"""Exercise the webp_agent loop with a scripted client and real local shell commands."""

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import pytest
from openai import AsyncOpenAI
from openai.types.responses import Response

import agents590.webp_agent as webp_agent
from agents590.contracts import RunContext
from agents590.webp_agent import WebpAgent, run_shell


def _response(output: list[dict[str, Any]], usage: dict[str, Any] | None = None) -> Response:
    return Response.model_validate(
        {
            "id": "resp_test",
            "object": "response",
            "created_at": 0,
            "model": "course-deployment",
            "output": output,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": usage,
        }
    )


def _usage(prompt: int, completion: int, cached: int) -> dict[str, Any]:
    return {
        "input_tokens": prompt,
        "input_tokens_details": {"cached_tokens": cached, "cache_write_tokens": 0},
        "output_tokens": completion,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": prompt + completion,
    }


def _shell_call(call_id: str, command: str) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": "shell",
        "arguments": json.dumps({"command": command}),
        "status": "completed",
    }


def _message(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


class ScriptedResponses:
    def __init__(self, responses: list[Response]) -> None:
        self.responses = responses
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: object) -> Response:
        self.requests.append(json.loads(json.dumps(kwargs)))
        return self.responses.pop(0)


class ScriptedClient:
    def __init__(self, responses: list[Response]) -> None:
        self.responses = ScriptedResponses(responses)


@pytest.fixture(autouse=True)
def configured_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-unc-key")
    monkeypatch.setenv("OPENAI_MODEL", "course-deployment")


def test_loop_runs_shell_calls_in_cwd_and_returns_final_text(tmp_path: Path) -> None:
    client = ScriptedClient(
        [
            _response([_shell_call("call_1", "echo fixed > result.txt && pwd")]),
            _response([_message("Done.")]),
        ]
    )
    agent = WebpAgent(client=cast(AsyncOpenAI, client))
    output = asyncio.run(agent.run("Fix it", RunContext(cwd=tmp_path, model="openai/other")))

    assert output == "Done."
    assert (tmp_path / "result.txt").read_text() == "fixed\n"
    first, second = client.responses.requests
    assert first["model"] == "other"
    assert first["store"] is False and first["background"] is False
    assert first["input"] == [{"role": "user", "content": "Fix it"}]
    tool_result = second["input"][-1]
    assert tool_result["type"] == "function_call_output"
    assert tool_result["call_id"] == "call_1"
    assert "exit_code: 0" in tool_result["output"]
    assert str(tmp_path) in tool_result["output"]


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        ({**_shell_call("c", ""), "arguments": "not json"}, "Invalid arguments"),
        ({**_shell_call("c", ""), "name": "python"}, "Unknown tool"),
    ],
)
def test_malformed_tool_calls_are_reported_to_the_model(
    tmp_path: Path, call: dict[str, Any], expected: str
) -> None:
    client = ScriptedClient([_response([call]), _response([_message("ok")])])
    asyncio.run(WebpAgent(client=cast(AsyncOpenAI, client)).run("x", RunContext(cwd=tmp_path)))
    assert expected in client.responses.requests[1]["input"][-1]["output"]


def test_run_shell_reports_failures_and_timeouts(tmp_path: Path) -> None:
    failed = asyncio.run(run_shell("echo oops >&2; exit 3", tmp_path))
    assert "exit_code: 3" in failed and "oops" in failed
    assert "timed out" in asyncio.run(run_shell("sleep 5", tmp_path, timeout=0.1))


def test_each_iteration_appends_an_atif_step(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.json"
    summary = [{"type": "summary_text", "text": "Look first."}]
    reasoning = {"type": "reasoning", "id": "rs_1", "summary": summary}
    malformed = _shell_call("call_2", "")
    client = ScriptedClient(
        [
            _response([reasoning, _shell_call("call_1", "echo hi")], _usage(100, 20, 40)),
            _response([{**malformed, "arguments": "not json"}], _usage(150, 10, 100)),
            _response([_message("Done.")], _usage(200, 5, 150)),
        ]
    )
    agent = WebpAgent(client=cast(AsyncOpenAI, client), trajectory_path=path)
    asyncio.run(agent.run("Fix it", RunContext(cwd=tmp_path)))

    trajectory = json.loads(path.read_text())
    steps = trajectory["steps"]
    assert trajectory["schema_version"] == "ATIF-v1.8"
    assert trajectory["agent"]["name"] == "webp_agent"
    assert [step["step_id"] for step in steps] == [1, 2, 3, 4, 5]
    assert [step["source"] for step in steps] == ["system", "user", "agent", "agent", "agent"]
    assert steps[1]["message"] == "Fix it"
    first = steps[2]
    assert first["reasoning_content"] == "Look first."
    assert first["tool_calls"] == [
        {"tool_call_id": "call_1", "function_name": "shell", "arguments": {"command": "echo hi"}}
    ]
    assert first["observation"]["results"][0]["source_call_id"] == "call_1"
    assert "hi" in first["observation"]["results"][0]["content"]
    assert steps[3]["tool_calls"][0]["arguments"] == {"raw": "not json"}
    assert steps[4]["message"] == "Done." and "tool_calls" not in steps[4]
    assert trajectory["final_metrics"] == {
        "total_prompt_tokens": 450,
        "total_completion_tokens": 35,
        "total_cached_tokens": 290,
        "total_steps": 5,
    }
    assert not (tmp_path / "trajectory.json.partial").exists()


def test_trajectory_is_only_written_inside_harbor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(webp_agent, "HARBOR_AGENT_LOGS", tmp_path / "missing")
    client = ScriptedClient([_response([_message("ok")])])
    asyncio.run(WebpAgent(client=cast(AsyncOpenAI, client)).run("x", RunContext(cwd=tmp_path)))
    assert list(tmp_path.iterdir()) == []

    logs = tmp_path / "logs"
    logs.mkdir()
    monkeypatch.setattr(webp_agent, "HARBOR_AGENT_LOGS", logs)
    client = ScriptedClient([_response([_message("ok")])])
    asyncio.run(WebpAgent(client=cast(AsyncOpenAI, client)).run("x", RunContext(cwd=tmp_path)))
    assert (logs / "trajectory.json").is_file()
