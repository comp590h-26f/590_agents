"""Agent that converts Python variables from snake_case to camelCase."""

# Analyze Python Code
import ast
import json

# Regular Expressions to quickly find strings
import re
from pathlib import Path
from typing import Any

# AI Model related dependancies
from openai.types.responses import ResponseFunctionToolCall

from agents590.contracts import RunContext
from agents590.provider import UNCOpenAISettings

SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)+$")

# System Prompt
SYSTEM = """Convert function parameters and local variables from snake_case
to camelCase, including loop, comprehension, and lambda variables.
Rename their references as well.

Do not change function names, constants, strings, dictionary keys,
keyword arguments, or program behavior.

Submit the complete file using submit_file. If validation fails,
fix the reported problems and submit the file again."""


TOOLS: list[Any] = [
    {
        # Specify tool that the AI Model can call
        "type": "function",
        "name": "submit_file",
        "description": "Submit the converted Python file for validation.",

        # Input parameter format for tool
        "parameters": {
            "type": "object",
            "properties": {"content": {"type": "string"}},
            "required": ["content"],
        },
    }
]


def protected_items(tree: ast.AST) -> list[tuple[str, str]]:
    """Collect values that must remain unchanged."""
    items: list[tuple[str, str]] = []

    # Identifies all string literals, keywords, function names, and constants in the file
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            items.append(("literal", repr(node.value)))
        elif isinstance(node, ast.keyword) and node.arg is not None:
            items.append(("keyword", node.arg))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            items.append(("function", node.name))
        elif isinstance(node, ast.Name) and node.id.isupper():
            items.append(("constant", node.id))

    return sorted(items)


def variable_names(tree: ast.AST) -> set[str]:
    """Find parameter names and assigned variable names."""
    names = {node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)}

    # Collect all possible candidate variables for conversion to camelCase
    names.update(
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    )

    return names

# Compare AI Model Changes to identify mistakes and possibly rerun the model
def problems(original: str, converted: str) -> str:
    """Return a validation error, or an empty string if valid."""
    try:
        # Check if the code is valid Python
        old_tree = ast.parse(original)
        new_tree = ast.parse(converted)
    except SyntaxError as error:
        return f"Syntax error: {error}"

    # Identify unchanged snake case variables
    remaining = sorted(name for name in variable_names(new_tree) if SNAKE_CASE.fullmatch(name))
    if remaining:
        return f"Variables still use snake_case: {remaining}"

    # Identify if a protected item has been changed by the model
    if protected_items(old_tree) != protected_items(new_tree):
        return "Function names, constants, keywords, or literals changed."

    return ""


class leoa_CamelAgent:
    # Configuration
    def provider_settings(self, context: RunContext) -> UNCOpenAISettings:
        return UNCOpenAISettings.from_env(model=context.model)

    
    async def run(self, instruction: str, context: RunContext) -> str:
        try:
            # Identify Python file to be changed according to instruction
            matches = re.findall(r"[\w./\\-]+\.py", instruction)
            if not matches:
                return "FAILURE: No Python file specified."

            # Find/Read Python file
            path = context.cwd / Path(matches[0]).name
            original = path.read_text()

            # Logs communication with the AI model
            history: list[Any] = [{"role": "user", "content": f"{instruction}\n\n{original}"}]
            settings = self.provider_settings(context)

            async with settings.create_async_client() as client:
                # 6 tries for the model to return a converted file correctly
                for _ in range(6):
                    # Request for the AI Model
                    response = await client.responses.create(
                        model=settings.model,
                        instructions=SYSTEM,
                        input=history,
                        tools=TOOLS,
                        tool_choice="required",
                        store=False,
                        background=False,
                    )
                    # Logging
                    history.extend(response.output)

                    for item in response.output:
                        if not isinstance(item, ResponseFunctionToolCall):
                            continue

                        # Extract output and validate
                        converted = json.loads(item.arguments)["content"]
                        error = problems(original, converted)

                        if error:
                            # Store failure message
                            result = f"Rejected: {error} Fix the issue and resubmit."
                        else:
                            # If successful, update the original file
                            path.write_text(converted.rstrip() + "\n")
                            return f"Converted {path.name}."

                        # Logging
                        history.append(
                            {
                                "type": "function_call_output",
                                "call_id": item.call_id,
                                "output": result,
                            }
                        )
                        # Retry again if unsucessful

        except Exception as error:
            return f"FAILURE: {error}"

        # Fail if the model never returned valid output
        return "FAILURE: File left unchanged."
