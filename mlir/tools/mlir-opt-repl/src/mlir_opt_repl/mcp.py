import json
import sys

from mlir_opt_repl.engine import check_mlir_opt, handle_tool_call

TOOLS = [
    {
        "name": "run_pipeline",
        "description": "Run MLIR source through mlir-opt with the given passes. Sets the result as the current IR state for subsequent chain_pipeline calls.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mlir": {
                    "type": "string",
                    "description": "MLIR source text to process",
                },
                "passes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of pass flags (e.g. ['--convert-arith-to-llvm', '--convert-func-to-llvm']). Can also be a single pass-pipeline string.",
                },
                "extra_args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Additional mlir-opt arguments (e.g. ['--allow-unregistered-dialect'])",
                    "default": [],
                },
            },
            "required": ["mlir", "passes"],
        },
    },
    {
        "name": "chain_pipeline",
        "description": "Feed the current IR state (from a previous run_pipeline or chain_pipeline) through additional passes. Use this to incrementally lower IR step by step.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "passes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of pass flags to apply to the current IR state",
                },
                "extra_args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Additional mlir-opt arguments",
                    "default": [],
                },
            },
            "required": ["passes"],
        },
    },
    {
        "name": "get_current_ir",
        "description": "Return the current IR state without running any passes.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "reset",
        "description": "Clear the current IR state.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "list_passes",
        "description": "List available mlir-opt passes (conversion passes, canonicalize, cse, etc.)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filter": {
                    "type": "string",
                    "description": "Optional substring to filter pass names",
                    "default": "",
                }
            },
        },
    },
    {
        "name": "rewind",
        "description": "Rewind the IR state by N steps or to a named bookmark.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "integer",
                    "description": "Number of steps to rewind (default 1)",
                    "default": 1,
                    "minimum": 1,
                },
                "target": {
                    "type": "string",
                    "description": "Bookmark name to rewind to (overrides steps)",
                },
            },
        },
    },
    {
        "name": "bookmark",
        "description": "Bookmark the current history step with a name, or list all bookmarks (no name).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name for the bookmark. Omit to list existing bookmarks.",
                    "default": "",
                }
            },
        },
    },
    {
        "name": "save",
        "description": "Save the current IR state to a file.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path to write the current IR to",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "verify",
        "description": "Verify that the current IR is valid (runs mlir-opt --verify-diagnostics).",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "history",
        "description": "Show the history of pass applications and their IR states.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "show_ir": {
                    "type": "boolean",
                    "description": "Include full IR text in each history entry",
                    "default": False,
                },
                "format": {
                    "type": "string",
                    "enum": ["unified", "side_by_side"],
                    "description": "Diff format: 'unified' for standard unified diff, 'side_by_side' for two-column comparison",
                },
                "pretty": {
                    "type": "boolean",
                    "description": "Use ANSI colors and box-drawing characters for terminal display",
                    "default": False,
                },
                "width": {
                    "type": "integer",
                    "description": "Terminal width for side-by-side mode (auto-detected if omitted)",
                },
            },
        },
    },
]


def send(msg):
    text = json.dumps(msg)
    sys.stdout.write(f"Content-Length: {len(text.encode())}\r\n\r\n{text}")
    sys.stdout.flush()


def read_message():
    headers = {}
    while True:
        line = sys.stdin.readline()
        if not line:
            return None
        line = line.strip()
        if line == "":
            break
        if ":" in line:
            key, val = line.split(":", 1)
            headers[key.strip()] = val.strip()

    length = int(headers.get("Content-Length", 0))
    if length == 0:
        return None
    body = sys.stdin.read(length)
    return json.loads(body)


def dispatch(msg):
    method = msg.get("method", "")
    msg_id = msg.get("id")

    if method == "initialize":
        send(
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "mlir-opt-repl", "version": "0.1.0"},
                },
            }
        )
    elif method == "notifications/initialized":
        pass
    elif method == "tools/list":
        send(
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"tools": TOOLS},
            }
        )
    elif method == "tools/call":
        params = msg.get("params", {})
        result = handle_tool_call(params["name"], params.get("arguments", {}))
        send({"jsonrpc": "2.0", "id": msg_id, "result": result})
    elif msg_id is not None:
        send(
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        )


def mcp_main():
    check_mlir_opt()
    while True:
        msg = read_message()
        if msg is None:
            break
        dispatch(msg)
