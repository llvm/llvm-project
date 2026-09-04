import json
import re
import sys
from contextlib import contextmanager
from io import StringIO
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).parent.parent.parent / "tools" / "mlir-opt-repl" / "src")
)

import mlir_opt_repl.engine as engine
from mlir_opt_repl.mcp import mcp_main
from mlir_opt_repl.repl import interactive_main

SAMPLE_MLIR = "func.func @test(%arg0: f32, %arg1: f32) -> f32 { %0 = arith.addf %arg0, %arg1 : f32 return %0 : f32 }"
INIT_MSG = {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}}


@pytest.fixture(autouse=True)
def reset_engine():
    engine.current_ir = None
    engine.ir_history = []
    yield
    engine.current_ir = None
    engine.ir_history = []


@contextmanager
def capture_stdio(input_text):
    old_stdin, old_stdout = sys.stdin, sys.stdout
    sys.stdin = StringIO(input_text)
    sys.stdout = StringIO()
    try:
        yield sys.stdout
    finally:
        sys.stdin, sys.stdout = old_stdin, old_stdout


def run_mcp(*messages):
    input_text = ""
    for m in messages:
        text = json.dumps(m)
        input_text += f"Content-Length: {len(text.encode())}\r\n\r\n{text}"

    with capture_stdio(input_text) as stdout:
        try:
            mcp_main()
        except Exception:
            pass
    return stdout.getvalue()


def run_repl(input_text):
    with capture_stdio(input_text) as stdout:
        try:
            interactive_main()
        except (EOFError, SystemExit):
            pass
    return stdout.getvalue()


def parse_responses(output):
    responses = []
    for m in re.finditer(r"Content-Length: (\d+)\r?\n\r?\n", output):
        start = m.end()
        length = int(m.group(1))
        body = output[start : start + length]
        responses.append(json.loads(body))
    return responses


def tool_call(id, name, arguments=None):
    return {
        "jsonrpc": "2.0",
        "id": id,
        "method": "tools/call",
        "params": {
            "name": name,
            "arguments": arguments or {},
        },
    }
