# mlir-opt-repl

An interactive MLIR pass pipeline explorer. Feed MLIR through passes
incrementally, inspect intermediate states, rewind to try different lowering
paths, and diff what each pass changed.

Also functions as an MCP (Model Context Protocol) server for Claude Code.

## Installation

```bash
pip install mlir-opt-repl
```

`mlir-opt` must be on your PATH, or set via `MLIR_OPT`:

```bash
export MLIR_OPT=/path/to/llvm-project/build/bin/mlir-opt
```

## Interactive REPL (default)

```bash
mlir-opt-repl
```

Commands: `load`, `run`, `ir`, `history`, `diff`, `sbs`, `rewind`, `bookmark`,
`save`, `verify`, `reset`, `passes`, `help`, `quit`.

Tab completion is supported for commands and pass names.
Diffs are rendered with ANSI colors (red/green for changes, dim for context).

```
mlir-opt-repl> load test.mlir
mlir-opt-repl> run canonicalize
mlir-opt-repl> run convert-arith-to-llvm
mlir-opt-repl> diff
mlir-opt-repl> sbs 0 2
mlir-opt-repl> bookmark pre-lower
mlir-opt-repl> run convert-func-to-llvm
mlir-opt-repl> rewind pre-lower
mlir-opt-repl> save output.mlir
mlir-opt-repl> verify
```

Pass-pipeline syntax is also supported:

```
mlir-opt-repl> run builtin.module(canonicalize,cse,convert-arith-to-llvm)
```

## MCP Server (for Claude Code)

```bash
mlir-opt-repl mcp
```

To configure Claude Code, add to `.claude/settings.local.json`:

```json
{
  "mcpServers": {
    "mlir-opt-repl": {
      "command": "mlir-opt-repl",
      "args": ["mcp"],
      "env": {
        "MLIR_OPT": "/path/to/build/bin/mlir-opt"
      }
    }
  }
}
```

For development without installing:

```json
{
  "mcpServers": {
    "mlir-opt-repl": {
      "command": "python3",
      "args": ["-m", "mlir_opt_repl", "mcp"],
      "env": {
        "PYTHONPATH": "/path/to/llvm-project/mlir/tools/mlir-opt-repl/src",
        "MLIR_OPT": "/path/to/build/bin/mlir-opt"
      }
    }
  }
}
```

### MCP Tools

| Tool | Description |
|------|-------------|
| `run_pipeline` | Run MLIR source through passes (supports pass-pipeline strings), stores result |
| `chain_pipeline` | Apply additional passes to the current IR state |
| `get_current_ir` | Show the current IR without running passes |
| `reset` | Clear current IR state, history, and bookmarks |
| `rewind` | Undo the last N steps, or rewind to a named bookmark |
| `bookmark` | Bookmark the current history step (or list bookmarks) |
| `save` | Save current IR to a file |
| `verify` | Verify current IR is valid |
| `history` | Show pass application timeline, with unified or side-by-side diffs |
| `list_passes` | List available mlir-opt passes with optional filter |

### Example MCP Session

```
run_pipeline(mlir="func.func @f(...) ...", passes=["canonicalize"])
  → canonicalized IR (saved as state)

chain_pipeline(passes=["builtin.module(convert-arith-to-llvm)"])
  → arith ops converted via pass-pipeline syntax

bookmark(name="pre-func-lower")
  → bookmarks current step

chain_pipeline(passes=["convert-func-to-llvm"])
  → fully lowered

rewind(target="pre-func-lower")
  → back to bookmarked state

save(path="output.mlir")
  → writes current IR to file

verify()
  → confirms IR is valid
```

## Project Structure

```
pyproject.toml
src/mlir_opt_repl/
  __init__.py
  __main__.py    — Click CLI: mlir-opt-repl [mcp|repl]
  engine.py      — Core state + logic: run_mlir_opt, handle_tool_call, bookmarks
  mcp.py         — MCP protocol: TOOLS schema, send/recv, dispatch
  repl.py        — Interactive terminal REPL with tab completion
  diff.py        — Side-by-side and unified diff renderers
  render.py      — ANSI color constants
```

## Testing

Tests live in `mlir/test/mlir-opt-repl/` and use pytest with 100% line coverage:

```bash
PYTHONPATH=mlir/tools/mlir-opt-repl/src \
MLIR_OPT=/path/to/build/bin/mlir-opt \
  python3 -m pytest mlir/test/mlir-opt-repl -n auto \
    --cov=mlir_opt_repl \
    --cov-config=mlir/tools/mlir-opt-repl/pyproject.toml \
    --cov-fail-under=100
```
