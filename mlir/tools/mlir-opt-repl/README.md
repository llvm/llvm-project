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

Commands: `load`, `run`, `ir`, `history`, `diff`, `sbs`, `rewind`, `reset`, `passes`, `quit`.
Diffs are rendered with ANSI colors (red/green for changes, dim for context).

```
mlir-opt-repl> load test.mlir
mlir-opt-repl> run canonicalize
mlir-opt-repl> run convert-arith-to-llvm
mlir-opt-repl> diff
mlir-opt-repl> sbs 0 2
mlir-opt-repl> rewind 1
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
| `run_pipeline` | Run MLIR source through passes, stores result as current state |
| `chain_pipeline` | Apply additional passes to the current IR state |
| `get_current_ir` | Show the current IR without running passes |
| `reset` | Clear current IR state and history |
| `rewind` | Undo the last N pass applications |
| `history` | Show pass application timeline, with unified or side-by-side diffs |
| `list_passes` | List available mlir-opt passes with optional filter |

### Example MCP Session

```
run_pipeline(mlir="func.func @f(...) ...", passes=["canonicalize"])
  → canonicalized IR (saved as state)

chain_pipeline(passes=["convert-arith-to-llvm"])
  → arith ops converted to LLVM dialect

history(format="unified")
  → shows unified diff between each step

rewind(steps=1)
  → back to post-canonicalize state, try a different path

history(format="side_by_side", width=120)
  → two-column comparison of each step
```

## Project Structure

```
pyproject.toml
src/mlir_opt_repl/
  __init__.py
  __main__.py    — Click CLI: mlir-opt-repl [mcp|repl]
  engine.py      — Core state + logic: run_mlir_opt, list_passes, handle_tool_call
  mcp.py         — MCP protocol: TOOLS schema, send/recv, dispatch
  repl.py        — Interactive terminal REPL
  diff.py        — Side-by-side and unified diff renderers
  render.py      — ANSI color constants
```

## Testing

Tests live in `mlir/test/mlir-opt-repl/` and use pytest with 100% line coverage:

```bash
PYTHONPATH=mlir/tools/mlir-opt-repl/src \
MLIR_OPT=/path/to/build/bin/mlir-opt \
  python3 -m pytest mlir/test/mlir-opt-repl \
    --cov=mlir_opt_repl \
    --cov-config=mlir/tools/mlir-opt-repl/pyproject.toml \
    --cov-fail-under=100
```
