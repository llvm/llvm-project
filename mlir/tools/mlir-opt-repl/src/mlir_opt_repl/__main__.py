"""Entry point for mlir-opt-repl.

Provides two modes:
  - MCP server: speaks Model Context Protocol over stdio,
    for use as a Claude Code tool server.
  - Interactive REPL (default): a terminal interface for exploring MLIR pass
    pipelines with colored diffs and history navigation.
"""

import json
import os
import shutil
from pathlib import Path

import click

from mlir_opt_repl.mcp import mcp_main
from mlir_opt_repl.repl import interactive_main


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """mlir-opt-repl: Interactive MLIR pass pipeline explorer.

    An MCP server and terminal REPL for running mlir-opt passes
    incrementally, inspecting intermediate IR, rewinding to try
    different lowering paths, and diffing what each pass changed.

    By default (no subcommand), starts the interactive REPL.
    Use 'mlir-opt-repl mcp' to start the MCP server for Claude Code.
    Use 'mlir-opt-repl mcp-config' to print the Claude Code configuration JSON.
    """
    if ctx.invoked_subcommand is None and not ctx.resilient_parsing:
        ctx.invoke(repl)


@cli.command()
def mcp():
    """Start the MCP (Model Context Protocol) server over stdio.

    This is the mode used by Claude Code. The server accepts JSON-RPC
    messages on stdin and responds on stdout, exposing tools for running
    MLIR through pass pipelines, chaining passes, rewinding, and diffing.

    To use with Claude Code, add to .claude/settings.local.json:

    \b
      {
        "mcpServers": {
          "mlir-opt-repl": {
            "command": "mlir-opt-repl",
            "args": ["mcp"],
            "env": {"MLIR_OPT": "/path/to/mlir-opt"}
          }
        }
      }
    """
    mcp_main()


@cli.command()
def repl():
    """Start an interactive terminal REPL.

    Provides a command-line interface for loading MLIR, applying passes
    step by step, viewing colored diffs (unified or side-by-side),
    rewinding to try different lowering paths, and browsing history.

    \b
    Commands:
      load <file.mlir>    Load MLIR from a file
      load -              Load MLIR from stdin (blank line to finish)
      run <passes...>     Apply passes to current IR
      ir                  Show current IR
      history             Show pass application history
      diff [a b]          Unified diff between steps
      sbs [a b]           Side-by-side diff between steps
      rewind [N]          Undo last N steps (default 1)
      reset               Clear all state
      passes [filter]     List available passes
      quit                Exit

    \b
    Example:
      $ MLIR_OPT=/path/to/mlir-opt mlir-opt-repl
      mlir-opt-repl> load test.mlir
      mlir-opt-repl> run canonicalize
      mlir-opt-repl> run convert-arith-to-llvm
      mlir-opt-repl> diff
    """
    interactive_main()


@cli.command(name="mcp-config")
def mcp_config():
    """Print the Claude Code MCP server configuration JSON.

    Outputs a ready-to-paste JSON block for .claude/settings.local.json,
    with the correct absolute path to this installation and mlir-opt
    (if found on PATH or via MLIR_OPT).
    """
    main_py = str(Path(__file__).resolve())
    mlir_opt = os.environ.get("MLIR_OPT") or shutil.which("mlir-opt")

    server_config = {
        "command": "python3",
        "args": [main_py, "mcp"],
    }
    if mlir_opt:
        server_config["env"] = {"MLIR_OPT": str(Path(mlir_opt).resolve())}

    output = {"mcpServers": {"mlir-opt-repl": server_config}}
    click.echo(json.dumps(output, indent=2))


def main():
    cli()


if __name__ == "__main__":
    main()
