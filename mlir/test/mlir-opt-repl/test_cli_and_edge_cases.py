"""Tests for the click CLI entry point and edge cases in engine/diff."""

import json
import subprocess
from unittest.mock import patch

from click.testing import CliRunner

import mlir_opt_repl.engine as engine
from conftest import capture_stdio
from mlir_opt_repl.__main__ import cli
from mlir_opt_repl.diff import render_side_by_side, render_unified_diff
from mlir_opt_repl.mcp import mcp_main
from mlir_opt_repl.repl import interactive_main


class TestClickCLI:
    def test_help(self):
        result = CliRunner().invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "mcp" in result.output
        assert "repl" in result.output

    def test_mcp_help(self):
        result = CliRunner().invoke(cli, ["mcp", "--help"])
        assert result.exit_code == 0
        assert "MCP" in result.output

    def test_repl_help(self):
        result = CliRunner().invoke(cli, ["repl", "--help"])
        assert result.exit_code == 0
        assert "load" in result.output

    def test_default_is_repl(self):
        result = CliRunner().invoke(cli, [], input="quit\n")
        assert "mlir-opt-repl" in result.output

    def test_mcp_subcommand(self):
        msg = json.dumps(
            {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}}
        )
        input_text = f"Content-Length: {len(msg.encode())}\r\n\r\n{msg}"
        result = CliRunner().invoke(cli, ["mcp"], input=input_text)
        assert "mlir-opt-repl" in result.output


class TestEngineEdgeCases:
    def test_mlir_opt_not_found(self):
        old = engine.MLIR_OPT
        engine.MLIR_OPT = "/nonexistent/mlir-opt"
        try:
            result = engine.handle_tool_call(
                "run_pipeline", {"mlir": "module {}", "passes": ["canonicalize"]}
            )
            assert result["isError"]
            assert "not found" in result["content"][0]["text"]
        finally:
            engine.MLIR_OPT = old

    def test_check_mlir_opt_exits(self):
        import pytest

        old = engine.MLIR_OPT
        engine.MLIR_OPT = "/nonexistent/mlir-opt"
        try:
            with pytest.raises(SystemExit):
                engine.check_mlir_opt()
        finally:
            engine.MLIR_OPT = old

    def test_check_mlir_opt_passes(self):
        engine.check_mlir_opt()

    def test_list_passes_error(self):
        old = engine.MLIR_OPT
        engine.MLIR_OPT = "/nonexistent/mlir-opt"
        try:
            passes = engine.list_passes()
            assert len(passes) == 1
            assert "error" in passes[0]
        finally:
            engine.MLIR_OPT = old

    def test_timeout(self):
        with patch(
            "mlir_opt_repl.engine.subprocess.run",
            side_effect=subprocess.TimeoutExpired("cmd", 30),
        ):
            result = engine.handle_tool_call(
                "run_pipeline", {"mlir": "module {}", "passes": ["canonicalize"]}
            )
        assert result["isError"]
        assert "timed out" in result["content"][0]["text"]


class TestMCPEdgeCases:
    def test_empty_content_length(self):
        with capture_stdio("Content-Length: 0\r\n\r\n") as stdout:
            try:
                mcp_main()
            except Exception:
                pass
        assert stdout.getvalue() == ""


class TestReplLoadEOF:
    def test_load_stdin_eof(self):
        with capture_stdio("load -\nfunc.func @f() { return }") as stdout:
            try:
                interactive_main()
            except (EOFError, SystemExit):
                pass
        assert "Loaded" in stdout.getvalue()


class TestDiffEdgeCases:
    def test_non_pretty_side_by_side(self):
        result = render_side_by_side(
            ["a", "b"], ["a", "c"], "left", "right", width=60, pretty=False
        )
        assert "|" in result

    def test_delete_only(self):
        result = render_side_by_side(
            ["a", "b", "c"], ["a"], "left", "right", width=80, pretty=True
        )
        assert "b" in result and "c" in result

    def test_insert_only(self):
        result = render_side_by_side(
            ["a"], ["a", "b", "c"], "left", "right", width=80, pretty=True
        )
        assert "b" in result and "c" in result

    def test_non_pretty_unified(self):
        result = render_unified_diff(["a", "b"], ["a", "c"], "f1", "f2", pretty=False)
        assert "-b" in result and "+c" in result
        assert "\033[" not in result

    def test_truncation(self):
        result = render_side_by_side(
            ["x" * 200], ["short"], "left", "right", width=60, pretty=False
        )
        assert "…" in result

    def test_replace_uneven_more_left(self):
        result = render_side_by_side(
            ["a", "b", "c"], ["x"], "left", "right", width=60, pretty=False
        )
        assert "b" in result and "c" in result
