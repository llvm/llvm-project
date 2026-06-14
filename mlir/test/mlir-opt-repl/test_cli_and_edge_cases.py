"""Tests for the click CLI entry point and edge cases in engine/diff."""

import json
import subprocess
from unittest.mock import patch

import pytest
from click.testing import CliRunner

import mlir_opt_repl.engine as engine
from conftest import (
    INIT_MSG,
    capture_stdio,
    parse_responses,
    run_mcp,
    run_repl,
    tool_call,
)
from mlir_opt_repl import repl as repl_module
from mlir_opt_repl.__main__ import cli
from mlir_opt_repl.diff import render_side_by_side, render_unified_diff
from mlir_opt_repl.mcp import mcp_main
from mlir_opt_repl.repl import _completer, _get_pass_names, interactive_main


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


class TestCompleter:
    def test_command_completion(self):

        with patch("mlir_opt_repl.repl.readline") as mock_rl:
            mock_rl.get_line_buffer.return_value = "lo"
            result = _completer("lo", 0)
            assert result == "load "

    def test_pass_completion(self):

        with patch("mlir_opt_repl.repl.readline") as mock_rl:
            mock_rl.get_line_buffer.return_value = "run convert-arith"
            result = _completer("convert-arith", 0)
            assert result is not None
            assert "convert-arith" in result

    def test_no_match(self):

        with patch("mlir_opt_repl.repl.readline") as mock_rl:
            mock_rl.get_line_buffer.return_value = "zzz"
            result = _completer("zzz", 0)
            assert result is None

    def test_bookmark_completion(self):

        repl_module.bookmarks = {"mymark": 0}
        with patch("mlir_opt_repl.repl.readline") as mock_rl:
            mock_rl.get_line_buffer.return_value = "rewind my"
            result = _completer("my", 0)
            assert result == "mymark "

    def test_state_out_of_range(self):

        with patch("mlir_opt_repl.repl.readline") as mock_rl:
            mock_rl.get_line_buffer.return_value = "lo"
            result = _completer("lo", 99)
            assert result is None

    def test_other_command_no_completions(self):
        with patch("mlir_opt_repl.repl.readline") as mock_rl:
            mock_rl.get_line_buffer.return_value = "save "
            result = _completer("", 0)
            assert result is None


class TestBookmarkInvalidIndex:
    def test_rewind_to_invalid_bookmark_mcp(self):

        engine.ir_history = [("initial", "module {}")]
        engine.current_ir = "module {}"
        engine.bookmarks = {"stale": 99}
        result = engine.handle_tool_call("rewind", {"target": "stale"})
        assert result["isError"] is True
        assert "invalid index" in result["content"][0]["text"]

    def test_rewind_to_invalid_bookmark_repl(self):

        engine.ir_history = [("initial", "module {}")]
        engine.current_ir = "module {}"
        repl_module.bookmarks = {"stale": 99}
        output = run_repl("rewind stale\nquit\n")
        assert "invalid index" in output


class TestVerifyInvalid:
    def test_verify_fails_on_bad_ir(self):
        engine.current_ir = "func.func @f() -> i32 { return }"
        engine.ir_history = [("initial", engine.current_ir)]
        result = engine.handle_tool_call("verify", {})
        assert result["isError"] is True
        assert "Verification failed" in result["content"][0]["text"]


class TestBookmarkNoBookmarksMCP:
    def test_list_empty(self):
        output = run_mcp(
            INIT_MSG,
            tool_call(
                1,
                "run_pipeline",
                {"mlir": "func.func @f() { return }", "passes": ["canonicalize"]},
            ),
            tool_call(2, "bookmark"),
        )
        assert (
            "(no bookmarks)"
            in parse_responses(output)[2]["result"]["content"][0]["text"]
        )


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
