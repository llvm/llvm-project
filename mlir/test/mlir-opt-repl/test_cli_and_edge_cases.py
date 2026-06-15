"""Tests for the click CLI entry point and edge cases in engine/diff."""

import json
import subprocess
from unittest.mock import patch

import pytest
from click.testing import CliRunner

import mlir_opt_repl.engine as engine_module
from mlir_opt_repl.engine import state, handle_tool_call
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
        old = engine_module.MLIR_OPT
        engine_module.MLIR_OPT = "/nonexistent/mlir-opt"
        try:
            result = handle_tool_call(
                "run_pipeline", {"mlir": "module {}", "passes": ["canonicalize"]}
            )
            assert result["isError"]
            assert "not found" in result["content"][0]["text"]
        finally:
            engine_module.MLIR_OPT = old

    def test_check_mlir_opt_exits(self):

        old = engine_module.MLIR_OPT
        engine_module.MLIR_OPT = "/nonexistent/mlir-opt"
        try:
            with pytest.raises(SystemExit):
                engine_module.check_mlir_opt()
        finally:
            engine_module.MLIR_OPT = old

    def test_check_mlir_opt_passes(self):
        engine_module.check_mlir_opt()

    def test_list_passes_error(self):
        old = engine_module.MLIR_OPT
        engine_module.MLIR_OPT = "/nonexistent/mlir-opt"
        try:
            passes = engine_module.list_passes()
            assert len(passes) == 1
            assert "error" in passes[0]
        finally:
            engine_module.MLIR_OPT = old

    def test_get_help_text_error(self):
        old = engine_module.MLIR_OPT
        engine_module.MLIR_OPT = "/nonexistent/mlir-opt"
        try:
            text = engine_module.get_help_text()
            assert "error" in text
        finally:
            engine_module.MLIR_OPT = old

    def test_timeout(self):
        with patch(
            "mlir_opt_repl.engine.subprocess.run",
            side_effect=subprocess.TimeoutExpired("cmd", 30),
        ):
            result = handle_tool_call(
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

        state.history_clear()
        state.history_append("initial", "module {}")
        state.bookmarks = {"stale": 99}
        result = handle_tool_call("rewind", {"target": "stale"})
        assert result["isError"] is True
        assert "invalid index" in result["content"][0]["text"]

    def test_rewind_to_invalid_bookmark_repl(self):

        state.history_clear()
        state.history_append("initial", "module {}")
        repl_module.bookmarks = {"stale": 99}
        output = run_repl("rewind stale\nquit\n")
        assert "invalid index" in output


class TestVerifyInvalid:
    def test_verify_fails_on_bad_ir(self):
        state.history_clear()
        state.history_append("initial", "func.func @f() -> i32 { return }")
        result = handle_tool_call("verify", {})
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


class TestIRHistory:
    def test_negative_getitem(self):
        state.history_clear()
        state.history_append("initial", "aaa")
        state.history_append("step1", "bbb")
        desc, ir = state.history_get(-1)
        assert desc == "step1"
        assert ir == "bbb"

    def test_iter(self):
        state.history_clear()
        state.history_append("initial", "aaa")
        state.history_append("step1", "bbb")
        items = list(state._ir_history)
        assert len(items) == 2
        assert items[0] == ("initial", "aaa")
        assert items[1] == ("step1", "bbb")

    def test_negative_get_description(self):
        state.history_clear()
        state.history_append("initial", "aaa")
        state.history_append("step1", "bbb")
        assert state.history_get_description(-1) == "step1"

    def test_out_of_bounds_get_description(self):
        state.history_clear()
        state.history_append("initial", "aaa")
        assert state.history_get_description(99) == "?"

    def test_truncate_noop(self):
        state.history_clear()
        state.history_append("initial", "aaa")
        state.history_truncate(5)
        assert state.history_len() == 1
        assert state.get_current_ir() == "aaa"

    def test_truncate_to_empty(self):
        state.history_clear()
        state.history_append("initial", "aaa")
        state.history_truncate(0)
        assert state.history_len() == 0
        assert state.get_current_ir() is None

    def test_reconstruct_negative_index(self):
        state.history_clear()
        state.history_append("initial", "aaa")
        state.history_append("step1", "bbb")
        ir = state._ir_history.reconstruct_ir(-1)
        assert ir == "bbb"

    def test_iter_with_ir_empty(self):
        state.history_clear()
        assert list(state.history_iter_with_ir()) == []

    def test_reconstruct_middle_entry(self):
        state.history_clear()
        state.history_append("initial", "line1\nline2\n")
        state.history_append("step1", "line1\nline2\nline3\n")
        state.history_append("step2", "line1\nline2\nline3\nline4\n")
        desc, ir = state.history_get(1)
        assert ir == "line1\nline2\nline3\n"

    def test_delete_and_insert_patches(self):
        state.history_clear()
        state.history_append("initial", "a\nb\nc\n")
        state.history_append("deleted", "a\nc\n")
        state.history_append("inserted", "a\nx\ny\nc\n")
        assert state.get_current_ir() == "a\nx\ny\nc\n"
        desc, ir = state.history_get(1)
        assert ir == "a\nc\n"
        state.history_truncate(2)
        assert state.get_current_ir() == "a\nc\n"
        state.history_truncate(1)
        assert state.get_current_ir() == "a\nb\nc\n"

    def test_redo_stack_cleared_on_append(self):
        state.history_clear()
        state.history_append("initial", "aaa")
        state.history_append("step1", "bbb")
        state.history_truncate(1)
        assert state._ir_history._redo_stack
        state.history_append("step2", "ccc")
        assert not state._ir_history._redo_stack

    def test_roundtrip_fidelity(self):
        texts = ["hello\n", "hello\nworld\n", "world\n", "completely different"]
        state.history_clear()
        for i, t in enumerate(texts):
            state.history_append(f"step{i}", t)
        for i, t in enumerate(texts):
            assert state._ir_history.reconstruct_ir(i) == t

    def test_iter_with_ir_full_ir_entry(self):
        from mlir_opt_repl.history import HistoryEntry

        state.history_clear()
        state.history_append("initial", "aaa\n")
        state._ir_history._entries.append(
            HistoryEntry(description="checkpoint", full_ir="bbb\n", parent_index=0)
        )
        state._ir_history._current_ir = "bbb\n"
        items = list(state.history_iter_with_ir())
        assert items[1] == (1, "checkpoint", "bbb\n")
