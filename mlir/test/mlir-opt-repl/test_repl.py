"""Tests for the interactive REPL."""

from conftest import SAMPLE_MLIR, run_repl

LOAD_SAMPLE = f"load -\n{SAMPLE_MLIR}\n\n"


class TestHelp:
    def test_shows_commands(self):
        output = run_repl("help\nquit\n")
        for cmd in ("load", "run", "diff", "sbs", "rewind", "quit"):
            assert cmd in output


class TestLoad:
    def test_stdin(self):
        output = run_repl("load -\nfunc.func @f() { return }\n\nir\nquit\n")
        assert "Loaded" in output
        assert "func.func @f" in output

    def test_multiline(self):
        output = run_repl(
            "load -\nmodule {\n  func.func @f() {\n    return\n  }\n}\n\nir\nquit\n"
        )
        assert "Loaded" in output
        assert "module {" in output

    def test_file(self, tmp_path):
        f = tmp_path / "test.mlir"
        f.write_text("module {}")
        output = run_repl(f"load {f}\nir\nquit\n")
        assert "Loaded" in output
        assert "module {}" in output

    def test_file_not_found(self):
        output = run_repl("load /nonexistent/path.mlir\nquit\n")
        assert "File not found" in output

    def test_no_arg(self):
        output = run_repl("load\nquit\n")
        assert "Usage:" in output

    def test_invalid_mlir(self):
        output = run_repl("load -\nhello\n\nquit\n")
        assert "Invalid MLIR" in output

    def test_stdin_eof(self):
        output = run_repl("load -\nfunc.func @f() { return }")
        assert "Loaded" in output


class TestRun:
    def test_basic(self):
        output = run_repl(LOAD_SAMPLE + "run canonicalize\nquit\n")
        assert "arith.addf" in output

    def test_multiple_passes(self):
        output = run_repl(
            LOAD_SAMPLE + "run convert-arith-to-llvm convert-func-to-llvm\nquit\n"
        )
        assert "llvm.fadd" in output

    def test_no_ir(self):
        output = run_repl("run canonicalize\nquit\n")
        assert "No IR loaded" in output

    def test_no_arg(self):
        output = run_repl("run\nquit\n")
        assert "Usage:" in output

    def test_invalid_pass(self):
        output = run_repl(LOAD_SAMPLE + "run nonexistent-pass-xyz\nquit\n")
        assert "Current IR unchanged" in output


class TestHistory:
    def test_shows_steps(self):
        output = run_repl(
            LOAD_SAMPLE + "run canonicalize\nrun convert-arith-to-llvm\nhistory\nquit\n"
        )
        assert "initial" in output
        assert "--canonicalize" in output
        assert "--convert-arith-to-llvm" in output
        assert "<-- current" in output

    def test_no_history(self):
        output = run_repl("history\nquit\n")
        assert "(no history)" in output


class TestDiff:
    def test_last_step(self):
        output = run_repl(
            LOAD_SAMPLE + "run canonicalize\nrun convert-arith-to-llvm\ndiff\nquit\n"
        )
        assert "---" in output
        assert "+++" in output

    def test_with_indices(self):
        output = run_repl(
            LOAD_SAMPLE
            + "run canonicalize\nrun convert-arith-to-llvm\nrun convert-func-to-llvm\ndiff 0 2\nquit\n"
        )
        assert "--- initial" in output

    def test_no_history(self):
        output = run_repl("diff\nquit\n")
        assert "need at least 2 history entries" in output

    def test_invalid_indices(self):
        output = run_repl(
            LOAD_SAMPLE
            + "run canonicalize\nrun convert-arith-to-llvm\ndiff 0 99\nquit\n"
        )
        assert "Invalid indices" in output


class TestSbs:
    def test_last_step(self):
        output = run_repl(
            LOAD_SAMPLE + "run canonicalize\nrun convert-arith-to-llvm\nsbs\nquit\n"
        )
        assert "--canonicalize" in output
        assert "--convert-arith-to-llvm" in output

    def test_with_indices(self):
        output = run_repl(
            LOAD_SAMPLE
            + "run canonicalize\nrun convert-arith-to-llvm\nrun convert-func-to-llvm\nsbs 0 2\nquit\n"
        )
        assert "initial" in output

    def test_no_history(self):
        output = run_repl("sbs\nquit\n")
        assert "need at least 2 history entries" in output

    def test_invalid_indices(self):
        output = run_repl(
            LOAD_SAMPLE
            + "run canonicalize\nrun convert-arith-to-llvm\nsbs 0 99\nquit\n"
        )
        assert "Invalid indices" in output


class TestRewind:
    def test_rewind_one(self):
        output = run_repl(
            LOAD_SAMPLE + "run canonicalize\nrun convert-arith-to-llvm\nrewind\nquit\n"
        )
        assert "Rewound 1 step(s)" in output

    def test_rewind_past_beginning(self):
        output = run_repl(LOAD_SAMPLE + "run canonicalize\nrewind 99\nquit\n")
        assert "Rewound to beginning" in output

    def test_no_history(self):
        output = run_repl("rewind\nquit\n")
        assert "No history to rewind" in output


class TestReset:
    def test_clears(self):
        output = run_repl(LOAD_SAMPLE + "run canonicalize\nreset\nir\nquit\n")
        assert "IR state cleared" in output
        assert "(no IR state set)" in output


class TestIR:
    def test_no_state(self):
        output = run_repl("ir\nquit\n")
        assert "(no IR state set)" in output

    def test_with_state(self):
        output = run_repl(LOAD_SAMPLE + "ir\nquit\n")
        assert "arith.addf" in output


class TestPasses:
    def test_filter(self):
        output = run_repl("passes arith-to-llvm\nquit\n")
        assert "convert-arith-to-llvm" in output

    def test_no_match(self):
        output = run_repl("passes zzz-nonexistent\nquit\n")
        assert "(no passes matched)" in output


class TestMisc:
    def test_unknown_command(self):
        output = run_repl("badcommand\nquit\n")
        assert "Unknown command" in output

    def test_eof_exits(self):
        output = run_repl("")
        assert "mlir-opt-repl" in output

    def test_empty_lines_ignored(self):
        output = run_repl("\n\n\nquit\n")
        assert "mlir-opt-repl" in output

    def test_exit_alias(self):
        output = run_repl("exit\n")
        assert "mlir-opt-repl" in output
