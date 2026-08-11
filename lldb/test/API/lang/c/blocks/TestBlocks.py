"""Test that lldb can invoke blocks and read variables captured by blocks."""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


@skipIfWasm  # no expression evaluation
class BlocksTestCase(TestBase):
    @skipUnlessDarwin
    def test(self):
        self.build()
        src = lldb.SBFileSpec("main.c")
        _, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "Set breakpoint 0 here.", src
        )

        # Inside the 'add' block we can read its arguments and the captured
        # variable 'c'.
        self.expect_expr("a + b", result_type="int", result_value="7")
        self.expect_expr("c", result_type="int", result_value="1")

        # Calling a block from the expression evaluator works.
        lldbutil.continue_to_source_breakpoint(
            self, process, "Set breakpoint 1 here.", src
        )
        self.expect_expr("(int)neg(-12)", result_type="int", result_value="12")

        # A block taking a struct argument can be called from the evaluator.
        lldbutil.continue_to_source_breakpoint(
            self, process, "Set breakpoint 2 here.", src
        )
        self.expect_expr("h(cg)", result_type="int", result_value="42")

        # Inside a block we can read captured variables of various types.
        lldbutil.continue_to_source_breakpoint(
            self, process, "Set breakpoint 3 here.", src
        )
        self.expect_var_path("captured_char", type="char", value="'a'")
        self.expect_var_path("captured_int", type="int", value="42")
        self.expect_var_path("captured_double", type="double", value="3.5")
        self.expect_var_path("*captured_ptr", type="int", value="42")
        self.expect_var_path("captured_struct.x", type="int", value="10")
        self.expect_var_path("captured_struct.y", type="int", value="20")

        # The same captured variables are also reachable from the expression
        # evaluator.
        self.expect_expr("captured_char", result_type="char", result_value="'a'")
        self.expect_expr("captured_int", result_type="int", result_value="42")
        self.expect_expr("captured_double", result_type="double", result_value="3.5")
        self.expect_expr("*captured_ptr", result_type="int", result_value="42")
        self.expect_expr("captured_struct.x", result_type="int", result_value="10")

    @skipUnlessDarwin
    def test_define(self):
        """Test defining and calling a block from the expression evaluator."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "Set breakpoint 0 here.", lldb.SBFileSpec("main.c")
        )

        self.runCmd(
            "expression int (^$add)(int, int) = ^int(int a, int b) { return a + b; };"
        )
        self.expect_expr("$add(2,3)", result_type="int", result_value="5")

        # Blocks defined in the expression evaluator cannot capture persistent
        # variables.
        self.runCmd("expression int $a = 3")
        self.expect(
            "expression int (^$addA)(int) = ^int(int b) { return $a + b; };",
            "Proper error is reported on capture",
            error=True,
        )
