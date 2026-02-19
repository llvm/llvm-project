"""
Test that expressions that call functions which fork
can be evaluated successfully.

This tests the ThreadPlanCallFunction handling of fork/vfork/vforkdone
stop reasons, which should be silently resumed rather than causing the
expression evaluation to fail.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ExprWithForkTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipIfWindows
    @add_test_categories(["fork"])
    def test_expr_with_fork(self):
        """Test that expression evaluation succeeds when the expression calls fork()."""
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        # Evaluate an expression that calls fork() inside a user function.
        # The fork will generate a fork stop event which ThreadPlanCallFunction
        # must handle transparently for the expression to complete.
        self.expect_expr(
            "fork_and_return(42)", result_type="int", result_value="42"
        )
