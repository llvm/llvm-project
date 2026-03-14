"""
Test handling of the situation where the main thread exits but the other threads
in the process keep running.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class ThreadExitTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    # Needs os-specific implementation in the inferior
    @skipIf(oslist=no_match(["linux"]))
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        # There should be one (non-main) thread left
        self.assertEqual(self.process().GetNumThreads(), 1)

        # Ensure we can evaluate_expressions in this state
        self.expect_expr("call_me()", result_value="12345")

        self.runCmd("continue")
        # Exit code depends on the version of the linux kernel
        self.assertIn(self.process().GetExitStatus(), [42, 47])
