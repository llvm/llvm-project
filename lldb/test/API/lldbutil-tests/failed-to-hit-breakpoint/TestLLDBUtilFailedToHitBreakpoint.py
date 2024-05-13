"""
Tests lldbutil's behavior when running to a source breakpoint fails.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class LLDBUtilFailedToHitBreakpointTest(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @expectedFailureAll(oslist=["windows"])
    def test_error_message(self):
        """
        Tests that run_to_source_breakpoint prints the right error message
        when failing to hit the wanted breakpoint.
        """
        self.build()
        with self.assertRaisesRegex(
            AssertionError,
            "Test process is not stopped at breakpoint: state: exited, exit code: 0, stdout: 'stdout_needlestderr_needle'",
        ):
            lldbutil.run_to_source_breakpoint(
                self, "// break here", lldb.SBFileSpec("main.cpp")
            )
