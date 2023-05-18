"""
Test calling a function, stopping in the call, continue and gather the result on stop.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ExprCommandCallStopContinueTestCase(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.c.

    def test(self):
        """Test gathering result from interrupted function call."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        lldbutil.run_break_set_by_file_and_line(
            self,
            "main.cpp",
            line_number("main.cpp", '{5, "five"}'),
            num_expected_locations=-1,
            loc_exact=True,
        )

        self.expect(
            "expr -i false -- returnsFive()",
            error=True,
            substrs=["Execution was interrupted, reason: breakpoint"],
        )

        self.runCmd("continue", "Continue completed")
        self.expect(
            "thread list",
            substrs=[
                "stop reason = User Expression thread plan",
                r'Completed expression: (Five) $0 = (number = 5, name = "five")',
            ],
        )
