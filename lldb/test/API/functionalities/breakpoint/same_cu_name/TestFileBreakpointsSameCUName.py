"""
Test setting a breakpoint by file and line when many instances of the
same file name exist in the CU list.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestBreakpointSameCU(TestBase):
    def test_breakpoint_same_cu(self):
        self.build()
        target = self.createTestTarget()

        # Break both on the line before the code:
        comment_line = line_number("common.cpp", "// A comment here")
        self.assertNotEqual(comment_line, 0, "line_number worked")
        bkpt = target.BreakpointCreateByLocation("common.cpp", comment_line)
        self.assertEqual(
            bkpt.GetNumLocations(), 4, "Got the right number of breakpoints"
        )

        # And break on the code, both should work:
        code_line = line_number("common.cpp", "// The line with code")
        self.assertNotEqual(comment_line, 0, "line_number worked again")
        bkpt = target.BreakpointCreateByLocation("common.cpp", code_line)
        self.assertEqual(
            bkpt.GetNumLocations(), 4, "Got the right number of breakpoints"
        )
