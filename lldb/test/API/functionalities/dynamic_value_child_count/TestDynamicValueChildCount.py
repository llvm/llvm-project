"""
Test that dynamic values update their child count correctly
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class DynamicValueChildCountTestCase(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

        # Find the line number to break for main.c.

        self.main_third_call_line = line_number(
            "pass-to-base.cpp", "// Break here and check b has 0 children"
        )
        self.main_fourth_call_line = line_number(
            "pass-to-base.cpp", "// Break here and check b still has 0 children"
        )
        self.main_fifth_call_line = line_number(
            "pass-to-base.cpp", "// Break here and check b has one child now"
        )
        self.main_sixth_call_line = line_number(
            "pass-to-base.cpp", "// Break here and check b has 0 children again"
        )

    @add_test_categories(["pyapi"])
    def test_get_dynamic_vals(self):
        """Test fetching C++ dynamic values from pointers & references."""
        """Get argument vals for the call stack when stopped on a breakpoint."""
        self.build()

        # Set up our breakpoints:

        target, _, _, _ = lldbutil.run_to_line_breakpoint(
            self, lldb.SBFileSpec("pass-to-base.cpp", False), self.main_third_call_line
        )
        fourth_call_bpt = target.BreakpointCreateByLocation(
            "pass-to-base.cpp", self.main_fourth_call_line
        )
        self.assertTrue(fourth_call_bpt, VALID_BREAKPOINT)
        fifth_call_bpt = target.BreakpointCreateByLocation(
            "pass-to-base.cpp", self.main_fifth_call_line
        )
        self.assertTrue(fifth_call_bpt, VALID_BREAKPOINT)
        sixth_call_bpt = target.BreakpointCreateByLocation(
            "pass-to-base.cpp", self.main_sixth_call_line
        )
        self.assertTrue(sixth_call_bpt, VALID_BREAKPOINT)

        b = self.frame().FindVariable("b").GetDynamicValue(lldb.eDynamicCanRunTarget)
        self.assertEqual(b.GetNumChildren(), 0, "b has 0 children")
        self.runCmd("continue")
        self.assertEqual(b.GetNumChildren(), 0, "b still has 0 children")
        self.runCmd("continue")
        self.assertNotEqual(b.GetNumChildren(), 0, "b now has 1 child")
        self.runCmd("continue")
        self.assertEqual(b.GetNumChildren(), 0, "b didn't go back to 0 children")
