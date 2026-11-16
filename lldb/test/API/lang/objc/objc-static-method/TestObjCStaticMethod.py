"""Test calling functions in static methods."""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestObjCStaticMethod(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers to break inside main().
        self.main_source = "static.m"
        self.break_line = line_number(self.main_source, "// Set breakpoint here.")

    @add_test_categories(["pyapi"])
    # <rdar://problem/9745789> "expression" can't call functions in class methods
    def test_with_python_api(self):
        """Test calling functions in static methods."""
        self.build()
        _, _, thread, _ = lldbutil.run_to_line_breakpoint(
            self, lldb.SBFileSpec(self.main_source), self.break_line
        )

        # Now make sure we can call a function in the static method we've
        # stopped in.
        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame, "Got a valid frame 0 frame.")

        cmd_value = frame.EvaluateExpression("(char *) sel_getName (_cmd)")
        self.assertTrue(cmd_value.IsValid())
        sel_name = cmd_value.GetSummary()
        self.assertEqual(
            sel_name,
            '"doSomethingWithString:"',
            "Got the right value for the selector as string.",
        )

        cmd_value = frame.EvaluateExpression("[self doSomethingElseWithString:string]")
        self.assertTrue(cmd_value.IsValid())
        string_length = cmd_value.GetValueAsUnsigned()
        self.assertEqual(
            string_length,
            27,
            "Got the right value from another class method on the same class.",
        )
