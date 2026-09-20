"""Test calling methods on super."""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestObjCSuperMethod(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers to break inside main().
        self.main_source = "class.m"
        self.break_line = line_number(self.main_source, "// Set breakpoint here.")

    @add_test_categories(["pyapi"])
    def test_with_python_api(self):
        """Test calling methods on super."""
        self.build()
        _, _, thread, _ = lldbutil.run_to_line_breakpoint(
            self, lldb.SBFileSpec(self.main_source), self.break_line
        )

        # Now make sure we can call a function in the class method we've
        # stopped in.
        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame, "Got a valid frame 0 frame.")

        cmd_value = frame.EvaluateExpression("[self get]")
        self.assertTrue(cmd_value.IsValid())
        self.assertEqual(cmd_value.GetValueAsUnsigned(), 2)

        cmd_value = frame.EvaluateExpression("[super get]")
        self.assertTrue(cmd_value.IsValid())
        self.assertEqual(cmd_value.GetValueAsUnsigned(), 1)
