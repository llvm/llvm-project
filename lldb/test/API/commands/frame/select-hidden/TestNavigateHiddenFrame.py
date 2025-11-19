import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class NavigateHiddenFrameTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @add_test_categories(["libc++"])
    def test(self):
        """Test going up/down a backtrace but we started in a hidden frame."""
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Break here", lldb.SBFileSpec("main.cpp")
        )
        # up
        self.assertIn("__impl2", thread.selected_frame.GetFunctionName())
        self.expect("up")
        self.assertIn("__impl1", thread.selected_frame.GetFunctionName())
        self.expect("up")
        self.assertIn("__impl", thread.selected_frame.GetFunctionName())
        self.expect("up")
        self.assertIn("non_impl", thread.selected_frame.GetFunctionName())

        # Back down again.
        self.expect("down")
        self.assertIn("__impl", thread.selected_frame.GetFunctionName())
        self.expect("down")
        self.assertIn("__impl1", thread.selected_frame.GetFunctionName())
        self.expect("down")
        self.assertIn("__impl2", thread.selected_frame.GetFunctionName())
