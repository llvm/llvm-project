"""
Test stepping out of an inline frame that is not frame zero.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestThreadStepOutInline(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def step_out_of_frame(self, frame_idx, function, marker):
        self.build()
        _, _, thread, _ = lldbutil.run_to_name_breakpoint(self, "sink")
        self.assertTrue(thread.GetFrameAtIndex(frame_idx).IsInlined())

        thread.StepOutOfFrame(thread.GetFrameAtIndex(frame_idx))

        self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonPlanComplete)
        frame = thread.GetFrameAtIndex(0)
        self.assertEqual(frame.GetFunctionName(), function)
        self.assertEqual(frame.GetLineEntry().GetLine(), line_number("main.c", marker))

    def test_step_out_of_frame_1(self):
        self.step_out_of_frame(1, "level1", "// after level2")

    def test_step_out_of_frame_2(self):
        self.step_out_of_frame(2, "main", "// after level1")
