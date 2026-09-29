"""
Test thread until with inlined functions.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestInlineFrameUntil(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def run_to_sink(self):
        self.build()
        _, _, thread, _ = lldbutil.run_to_name_breakpoint(self, "sink")
        return thread

    def until_from_frame(self, frame_idx, marker):
        thread = self.run_to_sink()
        self.runCmd(f"thread until -f {frame_idx} {line_number('main.c', marker)}")
        return thread.GetFrameAtIndex(0)

    def check_stops_at(self, frame_idx, marker):
        frame = self.until_from_frame(frame_idx, marker)
        self.assertEqual(frame.GetLineEntry().GetLine(), line_number("main.c", marker))

    def check_rejected(self, frame_idx, marker):
        self.run_to_sink()
        self.expect(
            f"thread until -f {frame_idx} {line_number('main.c', marker)}",
            error=True,
            substrs=["Until target outside of the current function"],
        )

    def test_until_in_same_inline_frame(self):
        self.check_stops_at(1, "// until in level3")

    def test_until_in_parent_inline_frame(self):
        self.check_stops_at(2, "// until in level2")

    # If an inlined frame is frame zero and thread until targets a "continue"
    # line, there are two breakpoints there: one for the until breakpoint, one
    # for the step over range. The step over range explains the stop first, so
    # the until plan misses it.
    # def test_until_jump_in_inline_frame(self):
    #     self.check_stops_at(1, "// until jump in level3")

    def test_until_target_already_ran(self):
        frame = self.until_from_frame(1, "// before sink")
        self.assertEqual(frame.GetFunctionName(), "level2")

    def test_until_return_address_in_inline_frame(self):
        thread = self.run_to_sink()
        return_address = thread.GetFrameAtIndex(1).GetPC()
        self.runCmd(f"thread until -f 1 -a {return_address}")
        frame = thread.GetFrameAtIndex(0)
        self.assertEqual(frame.GetFunctionName(), "level3")
        self.assertEqual(frame.GetPC(), return_address)

    def test_until_in_inlining_caller(self):
        self.check_rejected(1, "// until in level2")

    def test_until_in_inlined_callee(self):
        self.check_rejected(2, "// until in level3")

    def test_step_over_until_in_inlining_caller(self):
        thread = self.run_to_sink()
        error = thread.StepOverUntil(
            thread.GetFrameAtIndex(1),
            lldb.SBFileSpec("main.c"),
            line_number("main.c", "// until in level2"),
        )
        self.assertIn(
            "Until target outside of the current function", error.GetCString()
        )
