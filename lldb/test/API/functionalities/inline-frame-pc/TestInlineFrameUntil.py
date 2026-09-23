"""
Test thread until from an inline frame.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestInlineFrameUntil(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def until_from_frame(self, frame_idx, marker):
        self.build()
        _, _, thread, _ = lldbutil.run_to_name_breakpoint(self, "sink")
        self.runCmd(f"thread until -f {frame_idx} {line_number('main.c', marker)}")
        return thread.GetFrameAtIndex(0)

    def check_stops_at(self, frame_idx, marker):
        frame = self.until_from_frame(frame_idx, marker)
        self.assertEqual(frame.GetLineEntry().GetLine(), line_number("main.c", marker))

    def test_until_in_same_inline_frame(self):
        self.check_stops_at(1, "// until in level3")

    def test_until_in_parent_inline_frame(self):
        self.check_stops_at(2, "// until in level2")

    def test_until_in_inlining_caller(self):
        self.check_stops_at(1, "// until in level2")

    def test_until_in_inlined_callee(self):
        self.check_stops_at(2, "// until in level3")

    def test_until_target_already_ran(self):
        frame = self.until_from_frame(1, "// before sink")
        self.assertEqual(frame.GetFunctionName(), "main")
