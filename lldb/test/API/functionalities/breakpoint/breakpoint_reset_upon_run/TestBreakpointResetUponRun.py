"""
Test breakpoint hit count is reset when target runs.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class HitcountResetUponRun(TestBase):
    BREAKPOINT_TEXT = "Set a breakpoint here"

    def check_stopped_at_breakpoint_and_hit_once(self, thread, breakpoint):
        frame0 = thread.GetFrameAtIndex(0)
        location1 = breakpoint.FindLocationByAddress(frame0.GetPC())
        self.assertTrue(location1)
        self.assertEqual(location1.GetHitCount(), 1)
        self.assertEqual(breakpoint.GetHitCount(), 1)

    def test_hitcount_reset_upon_run(self):
        self.build()

        # Verify 1st breakpoint location is hit.
        target, process, thread, breakpoint = lldbutil.run_to_source_breakpoint(
            self, self.BREAKPOINT_TEXT, lldb.SBFileSpec("main.cpp")
        )
        self.assertEqual(breakpoint.GetNumLocations(), 1)
        self.check_stopped_at_breakpoint_and_hit_once(thread, breakpoint)

        # Relaunch
        process.Kill()

        # Verify the hit counts are still one.
        _, process, thread, _ = lldbutil.run_to_breakpoint_do_run(
            self, target, breakpoint
        )
        self.check_stopped_at_breakpoint_and_hit_once(thread, breakpoint)
