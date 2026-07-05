"""
Test breakpoint hit count is reset when target runs.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


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

        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateBySourceRegex(
            self.BREAKPOINT_TEXT, lldb.SBFileSpec("main.cpp")
        )
        self.assertTrue(
            breakpoint.IsValid() and breakpoint.GetNumLocations() == 1, VALID_BREAKPOINT
        )

        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        from lldbsuite.test.lldbutil import get_stopped_thread

        # Verify 1st breakpoint location is hit.
        thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(
            thread.IsValid(), "There should be a thread stopped due to breakpoint"
        )
        self.check_stopped_at_breakpoint_and_hit_once(thread, breakpoint)

        # Relaunch
        process.Kill()
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # Verify the hit counts are still one.
        thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(
            thread.IsValid(), "There should be a thread stopped due to breakpoint"
        )
        self.check_stopped_at_breakpoint_and_hit_once(thread, breakpoint)
