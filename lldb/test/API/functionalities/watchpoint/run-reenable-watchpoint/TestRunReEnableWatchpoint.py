"""
Test that a watchpoint created in one Process can be
re-enabled in a second Process launch and behave
correctly.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class RunReEnableWatchpointTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def continue_and_report_stop_reason(self, process, iter_str):
        process.Continue()
        self.assertIn(
            process.GetState(), [lldb.eStateStopped, lldb.eStateExited], iter_str
        )
        thread = process.GetSelectedThread()
        return thread.GetStopReason()

    # We must be able to launch the inferior with ASLR disabled
    # so the static array lands at the same address after relaunch.
    @skipUnlessPlatform(["macosx"])
    def test_rerun_enable_watchpoint(self):
        """Test set watchpoint, re-run, re-enable wp, hit it."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")
        li = lldb.SBLaunchInfo(None)
        li.SetLaunchFlags(lldb.eLaunchFlagDisableASLR)
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "break here", self.main_source_file, launch_info=li
        )

        frame = thread.GetFrameAtIndex(0)
        self.runCmd("watch set variable arr")

        reason = self.continue_and_report_stop_reason(process, "continue first-launch")
        self.assertEqual(reason, lldb.eStopReasonWatchpoint)

        process.Kill()
        self.runCmd("process launch --disable-aslr true")
        process = target.GetProcess()
        self.assertTrue(process.IsValid())

        target.EnableAllWatchpoints()

        reason = self.continue_and_report_stop_reason(process, "continue second-launch")
        self.assertEqual(reason, lldb.eStopReasonWatchpoint)
