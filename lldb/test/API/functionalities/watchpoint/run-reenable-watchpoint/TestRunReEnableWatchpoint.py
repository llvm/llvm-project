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
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        self.assertGreater(target.GetNumModules(), 0)

        src = lldb.SBFileSpec("main.c")
        bkpt1 = target.BreakpointCreateBySourceRegex("break here launch1", src)
        self.assertEqual(bkpt1.GetNumLocations(), 1)
        self.assertTrue(bkpt1.GetLocationAtIndex(0).IsEnabled())

        self.runCmd("process launch --disable-aslr true")
        process = target.GetProcess()
        self.assertTrue(process.IsValid())
        self.assertEqual(process.GetState(), lldb.eStateStopped)
        thread = process.GetSelectedThread()
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonBreakpoint)

        frame = thread.GetFrameAtIndex(0)
        self.runCmd("watch set variable arr")

        reason = self.continue_and_report_stop_reason(process, "continue first-launch")
        self.assertEqual(reason, lldb.eStopReasonWatchpoint)

        var_list = target.FindGlobalVariables("arr", 1)
        self.assertEqual(var_list.GetSize(), 1)
        arr = var_list.GetValueAtIndex(0)
        self.assertEqual(arr.GetNumChildren(), 6)
        self.assertEqual(arr.GetChildAtIndex(0).GetValueAsUnsigned(), 2)

        target.DeleteAllBreakpoints()
        process.Kill()

        bkpt2 = target.BreakpointCreateBySourceRegex("break here launch2", src)
        self.assertEqual(bkpt2.GetNumLocations(), 1)
        self.assertTrue(bkpt2.GetLocationAtIndex(0).IsEnabled())

        self.runCmd("process launch --disable-aslr true")
        process = target.GetProcess()
        self.assertTrue(process.IsValid())
        self.assertEqual(process.GetState(), lldb.eStateStopped)
        thread = process.GetSelectedThread()
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonBreakpoint)

        target.EnableAllWatchpoints()

        reason = self.continue_and_report_stop_reason(process, "continue second-launch")
        self.assertEqual(reason, lldb.eStopReasonWatchpoint)

        var_list = target.FindGlobalVariables("arr", 1)
        self.assertEqual(var_list.GetSize(), 1)
        arr = var_list.GetValueAtIndex(0)
        self.assertEqual(arr.GetNumChildren(), 6)
        self.assertEqual(arr.GetChildAtIndex(0).GetValueAsUnsigned(), 4)
