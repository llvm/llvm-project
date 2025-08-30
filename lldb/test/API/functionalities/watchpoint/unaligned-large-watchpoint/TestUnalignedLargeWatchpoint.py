"""
Watch a large unaligned memory region that
lldb will need multiple hardware watchpoints
to cover.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbwatchpointutils import *


class UnalignedLargeWatchpointTestCase(TestBase):
    def continue_and_report_stop_reason(self, process, iter_str):
        if self.TraceOn():
            self.runCmd("script print('continue')")
        process.Continue()
        self.assertIn(
            process.GetState(), [lldb.eStateStopped, lldb.eStateExited], iter_str
        )
        thread = process.GetSelectedThread()
        return thread.GetStopReason()

    NO_DEBUG_INFO_TESTCASE = True

    # The Windows process plugins haven't been updated to break
    # watchpoints into WatchpointResources yet.
    @skipIfWindows

    # Test on 64-bit targets where we probably have
    # four watchpoint registers that can watch doublewords (8-byte).
    @skipIf(archs=no_match(["arm64", "arm64e", "aarch64", "x86_64"]))
    def test_unaligned_large_hw_watchpoint(self):
        self.do_unaligned_large_watchpoint(
            WatchpointType.MODIFY, lldb.eWatchpointModeHardware
        )

    def test_unaligned_large_sw_watchpoint(self):
        self.do_unaligned_large_watchpoint(
            WatchpointType.MODIFY, lldb.eWatchpointModeSoftware
        )

    def do_unaligned_large_watchpoint(self, wp_type, wp_mode):
        """Test watching an unaligned region of memory that requires multiple watchpoints."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "break here", self.main_source_file
        )
        self.runCmd("break set -p done")
        self.runCmd("break set -p exiting")

        frame = thread.GetFrameAtIndex(0)

        array_addr = frame.GetValueForVariablePath("array").GetValueAsUnsigned()

        # Don't assume that the heap allocated array is aligned
        # to a 1024 byte boundary to begin with, force alignment.
        # wa_addr = (array_addr + 1024) & ~(1024 - 1)
        wa_addr = array_addr

        # Now make the start address unaligned.
        wa_addr = wa_addr + 7

        err = lldb.SBError()
        wp = set_watchpoint_by_address(target, wa_addr, 22, wp_type, wp_mode, err)
        self.assertTrue(wp.IsValid())
        self.assertSuccess(err)
        if self.TraceOn():
            self.runCmd("watch list -v")

        c_count = 0
        reason = self.continue_and_report_stop_reason(process, "continue #%d" % c_count)
        while reason == lldb.eStopReasonWatchpoint:
            c_count = c_count + 1
            reason = self.continue_and_report_stop_reason(
                process, "continue #%d" % c_count
            )
            self.assertLessEqual(c_count, 22)

        self.assertEqual(c_count, 22)
        self.expect("watchpoint list -v", substrs=["hit_count = 22"])
        self.assertEqual(wp.GetHitCount(), 22)

        target.DeleteWatchpoint(wp.GetID())

        # Now try watching a 16 byte variable
        # (not unaligned, but a good check to do anyway)
        frame0 = thread.GetFrameAtIndex(0)
        value = frame0.FindValue("variable", lldb.eValueTypeVariableLocal)
        err = lldb.SBError()
        wp = set_watchpoint_at_value(value, wp_type, wp_mode, err)
        self.assertTrue(
            value and wp, "Successfully found the variable and set a watchpoint"
        )
        self.assertSuccess(err)
        if self.TraceOn():
            self.runCmd("frame select 0")
            self.runCmd("watchpoint list")

        c_count = 0
        reason = self.continue_and_report_stop_reason(process, "continue #%d" % c_count)
        while reason == lldb.eStopReasonWatchpoint:
            c_count = c_count + 1
            reason = self.continue_and_report_stop_reason(
                process, "continue #%d" % c_count
            )
            self.assertLessEqual(c_count, 4)

        if self.TraceOn():
            self.runCmd("frame select 0")

        self.assertEqual(c_count, 4)
        self.expect("watchpoint list -v", substrs=["hit_count = 4"])
        self.assertEqual(wp.GetHitCount(), 4)
