import time

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbwatchpointutils import *


class TestWatchpointCount(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipIf(
        oslist=["freebsd", "linux"],
        archs=["arm$", "aarch64"],
        bugnumber="llvm.org/pr26031",
    )
    @expectedFailureAll(archs="^riscv.*")
    def test_hw_watchpoint_count(self):
        self.do_watchpoint_count(WatchpointType.MODIFY, lldb.eWatchpointModeHardware)

    def test_sw_watchpoint_count(self):
        self.do_watchpoint_count(WatchpointType.MODIFY, lldb.eWatchpointModeSoftware)

    def do_watchpoint_count(self, wp_type, wp_mode):
        self.build()
        (_, process, thread, _) = lldbutil.run_to_source_breakpoint(
            self, "patatino", lldb.SBFileSpec("main.c")
        )
        frame = thread.GetFrameAtIndex(0)
        first_var = frame.FindVariable("x1")
        second_var = frame.FindVariable("x2")

        error = lldb.SBError()
        first_watch = set_watchpoint_at_value(first_var, wp_type, wp_mode, error)
        if not error.Success():
            self.fail("Failed to make watchpoint for x1: %s" % (error.GetCString()))

        second_watch = set_watchpoint_at_value(second_var, wp_type, wp_mode, error)
        if not error.Success():
            self.fail("Failed to make watchpoint for x2: %s" % (error.GetCString()))
        process.Continue()

        stop_reason = thread.GetStopReason()
        self.assertStopReason(
            stop_reason, lldb.eStopReasonWatchpoint, "watchpoint for x1 not hit"
        )
        stop_reason_descr = thread.stop_description
        self.assertEqual(stop_reason_descr, "watchpoint 1")

        process.Continue()
        stop_reason = thread.GetStopReason()
        self.assertStopReason(
            stop_reason, lldb.eStopReasonWatchpoint, "watchpoint for x2 not hit"
        )
        stop_reason_descr = thread.stop_description
        self.assertEqual(stop_reason_descr, "watchpoint 2")
