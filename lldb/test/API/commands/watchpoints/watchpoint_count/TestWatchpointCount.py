import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestWatchpointCount(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)

    @skipIf(oslist=["freebsd", "linux"], archs=["arm", "aarch64"],
            bugnumber="llvm.org/pr26031")
    def test_watchpoint_count(self):
        self.build()
        (_, process, thread, _) = lldbutil.run_to_source_breakpoint(self, "patatino", lldb.SBFileSpec("main.c"))
        frame = thread.GetFrameAtIndex(0)
        first_var = frame.FindVariable("x1")
        second_var = frame.FindVariable("x2")

        error = lldb.SBError()
        first_watch = first_var.Watch(True, False, True, error)
        if not error.Success():
            self.fail(
                "Failed to make watchpoint for x1: %s" %
                (error.GetCString()))

        second_watch = second_var.Watch(True, False, True, error)
        if not error.Success():
            self.fail(
                "Failed to make watchpoint for x2: %s" %
                (error.GetCString()))
        process.Continue()

        stop_reason = thread.GetStopReason()
        self.assertEqual(stop_reason, lldb.eStopReasonWatchpoint, "watchpoint for x1 not hit")
        stop_reason_descr = thread.GetStopDescription(256)
        self.assertEqual(stop_reason_descr, "watchpoint 1")

        process.Continue()
        stop_reason = thread.GetStopReason()
        self.assertEqual(stop_reason, lldb.eStopReasonWatchpoint, "watchpoint for x2 not hit")
        stop_reason_descr = thread.GetStopDescription(256)
        self.assertEqual(stop_reason_descr, "watchpoint 2")
