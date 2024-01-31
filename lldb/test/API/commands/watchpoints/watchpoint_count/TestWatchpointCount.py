import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestWatchpointCount(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipIf(
        oslist=["freebsd", "linux"],
        archs=["arm", "aarch64"],
        bugnumber="llvm.org/pr26031",
    )
    def test_watchpoint_count(self):
        self.build()
        (_, process, thread, _) = lldbutil.run_to_source_breakpoint(
            self, "patatino", lldb.SBFileSpec("main.c")
        )
        frame = thread.GetFrameAtIndex(0)
        first_var = frame.FindVariable("x1")
        second_var = frame.FindVariable("x2")

        self.runCmd("log enable -v lldb watch")
        self.addTearDownHook(lambda: self.runCmd("log disable lldb watch"))

        error = lldb.SBError()
        first_watch = first_var.Watch(True, False, True, error)
        if not error.Success():
            self.fail("Failed to make watchpoint for x1: %s" % (error.GetCString()))

        second_watch = second_var.Watch(True, False, True, error)
        if not error.Success():
            self.fail("Failed to make watchpoint for x2: %s" % (error.GetCString()))
        # LWP_TODO: Adding temporary prints to debug a test
        # failure on the x86-64 Debian bot.
        self.runCmd("p &x1")
        self.runCmd("p &x2")
        self.runCmd("watchpoint list")
        self.runCmd("frame select 0")
        self.runCmd("bt")

        process.Continue()

        stop_reason = thread.GetStopReason()
        self.assertStopReason(
            stop_reason, lldb.eStopReasonWatchpoint, "watchpoint for x1 not hit"
        )
        stop_reason_descr = thread.GetStopDescription(256)
        self.assertEqual(stop_reason_descr, "watchpoint 1")

        process.Continue()
        # LWP_TODO: Adding temporary prints to debug a test
        # failure on the x86-64 Debian bot.
        self.runCmd("frame select 0")
        self.runCmd("bt")
        self.runCmd("disassemble")

        stop_reason = thread.GetStopReason()
        self.assertStopReason(
            stop_reason, lldb.eStopReasonWatchpoint, "watchpoint for x2 not hit"
        )
        stop_reason_descr = thread.GetStopDescription(256)
        self.assertEqual(stop_reason_descr, "watchpoint 2")
