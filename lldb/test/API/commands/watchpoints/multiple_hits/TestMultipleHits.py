"""
Test handling of cases when a single instruction triggers multiple watchpoints
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbwatchpointutils import *


class MultipleHitsTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipIf(
        bugnumber="llvm.org/pr30758",
        oslist=["linux"],
        archs=["arm$", "aarch64", "powerpc64le"],
    )
    @expectedFailureAll(archs="^riscv.*")
    @skipIfwatchOS
    def test_hw_watchpoint(self):
        self.do_test(WatchpointType.READ_WRITE, lldb.eWatchpointModeHardware)

    def test_sw_watchpoint(self):
        self.do_test(WatchpointType.MODIFY, lldb.eWatchpointModeSoftware)

    def do_test(self, wp_type, wp_mode):
        self.build()
        target = self.createTestTarget()

        bp = target.BreakpointCreateByName("main")
        self.assertTrue(bp and bp.IsValid(), "Breakpoint is valid")

        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertState(process.GetState(), lldb.eStateStopped)

        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertIsNotNone(thread)

        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame and frame.IsValid(), "Frame is valid")

        buf = frame.FindValue("buf", lldb.eValueTypeVariableGlobal)
        self.assertTrue(buf and buf.IsValid(), "buf is valid")

        for i in [0, target.GetAddressByteSize()]:
            member = buf.GetChildAtIndex(i)
            self.assertTrue(member and member.IsValid(), "member is valid")

            error = lldb.SBError()
            watch = set_watchpoint_at_value(member, wp_type, wp_mode, error)
            self.assertSuccess(error)

        process.Continue()
        self.assertState(process.GetState(), lldb.eStateStopped)
        self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonWatchpoint)
