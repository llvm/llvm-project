"""
Test handling of cases when a single instruction triggers multiple watchpoints
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MultipleHitsTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipIf(
        bugnumber="llvm.org/pr30758",
        oslist=["linux"],
        archs=["arm$", "aarch64", "powerpc64le"],
    )
    @skipIfwatchOS
    def test(self):
        self.build()
        target, process, thread, _ = lldbutil.run_to_name_breakpoint(self, "main")

        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame and frame.IsValid(), "Frame is valid")

        buf = frame.FindValue("buf", lldb.eValueTypeVariableGlobal)
        self.assertTrue(buf and buf.IsValid(), "buf is valid")

        for i in [0, target.GetAddressByteSize()]:
            member = buf.GetChildAtIndex(i)
            self.assertTrue(member and member.IsValid(), "member is valid")

            error = lldb.SBError()
            watch = member.Watch(True, True, True, error)
            self.assertSuccess(error)

        process.Continue()
        self.assertState(process.GetState(), lldb.eStateStopped)
        self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonWatchpoint)
