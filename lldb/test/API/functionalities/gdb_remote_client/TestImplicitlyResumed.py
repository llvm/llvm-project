"""
Test that lldb takes a new stop info from a thread that it suspended, and asks
that thread whether to stop, when a stop reply lists it in the
"implicitly_resumed_threads" key.

lldb keeps the stop info of a thread that it suspended, and does not ask that
thread whether the process should stop. The threads of a SIMD group, such as
the lanes of a GPU warp, run together, so when lldb resumes one of them, the
server lists the others. Here threads 1 and 2 are the two lanes of one SIMD
group, and lldb runs one lane while the server lists the other.
"""

import json

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase

# Threads 1 and 2 are lanes 0 and 1 of this SIMD group.
SIMD = 48


def lane(tid, **stop_info):
    """The jstopinfo dictionary of thread tid."""
    return dict(tid=tid, lane=tid - 1, simd=SIMD, **stop_info)


def stop_reply(tid, thread_infos, fields, signal=5):
    """Stop threads 1 and 2 for thread tid, with thread_infos as jstopinfo."""
    return "T%02xthread:%x;lane:%d;simd:%d;threads:1,2;%sjstopinfo:%s;" % (
        signal,
        tid,
        tid - 1,
        SIMD,
        fields,
        json.dumps(thread_infos).encode().hex(),
    )


def exception(description):
    return "reason:exception;description:%s;" % description.encode().hex()


class SoloResumeResponder(MockGDBServerResponder):
    """Accepts breakpoints and vCont, and lists the vCont continue packets."""

    def resumes(self):
        return [p for p in self.packetLog.get_received() if p.startswith("vCont;c")]

    def setBreakpoint(self, packet):
        return "OK"

    def other(self, packet):
        if packet == "vCont?":
            return "vCont;c;C;s;S"
        return ""


class TestImplicitlyResumed(GDBRemoteTestBase):
    def connect_x86_64(self):
        # Breakpoints and thread PCs need an architecture.
        target = self.dbg.CreateTargetWithFileAndTargetTriple("", "x86_64-pc-linux")
        return self.connect(target)

    def test_new_stop_reason_stops_the_process(self):
        """
        lldb runs thread 1 alone. Its SIMD group runs thread 2 too, and thread 2
        raises another exception. Unless the stop reply lists thread 2, lldb
        keeps thread 2's exception from the first stop, finds no reason to stop,
        and resumes the process again.
        """

        class MyResponder(SoloResumeResponder):
            def haltReason(self):
                return stop_reply(
                    2,
                    [lane(1), lane(2, reason="exception", description="first")],
                    fields=exception("first"),
                )

            def vCont(self, packet):
                if len(self.resumes()) > 1:
                    return "W00"
                return stop_reply(
                    2,
                    [lane(1), lane(2, reason="exception", description="second")],
                    fields="implicitly_resumed_threads:2;" + exception("second"),
                )

        self.server.responder = MyResponder()
        process = self.connect_x86_64()
        thread_1 = process.GetThreadByID(1)
        thread_2 = process.GetThreadByID(2)
        self.assertEqual(thread_2.GetStopDescription(100), "first")

        # Run thread 1 alone, which suspends thread 2.
        error = lldb.SBError()
        thread_1.RunToAddress(0x1000, error)
        self.assertEqual(self.server.responder.resumes(), ["vCont;c:1"])
        self.assertSuccess(error)
        self.assertStopReason(thread_2.GetStopReason(), lldb.eStopReasonException)
        self.assertEqual(thread_2.GetStopDescription(100), "second")
        self.assertEqual(process.GetSelectedThread().GetThreadID(), 2)

    def test_list_without_jstopinfo(self):
        """
        As in test_new_stop_reason_stops_the_process, but the stop reply is for
        thread 1 and has no jstopinfo. lldb reads thread 2's new exception with
        qThreadStopInfo, and asks for it only because the list names thread 2.
        """

        class MyResponder(SoloResumeResponder):
            def haltReason(self):
                return stop_reply(
                    2,
                    [lane(1), lane(2, reason="exception", description="first")],
                    fields=exception("first"),
                )

            def vCont(self, packet):
                if len(self.resumes()) > 1:
                    return "W00"
                return (
                    "T00thread:1;lane:0;simd:%d;threads:1,2;"
                    "implicitly_resumed_threads:2;" % SIMD
                )

            def threadStopInfo(self, threadnum):
                if threadnum != 2:
                    return ""
                return "T05thread:2;lane:1;simd:%d;%s" % (SIMD, exception("second"))

        self.server.responder = MyResponder()
        process = self.connect_x86_64()
        thread_1 = process.GetThreadByID(1)
        thread_2 = process.GetThreadByID(2)
        self.assertEqual(thread_2.GetStopDescription(100), "first")

        # Run thread 1 alone, which suspends thread 2.
        error = lldb.SBError()
        thread_1.RunToAddress(0x1000, error)
        self.assertEqual(self.server.responder.resumes(), ["vCont;c:1"])
        self.assertSuccess(error)
        self.assertStopReason(thread_2.GetStopReason(), lldb.eStopReasonException)
        self.assertEqual(thread_2.GetStopDescription(100), "second")

    def test_new_stop_reason_of_last_thread_stops_the_process(self):
        """
        lldb runs thread 1 alone. Its SIMD group runs thread 2 too, thread 1
        exits, and thread 2 raises another exception. The stop reply is for
        thread 2, the only thread left, and lists it as implicitly resumed.
        """

        class MyResponder(SoloResumeResponder):
            def haltReason(self):
                return stop_reply(
                    2,
                    [lane(1), lane(2, reason="exception", description="first")],
                    fields=exception("first"),
                )

            def vCont(self, packet):
                if len(self.resumes()) > 1:
                    return "W00"
                fields = "implicitly_resumed_threads:2;" + exception("second")
                return "T05thread:2;lane:1;simd:%d;threads:2;%s" % (SIMD, fields)

            # lldb resumes every thread, here only thread 2, with "c".
            def cont(self):
                return "W00"

        self.server.responder = MyResponder()
        process = self.connect_x86_64()
        thread_1 = process.GetThreadByID(1)
        thread_2 = process.GetThreadByID(2)
        self.assertEqual(thread_2.GetStopDescription(100), "first")

        # Run thread 1 alone, which suspends thread 2.
        error = lldb.SBError()
        thread_1.RunToAddress(0x1000, error)
        self.assertEqual(self.server.responder.resumes(), ["vCont;c:1"])
        self.assertSuccess(error)
        self.assertState(process.GetState(), lldb.eStateStopped)
        self.assertEqual(process.GetNumThreads(), 1)
        self.assertStopReason(thread_2.GetStopReason(), lldb.eStopReasonException)
        self.assertEqual(thread_2.GetStopDescription(100), "second")

    def test_new_breakpoint_hit_is_counted(self):
        """
        Thread 2 hits a breakpoint. lldb runs thread 1 alone, its SIMD group
        runs thread 2 too, and thread 2 goes around a loop and hits the same
        breakpoint again. lldb keeps a thread's breakpoint hit while the thread
        is still at that breakpoint, so unless the stop reply lists thread 2,
        lldb takes the new hit for the old one: it does not count it, or
        evaluate the breakpoint's condition and commands again. Thread 2 is not
        the thread of the stop reply, so it gets its stop info from jstopinfo
        when lldb first asks for it.
        """

        class MyResponder(SoloResumeResponder):
            def haltReason(self):
                return stop_reply(1, [lane(1), lane(2)], fields="thread-pcs:1000,1ff0;")

            def cont(self):
                return stop_reply(
                    2,
                    [lane(1), lane(2, reason="breakpoint")],
                    fields="thread-pcs:1000,2000;reason:breakpoint;",
                )

            def vCont(self, packet):
                if len(self.resumes()) > 1:
                    return "W00"
                return stop_reply(
                    1,
                    [lane(1), lane(2, reason="breakpoint")],
                    signal=0,
                    fields="implicitly_resumed_threads:2;thread-pcs:1050,2000;",
                )

        self.server.responder = MyResponder()
        process = self.connect_x86_64()
        thread_1 = process.GetThreadByID(1)
        thread_2 = process.GetThreadByID(2)
        breakpoint = process.GetTarget().BreakpointCreateByAddress(0x2000)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)
        self.assertSuccess(process.Continue())
        self.assertStopReason(thread_2.GetStopReason(), lldb.eStopReasonBreakpoint)
        self.assertEqual(breakpoint.GetHitCount(), 1)

        # Run thread 1 alone, which suspends thread 2.
        error = lldb.SBError()
        thread_1.RunToAddress(0x1100, error)
        self.assertEqual(self.server.responder.resumes(), ["vCont;c:1"])
        self.assertSuccess(error)
        self.assertStopReason(thread_2.GetStopReason(), lldb.eStopReasonBreakpoint)
        self.assertEqual(breakpoint.GetHitCount(), 2)
        self.assertEqual(process.GetSelectedThread().GetThreadID(), 2)
