"""
Test lldb-dap 'stopped' events.
"""

from typing import Sequence

from lldbsuite.test.decorators import (
    expectedFailureAll,
    expectedFailureNetBSD,
    skipIfLinux,
    skipIfWindows,
)
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase, DAPTestSession
from lldbsuite.test.tools.lldb_dap.types import LaunchArgs, StoppedEvent, ThreadsArgs


@skipIfWindows  # This is flakey on Windows: llvm.org/pr24668, llvm.org/pr38373
@skipIfLinux
class TestDAP_stopped_events(DAPTestCaseBase):
    """
    Test validates different operations that produce 'stopped' events.
    """

    def collect_two_stops(self, session: DAPTestSession, *, after):
        """Collect two consecutive StoppedEvents (one per thread that hit the
        breakpoint). Both threads release the barrier together so the events
        arrive close in time"""
        first = session.wait_for_stopped_event(
            after=after, timeout_msg="waiting for first stop"
        )
        second = session.wait_for_stopped_event(
            after=first, timeout_msg="waiting for second stop"
        )

        self.assertNotEqual(
            first.body.threadId,
            second.body.threadId,
            "expected stopped event on different threads.",
        )
        return first, second

    def verify_threads(
        self, session: DAPTestSession, stopped_events: Sequence[StoppedEvent]
    ):
        """Verify the threads response and the focused-thread invariants.

        One of `stopped_events` must carry threads[1].id as its threadId.
        That event must not set preserveFocusHint=True.

        Should return something like:
         Process 1234 stopped
           thread #1: tid = 0x01, 0x0a libsystem_pthread.dylib`pthread_mutex_lock + 12, queue = 'com.apple.main-thread'
         * thread #2: tid = 0x02, 0x0b a.out`add(a=1, b=2) at main.cpp:10:32, stop reason = breakpoint 1.1 2.1
           thread #3: tid = 0x03, 0x0c a.out`add(a=4, b=5) at main.cpp:10:32, stop reason = breakpoint 1.1 2.1
        """
        threads = session.send_request(ThreadsArgs()).result().body.threads
        expected_count = 3
        self.assertGreaterEqual(
            len(threads),
            expected_count,
            f"expected at least {expected_count} threads, got {threads!r}",
        )

        focused_tid = threads[1].id
        events_by_tid = {e.body.threadId: e for e in stopped_events}
        focused_event = self.expect_not_none(
            events_by_tid.get(focused_tid),
            f"expected a stopped event for focused thread {focused_tid}",
        )
        self.assertNotEqual(
            focused_event.body.preserveFocusHint,
            True,
            "focused thread's stopped event must not set preserveFocusHint=True",
        )

    @expectedFailureAll(
        oslist=["freebsd"],
        bugnumber="llvm.org/pr18190 thread states not properly maintained",
    )
    @expectedFailureNetBSD
    def test_multiple_threads_same_breakpoint(self):
        """
        Test that multiple threads being stopped on the same breakpoint
        produces multiple 'stopped' events.
        """
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        with session.configure(LaunchArgs(program)) as ctx:
            [bp] = session.resolve_function_breakpoints(["my_add"])
        process_event = ctx.process_event
        first, second = self.collect_two_stops(session, after=process_event)

        for event in (first, second):
            body = event.body
            self.assertEqual(body.reason, "breakpoint")
            self.assertEqual(body.text, "breakpoint 1.1")
            self.assertEqual(body.description, "breakpoint 1.1")
            self.assertEqual(body.hitBreakpointIds, [bp])
            self.assertIsNotNone(body.threadId)

        self.verify_threads(session, (first, second))
        session.continue_to_exit()

    @expectedFailureAll(
        oslist=["freebsd"],
        bugnumber="llvm.org/pr18190 thread states not properly maintained",
    )
    @expectedFailureNetBSD
    def test_multiple_breakpoints_same_location(self):
        """
        Test stopping at a location that reports multiple overlapping
        breakpoints.
        """
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        source = self.getSourcePath("main.cpp")
        bp_line = line_number(source, "breakpoint")

        with session.configure(LaunchArgs(program)) as ctx:
            [bp1] = session.resolve_source_breakpoints(source, [bp_line])
            [bp2] = session.resolve_function_breakpoints(["my_add"])
        first, second = self.collect_two_stops(session, after=ctx.process_event)

        for event in (first, second):
            body = event.body
            self.assertEqual(body.reason, "breakpoint")
            self.assertEqual(body.text, "breakpoint 1.1 2.1")
            self.assertEqual(body.description, "breakpoint 1.1 2.1")
            self.assertEqual(body.hitBreakpointIds, [bp1, bp2])
            self.assertIsNotNone(body.threadId)

        self.verify_threads(session, (first, second))
        session.continue_to_exit()
