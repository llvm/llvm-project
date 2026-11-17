"""
Test lldb-dap 'stopped' events.
"""

from typing import Sequence

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase, DAPTestSession
from lldbsuite.test.tools.lldb_dap.types import LaunchArgs, StoppedEvent, ThreadsArgs


# The in-process Windows plugin (ProcessWindows) reports one thread's exception
# per stop.
@skipIfWindows
@requireThreadSupport
class TestDAP_stopped_events(DAPTestCaseBase):
    """
    Test validates different operations that produce 'stopped' events.
    """

    # How often to launch the process to get both threads into one stop.
    MAX_LAUNCH_ATTEMPTS = 50

    def launch_until_threads_stop_together(self, set_breakpoints):
        """Launch the process until both threads hit the breakpoint in the same
        stop. Returns (session, breakpoint ids, first event, second event).

        Ideally both threads hit the breakpoint before the stop is reported,
        and lldb-dap sends one StoppedEvent per thread for that one stop.
        On a loaded machine, the second thread might reach the breakpoint only
        after the stop was reported, so the stop only has a single
        StoppedEvent. In that case, relaunch and try again.
        """
        self.build()
        program = self.getBuildArtifact("a.out")
        for attempt in range(1, self.MAX_LAUNCH_ATTEMPTS + 1):
            # The first attempt uses the default adapter.
            adapter = self.create_debug_adapter() if attempt > 1 else None
            session = self.create_session(adapter, disconnect_automatically=False)
            with session.configure(LaunchArgs(program)) as ctx:
                bp_ids = set_breakpoints(session)
            first = session.wait_for_stopped_event(
                after=ctx.process_event, timeout_msg="waiting for first stop"
            )
            # For each stop, lldb-dap sends one StoppedEvent per thread with a
            # stop reason (see SendThreadStoppedEvent). The focused thread's
            # event is always sent last and is the only one with
            # allThreadsStopped=True. So if the first event has
            # allThreadsStopped=False, at least one more event for this stop
            # follows, i.e. both threads hit the breakpoint. If it is True,
            # it was the only thread with a stop reason in this stop.
            if not first.body.allThreadsStopped:
                break
            self.logger.info(
                "Attempt %d: only one thread stopped at the breakpoint, relaunching.",
                attempt,
            )
            session.disconnect(terminateDebuggee=True)
            session.stop()
        else:
            self.fail(
                f"no stop with both threads at the breakpoint in "
                f"{self.MAX_LAUNCH_ATTEMPTS} launches"
            )

        second = session.wait_for_stopped_event(
            after=first, timeout_msg="waiting for second thread in same stop"
        )
        self.assertTrue(
            second.body.allThreadsStopped,
            "expected the second stopped event to complete the stop",
        )
        self.assertNotEqual(
            first.body.threadId,
            second.body.threadId,
            "expected stopped event on different threads.",
        )
        return session, bp_ids, first, second

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
        session, [bp], first, second = self.launch_until_threads_stop_together(
            lambda session: session.resolve_function_breakpoints(["my_add"])
        )

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
        source = self.getSourcePath("main.cpp")
        bp_line = line_number(source, "breakpoint")

        def set_breakpoints(session):
            [bp1] = session.resolve_source_breakpoints(source, [bp_line])
            [bp2] = session.resolve_function_breakpoints(["my_add"])
            return bp1, bp2

        session, (bp1, bp2), first, second = self.launch_until_threads_stop_together(
            set_breakpoints
        )

        for event in (first, second):
            body = event.body
            self.assertEqual(body.reason, "breakpoint")
            self.assertEqual(body.text, "breakpoint 1.1 2.1")
            self.assertEqual(body.description, "breakpoint 1.1 2.1")
            self.assertEqual(body.hitBreakpointIds, [bp1, bp2])
            self.assertIsNotNone(body.threadId)

        self.verify_threads(session, (first, second))
        session.continue_to_exit()
