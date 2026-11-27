"""
Test lldb-dap 'stopped' events.
"""

from typing import Sequence, List

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase, DAPTestSession
from lldbsuite.test.tools.lldb_dap.types import LaunchArgs, StoppedEvent, ThreadsArgs


# The in-process Windows plugin (ProcessWindows) reports one thread's exception
# per stop.
@skipIfWindows
@skipIfLinux
@requireThreadSupport
class TestDAP_stopped_events(DAPTestCaseBase):
    """
    Test validates different operations that produce 'stopped' events.
    """

    def collect_stopped_events(self, session: DAPTestSession, *, after):
        """Collect the stopped events until an event reports `allThreadsStopped`."""
        events: List[StoppedEvent] = []

        def all_threads_stopped(event: StoppedEvent):
            events.append(event)
            return bool(event.body.allThreadsStopped)

        session.wait_for_stopped_event(
            after=after,
            until=all_threads_stopped,
            timeout_msg="waiting until all threads stopped",
        )
        return events

    def verify_both_threads_hit_breakpoint(
        self, session: DAPTestSession, *, after
    ) -> list[StoppedEvent]:
        """Collect the stopped events of both threads hitting the breakpoint.

        The threads usually hit it in the same stop, but a thread that reaches
        the breakpoint after the process stopped only hits it in the next stop,
        so continue until both did."""
        events = self.collect_stopped_events(session, after=after)
        self.verify_focus(session, events)
        self.trace(f"threads that hit the breakpoint in the first stop: {len(events)}")

        if len(events) < 2:
            session.do_continue()
            # The next stop's events can arrive before the continue response,
            # so wait for those after the first stop's.
            next_events = self.collect_stopped_events(session, after=events[-1])
            self.verify_focus(session, next_events)
            events += next_events

        thread_count = len({event.body.threadId for event in events})
        self.assertEqual(
            (len(events), thread_count),
            (2, 2),
            f"expected one stopped event on each of two threads, got {events!r}",
        )
        return events

    def verify_focus(self, session: DAPTestSession, events: Sequence[StoppedEvent]):
        "The focus thread is the last stopped event and the first thread in threads request."
        threads = session.send_request(ThreadsArgs()).result().body.threads
        thread_ids = [t.id for t in threads]
        stopped_tids = [e.body.threadId for e in events]

        for tid in stopped_tids:
            self.assertIn(tid, thread_ids, f"unknown thread in {threads!r}")
        first_stopped_tid = next(tid for tid in thread_ids if tid in stopped_tids)

        *_, focused_event = events
        self.assertEqual(focused_event.body.threadId, first_stopped_tid)
        self.assertFalse(
            focused_event.body.preserveFocusHint,
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
        events = self.verify_both_threads_hit_breakpoint(session, after=process_event)

        for event in events:
            body = event.body
            self.assertEqual(body.reason, "breakpoint")
            self.assertEqual(body.text, "breakpoint 1.1")
            self.assertEqual(body.description, "breakpoint 1.1")
            self.assertEqual(body.hitBreakpointIds, [bp])
            self.assertIsNotNone(body.threadId)

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

        process_event = ctx.process_event
        events = self.verify_both_threads_hit_breakpoint(session, after=process_event)

        for event in events:
            body = event.body
            self.assertEqual(body.reason, "breakpoint")
            self.assertEqual(body.text, "breakpoint 1.1 2.1")
            self.assertEqual(body.description, "breakpoint 1.1 2.1")
            self.assertEqual(body.hitBreakpointIds, [bp1, bp2])
            self.assertIsNotNone(body.threadId)

        session.continue_to_exit()
