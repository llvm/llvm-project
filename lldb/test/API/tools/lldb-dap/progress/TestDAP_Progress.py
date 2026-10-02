"""
Test lldb-dap progress events (smoke test).

This test only verifies that `ProgressReport` events sent actually reaches the client
from lldb.

The throttling check is covered by `lldb/unittests/DAP/ProgressEventTest.cpp`.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase, DAPTestSession
from lldbsuite.test.tools.lldb_dap.types import *

_ProgressEvent = Union[ProgressStartEvent, ProgressUpdateEvent, ProgressEndEvent]


class TestDAP_Progress(DAPTestCaseBase):
    def collect_progress_events(self, session: DAPTestSession, *, after):
        """Collect ProgressXXXX events between `after` and the next ProgressEndEvent."""
        events: List[_ProgressEvent] = []

        def matches_progress_end(evt) -> bool:
            events.append(evt)
            return isinstance(evt, ProgressEndEvent)

        session.wait_for_any_event(
            (ProgressStartEvent, ProgressUpdateEvent, ProgressEndEvent),
            after=after,
            until=matches_progress_end,
            timeout_msg="Collecting ProgressXXXXEvents until ProgressEndEvent",
        )
        return events

    def verify_progress_events(
        self,
        events: List[_ProgressEvent],
        *,
        expected_title: str,
        expected_message: Optional[str] = None,
        expected_message_regex: Optional[str] = None,
        expected_not_in_message: Optional[str] = None,
    ):
        # A progress group is shaped: [ProgressStart, ProgressUpdate*, ProgressEnd].
        self.assertGreaterEqual(
            len(events), 3, "expected at least start + one update + end"
        )
        [start, *updates, end] = events

        self.assertIsInstance(start, ProgressStartEvent)
        self.assertIn(expected_title, start.body.title)
        self.assertIsInstance(end, ProgressEndEvent)

        for update in updates:
            self.assertIsInstance(update, ProgressUpdateEvent)
            message = update.body.message or ""

            if expected_message is not None:
                self.assertIn(expected_message, message)
            if expected_message_regex is not None:
                self.assertTrue(re.match(expected_message_regex, message))
            if expected_not_in_message is not None:
                self.assertNotIn(expected_not_in_message, message)

    @skipIfWindows
    def test_progress(self):
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        process_event = session.launch(LaunchArgs(program, stopOnEntry=True))
        stopped = session.verify_stopped_on_entry(after=process_event)

        progress_emitter = self.getSourcePath("Progress_emitter.py")
        session.evaluate(f"`command script import {progress_emitter}", context="repl")

        # Test details.
        # 1 progress every 200ms, 10 times = 2s.
        session.evaluate("`send-progress --total 10 --seconds 0.2", context="repl")
        events = self.collect_progress_events(session, after=stopped)
        self.verify_progress_events(
            events,
            expected_title="Progress tester",
            expected_not_in_message="Progress tester",
        )
