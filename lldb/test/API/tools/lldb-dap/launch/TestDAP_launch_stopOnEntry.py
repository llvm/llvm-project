"""
Test lldb-dap launch request.
"""

from lldbsuite.test.tools.lldb_dap.dap_types import (
    ExitedEvent,
    LaunchArgs,
    StoppedEvent,
)
from lldbsuite.test.tools.lldb_dap.lldb_dap_testcase import DAPTestCaseBase


class TestDAP_launch_stopOnEntry(DAPTestCaseBase):
    """
    Tests the default launch of a simple program that stops at the
    entry point instead of continuing.
    """

    def test(self):
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        process_event = session.launch(LaunchArgs(program, stopOnEntry=True))
        stop_event = session.verify_stopped_on_entry(after=process_event)
        exit_event = session.continue_to_exit()

        seen_stop_events = []

        def matches_exit_event(evt):
            if isinstance(evt, StoppedEvent):
                seen_stop_events.append(evt)
            return evt.seq == exit_event.seq

        # Collect stopped events until the ExitEvent.
        session.wait_for_any_event(
            (StoppedEvent, ExitedEvent), after=stop_event, until=matches_exit_event
        )
        # Verify we did not receive any other stop event.
        self.assertEqual(
            len(seen_stop_events),
            0,
            f"expected no new stopped events. seen events: {seen_stop_events}",
        )
