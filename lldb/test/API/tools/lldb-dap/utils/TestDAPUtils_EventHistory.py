import json
import threading
from typing import List, Union

from lldbsuite.test.tools.lldb_dap.types import (
    CapabilitiesEvent,
    DAPError,
    Event,
    ExitedEvent,
    InitializedEvent,
    ModuleEvent,
    OutputEvent,
    ProcessEvent,
    ProgressEndEvent,
    TerminatedEvent,
)
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.utils import EventHistory


class TestDAPUtils_EventHistory(DAPTestCaseBase):
    """
    This test suite verifies the behavior of EventHistory, including event ordering validation,
    querying specific events, handling closed histories, applying custom filtering conditions,
    and managing timeouts during asynchronous event additions.
    """

    USE_DEFAULT_DEBUG_ADAPTER = False

    def test_history_order(self):
        """Test that events added to EventHistory are in sequential order"""

        history = EventHistory(timeout=10)
        event_1 = Event.from_json(
            {
                "body": {
                    "category": "console",
                    "output": "Running preInitCommands:\n",
                },
                "event": "output",
                "seq": 2,
                "type": "event",
            }
        )
        event_2 = Event.from_json(
            {
                "body": {
                    "category": "console",
                    "output": "(lldb) log enable lldb process",
                },
                "event": "output",
                "seq": 1,
                "type": "event",
            }
        )

        history.record(event_1)
        with self.assertRaises(DAPError) as ctx:
            history.record(event_2)

        self.assertIn("older than last event", str(ctx.exception))

    def test_wait_for_X_event_on_history_closed(self):
        """Tests `wait_for_earliest_event`, `wait_for_event` and `wait_for_any_event` when the history is closed.

        It should return a DAPError (HistoryClosed) instead of waiting for new events if the
        history is closed and the requested event does not already exist in the history.
        """
        history = EventHistory(timeout=10)
        self.populate_history(history)

        # We only intend on working on existing events.
        # so we don't get a Timeout error.
        history.close()

        init_event = history.wait_for_earliest_event(InitializedEvent)
        self.assertEqual(init_event.seq, 9)
        self.assertIsInstance(init_event, InitializedEvent)
        self.assertEqual(init_event.type, "event")

        cap_event = history.wait_for_event(CapabilitiesEvent, after=init_event)
        self.assertGreater(cap_event.seq, init_event.seq)
        self.assertIsInstance(cap_event, CapabilitiesEvent)

        # There is only one capabilitiesEvent in the history.
        with self.assertRaises(DAPError):
            history.wait_for_event(CapabilitiesEvent, after=cap_event)

        # There is no progressEndEvent in history.
        with self.assertRaises(DAPError):
            history.wait_for_earliest_event(ProgressEndEvent)

        # Must wait for at least one event.
        with self.assertRaises(AssertionError):
            history.wait_for_any_event((), after=cap_event)

        # Test wait_for_any_event.
        # wait_or_any_event should return the first seen event that in the tuple.
        any_event = history.wait_for_any_event(
            (ProcessEvent, OutputEvent, ModuleEvent, TerminatedEvent), after=cap_event
        )
        # We should see the output event first.
        self.assertIsInstance(any_event, OutputEvent)

        any_event = history.wait_for_any_event(
            (ModuleEvent, TerminatedEvent), after=cap_event
        )

        # We should see the ModuleEvent event first.
        self.assertIsInstance(any_event, ModuleEvent)

    def test_wait_for_X_event_until_Y_condition(self):
        """
        Tests the `until` parameter in wait_for_XXXX methods.

        Verifies that `wait_for_event` and `wait_for_any_event` continue scanning
        through matching event types until the provided condition function returns True.
        or throws an error.
        """
        history = EventHistory(timeout=10)
        self.populate_history(history)
        history.close()

        start_event = history.wait_for_earliest_event(OutputEvent)
        self.assertEqual(start_event.seq, 1)
        self.assertIsInstance(start_event, OutputEvent)
        self.assertEqual(start_event.body.output, "Running preInitCommands:\n")

        # Test wait_for_event_until success.
        def has_got_token(event: OutputEvent):
            return "GOT TOKEN" in event.body.output

        output_event = history.wait_for_event(
            OutputEvent, after=start_event, until=has_got_token
        )
        self.assertEqual(output_event.seq, 44)
        self.assertIsInstance(output_event, OutputEvent)
        self.assertIn("GOT TOKEN", output_event.body.output)

        def sdl3_module_loaded(event: ModuleEvent):
            module_is_sdl = event.body.module.name == "sdl3"
            is_new_module = event.body.reason == "new"

            return module_is_sdl and is_new_module

        # Test wait_for_event_until failure.
        # There is no sdl3 module in the event history.
        with self.assertRaises(DAPError):
            history.wait_for_event(
                ModuleEvent, after=start_event, until=sdl3_module_loaded
            )

        # Test wait_for_any_event.
        def sdl3_module_loaded_or_process_exited(event):
            if isinstance(event, ModuleEvent):
                return sdl3_module_loaded(event)

            return True

        seen_event = history.wait_for_any_event(
            (ModuleEvent, ExitedEvent),
            after=start_event,
            until=sdl3_module_loaded_or_process_exited,
        )

        self.assertEqual(seen_event.seq, 47)
        self.assertIsInstance(seen_event, ExitedEvent)
        self.assertEqual(seen_event.body.exitCode, 0)  # type: ignore

    def test_wait_for_X_event_with_timeout(self):
        """Tests the timeout functionality when waiting for events.

        Verifies that a TimeoutError is raised if the expected event does not
        arrive within the specified timeout. Also ensures that the wait methods
        successfully block and retrieve events added asynchronously by another thread
        before the timeout expires.
        """
        history = EventHistory(timeout=10)
        self.populate_history(history)

        delayed_append = threading.Event()
        delayed_append.clear()

        new_cap_events = [
            Event.from_json(
                {
                    "body": {"capabilities": {"supportsModulesRequest": True}},
                    "event": "capabilities",
                    "seq": 50,
                    "type": "event",
                }
            ),
            Event.from_json(
                {
                    "body": {"capabilities": {"supportsRestartRequest": True}},
                    "event": "capabilities",
                    "seq": 52,
                    "type": "event",
                }
            ),
        ]

        def add_delayed_events(events: List[Event]):
            delayed_append.wait()
            for event in events:
                history.record(event)

        event_thread = threading.Thread(
            target=add_delayed_events, args=[new_cap_events]
        )
        event_thread.start()

        try:
            exited_event = history.wait_for_earliest_event(ExitedEvent)
            self.assertEqual(exited_event.seq, 47)
            self.assertEqual(exited_event.body.exitCode, 0)

            def supports_restart_request(event: CapabilitiesEvent):
                return event.body.capabilities.supportsRestartRequest == True

            # No new CapabilitiesEvents have been added after exited event, so this should time out.
            with self.assertRaises(TimeoutError):
                history.wait_for_event(
                    CapabilitiesEvent,
                    after=exited_event,
                    until=supports_restart_request,
                    timeout=0.05,
                )

            # Unblock and sync with the event thread.
            delayed_append.set()
            seen_event = history.wait_for_event(
                CapabilitiesEvent,
                after=exited_event,
                until=supports_restart_request,
                timeout=10,
            )

            # The supportsModuleRequest CapabilitiesEvent should be ignored.
            self.assertEqual(seen_event.seq, 52)
            self.assertIsInstance(seen_event, CapabilitiesEvent)
            self.assertTrue(seen_event.body.capabilities.supportsRestartRequest)
        finally:
            # Clean up.
            delayed_append.set()
            event_thread.join()

    def populate_history(self, history: EventHistory):
        message_log = self.getSourcePath("sample_dap_log.json")
        with open(message_log, "r", encoding="utf-8") as file:
            raw_events = json.loads(file.read())

        for raw_event in raw_events:
            event = Event.from_json(raw_event)
            history.record(event)
