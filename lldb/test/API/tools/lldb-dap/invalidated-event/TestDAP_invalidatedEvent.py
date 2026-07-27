"""
Test lldb-dap recieves invalidated-events when the area such as
stack, variables, threads has changes but the client does not
know about it.
"""

from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.types import LaunchArgs


class TestDAP_invalidatedEvent(DAPTestCaseBase):
    def test_invalidated_stack_area_event(self):
        """
        Test an invalidated event for the stack area.
        The event is sent when the command `thread return <expr>` is sent by the user.
        """
        other_source = "other.h"
        return_bp_line = line_number(other_source, "// thread return breakpoint")
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        with session.configure(LaunchArgs(program)) as ctx:
            session.resolve_source_breakpoints(other_source, [return_bp_line])

        stopped_event = session.verify_stopped_on_breakpoint(after=ctx.process_event)

        thread_ctx = session.thread_context_from(stopped_event)
        top_frame = thread_ctx.top_frame()
        self.assertRegex(top_frame.name, "add.*")

        last_event = session.last_event()
        # Run thread return.
        session.evaluate("thread return 20", context="repl")

        # Wait for the invalidated stack event.
        invalid_event = session.wait_for_invalidated_event(after=last_event)
        self.assertIsNotNone(invalid_event, "Expected an invalidated event.")
        event_body = invalid_event.body
        self.assertIsNotNone(event_body.areas)
        self.assertIn("stacks", event_body.areas or [])
        self.assertIsNotNone(event_body.threadId)
        self.assertEqual(
            thread_ctx.thread_id,
            event_body.threadId,
            f"Expected the event from thread {thread_ctx.thread_id}.",
        )

        # Confirm we are back at the main frame.
        top_frame = session.top_frame_from(invalid_event)
        self.assertRegex(top_frame.name, "main.*")

        session.continue_to_exit()
