"""
Test lldb-dap recieves invalidated-events when the area such as
stack, variables, threads has changes but the client does not
know about it.
"""

from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap.dap_types import LaunchArgs, StackTraceArgs
from lldbsuite.test.tools.lldb_dap.lldb_dap_testcase import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.session_helpers import DAPTestSession


class TestDAP_invalidatedEvent(DAPTestCaseBase):
    def verify_top_frame_name(
        self, session: DAPTestSession, frame_name: str, thread_id: int
    ):
        response = session.stack_trace(thread_id)
        all_frames = response.body.stackFrames

        self.assertGreaterEqual(len(all_frames), 1, "Expected at least one frame.")
        top_frame_name = all_frames[0].name
        self.assertRegex(top_frame_name, f"{frame_name}.*")
        return response

    def test_invalidated_stack_area_event(self):
        """
        Test an invalidated event for the stack area.
        The event is sent when the command `thread return <expr>` is sent by the user.
        """
        other_source = "other.h"
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        with session.configure(LaunchArgs(program)) as ctx:
            return_bp_line = line_number(other_source, "// thread return breakpoint")
            session.resolve_source_breakpoints(other_source, [return_bp_line])

        stopped_event = session.verify_stopped_on_breakpoint(after=ctx.process_event)

        thread_id = self.expect_not_none(
            stopped_event.body.threadId, "expected a thread id."
        )
        stack_response = self.verify_top_frame_name(session, "add", thread_id)

        # Run thread return.
        thread_command = "thread return 20"
        session.evaluate(thread_command, context="repl")

        # Wait for the invalidated stack event.
        invalid_event = session.wait_for_invalidated(after=stack_response)
        self.assertIsNotNone(invalid_event, "Expected an invalidated event.")
        event_body = invalid_event.body
        self.assertIsNotNone(event_body.areas)
        self.assertIn("stacks", event_body.areas or [])
        self.assertIsNotNone(event_body.threadId)
        self.assertEqual(
            thread_id,
            event_body.threadId,
            f"Expected the event from thread {thread_id}.",
        )

        # Confirm we are back at the main frame.
        thread_id = self.expect_not_none(
            invalid_event.body.threadId, "expected a thread id."
        )
        self.verify_top_frame_name(session, "main", thread_id)
        session.continue_to_exit()
