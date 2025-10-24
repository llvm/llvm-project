"""
Test lldb-dap recieves invalidated-events when the area such as
stack, variables, threads has changes but the client does not
know about it.
"""

import lldbdap_testcase
from lldbsuite.test.lldbtest import line_number
from dap_server import Event


class TestDAP_invalidatedEvent(lldbdap_testcase.DAPTestCaseBase):
    def verify_top_frame_name(self, frame_name: str):
        all_frames = self.get_stackFrames()
        self.assertGreaterEqual(len(all_frames), 1, "Expected at least one frame.")
        top_frame_name = all_frames[0]["name"]
        self.assertRegex(top_frame_name, f"{frame_name}.*")

    def test_invalidated_stack_area_event(self):
        """
        Test an invalidated event for the stack area.
        The event is sent when the command `thread return <expr>` is sent by the user.
        """
        other_source = "other.h"
        return_bp_line = line_number(other_source, "// thread return breakpoint")

        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        self.set_source_breakpoints(other_source, [return_bp_line])
        self.continue_to_next_stop()

        self.verify_top_frame_name("add")
        thread_id = self.dap_server.get_thread_id()
        self.assertIsNotNone(thread_id, "Exepected a thread id.")

        # run thread return
        thread_command = "thread return 20"
        eval_resp = self.dap_server.request_evaluate(thread_command, context="repl")
        self.assertTrue(eval_resp["success"], f"Failed to evaluate `{thread_command}`.")

        # wait for the invalidated stack event.
        stack_event = self.dap_server.wait_for_event(["invalidated"])
        self.assertIsNotNone(stack_event, "Expected an invalidated event.")
        event_body: Event = stack_event["body"]
        self.assertIn("stacks", event_body["areas"])
        self.assertIn("threadId", event_body.keys())
        self.assertEqual(
            thread_id,
            event_body["threadId"],
            f"Expected the event from thread {thread_id}.",
        )

        # confirm we are back at the main frame.
        self.verify_top_frame_name("main")
        self.continue_to_exit()
