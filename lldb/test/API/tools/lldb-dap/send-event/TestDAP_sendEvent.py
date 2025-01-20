"""
Test lldb-dap send-event integration.
"""

import json

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbdap_testcase


class TestDAP_sendEvent(lldbdap_testcase.DAPTestCaseBase):
    def test_send_event(self):
        """
        Test sending a custom event.
        """
        program = self.getBuildArtifact("a.out")
        source = "main.c"
        custom_event_body = {
            "key": 321,
            "arr": [True],
        }
        self.build_and_launch(
            program,
            stopCommands=[
                "lldb-dap send-event my-custom-event-no-body",
                "lldb-dap send-event my-custom-event '{}'".format(
                    json.dumps(custom_event_body)
                ),
            ],
        )

        breakpoint_line = line_number(source, "// breakpoint")

        self.set_source_breakpoints(source, [breakpoint_line])
        self.continue_to_next_stop()

        custom_event = self.dap_server.wait_for_event(
            filter=["my-custom-event-no-body"]
        )
        self.assertEqual(custom_event["event"], "my-custom-event-no-body")
        self.assertIsNone(custom_event.get("body", None))

        custom_event = self.dap_server.wait_for_event(filter=["my-custom-event"])
        self.assertEqual(custom_event["event"], "my-custom-event")
        self.assertEqual(custom_event["body"], custom_event_body)

    def test_send_internal_event(self):
        """
        Test sending an internal event produces an error.
        """
        program = self.getBuildArtifact("a.out")
        source = "main.c"
        self.build_and_launch(program)

        breakpoint_line = line_number(source, "// breakpoint")

        self.set_source_breakpoints(source, [breakpoint_line])
        self.continue_to_next_stop()

        resp = self.dap_server.request_evaluate(
            "`lldb-dap send-event stopped", context="repl"
        )
        self.assertRegex(
            resp["body"]["result"],
            r"Invalid use of lldb-dap send-event, event \"stopped\" should be handled by lldb-dap internally.",
        )
