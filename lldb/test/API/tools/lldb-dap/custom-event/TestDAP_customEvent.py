"""
Test lldb-dap custom-event integration.
"""

import json

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbdap_testcase


class TestDAP_customEvent(lldbdap_testcase.DAPTestCaseBase):
    def test_custom_event(self):
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
                "lldb-dap custom-event my-custom-event-no-body",
                "lldb-dap custom-event my-custom-event '{}'".format(
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
        self.assertEquals(custom_event["event"], "my-custom-event-no-body")
        self.assertIsNone(custom_event.get("body", None))

        custom_event = self.dap_server.wait_for_event(filter=["my-custom-event"])
        self.assertEquals(custom_event["event"], "my-custom-event")
        self.assertEquals(custom_event["body"], custom_event_body)
