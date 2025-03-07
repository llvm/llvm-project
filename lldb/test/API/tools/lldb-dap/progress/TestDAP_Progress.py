"""
Test lldb-dap output events
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import json
import os
import time

import lldbdap_testcase


class TestDAP_progress(lldbdap_testcase.DAPTestCaseBase):
    def verify_progress_events(
        self,
        expected_title,
        expected_message=None,
        expected_not_in_message=None,
        only_verify_first_update=False,
    ):
        self.dap_server.wait_for_event("progressEnd", 15)
        self.assertTrue(len(self.dap_server.progress_events) > 0)
        start_found = False
        update_found = False
        end_found = False
        for event in self.dap_server.progress_events:
            event_type = event["event"]
            if "progressStart" in event_type:
                title = event["body"]["title"]
                self.assertIn(expected_title, title)
                start_found = True
            if "progressUpdate" in event_type:
                message = event["body"]["message"]
                if only_verify_first_update and update_found:
                    continue
                if expected_message is not None:
                    self.assertIn(expected_message, message)
                if expected_not_in_message is not None:
                    self.assertNotIn(expected_not_in_message, message)
                update_found = True
            if "progressEnd" in event_type:
                end_found = True

        self.assertTrue(start_found)
        self.assertTrue(update_found)
        self.assertTrue(end_found)

    @skipIfWindows
    def test_output(self):
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        progress_emitter = os.path.join(os.getcwd(), "Progress_emitter.py")
        source = "main.cpp"
        breakpoint_ids = self.set_source_breakpoints(
            source, [line_number(source, "// break here")]
        )
        self.continue_to_breakpoints(breakpoint_ids)
        self.dap_server.request_evaluate(
            f"`command script import {progress_emitter}", context="repl"
        )
        self.dap_server.request_evaluate(
            "`test-progress --total 3 --seconds 1", context="repl"
        )

        self.verify_progress_events(
            expected_title="Progress tester",
            expected_not_in_message="Progress tester",
        )

    @skipIfWindows
    def test_output_nodetails(self):
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        progress_emitter = os.path.join(os.getcwd(), "Progress_emitter.py")
        source = "main.cpp"
        breakpoint_ids = self.set_source_breakpoints(
            source, [line_number(source, "// break here")]
        )
        self.continue_to_breakpoints(breakpoint_ids)
        self.dap_server.request_evaluate(
            f"`command script import {progress_emitter}", context="repl"
        )
        self.dap_server.request_evaluate(
            "`test-progress --total 3 --seconds 1 --no-details", context="repl"
        )

        self.verify_progress_events(
            expected_title="Progress tester",
            expected_message="Initial Detail",
        )

    @skipIfWindows
    def test_output_indeterminate(self):
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        progress_emitter = os.path.join(os.getcwd(), "Progress_emitter.py")
        source = "main.cpp"
        breakpoint_ids = self.set_source_breakpoints(
            source, [line_number(source, "// break here")]
        )
        self.continue_to_breakpoints(breakpoint_ids)
        self.dap_server.request_evaluate(
            f"`command script import {progress_emitter}", context="repl"
        )
        self.dap_server.request_evaluate("`test-progress --seconds 1", context="repl")

        self.verify_progress_events(
            expected_title="Progress tester: Initial Indeterminate Detail",
            expected_message="Step 1",
            only_verify_first_update=True,
        )

    @skipIfWindows
    def test_output_nodetails_indeterminate(self):
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        progress_emitter = os.path.join(os.getcwd(), "Progress_emitter.py")
        source = "main.cpp"
        breakpoint_ids = self.set_source_breakpoints(
            source, [line_number(source, "// break here")]
        )
        self.dap_server.request_evaluate(
            f"`command script import {progress_emitter}", context="repl"
        )

        self.dap_server.request_evaluate(
            "`test-progress --seconds 1 --no-details", context="repl"
        )

        self.verify_progress_events(
            expected_title="Progress tester: Initial Indeterminate Detail",
            expected_message="Initial Indeterminate Detail",
            only_verify_first_update=True,
        )
