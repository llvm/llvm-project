"""
Test lldb-dap output events
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import os
import time

import lldbdap_testcase


class TestDAP_progress(lldbdap_testcase.DAPTestCaseBase):
    @skipIfWindows
    def test_output(self):
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        progress_emitter = os.path.join(os.getcwd(), "Progress_emitter.py")
        print(f"Progress emitter path: {progress_emitter}")
        source = "main.cpp"
        # Set breakpoint in the thread function so we can step the threads
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

        self.dap_server.wait_for_event("progressEnd", 15)
        # Expect at least a start, an update, and end event
        # However because the underlying Progress instance is an RAII object and we can't guaruntee
        # it's deterministic destruction in the python API, we verify just start and update
        # otherwise this test could be flakey.
        self.assertTrue(len(self.dap_server.progress_events) > 0)
        start_found = False
        update_found = False
        for event in self.dap_server.progress_events:
            event_type = event["event"]
            if "progressStart" in event_type:
                start_found = True
            if "progressUpdate" in event_type:
                update_found = True

        self.assertTrue(start_found)
        self.assertTrue(update_found)
