import dap_server
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbdap_testcase
import re


class TestDAP_module_event(lldbdap_testcase.DAPTestCaseBase):
    @skipIfWindows
    def test_module_event(self):
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program, stopOnEntry=True)

        source = "main.cpp"
        breakpoint1_line = line_number(source, "// breakpoint 1")
        breakpoint2_line = line_number(source, "// breakpoint 2")
        breakpoint3_line = line_number(source, "// breakpoint 3")

        breakpoint_ids = self.set_source_breakpoints(
            source, [breakpoint1_line, breakpoint2_line, breakpoint3_line]
        )
        self.continue_to_breakpoints(breakpoint_ids)

        # We're now stopped at breakpoint 1 before the dlopen. Flush all the module events.
        event = self.dap_server.wait_for_event("module", 0.25)
        while event is not None:
            event = self.dap_server.wait_for_event("module", 0.25)

        # Continue to the second breakpoint, before the dlclose.
        self.continue_to_breakpoints(breakpoint_ids)

        # Make sure we got a module event for libother.
        event = self.dap_server.wait_for_event("module", 5)
        self.assertTrue(event, "didn't get a module event")
        module_name = event["body"]["module"]["name"]
        module_id = event["body"]["module"]["id"]
        self.assertEqual(event["body"]["reason"], "new")
        self.assertIn("libother", module_name)

        # Continue to the third breakpoint, after the dlclose.
        self.continue_to_breakpoints(breakpoint_ids)

        # Make sure we got a module event for libother.
        event = self.dap_server.wait_for_event("module", 5)
        self.assertTrue(event, "didn't get a module event")
        reason = event["body"]["reason"]
        self.assertEqual(event["body"]["reason"], "removed")
        self.assertEqual(event["body"]["module"]["id"], module_id)

        # The removed module event should omit everything but the module id.
        # Check that there's no module name in the event.
        self.assertNotIn("name", event["body"]["module"])

        self.continue_to_exit()
