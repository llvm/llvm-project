"""
Test 'module' events for dynamically loaded libraries.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbdap_testcase


class TestDAP_module_event(lldbdap_testcase.DAPTestCaseBase):
    @skipIfWindows
    def test_module_event(self):
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        self.continue_to_exit()

        # Module 'remove' events will only contain the 'id' not the 'name',
        # first lookup the module id to find all the events.
        a_out_id = next(
            e
            for e in self.dap_server.module_events
            if e["body"]["module"]["name"] == "a.out"
        )["body"]["module"]["id"]
        a_out_events = [
            e
            for e in self.dap_server.module_events
            if e["body"]["module"]["id"] == a_out_id
        ]

        self.assertIn(
            "new",
            [e["body"]["reason"] for e in a_out_events],
            "Expected a.out to load during the debug session.",
        )

        libother_id = next(
            e
            for e in self.dap_server.module_events
            if e["body"]["module"]["name"].startswith("libother.")
        )["body"]["module"]["id"]
        libother_events = [
            e
            for e in self.dap_server.module_events
            if e["body"]["module"]["id"] == libother_id
        ]

        self.assertTrue(libother_events, "Expected libother to produce module events.")
        self.assertEqual(
            [e["body"]["reason"] for e in libother_events],
            ["new", "removed"],
            "Expected libother to be loaded then unloaded during the debug session.",
        )
