"""
Test 'module' events for dynamically loaded libraries.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbdap_testcase


class TestDAP_module_event(lldbdap_testcase.DAPTestCaseBase):
    def lookup_module_id(self, name):
        """Returns the identifier for the first module event starting with the given name."""
        for event in self.dap_server.module_events:
            if self.get_dict_value(event, ["body", "module", "name"]).startswith(name):
                return self.get_dict_value(event, ["body", "module", "id"])
        self.fail(f"No module events matching name={name}")

    def module_events(self, id):
        """Finds all module events by identifier."""
        return [
            event
            for event in self.dap_server.module_events
            if self.get_dict_value(event, ["body", "module", "id"]) == id
        ]

    def module_reasons(self, events):
        """Returns the list of 'reason' values from the given events."""
        return [event["body"]["reason"] for event in events]

    @skipIfWindows
    def test_module_event(self):
        """
        Test that module events are fired on target load and when the list of
        dynamic libraries updates while running.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        # We can analyze the order of events after the process exits.
        self.continue_to_exit()

        a_out_id = self.lookup_module_id("a.out")
        a_out_events = self.module_events(id=a_out_id)

        self.assertIn(
            "new",
            self.module_reasons(a_out_events),
            "Expected a.out to load during the debug session.",
        )

        libother_id = self.lookup_module_id(
            "libother."  # libother.so or libother.dylib based on OS.
        )
        libother_events = self.module_events(id=libother_id)
        self.assertEqual(
            self.module_reasons(libother_events),
            ["new", "removed"],
            "Expected libother to be loaded then unloaded during the debug session.",
        )
