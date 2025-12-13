"""
Test lldb-dap setBreakpoints request in assembly source references.
"""

from lldbsuite.test.decorators import *
from dap_server import Source
import lldbdap_testcase


class TestDAP_setBreakpointsAssembly(lldbdap_testcase.DAPTestCaseBase):
    # When using PDB, we need to have debug information to break on assembly_func,
    # but this test relies on us not having debug information for that function.
    @skipIfWindows
    def test_can_break_in_source_references(self):
        """Tests hitting assembly source breakpoints"""
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)

        assmebly_func_breakpoints = self.set_function_breakpoints(["assembly_func"])
        self.continue_to_breakpoints(assmebly_func_breakpoints)

        assembly_func_frame = self.get_stackFrames()[0]
        self.assertIn(
            "sourceReference",
            assembly_func_frame.get("source"),
            "Expected assembly source frame",
        )

        line = assembly_func_frame["line"]

        # Set an assembly breakpoint in the next line and check that it's hit
        source_reference = assembly_func_frame["source"]["sourceReference"]
        assembly_breakpoint_ids = self.set_source_breakpoints_assembly(
            source_reference, [line + 1]
        )
        self.continue_to_breakpoints(assembly_breakpoint_ids)

        # Continue again and verify it hits in the next function call
        self.continue_to_breakpoints(assmebly_func_breakpoints)
        self.continue_to_breakpoints(assembly_breakpoint_ids)

        # Clear the breakpoint and then check that the assembly breakpoint does not hit next time
        self.set_source_breakpoints_assembly(source_reference, [])
        self.continue_to_breakpoints(assmebly_func_breakpoints)
        self.continue_to_exit()

    def test_break_on_invalid_source_reference(self):
        """Tests hitting assembly source breakpoints"""
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)

        # Verify that setting a breakpoint on an invalid source reference fails
        response = self.dap_server.request_setBreakpoints(
            Source.build(source_reference=-1), [1]
        )
        self.assertIsNotNone(response)
        breakpoints = response["body"]["breakpoints"]
        self.assertEqual(len(breakpoints), 1)
        breakpoint = breakpoints[0]
        self.assertFalse(
            breakpoint["verified"], "Expected breakpoint to not be verified"
        )
        self.assertIn("message", breakpoint, "Expected message to be present")
        self.assertEqual(
            breakpoint["message"],
            "Invalid sourceReference.",
        )

        # Verify that setting a breakpoint on a source reference that is not created fails
        response = self.dap_server.request_setBreakpoints(
            Source.build(source_reference=200), [1]
        )
        self.assertIsNotNone(response)
        breakpoints = response["body"]["breakpoints"]
        self.assertEqual(len(breakpoints), 1)
        break_point = breakpoints[0]
        self.assertFalse(
            break_point["verified"], "Expected breakpoint to not be verified"
        )
        self.assertIn("message", break_point, "Expected message to be present")
        self.assertEqual(
            break_point["message"],
            "Invalid sourceReference.",
        )

    @skipIfWindows
    def test_persistent_assembly_breakpoint(self):
        """Tests that assembly breakpoints are working persistently across sessions"""
        self.build()
        program = self.getBuildArtifact("a.out")
        self.create_debug_adapter()

        # Run the first session and set a persistent assembly breakpoint
        try:
            self.dap_server.request_initialize()
            self.dap_server.request_launch(program)

            assmebly_func_breakpoints = self.set_function_breakpoints(["assembly_func"])
            self.continue_to_breakpoints(assmebly_func_breakpoints)

            assembly_func_frame = self.get_stackFrames()[0]
            source_reference = assembly_func_frame["source"]["sourceReference"]

            # Set an assembly breakpoint in the middle of the assembly function
            persistent_breakpoint_line = 4
            persistent_breakpoint_ids = self.set_source_breakpoints_assembly(
                source_reference, [persistent_breakpoint_line]
            )

            self.assertEqual(
                len(persistent_breakpoint_ids),
                1,
                "Expected one assembly breakpoint to be set",
            )

            persistent_breakpoint_source = self.dap_server.resolved_breakpoints[
                persistent_breakpoint_ids[0]
            ]["source"]
            self.assertIn(
                "adapterData",
                persistent_breakpoint_source,
                "Expected assembly breakpoint to have persistent information",
            )
            self.assertIn(
                "persistenceData",
                persistent_breakpoint_source["adapterData"],
                "Expected assembly breakpoint to have persistent information",
            )

            self.continue_to_breakpoints(persistent_breakpoint_ids)
        finally:
            self.dap_server.request_disconnect(terminateDebuggee=True)
            self.dap_server.terminate()

        # Restart the session and verify the breakpoint is still there
        self.create_debug_adapter()
        try:
            self.dap_server.request_initialize()
            self.dap_server.request_launch(program)
            new_session_breakpoints_ids = self.set_source_breakpoints_from_source(
                Source(persistent_breakpoint_source),
                [persistent_breakpoint_line],
            )

            self.assertEqual(
                len(new_session_breakpoints_ids),
                1,
                "Expected one breakpoint to be set in the new session",
            )

            self.continue_to_breakpoints(new_session_breakpoints_ids)
            current_line = self.get_stackFrames()[0]["line"]
            self.assertEqual(
                current_line,
                persistent_breakpoint_line,
                "Expected to hit the persistent assembly breakpoint at the same line",
            )
        finally:
            self.dap_server.request_disconnect(terminateDebuggee=True)
            self.dap_server.terminate()
