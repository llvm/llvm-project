"""
Test lldb-dap setBreakpoints request in assembly source references.
"""

from lldbsuite.test.decorators import skipIfWindows
from lldbsuite.test.tools.lldb_dap.types import LaunchArgs
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase


class TestDAP_setBreakpointsAssembly(DAPTestCaseBase):
    # When using PDB, we need to have debug information to break on assembly_func,
    # but this test relies on us not having debug information for that function.
    @skipIfWindows
    def test_can_break_in_source_references(self):
        """Tests hitting assembly source breakpoints"""
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        with session.configure(LaunchArgs(program)) as ctx:
            [assembly_func_id] = session.resolve_function_breakpoints(["assembly_func"])
        stop_event = session.verify_stopped_on_breakpoint(
            assembly_func_id, after=ctx.process_event
        )

        top_frame = session.top_frame_from(stop_event).frame
        source_reference = self.expect_not_none(
            top_frame.source and top_frame.source.sourceReference,
            "expected an assembly source reference",
        )

        # Set an assembly breakpoint on the next line and check that it's hit.
        asm_bp_response = session.set_assembly_breakpoints(
            source_reference, [top_frame.line + 1]
        )
        [asm_bp_id] = session.breakpoints_to_ids(asm_bp_response.body.breakpoints)
        session.continue_to_breakpoint(asm_bp_id)

        # Continue again and verify it hits in the next function call.
        session.continue_to_breakpoint(assembly_func_id)
        session.continue_to_breakpoint(asm_bp_id)

        # Clear the assembly breakpoint and verify it does not hit again.
        session.set_assembly_breakpoints(source_reference, [])
        session.continue_to_breakpoint(assembly_func_id)
        session.continue_to_exit()

    def test_break_on_invalid_source_reference(self):
        """Tests setting breakpoints on invalid source references fails cleanly."""
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        session.launch(LaunchArgs(program))

        # Verify that setting a breakpoint on an invalid source reference or
        # a source reference not created fails.
        for bad_ref in (-1, 200):
            response = session.set_assembly_breakpoints(bad_ref, [1])
            [bp] = response.body.breakpoints
            self.assertFalse(bp.verified, "expected breakpoint to not be verified")
            self.assertEqual(bp.message, "Invalid sourceReference.")

    @skipIfWindows
    def test_persistent_assembly_breakpoint(self):
        """Tests that assembly breakpoints persist across sessions."""
        self.build()
        program = self.getBuildArtifact("a.out")

        # Session 1: set the persistent assembly breakpoint.
        session = self.create_session(disconnect_automatically=False)
        with session.configure(LaunchArgs(program)) as ctx:
            function_bp_ids = session.resolve_function_breakpoints(["assembly_func"])
        stop_event = session.verify_stopped_on_breakpoint(
            function_bp_ids, after=ctx.process_event
        )

        top_frame = session.top_frame_from(stop_event).frame
        source = self.expect_not_none(top_frame.source)
        source_reference = self.expect_not_none(source.sourceReference)

        persistent_breakpoint_line = 4
        response = session.set_assembly_breakpoints(
            source_reference, [persistent_breakpoint_line]
        )
        [persistent_bp] = response.body.breakpoints
        persistent_source = self.expect_not_none(
            persistent_bp.source, "expected resolved breakpoint to carry a source"
        )
        adapter_data = self.expect_not_none(
            persistent_source.adapterData,
            "expected assembly breakpoint to carry persistence info",
        )
        self.assertIn(
            "persistenceData",
            adapter_data,
            "expected adapterData to include persistenceData",
        )

        session.continue_to_breakpoint(self.expect_not_none(persistent_bp.id))
        session.disconnect(terminateDebuggee=True)
        session.stop()

        # Session 2: replay the persisted source and verify the breakpoint hits.
        adapter = self.create_stdio_debug_adapter()
        session2 = self.create_session(adapter=adapter)
        with session2.configure(LaunchArgs(program)) as ctx:
            response = session2.set_assembly_breakpoints(
                persistent_source, [persistent_breakpoint_line]
            )
            [new_bp_id] = session2.breakpoints_to_ids(response.body.breakpoints)

        stop_event = session2.verify_stopped_on_breakpoint(
            new_bp_id, after=ctx.process_event
        )
        top_frame = session2.top_frame_from(stop_event).frame
        self.assertEqual(
            top_frame.line,
            persistent_breakpoint_line,
            "expected to hit the persistent assembly breakpoint at the same line",
        )
