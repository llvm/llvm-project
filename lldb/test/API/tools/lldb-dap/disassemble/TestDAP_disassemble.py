"""
Test lldb-dap disassemble request
"""

from lldbsuite.test.decorators import skipIfWindows
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap.types import LaunchArgs
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase


class TestDAP_disassemble(DAPTestCaseBase):
    @skipIfWindows
    def test_disassemble(self):
        """Disassembly at the current PC returns the expected source line, and
        clearing breakpoints doesn't change the instructions."""

        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        source = self.getSourcePath("main.c")
        bp_line = line_number(source, "// breakpoint 1")

        with session.configure(LaunchArgs(program)) as ctx:
            session.resolve_source_breakpoints(source, [bp_line])
        stop_event = session.verify_stopped_on_breakpoint(after=ctx.process_event)
        top_frame = session.thread_context_from(stop_event).top_frame()

        insts_with_bp = top_frame.disassemble()
        pc_with_bp = insts_with_bp[0]
        self.assertIsNotNone(pc_with_bp.location, "Source location missing.")
        self.assertEqual(pc_with_bp.line, bp_line, "Expects the same line number")
        self.assertTrue(pc_with_bp.instruction, "Assembly instruction missing.")

        cleared = session.set_source_breakpoints(source, [])
        self.assertEqual(len(cleared.body.breakpoints), 0, "Expects no breakpoints.")

        insts_no_bp = top_frame.disassemble()
        pc_no_bp = insts_no_bp[0]
        self.assertIsNotNone(pc_no_bp.location, "Source location missing.")
        self.assertEqual(pc_no_bp.line, bp_line, "Expects the same line number")
        self.assertTrue(pc_no_bp.instruction, "Assembly instruction missing.")

        # The disassembly instructions should be the same with breakpoint and
        # no breakpoint.
        self.assertEqual(
            insts_with_bp,
            insts_no_bp,
            "Expects instructions are the same after removing breakpoints.",
        )

        session.continue_to_exit()

    @skipIfWindows
    def test_disassemble_backwards(self):
        """
        Tests the 'disassemble' request with a backwards disassembly range.
        """
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        source = self.getSourcePath("main.c")
        bp_line = line_number(source, "// breakpoint 1")
        with session.configure(LaunchArgs(program)) as ctx:
            session.resolve_source_breakpoints(source, [bp_line])
        stop_event = session.verify_stopped_on_breakpoint(after=ctx.process_event)

        caller_frame = session.thread_context_from(stop_event).frames(levels=2)[1]
        instruction_pointer_ref = self.expect_not_none(
            caller_frame.frame.instructionPointerReference
        )

        backwards_instructions = 200
        instructions_count = 400
        instructions = session.disassemble(
            memoryReference=instruction_pointer_ref,
            instructionOffset=-backwards_instructions,
            instructionCount=instructions_count,
        )

        self.assertEqual(
            len(instructions),
            instructions_count,
            "Disassemble request should return the exact requested number of instructions.",
        )

        # The requested instruction pointer should land exactly at
        # `backwards_instructions` entries into the returned instructions.
        self.assertEqual(
            instructions[backwards_instructions].address,
            instruction_pointer_ref,
            f"expected instruction pointer at index {backwards_instructions}",
        )

        session.set_source_breakpoints(source, [])
        session.continue_to_exit()

    def test_disassemble_empty_memory_reference(self):
        """An empty `memoryReference` returns the requested count of invalid
        placeholder instructions instead of erroring out."""
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        source = self.getSourcePath("main.c")
        bp_line = line_number(source, "// breakpoint 1")
        with session.configure(LaunchArgs(program)) as ctx:
            session.resolve_source_breakpoints(source, [bp_line])
        session.verify_stopped_on_breakpoint(after=ctx.process_event)

        instructions = session.disassemble(
            memoryReference="", instructionOffset=0, instructionCount=50
        )
        self.assertEqual(len(instructions), 50)
        for instruction in instructions:
            self.assertEqual(instruction.presentationHint, "invalid")

        # Clear breakpoints and exit.
        session.set_source_breakpoints(source, [])
        session.continue_to_exit()
