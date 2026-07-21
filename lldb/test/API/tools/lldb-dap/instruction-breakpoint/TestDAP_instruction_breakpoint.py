"""
Test lldb-dap instruction breakpoints.
"""

import os

from lldbsuite.test.decorators import skipIfWindows
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.types import LaunchArgs


class TestDAP_InstructionBreakpointTestCase(DAPTestCaseBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipIfWindows
    def test_instruction_breakpoint(self):
        """Set a source breakpoint, then use the disassembly to set an
        instruction breakpoint on the next instruction and verify we hit it."""
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()

        main_basename = "main-copy.cpp"
        main_path = os.path.realpath(self.getBuildArtifact(main_basename))
        main_line = line_number("main.cpp", "breakpoint 1")

        # Set a source breakpoint and check it was resolved against the
        # renamed source file.
        with session.configure(LaunchArgs(program)) as ctx:
            response = session.set_source_breakpoints(main_path, [main_line])
            [source_bp] = response.body.breakpoints
            self.assertTrue(source_bp.verified, "breakpoint is not verified")
            self.assertEqual(source_bp.line, main_line, "incorrect breakpoint line")

            bp_source = self.expect_not_none(source_bp.source)
            self.assertEqual(bp_source.name, main_basename, "incorrect source name")
            self.assertEqual(bp_source.path, main_path, "incorrect source path")

            source_bp_id = self.expect_not_none(source_bp.id)

        # Run to the source breakpoint, the stack frame should also report
        # the renamed source.
        stop_event = session.verify_stopped_on_breakpoint(
            source_bp_id, after=ctx.process_event
        )
        top_frame_ctx = session.top_frame_from(stop_event)
        top_frame = top_frame_ctx.frame

        frame_source = self.expect_not_none(top_frame.source)
        self.assertEqual(frame_source.name, main_basename, "incorrect source name")
        self.assertEqual(frame_source.path, main_path, "incorrect source path")

        # Disassemble at the current PC and use the address of the following
        # instruction as an instruction breakpoint target.
        disasm = top_frame_ctx.disassemble()
        current_inst, next_inst = disasm[0], disasm[1]

        self.assertEqual(
            current_inst.address,
            top_frame.instructionPointerReference,
            "disassembly does not begin at the current instruction",
        )
        self.assertGreater(len(next_inst.address), 2)
        self.assertNotEqual(next_inst.presentationHint, "invalid")

        bp_response = session.set_instruction_breakpoints([next_inst.address])
        [inst_bp] = bp_response.body.breakpoints
        self.assertEqual(
            inst_bp.instructionReference,
            next_inst.address,
            "instruction breakpoint was not resolved to the expected address",
        )

        inst_bp_id = self.expect_not_none(inst_bp.id)
        session.continue_to_breakpoint(inst_bp_id)

        session.set_source_breakpoints(main_path, [])
        session.set_instruction_breakpoints([])
        session.continue_to_exit(exitCode=3)
