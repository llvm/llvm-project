"""
Test lldb-dap disassemble request
"""

from lldbsuite.test.decorators import skipIfWindows
from lldbsuite.test.lldbtest import line_number
import lldbdap_testcase


class TestDAP_disassemble(lldbdap_testcase.DAPTestCaseBase):
    @skipIfWindows
    def test_disassemble(self):
        """
        Tests the 'disassemble' request.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = "main.c"
        self.set_source_breakpoints(source, [line_number(source, "// breakpoint 1")])
        self.continue_to_next_stop()

        insts_with_bp, pc_with_bp_assembly = self.disassemble(frameIndex=0)
        no_bp = self.set_source_breakpoints(source, [])
        self.assertEqual(len(no_bp), 0, "expect no breakpoints.")
        self.assertIn(
            "instruction", pc_with_bp_assembly, "Assembly instruction missing."
        )

        # the disassembly instructions should be the same even if there is a breakpoint;
        insts_no_bp, pc_no_bp_assembly = self.disassemble(frameIndex=0)
        self.assertDictEqual(
            insts_with_bp,
            insts_no_bp,
            "Expects instructions are the same after removing breakpoints.",
        )
        self.assertIn("instruction", pc_no_bp_assembly, "Assembly instruction missing.")

        self.continue_to_exit()

    @skipIfWindows
    def test_disassemble_backwards(self):
        """
        Tests the 'disassemble' request with a backwards disassembly range.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = "main.c"
        self.set_source_breakpoints(source, [line_number(source, "// breakpoint 1")])
        self.continue_to_next_stop()

        instruction_pointer_reference = self.get_stackFrames()[1][
            "instructionPointerReference"
        ]
        backwards_instructions = 200
        instructions_count = 400
        instructions = self.dap_server.request_disassemble(
            memoryReference=instruction_pointer_reference,
            instructionOffset=-backwards_instructions,
            instructionCount=instructions_count,
        )

        self.assertEqual(
            len(instructions),
            instructions_count,
            "Disassemble request should return the exact requested number of instructions.",
        )

        frame_instruction_index = next(
            (
                i
                for i, instruction in enumerate(instructions)
                if instruction["address"] == instruction_pointer_reference
            ),
            -1,
        )
        self.assertEqual(
            frame_instruction_index,
            backwards_instructions,
            f"requested instruction should be preceeded by {backwards_instructions} instructions. Actual index: {frame_instruction_index}",
        )

        # clear breakpoints
        self.set_source_breakpoints(source, [])
        self.continue_to_exit()
