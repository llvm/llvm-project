"""
Test lldb-dap disassemble request
"""


import dap_server
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbdap_testcase
import os


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

        _, pc_assembly = self.disassemble(frameIndex=0)
        self.assertIn("location", pc_assembly, "Source location missing.")
        self.assertIn("instruction", pc_assembly, "Assembly instruction missing.")

        # The calling frame (qsort) is coming from a system library, as a result
        # we should not have a source location.
        _, qsort_assembly = self.disassemble(frameIndex=1)
        self.assertNotIn("location", qsort_assembly, "Source location not expected.")
        self.assertIn("instruction", pc_assembly, "Assembly instruction missing.")

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
