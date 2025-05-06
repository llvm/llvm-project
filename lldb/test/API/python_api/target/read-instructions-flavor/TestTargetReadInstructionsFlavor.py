"""
Test SBTarget Read Instruction.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TargetReadInstructionsFlavor(TestBase):
    @skipIfWindows
    @skipIf(archs=no_match(["x86_64", "x86", "i386"]))
    def test_read_instructions_with_flavor(self):
        self.build()
        executable = self.getBuildArtifact("a.out")

        # create a target
        target = self.dbg.CreateTarget(executable)
        self.assertTrue(target.IsValid(), "target is not valid")

        functions = target.FindFunctions("test_add")
        self.assertEqual(len(functions), 1)
        test_add = functions[0]

        test_add_symbols = test_add.GetSymbol()
        self.assertTrue(
            test_add_symbols.IsValid(), "test_add function symbols is not valid"
        )

        expected_instructions = (("mov", "eax, edi"), ("add", "eax, esi"), ("ret", ""))
        test_add_insts = test_add_symbols.GetInstructions(target, "intel")
        # clang adds an extra nop instruction but gcc does not. It makes more sense
        # to check if it is at least 3
        self.assertLessEqual(len(expected_instructions), len(test_add_insts))

        # compares only the expected instructions
        for expected_instr, instr in zip(expected_instructions, test_add_insts):
            self.assertTrue(instr.IsValid(), "instruction is not valid")
            expected_mnemonic, expected_op_str = expected_instr
            self.assertEqual(instr.GetMnemonic(target), expected_mnemonic)
            self.assertEqual(instr.GetOperands(target), expected_op_str)
