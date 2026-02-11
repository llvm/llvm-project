"""
Tests that LLDB can correctly set up a disassembler using extensions from the .riscv.attributes section.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestDisassembler(TestBase):
    expected_zbb_instrs = ["andn", "orn", "xnor", "rol", "ror"]

    @skipIfLLVMTargetMissing("RISCV")
    def test_without_riscv_attributes(self):
        """
        Tests disassembly of a riscv binary without the .riscv.attributes.
        Without the .riscv.attributes section lldb won't set up a disassembler to
        handle the bitmanip extension, so it is not expected to see zbb instructions
        in the output.
        """
        self.build(dictionary={"CFLAGS_EXTRAS": "-march=rv64gc_zbb"})

        target = self.dbg.CreateTarget(self.getBuildArtifact("stripped.out"))

        self.expect("disassemble --name do_zbb_stuff")
        output = self.res.GetOutput()

        for instr in self.expected_zbb_instrs:
            self.assertFalse(
                instr in output, "Zbb instructions should not be disassembled"
            )

    @skipIfLLVMTargetMissing("RISCV")
    def test_with_riscv_attributes(self):
        """
        Tests disassembly of a riscv binary with the .riscv.attributes.
        """
        self.build(dictionary={"CFLAGS_EXTRAS": "-march=rv64gc_zbb"})

        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))

        self.expect("disassemble --name do_zbb_stuff")
        output = self.res.GetOutput()

        for instr in self.expected_zbb_instrs:
            self.assertTrue(instr in output, "Invalid disassembler output")
