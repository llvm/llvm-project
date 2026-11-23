"""
Tests that LLDB can correctly set up a disassembler using extensions from the .riscv.attributes section.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import os


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
        yaml = os.path.join(self.getSourceDir(), "stripped.out.yaml")
        exe = self.getBuildArtifact("stripped.out")
        self.yaml2obj(yaml, exe)

        target = self.dbg.CreateTarget(exe)

        self.expect("disassemble --name do_zbb_stuff")
        output = self.res.GetOutput()

        for instr in self.expected_zbb_instrs:
            self.assertFalse(
                instr in output, "Zbb instructions should not be disassembled"
            )

        self.assertEqual(
            output.count("unknown"),
            len(self.expected_zbb_instrs),
            "Instructions from the Zbb extension should be displayed as <unknown>",
        )

    @skipIfLLVMTargetMissing("RISCV")
    def test_with_riscv_attributes(self):
        """
        Tests disassembly of a riscv binary with the .riscv.attributes.
        """
        yaml = os.path.join(self.getSourceDir(), "a.out.yaml")
        exe = self.getBuildArtifact("a.out")
        self.yaml2obj(yaml, exe)

        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))

        self.expect("disassemble --name do_zbb_stuff")
        output = self.res.GetOutput()

        for instr in self.expected_zbb_instrs:
            self.assertTrue(instr in output, "Invalid disassembler output")

    @skipIfLLVMTargetMissing("RISCV")
    def test_conflicting_extensions(self):
        """
        This test demonstrates the scenario where:
        1. file_with_zcd.c is compiled with rv64gc (includes C and D).
        2. file_with_zcmp.c is compiled with rv64imad_zcmp (includes Zcmp).
        3. The linker merges .riscv.attributes, creating the union: C + D + Zcmp.

        The Zcmp extension is incompatible with the C extension when the D extension is enabled.
        Therefore, the arch string contains conflicting extensions, and LLDB should
        display an appropriate warning in this case.
        """
        yaml = os.path.join(self.getSourceDir(), "conflicting.out.yaml")
        exe = self.getBuildArtifact("conflicting.out")
        self.yaml2obj(yaml, exe)

        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        output = self.res.GetOutput()

        self.assertIn(
            output,
            "The .riscv.attributes section contains an invalid RISC-V arch string",
        )
