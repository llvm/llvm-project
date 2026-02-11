"""
Tests that LLDB displays an appropriate warning when .riscv.attributes contains an invalid RISC-V arch string.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestRISCVConflictingExtensions(TestBase):
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
        self.build()

        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        output = self.res.GetOutput()

        self.assertIn(
            output,
            "The .riscv.attributes section contains an invalid RISC-V arch string",
        )
