"""
Tests that LLDB can correctly set up a disassembler using extensions from the .riscv.attributes section.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestDisassembler(TestBase):
    expected_zbb_instrs = ["andn", "orn", "xnor", "rol", "ror", "ret"]

    def _get_llvm_tool(self, tool):
        clang = self.getCompiler()
        bindir = os.path.dirname(clang)
        candidate = os.path.join(bindir, tool)
        if os.path.exists(candidate):
            return candidate
        return lldbutil.which(tool)

    def _strip_riscv_attributes(self):
        """
        Strips the .riscv.attributes section.
        """
        exe = self.getBuildArtifact("a.out")
        stripped = self.getBuildArtifact("stripped.out")

        objcopy_path = self._get_llvm_tool("llvm-objcopy")
        self.assertTrue(objcopy_path, "llvm-objcopy not found")

        out = subprocess.run(
            [objcopy_path, "--remove-section=.riscv.attributes", exe, stripped],
            check=True,
        )

        return os.path.basename(stripped)

    @skipIf(archs=no_match("^riscv.*"))
    def test_without_riscv_attributes(self):
        """
        Tests disassembly of a riscv binary without the .riscv.attributes.
        Without the .riscv.attributes section lldb won't set up a disassembler to
        handle the bitmanip extension, so it is not expected to see zbb instructions
        in the output.
        """
        self.build(dictionary={"CFLAGS_EXTRAS": "-march=rv64gc_zbb"})
        stripped_exe = self._strip_riscv_attributes()

        lldbutil.run_to_name_breakpoint(self, "main", exe_name=stripped_exe)

        self.expect("disassemble --name do_zbb_stuff")
        output = self.res.GetOutput()

        for instr in self.expected_zbb_instrs:
            self.assertFalse(instr in output, "Zbb instructions should not be disassembled")

    @skipIf(archs=no_match("^riscv.*"))
    def test_with_riscv_attributes(self):
        """
        Tests disassembly of a riscv binary with the .riscv.attributes.
        """
        self.build(dictionary={"CFLAGS_EXTRAS": "-march=rv64gc_zbb"})

        lldbutil.run_to_name_breakpoint(self, "main")

        self.expect("disassemble --name do_zbb_stuff")
        output = self.res.GetOutput()

        for instr in self.expected_zbb_instrs:
            self.assertTrue(instr in output, "Invalid disassembler output")
