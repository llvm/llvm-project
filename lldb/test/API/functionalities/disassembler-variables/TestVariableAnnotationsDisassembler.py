from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldb
import os
import re


# Requires ELF assembler directives (.section … @progbits, .ident, etc.);
# not compatible with COFF/Mach-O toolchains.
@skipUnlessPlatform(["linux", "android", "freebsd", "netbsd"])
class TestVariableAnnotationsDisassembler(TestBase):
    def _build_obj(self, obj_name: str) -> str:
        # Let the Makefile build all .o’s (pattern rule). Then grab the one we need.
        self.build()
        obj = self.getBuildArtifact(obj_name)
        self.assertTrue(os.path.exists(obj), f"missing object: {obj}")
        return obj

    def _create_target(self, path):
        target = self.dbg.CreateTarget(path)
        self.assertTrue(target, f"failed to create target for {path}")
        return target

    def _disassemble_verbose_symbol(self, symname):
        self.runCmd(f"disassemble -n {symname} -v", check=True)
        return self.res.GetOutput()

    @skipIf(archs=no_match(["x86_64"]))
    def test_d_original_example_O1(self):
        obj = self._build_obj("d_original_example.o")
        target = self._create_target(obj)
        out = self._disassemble_verbose_symbol("main")
        print(out)
        self.assertIn("argc = ", out)
        self.assertIn("argv = ", out)
        self.assertIn("i = ", out)
        self.assertNotIn("<decoding error>", out)

    @no_debug_info_test
    @skipIf(archs=no_match(["x86_64"]))
    def test_regs_int_params(self):
        obj = self._build_obj("regs_int_params.o")
        target = self._create_target(obj)
        out = self._disassemble_verbose_symbol("regs_int_params")
        print(out)
        self.assertRegex(out, r"\ba\s*=\s*(DW_OP_reg5\b|RDI\b)")
        self.assertRegex(out, r"\bb\s*=\s*(DW_OP_reg4\b|RSI\b)")
        self.assertRegex(out, r"\bc\s*=\s*(DW_OP_reg1\b|RDX\b)")
        self.assertRegex(out, r"\bd\s*=\s*(DW_OP_reg2\b|RCX\b)")
        self.assertRegex(out, r"\be\s*=\s*(DW_OP_reg8\b|R8\b)")
        self.assertRegex(out, r"\bf\s*=\s*(DW_OP_reg9\b|R9\b)")
        self.assertNotIn("<decoding error>", out)

    @no_debug_info_test
    @skipIf(archs=no_match(["x86_64"]))
    def test_regs_fp_params(self):
        obj = self._build_obj("regs_fp_params.o")
        target = self._create_target(obj)
        out = self._disassemble_verbose_symbol("regs_fp_params")
        print(out)
        self.assertRegex(out, r"\ba\s*=\s*(DW_OP_reg17\b|XMM0\b)")
        self.assertRegex(out, r"\bb\s*=\s*(DW_OP_reg18\b|XMM1\b)")
        self.assertRegex(out, r"\bc\s*=\s*(DW_OP_reg19\b|XMM2\b)")
        self.assertRegex(out, r"\bd\s*=\s*(DW_OP_reg20\b|XMM3\b)")
        self.assertRegex(out, r"\be\s*=\s*(DW_OP_reg21\b|XMM4\b)")
        self.assertRegex(out, r"\bf\s*=\s*(DW_OP_reg22\b|XMM5\b)")
        self.assertNotIn("<decoding error>", out)

    @no_debug_info_test
    @skipIf(archs=no_match(["x86_64"]))
    def test_regs_mixed_params(self):
        obj = self._build_obj("regs_mixed_params.o")
        target = self._create_target(obj)
        out = self._disassemble_verbose_symbol("regs_mixed_params")
        print(out)
        self.assertRegex(out, r"\ba\s*=\s*(DW_OP_reg5\b|RDI\b)")
        self.assertRegex(out, r"\bb\s*=\s*(DW_OP_reg4\b|RSI\b)")
        self.assertRegex(out, r"\bx\s*=\s*(DW_OP_reg17\b|XMM0\b|DW_OP_reg\d+\b)")
        self.assertRegex(out, r"\by\s*=\s*(DW_OP_reg18\b|XMM1\b|DW_OP_reg\d+\b)")
        self.assertRegex(out, r"\bc\s*=\s*(DW_OP_reg1\b|RDX\b)")
        self.assertRegex(out, r"\bz\s*=\s*(DW_OP_reg19\b|XMM2\b|DW_OP_reg\d+\b)")
        self.assertNotIn("<decoding error>", out)

    @no_debug_info_test
    @skipIf(archs=no_match(["x86_64"]))
    def test_live_across_call(self):
        obj = self._build_obj("live_across_call.o")
        target = self._create_target(obj)
        out = self._disassemble_verbose_symbol("live_across_call")
        print(out)
        self.assertRegex(out, r"\bx\s*=\s*(DW_OP_reg5\b|RDI\b)")
        self.assertIn("call", out)
        self.assertRegex(out, r"\br\s*=\s*(DW_OP_reg0\b|RAX\b|DW_OP_reg\d+\b)")
        self.assertNotIn("<decoding error>", out)

    @no_debug_info_test
    @skipIf(archs=no_match(["x86_64"]))
    def test_loop_reg_rotate(self):
        obj = self._build_obj("loop_reg_rotate.o")
        target = self._create_target(obj)
        out = self._disassemble_verbose_symbol("loop_reg_rotate")
        print(out)
        self.assertRegex(out, r"\bn\s*=\s*(DW_OP_reg\d+\b|R[A-Z0-9]+)")
        self.assertRegex(out, r"\bseed\s*=\s*(DW_OP_reg\d+\b|R[A-Z0-9]+)")
        self.assertRegex(out, r"\bk\s*=\s*(DW_OP_reg\d+\b|R[A-Z0-9]+)")
        self.assertRegex(out, r"\bj\s*=\s*(DW_OP_reg\d+\b|R[A-Z0-9]+)")
        self.assertRegex(out, r"\bi\s*=\s*(DW_OP_reg\d+\b|R[A-Z0-9]+)")
        self.assertNotIn("<decoding error>", out)

    @no_debug_info_test
    @skipIf(archs=no_match(["x86_64"]))
    def test_seed_reg_const_undef(self):
        obj = self._build_obj("seed_reg_const_undef.o")
        target = self._create_target(obj)
        out = self._disassemble_verbose_symbol("main")
        print(out)
        self.assertRegex(out, r"\b(i|argc)\s*=\s*(DW_OP_reg\d+\b|R[A-Z0-9]+)")
        self.assertNotIn("<decoding error>", out)
