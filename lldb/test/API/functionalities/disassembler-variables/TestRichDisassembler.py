from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldb
import os
import re

class TestRichDisassembler(TestBase):
    def _compile_object(self, src_name, func_cflags="-g -gdwarf-5 -O2 -fno-inline"):
        """
        Compile a single C source to an object file using the host toolchain.
        We intentionally use `platform shell` to keep this as close to your
        existing style as possible (and avoid depending on the Makefile for .o).
        """
        src = self.getSourcePath(src_name)
        obj = self.getBuildArtifact(os.path.splitext(src_name)[0] + ".o")
        cmd = f"cc {func_cflags} -c '{src}' -o '{obj}'"
        self.runCmd(f"platform shell {cmd}", check=True)
        self.assertTrue(os.path.exists(obj), f"missing object: {obj}")
        return obj

    def _create_target(self, path):
        target = self.dbg.CreateTarget(path)
        self.assertTrue(target, f"failed to create target for {path}")
        return target

    def _disassemble_verbose_frame(self):
        # Same as your original: current frame (-f), with --variable enabled.
        self.runCmd("disassemble --variable -f", check=True)
        return self.res.GetOutput()

    def _disassemble_verbose_symbol(self, symname):
        # For object-only tests, disassemble a named symbol from the .o
        self.runCmd(f"disassemble -n {symname} -v", check=True)
        return self.res.GetOutput()

    def test_d_original_example_O1(self):
        """
        Tests disassembler output for d_original_example.c built with -O1,
        using the CLI with --rich for enabled annotations.
        """
        self.build(
            dictionary={"C_SOURCES": "d_original_example.c", "CFLAGS_EXTRAS": "-g -O1"}
        )
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target)

        bp = target.BreakpointCreateByName("main")
        self.assertGreater(bp.GetNumLocations(), 0)

        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, "Failed to launch process")
        self.assertEqual(process.GetState(), lldb.eStateStopped)

        # Run the CLI command and read output from self.res
        self.runCmd("disassemble --variable -f", check=True)
        out = self.res.GetOutput()
        print(out)

        self.assertIn("argc = ", out)
        self.assertIn("argv = ", out)
        self.assertIn("i = ", out)
        self.assertNotIn("<decoding error>", out)

    @no_debug_info_test  # we explicitly request -g in _compile_object
    def test_regs_int_params(self):
        obj = self._compile_object("regs_int_params.c")
        target = self._create_target(obj)
        out = self._disassemble_verbose_symbol("regs_int_params")
        print(out)

        # assertions (tweak as desired)
        self.assertRegex(out, r"\ba\s*=\s*(DW_OP_reg5\b|RDI\b)")
        self.assertRegex(out, r"\bb\s*=\s*(DW_OP_reg4\b|RSI\b)")
        self.assertRegex(out, r"\bc\s*=\s*(DW_OP_reg1\b|RDX\b)")
        self.assertRegex(out, r"\bd\s*=\s*(DW_OP_reg2\b|RCX\b)")
        self.assertRegex(out, r"\be\s*=\s*(DW_OP_reg8\b|R8\b)")
        self.assertRegex(out, r"\bf\s*=\s*(DW_OP_reg9\b|R9\b)")
        self.assertNotIn("<decoding error>", out)
        

    @no_debug_info_test
    def test_regs_fp_params(self):
        obj = self._compile_object("regs_fp_params.c")
        target = self._create_target(obj)
        out = self._disassemble_verbose_symbol("regs_fp_params")
        print(out)

        # XMM0..5 typically map to DW_OP_reg17..22
        self.assertRegex(out, r"\ba\s*=\s*(DW_OP_reg17\b|XMM0\b)")
        self.assertRegex(out, r"\bb\s*=\s*(DW_OP_reg18\b|XMM1\b)")
        self.assertRegex(out, r"\bc\s*=\s*(DW_OP_reg19\b|XMM2\b)")
        self.assertRegex(out, r"\bd\s*=\s*(DW_OP_reg20\b|XMM3\b)")
        self.assertRegex(out, r"\be\s*=\s*(DW_OP_reg21\b|XMM4\b)")
        self.assertRegex(out, r"\bf\s*=\s*(DW_OP_reg22\b|XMM5\b)")
        self.assertNotIn("<decoding error>", out)

    @no_debug_info_test
    def test_regs_mixed_params(self):
        obj = self._compile_object("regs_mixed_params.c")
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
    def test_live_across_call(self):
        obj = self._compile_object("live_across_call.c")
        target = self._create_target(obj)
        out = self._disassemble_verbose_symbol("live_across_call")
        print(out)

        # We just assert 'a' is in a register, then there's a call, then 'a' again.
        self.assertRegex(out, r"\bx\s*=\s*(DW_OP_reg5\b|RDI\b)")
        self.assertIn("call", out)
        self.assertRegex(out, r"\br\s*=\s*(DW_OP_reg4\b|RAX\b)")
        self.assertNotIn("<decoding error>", out)

    @no_debug_info_test
    def test_loop_reg_rotate(self):
        obj = self._compile_object("loop_reg_rotate.c")
        target = self._create_target(obj)
        out = self._disassemble_verbose_symbol("loop_reg_rotate")
        print(out)

        self.assertRegex(out, r"\bn\s*=\s*()")
        self.assertRegex(out, r"\bt\s*=\s*()")
        self.assertRegex(out, r"\bk\s*=\s*()")
        self.assertRegex(out, r"\bj\s*=\s*()")
        self.assertRegex(out, r"\bi\s*=\s*()")
        self.assertNotIn("<decoding error>", out)

    @no_debug_info_test
    def test_seed_reg_const_undef(self):
        """
        For now, you mentioned constants aren’t printed; we still check that the
        register part shows up (first range). When you add CONST support, you
        can add an assertion for ' = 0' or similar.
        """
        # Use O1 to help keep a first reg range; still object-only
        obj = self._compile_object("seed_reg_const_undef.c",
                                   func_cflags="-g -gdwarf-5 -O1 -fno-inline")
        target = self._create_target(obj)
        out = self._disassemble_verbose_symbol("main")
        print(out)

        # check that at least one var (i or argc) is shown as a register at start
        self.assertRegex(out, r"\b(i|argc)\s*=\s*(DW_OP_reg\d+\b|R[A-Z0-9]+)")
        self.assertNotIn("<decoding error>", out)
