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

    @no_debug_info_test
    @skipIf(archs=no_match(["x86_64"]))
    def test_structured_annotations_api(self):
        """Test SBInstruction.variable_annotations() Python API."""
        obj = self._build_obj("d_original_example.o")
        target = self._create_target(obj)

        main_symbols = target.FindSymbols("main")
        self.assertTrue(
            main_symbols.IsValid() and main_symbols.GetSize() > 0,
            "Could not find 'main' symbol",
        )

        main_symbol = main_symbols.GetContextAtIndex(0).GetSymbol()
        start_addr = main_symbol.GetStartAddress()
        self.assertTrue(start_addr.IsValid(), "Invalid start address for main")

        instructions = target.ReadInstructions(start_addr, 16)
        self.assertGreater(instructions.GetSize(), 0, "No instructions read")

        if self.TraceOn():
            print(
                f"\nTesting SBInstruction.variable_annotations on {instructions.GetSize()} instructions"
            )

        expected_vars = ["argc", "argv", "i"]

        # Track current state of variables across instructions.
        found_variables = set()

        # Test each instruction.
        for i in range(instructions.GetSize()):
            inst = instructions.GetInstructionAtIndex(i)
            self.assertTrue(inst.IsValid(), f"Invalid instruction at index {i}")

            # Get annotations as Python list of dicts.
            annotations = inst.variable_annotations()

            for ann in annotations:
                # Validate required fields are present.
                self.assertIn("variable_name", ann, "Missing 'variable_name' field")
                self.assertIn(
                    "location_description", ann, "Missing 'location_description' field"
                )
                self.assertIn("start_address", ann, "Missing 'start_address' field")
                self.assertIn("end_address", ann, "Missing 'end_address' field")
                self.assertIn("register_kind", ann, "Missing 'register_kind' field")

                var_name = ann["variable_name"]

                # Validate types and values.
                self.assertIsInstance(var_name, str, "variable_name should be string")
                self.assertIsInstance(
                    ann["location_description"],
                    str,
                    "location_description should be string",
                )
                self.assertIsInstance(
                    ann["start_address"], int, "start_address should be integer"
                )
                self.assertIsInstance(
                    ann["end_address"], int, "end_address should be integer"
                )
                self.assertIsInstance(
                    ann["register_kind"], int, "register_kind should be integer"
                )

                self.assertGreater(
                    len(var_name), 0, "variable_name should not be empty"
                )
                self.assertGreater(
                    len(ann["location_description"]),
                    0,
                    "location_description should not be empty",
                )
                self.assertGreater(
                    ann["end_address"],
                    ann["start_address"],
                    "end_address should be > start_address",
                )

                self.assertIn(
                    var_name, expected_vars, f"Unexpected variable name: {var_name}"
                )

                found_variables.add(var_name)

        # Validate we find all expected variables.
        self.assertEqual(
            found_variables,
            set(expected_vars),
            f"Did not find all expected variables. Expected: {expected_vars}, find: {found_variables}",
        )

        if self.TraceOn():
            print(f"\nTest complete. All expected variables found: {found_variables}")
