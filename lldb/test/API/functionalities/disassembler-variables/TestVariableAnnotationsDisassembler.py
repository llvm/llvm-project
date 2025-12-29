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
        """Test SBVariableAnnotator::AnnotateStructured API returns structured data"""
        obj = self._build_obj("d_original_example.o")
        target = self._create_target(obj)
        annotator = lldb.SBVariableAnnotator()

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
                f"\nTesting SBVariableAnnotator::AnnotateStructured API on {instructions.GetSize()} instructions"
            )

        expected_vars = ["argc", "argv", "i"]
        found_variables = set()

        # Test each instruction.
        for i in range(instructions.GetSize()):
            inst = instructions.GetInstructionAtIndex(i)
            self.assertTrue(inst.IsValid(), f"Invalid instruction at index {i}")

            # TODO use more python convinient get_annotations_list defined in Extensions file.
            annotations = annotator.AnnotateStructured(inst)

            self.assertIsInstance(
                annotations,
                lldb.SBStructuredData,
                "AnnotateStructured should return SBStructuredData",
            )

            self.assertTrue(
                annotations.GetSize() > 0,
                "AnnotateStructured should return non empty array",
            )

            if annotations.GetSize() > 0:
                # Validate each annotation.
                for j in range(annotations.GetSize()):
                    ann = annotations.GetItemAtIndex(j)
                    self.assertTrue(ann.IsValid(), f"Invalid annotation at index {j}")

                    self.assertEqual(
                        ann.GetType(),
                        lldb.eStructuredDataTypeDictionary,
                        "Each annotation should be a dictionary",
                    )

                    var_name_obj = ann.GetValueForKey("variable_name")
                    self.assertTrue(
                        var_name_obj.IsValid(), "Missing 'variable_name' field"
                    )

                    location_obj = ann.GetValueForKey("location_description")
                    self.assertTrue(
                        location_obj.IsValid(), "Missing 'location_description' field"
                    )

                    is_live_obj = ann.GetValueForKey("is_live")
                    self.assertTrue(is_live_obj.IsValid(), "Missing 'is_live' field")

                    start_addr_obj = ann.GetValueForKey("start_address")
                    self.assertTrue(
                        start_addr_obj.IsValid(), "Missing 'start_address' field"
                    )

                    end_addr_obj = ann.GetValueForKey("end_address")
                    self.assertTrue(
                        end_addr_obj.IsValid(), "Missing 'end_address' field"
                    )

                    register_kind_obj = ann.GetValueForKey("register_kind")
                    self.assertTrue(
                        register_kind_obj.IsValid(), "Missing 'register_kind' field"
                    )

                    var_name = var_name_obj.GetStringValue(1024)

                    # Check for expected variables in this function.
                    self.assertIn(
                        var_name, expected_vars, f"Unexpected variable name: {var_name}"
                    )

                    found_variables.add(var_name)

        if self.TraceOn():
            print(f"\nTest complete. Found variables: {found_variables}")