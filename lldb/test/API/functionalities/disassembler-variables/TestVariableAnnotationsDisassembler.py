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
        """Test GetVariableAnnotations() API returns structured data"""
        obj = self._build_obj("d_original_example.o")
        target = self._create_target(obj)

        main_symbols = target.FindSymbols("main")
        self.assertTrue(main_symbols.IsValid() and main_symbols.GetSize() > 0,
                       "Could not find 'main' symbol")

        main_symbol = main_symbols.GetContextAtIndex(0).GetSymbol()
        start_addr = main_symbol.GetStartAddress()
        self.assertTrue(start_addr.IsValid(), "Invalid start address for main")

        instructions = target.ReadInstructions(start_addr, 16)
        self.assertGreater(instructions.GetSize(), 0, "No instructions read")

        print(f"\nTesting GetVariableAnnotations() API on {instructions.GetSize()} instructions")

        # Track what we find
        found_annotations = False
        found_variables = set()

        # Track variable locations to detect changes (for selective printing)
        prev_locations = {}

        # Test each instruction
        for i in range(instructions.GetSize()):
            inst = instructions.GetInstructionAtIndex(i)
            self.assertTrue(inst.IsValid(), f"Invalid instruction at index {i}")

            annotations = inst.GetVariableAnnotations(target)

            self.assertIsInstance(annotations, lldb.SBStructuredData,
                                "GetVariableAnnotations should return SBStructuredData")

            if annotations.GetSize() > 0:
                found_annotations = True

                # Track current locations and detect changes
                current_locations = {}
                should_print = False

                # Validate each annotation
                for j in range(annotations.GetSize()):
                    ann = annotations.GetItemAtIndex(j)
                    self.assertTrue(ann.IsValid(),
                                  f"Invalid annotation at index {j}")

                    self.assertEqual(ann.GetType(), lldb.eStructuredDataTypeDictionary,
                                   "Each annotation should be a dictionary")

                    var_name_obj = ann.GetValueForKey("variable_name")
                    self.assertTrue(var_name_obj.IsValid(),
                                  "Missing 'variable_name' field")

                    location_obj = ann.GetValueForKey("location_description")
                    self.assertTrue(location_obj.IsValid(),
                                  "Missing 'location_description' field")

                    is_live_obj = ann.GetValueForKey("is_live")
                    self.assertTrue(is_live_obj.IsValid(),
                                  "Missing 'is_live' field")

                    start_addr_obj = ann.GetValueForKey("start_address")
                    self.assertTrue(start_addr_obj.IsValid(),
                                  "Missing 'start_address' field")

                    end_addr_obj = ann.GetValueForKey("end_address")
                    self.assertTrue(end_addr_obj.IsValid(),
                                  "Missing 'end_address' field")

                    register_kind_obj = ann.GetValueForKey("register_kind")
                    self.assertTrue(register_kind_obj.IsValid(),
                                  "Missing 'register_kind' field")

                    # Extract and validate values
                    var_name = var_name_obj.GetStringValue(1024)
                    location = location_obj.GetStringValue(1024)
                    is_live = is_live_obj.GetBooleanValue()
                    start_addr = start_addr_obj.GetUnsignedIntegerValue()
                    end_addr = end_addr_obj.GetUnsignedIntegerValue()
                    register_kind = register_kind_obj.GetUnsignedIntegerValue()

                    # Validate types and values
                    self.assertIsInstance(var_name, str, "variable_name should be string")
                    self.assertGreater(len(var_name), 0, "variable_name should not be empty")

                    self.assertIsInstance(location, str, "location_description should be string")
                    self.assertGreater(len(location), 0, "location_description should not be empty")

                    self.assertIsInstance(is_live, bool, "is_live should be boolean")

                    self.assertIsInstance(start_addr, int, "start_address should be integer")
                    self.assertIsInstance(end_addr, int, "end_address should be integer")
                    self.assertGreater(end_addr, start_addr,
                                     "end_address should be greater than start_address")

                    self.assertIsInstance(register_kind, int, "register_kind should be integer")

                    # Check for expected variables in this function
                    self.assertIn(var_name, ["argc", "argv", "i"],
                                f"Unexpected variable name: {var_name}")

                    found_variables.add(var_name)

                    # Track current location
                    current_locations[var_name] = location

                    # Detect if this is a new variable or location changed
                    if var_name not in prev_locations or prev_locations[var_name] != location:
                        should_print = True

                    # Check optional fields (may or may not be present)
                    decl_file_obj = ann.GetValueForKey("decl_file")
                    if decl_file_obj.IsValid():
                        decl_file = decl_file_obj.GetStringValue(1024)
                        self.assertIsInstance(decl_file, str)
                        self.assertIn("d_original_example.c", decl_file,
                                    f"Expected source file d_original_example.c in {decl_file}")

                    decl_line_obj = ann.GetValueForKey("decl_line")
                    if decl_line_obj.IsValid():
                        decl_line = decl_line_obj.GetUnsignedIntegerValue()
                        self.assertIsInstance(decl_line, int)

                        # Validate declaration line matches the source code (according to d_original_example.c)
                        if var_name == "argc":
                            self.assertEqual(decl_line, 3, "argc should be declared on line 3")
                        elif var_name == "argv":
                            self.assertEqual(decl_line, 3, "argv should be declared on line 3")
                        elif var_name == "i":
                            self.assertEqual(decl_line, 4, "i should be declared on line 4")

                    type_name_obj = ann.GetValueForKey("type_name")
                    if type_name_obj.IsValid():
                        type_name = type_name_obj.GetStringValue(1024)
                        self.assertIsInstance(type_name, str)

                        # Validate declaration line matches the source code (according to d_original_example.c)
                        if var_name == "argc":
                            self.assertEqual(type_name, "int", "argc should be type 'int'")
                        elif var_name == "argv":
                            self.assertEqual(type_name, "char **", "argv should be type 'char **'")
                        elif var_name == "i":
                            self.assertEqual(type_name, "int", "i should be type 'int'")

                # Only print if something happened (location changed or variable appeared/disappeared)
                if should_print or len(current_locations) != len(prev_locations):
                    print(f"\nInstruction {i} at {inst.GetAddress()}: {annotations.GetSize()} annotations")
                    for var_name, location in current_locations.items():
                        change_marker = " <- CHANGED" if var_name in prev_locations and prev_locations[var_name] != location else ""
                        new_marker = " <- NEW" if var_name not in prev_locations else ""
                        print(f"  {var_name} = {location}{change_marker}{new_marker}")
                    # Check for disappeared variables
                    for var_name in prev_locations:
                        if var_name not in current_locations:
                            print(f"  {var_name} <- GONE")

                # Update tracking
                prev_locations = current_locations.copy()

        self.assertTrue(found_annotations,
                       "Should find at least one instruction with variable annotations")

        self.assertGreater(len(found_variables), 0,
                         "Should find at least one variable")

        print(f"\nTest complete. Found variables: {found_variables}")
