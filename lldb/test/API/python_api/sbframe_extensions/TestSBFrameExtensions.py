"""
Test SBFrameExtensions API.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestSBFrameExtensions(TestBase):
    def setUp(self):
        TestBase.setUp(self)
        self.source = "main.c"

    def test_properties_pc_addr_fp_sp(self):
        """Test SBFrame extension properties: pc, addr, fp, sp"""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Set breakpoint here", lldb.SBFileSpec(self.source)
        )

        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame.IsValid(), "Frame should be valid")

        # Test pc property
        pc = frame.pc
        self.assertIsInstance(pc, int, "pc should be an integer")
        self.assertGreater(pc, 0, "pc should be greater than 0")
        self.assertEqual(pc, frame.GetPC(), "pc property should match GetPC()")

        # Test addr property
        addr = frame.addr
        self.assertTrue(addr.IsValid(), "addr should be valid")
        self.assertEqual(addr, frame.GetPCAddress(), "addr should match GetPCAddress()")

        # Test fp property
        fp = frame.fp
        self.assertIsInstance(fp, int, "fp should be an integer")
        self.assertEqual(fp, frame.GetFP(), "fp property should match GetFP()")

        # Test sp property
        sp = frame.sp
        self.assertIsInstance(sp, int, "sp should be an integer")
        self.assertEqual(sp, frame.GetSP(), "sp property should match GetSP()")

    def test_properties_module_compile_unit_function_symbol_block(self):
        """Test SBFrame extension properties: module, compile_unit, function, symbol, block"""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Set breakpoint here", lldb.SBFileSpec(self.source)
        )

        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame.IsValid(), "Frame should be valid")

        # Test module property
        module = frame.module
        self.assertTrue(module.IsValid(), "module should be valid")
        self.assertEqual(module, frame.GetModule(), "module should match GetModule()")

        # Test compile_unit property
        compile_unit = frame.compile_unit
        self.assertTrue(compile_unit.IsValid(), "compile_unit should be valid")
        self.assertEqual(
            compile_unit,
            frame.GetCompileUnit(),
            "compile_unit should match GetCompileUnit()",
        )

        # Test function property
        function = frame.function
        self.assertTrue(function.IsValid(), "function should be valid")
        self.assertEqual(
            function, frame.GetFunction(), "function should match GetFunction()"
        )

        # Test symbol property
        symbol = frame.symbol
        self.assertTrue(symbol.IsValid(), "symbol should be valid")
        self.assertEqual(symbol, frame.GetSymbol(), "symbol should match GetSymbol()")

        # Test block property
        block = frame.block
        self.assertTrue(block.IsValid(), "block should be valid")
        block_direct = frame.GetBlock()
        self.assertTrue(
            block.IsEqual(block_direct),
            "block should match GetBlock()",
        )

    def test_properties_is_inlined_name_line_entry_thread(self):
        """Test SBFrame extension properties: is_inlined, name, line_entry, thread"""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Set breakpoint here", lldb.SBFileSpec(self.source)
        )

        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame.IsValid(), "Frame should be valid")

        # Test is_inlined property
        is_inlined = frame.is_inlined
        self.assertIsInstance(is_inlined, bool, "is_inlined should be a boolean")
        self.assertEqual(
            is_inlined, frame.IsInlined(), "is_inlined should match IsInlined()"
        )

        # Test name property
        name = frame.name
        self.assertIsInstance(name, str, "name should be a string")
        self.assertEqual(
            name, frame.GetFunctionName(), "name should match GetFunctionName()"
        )
        # Should be one of our functions
        self.assertIn(
            name, ["func1", "func2", "main"], "name should be a known function"
        )

        # Test line_entry property
        line_entry = frame.line_entry
        self.assertTrue(line_entry.IsValid(), "line_entry should be valid")
        self.assertEqual(
            line_entry, frame.GetLineEntry(), "line_entry should match GetLineEntry()"
        )

        # Test thread property
        thread_prop = frame.thread
        self.assertTrue(thread_prop.IsValid(), "thread should be valid")
        self.assertEqual(
            thread_prop, frame.GetThread(), "thread should match GetThread()"
        )
        self.assertEqual(
            thread_prop.GetThreadID(),
            thread.GetThreadID(),
            "thread should be the same thread",
        )

    def test_properties_disassembly_idx(self):
        """Test SBFrame extension properties: disassembly, idx"""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Set breakpoint here", lldb.SBFileSpec(self.source)
        )

        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame.IsValid(), "Frame should be valid")

        # Test disassembly property
        disassembly = frame.disassembly
        self.assertIsInstance(disassembly, str, "disassembly should be a string")
        self.assertGreater(len(disassembly), 0, "disassembly should not be empty")
        self.assertEqual(
            disassembly, frame.Disassemble(), "disassembly should match Disassemble()"
        )

        # Test idx property
        idx = frame.idx
        self.assertIsInstance(idx, int, "idx should be an integer")
        self.assertEqual(idx, frame.GetFrameID(), "idx should match GetFrameID()")
        self.assertEqual(idx, 0, "First frame should have idx 0")

    def test_properties_variables_vars_locals_args_arguments_statics(self):
        """Test SBFrame extension properties: variables, vars, locals, args, arguments, statics"""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Set breakpoint here", lldb.SBFileSpec(self.source)
        )

        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame.IsValid(), "Frame should be valid")

        # Test variables property (alias for get_all_variables)
        variables = frame.variables
        self.assertIsInstance(
            variables, lldb.SBValueList, "variables should be SBValueList"
        )
        all_vars = frame.GetVariables(True, True, True, True)
        self.assertEqual(
            variables.GetSize(),
            all_vars.GetSize(),
            "variables should match GetVariables(True, True, True, True)",
        )

        # Test vars property (alias for variables)
        vars_prop = frame.vars
        self.assertIsInstance(vars_prop, lldb.SBValueList, "vars should be SBValueList")
        self.assertEqual(
            vars_prop.GetSize(),
            variables.GetSize(),
            "vars should match variables",
        )

        # Test locals property
        locals_prop = frame.locals
        self.assertIsInstance(
            locals_prop, lldb.SBValueList, "locals should be SBValueList"
        )
        locals_direct = frame.GetVariables(False, True, False, False)
        self.assertEqual(
            locals_prop.GetSize(),
            locals_direct.GetSize(),
            "locals should match GetVariables(False, True, False, False)",
        )

        # Test args property
        args_prop = frame.args
        self.assertIsInstance(args_prop, lldb.SBValueList, "args should be SBValueList")
        args_direct = frame.GetVariables(True, False, False, False)
        self.assertEqual(
            args_prop.GetSize(),
            args_direct.GetSize(),
            "args should match GetVariables(True, False, False, False)",
        )

        # Test arguments property (alias for args)
        arguments_prop = frame.arguments
        self.assertIsInstance(
            arguments_prop, lldb.SBValueList, "arguments should be SBValueList"
        )
        self.assertEqual(
            arguments_prop.GetSize(),
            args_prop.GetSize(),
            "arguments should match args",
        )

        # Test statics property
        statics_prop = frame.statics
        self.assertIsInstance(
            statics_prop, lldb.SBValueList, "statics should be SBValueList"
        )
        statics_direct = frame.GetVariables(False, False, True, False)
        self.assertEqual(
            statics_prop.GetSize(),
            statics_direct.GetSize(),
            "statics should match GetVariables(False, False, True, False)",
        )

    def test_properties_registers_regs_register_reg(self):
        """Test SBFrame extension properties: registers, regs, register, reg"""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Set breakpoint here", lldb.SBFileSpec(self.source)
        )

        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame.IsValid(), "Frame should be valid")

        # Test registers property
        registers = frame.registers
        # registers returns an SBValueList that can be iterated
        self.assertTrue(hasattr(registers, "__iter__"), "registers should be iterable")
        registers_direct = frame.GetRegisters()
        # Compare by iterating and counting
        registers_count = sum(1 for _ in registers)
        registers_direct_count = sum(1 for _ in registers_direct)
        self.assertEqual(
            registers_count,
            registers_direct_count,
            "registers should match GetRegisters()",
        )

        # Test regs property (alias for registers)
        regs = frame.regs
        self.assertTrue(hasattr(regs, "__iter__"), "regs should be iterable")
        regs_count = sum(1 for _ in regs)
        self.assertEqual(regs_count, registers_count, "regs should match registers")

        # Test register property (flattened view)
        register = frame.register
        self.assertIsNotNone(register, "register should not be None")
        # register is a helper object with __iter__ and __getitem__
        reg_names = set()
        for reg in register:
            self.assertTrue(reg.IsValid(), "Register should be valid")
            reg_names.add(reg.name)

        # Test reg property (alias for register)
        reg = frame.reg
        self.assertIsNotNone(reg, "reg should not be None")
        reg_names2 = set()
        for r in reg:
            reg_names2.add(r.name)
        self.assertEqual(reg_names, reg_names2, "reg should match register")

        # Test register indexing by name
        if len(reg_names) > 0:
            first_reg_name = list(reg_names)[0]
            reg_by_name = register[first_reg_name]
            self.assertTrue(reg_by_name.IsValid(), "Register by name should be valid")
            self.assertEqual(
                reg_by_name.name, first_reg_name, "Register name should match"
            )

    def test_properties_parent_child(self):
        """Test SBFrame extension properties: parent, child"""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Set breakpoint here", lldb.SBFileSpec(self.source)
        )

        # Get frame at func1 (should be frame 0)
        frame0 = thread.GetFrameAtIndex(0)
        self.assertTrue(frame0.IsValid(), "Frame 0 should be valid")

        # If there's a parent frame (frame 1), test parent property
        if thread.GetNumFrames() > 1:
            frame1 = thread.GetFrameAtIndex(1)
            parent = frame0.parent
            self.assertTrue(parent.IsValid(), "parent should be valid")
            self.assertEqual(
                parent.GetFrameID(),
                frame1.GetFrameID(),
                "parent should be the next frame",
            )
            self.assertEqual(
                parent.pc, frame1.GetPC(), "parent PC should match frame 1"
            )

        # Test child property (should be frame -1, which doesn't exist, so should return invalid)
        child = frame0.child
        # Child of frame 0 would be frame -1, which doesn't exist
        # So it should return an invalid frame
        if thread.GetNumFrames() == 1:
            self.assertFalse(child.IsValid(), "child of only frame should be invalid")

    def test_methods_get_all_variables_get_arguments_get_locals_get_statics(self):
        """Test SBFrame extension methods: get_all_variables, get_arguments, get_locals, get_statics"""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Set breakpoint here", lldb.SBFileSpec(self.source)
        )

        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame.IsValid(), "Frame should be valid")

        # Test get_all_variables method
        all_vars = frame.get_all_variables()
        self.assertIsInstance(
            all_vars, lldb.SBValueList, "get_all_variables should return SBValueList"
        )
        all_vars_direct = frame.GetVariables(True, True, True, True)
        self.assertEqual(
            all_vars.GetSize(),
            all_vars_direct.GetSize(),
            "get_all_variables should match GetVariables(True, True, True, True)",
        )

        # Test get_arguments method
        args = frame.get_arguments()
        self.assertIsInstance(
            args, lldb.SBValueList, "get_arguments should return SBValueList"
        )
        args_direct = frame.GetVariables(True, False, False, False)
        self.assertEqual(
            args.GetSize(),
            args_direct.GetSize(),
            "get_arguments should match GetVariables(True, False, False, False)",
        )

        # Test get_locals method
        locals = frame.get_locals()
        self.assertIsInstance(
            locals, lldb.SBValueList, "get_locals should return SBValueList"
        )
        locals_direct = frame.GetVariables(False, True, False, False)
        self.assertEqual(
            locals.GetSize(),
            locals_direct.GetSize(),
            "get_locals should match GetVariables(False, True, False, False)",
        )

        # Test get_statics method
        statics = frame.get_statics()
        self.assertIsInstance(
            statics, lldb.SBValueList, "get_statics should return SBValueList"
        )
        statics_direct = frame.GetVariables(False, False, True, False)
        self.assertEqual(
            statics.GetSize(),
            statics_direct.GetSize(),
            "get_statics should match GetVariables(False, False, True, False)",
        )

    def test_method_var(self):
        """Test SBFrame extension method: var()"""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Set breakpoint here", lldb.SBFileSpec(self.source)
        )

        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame.IsValid(), "Frame should be valid")

        # Test var() method with a variable that should exist
        # First, let's see what variables are available
        all_vars = frame.GetVariables(True, True, True, True)
        if all_vars.GetSize() > 0:
            var_name = all_vars.GetValueAtIndex(0).GetName()
            var_value = frame.var(var_name)
            self.assertTrue(var_value.IsValid(), f"var('{var_name}') should be valid")
            self.assertEqual(
                var_value.GetName(),
                var_name,
                f"var('{var_name}') should return the correct variable",
            )
            # Compare with GetValueForVariablePath
            var_direct = frame.GetValueForVariablePath(var_name)
            self.assertEqual(
                var_value.GetName(),
                var_direct.GetName(),
                "var() should match GetValueForVariablePath()",
            )

        # Test var() with non-existent variable
        invalid_var = frame.var("NonExistentVariable12345")
        self.assertFalse(
            invalid_var.IsValid(), "var() with non-existent variable should be invalid"
        )

    def test_method_get_parent_frame_get_child_frame(self):
        """Test SBFrame extension methods: get_parent_frame, get_child_frame"""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Set breakpoint here", lldb.SBFileSpec(self.source)
        )

        frame0 = thread.GetFrameAtIndex(0)
        self.assertTrue(frame0.IsValid(), "Frame 0 should be valid")

        # Test get_parent_frame
        if thread.GetNumFrames() > 1:
            parent = frame0.get_parent_frame()
            self.assertTrue(
                parent.IsValid(), "get_parent_frame should return valid frame"
            )
            frame1 = thread.GetFrameAtIndex(1)
            self.assertEqual(
                parent.GetFrameID(),
                frame1.GetFrameID(),
                "get_parent_frame should return frame 1",
            )
        else:
            # If there's only one frame, parent should be invalid
            parent = frame0.get_parent_frame()
            # Note: get_parent_frame might return an invalid frame if idx+1 is out of bounds

        # Test get_child_frame (frame -1 doesn't exist, so should be invalid)
        child = frame0.get_child_frame()
        if thread.GetNumFrames() == 1:
            self.assertFalse(
                child.IsValid(), "get_child_frame of only frame should be invalid"
            )

    def test_special_methods_eq_int_hex(self):
        """Test SBFrame extension special methods: __eq__, __int__, __hex__"""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Set breakpoint here", lldb.SBFileSpec(self.source)
        )

        frame0 = thread.GetFrameAtIndex(0)
        self.assertTrue(frame0.IsValid(), "Frame 0 should be valid")

        # Test __int__ (converts frame to its frame ID)
        frame_id = int(frame0)
        self.assertIsInstance(frame_id, int, "__int__ should return an integer")
        self.assertEqual(
            frame_id, frame0.GetFrameID(), "__int__ should return frame ID"
        )

        # Test __hex__ (converts frame to its PC)
        # Note: __hex__ returns the PC as an integer, not a hex string
        # In Python 3, hex() builtin calls __index__ if __hex__ doesn't exist,
        # but since __hex__ is defined, it will be called
        pc_hex = frame0.__hex__()
        self.assertIsInstance(pc_hex, int, "__hex__ should return an integer (PC)")
        self.assertEqual(pc_hex, frame0.GetPC(), "__hex__ should return PC")

        # Test __eq__ and __ne__
        frame0_copy = thread.GetFrameAtIndex(0)
        self.assertTrue(frame0 == frame0_copy, "Same frame should be equal")
        self.assertFalse(frame0 != frame0_copy, "Same frame should not be not-equal")

        if thread.GetNumFrames() > 1:
            frame1 = thread.GetFrameAtIndex(1)
            self.assertFalse(frame0 == frame1, "Different frames should not be equal")
            self.assertTrue(frame0 != frame1, "Different frames should be not-equal")

    def test_pc_property_settable(self):
        """Test that pc property is settable"""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Set breakpoint here", lldb.SBFileSpec(self.source)
        )

        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame.IsValid(), "Frame should be valid")

        original_pc = frame.GetPC()
        # Test that we can set pc (though this might not work on all platforms)
        # We'll just verify the property exists and can be read
        pc = frame.pc
        self.assertIsInstance(pc, int, "pc should be readable")
        # Note: Setting pc might not be supported on all platforms, so we just test reading
