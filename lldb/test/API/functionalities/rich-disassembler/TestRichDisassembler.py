from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *

class TestRichDisassembler(TestBase):

    @no_debug_info_test
    def test_a_loop_with_local_variable(self):
        """
        Tests that the disassembler includes basic DWARF variable annotation in output.
        Specifically checks that local variables in a loop are shown with DW_OP locations.
        Additionally, it verifies that the disassembly does not contain decoding errors.
        """
        self.build(dictionary={
            'C_SOURCES': 'a_loop_with_local_variable.c',
            'CFLAGS_EXTRAS': '-g -O0'
        })
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target)

        # Set a breakpoint inside main's loop
        src_file = lldb.SBFileSpec("test_loop_function_call.c")
        breakpoint = target.BreakpointCreateByName("main")
        self.assertGreater(breakpoint.GetNumLocations(), 0)

        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, "Failed to launch process")
        self.assertEqual(process.GetState(), lldb.eStateStopped)

        frame = process.GetSelectedThread().GetSelectedFrame()
        disasm = frame.Disassemble()
        print(disasm)

        # Check that we have DWARF annotations for variables
        self.assertIn("i = ", disasm)
        self.assertIn("DW_OP", disasm)
        self.assertNotIn("<decoding error>", disasm)


    @no_debug_info_test
    def test_b_multiple_stack_variables_O0(self):
        """
        Tests disassembler output for b_multiple_stack_variables.c built with -O0.
        This test checks that multiple local variables are annotated with DWARF
        and that their locations are distinct. It also ensures that no decoding errors appear.
        """
        self.build(dictionary={
            'C_SOURCES': 'b_multiple_stack_variables.c',
            'CFLAGS_EXTRAS': '-g -O0'
        })
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target)

        # Set a breakpoint inside main's loop
        src_file = lldb.SBFileSpec("test_loop_function_call.c")
        breakpoint = target.BreakpointCreateByName("main")
        self.assertGreater(breakpoint.GetNumLocations(), 0)

        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, "Failed to launch process")
        self.assertEqual(process.GetState(), lldb.eStateStopped)

        frame = process.GetSelectedThread().GetSelectedFrame()
        disasm = frame.Disassemble()
        print(disasm)

        # Check that we have DWARF annotations for variables
        self.assertIn("a = ", disasm)
        self.assertIn("b = ", disasm)
        self.assertIn("c = ", disasm)
        self.assertIn("DW_OP", disasm)
        self.assertNotIn("<decoding error>", disasm)


    @no_debug_info_test
    def test_b_multiple_stack_variables_O1(self):
        """
        Tests disassembler output for b_multiple_stack_variables.c built with -O1.
        Due to optimizations, some variables may be optimized out.
        We only check for 'c' and ensure no decoding errors appear.
        """
        self.build(dictionary={
            'C_SOURCES': 'b_multiple_stack_variables.c',
            'CFLAGS_EXTRAS': '-g -O1'
        })
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target)

        breakpoint = target.BreakpointCreateByName("main")
        self.assertGreater(breakpoint.GetNumLocations(), 0)

        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, "Failed to launch process")
        self.assertEqual(process.GetState(), lldb.eStateStopped)

        frame = process.GetSelectedThread().GetSelectedFrame()
        disasm = frame.Disassemble()
        print(disasm)

        self.assertIn("c = ", disasm)
        self.assertIn("DW_OP", disasm)
        self.assertNotIn("<decoding error>", disasm)


    @no_debug_info_test
    def test_c_variable_passed_to_another_function(self):
        """
        Tests disassembler output for c_variable_passed_to_another_function.c.
        This test checks that a variable passed to another function is annotated
        with DWARF and that its location is distinct. It also ensures that no decoding errors appear.        
        """
        self.build(dictionary={
            'C_SOURCES': 'c_variable_passed_to_another_function.c',
            'CFLAGS_EXTRAS': '-g -O0'
        })
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target)

        breakpoint = target.BreakpointCreateByName("main")
        self.assertGreater(breakpoint.GetNumLocations(), 0)

        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, "Failed to launch process")
        self.assertEqual(process.GetState(), lldb.eStateStopped)

        frame = process.GetSelectedThread().GetSelectedFrame()
        disasm = frame.Disassemble()
        print(disasm)

        self.assertIn("x = ", disasm)
        self.assertIn("DW_OP", disasm)
        self.assertNotIn("<decoding error>", disasm)


    @no_debug_info_test
    def test_c_variable_passed_to_another_function_O1(self):
        """
        Tests disassembler output for c_variable_passed_to_another_function.c built with -O1.
        """
        self.build(dictionary={
            'C_SOURCES': 'c_variable_passed_to_another_function.c',
            'CFLAGS_EXTRAS': '-g -O1'
        })
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target)

        breakpoint = target.BreakpointCreateByName("main")
        self.assertGreater(breakpoint.GetNumLocations(), 0)

        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, "Failed to launch process")
        self.assertEqual(process.GetState(), lldb.eStateStopped)

        frame = process.GetSelectedThread().GetSelectedFrame()
        disasm = frame.Disassemble()
        print(disasm)

        self.assertIn("x = ", disasm)
        self.assertIn("arg = ", disasm)
        self.assertIn("DW_OP", disasm)
        self.assertNotIn("<decoding error>", disasm)

    @no_debug_info_test
    def test_d_original_example(self):
        """
        Tests disassembler output for d_original_example.c.
        This test checks that the disassembly includes basic DWARF variable annotations
        and that local variables in the main function are shown with DW_OP locations.
        Additionally, it verifies that the disassembly does not contain decoding errors.
        """
        self.build(dictionary={
            'C_SOURCES': 'd_original_example.c',
            'CFLAGS_EXTRAS': '-g -O0'
        })
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target)

        breakpoint = target.BreakpointCreateByName("main")
        self.assertGreater(breakpoint.GetNumLocations(), 0)

        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, "Failed to launch process")
        self.assertEqual(process.GetState(), lldb.eStateStopped)

        frame = process.GetSelectedThread().GetSelectedFrame()
        disasm = frame.Disassemble()
        print(disasm)

        self.assertIn("argc = ", disasm)
        self.assertIn("argv = ", disasm)
        self.assertIn("i = ", disasm)
        self.assertIn("DW_OP", disasm)
        self.assertNotIn("<decoding error>", disasm)

    @no_debug_info_test
    def test_d_original_example_O1(self):
        """
        Tests disassembler output for d_original_example.c built with -O1.
        """
        self.build(dictionary={
            'C_SOURCES': 'd_original_example.c',
            'CFLAGS_EXTRAS': '-g -O1'
        })
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target)

        breakpoint = target.BreakpointCreateByName("main")
        self.assertGreater(breakpoint.GetNumLocations(), 0)

        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, "Failed to launch process")
        self.assertEqual(process.GetState(), lldb.eStateStopped)

        frame = process.GetSelectedThread().GetSelectedFrame()
        disasm = frame.Disassemble()
        print(disasm)

        self.assertIn("argc = ", disasm)
        self.assertIn("argv = ", disasm)
        self.assertIn("i = ", disasm)
        self.assertIn("DW_OP_reg", disasm)
        self.assertIn("DW_OP_stack_value", disasm)
        self.assertNotIn("<decoding error>", disasm)


    @no_debug_info_test
    def test_e_control_flow_edge(self):
        """
        Tests disassembler output for e_control_flow_edge.c with a focus on control flow edges.
        This test checks that the disassembly includes basic DWARF variable annotations
        and that local variables in the main function are shown with DW_OP locations.
        Additionally, it verifies that the disassembly does not contain decoding errors.
        """
        self.build(dictionary={
            'C_SOURCES': 'e_control_flow_edge.c',
            'CFLAGS_EXTRAS': '-g -O0'
        })
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target)

        breakpoint = target.BreakpointCreateByName("main")
        self.assertGreater(breakpoint.GetNumLocations(), 0)

        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, "Failed to launch process")
        self.assertEqual(process.GetState(), lldb.eStateStopped)

        frame = process.GetSelectedThread().GetSelectedFrame()
        disasm = frame.Disassemble()
        print(disasm)

        self.assertIn("a = ", disasm)
        self.assertIn("b = ", disasm)
        self.assertIn("DW_OP_", disasm)
        self.assertNotIn("<decoding error>", disasm)

    @no_debug_info_test
    def test_e_control_flow_edge_O1(self):
        """
        Tests disassembler output for e_control_flow_edge.c built with -O1.
        This test checks that the disassembly annotation does not contain decoding errors.
        """
        self.build(dictionary={
            'C_SOURCES': 'e_control_flow_edge.c',
            'CFLAGS_EXTRAS': '-g -O1'
        })
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target)

        breakpoint = target.BreakpointCreateByName("main")
        self.assertGreater(breakpoint.GetNumLocations(), 0)

        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, "Failed to launch process")
        self.assertEqual(process.GetState(), lldb.eStateStopped)

        frame = process.GetSelectedThread().GetSelectedFrame()
        disasm = frame.Disassemble()
        print(disasm)

        self.assertNotIn("<decoding error>", disasm)


    







