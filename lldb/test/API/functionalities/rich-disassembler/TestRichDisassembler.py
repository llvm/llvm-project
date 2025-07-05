from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *

class TestRichDisassembler(TestBase):
    """
    Tests that the disassembler includes DWARF variable annotations in output.
    Specifically checks that variables like 'a' and 'temp' are shown with DW_OP locations.
    """
    def test_variable_annotation(self):
        print("Building with:", self.getCompiler())
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target)

        src_file = lldb.SBFileSpec("main.cpp")
        breakpoint = target.BreakpointCreateByName("test")
        print("Breakpoint locations:", breakpoint.GetNumLocations())
        self.assertGreater(breakpoint.GetNumLocations(), 0)

        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        print("Process state:", process.GetState())
        print("Exit status:", process.GetExitStatus())
        print("Exit description:", process.GetExitDescription())

        self.assertTrue(process, "Failed to launch process")
        self.assertEqual(process.GetState(), lldb.eStateStopped, "Process did not stop")

        frame = process.GetSelectedThread().GetSelectedFrame()
        disasm = frame.Disassemble()
        print(disasm)

        # Check that at least one DWARF annotation is shown.
        self.assertIn("DW_OP_", disasm)

        # Check that at least one variable name is annotated.
        self.assertRegex(disasm, r'[a-zA-Z_]\w*\s*=\s*DW_OP_')

