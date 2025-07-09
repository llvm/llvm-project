from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *

class TestRichDisassembler(TestBase):
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
        # self.assertIn("DW_OP_reg", disasm)
        # self.assertIn("DW_OP_stack_value", disasm)
        self.assertNotIn("<decoding error>", disasm)


    







