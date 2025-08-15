from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldb


class TestRichDisassembler(TestBase):
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
