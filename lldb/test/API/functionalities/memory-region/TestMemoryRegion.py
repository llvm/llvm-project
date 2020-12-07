"""
Test the 'memory region' command.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MemoryCommandRegion(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)
        # Find the line number to break for main.c.
        self.line = line_number(
            'main.cpp',
            '// Run here before printing memory regions')

    @expectedFailureAll(oslist=["freebsd"])
    def test(self):
        self.build()

        # Set breakpoint in main and run
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=-1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        interp = self.dbg.GetCommandInterpreter()
        result = lldb.SBCommandReturnObject()

        # Test that the first 'memory region' command prints the usage.
        interp.HandleCommand("memory region", result)
        self.assertFalse(result.Succeeded())
        self.assertRegexpMatches(result.GetError(), "Usage: memory region ADDR")

        # Test that when the address fails to parse, we show an error and do not continue
        interp.HandleCommand("memory region not_an_address", result)
        self.assertFalse(result.Succeeded())
        self.assertEqual(result.GetError(),
                "error: invalid address argument \"not_an_address\": address expression \"not_an_address\" evaluation failed\n")

        # Now let's print the memory region starting at 0 which should always work.
        interp.HandleCommand("memory region 0x0", result)
        self.assertTrue(result.Succeeded())
        self.assertRegexpMatches(result.GetOutput(), "\\[0x0+-")

        # Keep printing memory regions until we printed all of them.
        while True:
            interp.HandleCommand("memory region", result)
            if not result.Succeeded():
                break

        # Now that we reached the end, 'memory region' should again print the usage.
        interp.HandleCommand("memory region", result)
        self.assertFalse(result.Succeeded())
        self.assertRegexpMatches(result.GetError(), "Usage: memory region ADDR")
