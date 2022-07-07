"""
Test the 'memory region' command.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MemoryCommandRegion(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)
        # Find the line number to break for main.c.
        self.line = line_number(
            'main.cpp',
            '// Run here before printing memory regions')

    def test_help(self):
        """ Test that help shows you must have one of address or --all, not both."""
        self.expect("help memory region",
            substrs=["memory region <address-expression>",
                     "memory region -a"])

    def setup_program(self):
        self.build()

        # Set breakpoint in main and run
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=-1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

    def test_command(self):
        self.setup_program()

        interp = self.dbg.GetCommandInterpreter()
        result = lldb.SBCommandReturnObject()

        # Test that the first 'memory region' command prints the usage.
        interp.HandleCommand("memory region", result)
        self.assertFalse(result.Succeeded())
        self.assertEqual(result.GetError(),
                    "error: 'memory region' takes one argument or \"--all\" option:\n"
                    "Usage: memory region <address-expression> (or --all)\n")

        # We allow --all or an address argument, not both
        interp.HandleCommand("memory region --all 0", result)
        self.assertFalse(result.Succeeded())
        self.assertRegexpMatches(result.GetError(),
                "The \"--all\" option cannot be used when an address argument is given")

        # Test that when the address fails to parse, we show an error and do not continue
        interp.HandleCommand("memory region not_an_address", result)
        self.assertFalse(result.Succeeded())
        self.assertEqual(result.GetError(),
                "error: invalid address argument \"not_an_address\": address expression \"not_an_address\" evaluation failed\n")

        # Accumulate the results to compare with the --all output
        all_regions = ""

        # Now let's print the memory region starting at 0 which should always work.
        interp.HandleCommand("memory region 0x0", result)
        self.assertTrue(result.Succeeded())
        self.assertRegexpMatches(result.GetOutput(), "\\[0x0+-")
        all_regions += result.GetOutput()

        # Keep printing memory regions until we printed all of them.
        while True:
            interp.HandleCommand("memory region", result)
            if not result.Succeeded():
                break
            all_regions += result.GetOutput()

        # Now that we reached the end, 'memory region' should again print the usage.
        interp.HandleCommand("memory region", result)
        self.assertFalse(result.Succeeded())
        self.assertRegexpMatches(result.GetError(), "Usage: memory region <address\-expression> \(or \-\-all\)")

        # --all should match what repeating the command gives you
        interp.HandleCommand("memory region --all", result)
        self.assertTrue(result.Succeeded())
        self.assertEqual(result.GetOutput(), all_regions)

    def test_no_overlapping_regions(self):
        # In the past on Windows we were recording AllocationBase as the base address
        # of the current region, not BaseAddress. So if a range of pages was split
        # into regions you would see several regions with the same base address.
        # This checks that this no longer happens (and it shouldn't happen on any
        # other OS either).
        self.setup_program()

        regions = self.process().GetMemoryRegions()
        num_regions = regions.GetSize()

        if num_regions:
            region = lldb.SBMemoryRegionInfo()
            regions.GetMemoryRegionAtIndex(0, region)
            previous_base = region.GetRegionBase()
            previous_end = region.GetRegionEnd()

            for idx in range(1, regions.GetSize()):
                regions.GetMemoryRegionAtIndex(idx, region)

                # Check that it does not overlap the previous region.
                # This could happen if we got the base addresses or size wrong.
                # Also catches the base addresses being the same.
                region_base = region.GetRegionBase()
                region_end = region.GetRegionEnd()

                self.assertFalse(
                    (region_base < previous_end) and (previous_base < region_end),
                    "Unexpected overlapping memory region found.")

                previous_base = region_base
                previous_end = region_end