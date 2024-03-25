"""
Test that "memory region" lookup uses the ABI plugin to remove
non address bits from addresses before lookup.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class AArch64LinuxTaggedMemoryRegionTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    # AArch64 Linux always enables the top byte ignore feature
    @skipUnlessArch("aarch64")
    @skipUnlessPlatform(["linux"])
    def test_mte_regions(self):
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self,
            "main.c",
            line_number("main.c", "// Set break point at this line."),
            num_expected_locations=1,
        )

        self.runCmd("run", RUN_SUCCEEDED)

        if self.process().GetState() == lldb.eStateExited:
            self.fail("Test program failed to run.")

        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        # Despite the non address bits we should find a region
        self.expect(
            "memory region the_page", patterns=["\[0x[0-9A-Fa-f]+-0x[0-9A-Fa-f]+\) r-x"]
        )

        # Check that the usual error message is displayed after repeating
        # the command until the last region.
        self.runCmd("memory region 0")

        # Count the number of repeats for use in the next check
        repeats = 0
        interp = self.dbg.GetCommandInterpreter()
        result = lldb.SBCommandReturnObject()

        while True:
            interp.HandleCommand("memory region", result)
            if result.Succeeded():
                repeats += 1
            else:
                self.assertRegex(result.GetError(), "Usage: memory region")
                break

        # This time repeat until we get the last region. At that
        # point the previous address will have non-address bits in it.
        self.runCmd("memory region 0")
        for i in range(repeats):
            self.runCmd("memory region")

        # This should not error, since the user supplied address overrides
        # the previous end address.
        self.expect(
            "memory region the_page", patterns=["\[0x[0-9A-Fa-f]+-0x[0-9A-Fa-f]+\) r-x"]
        )
