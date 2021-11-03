"""
Test that "memory region" lookup uses the ABI plugin to remove
non address bits from addresses before lookup.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class AArch64LinuxTaggedMemoryRegionTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    # AArch64 Linux always enables the top byte ignore feature
    @skipUnlessArch("aarch64")
    @skipUnlessPlatform(["linux"])
    def test_mte_regions(self):
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(self, "main.c",
            line_number('main.c', '// Set break point at this line.'),
            num_expected_locations=1)

        self.runCmd("run", RUN_SUCCEEDED)

        if self.process().GetState() == lldb.eStateExited:
            self.fail("Test program failed to run.")

        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs=['stopped',
                     'stop reason = breakpoint'])

        # Despite the non address bits we should find a region
        self.expect("memory region the_page", patterns=[
            "\[0x[0-9A-Fa-f]+-0x[0-9A-Fa-f]+\) r-x"])
