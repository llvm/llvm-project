"""
Test that "memory region" command can show memory tagged regions
on AArch64 Linux.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class AArch64LinuxMTEMemoryRegionTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessArch("aarch64")
    @skipUnlessPlatform(["linux"])
    @skipUnlessAArch64MTELinuxCompiler
    def test_mte_regions(self):
        if not self.isAArch64MTE():
            self.skipTest('Target must support MTE.')
        if not self.hasLinuxVmFlags():
            self.skipTest('/proc/{pid}/smaps VmFlags must be present')

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

        substrs = ["memory tagging: enabled"]
        # The new page will be tagged
        self.expect("memory region the_page", substrs=substrs)
        # Code page will not be
        self.expect("memory region main", substrs=substrs, matching=False)
