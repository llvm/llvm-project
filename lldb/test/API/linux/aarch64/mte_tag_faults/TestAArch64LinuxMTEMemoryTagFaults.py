"""
Test reporting of MTE tag access faults.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class AArch64LinuxMTEMemoryTagFaultsTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setup_mte_test(self, fault_type):
        if not self.isAArch64MTE():
            self.skipTest("Target must support MTE.")

        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self,
            "main.c",
            line_number("main.c", "// Breakpoint here"),
            num_expected_locations=1,
        )

        self.runCmd("run {}".format(fault_type), RUN_SUCCEEDED)

        if self.process().GetState() == lldb.eStateExited:
            self.fail("Test program failed to run.")

        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

    @skipUnlessArch("aarch64")
    @skipUnlessPlatform(["linux"])
    @skipUnlessAArch64MTELinuxCompiler
    def test_mte_tag_fault_sync(self):
        self.setup_mte_test("sync")
        # The logical tag should be included in the fault address
        # and we know what the bottom byte should be.
        # It will be 0x10 (to be in the 2nd granule), +1 to be 0x11.
        # Which tests that lldb-server handles fault addresses that
        # are not granule aligned.
        self.expect(
            "continue",
            patterns=[
                "\* thread #1, name = 'a.out', stop reason = signal SIGSEGV: "
                "sync tag check fault \(fault address=0x9[0-9A-Fa-f]+11\ "
                "logical tag=0x9 allocation tag=0xa\)"
            ],
        )

    @skipUnlessArch("aarch64")
    @skipUnlessPlatform(["linux"])
    @skipUnlessAArch64MTELinuxCompiler
    def test_mte_tag_fault_async(self):
        self.setup_mte_test("async")
        self.expect(
            "continue",
            substrs=[
                "* thread #1, name = 'a.out', stop reason = "
                "signal SIGSEGV: async tag check fault"
            ],
        )
