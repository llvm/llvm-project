"""
Test that LLDB correctly reads, writes and restores the MTE control register on
AArch64 Linux.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MTECtrlRegisterTestCase(TestBase):
    @no_debug_info_test
    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(["linux"]))
    def test_mte_ctrl_register(self):
        if not self.isAArch64MTE():
            self.skipTest("Target must support MTE.")

        self.build()
        self.line = line_number("main.c", "// Set a break point here.")

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", self.line, num_expected_locations=1
        )
        self.runCmd("run", RUN_SUCCEEDED)

        self.expect(
            "process status",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stop reason = breakpoint 1."],
        )

        # Bit 0 = tagged addressing enabled
        # Bit 1 = synchronous faults
        # Bit 2 = asynchronous faults
        # We start enabled with synchronous faults.
        self.expect("register read mte_ctrl", substrs=["0x0000000000000003"])

        # Change to asynchronous faults.
        self.runCmd("register write mte_ctrl 5")
        self.expect("register read mte_ctrl", substrs=["0x0000000000000005"])

        # This would return to synchronous faults if we did not restore the
        # previous value.
        self.expect("expression setup_mte()", substrs=["= 0"])
        self.expect("register read mte_ctrl", substrs=["0x0000000000000005"])