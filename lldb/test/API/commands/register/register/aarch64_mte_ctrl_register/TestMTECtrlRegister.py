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

        has_store_only = self.isAArch64MTEStoreOnly()

        def check_mte_ctrl(async_err, sync_err, store_only):
            # Bit 0 = tagged addressing enabled
            # Bit 1 = synchronous faults
            # Bit 2 = asynchronous faults
            # Bit 19 = store only checking mode
            value = "0x{:016x}".format(
                (store_only << 19) | (async_err << 2) | (sync_err << 1) | 1
            )
            expected = [value]

            if self.hasXMLSupport():
                fields = "("
                if has_store_only:
                    fields += f"STORE_ONLY = {store_only}, "

                tfc_modes = ["NONE", "SYNC", "ASYNC", "ASYMM"]
                fields += f"TAGS = 0, TCF = TCF_{tfc_modes[async_err << 1 | sync_err]}, TAGGED_ADDR_ENABLE = 1)"

                expected.append(fields)

            self.expect("register read mte_ctrl", substrs=expected)

        # We start enabled with synchronous faults.
        check_mte_ctrl(0, 1, 0)
        # Change to asynchronous faults.
        self.runCmd("register write mte_ctrl 5")
        check_mte_ctrl(1, 0, 0)
        # This would return to synchronous faults if we did not restore the
        # previous value.
        self.expect("expression setup_mte()", substrs=["= 0"])
        check_mte_ctrl(1, 0, 0)

        # Store only checking requires FEAT_MTE_STORE_ONLY.
        if has_store_only:
            self.runCmd(f"register write mte_ctrl {1 | (1 << 19)}")
            check_mte_ctrl(0, 0, 1)
