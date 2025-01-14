"""
Check that lldb features work when the AArch64 Guarded Control Stack (GCS)
extension is enabled.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class AArch64LinuxGCSTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessArch("aarch64")
    @skipUnlessPlatform(["linux"])
    def test_gcs_region(self):
        if not self.isAArch64GCS():
            self.skipTest("Target must support GCS.")

        # This test assumes that we have /proc/<PID>/smaps files
        # that include "VmFlags:" lines.
        # AArch64 kernel config defaults to enabling smaps with
        # PROC_PAGE_MONITOR and "VmFlags" was added in kernel 3.8,
        # before GCS was supported at all.

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

        # By now either the program or the system C library enabled GCS and there
        # should be one region marked for use by it (we cannot predict exactly
        # where it will be).
        self.runCmd("memory region --all")
        found_ss = False
        for line in self.res.GetOutput().splitlines():
            if line.strip() == "shadow stack: yes":
                if found_ss:
                    self.fail("Found more than one shadow stack region.")
                found_ss = True

        self.assertTrue(found_ss, "Failed to find a shadow stack region.")

        # Note that we must let the debugee get killed here as it cannot exit
        # cleanly if GCS was manually enabled.
