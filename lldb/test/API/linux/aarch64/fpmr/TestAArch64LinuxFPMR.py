"""
Test lldb's ability to read and write the AArch64 FPMR register.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class AArch64LinuxFPMR(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessArch("aarch64")
    @skipUnlessPlatform(["linux"])
    def test_fpmr_register(self):
        if not self.isAArch64FPMR():
            self.skipTest("FPMR must be present.")

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

        # This has been set by the program.
        expected_fpmr = (0b101010 << 32) | 0b101
        self.expect(
            "register read --all",
            substrs=["Floating Point Mode Register", f"fpmr = {expected_fpmr:#018x}"],
        )

        # Write a value for the program to find. Same fields but with bit values
        # inverted.
        new_fpmr = (0b010101 << 32) | 0b010
        self.runCmd(f"register write fpmr {new_fpmr:#x}")

        # This value should be saved and restored after expressions.
        self.runCmd("p expr_func()")
        self.expect("register read fpmr", substrs=[f"fpmr = {new_fpmr:#018x}"])

        # 0 means the program found the new value in the sysreg as expected.
        self.expect("continue", substrs=["exited with status = 0"])
