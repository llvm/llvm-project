"""
Test lldb's ability to read and write the AArch64 FPMR register.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class AArch64LinuxFPMR(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    # The value set by the inferior.
    EXPECTED_FPMR = (0b101010 << 32) | 0b101
    EXPECTED_FPMR_FIELDS = ["LSCALE2 = 42", "F8S1 = FP8_E4M3 | 0x4"]

    @skipUnlessArch("aarch64")
    @skipUnlessPlatform(["linux"])
    def test_fpmr_register_live(self):
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
        self.expect(
            "register read --all",
            substrs=[
                "Floating Point Mode Register",
                f"fpmr = {self.EXPECTED_FPMR:#018x}",
            ],
        )

        if self.hasXMLSupport():
            self.expect("register read fpmr", substrs=self.EXPECTED_FPMR_FIELDS)

        # Write a value for the program to find. Same fields but with bit values
        # inverted.
        new_fpmr = (0b010101 << 32) | 0b010
        self.runCmd(f"register write fpmr {new_fpmr:#x}")

        # This value should be saved and restored after expressions.
        self.runCmd("p expr_func()")
        self.expect("register read fpmr", substrs=[f"fpmr = {new_fpmr:#018x}"])

        # 0 means the program found the new value in the sysreg as expected.
        self.expect("continue", substrs=["exited with status = 0"])

    @skipIfLLVMTargetMissing("AArch64")
    def test_fpmr_register_core(self):
        if not self.isAArch64FPMR():
            self.skipTest("FPMR must be present.")

        self.runCmd("target create --core corefile")

        self.expect(
            "register read --all",
            substrs=[
                "Floating Point Mode Register",
                f"fpmr = {self.EXPECTED_FPMR:#018x}",
            ],
        )
        self.expect("register read fpmr", substrs=self.EXPECTED_FPMR_FIELDS)
