"""
Test lldb's ability to read and write the LoongArch LASX registers.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LoongArch64LinuxRegisters(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def make_lasx_value(self, n):
        return "{" + " ".join(["0x{:02x}".format(n)] * 32) + "}"

    def check_lasx_values(self, value_offset):
        for i in range(32):
            self.expect(
                "register read xr{}".format(i),
                substrs=[self.make_lasx_value(i + value_offset)],
            )

    def lasx_registers_impl(self):
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

        self.check_lasx_values(0)
        self.runCmd("expression write_lasx_regs(2)")
        self.check_lasx_values(2)

        for i in range(32):
            self.runCmd(
                'register write xr{} "{}"'.format(i, self.make_lasx_value(i + 1))
            )

        # Should be visible within lldb.
        self.check_lasx_values(1)

        # The program should agree with lldb.
        self.expect("continue", substrs=["exited with status = 0"])

    @skipUnlessArch("loongarch64")
    @skipUnlessPlatform(["linux"])
    def test_lasx(self):
        if not self.isLoongArchLASX():
            self.skipTest("LASX must be present.")
        self.lasx_registers_impl()
