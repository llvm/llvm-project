"""
Test lldb's ability to read and write the LoongArch SIMD registers.
"""

from enum import Enum
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class Mode(Enum):
    LSX = 0
    LASX = 1


class LoongArch64LinuxRegisters(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def get_build_flags(self, mode):
        cflags = "-march=la464"
        if mode == Mode.LASX:
            cflags += " -DLASX"

        return {"CFLAGS_EXTRAS": cflags}

    def make_simd_value(self, n, mode):
        count = 32 if mode == Mode.LASX else 16
        return "{" + " ".join(["0x{:02x}".format(n)] * count) + "}"

    def check_simd_values(self, value_offset, mode):
        reg_prefix = "xr" if mode == Mode.LASX else "vr"
        for i in range(32):
            self.expect(
                "register read {}{}".format(reg_prefix, i),
                substrs=[self.make_simd_value(i + value_offset, mode)],
            )

    def simd_registers_impl(self, mode):
        self.build(dictionary=self.get_build_flags(mode))
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

        self.check_simd_values(0, mode)
        self.runCmd("expression write_simd_regs(1)")
        self.check_simd_values(0, mode)

        reg_prefix = "xr" if mode == Mode.LASX else "vr"
        for i in range(32):
            self.runCmd(
                'register write {}{} "{}"'.format(
                    reg_prefix, i, self.make_simd_value(i + 1, mode)
                )
            )

        # Should be visible within lldb.
        self.check_simd_values(1, mode)

        # The program should agree with lldb.
        self.expect("continue", substrs=["exited with status = 0"])

    @skipUnlessArch("loongarch64")
    @skipUnlessPlatform(["linux"])
    def test_lsx(self):
        """Test read/write of LSX registers."""
        if not self.isLoongArchLSX():
            self.skipTest("LSX must be present.")
        self.simd_registers_impl(Mode.LSX)

    @skipUnlessArch("loongarch64")
    @skipUnlessPlatform(["linux"])
    def test_lasx(self):
        """Test read/write of LASX registers."""
        if not self.isLoongArchLASX():
            self.skipTest("LASX must be present.")
        self.simd_registers_impl(Mode.LASX)
