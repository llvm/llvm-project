"""
Test that LLDB correctly reads and writes and restores AArch64 SIMD registers
in SVE, streaming SVE and normal SIMD modes.

There are a few operating modes and we use different strategies for each:
* Without SVE, in SIMD mode - read the SIMD regset.
* With SVE, but SVE is inactive - read the SVE regset, but get SIMD data from it.
* With SVE, SVE is active - read the SVE regset, use the bottom 128 bits of the
  Z registers.
* With streaming SVE active - read the SSVE regset, use the bottom 128 bits of
  the Z registers.

This text excercise most of those.
"""

from enum import Enum
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class Mode(Enum):
    SIMD = 0
    SVE = 1
    SSVE = 2


class SVESIMDRegistersTestCase(TestBase):
    def get_build_flags(self, mode):
        cflags = "-march=armv8-a+sve"
        if mode == Mode.SSVE:
            cflags += " -DSSVE"
        elif mode == Mode.SVE:
            cflags += " -DSVE"
        # else we want SIMD mode, which processes start up in already.

        return {"CFLAGS_EXTRAS": cflags}

    def skip_if_needed(self, mode):
        if (mode == Mode.SVE) and not self.isAArch64SVE():
            self.skipTest("SVE registers must be supported.")

        if (mode == Mode.SSVE) and not self.isAArch64SME():
            self.skipTest("SSVE registers must be supported.")

    def make_simd_value(self, n):
        pad = " ".join(["0x00"] * 7)
        return "{{0x{:02x} {} 0x{:02x} {}}}".format(n, pad, n, pad)

    def check_simd_values(self, value_offset):
        # These are 128 bit registers, so getting them from the API as unsigned
        # values doesn't work. Check the command output instead.
        for i in range(32):
            self.expect(
                "register read v{}".format(i),
                substrs=[self.make_simd_value(i + value_offset)],
            )

    def sve_simd_registers_impl(self, mode):
        self.skip_if_needed(mode)

        self.build(dictionary=self.get_build_flags(mode))
        self.line = line_number("main.c", "// Set a break point here.")

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", self.line, num_expected_locations=1
        )
        self.runCmd("run", RUN_SUCCEEDED)

        self.expect(
            "thread backtrace",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stop reason = breakpoint 1."],
        )

        self.check_simd_values(0)
        self.runCmd("expression write_simd_regs(1)")
        self.check_simd_values(0)

        # Write a new set of values. The kernel will move the program back to
        # non-streaming mode here.
        for i in range(32):
            self.runCmd(
                'register write v{} "{}"'.format(i, self.make_simd_value(i + 1))
            )

        # Should be visible within lldb.
        self.check_simd_values(1)

        # The program should agree with lldb.
        self.expect("continue", substrs=["exited with status = 0"])

    @no_debug_info_test
    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(["linux"]))
    def test_simd_registers_sve(self):
        """Test read/write of SIMD registers when in SVE mode."""
        self.sve_simd_registers_impl(Mode.SVE)

    @no_debug_info_test
    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(["linux"]))
    def test_simd_registers_ssve(self):
        """Test read/write of SIMD registers when in SSVE mode."""
        self.sve_simd_registers_impl(Mode.SSVE)

    @no_debug_info_test
    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(["linux"]))
    def test_simd_registers_simd(self):
        """Test read/write of SIMD registers when in SIMD mode."""
        self.sve_simd_registers_impl(Mode.SIMD)
