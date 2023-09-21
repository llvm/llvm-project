"""
Test the AArch64 SME ZA register is saved and restored around expressions.

This attempts to cover expressions that change the following:
* ZA enabled or not.
* Streaming mode or not.
* Streaming vector length (increasing and decreasing).
* Some combintations of the above.
"""

from enum import IntEnum
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


# These enum values match the flag values used in the test program.
class Mode(IntEnum):
    SVE = 0
    SSVE = 1


class ZA(IntEnum):
    Disabled = 0
    Enabled = 1


class AArch64ZATestCase(TestBase):
    def get_supported_svg(self):
        # Always build this probe program to start as streaming SVE.
        # We will read/write "vg" here but since we are in streaming mode "svg"
        # is really what we are writing ("svg" is a read only pseudo).
        self.build()

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)
        # Enter streaming mode, don't enable ZA, start_vl and other_vl don't
        # matter here.
        self.runCmd("settings set target.run-args 1 0 0 0")

        stop_line = line_number("main.c", "// Set a break point here.")
        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", stop_line, num_expected_locations=1
        )

        self.runCmd("run", RUN_SUCCEEDED)

        self.expect(
            "thread info 1",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stop reason = breakpoint"],
        )

        # Write back the current vg to confirm read/write works at all.
        current_svg = self.match("register read vg", ["(0x[0-9]+)"])
        self.assertTrue(current_svg is not None)
        self.expect("register write vg {}".format(current_svg.group()))

        # Aka 128, 256 and 512 bit.
        supported_svg = []
        for svg in [2, 4, 8]:
            # This could mask other errors but writing vg is tested elsewhere
            # so we assume the hardware rejected the value.
            self.runCmd("register write vg {}".format(svg), check=False)
            if not self.res.GetError():
                supported_svg.append(svg)

        self.runCmd("breakpoint delete 1")
        self.runCmd("continue")

        return supported_svg

    def read_vg(self):
        process = self.dbg.GetSelectedTarget().GetProcess()
        registerSets = process.GetThreadAtIndex(0).GetFrameAtIndex(0).GetRegisters()
        sve_registers = registerSets.GetFirstValueByName(
            "Scalable Vector Extension Registers"
        )
        return sve_registers.GetChildMemberWithName("vg").GetValueAsUnsigned()

    def read_svg(self):
        process = self.dbg.GetSelectedTarget().GetProcess()
        registerSets = process.GetThreadAtIndex(0).GetFrameAtIndex(0).GetRegisters()
        sve_registers = registerSets.GetFirstValueByName(
            "Scalable Matrix Extension Registers"
        )
        return sve_registers.GetChildMemberWithName("svg").GetValueAsUnsigned()

    def make_za_value(self, vl, generator):
        # Generate a vector value string "{0x00 0x01....}".
        rows = []
        for row in range(vl):
            byte = "0x{:02x}".format(generator(row))
            rows.append(" ".join([byte] * vl))
        return "{" + " ".join(rows) + "}"

    def check_za(self, vl):
        # We expect an increasing value starting at 1. Row 0=1, row 1 = 2, etc.
        self.expect(
            "register read za", substrs=[self.make_za_value(vl, lambda row: row + 1)]
        )

    def check_za_disabled(self, vl):
        # When ZA is disabled, lldb will show ZA as all 0s.
        self.expect("register read za", substrs=[self.make_za_value(vl, lambda row: 0)])

    def za_expr_test_impl(self, sve_mode, za_state, swap_start_vl):
        if not self.isAArch64SME():
            self.skipTest("SME must be present.")

        supported_svg = self.get_supported_svg()
        if len(supported_svg) < 2:
            self.skipTest("Target must support at least 2 streaming vector lengths.")

        # vg is in units of 8 bytes.
        start_vl = supported_svg[1] * 8
        other_vl = supported_svg[2] * 8

        if swap_start_vl:
            start_vl, other_vl = other_vl, start_vl

        self.line = line_number("main.c", "// Set a break point here.")

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)
        self.runCmd(
            "settings set target.run-args {} {} {} {}".format(
                sve_mode, za_state, start_vl, other_vl
            )
        )

        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", self.line, num_expected_locations=1
        )
        self.runCmd("run", RUN_SUCCEEDED)

        self.expect(
            "thread backtrace",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stop reason = breakpoint 1."],
        )

        exprs = [
            "expr_disable_za",
            "expr_enable_za",
            "expr_start_vl",
            "expr_other_vl",
            "expr_enable_sm",
            "expr_disable_sm",
        ]

        # This may be the streaming or non-streaming vg. All that matters is
        # that it is saved and restored, remaining constant throughout.
        start_vg = self.read_vg()

        # Check SVE registers to make sure that combination of scaling SVE
        # and scaling ZA works properly. This is a brittle check, but failures
        # are likely to be catastrophic when they do happen anyway.
        sve_reg_names = "ffr {} {}".format(
            " ".join(["z{}".format(n) for n in range(32)]),
            " ".join(["p{}".format(n) for n in range(16)]),
        )
        self.runCmd("register read " + sve_reg_names)
        sve_values = self.res.GetOutput()

        def check_regs():
            if za_state == ZA.Enabled:
                self.check_za(start_vl)
            else:
                self.check_za_disabled(start_vl)

            # svg and vg are in units of 8 bytes.
            self.assertEqual(start_vl, self.read_svg() * 8)
            self.assertEqual(start_vg, self.read_vg())

            self.expect("register read " + sve_reg_names, substrs=[sve_values])

        for expr in exprs:
            expr_cmd = "expression {}()".format(expr)

            # We do this twice because there were issues in development where
            # using data stored by a previous WriteAllRegisterValues would crash
            # the second time around.
            self.runCmd(expr_cmd)
            check_regs()
            self.runCmd(expr_cmd)
            check_regs()

        # Run them in sequence to make sure there is no state lingering between
        # them after a restore.
        for expr in exprs:
            self.runCmd("expression {}()".format(expr))
            check_regs()

        for expr in reversed(exprs):
            self.runCmd("expression {}()".format(expr))
            check_regs()

    # These tests start with the 1st supported SVL and change to the 2nd
    # supported SVL as needed.

    @no_debug_info_test
    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(["linux"]))
    def test_za_expr_ssve_za_enabled(self):
        self.za_expr_test_impl(Mode.SSVE, ZA.Enabled, False)

    @no_debug_info_test
    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(["linux"]))
    def test_za_expr_ssve_za_disabled(self):
        self.za_expr_test_impl(Mode.SSVE, ZA.Disabled, False)

    @no_debug_info_test
    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(["linux"]))
    def test_za_expr_sve_za_enabled(self):
        self.za_expr_test_impl(Mode.SVE, ZA.Enabled, False)

    @no_debug_info_test
    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(["linux"]))
    def test_za_expr_sve_za_disabled(self):
        self.za_expr_test_impl(Mode.SVE, ZA.Disabled, False)

    # These tests start in the 2nd supported SVL and change to the 1st supported
    # SVL as needed.

    @no_debug_info_test
    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(["linux"]))
    def test_za_expr_ssve_za_enabled_different_vl(self):
        self.za_expr_test_impl(Mode.SSVE, ZA.Enabled, True)

    @no_debug_info_test
    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(["linux"]))
    def test_za_expr_ssve_za_disabled_different_vl(self):
        self.za_expr_test_impl(Mode.SSVE, ZA.Disabled, True)

    @no_debug_info_test
    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(["linux"]))
    def test_za_expr_sve_za_enabled_different_vl(self):
        self.za_expr_test_impl(Mode.SVE, ZA.Enabled, True)

    @no_debug_info_test
    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(["linux"]))
    def test_za_expr_sve_za_disabled_different_vl(self):
        self.za_expr_test_impl(Mode.SVE, ZA.Disabled, True)
