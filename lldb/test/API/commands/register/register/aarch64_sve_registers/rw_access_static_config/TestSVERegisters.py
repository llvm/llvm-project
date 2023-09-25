"""
Test the AArch64 SVE registers.
"""

from enum import Enum
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class Mode(Enum):
    SVE = 0
    SSVE = 1


class RegisterCommandsTestCase(TestBase):
    def check_sve_register_size(self, set, name, expected):
        reg_value = set.GetChildMemberWithName(name)
        self.assertTrue(
            reg_value.IsValid(), 'Verify we have a register named "%s"' % (name)
        )
        self.assertEqual(
            reg_value.GetByteSize(), expected, 'Verify "%s" == %i' % (name, expected)
        )

    def check_sve_regs_read(self, z_reg_size, expected_mode):
        if self.isAArch64SME():
            # This test uses SMSTART SM, which only enables streaming mode,
            # leaving ZA disabled.
            expected_value = "1" if expected_mode == Mode.SSVE else "0"
            self.expect(
                "register read svcr", substrs=["0x000000000000000" + expected_value]
            )

        p_reg_size = int(z_reg_size / 8)

        for i in range(32):
            z_regs_value = (
                "{"
                + " ".join("0x{:02x}".format(i + 1) for _ in range(z_reg_size))
                + "}"
            )
            self.expect("register read " + "z%i" % (i), substrs=[z_regs_value])

        p_value_bytes = ["0xff", "0x55", "0x11", "0x01", "0x00"]
        for i in range(16):
            p_regs_value = (
                "{" + " ".join(p_value_bytes[i % 5] for _ in range(p_reg_size)) + "}"
            )
            self.expect("register read " + "p%i" % (i), substrs=[p_regs_value])

        self.expect("register read ffr", substrs=[p_regs_value])

    def check_sve_regs_read_after_write(self, z_reg_size):
        p_reg_size = int(z_reg_size / 8)

        z_regs_value = "{" + " ".join(("0x9d" for _ in range(z_reg_size))) + "}"

        p_regs_value = "{" + " ".join(("0xee" for _ in range(p_reg_size))) + "}"

        for i in range(32):
            self.runCmd("register write " + "z%i" % (i) + " '" + z_regs_value + "'")

        for i in range(32):
            self.expect("register read " + "z%i" % (i), substrs=[z_regs_value])

        for i in range(16):
            self.runCmd("register write " + "p%i" % (i) + " '" + p_regs_value + "'")

        for i in range(16):
            self.expect("register read " + "p%i" % (i), substrs=[p_regs_value])

        self.runCmd("register write " + "ffr " + "'" + p_regs_value + "'")

        self.expect("register read " + "ffr", substrs=[p_regs_value])

    def get_build_flags(self, mode):
        cflags = "-march=armv8-a+sve"
        if mode == Mode.SSVE:
            cflags += " -DSTART_SSVE"
        return {"CFLAGS_EXTRAS": cflags}

    def skip_if_needed(self, mode):
        if (mode == Mode.SVE) and not self.isAArch64SVE():
            self.skipTest("SVE registers must be supported.")

        if (mode == Mode.SSVE) and not self.isAArch64SME():
            self.skipTest("SSVE registers must be supported.")

    def sve_registers_configuration_impl(self, mode):
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

        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()
        thread = process.GetThreadAtIndex(0)
        currentFrame = thread.GetFrameAtIndex(0)

        registerSets = process.GetThreadAtIndex(0).GetFrameAtIndex(0).GetRegisters()
        sve_registers = registerSets.GetFirstValueByName(
            "Scalable Vector Extension Registers"
        )
        self.assertTrue(sve_registers)

        vg_reg_value = sve_registers.GetChildMemberWithName("vg").GetValueAsUnsigned()

        z_reg_size = vg_reg_value * 8
        for i in range(32):
            self.check_sve_register_size(sve_registers, "z%i" % (i), z_reg_size)

        p_reg_size = z_reg_size / 8
        for i in range(16):
            self.check_sve_register_size(sve_registers, "p%i" % (i), p_reg_size)

        self.check_sve_register_size(sve_registers, "ffr", p_reg_size)

    @no_debug_info_test
    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(["linux"]))
    def test_sve_registers_configuration(self):
        """Test AArch64 SVE registers size configuration."""
        self.sve_registers_configuration_impl(Mode.SVE)

    @no_debug_info_test
    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(["linux"]))
    def test_ssve_registers_configuration(self):
        """Test AArch64 SSVE registers size configuration."""
        self.sve_registers_configuration_impl(Mode.SSVE)

    def sve_registers_read_write_impl(self, start_mode, eval_mode):
        self.skip_if_needed(start_mode)
        self.skip_if_needed(eval_mode)
        self.build(dictionary=self.get_build_flags(start_mode))

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        self.line = line_number("main.c", "// Set a break point here.")
        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", self.line, num_expected_locations=1
        )
        self.runCmd("run", RUN_SUCCEEDED)

        self.expect(
            "thread backtrace",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stop reason = breakpoint 1."],
        )

        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()

        registerSets = process.GetThreadAtIndex(0).GetFrameAtIndex(0).GetRegisters()
        sve_registers = registerSets.GetFirstValueByName(
            "Scalable Vector Extension Registers"
        )
        self.assertTrue(sve_registers)

        vg_reg_value = sve_registers.GetChildMemberWithName("vg").GetValueAsUnsigned()
        z_reg_size = vg_reg_value * 8
        self.check_sve_regs_read(z_reg_size, start_mode)

        # Evaluate simple expression and print function expr_eval_func address.
        self.expect("expression expr_eval_func", substrs=["= 0x"])

        # Evaluate expression call function expr_eval_func.
        self.expect_expr(
            "expr_eval_func({})".format(
                "true" if (eval_mode == Mode.SSVE) else "false"
            ),
            result_type="int",
            result_value="1",
        )

        # We called a jitted function above which must not have changed SVE
        # vector length or register values.
        self.check_sve_regs_read(z_reg_size, start_mode)

        self.check_sve_regs_read_after_write(z_reg_size)

    # The following tests all setup some register values then evaluate an
    # expression. After the expression, the mode and register values should be
    # the same as before. Finally they read/write some values in the registers.
    # The only difference is the mode we start the program in, and the mode
    # the expression function uses.

    @no_debug_info_test
    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(["linux"]))
    def test_registers_expr_read_write_sve_sve(self):
        self.sve_registers_read_write_impl(Mode.SVE, Mode.SVE)

    @no_debug_info_test
    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(["linux"]))
    def test_registers_expr_read_write_ssve_ssve(self):
        self.sve_registers_read_write_impl(Mode.SSVE, Mode.SSVE)

    @no_debug_info_test
    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(["linux"]))
    def test_registers_expr_read_write_sve_ssve(self):
        self.sve_registers_read_write_impl(Mode.SVE, Mode.SSVE)

    @no_debug_info_test
    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(["linux"]))
    def test_registers_expr_read_write_ssve_sve(self):
        self.sve_registers_read_write_impl(Mode.SSVE, Mode.SVE)
