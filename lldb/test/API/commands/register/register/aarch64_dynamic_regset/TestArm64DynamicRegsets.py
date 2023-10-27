"""
Test AArch64 dynamic register sets
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class RegisterCommandsTestCase(TestBase):
    def check_sve_register_size(self, set, name, expected):
        reg_value = set.GetChildMemberWithName(name)
        self.assertTrue(reg_value.IsValid(), "Expected a register named %s" % (name))
        self.assertEqual(
            reg_value.GetByteSize(),
            expected,
            "Expected a register %s size == %i bytes" % (name, expected),
        )

    def sve_regs_read_dynamic(self, sve_registers):
        vg_reg = sve_registers.GetChildMemberWithName("vg")
        vg_reg_value = sve_registers.GetChildMemberWithName("vg").GetValueAsUnsigned()

        z_reg_size = vg_reg_value * 8
        p_reg_size = int(z_reg_size / 8)

        for i in range(32):
            z_regs_value = (
                "{"
                + " ".join("0x{:02x}".format(i + 1) for _ in range(z_reg_size))
                + "}"
            )
            self.expect("register read z%i" % (i), substrs=[z_regs_value])

        # Set P registers with random test values. The P registers are predicate
        # registers, which hold one bit for each byte available in a Z register.
        # For below mentioned values of P registers, P(0,5,10,15) will have all
        # Z register lanes set while P(4,9,14) will have no lanes set.
        p_value_bytes = ["0xff", "0x55", "0x11", "0x01", "0x00"]
        for i in range(16):
            p_regs_value = (
                "{" + " ".join(p_value_bytes[i % 5] for _ in range(p_reg_size)) + "}"
            )
            self.expect("register read p%i" % (i), substrs=[p_regs_value])

        self.expect("register read ffr", substrs=[p_regs_value])

        for i in range(32):
            z_regs_value = (
                "{"
                + " ".join("0x{:02x}".format(32 - i) for _ in range(z_reg_size))
                + "}"
            )
            self.runCmd("register write z%i '%s'" % (i, z_regs_value))
            self.expect("register read z%i" % (i), substrs=[z_regs_value])

        for i in range(16):
            p_regs_value = (
                "{"
                + " ".join("0x{:02x}".format(16 - i) for _ in range(p_reg_size))
                + "}"
            )
            self.runCmd("register write p%i '%s'" % (i, p_regs_value))
            self.expect("register read p%i" % (i), substrs=[p_regs_value])

        p_regs_value = (
            "{" + " ".join("0x{:02x}".format(8) for _ in range(p_reg_size)) + "}"
        )
        self.runCmd("register write ffr " + "'" + p_regs_value + "'")
        self.expect("register read ffr", substrs=[p_regs_value])

    def setup_register_config_test(self, run_args=None):
        self.build()
        self.line = line_number("main.c", "// Set a break point here.")

        exe = self.getBuildArtifact("a.out")
        if run_args is not None:
            self.runCmd("settings set target.run-args " + run_args)
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

        return self.thread().GetSelectedFrame().GetRegisters()

    @no_debug_info_test
    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(["linux"]))
    def test_aarch64_dynamic_regset_config(self):
        """Test AArch64 Dynamic Register sets configuration."""
        register_sets = self.setup_register_config_test()

        for registerSet in register_sets:
            if "Scalable Vector Extension Registers" in registerSet.GetName():
                self.assertTrue(
                    self.isAArch64SVE(),
                    "LLDB enabled AArch64 SVE register set when it was disabled by target.",
                )
                self.sve_regs_read_dynamic(registerSet)
            if "MTE Control Register" in registerSet.GetName():
                self.assertTrue(
                    self.isAArch64MTE(),
                    "LLDB enabled AArch64 MTE register set when it was disabled by target.",
                )
                self.runCmd("register write mte_ctrl 0x7fff9")
                self.expect(
                    "register read mte_ctrl", substrs=["mte_ctrl = 0x000000000007fff9"]
                )
            if "Pointer Authentication Registers" in registerSet.GetName():
                self.assertTrue(
                    self.isAArch64PAuth(),
                    "LLDB enabled AArch64 Pointer Authentication register set when it was disabled by target.",
                )
                self.expect("register read data_mask", substrs=["data_mask = 0x"])
                self.expect("register read code_mask", substrs=["code_mask = 0x"])
            if "Scalable Matrix Extension Registers" in registerSet.GetName():
                self.assertTrue(
                    self.isAArch64SME(),
                    "LLDB Enabled SME register set when it was disabled by target",
                )

    def make_za_value(self, vl, generator):
        # Generate a vector value string "{0x00 0x01....}".
        rows = []
        for row in range(vl):
            byte = "0x{:02x}".format(generator(row))
            rows.append(" ".join([byte] * vl))
        return "{" + " ".join(rows) + "}"

    @no_debug_info_test
    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(["linux"]))
    def test_aarch64_dynamic_regset_config_sme(self):
        """Test AArch64 Dynamic Register sets configuration, but only SME
        registers."""
        if not self.isAArch64SME():
            self.skipTest("SME must be present.")

        register_sets = self.setup_register_config_test("sme")

        ssve_registers = register_sets.GetFirstValueByName(
            "Scalable Vector Extension Registers"
        )
        self.assertTrue(ssve_registers.IsValid())
        self.sve_regs_read_dynamic(ssve_registers)

        sme_registers = register_sets.GetFirstValueByName(
            "Scalable Matrix Extension Registers"
        )
        self.assertTrue(sme_registers.IsValid())

        vg = ssve_registers.GetChildMemberWithName("vg").GetValueAsUnsigned()
        vl = vg * 8
        # When first enabled it is all 0s.
        self.expect("register read za", substrs=[self.make_za_value(vl, lambda r: 0)])
        za_value = self.make_za_value(vl, lambda r: r + 1)
        self.runCmd("register write za '{}'".format(za_value))
        self.expect("register read za", substrs=[za_value])

        # SVG should match VG because we're in streaming mode.

        self.assertTrue(sme_registers.IsValid())
        svg = sme_registers.GetChildMemberWithName("svg").GetValueAsUnsigned()
        self.assertEqual(vg, svg)

        # SVCR should be SVCR.SM | SVCR.ZA aka 3 because streaming mode is on
        # and ZA is enabled.
        svcr = sme_registers.GetChildMemberWithName("svcr").GetValueAsUnsigned()
        self.assertEqual(3, svcr)

        # SVCR is read only so we do not test writing to it.

    @no_debug_info_test
    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(["linux"]))
    def test_aarch64_dynamic_regset_config_sme_za_disabled(self):
        """Test that ZA shows as 0s when disabled and can be enabled by writing
        to it."""
        if not self.isAArch64SME():
            self.skipTest("SME must be present.")

        # No argument, so ZA will be disabled when we break.
        register_sets = self.setup_register_config_test()

        # vg is the non-streaming vg as we are in non-streaming mode, so we need
        # to use svg.
        sme_registers = register_sets.GetFirstValueByName(
            "Scalable Matrix Extension Registers"
        )
        self.assertTrue(sme_registers.IsValid())
        svg = sme_registers.GetChildMemberWithName("svg").GetValueAsUnsigned()

        # We are not in streaming mode, ZA is disabled, so this should be 0.
        svcr = sme_registers.GetChildMemberWithName("svcr").GetValueAsUnsigned()
        self.assertEqual(0, svcr)

        svl = svg * 8
        # A disabled ZA is shown as all 0s.
        self.expect("register read za", substrs=[self.make_za_value(svl, lambda r: 0)])
        za_value = self.make_za_value(svl, lambda r: r + 1)
        # Writing to it enables ZA, so the value should be there when we read
        # it back.
        self.runCmd("register write za '{}'".format(za_value))
        self.expect("register read za", substrs=[za_value])

        # Now SVCR.ZA should be set, which is bit 1.
        self.expect("register read svcr", substrs=["0x0000000000000002"])

        # SVCR is read only so we do not test writing to it.
