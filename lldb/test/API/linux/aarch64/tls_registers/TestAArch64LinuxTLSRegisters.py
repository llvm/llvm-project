"""
Test lldb's ability to read and write the AArch64 TLS registers.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class AArch64LinuxTLSRegisters(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setup(self, registers):
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self,
            "main.c",
            line_number("main.c", "// Set break point at this line."),
            num_expected_locations=1,
        )

        lldbutil.run_break_set_by_file_and_line(
            self,
            "main.c",
            line_number("main.c", "// Set break point 2 at this line."),
            num_expected_locations=1,
        )

        if "tpidr2" in registers:
            self.runCmd("settings set target.run-args 1")
        self.runCmd("run", RUN_SUCCEEDED)

        if self.process().GetState() == lldb.eStateExited:
            self.fail("Test program failed to run.")

        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

    def check_registers(self, registers, values):
        regs = self.thread().GetSelectedFrame().GetRegisters()
        tls_regs = regs.GetFirstValueByName("Thread Local Storage Registers")
        self.assertTrue(tls_regs.IsValid(), "No TLS registers found.")

        for register in registers:
            tls_reg = tls_regs.GetChildMemberWithName(register)
            self.assertTrue(
                tls_reg.IsValid(), "{} register not found.".format(register)
            )
            self.assertEqual(tls_reg.GetValueAsUnsigned(), values[register])

    def check_tls_reg(self, registers):
        self.setup(registers)

        # Since we can't predict what the value will be, the program has set
        # a target value for us to find.
        initial_values = {
            "tpidr": 0x1122334455667788,
            "tpidr2": 0x8877665544332211,
        }

        self.check_registers(registers, initial_values)

        # Their values should be restored if an expression modifies them.
        self.runCmd("expression expr_func()")

        self.check_registers(registers, initial_values)

        set_values = {
            "tpidr": 0x1111222233334444,
            "tpidr2": 0x4444333322221111,
        }

        # Set our own value(s) for the program to find.
        for register in registers:
            self.expect(
                "register write {} 0x{:x}".format(register, set_values[register])
            )

        self.expect("continue")

        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        for register in registers:
            self.expect("p {}_was_set".format(register), substrs=["true"])

    @skipUnlessArch("aarch64")
    @skipUnlessPlatform(["linux"])
    def test_tls_no_sme(self):
        if self.isAArch64SME():
            self.skipTest("SME must not be present.")

        self.check_tls_reg(["tpidr"])

    @skipUnlessArch("aarch64")
    @skipUnlessPlatform(["linux"])
    def test_tls_sme(self):
        if not self.isAArch64SME():
            self.skipTest("SME must be present.")

        self.check_tls_reg(["tpidr", "tpidr2"])

    @skipUnlessArch("aarch64")
    @skipUnlessPlatform(["linux"])
    def test_tpidr2_no_sme(self):
        if self.isAArch64SME():
            self.skipTest("SME must not be present.")

        self.setup("tpidr")

        regs = self.thread().GetSelectedFrame().GetRegisters()
        tls_regs = regs.GetFirstValueByName("Thread Local Storage Registers")
        self.assertTrue(tls_regs.IsValid(), "No TLS registers found.")
        tls_reg = tls_regs.GetChildMemberWithName("tpidr2")
        self.assertFalse(tls_reg.IsValid(), "tpdir2 should not be present without SME")
