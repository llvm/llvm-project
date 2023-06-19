"""
Test lldb's ability to read and write the AArch64 TLS register tpidr.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class AArch64LinuxTLSRegister(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessArch("aarch64")
    @skipUnlessPlatform(["linux"])
    def test_tls(self):
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

        self.runCmd("run", RUN_SUCCEEDED)

        if self.process().GetState() == lldb.eStateExited:
            self.fail("Test program failed to run.")

        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        # Since we can't predict what the value will be, the program has set
        # a target value for us to find.

        regs = self.thread().GetSelectedFrame().GetRegisters()
        tls_regs = regs.GetFirstValueByName("Thread Local Storage Registers")
        self.assertTrue(tls_regs.IsValid(), "No TLS registers found.")
        tpidr = tls_regs.GetChildMemberWithName("tpidr")
        self.assertTrue(tpidr.IsValid(), "No tpidr register found.")

        self.assertEqual(tpidr.GetValueAsUnsigned(), 0x1122334455667788)

        # Set our own value for the program to find.
        self.expect("register write tpidr 0x{:x}".format(0x8877665544332211))
        self.expect("continue")

        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        self.expect("p tpidr_was_set", substrs=["true"])