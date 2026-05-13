"""
Test lldb's ability to read the Arm TLS register.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@skipUnlessArch("arm")
@skipUnlessPlatform(["linux"])
class ArmLinuxTLSRegister(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_tls_reg_access(self):
        self.build()
        (_, _, thread, _) = lldbutil.run_to_source_breakpoint(
            self, "// Set breakpoint here.", lldb.SBFileSpec("main.c")
        )
        frame = thread.GetFrameAtIndex(0)

        tpidruro_var = frame.FindVariable("tpidruro")
        self.assertTrue(tpidruro_var.IsValid())

        regs = frame.GetRegisters()
        tls_regs = regs.GetFirstValueByName("Thread Local Storage Registers")
        self.assertTrue(tls_regs.IsValid(), "No TLS registers found.")
        tpidruro_reg = tls_regs.GetChildMemberWithName("tpidruro")
        self.assertTrue(tpidruro_reg.IsValid(), "tpidruro register not found.")

        val = tpidruro_var.GetValueAsUnsigned()
        self.assertEqual(tpidruro_reg.GetValueAsUnsigned(), val)
        self.expect("reg read tp", substrs=[hex(val)])

        self.expect(
            "register write tpidruro 0x1234",
            error=True,
            substrs=["Failed to write register 'tpidruro'"],
        )
