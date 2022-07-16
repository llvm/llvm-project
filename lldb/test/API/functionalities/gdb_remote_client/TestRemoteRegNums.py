from __future__ import print_function
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


# This test case checks for register number mismatch between lldb and gdb stub.
# LLDB client assigns register numbers to target xml registers in increasing
# order starting with regnum = 0, while gdb-remote may specify different regnum
# which is stored as eRegisterKindProcessPlugin. Remote side will use its
# register number in expedited register list, value_regs and invalidate_regnums.
#
# This test creates a ficticious target xml with non-sequential regnums to test
# that correct registers are accessed in all of above mentioned cases.

class TestRemoteRegNums(GDBRemoteTestBase):

    @skipIfXmlSupportMissing
    def test(self):
        class MyResponder(MockGDBServerResponder):
            def haltReason(self):
                return "T02thread:1ff0d;threads:1ff0d;thread-pcs:000000010001bc00;00:00bc010001000000;09:c04825ebfe7f0000;"

            def threadStopInfo(self, threadnum):
                return "T02thread:1ff0d;threads:1ff0d;thread-pcs:000000010001bc00;00:00bc010001000000;09:c04825ebfe7f0000;"

            def writeRegisters(self):
                return "E02"

            def readRegisters(self):
                return "E01"

            rax_regnum2_val = "7882773ce0ffffff"
            rbx_regnum4_val = "1122334455667788"

            def readRegister(self, regnum):
                # lldb will try sending "p0" to see if the p packet is supported,
                # give a bogus value; in theory lldb could use this value in the
                # register context and that would be valid behavior.

                # notably, don't give values for registers 1 & 3 -- lldb should
                # get those from the ? stop packet ("T11") and it is a pref regression
                # if lldb is asking for these register values.
                if regnum == 0:
                    return "5555555555555555"
                if regnum == 2:
                    return self.rax_regnum2_val
                if regnum == 4:
                    return self.rbx_regnum4_val

                return "E03"

            def writeRegister(self, regnum, value_hex):
                if regnum == 2:
                    self.rax_regnum2_val = value_hex
                if regnum == 4:
                    self.rbx_regnum4_val = value_hex

                return "OK"

            def qXferRead(self, obj, annex, offset, length):
                if annex == "target.xml":
                    return """<?xml version="1.0"?>
                        <target version="1.0">
                          <architecture>i386:x86-64</architecture>
                          <feature name="org.gnu.gdb.i386.core">
                            <reg name="rip" bitsize="64" regnum="0" type="code_ptr" group="general" altname="pc" generic="pc"/>
                            <reg name="rax" bitsize="64" regnum="2" type="code_ptr" group="general"/>
                            <reg name="rbx" bitsize="64" regnum="4" type="code_ptr" group="general"/>
                            <reg name="eax" bitsize="32" regnum="5" value_regnums="2" invalidate_regnums="2" type="code_ptr" group="general"/>
                            <reg name="ebx" bitsize="32" regnum="7" value_regnums="4" invalidate_regnums="4" type="code_ptr" group="general"/>
                            <reg name="rsi" bitsize="64" regnum="9" type="code_ptr" group="general"/>
                          </feature>
                        </target>""", False
                else:
                    return None, False

        self.server.responder = MyResponder()
        target = self.dbg.CreateTarget('')
        if self.TraceOn():
            self.runCmd("log enable gdb-remote packets")
            self.addTearDownHook(
                lambda: self.runCmd("log disable gdb-remote packets"))
        process = self.connect(target)

        thread = process.GetThreadAtIndex(0)
        frame = thread.GetFrameAtIndex(0)
        rax = frame.FindRegister("rax").GetValueAsUnsigned()
        eax = frame.FindRegister("eax").GetValueAsUnsigned()
        rbx = frame.FindRegister("rbx").GetValueAsUnsigned()
        ebx = frame.FindRegister("ebx").GetValueAsUnsigned()
        rsi = frame.FindRegister("rsi").GetValueAsUnsigned()
        pc = frame.GetPC()
        rip = frame.FindRegister("rip").GetValueAsUnsigned()

        if self.TraceOn():
            print("Register values: rax == 0x%x, rbx == 0x%x, rsi == 0x%x, pc == 0x%x, rip == 0x%x" % (
                rax, rbx, rsi, pc, rip))

        self.assertEqual(rax, 0xffffffe03c778278)
        self.assertEqual(rbx, 0x8877665544332211)
        self.assertEqual(eax, 0x3c778278)
        self.assertEqual(ebx, 0x44332211)
        self.assertEqual(rsi, 0x00007ffeeb2548c0)
        self.assertEqual(pc, 0x10001bc00)
        self.assertEqual(rip, 0x10001bc00)

        frame.FindRegister("eax").SetValueFromCString("1")
        frame.FindRegister("ebx").SetValueFromCString("0")
        eax = frame.FindRegister("eax").GetValueAsUnsigned()
        ebx = frame.FindRegister("ebx").GetValueAsUnsigned()
        rax = frame.FindRegister("rax").GetValueAsUnsigned()
        rbx = frame.FindRegister("rbx").GetValueAsUnsigned()

        if self.TraceOn():
            print("Register values: rax == 0x%x, rbx == 0x%x, rsi == 0x%x, pc == 0x%x, rip == 0x%x" % (
                rax, rbx, rsi, pc, rip))

        self.assertEqual(rax, 0xffffffe000000001)
        self.assertEqual(rbx, 0x8877665500000000)
        self.assertEqual(eax, 0x00000001)
        self.assertEqual(ebx, 0x00000000)
        self.assertEqual(rsi, 0x00007ffeeb2548c0)
        self.assertEqual(pc, 0x10001bc00)
        self.assertEqual(rip, 0x10001bc00)
