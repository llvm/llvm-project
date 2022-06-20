from __future__ import print_function
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


# This test case is testing three things:
#
#  1. three register values will be provided in the ? stop packet (T11) -
#     registers 0 ("rax"), 1 ("rbx"), and 3 ("rip")
#  2. ReadRegister packet will provide the value of register 2 ("rsi")
#  3. The "g" read-all-registers packet is not supported; p must be used
#     to get the value of register 2 ("rsi")
#
# Forcing lldb to use the expedited registers in the stop packet and
# marking it an error to request that register value is to prevent
# performance regressions.
# 
# Some gdb RSP stubs only implement p/P, they do not support g/G.
# lldb must be able to work with either.

class TestNoGPacketSupported(GDBRemoteTestBase):

    @skipIfXmlSupportMissing
    def test(self):
        class MyResponder(MockGDBServerResponder):
            def haltReason(self):
                return "T02thread:1ff0d;threads:1ff0d;thread-pcs:000000010001bc00;00:7882773ce0ffffff;01:1122334455667788;03:00bc010001000000;"

            def threadStopInfo(self, threadnum):
                return "T02thread:1ff0d;threads:1ff0d;thread-pcs:000000010001bc00;00:7882773ce0ffffff;01:1122334455667788;03:00bc010001000000;"

            def writeRegisters(self):
                return "E02"

            def readRegisters(self):
                return "E01"

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
                    return "c04825ebfe7f0000" # 0x00007ffeeb2548c0

                return "E03"

            def writeRegister(self, regnum):
                return "OK"

            def qXferRead(self, obj, annex, offset, length):
                if annex == "target.xml":
                    return """<?xml version="1.0"?>
                        <target version="1.0">
                          <architecture>i386:x86-64</architecture>
                          <feature name="org.gnu.gdb.i386.core">
                            <reg name="rax" bitsize="64" regnum="0" type="code_ptr" group="general"/>
                            <reg name="rbx" bitsize="64" regnum="1" type="code_ptr" group="general"/>
                            <reg name="rsi" bitsize="64" regnum="2" type="code_ptr" group="general"/>
                            <reg name="rip" bitsize="64" regnum="3" type="code_ptr" group="general" altname="pc" generic="pc"/>
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
        rbx = frame.FindRegister("rbx").GetValueAsUnsigned()
        rsi = frame.FindRegister("rsi").GetValueAsUnsigned()
        pc = frame.GetPC()
        rip = frame.FindRegister("rip").GetValueAsUnsigned()

        if self.TraceOn():
            print("Register values: rax == 0x%x, rbx == 0x%x, rsi == 0x%x, pc == 0x%x, rip == 0x%x" % (rax, rbx, rsi, pc, rip))

        self.assertEqual(rax, 0xffffffe03c778278)
        self.assertEqual(rbx, 0x8877665544332211)
        self.assertEqual(rsi, 0x00007ffeeb2548c0)
        self.assertEqual(pc, 0x10001bc00)
        self.assertEqual(rip, 0x10001bc00)
