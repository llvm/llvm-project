import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


class TestPartialGPacket(GDBRemoteTestBase):

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test(self):
        """
        Test GDB remote fallback to 'p' packet when 'g' packet does not include all registers.
        """
        class MyResponder(MockGDBServerResponder):

            def qXferRead(self, obj, annex, offset, length):
                if annex == "target.xml":
                    return """<?xml version="1.0"?>
                        <!DOCTYPE feature SYSTEM "gdb-target.dtd">
                        <target>
                        <architecture>arm</architecture>
                        <feature name="org.gnu.gdb.arm.m-profile">
                        <reg name="r0" bitsize="32" type="uint32" group="general"/>
                        <reg name="r1" bitsize="32" type="uint32" group="general"/>
                        <reg name="r2" bitsize="32" type="uint32" group="general"/>
                        <reg name="r3" bitsize="32" type="uint32" group="general"/>
                        <reg name="r4" bitsize="32" type="uint32" group="general"/>
                        <reg name="r5" bitsize="32" type="uint32" group="general"/>
                        <reg name="r6" bitsize="32" type="uint32" group="general"/>
                        <reg name="r7" bitsize="32" type="uint32" group="general"/>
                        <reg name="r8" bitsize="32" type="uint32" group="general"/>
                        <reg name="r9" bitsize="32" type="uint32" group="general"/>
                        <reg name="r10" bitsize="32" type="uint32" group="general"/>
                        <reg name="r11" bitsize="32" type="uint32" group="general"/>
                        <reg name="r12" bitsize="32" type="uint32" group="general"/>
                        <reg name="sp" bitsize="32" type="data_ptr" group="general"/>
                        <reg name="lr" bitsize="32" type="uint32" group="general"/>
                        <reg name="pc" bitsize="32" type="code_ptr" group="general"/>
                        <reg name="xpsr" bitsize="32" regnum="25" type="uint32" group="general"/>
                        <reg name="MSP" bitsize="32" regnum="26" type="uint32" group="general"/>
                        <reg name="PSP" bitsize="32" regnum="27" type="uint32" group="general"/>
                        <reg name="PRIMASK" bitsize="32" regnum="28" type="uint32" group="general"/>
                        <reg name="BASEPRI" bitsize="32" regnum="29" type="uint32" group="general"/>
                        <reg name="FAULTMASK" bitsize="32" regnum="30" type="uint32" group="general"/>
                        <reg name="CONTROL" bitsize="32" regnum="31" type="uint32" group="general"/>
                        </feature>
                        </target>""", False
                else:
                    return None, False

            def readRegister(self, regnum):
                if regnum == 31:
                    return "cdcc8c3f00000000"
                return "E01"

            def readRegisters(self):
                return "20000000f8360020001000002fcb0008f8360020a0360020200c0020000000000000000000000000000000000000000000000000b87f0120b7d100082ed2000800000001"

            def haltReason(self):
                return "S05"

            def qfThreadInfo(self):
                return "mdead"

            def qC(self):
                return ""

            def qSupported(self, client_supported):
                return "PacketSize=4000;qXfer:memory-map:read-;QStartNoAckMode+;qXfer:threads:read+;hwbreak+;qXfer:features:read+"

            def QThreadSuffixSupported(self):
                return "OK"

            def QListThreadsInStopReply(self):
                return "OK"

        self.server.responder = MyResponder()
        if self.TraceOn():
            self.runCmd("log enable gdb-remote packets")
            self.addTearDownHook(
                lambda: self.runCmd("log disable gdb-remote packets"))

        self.dbg.SetDefaultArchitecture("armv7em")
        target = self.dbg.CreateTargetWithFileAndArch(None, None)

        process = self.connect(target)

        if self.TraceOn():
            interp = self.dbg.GetCommandInterpreter()
            result = lldb.SBCommandReturnObject()
            interp.HandleCommand("target list", result)
            print(result.GetOutput())

        r0_valobj = process.GetThreadAtIndex(
            0).GetFrameAtIndex(0).FindRegister("r0")
        self.assertEqual(r0_valobj.GetValueAsUnsigned(), 0x20)

        pc_valobj = process.GetThreadAtIndex(
            0).GetFrameAtIndex(0).FindRegister("pc")
        self.assertEqual(pc_valobj.GetValueAsUnsigned(), 0x0800d22e)

        pc_valobj = process.GetThreadAtIndex(
            0).GetFrameAtIndex(0).FindRegister("CONTROL")
        self.assertEqual(pc_valobj.GetValueAsUnsigned(), 0x3f8ccccd)
