from __future__ import print_function
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase

class TestArmRegisterDefinition(GDBRemoteTestBase):

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test(self):
        """
        Test lldb's parsing of the <architecture> tag in the target.xml register
        description packet.
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
                        <reg name="SYS0" bitsize="9" regnum="21" type="uint32" group="system"/>
                        <reg name="SYS1" bitsize="8" regnum="22" type="uint32" group="system"/>
                        <reg name="SYS2" bitsize="1" regnum="23" type="uint32" group="system"/>
                        <reg name="SYS3" bitsize="7" regnum="24" type="uint32" group="system"/>
                        <reg name="xpsr" bitsize="32" regnum="25" type="uint32" group="general"/>
                        <reg name="MSP" bitsize="32" regnum="26" type="uint32" group="general"/>
                        <reg name="PSP" bitsize="32" regnum="27" type="uint32" group="general"/>
                        <reg name="PRIMASK" bitsize="32" regnum="28" type="uint32" group="general"/>
                        <reg name="BASEPRI" bitsize="32" regnum="29" type="uint32" group="general"/>
                        <reg name="FAULTMASK" bitsize="32" regnum="30" type="uint32" group="general"/>
                        <reg name="CONTROL" bitsize="32" regnum="31" type="uint32" group="general"/>
                        <reg name="FPSCR" bitsize="32" type="uint32" group="float"/>
                        <reg name="s0" bitsize="32" type="float" group="float"/>
                        <reg name="s1" bitsize="32" type="float" group="float"/>
                        <reg name="s2" bitsize="32" type="float" group="float"/>
                        <reg name="s3" bitsize="32" type="float" group="float"/>
                        <reg name="s4" bitsize="32" type="float" group="float"/>
                        <reg name="s5" bitsize="32" type="float" group="float"/>
                        <reg name="s6" bitsize="32" type="float" group="float"/>
                        <reg name="s7" bitsize="32" type="float" group="float"/>
                        <reg name="s8" bitsize="32" type="float" group="float"/>
                        <reg name="s9" bitsize="32" type="float" group="float"/>
                        <reg name="s10" bitsize="32" type="float" group="float"/>
                        <reg name="s11" bitsize="32" type="float" group="float"/>
                        <reg name="s12" bitsize="32" type="float" group="float"/>
                        <reg name="s13" bitsize="32" type="float" group="float"/>
                        <reg name="s14" bitsize="32" type="float" group="float"/>
                        <reg name="s15" bitsize="32" type="float" group="float"/>
                        <reg name="s16" bitsize="32" type="float" group="float"/>
                        <reg name="s17" bitsize="32" type="float" group="float"/>
                        <reg name="s18" bitsize="32" type="float" group="float"/>
                        <reg name="s19" bitsize="32" type="float" group="float"/>
                        <reg name="s20" bitsize="32" type="float" group="float"/>
                        <reg name="s21" bitsize="32" type="float" group="float"/>
                        <reg name="s22" bitsize="32" type="float" group="float"/>
                        <reg name="s23" bitsize="32" type="float" group="float"/>
                        <reg name="s24" bitsize="32" type="float" group="float"/>
                        <reg name="s25" bitsize="32" type="float" group="float"/>
                        <reg name="s26" bitsize="32" type="float" group="float"/>
                        <reg name="s27" bitsize="32" type="float" group="float"/>
                        <reg name="s28" bitsize="32" type="float" group="float"/>
                        <reg name="s29" bitsize="32" type="float" group="float"/>
                        <reg name="s30" bitsize="32" type="float" group="float"/>
                        <reg name="s31" bitsize="32" type="float" group="float"/>
                        </feature>
                        </target>""", False
                else:
                    return None, False

            def readRegister(self, regnum):
                return "E01"

            def readRegisters(self):
                return "20000000f8360020001000002fcb0008f8360020a0360020200c0020000000000000000000000000000000000000000000000000b87f0120b7d100082ed20008addebeafbc00000001b87f01200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"

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

        r0_valobj = process.GetThreadAtIndex(0).GetFrameAtIndex(0).FindRegister("r0")
        self.assertEqual(r0_valobj.GetValueAsUnsigned(), 0x20)

        pc_valobj = process.GetThreadAtIndex(0).GetFrameAtIndex(0).FindRegister("pc")
        self.assertEqual(pc_valobj.GetValueAsUnsigned(), 0x0800d22e)

        sys_valobj = process.GetThreadAtIndex(0).GetFrameAtIndex(0).FindRegister("SYS0")
        self.assertEqual(sys_valobj.GetValueAsUnsigned(), 0xdead)

        sys_valobj = process.GetThreadAtIndex(0).GetFrameAtIndex(0).FindRegister("SYS1")
        self.assertEqual(sys_valobj.GetValueAsUnsigned(), 0xbe)

        sys_valobj = process.GetThreadAtIndex(0).GetFrameAtIndex(0).FindRegister("SYS2")
        self.assertEqual(sys_valobj.GetValueAsUnsigned(), 0xaf)

        sys_valobj = process.GetThreadAtIndex(0).GetFrameAtIndex(0).FindRegister("SYS3")
        self.assertEqual(sys_valobj.GetValueAsUnsigned(), 0xbc)
