import lldb
import time
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase

class TestRegDefinitionInParts(GDBRemoteTestBase):

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test(self):
        """
        Test that lldb correctly fetches the target definition file
        in multiple chunks if the remote server only provides the 
        content in small parts, and the small parts it provides is
        smaller than the maximum packet size that it declared at
        the start of the debug session.  qemu does this.
        """
        class MyResponder(MockGDBServerResponder):

            def qXferRead(self, obj, annex, offset, length):
                if annex == "target.xml":
                    return """<?xml version="1.0"?>
                              <!DOCTYPE feature SYSTEM "gdb-target.dtd">
                              <target version="1.0">
                              <architecture>i386:x86-64</architecture>
                              <xi:include href="i386-64bit-core.xml"/>
                              </target>""", False

                if annex == "i386-64bit-core.xml" and offset == 0:
                    return """<?xml version="1.0"?>
<!-- Copyright (C) 2010-2015 Free Software Foundation, Inc.

     Copying and distribution of this file, with or without modification,
     are permitted in any medium without royalty provided the copyright
     notice and this notice are preserved.  -->

<!DOCTYPE feature SYSTEM "gdb-target.dtd">
<feature name="org.gnu.gdb.i386.core">
  <flags id="i386_eflags" size="4">
    <field name="CF" start="0" end="0"/>
    <field name="" start="1" end="1"/>
    <field name="PF" start="2" end="2"/>
    <field name="AF" start="4" end="4"/>
    <field name="ZF" start="6" end="6"/>
    <field name="SF" start="7" end="7"/>
    <field name="TF" start="8" end="8"/>
    <field name="IF" start="9" end="9"/>
    <field name="DF" start="10" end="10"/>
    <field name="OF" start="11" end="11"/>
    <field name="NT" start="14" end="14"/>
    <field name="RF" start="16" end="16"/>
    <field name="VM" start="17" end="17"/>
    <field name="AC" start="18" end="18"/>
    <field name="VIF" start="19" end="19"/>
    <field name="VIP" start="20" end="20"/>
    <field name="ID" start="21" end="21"/>
  </flags>

  <reg name="rax" bitsize="64" type="int64"/>
  <reg name="rbx" bitsize="64" type="int64"/>
  <reg name="rcx" bitsize="64" type="int64"/>
  <reg name="rdx" bitsize="64" type="int64"/>
  <reg name="rsi" bitsize="64" type="int64"/>
  <reg name="rdi" bitsize="64" type="int64"/>
  <reg name="rbp" bitsize="64" type="data_ptr"/>
  <reg name="rsp" bitsize="64" type="data_ptr"/>
  <reg name="r8" bitsize="64" type="int64"/>
  <reg name="r9" bitsize="64" type="int64"/>
  <reg name="r10" bitsize="64" type="int64"/>
  <reg name="r11" bitsize="64" type="int64"/>
  <reg name="r12" bitsize="64" type="int64"/>
  <reg name="r13" bitsize="64" type="int64"/>
  <reg name="r14" bitsize="64" type="int64"/>
  <reg name="r15" bitsize="64" type="int64"/>

  <reg name="rip" bitsize="64" type="code_ptr"/>
  <reg name="eflags" bitsize="32" type="i386_eflags"/>
  <reg name="cs" bitsize="32" type="int32"/>
  <reg name="ss" bitsize="32" ty""", True

                if annex == "i386-64bit-core.xml" and offset == 2045:
                    return """pe="int32"/>
  <reg name="ds" bitsize="32" type="int32"/>
  <reg name="es" bitsize="32" type="int32"/>
  <reg name="fs" bitsize="32" type="int32"/>
  <reg name="gs" bitsize="32" type="int32"/>

  <reg name="st0" bitsize="80" type="i387_ext"/>
  <reg name="st1" bitsize="80" type="i387_ext"/>
  <reg name="st2" bitsize="80" type="i387_ext"/>
  <reg name="st3" bitsize="80" type="i387_ext"/>
  <reg name="st4" bitsize="80" type="i387_ext"/>
  <reg name="st5" bitsize="80" type="i387_ext"/>
  <reg name="st6" bitsize="80" type="i387_ext"/>
  <reg name="st7" bitsize="80" type="i387_ext"/>

  <reg name="fctrl" bitsize="32" type="int" group="float"/>
  <reg name="fstat" bitsize="32" type="int" group="float"/>
  <reg name="ftag" bitsize="32" type="int" group="float"/>
  <reg name="fiseg" bitsize="32" type="int" group="float"/>
  <reg name="fioff" bitsize="32" type="int" group="float"/>
  <reg name="foseg" bitsize="32" type="int" group="float"/>
  <reg name="fooff" bitsize="32" type="int" group="float"/>
  <reg name="fop" bitsize="32" type="int" group="float"/>
</feature>""", False

                return None, False

            def readRegister(self, regnum):
                return ""

            def readRegisters(self):
                return "0600000000000000c0b7c00080fffffff021c60080ffffff1a00000000000000020000000000000078b7c00080ffffff203f8ca090ffffff103f8ca090ffffff3025990a80ffffff809698000000000070009f0a80ffffff020000000000000000eae10080ffffff00000000000000001822d74f1a00000078b7c00080ffffff0e12410080ffff004602000011111111222222223333333300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000007f0300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000801f0000"

            def haltReason(self):
                return "T02thread:dead;threads:dead;"

            def qfThreadInfo(self):
                return "mdead"
            
            def qC(self):
                return ""

            def qSupported(self, client_supported):
                return "PacketSize=1000;qXfer:features:read+"

            def QThreadSuffixSupported(self):
                return "OK"

            def QListThreadsInStopReply(self):
                return "OK"

        self.server.responder = MyResponder()
        if self.TraceOn():
            self.runCmd("log enable gdb-remote packets")
            time.sleep(10)
            self.addTearDownHook(
                    lambda: self.runCmd("log disable gdb-remote packets"))

        target = self.dbg.CreateTargetWithFileAndArch(None, None)

        process = self.connect(target)

        if self.TraceOn():
            interp = self.dbg.GetCommandInterpreter()
            result = lldb.SBCommandReturnObject()
            interp.HandleCommand("target list", result)
            print(result.GetOutput())

        rip_valobj = process.GetThreadAtIndex(0).GetFrameAtIndex(0).FindRegister("rip")
        self.assertEqual(rip_valobj.GetValueAsUnsigned(), 0x00ffff800041120e)

        ss_valobj = process.GetThreadAtIndex(0).GetFrameAtIndex(0).FindRegister("ss")
        self.assertEqual(ss_valobj.GetValueAsUnsigned(), 0x22222222)

        if self.TraceOn():
            print("rip is 0x%x" % rip_valobj.GetValueAsUnsigned())
            print("ss is 0x%x" % ss_valobj.GetValueAsUnsigned())
