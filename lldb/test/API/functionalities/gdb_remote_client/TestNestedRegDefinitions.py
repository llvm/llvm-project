from __future__ import print_function
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase

class TestNestedRegDefinitions(GDBRemoteTestBase):

    mydir = TestBase.compute_mydir(__file__)

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
                    return """<?xml version="1.0"?><!DOCTYPE target SYSTEM "gdb-target.dtd"><target><architecture>i386:x86-64</architecture><xi:include href="i386-64bit.xml"/></target>""", False

                if annex == "i386-64bit.xml":
                    return """<?xml version="1.0"?>
<!-- Copyright (C) 2010-2017 Free Software Foundation, Inc.

     Copying and distribution of this file, with or without modification,
     are permitted in any medium without royalty provided the copyright
     notice and this notice are preserved.  -->

<!-- I386 64bit -->

<!DOCTYPE target SYSTEM "gdb-target.dtd">
<feature name="org.gnu.gdb.i386.64bit">
  <xi:include href="i386-64bit-core.xml"/>
  <xi:include href="i386-64bit-sse.xml"/>
</feature>""", False

                if annex == "i386-64bit-core.xml":
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
  <reg name="ss" bitsize="32" type="int32"/>
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

                if annex == "i386-64bit-sse.xml":
                    return """<?xml version="1.0"?>
<!-- Copyright (C) 2010-2017 Free Software Foundation, Inc.

     Copying and distribution of this file, with or without modification,
     are permitted in any medium without royalty provided the copyright
     notice and this notice are preserved.  -->

<!DOCTYPE feature SYSTEM "gdb-target.dtd">
<feature name="org.gnu.gdb.i386.64bit.sse">
  <vector id="v4f" type="ieee_single" count="4"/>
  <vector id="v2d" type="ieee_double" count="2"/>
  <vector id="v16i8" type="int8" count="16"/>
  <vector id="v8i16" type="int16" count="8"/>
  <vector id="v4i32" type="int32" count="4"/>
  <vector id="v2i64" type="int64" count="2"/>
  <union id="vec128">
    <field name="v4_float" type="v4f"/>
    <field name="v2_double" type="v2d"/>
    <field name="v16_int8" type="v16i8"/>
    <field name="v8_int16" type="v8i16"/>
    <field name="v4_int32" type="v4i32"/>
    <field name="v2_int64" type="v2i64"/>
    <field name="uint128" type="uint128"/>
  </union>
  <flags id="i386_mxcsr" size="4">
    <field name="IE" start="0" end="0"/>
    <field name="DE" start="1" end="1"/>
    <field name="ZE" start="2" end="2"/>
    <field name="OE" start="3" end="3"/>
    <field name="UE" start="4" end="4"/>
    <field name="PE" start="5" end="5"/>
    <field name="DAZ" start="6" end="6"/>
    <field name="IM" start="7" end="7"/>
    <field name="DM" start="8" end="8"/>
    <field name="ZM" start="9" end="9"/>
    <field name="OM" start="10" end="10"/>
    <field name="UM" start="11" end="11"/>
    <field name="PM" start="12" end="12"/>
    <field name="FZ" start="15" end="15"/>
  </flags>

  <reg name="xmm0" bitsize="128" type="vec128" regnum="40"/>
  <reg name="xmm1" bitsize="128" type="vec128"/>
  <reg name="xmm2" bitsize="128" type="vec128"/>
  <reg name="xmm3" bitsize="128" type="vec128"/>
  <reg name="xmm4" bitsize="128" type="vec128"/>
  <reg name="xmm5" bitsize="128" type="vec128"/>
  <reg name="xmm6" bitsize="128" type="vec128"/>
  <reg name="xmm7" bitsize="128" type="vec128"/>
  <reg name="xmm8" bitsize="128" type="vec128"/>
  <reg name="xmm9" bitsize="128" type="vec128"/>
  <reg name="xmm10" bitsize="128" type="vec128"/>
  <reg name="xmm11" bitsize="128" type="vec128"/>
  <reg name="xmm12" bitsize="128" type="vec128"/>
  <reg name="xmm13" bitsize="128" type="vec128"/>
  <reg name="xmm14" bitsize="128" type="vec128"/>
  <reg name="xmm15" bitsize="128" type="vec128"/>

  <reg name="mxcsr" bitsize="32" type="i386_mxcsr" group="vector"/>
</feature>""", False

                return None, False

            def readRegister(self, regnum):
                return ""

            def readRegisters(self):
                return "0600000000000000c0b7c00080fffffff021c60080ffffff1a00000000000000020000000000000078b7c00080ffffff203f8ca090ffffff103f8ca090ffffff3025990a80ffffff809698000000000070009f0a80ffffff020000000000000000eae10080ffffff00000000000000001822d74f1a00000078b7c00080ffffff0e12410080ffff004602000008000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000007f0300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000801f0000"

            def haltReason(self):
                return "T02thread:dead;threads:dead;"

            def qfThreadInfo(self):
                return "mdead"
            
            def qC(self):
                return ""

            def qSupported(self, client_supported):
                return "PacketSize=4000;qXfer:features:read+"

            def QThreadSuffixSupported(self):
                return "OK"

            def QListThreadsInStopReply(self):
                return "OK"

        self.server.responder = MyResponder()
        if self.TraceOn():
            self.runCmd("log enable gdb-remote packets")
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

        r15_valobj = process.GetThreadAtIndex(0).GetFrameAtIndex(0).FindRegister("r15")
        self.assertEqual(r15_valobj.GetValueAsUnsigned(), 0xffffff8000c0b778)

        mxcsr_valobj = process.GetThreadAtIndex(0).GetFrameAtIndex(0).FindRegister("mxcsr")
        self.assertEqual(mxcsr_valobj.GetValueAsUnsigned(), 0x00001f80)

        gpr_reg_set_name = process.GetThreadAtIndex(0).GetFrameAtIndex(0).GetRegisters().GetValueAtIndex(0).GetName()
        self.assertEqual(gpr_reg_set_name, "general")

        float_reg_set_name = process.GetThreadAtIndex(0).GetFrameAtIndex(0).GetRegisters().GetValueAtIndex(1).GetName()
        self.assertEqual(float_reg_set_name, "float")

        vector_reg_set_name = process.GetThreadAtIndex(0).GetFrameAtIndex(0).GetRegisters().GetValueAtIndex(2).GetName()
        self.assertEqual(vector_reg_set_name, "vector")

        if self.TraceOn():
            print("rip is 0x%x" % rip_valobj.GetValueAsUnsigned())
            print("r15 is 0x%x" % r15_valobj.GetValueAsUnsigned())
            print("mxcsr is 0x%x" % mxcsr_valobj.GetValueAsUnsigned())
