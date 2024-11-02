import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase

class TestJLink6Armv7RegisterDefinition(GDBRemoteTestBase):

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test(self):
        """
        Test lldb's parsing of SEGGER J-Link v6.54 register
        definition for a Cortex M-4 dev board, and the fact
        that the J-Link only supports g/G for reading/writing
        register AND the J-Link v6.54 doesn't provide anything
        but the general purpose registers."""
        class MyResponder(MockGDBServerResponder):

            def qXferRead(self, obj, annex, offset, length):
                if annex == "target.xml":
                    return """<?xml version="1.0"?>
<!-- Copyright (C) 2008 Free Software Foundation, Inc.

     Copying and distribution of this file, with or without modification,
     are permitted in any medium without royalty provided the copyright
     notice and this notice are preserved.  -->

<!DOCTYPE feature SYSTEM "gdb-target.dtd">
<target version="1.0">
  <architecture>arm</architecture>
  <feature name="org.gnu.gdb.arm.m-profile">
    <reg name="r0" bitsize="32" regnum="0" type="uint32" group="general"/>
    <reg name="r1" bitsize="32" regnum="1" type="uint32" group="general"/>
    <reg name="r2" bitsize="32" regnum="2" type="uint32" group="general"/>
    <reg name="r3" bitsize="32" regnum="3" type="uint32" group="general"/>
    <reg name="r4" bitsize="32" regnum="4" type="uint32" group="general"/>
    <reg name="r5" bitsize="32" regnum="5" type="uint32" group="general"/>
    <reg name="r6" bitsize="32" regnum="6" type="uint32" group="general"/>
    <reg name="r7" bitsize="32" regnum="7" type="uint32" group="general"/>
    <reg name="r8" bitsize="32" regnum="8" type="uint32" group="general"/>
    <reg name="r9" bitsize="32" regnum="9" type="uint32" group="general"/>
    <reg name="r10" bitsize="32" regnum="10" type="uint32" group="general"/>
    <reg name="r11" bitsize="32" regnum="11" type="uint32" group="general"/>
    <reg name="r12" bitsize="32" regnum="12" type="uint32" group="general"/>
    <reg name="sp" bitsize="32" regnum="13" type="data_ptr" group="general"/>
    <reg name="lr" bitsize="32" regnum="14" type="uint32" group="general"/>
    <reg name="pc" bitsize="32" regnum="15" type="code_ptr" group="general"/>
    <reg name="xpsr" bitsize="32" regnum="25" type="uint32" group="general"/>
  </feature>
  <feature name="org.gnu.gdb.arm.m-system">
    <reg name="msp" bitsize="32" regnum="26" type="uint32" group="general"/>
    <reg name="psp" bitsize="32" regnum="27" type="uint32" group="general"/>
    <reg name="primask" bitsize="32" regnum="28" type="uint32" group="general"/>
    <reg name="basepri" bitsize="32" regnum="29" type="uint32" group="general"/>
    <reg name="faultmask" bitsize="32" regnum="30" type="uint32" group="general"/>
    <reg name="control" bitsize="32" regnum="31" type="uint32" group="general"/>
  </feature>
  <feature name="org.gnu.gdb.arm.m-float">
    <reg name="fpscr" bitsize="32" regnum="32" type="uint32" group="float"/>
    <reg name="s0" bitsize="32" regnum="33" type="float" group="float"/>
    <reg name="s1" bitsize="32" regnum="34" type="float" group="float"/>
    <reg name="s2" bitsize="32" regnum="35" type="float" group="float"/>
    <reg name="s3" bitsize="32" regnum="36" type="float" group="float"/>
    <reg name="s4" bitsize="32" regnum="37" type="float" group="float"/>
    <reg name="s5" bitsize="32" regnum="38" type="float" group="float"/>
    <reg name="s6" bitsize="32" regnum="39" type="float" group="float"/>
    <reg name="s7" bitsize="32" regnum="40" type="float" group="float"/>
    <reg name="s8" bitsize="32" regnum="41" type="float" group="float"/>
    <reg name="s9" bitsize="32" regnum="42" type="float" group="float"/>
    <reg name="s10" bitsize="32" regnum="43" type="float" group="float"/>
    <reg name="s11" bitsize="32" regnum="44" type="float" group="float"/>
    <reg name="s12" bitsize="32" regnum="45" type="float" group="float"/>
    <reg name="s13" bitsize="32" regnum="46" type="float" group="float"/>
    <reg name="s14" bitsize="32" regnum="47" type="float" group="float"/>
    <reg name="s15" bitsize="32" regnum="48" type="float" group="float"/>
    <reg name="s16" bitsize="32" regnum="49" type="float" group="float"/>
    <reg name="s17" bitsize="32" regnum="50" type="float" group="float"/>
    <reg name="s18" bitsize="32" regnum="51" type="float" group="float"/>
    <reg name="s19" bitsize="32" regnum="52" type="float" group="float"/>
    <reg name="s20" bitsize="32" regnum="53" type="float" group="float"/>
    <reg name="s21" bitsize="32" regnum="54" type="float" group="float"/>
    <reg name="s22" bitsize="32" regnum="55" type="float" group="float"/>
    <reg name="s23" bitsize="32" regnum="56" type="float" group="float"/>
    <reg name="s24" bitsize="32" regnum="57" type="float" group="float"/>
    <reg name="s25" bitsize="32" regnum="58" type="float" group="float"/>
    <reg name="s26" bitsize="32" regnum="59" type="float" group="float"/>
    <reg name="s27" bitsize="32" regnum="60" type="float" group="float"/>
    <reg name="s28" bitsize="32" regnum="61" type="float" group="float"/>
    <reg name="s29" bitsize="32" regnum="62" type="float" group="float"/>
    <reg name="s30" bitsize="32" regnum="63" type="float" group="float"/>
    <reg name="s31" bitsize="32" regnum="64" type="float" group="float"/>
    <reg name="d0" bitsize="64" regnum="65" type="ieee_double" group="float"/>
    <reg name="d1" bitsize="64" regnum="66" type="ieee_double" group="float"/>
    <reg name="d2" bitsize="64" regnum="67" type="ieee_double" group="float"/>
    <reg name="d3" bitsize="64" regnum="68" type="ieee_double" group="float"/>
    <reg name="d4" bitsize="64" regnum="69" type="ieee_double" group="float"/>
    <reg name="d5" bitsize="64" regnum="70" type="ieee_double" group="float"/>
    <reg name="d6" bitsize="64" regnum="71" type="ieee_double" group="float"/>
    <reg name="d7" bitsize="64" regnum="72" type="ieee_double" group="float"/>
    <reg name="d8" bitsize="64" regnum="73" type="ieee_double" group="float"/>
    <reg name="d9" bitsize="64" regnum="74" type="ieee_double" group="float"/>
    <reg name="d10" bitsize="64" regnum="75" type="ieee_double" group="float"/>
    <reg name="d11" bitsize="64" regnum="76" type="ieee_double" group="float"/>
    <reg name="d12" bitsize="64" regnum="77" type="ieee_double" group="float"/>
    <reg name="d13" bitsize="64" regnum="78" type="ieee_double" group="float"/>
    <reg name="d14" bitsize="64" regnum="79" type="ieee_double" group="float"/>
    <reg name="d15" bitsize="64" regnum="80" type="ieee_double" group="float"/>
  </feature>
</target>""", False
                else:
                    return None, False

            def readRegister(self, regnum):
                return "E01"

            # Initial r1 bytes, in little-endian order
            r1_bytes = "01000000"

            ## readRegisters only provides reg values up through xpsr (0x61000000)
            ## it doesn't send up any of the exception registers or floating point
            ## registers that the above register xml describes.
            def readRegisters(self):
                return "00000000" + self.r1_bytes + "010000000100000001000000000000008c080020a872012000000000a0790120000000008065012041ad0008a0720120692a00089e26000800000061"

            ## the J-Link accepts a register write packet with just the GPRs
            ## defined.
            def writeRegisters(self, registers_hex):
                # Check that lldb returns the full 704 hex-byte register context,
                # or the 136 hex-byte general purpose register reg ctx.
                if len(registers_hex) != 704 and len(register_hex) != 136:
                    return "E06"
                if registers_hex.startswith("0000000044332211010000000100000001000000000000008c080020a872012000000000a0790120000000008065012041ad0008a0720120692a00089e26000800000061"):
                    self.r1_bytes = "44332211"
                    return "OK"
                else:
                    return "E07"

            def haltReason(self):
                return "S05"

            def qfThreadInfo(self):
                return "mdead"

            def qC(self):
                return ""

            def qSupported(self, client_supported):
                return "PacketSize=4000;qXfer:memory-map:read-;QStartNoAckMode+;hwbreak+;qXfer:features:read+"

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

        r1_valobj = process.GetThreadAtIndex(0).GetFrameAtIndex(0).FindRegister("r1")
        self.assertEqual(r1_valobj.GetValueAsUnsigned(), 1)

        pc_valobj = process.GetThreadAtIndex(0).GetFrameAtIndex(0).FindRegister("pc")
        self.assertEqual(pc_valobj.GetValueAsUnsigned(), 0x0800269e)

        xpsr_valobj = process.GetThreadAtIndex(0).GetFrameAtIndex(0).FindRegister("xpsr")
        self.assertEqual(xpsr_valobj.GetValueAsUnsigned(), 0x61000000)

        msp_valobj = process.GetThreadAtIndex(0).GetFrameAtIndex(0).FindRegister("msp")
        err = msp_valobj.GetError()
        self.assertTrue(err.Fail(), "lldb should not be able to fetch the msp register")

        val = b'\x11\x22\x33\x44'
        error = lldb.SBError()
        data = lldb.SBData()
        data.SetData(error, val, lldb.eByteOrderBig, 4)
        self.assertEqual(r1_valobj.SetData(data, error), True)
        self.assertSuccess(error)

        r1_valobj = process.GetThreadAtIndex(0).GetFrameAtIndex(0).FindRegister("r1")
        self.assertEqual(r1_valobj.GetValueAsUnsigned(), 0x11223344)

