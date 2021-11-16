from __future__ import print_function
from textwrap import dedent
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


class MyResponder(MockGDBServerResponder):
    def qXferRead(self, obj, annex, offset, length):
        if annex == "target.xml":
            return dedent("""\
                <?xml version="1.0"?>
                  <target version="1.0">
                    <architecture>aarch64</architecture>
                    <feature name="org.gnu.gdb.aarch64.core">
                      <reg name="cpsr" regnum="33" bitsize="32"/>
                      <reg name="x0" regnum="0" bitsize="64"/>
                      <reg name="x1" bitsize="64"/>
                      <reg name="x2" bitsize="64"/>
                      <reg name="x3" bitsize="64"/>
                      <reg name="x4" bitsize="64"/>
                      <reg name="x5" bitsize="64"/>
                      <reg name="x6" bitsize="64"/>
                      <reg name="x7" bitsize="64"/>
                      <reg name="x8" bitsize="64"/>
                      <reg name="x9" bitsize="64"/>
                      <reg name="x10" bitsize="64"/>
                      <reg name="x11" bitsize="64"/>
                      <reg name="x12" bitsize="64"/>
                      <reg name="x13" bitsize="64"/>
                      <reg name="x14" bitsize="64"/>
                      <reg name="x15" bitsize="64"/>
                      <reg name="x16" bitsize="64"/>
                      <reg name="x17" bitsize="64"/>
                      <reg name="x18" bitsize="64"/>
                      <reg name="x19" bitsize="64"/>
                      <reg name="x20" bitsize="64"/>
                      <reg name="x21" bitsize="64"/>
                      <reg name="x22" bitsize="64"/>
                      <reg name="x23" bitsize="64"/>
                      <reg name="x24" bitsize="64"/>
                      <reg name="x25" bitsize="64"/>
                      <reg name="x26" bitsize="64"/>
                      <reg name="x27" bitsize="64"/>
                      <reg name="x28" bitsize="64"/>
                      <reg name="x29" bitsize="64"/>
                      <reg name="x30" bitsize="64"/>
                      <reg name="sp" bitsize="64"/>
                      <reg name="pc" bitsize="64"/>
                      <reg name="w0" bitsize="32" value_regnums="0" invalidate_regnums="0" regnum="34"/>
                      <reg name="w1" bitsize="32" value_regnums="1" invalidate_regnums="1"/>
                      <reg name="w2" bitsize="32" value_regnums="2" invalidate_regnums="2"/>
                      <reg name="w3" bitsize="32" value_regnums="3" invalidate_regnums="3"/>
                      <reg name="w4" bitsize="32" value_regnums="4" invalidate_regnums="4"/>
                      <reg name="w5" bitsize="32" value_regnums="5" invalidate_regnums="5"/>
                      <reg name="w6" bitsize="32" value_regnums="6" invalidate_regnums="6"/>
                      <reg name="w7" bitsize="32" value_regnums="7" invalidate_regnums="7"/>
                      <reg name="w8" bitsize="32" value_regnums="8" invalidate_regnums="8"/>
                      <reg name="w9" bitsize="32" value_regnums="9" invalidate_regnums="9"/>
                      <reg name="w10" bitsize="32" value_regnums="10" invalidate_regnums="10"/>
                      <reg name="w11" bitsize="32" value_regnums="11" invalidate_regnums="11"/>
                      <reg name="w12" bitsize="32" value_regnums="12" invalidate_regnums="12"/>
                      <reg name="w13" bitsize="32" value_regnums="13" invalidate_regnums="13"/>
                      <reg name="w14" bitsize="32" value_regnums="14" invalidate_regnums="14"/>
                      <reg name="w15" bitsize="32" value_regnums="15" invalidate_regnums="15"/>
                      <reg name="w16" bitsize="32" value_regnums="16" invalidate_regnums="16"/>
                      <reg name="w17" bitsize="32" value_regnums="17" invalidate_regnums="17"/>
                      <reg name="w18" bitsize="32" value_regnums="18" invalidate_regnums="18"/>
                      <reg name="w19" bitsize="32" value_regnums="19" invalidate_regnums="19"/>
                      <reg name="w20" bitsize="32" value_regnums="20" invalidate_regnums="20"/>
                      <reg name="w21" bitsize="32" value_regnums="21" invalidate_regnums="21"/>
                      <reg name="w22" bitsize="32" value_regnums="22" invalidate_regnums="22"/>
                      <reg name="w23" bitsize="32" value_regnums="23" invalidate_regnums="23"/>
                      <reg name="w24" bitsize="32" value_regnums="24" invalidate_regnums="24"/>
                      <reg name="w25" bitsize="32" value_regnums="25" invalidate_regnums="25"/>
                      <reg name="w26" bitsize="32" value_regnums="26" invalidate_regnums="26"/>
                      <reg name="w27" bitsize="32" value_regnums="27" invalidate_regnums="27"/>
                      <reg name="w28" bitsize="32" value_regnums="28" invalidate_regnums="28"/>
                    </feature>
                  </target>
                """), False
        else:
            return None,

    def readRegister(self, regnum):
        return "E01"

    def readRegisters(self):
        return "20000000000000002000000000000000f0c154bfffff00005daa985a8fea0b48f0b954bfffff0000ad13cce570150b48380000000000000070456abfffff0000a700000000000000000000000000000001010101010101010000000000000000f0c154bfffff00000f2700000000000008e355bfffff0000080e55bfffff0000281041000000000010de61bfffff00005c05000000000000f0c154bfffff000090fcffffffff00008efcffffffff00008ffcffffffff00000000000000000000001000000000000090fcffffffff000000d06cbfffff0000f0c154bfffff00000100000000000000d0b954bfffff0000e407400000000000d0b954bfffff0000e40740000000000000100000"


class TestAArch64XMLRegOffsets(GDBRemoteTestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfXmlSupportMissing
    @skipIfRemote
    @skipIfLLVMTargetMissing("AArch64")
    def test_register_gpacket_offsets(self):
        """
        Test that we correctly associate the register info with the eh_frame
        register numbers.
        """

        target = self.createTarget("basic_eh_frame-aarch64.yaml")
        self.server.responder = MyResponder()

        if self.TraceOn():
            self.runCmd("log enable gdb-remote packets")
            self.addTearDownHook(
                lambda: self.runCmd("log disable gdb-remote packets"))

        process = self.connect(target)
        lldbutil.expect_state_changes(self, self.dbg.GetListener(), process,
                                      [lldb.eStateStopped])

        registerSet = process.GetThreadAtIndex(
            0).GetFrameAtIndex(0).GetRegisters().GetValueAtIndex(0)

        reg_val_dict = {
            "x0": 0x0000000000000020, "x1": 0x0000000000000020,
            "x2": 0x0000ffffbf54c1f0, "x3": 0x480bea8f5a98aa5d,
            "x4": 0x0000ffffbf54b9f0, "x5": 0x480b1570e5cc13ad,
            "x6": 0x0000000000000038, "x7": 0x0000ffffbf6a4570,
            "x8": 0x00000000000000a7, "x9": 0x0000000000000000,
            "x10": 0x0101010101010101, "x11": 0x0000000000000000,
            "x12": 0x0000ffffbf54c1f0, "x13": 0x000000000000270f,
            "x14": 0x0000ffffbf55e308, "x15": 0x0000ffffbf550e08,
            "x16": 0x0000000000411028, "x17": 0x0000ffffbf61de10,
            "x18": 0x000000000000055c, "x19": 0x0000ffffbf54c1f0,
            "x20": 0x0000fffffffffc90, "x21": 0x0000fffffffffc8e,
            "x22": 0x0000fffffffffc8f, "x23": 0x0000000000000000,
            "x24": 0x0000000000001000, "x25": 0x0000fffffffffc90,
            "x26": 0x0000ffffbf6cd000, "x27": 0x0000ffffbf54c1f0,
            "x28": 0x0000000000000001, "x29": 0x0000ffffbf54b9d0,
            "x30": 0x00000000004007e4, "sp": 0x0000ffffbf54b9d0,
            "pc": 0x00000000004007e4, "cpsr": 0x00001000, "w0": 0x00000020,
            "w1": 0x00000020, "w2": 0xbf54c1f0, "w3": 0x5a98aa5d,
            "w4": 0xbf54b9f0, "w5": 0xe5cc13ad, "w6": 0x00000038,
            "w7": 0xbf6a4570, "w8": 0x000000a7, "w9": 0x00000000,
            "w10": 0x01010101, "w11": 0x00000000, "w12": 0xbf54c1f0,
            "w13": 0x0000270f, "w14": 0xbf55e308, "w15": 0xbf550e08,
            "w16": 0x00411028, "w17": 0xbf61de10, "w18": 0x0000055c,
            "w19": 0xbf54c1f0, "w20": 0xfffffc90, "w21": 0xfffffc8e,
            "w22": 0xfffffc8f, "w23": 0x00000000, "w24": 0x00001000,
            "w25": 0xfffffc90, "w26": 0xbf6cd000, "w27": 0xbf54c1f0,
            "w28": 0x00000001
        }

        for reg in registerSet:
            self.assertEqual(reg.GetValueAsUnsigned(),
                             reg_val_dict[reg.GetName()])
