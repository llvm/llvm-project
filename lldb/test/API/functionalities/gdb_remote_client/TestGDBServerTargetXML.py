from __future__ import print_function
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from gdbclientutils import *


class TestGDBServerTargetXML(GDBRemoteTestBase):
    @skipIfXmlSupportMissing
    @skipIfRemote
    @skipIfLLVMTargetMissing("X86")
    def test_x86_64_regs(self):
        """Test grabbing various x86_64 registers from gdbserver."""
        reg_data = [
            "0102030405060708",  # rcx
            "1112131415161718",  # rdx
            "2122232425262728",  # rsi
            "3132333435363738",  # rdi
            "4142434445464748",  # rbp
            "5152535455565758",  # rsp
            "6162636465666768",  # r8
            "7172737475767778",  # r9
            "8182838485868788",  # rip
            "91929394",  # eflags
            "0102030405060708090a",  # st0
            "1112131415161718191a",  # st1
        ] + 6 * [
            "2122232425262728292a"  # st2..st7
        ] + [
            "8182838485868788898a8b8c8d8e8f90",  # xmm0
            "9192939495969798999a9b9c9d9e9fa0",  # xmm1
        ] + 14 * [
            "a1a2a3a4a5a6a7a8a9aaabacadaeafb0",  # xmm2..xmm15
        ] + [
            "00000000",  # mxcsr
        ] + [
            "b1b2b3b4b5b6b7b8b9babbbcbdbebfc0",  # ymm0h
            "c1c2c3c4c5c6c7c8c9cacbcccdcecfd0",  # ymm1h
        ] + 14 * [
            "d1d2d3d4d5d6d7d8d9dadbdcdddedfe0",  # ymm2h..ymm15h
        ]

        class MyResponder(MockGDBServerResponder):
            def qXferRead(self, obj, annex, offset, length):
                if annex == "target.xml":
                    return """<?xml version="1.0"?>
                        <!DOCTYPE feature SYSTEM "gdb-target.dtd">
                        <target>
                          <architecture>i386:x86-64</architecture>
                          <osabi>GNU/Linux</osabi>
                          <feature name="org.gnu.gdb.i386.core">
                            <reg name="rcx" bitsize="64" type="int64" regnum="2"/>
                            <reg name="rdx" bitsize="64" type="int64" regnum="3"/>
                            <reg name="rsi" bitsize="64" type="int64" regnum="4"/>
                            <reg name="rdi" bitsize="64" type="int64" regnum="5"/>
                            <reg name="rbp" bitsize="64" type="data_ptr" regnum="6"/>
                            <reg name="rsp" bitsize="64" type="data_ptr" regnum="7"/>
                            <reg name="r8" bitsize="64" type="int64" regnum="8"/>
                            <reg name="r9" bitsize="64" type="int64" regnum="9"/>
                            <reg name="rip" bitsize="64" type="code_ptr" regnum="16"/>
                            <reg name="eflags" bitsize="32" type="i386_eflags" regnum="17"/>
                            <reg name="st0" bitsize="80" type="i387_ext" regnum="24"/>
                            <reg name="st1" bitsize="80" type="i387_ext" regnum="25"/>
                            <reg name="st2" bitsize="80" type="i387_ext" regnum="26"/>
                            <reg name="st3" bitsize="80" type="i387_ext" regnum="27"/>
                            <reg name="st4" bitsize="80" type="i387_ext" regnum="28"/>
                            <reg name="st5" bitsize="80" type="i387_ext" regnum="29"/>
                            <reg name="st6" bitsize="80" type="i387_ext" regnum="30"/>
                            <reg name="st7" bitsize="80" type="i387_ext" regnum="31"/>
                          </feature>
                          <feature name="org.gnu.gdb.i386.sse">
                            <reg name="xmm0" bitsize="128" type="vec128" regnum="40"/>
                            <reg name="xmm1" bitsize="128" type="vec128" regnum="41"/>
                            <reg name="xmm2" bitsize="128" type="vec128" regnum="42"/>
                            <reg name="xmm3" bitsize="128" type="vec128" regnum="43"/>
                            <reg name="xmm4" bitsize="128" type="vec128" regnum="44"/>
                            <reg name="xmm5" bitsize="128" type="vec128" regnum="45"/>
                            <reg name="xmm6" bitsize="128" type="vec128" regnum="46"/>
                            <reg name="xmm7" bitsize="128" type="vec128" regnum="47"/>
                            <reg name="xmm8" bitsize="128" type="vec128" regnum="48"/>
                            <reg name="xmm9" bitsize="128" type="vec128" regnum="49"/>
                            <reg name="xmm10" bitsize="128" type="vec128" regnum="50"/>
                            <reg name="xmm11" bitsize="128" type="vec128" regnum="51"/>
                            <reg name="xmm12" bitsize="128" type="vec128" regnum="52"/>
                            <reg name="xmm13" bitsize="128" type="vec128" regnum="53"/>
                            <reg name="xmm14" bitsize="128" type="vec128" regnum="54"/>
                            <reg name="xmm15" bitsize="128" type="vec128" regnum="55"/>
                            <reg name="mxcsr" bitsize="32" type="i386_mxcsr" regnum="56" group="vector"/>
                          </feature>
                          <feature name="org.gnu.gdb.i386.avx">
                            <reg name="ymm0h" bitsize="128" type="uint128" regnum="60"/>
                            <reg name="ymm1h" bitsize="128" type="uint128" regnum="61"/>
                            <reg name="ymm2h" bitsize="128" type="uint128" regnum="62"/>
                            <reg name="ymm3h" bitsize="128" type="uint128" regnum="63"/>
                            <reg name="ymm4h" bitsize="128" type="uint128" regnum="64"/>
                            <reg name="ymm5h" bitsize="128" type="uint128" regnum="65"/>
                            <reg name="ymm6h" bitsize="128" type="uint128" regnum="66"/>
                            <reg name="ymm7h" bitsize="128" type="uint128" regnum="67"/>
                            <reg name="ymm8h" bitsize="128" type="uint128" regnum="68"/>
                            <reg name="ymm9h" bitsize="128" type="uint128" regnum="69"/>
                            <reg name="ymm10h" bitsize="128" type="uint128" regnum="70"/>
                            <reg name="ymm11h" bitsize="128" type="uint128" regnum="71"/>
                            <reg name="ymm12h" bitsize="128" type="uint128" regnum="72"/>
                            <reg name="ymm13h" bitsize="128" type="uint128" regnum="73"/>
                            <reg name="ymm14h" bitsize="128" type="uint128" regnum="74"/>
                            <reg name="ymm15h" bitsize="128" type="uint128" regnum="75"/>
                          </feature>
                        </target>""", False
                else:
                    return None, False

            def readRegister(self, regnum):
                return ""

            def readRegisters(self):
                return "".join(reg_data)

            def writeRegisters(self, reg_hex):
                return "OK"

            def haltReason(self):
                return "T02thread:1ff0d;threads:1ff0d;thread-pcs:000000010001bc00;07:0102030405060708;10:1112131415161718;"

        self.server.responder = MyResponder()

        target = self.createTarget("basic_eh_frame.yaml")
        process = self.connect(target)
        lldbutil.expect_state_changes(self, self.dbg.GetListener(), process,
                                      [lldb.eStateStopped])

        # test generic aliases
        self.match("register read arg4",
                   ["rcx = 0x0807060504030201"])
        self.match("register read arg3",
                   ["rdx = 0x1817161514131211"])
        self.match("register read arg2",
                   ["rsi = 0x2827262524232221"])
        self.match("register read arg1",
                   ["rdi = 0x3837363534333231"])
        self.match("register read fp",
                   ["rbp = 0x4847464544434241"])
        self.match("register read sp",
                   ["rsp = 0x5857565554535251"])
        self.match("register read arg5",
                   ["r8 = 0x6867666564636261"])
        self.match("register read arg6",
                   ["r9 = 0x7877767574737271"])
        self.match("register read pc",
                   ["rip = 0x8887868584838281"])
        self.match("register read flags",
                   ["eflags = 0x94939291"])

        # both stX and xmmX should be displayed as vectors
        self.match("register read st0",
                   ["st0 = {0x01 0x02 0x03 0x04 0x05 0x06 0x07 0x08 0x09 0x0a}"])
        self.match("register read st1",
                   ["st1 = {0x11 0x12 0x13 0x14 0x15 0x16 0x17 0x18 0x19 0x1a}"])
        self.match("register read xmm0",
                   ["xmm0 = {0x81 0x82 0x83 0x84 0x85 0x86 0x87 0x88 "
                    "0x89 0x8a 0x8b 0x8c 0x8d 0x8e 0x8f 0x90}"])
        self.match("register read xmm1",
                   ["xmm1 = {0x91 0x92 0x93 0x94 0x95 0x96 0x97 0x98 "
                    "0x99 0x9a 0x9b 0x9c 0x9d 0x9e 0x9f 0xa0}"])

        # test pseudo-registers
        self.filecheck("register read --all",
                       os.path.join(os.path.dirname(__file__),
                                    "amd64-partial-regs.FileCheck"))

        # test writing into pseudo-registers
        self.runCmd("register write ecx 0xfffefdfc")
        reg_data[0] = "fcfdfeff05060708"
        self.assertPacketLogContains(["G" + "".join(reg_data)])
        self.match("register read rcx",
                   ["rcx = 0x08070605fffefdfc"])

        self.runCmd("register write cx 0xfbfa")
        reg_data[0] = "fafbfeff05060708"
        self.assertPacketLogContains(["G" + "".join(reg_data)])
        self.match("register read ecx",
                   ["ecx = 0xfffefbfa"])
        self.match("register read rcx",
                   ["rcx = 0x08070605fffefbfa"])

        self.runCmd("register write ch 0xf9")
        reg_data[0] = "faf9feff05060708"
        self.assertPacketLogContains(["G" + "".join(reg_data)])
        self.match("register read cx",
                   ["cx = 0xf9fa"])
        self.match("register read ecx",
                   ["ecx = 0xfffef9fa"])
        self.match("register read rcx",
                   ["rcx = 0x08070605fffef9fa"])

        self.runCmd("register write cl 0xf8")
        reg_data[0] = "f8f9feff05060708"
        self.assertPacketLogContains(["G" + "".join(reg_data)])
        self.match("register read cx",
                   ["cx = 0xf9f8"])
        self.match("register read ecx",
                   ["ecx = 0xfffef9f8"])
        self.match("register read rcx",
                   ["rcx = 0x08070605fffef9f8"])

        self.runCmd("register write mm0 0xfffefdfcfbfaf9f8")
        reg_data[10] = "f8f9fafbfcfdfeff090a"
        self.assertPacketLogContains(["G" + "".join(reg_data)])
        self.match("register read st0",
                   ["st0 = {0xf8 0xf9 0xfa 0xfb 0xfc 0xfd 0xfe 0xff 0x09 0x0a}"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    @skipIfLLVMTargetMissing("X86")
    def test_i386_regs(self):
        """Test grabbing various i386 registers from gdbserver."""
        reg_data = [
            "01020304",  # eax
            "11121314",  # ecx
            "21222324",  # edx
            "31323334",  # ebx
            "41424344",  # esp
            "51525354",  # ebp
            "61626364",  # esi
            "71727374",  # edi
            "81828384",  # eip
            "91929394",  # eflags
            "0102030405060708090a",  # st0
            "1112131415161718191a",  # st1
        ] + 6 * [
            "2122232425262728292a"  # st2..st7
        ] + [
            "8182838485868788898a8b8c8d8e8f90",  # xmm0
            "9192939495969798999a9b9c9d9e9fa0",  # xmm1
        ] + 6 * [
            "a1a2a3a4a5a6a7a8a9aaabacadaeafb0",  # xmm2..xmm7
        ] + [
            "00000000",  # mxcsr
        ] + [
            "b1b2b3b4b5b6b7b8b9babbbcbdbebfc0",  # ymm0h
            "c1c2c3c4c5c6c7c8c9cacbcccdcecfd0",  # ymm1h
        ] + 6 * [
            "d1d2d3d4d5d6d7d8d9dadbdcdddedfe0",  # ymm2h..ymm7h
        ]

        class MyResponder(MockGDBServerResponder):
            def qXferRead(self, obj, annex, offset, length):
                if annex == "target.xml":
                    return """<?xml version="1.0"?>
                        <!DOCTYPE feature SYSTEM "gdb-target.dtd">
                        <target>
                          <architecture>i386</architecture>
                          <osabi>GNU/Linux</osabi>
                          <feature name="org.gnu.gdb.i386.core">
                            <reg name="eax" bitsize="32" type="int32" regnum="0"/>
                            <reg name="ecx" bitsize="32" type="int32" regnum="1"/>
                            <reg name="edx" bitsize="32" type="int32" regnum="2"/>
                            <reg name="ebx" bitsize="32" type="int32" regnum="3"/>
                            <reg name="esp" bitsize="32" type="data_ptr" regnum="4"/>
                            <reg name="ebp" bitsize="32" type="data_ptr" regnum="5"/>
                            <reg name="esi" bitsize="32" type="int32" regnum="6"/>
                            <reg name="edi" bitsize="32" type="int32" regnum="7"/>
                            <reg name="eip" bitsize="32" type="code_ptr" regnum="8"/>
                            <reg name="eflags" bitsize="32" type="i386_eflags" regnum="9"/>
                            <reg name="st0" bitsize="80" type="i387_ext" regnum="16"/>
                            <reg name="st1" bitsize="80" type="i387_ext" regnum="17"/>
                            <reg name="st2" bitsize="80" type="i387_ext" regnum="18"/>
                            <reg name="st3" bitsize="80" type="i387_ext" regnum="19"/>
                            <reg name="st4" bitsize="80" type="i387_ext" regnum="20"/>
                            <reg name="st5" bitsize="80" type="i387_ext" regnum="21"/>
                            <reg name="st6" bitsize="80" type="i387_ext" regnum="22"/>
                            <reg name="st7" bitsize="80" type="i387_ext" regnum="23"/>
                          </feature>
                          <feature name="org.gnu.gdb.i386.sse">
                            <reg name="xmm0" bitsize="128" type="vec128" regnum="32"/>
                            <reg name="xmm1" bitsize="128" type="vec128" regnum="33"/>
                            <reg name="xmm2" bitsize="128" type="vec128" regnum="34"/>
                            <reg name="xmm3" bitsize="128" type="vec128" regnum="35"/>
                            <reg name="xmm4" bitsize="128" type="vec128" regnum="36"/>
                            <reg name="xmm5" bitsize="128" type="vec128" regnum="37"/>
                            <reg name="xmm6" bitsize="128" type="vec128" regnum="38"/>
                            <reg name="xmm7" bitsize="128" type="vec128" regnum="39"/>
                            <reg name="mxcsr" bitsize="32" type="i386_mxcsr" regnum="40" group="vector"/>
                          </feature>
                          <feature name="org.gnu.gdb.i386.avx">
                            <reg name="ymm0h" bitsize="128" type="uint128" regnum="42"/>
                            <reg name="ymm1h" bitsize="128" type="uint128" regnum="43"/>
                            <reg name="ymm2h" bitsize="128" type="uint128" regnum="44"/>
                            <reg name="ymm3h" bitsize="128" type="uint128" regnum="45"/>
                            <reg name="ymm4h" bitsize="128" type="uint128" regnum="46"/>
                            <reg name="ymm5h" bitsize="128" type="uint128" regnum="47"/>
                            <reg name="ymm6h" bitsize="128" type="uint128" regnum="48"/>
                            <reg name="ymm7h" bitsize="128" type="uint128" regnum="49"/>
                          </feature>
                        </target>""", False
                else:
                    return None, False

            def readRegister(self, regnum):
                return ""

            def readRegisters(self):
                return "".join(reg_data)

            def writeRegisters(self, reg_hex):
                return "OK"

            def haltReason(self):
                return "T02thread:1ff0d;threads:1ff0d;thread-pcs:000000010001bc00;07:0102030405060708;10:1112131415161718;"

        self.server.responder = MyResponder()

        target = self.createTarget("basic_eh_frame-i386.yaml")
        process = self.connect(target)
        lldbutil.expect_state_changes(self, self.dbg.GetListener(), process,
                                      [lldb.eStateStopped])

        # test generic aliases
        self.match("register read fp",
                   ["ebp = 0x54535251"])
        self.match("register read sp",
                   ["esp = 0x44434241"])
        self.match("register read pc",
                   ["eip = 0x84838281"])
        self.match("register read flags",
                   ["eflags = 0x94939291"])

        # test pseudo-registers
        self.match("register read cx",
                   ["cx = 0x1211"])
        self.match("register read ch",
                   ["ch = 0x12"])
        self.match("register read cl",
                   ["cl = 0x11"])
        self.match("register read mm0",
                   ["mm0 = 0x0807060504030201"])
        self.match("register read mm1",
                   ["mm1 = 0x1817161514131211"])

        # both stX and xmmX should be displayed as vectors
        self.match("register read st0",
                   ["st0 = {0x01 0x02 0x03 0x04 0x05 0x06 0x07 0x08 0x09 0x0a}"])
        self.match("register read st1",
                   ["st1 = {0x11 0x12 0x13 0x14 0x15 0x16 0x17 0x18 0x19 0x1a}"])
        self.match("register read xmm0",
                   ["xmm0 = {0x81 0x82 0x83 0x84 0x85 0x86 0x87 0x88 "
                    "0x89 0x8a 0x8b 0x8c 0x8d 0x8e 0x8f 0x90}"])
        self.match("register read xmm1",
                   ["xmm1 = {0x91 0x92 0x93 0x94 0x95 0x96 0x97 0x98 "
                    "0x99 0x9a 0x9b 0x9c 0x9d 0x9e 0x9f 0xa0}"])

        # test writing into pseudo-registers
        self.runCmd("register write cx 0xfbfa")
        reg_data[1] = "fafb1314"
        self.assertPacketLogContains(["G" + "".join(reg_data)])
        self.match("register read ecx",
                   ["ecx = 0x1413fbfa"])

        self.runCmd("register write ch 0xf9")
        reg_data[1] = "faf91314"
        self.assertPacketLogContains(["G" + "".join(reg_data)])
        self.match("register read cx",
                   ["cx = 0xf9fa"])
        self.match("register read ecx",
                   ["ecx = 0x1413f9fa"])

        self.runCmd("register write cl 0xf8")
        reg_data[1] = "f8f91314"
        self.assertPacketLogContains(["G" + "".join(reg_data)])
        self.match("register read cx",
                   ["cx = 0xf9f8"])
        self.match("register read ecx",
                   ["ecx = 0x1413f9f8"])

        self.runCmd("register write mm0 0xfffefdfcfbfaf9f8")
        reg_data[10] = "f8f9fafbfcfdfeff090a"
        self.assertPacketLogContains(["G" + "".join(reg_data)])
        self.match("register read st0",
                   ["st0 = {0xf8 0xf9 0xfa 0xfb 0xfc 0xfd 0xfe 0xff 0x09 0x0a}"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    @skipIfLLVMTargetMissing("AArch64")
    def test_aarch64_regs(self):
        """Test grabbing various aarch64 registers from gdbserver."""
        class MyResponder(MockGDBServerResponder):
            reg_data = (
                "0102030405060708"  # x0
                "1112131415161718"  # x1
            ) + 27 * (
                "2122232425262728"  # x2..x28
            ) + (
                "3132333435363738"  # x29 (fp)
                "4142434445464748"  # x30 (lr)
                "5152535455565758"  # x31 (sp)
                "6162636465666768"  # pc
                "71727374"  # cpsr
                "8182838485868788898a8b8c8d8e8f90"  # v0
                "9192939495969798999a9b9c9d9e9fa0"  # v1
            ) + 30 * (
                "a1a2a3a4a5a6a7a8a9aaabacadaeafb0"  # v2..v31
            ) + (
                "00000000"  # fpsr
                "00000000"  # fpcr
            )

            def qXferRead(self, obj, annex, offset, length):
                if annex == "target.xml":
                    return """<?xml version="1.0"?>
                        <!DOCTYPE feature SYSTEM "gdb-target.dtd">
                        <target>
                          <architecture>aarch64</architecture>
                          <feature name="org.gnu.gdb.aarch64.core">
                            <reg name="x0" bitsize="64" type="int" regnum="0"/>
                            <reg name="x1" bitsize="64" type="int" regnum="1"/>
                            <reg name="x2" bitsize="64" type="int" regnum="2"/>
                            <reg name="x3" bitsize="64" type="int" regnum="3"/>
                            <reg name="x4" bitsize="64" type="int" regnum="4"/>
                            <reg name="x5" bitsize="64" type="int" regnum="5"/>
                            <reg name="x6" bitsize="64" type="int" regnum="6"/>
                            <reg name="x7" bitsize="64" type="int" regnum="7"/>
                            <reg name="x8" bitsize="64" type="int" regnum="8"/>
                            <reg name="x9" bitsize="64" type="int" regnum="9"/>
                            <reg name="x10" bitsize="64" type="int" regnum="10"/>
                            <reg name="x11" bitsize="64" type="int" regnum="11"/>
                            <reg name="x12" bitsize="64" type="int" regnum="12"/>
                            <reg name="x13" bitsize="64" type="int" regnum="13"/>
                            <reg name="x14" bitsize="64" type="int" regnum="14"/>
                            <reg name="x15" bitsize="64" type="int" regnum="15"/>
                            <reg name="x16" bitsize="64" type="int" regnum="16"/>
                            <reg name="x17" bitsize="64" type="int" regnum="17"/>
                            <reg name="x18" bitsize="64" type="int" regnum="18"/>
                            <reg name="x19" bitsize="64" type="int" regnum="19"/>
                            <reg name="x20" bitsize="64" type="int" regnum="20"/>
                            <reg name="x21" bitsize="64" type="int" regnum="21"/>
                            <reg name="x22" bitsize="64" type="int" regnum="22"/>
                            <reg name="x23" bitsize="64" type="int" regnum="23"/>
                            <reg name="x24" bitsize="64" type="int" regnum="24"/>
                            <reg name="x25" bitsize="64" type="int" regnum="25"/>
                            <reg name="x26" bitsize="64" type="int" regnum="26"/>
                            <reg name="x27" bitsize="64" type="int" regnum="27"/>
                            <reg name="x28" bitsize="64" type="int" regnum="28"/>
                            <reg name="x29" bitsize="64" type="int" regnum="29"/>
                            <reg name="x30" bitsize="64" type="int" regnum="30"/>
                            <reg name="sp" bitsize="64" type="data_ptr" regnum="31"/>
                            <reg name="pc" bitsize="64" type="code_ptr" regnum="32"/>
                            <reg name="cpsr" bitsize="32" type="cpsr_flags" regnum="33"/>
                          </feature>
                          <feature name="org.gnu.gdb.aarch64.fpu">
                            <reg name="v0" bitsize="128" type="aarch64v" regnum="34"/>
                            <reg name="v1" bitsize="128" type="aarch64v" regnum="35"/>
                            <reg name="v2" bitsize="128" type="aarch64v" regnum="36"/>
                            <reg name="v3" bitsize="128" type="aarch64v" regnum="37"/>
                            <reg name="v4" bitsize="128" type="aarch64v" regnum="38"/>
                            <reg name="v5" bitsize="128" type="aarch64v" regnum="39"/>
                            <reg name="v6" bitsize="128" type="aarch64v" regnum="40"/>
                            <reg name="v7" bitsize="128" type="aarch64v" regnum="41"/>
                            <reg name="v8" bitsize="128" type="aarch64v" regnum="42"/>
                            <reg name="v9" bitsize="128" type="aarch64v" regnum="43"/>
                            <reg name="v10" bitsize="128" type="aarch64v" regnum="44"/>
                            <reg name="v11" bitsize="128" type="aarch64v" regnum="45"/>
                            <reg name="v12" bitsize="128" type="aarch64v" regnum="46"/>
                            <reg name="v13" bitsize="128" type="aarch64v" regnum="47"/>
                            <reg name="v14" bitsize="128" type="aarch64v" regnum="48"/>
                            <reg name="v15" bitsize="128" type="aarch64v" regnum="49"/>
                            <reg name="v16" bitsize="128" type="aarch64v" regnum="50"/>
                            <reg name="v17" bitsize="128" type="aarch64v" regnum="51"/>
                            <reg name="v18" bitsize="128" type="aarch64v" regnum="52"/>
                            <reg name="v19" bitsize="128" type="aarch64v" regnum="53"/>
                            <reg name="v20" bitsize="128" type="aarch64v" regnum="54"/>
                            <reg name="v21" bitsize="128" type="aarch64v" regnum="55"/>
                            <reg name="v22" bitsize="128" type="aarch64v" regnum="56"/>
                            <reg name="v23" bitsize="128" type="aarch64v" regnum="57"/>
                            <reg name="v24" bitsize="128" type="aarch64v" regnum="58"/>
                            <reg name="v25" bitsize="128" type="aarch64v" regnum="59"/>
                            <reg name="v26" bitsize="128" type="aarch64v" regnum="60"/>
                            <reg name="v27" bitsize="128" type="aarch64v" regnum="61"/>
                            <reg name="v28" bitsize="128" type="aarch64v" regnum="62"/>
                            <reg name="v29" bitsize="128" type="aarch64v" regnum="63"/>
                            <reg name="v30" bitsize="128" type="aarch64v" regnum="64"/>
                            <reg name="v31" bitsize="128" type="aarch64v" regnum="65"/>
                            <reg name="fpsr" bitsize="32" type="int" regnum="66"/>
                            <reg name="fpcr" bitsize="32" type="int" regnum="67"/>
                          </feature>
                        </target>""", False
                else:
                    return None, False

            def readRegister(self, regnum):
                return ""

            def readRegisters(self):
                return self.reg_data

            def writeRegisters(self, reg_hex):
                self.reg_data = reg_hex
                return "OK"

            def haltReason(self):
                return "T02thread:1ff0d;threads:1ff0d;thread-pcs:000000010001bc00;07:0102030405060708;10:1112131415161718;"

        self.server.responder = MyResponder()

        target = self.createTarget("basic_eh_frame-aarch64.yaml")
        process = self.connect(target)
        lldbutil.expect_state_changes(self, self.dbg.GetListener(), process,
                                      [lldb.eStateStopped])

        # test GPRs
        self.match("register read x0",
                   ["x0 = 0x0807060504030201"])
        self.match("register read x1",
                   ["x1 = 0x1817161514131211"])
        self.match("register read x29",
                   ["x29 = 0x3837363534333231"])
        self.match("register read x30",
                   ["x30 = 0x4847464544434241"])
        self.match("register read x31",
                   ["sp = 0x5857565554535251"])
        self.match("register read sp",
                   ["sp = 0x5857565554535251"])
        self.match("register read pc",
                   ["pc = 0x6867666564636261"])
        self.match("register read cpsr",
                   ["cpsr = 0x74737271"])

        # test generic aliases
        self.match("register read arg1",
                   ["x0 = 0x0807060504030201"])
        self.match("register read arg2",
                   ["x1 = 0x1817161514131211"])
        self.match("register read fp",
                   ["x29 = 0x3837363534333231"])
        self.match("register read lr",
                   ["x30 = 0x4847464544434241"])
        self.match("register read ra",
                   ["x30 = 0x4847464544434241"])
        self.match("register read flags",
                   ["cpsr = 0x74737271"])

        # test vector registers
        self.match("register read v0",
                   ["v0 = {0x81 0x82 0x83 0x84 0x85 0x86 0x87 0x88 0x89 0x8a 0x8b 0x8c 0x8d 0x8e 0x8f 0x90}"])
        self.match("register read v31",
                   ["v31 = {0xa1 0xa2 0xa3 0xa4 0xa5 0xa6 0xa7 0xa8 0xa9 0xaa 0xab 0xac 0xad 0xae 0xaf 0xb0}"])

        # test partial registers
        self.match("register read w0",
                   ["w0 = 0x04030201"])
        self.runCmd("register write w0 0xfffefdfc")
        self.match("register read x0",
                   ["x0 = 0x08070605fffefdfc"])

        self.match("register read w1",
                   ["w1 = 0x14131211"])
        self.runCmd("register write w1 0xefeeedec")
        self.match("register read x1",
                   ["x1 = 0x18171615efeeedec"])

        self.match("register read w30",
                   ["w30 = 0x44434241"])
        self.runCmd("register write w30 0xdfdedddc")
        self.match("register read x30",
                   ["x30 = 0x48474645dfdedddc"])

        self.match("register read w31",
                   ["w31 = 0x54535251"])
        self.runCmd("register write w31 0xcfcecdcc")
        self.match("register read x31",
                   ["sp = 0x58575655cfcecdcc"])

        # test FPU registers (overlapping with vector registers)
        self.runCmd("register write d0 16")
        self.match("register read v0",
                   ["v0 = {0x00 0x00 0x00 0x00 0x00 0x00 0x30 0x40 0x89 0x8a 0x8b 0x8c 0x8d 0x8e 0x8f 0x90}"])
        self.runCmd("register write v31 '{0x00 0x00 0x00 0x00 0x00 0x00 0x50 0x40 0xff 0xff 0xff 0xff 0xff 0xff 0xff 0xff}'")
        self.match("register read d31",
                   ["d31 = 64"])

        self.runCmd("register write s0 32")
        self.match("register read v0",
                   ["v0 = {0x00 0x00 0x00 0x42 0x00 0x00 0x30 0x40 0x89 0x8a 0x8b 0x8c 0x8d 0x8e 0x8f 0x90}"])
        self.runCmd("register write v31 '{0x00 0x00 0x00 0x43 0xff 0xff 0xff 0xff 0xff 0xff 0xff 0xff 0xff 0xff 0xff 0xff}'")
        self.match("register read s31",
                   ["s31 = 128"])
