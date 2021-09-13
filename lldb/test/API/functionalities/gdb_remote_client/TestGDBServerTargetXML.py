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
