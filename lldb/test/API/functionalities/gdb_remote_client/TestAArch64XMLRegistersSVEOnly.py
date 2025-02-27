""" Check that when a debug server provides XML that only defines SVE Z registers,
    and does not include Neon V registers, lldb creates sub-registers to represent
    the V registers as the bottom 128 bits of the Z registers.

    qemu-aarch64 is one such debug server.

    This also doubles as a test that lldb has a fallback path for registers of
    unknown type that are > 128 bits, as the SVE registers are here.
"""

from textwrap import dedent
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


class Responder(MockGDBServerResponder):
    def __init__(self):
        super().__init__()
        self.vg = 4
        self.pc = 0xA0A0A0A0A0A0A0A0

    def qXferRead(self, obj, annex, offset, length):
        if annex == "target.xml":
            # Note that QEMU sends the current SVE size in XML and the debugger
            # then reads vg to know the latest size.
            return (
                dedent(
                    """\
              <?xml version="1.0"?>
              <target version="1.0">
                <architecture>aarch64</architecture>
                <feature name="org.gnu.gdb.aarch64.core">
                  <reg name="pc" regnum="0" bitsize="64"/>
                  <reg name="vg" regnum="1" bitsize="64"/>
                  <reg name="z0" regnum="2" bitsize="2048" type="not_a_type"/>
                </feature>
              </target>"""
                ),
                False,
            )

        return (None,)

    def readRegister(self, regnum):
        return "E01"

    def readRegisters(self):
        return "".join(
            [
                # 64 bit PC.
                f"{self.pc:x}",
                # 64 bit vg
                f"0{self.vg}00000000000000",
                # Enough data for 256 and 512 bit SVE.
                "".join([f"{n:02x}" * 4 for n in range(1, 17)]),
            ]
        )

    def cont(self):
        # vg is expedited so that lldb can resize the SVE registers.
        return f"T02thread:1ff0d;threads:1ff0d;thread-pcs:{self.pc};01:0{self.vg}00000000000000;"

    def writeRegisters(self, registers_hex):
        # We get a block of data containing values in regnum order.
        self.vg = int(registers_hex[16:18])
        return "OK"


class TestXMLRegisterFlags(GDBRemoteTestBase):
    def check_regs(self, vg):
        # Each 32 bit chunk repeats n.
        z0_value = " ".join(
            [" ".join([f"0x{n:02x}"] * 4) for n in range(1, (vg * 2) + 1)]
        )

        self.expect(
            "register read vg z0 v0 s0 d0",
            substrs=[
                f"      vg = 0x000000000000000{vg}\n"
                "      z0 = {" + z0_value + "}\n"
                "      v0 = {0x01 0x01 0x01 0x01 0x02 0x02 0x02 0x02 0x03 0x03 0x03 0x03 0x04 0x04 0x04 0x04}\n"
                "      s0 = 2.36942783E-38\n"
                "      d0 = 5.3779407333977203E-299\n"
            ],
        )

        self.expect("register read s0 --format uint32", substrs=["s0 = {0x01010101}"])
        self.expect(
            "register read d0 --format uint64",
            substrs=["d0 = {0x0202020201010101}"],
        )

    @skipIfXmlSupportMissing
    @skipIfRemote
    @skipIfLLVMTargetMissing("AArch64")
    def test_v_sub_registers(self):
        self.server.responder = Responder()
        target = self.dbg.CreateTarget("")

        if self.TraceOn():
            self.runCmd("log enable gdb-remote packets")
            self.addTearDownHook(lambda: self.runCmd("log disable gdb-remote packets"))

        process = self.connect(target)
        lldbutil.expect_state_changes(
            self, self.dbg.GetListener(), process, [lldb.eStateStopped]
        )

        self.check_regs(4)

        # Now increase the SVE length and continue. The mock will respond with a new
        # vg and lldb will reconfigure the register defs. This should not break the
        # sub-registers.

        self.runCmd("register write vg 8")
        self.expect("continue", substrs=["stop reason = signal SIGINT"])

        self.check_regs(8)
