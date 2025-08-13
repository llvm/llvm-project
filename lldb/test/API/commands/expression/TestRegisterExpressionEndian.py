""" Check that registers written to memory for expression evaluation are
    written using the target's endian not the host's.
"""

from enum import Enum
from textwrap import dedent
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


class Endian(Enum):
    BIG = 0
    LITTLE = 1


class Responder(MockGDBServerResponder):
    def __init__(self, doc, endian):
        super().__init__()
        self.target_xml = doc
        self.endian = endian

    def qXferRead(self, obj, annex, offset, length):
        if annex == "target.xml":
            return self.target_xml, False
        return (None,)

    def readRegister(self, regnum):
        return "E01"

    def readRegisters(self):
        # 64 bit pc value.
        data = ["00", "00", "00", "00", "00", "00", "12", "34"]
        if self.endian == Endian.LITTLE:
            data.reverse()
        return "".join(data)


class TestXMLRegisterFlags(GDBRemoteTestBase):
    def do_endian_test(self, endian):
        architecture, pc_reg_name = {
            Endian.BIG: ("s390x", "pswa"),
            Endian.LITTLE: ("aarch64", "pc"),
        }[endian]

        self.server.responder = Responder(
            dedent(
                f"""\
            <?xml version="1.0"?>
              <target version="1.0">
                <architecture>{architecture}</architecture>
                <feature>
                  <reg name="{pc_reg_name}" bitsize="64"/>
                </feature>
            </target>"""
            ),
            endian,
        )
        target = self.dbg.CreateTarget("")
        process = self.connect(target)
        lldbutil.expect_state_changes(
            self, self.dbg.GetListener(), process, [lldb.eStateStopped]
        )

        # If expressions convert register values into target endian, the
        # result of register read and expr should be the same.
        pc_value = "0x0000000000001234"
        self.expect(
            "register read pc",
            substrs=[pc_value],
        )
        self.expect("expr --format hex -- $pc", substrs=[pc_value])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_little_endian_target(self):
        self.do_endian_test(Endian.LITTLE)

    @skipIfXmlSupportMissing
    @skipIfRemote
    # Unlike AArch64, we do need the backend present for this test to work.
    @skipIfLLVMTargetMissing("SystemZ")
    def test_big_endian_target(self):
        self.do_endian_test(Endian.BIG)
