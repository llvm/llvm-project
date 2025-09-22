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
        architecture, pc_reg_name, yaml_file, data, machine = {
            Endian.BIG: ("s390x", "pswa", "s390x.yaml", "ELFDATA2MSB", "EM_S390"),
            Endian.LITTLE: (
                "aarch64",
                "pc",
                "aarch64.yaml",
                "ELFDATA2LSB",
                "EM_AARCH64",
            ),
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

        # We need to have a program file, so that we have a full type system,
        # so that we can do the casts later.
        obj_path = self.getBuildArtifact("main.o")
        yaml_path = self.getBuildArtifact(yaml_file)
        with open(yaml_path, "w") as f:
            f.write(
                dedent(
                    f"""\
                --- !ELF
                FileHeader:
                  Class:    ELFCLASS64
                  Data:     {data}
                  Type:     ET_REL
                  Machine:  {machine}
                ...
                """
                )
            )
        self.yaml2obj(yaml_path, obj_path)
        target = self.dbg.CreateTarget(obj_path)

        process = self.connect(target)
        lldbutil.expect_state_changes(
            self, self.dbg.GetListener(), process, [lldb.eStateStopped]
        )

        # If expressions convert register values into target endian, the
        # result of register read, expr and casts should be the same.
        pc_value = "0x0000000000001234"
        self.expect(
            "register read pc",
            substrs=[pc_value],
        )
        self.expect("expr --format hex -- $pc", substrs=[pc_value])

        pc = (
            process.thread[0]
            .frame[0]
            .GetRegisters()
            .GetValueAtIndex(0)
            .GetChildMemberWithName("pc")
        )
        ull = target.FindTypes("unsigned long long").GetTypeAtIndex(0)
        pc_ull = pc.Cast(ull)

        self.assertEqual(pc.GetValue(), pc_ull.GetValue())
        self.assertEqual(pc.GetValueAsAddress(), pc_ull.GetValueAsAddress())
        self.assertEqual(pc.GetValueAsSigned(), pc_ull.GetValueAsSigned())
        self.assertEqual(pc.GetValueAsUnsigned(), pc_ull.GetValueAsUnsigned())

    @skipIfXmlSupportMissing
    @skipIfRemote
    @skipIfLLVMTargetMissing("AArch64")
    def test_little_endian_target(self):
        self.do_endian_test(Endian.LITTLE)

    @skipIfXmlSupportMissing
    @skipIfRemote
    @skipIfLLVMTargetMissing("SystemZ")
    def test_big_endian_target(self):
        self.do_endian_test(Endian.BIG)
