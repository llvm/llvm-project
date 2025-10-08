""" Check that registers written to memory for expression evaluation are
    written using the target's endian not the host's.
"""

from dataclasses import dataclass
from enum import Enum
from textwrap import dedent
import lldb
from lldbsuite.test.lldbtest import lldbutil
from lldbsuite.test.decorators import (
    skipIfXmlSupportMissing,
    skipIfRemote,
    skipIfLLVMTargetMissing,
)
from lldbsuite.test.gdbclientutils import MockGDBServerResponder
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


class Endian(Enum):
    BIG = 0
    LITTLE = 1


class ElementOrder(Enum):
    # Memory is laid out as [0, 1, ... N-1]
    ZEROFIRST = 0
    # Memory is laid out as [N-1, N-2, ..., 0]
    LASTFIRST = 1


@dataclass
class Config:
    architecture: str
    pc_reg_name: str
    yaml_file: str
    data: str
    machine: str
    endian: Endian
    element_order: ElementOrder


class Responder(MockGDBServerResponder):
    def __init__(self, doc: str, endian: Endian, element_order: ElementOrder):
        super().__init__()
        self.target_xml = doc
        self.endian = endian
        self.element_order = element_order

    def qXferRead(self, obj, annex, offset, length) -> tuple[str | None, bool]:
        if obj == 'features' and annex == "target.xml":
            more = offset + length < len(self.target_xml)
            return self.target_xml[offset:offset+length], more
        return (None, False)

    def readRegister(self, register: int) -> str:
        _ = register # Silence unused parameter hint
        return "E01"

    def readRegisters(self) -> str:
        # 64 bit pc value.
        data = ["00", "00", "00", "00", "00", "00", "12", "34"]
        if self.endian == Endian.LITTLE:
            data.reverse()
        return "".join(data)



class TestXMLRegisterFlags(GDBRemoteTestBase):
    def do_expr_eval(self, config_name: str):
        cfg = {
            # AArch64 stores elements in little-endian, zero-first order.
            "aarch64-le": Config(
                architecture="aarch64",
                pc_reg_name="pc",
                yaml_file="aarch64.yaml",
                data="ELFDATA2LSB",
                machine="EM_AARCH64",
                endian=Endian.LITTLE,
                element_order=ElementOrder.ZEROFIRST,
            ),
            # AArch64 stores elements in big-endian but the vector remains in
            # the same zero-first order.
            "aarch64-be": Config(
                architecture="aarch64_be",
                pc_reg_name="pc",
                yaml_file="aarch64be.yaml",
                data="ELFDATA2MSB",
                machine="EM_AARCH64",
                endian=Endian.BIG,
                element_order=ElementOrder.ZEROFIRST,
            ),
            # PowerPC stores the whole vector as little-endian
            "ppc-le": Config(
                architecture="ppc",
                pc_reg_name="pc",
                yaml_file="ppc.yaml",
                data="ELFDATA2LSB",
                machine="EM_PPC64",
                endian=Endian.LITTLE,
                element_order=ElementOrder.ZEROFIRST,
            ),
            # PowerPC stores the whole vector as big-endian which reverses
            # element order
            "ppc-be": Config(
                architecture="ppc",
                pc_reg_name="pc",
                yaml_file="ppcbe.yaml",
                data="ELFDATA2MSB",
                machine="EM_PPC64",
                endian=Endian.BIG,
                element_order=ElementOrder.LASTFIRST,
            ),
            # Vectors are stored in the same element order as arrays.
            # Note: WebAssembly loads vectors in reverse order because it
            #       requires that vectors behave like little-endian first-zero
            #       even when reinterpreting memory as a vector with different
            #       element sizes.
            #       C++ on the other hand effectively shuffles the bytes like
            #       AArch64 Big Endian does
            "systemz-be": Config(
                architecture="s390x",
                pc_reg_name="pswa",
                yaml_file="s390x.yaml",
                data="ELFDATA2MSB",
                machine="EM_S390",
                endian=Endian.BIG,
                element_order=ElementOrder.ZEROFIRST,
            ),
        }[config_name]

        assert self.server is not None
        self.server.responder = Responder(
            dedent(
                f"""\
            <?xml version="1.0"?>
              <target version="1.0">
                <architecture>{cfg.architecture}</architecture>
                <feature>
                  <reg name="{cfg.pc_reg_name}" bitsize="64"/>
                </feature>
            </target>"""
            ),
            cfg.endian,
            cfg.element_order,
        )

        # We need to have a program file, so that we have a full type system,
        # so that we can do the casts later.
        obj_path = self.getBuildArtifact("main.o")
        yaml_path = self.getBuildArtifact(cfg.yaml_file)
        with open(yaml_path, "w") as f:
            f.write(
                dedent(
                    f"""\
                --- !ELF
                FileHeader:
                  Class:    ELFCLASS64
                  Data:     {cfg.data}
                  Type:     ET_REL
                  Machine:  {cfg.machine}
                DWARF:
                  debug_abbrev:
                    - Table:
                        - Code:            1
                          Tag:             DW_TAG_compile_unit
                          Children:        DW_CHILDREN_yes
                          Attributes:
                            - Attribute:   DW_AT_language
                              Form:        DW_FORM_data2
                        - Code:            2
                          Tag:             DW_TAG_typedef
                          Children:        DW_CHILDREN_no
                          Attributes:
                            - Attribute:   DW_AT_type
                              Form:        DW_FORM_ref4
                            - Attribute:   DW_AT_name
                              Form:        DW_FORM_string
                        - Code:            3
                          Tag:             DW_TAG_array_type
                          Children:        DW_CHILDREN_yes
                          Attributes:
                            - Attribute:   DW_AT_GNU_vector
                              Form:        DW_FORM_flag_present
                            - Attribute:   DW_AT_type
                              Form:        DW_FORM_ref4
                        - Code:            4
                          Tag:             DW_TAG_subrange_type
                          Children:        DW_CHILDREN_no
                          Attributes:
                            - Attribute:   DW_AT_count
                              Form:        DW_FORM_data1
                        - Code:            5
                          Tag:             DW_TAG_base_type
                          Children:        DW_CHILDREN_no
                          Attributes:
                            - Attribute:   DW_AT_name
                              Form:        DW_FORM_string
                            - Attribute:   DW_AT_encoding
                              Form:        DW_FORM_data1
                            - Attribute:   DW_AT_byte_size
                              Form:        DW_FORM_data1
                  debug_info:
                    - Version:           4
                      AbbrevTableID:     0
                      AbbrOffset:        0x0
                      AddrSize:          8
                      Entries:
                        - AbbrCode:      1
                          Values:
                            - Value:     0x0004  # DW_LANG_C_plus_plus
                        - AbbrCode:      2  # typedef
                          Values:
                            - Value:     27  # DW_AT_type: Reference to array type at 0x1b
                            - CStr:      v4float  # DW_AT_name
                        - AbbrCode:      3  # array_type
                          Values:
                            - Value:     0x01  # DW_AT_GNU_vector (flag_present)
                            - Value:     35  # DW_AT_type: Reference to float at 0x23
                        - AbbrCode:      4  # subrange_type
                          Values:
                            - Value:     0x04  # DW_AT_count: 4 elements
                        - AbbrCode:      0  # End of array_type children
                        - AbbrCode:      5  # base_type (float)
                          Values:
                            - CStr:      float  # DW_AT_name
                            - Value:     0x04  # DW_AT_encoding: DW_ATE_float
                            - Value:     0x04  # DW_AT_byte_size: 4 bytes
                        - AbbrCode:      0  # End of compile unit
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

        # Enable logging to debug type lookup and expression evaluation
        self.TraceOn()
        log_file = self.getBuildArtifact("lldb.log")
        self.runCmd(f"log enable lldb types expr -f {log_file}")
        self.runCmd("image dump symtab", check=False)
        self.runCmd("image lookup -t v4float", check=False)
        self.runCmd("image lookup -t float", check=False)

        # If expressions convert register values into target endian, the
        # vector should be stored correctly in memory.
        self.expect("expr --language c++ -- (v4float){0.25, 0.5, 0.75, 1.0}", substrs=["0.25", "0.5", "0.75", "1"])

        # Check the raw bytes to verify endianness
        result = self.frame().EvaluateExpression("(v4float){0.25, 0.5, 0.75, 1.0}", lldb.eDynamicCanRunTarget)
        self.assertTrue(result.IsValid())
        error = lldb.SBError()
        data = result.GetData()
        bytes_list = [data.GetUnsignedInt8(error, i) for i in range(16)]
        # For big-endian: 0x3e800000, 0x3f000000, 0x3f400000, 0x3f800000
        # For little-endian: bytes are reversed within each float
        expected_big = [0x3e, 0x80, 0x00, 0x00, 0x3f, 0x00, 0x00, 0x00, 0x3f, 0x40, 0x00, 0x00, 0x3f, 0x80, 0x00, 0x00]
        expected_little = [0x00, 0x00, 0x80, 0x3e, 0x00, 0x00, 0x00, 0x3f, 0x00, 0x00, 0x40, 0x3f, 0x00, 0x00, 0x80, 0x3f]
        if cfg.endian == Endian.BIG:
            self.assertEqual(bytes_list, expected_big)
        else:
            self.assertEqual(bytes_list, expected_little)

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
    def test_aarch64_little_endian_target(self):
        self.do_expr_eval("aarch64-le")

    # AArch64 doesn't seem to have implemented big-endian in lldb
    # Both big-endian and little-endian triples select the same ArchSpec.
    #@skipIfXmlSupportMissing
    #@skipIfRemote
    #@skipIfLLVMTargetMissing("AArch64")
    #def test_aarch64_big_endian(self):
    #    self.do_expr_eval("aarch64-be")

    @skipIfXmlSupportMissing
    @skipIfRemote
    @skipIfLLVMTargetMissing("PowerPC")
    def test_ppc_little_endian(self):
        self.do_expr_eval("ppc-le")

    @skipIfXmlSupportMissing
    @skipIfRemote
    @skipIfLLVMTargetMissing("PowerPC")
    def test_ppc_big_endian_target(self):
        self.do_expr_eval("ppc-be")

    @skipIfXmlSupportMissing
    @skipIfRemote
    @skipIfLLVMTargetMissing("SystemZ")
    def test_systemz_big_endian_target(self):
        self.do_expr_eval("systemz-be")
