"""Test vector metadata from GDB remote target description XML."""

from textwrap import dedent

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase
from lldbsuite.test.lldbtest import *


class MultiDocResponder(MockGDBServerResponder):
    def __init__(self, docs, register_data):
        super().__init__()
        self.docs = docs
        self.register_data = register_data

    def qXferRead(self, obj, annex, offset, length):
        try:
            return self.docs[annex], False
        except KeyError:
            return (None,)

    def readRegister(self, regnum):
        return "E01"

    def readRegisters(self):
        return self.register_data


class TestXMLRegisterVector(GDBRemoteTestBase):
    def setup_multidoc_test(self, docs, register_data):
        self.server.responder = MultiDocResponder(docs, register_data)
        target = self.dbg.CreateTarget("")

        if self.TraceOn():
            self.runCmd("log enable gdb-remote packets process")
            self.addTearDownHook(
                lambda: self.runCmd("log disable gdb-remote packets process")
            )

        process = self.connect(target)
        lldbutil.expect_state_changes(
            self, self.dbg.GetListener(), process, [lldb.eStateStopped]
        )

    def setup_register_test(self, definitions, register_data):
        self.setup_multidoc_test(
            {
                "target.xml": dedent(
                    """\
                <?xml version="1.0"?>
                <target version="1.0">
                  <architecture>aarch64</architecture>
                  <feature name="test.register.vectors">
                    {}
                  </feature>
                </target>"""
                ).format(definitions)
            },
            register_data,
        )

    def assert_vector_info(self, name, byte_size, count):
        self.expect(
            "register info {}".format(name),
            substrs=[
                "Size: {} bytes".format(byte_size),
                "Vector elements: {}".format(count),
            ],
        )

    def assert_no_vector_info(self, name):
        self.expect("register info {}".format(name), substrs=["Name: {}".format(name)])
        self.expect(
            "register info {}".format(name),
            matching=False,
            substrs=["Vector elements:"],
        )

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_direct_and_nested_vector_metadata(self):
        self.setup_register_test(
            """\
            <vector id="v2f" type="ieee_single" count="2"/>
            <vector id="v3v2f" type="v2f" count="3"/>
            <reg name="direct" regnum="0" bitsize="64" type="v2f"/>
            <reg name="nested" regnum="1" bitsize="192" type="v3v2f"/>
            <reg name="pc" bitsize="64"/>""",
            "00" * 40,
        )

        self.assert_vector_info("direct", 8, 2)
        self.assert_vector_info("nested", 24, 3)

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_invalid_vectors_are_ignored(self):
        self.setup_register_test(
            """\
            <vector id="bad_count" type="ieee_single" count="nope"/>
            <vector id="zero" type="ieee_single" count="0"/>
            <vector id="too_many" type="ieee_single" count="65537"/>
            <vector id="unknown" type="missing" count="2"/>
            <vector id="forward" type="later" count="1"/>
            <vector id="missing_type" count="2"/>
            <vector id="missing_count" type="ieee_single"/>
            <vector id="too_large" type="uint128" count="4097"/>
            <vector type="ieee_single" count="2"/>
            <vector id="later" type="ieee_single" count="2"/>
            <enum id="enumeration" size="4">
              <evalue name="zero" value="0"/>
            </enum>
            <flags id="flags" size="4">
              <field name="bit" start="0" end="0"/>
            </flags>
            <vector id="enum_vector" type="enumeration" count="2"/>
            <vector id="flags_vector" type="flags" count="2"/>
            <vector id="bad_pointer_size" type="data_ptr" count="3"/>
            <vector id="pointer_pair" type="data_ptr" count="2"/>
            <vector id="bad_nested_pointer_size" type="pointer_pair" count="3"/>
            <vector id="v4f" type="ieee_single" count="4"/>
            <reg name="bad_count" regnum="0" bitsize="64" type="bad_count"/>
            <reg name="zero" regnum="1" bitsize="64" type="zero"/>
            <reg name="too_many" regnum="2" bitsize="64" type="too_many"/>
            <reg name="unknown" regnum="3" bitsize="64" type="unknown"/>
            <reg name="forward" regnum="4" bitsize="64" type="forward"/>
            <reg name="missing_type" regnum="5" bitsize="64"
                 type="missing_type"/>
            <reg name="missing_count" regnum="6" bitsize="64"
                 type="missing_count"/>
            <reg name="too_large" regnum="7" bitsize="64" type="too_large"/>
            <reg name="later" regnum="8" bitsize="64" type="later"/>
            <reg name="enum_vector" regnum="9" bitsize="64"
                 type="enum_vector"/>
            <reg name="flags_vector" regnum="10" bitsize="64"
                 type="flags_vector"/>
            <reg name="bad_size" regnum="11" bitsize="64" type="v4f"/>
            <reg name="bad_pointer_size" regnum="12" bitsize="80"
                 type="bad_pointer_size"/>
            <reg name="bad_nested_pointer_size" regnum="13" bitsize="120"
                 type="bad_nested_pointer_size"/>
            <reg name="pc" bitsize="64"/>""",
            "00" * 129,
        )

        for name in [
            "bad_count",
            "zero",
            "too_many",
            "unknown",
            "forward",
            "missing_type",
            "missing_count",
            "too_large",
            "enum_vector",
            "flags_vector",
            "bad_size",
            "bad_pointer_size",
            "bad_nested_pointer_size",
        ]:
            self.assert_no_vector_info(name)

        self.assert_vector_info("later", 8, 2)

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_duplicate_vector_id_uses_first_definition(self):
        self.setup_register_test(
            """\
            <vector id="shared" type="ieee_single" count="2"/>
            <vector id="shared" type="uint16" count="4"/>
            <reg name="v0" regnum="0" bitsize="64" type="shared"/>
            <reg name="pc" bitsize="64"/>""",
            "00" * 16,
        )

        self.assert_vector_info("v0", 8, 2)
        self.expect("register info v0", matching=False, substrs=["Vector elements: 4"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_target_type_precedes_builtin_type(self):
        self.setup_register_test(
            """\
            <vector id="ieee_single" type="uint8" count="2"/>
            <vector id="nested" type="ieee_single" count="3"/>
            <reg name="v0" regnum="0" bitsize="48" type="nested"/>
            <reg name="pc" bitsize="64"/>""",
            "00" * 14,
        )

        self.assert_vector_info("v0", 6, 3)

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_vector_ids_are_scoped_to_included_feature(self):
        self.setup_multidoc_test(
            {
                "target.xml": dedent(
                    """\
                <?xml version="1.0"?>
                <target version="1.0">
                  <architecture>aarch64</architecture>
                  <xi:include href="first.xml"/>
                  <xi:include href="second.xml"/>
                  <xi:include href="third.xml"/>
                </target>"""
                ),
                "first.xml": dedent(
                    """\
                <?xml version="1.0"?>
                <feature name="feature.first">
                  <vector id="shared" type="ieee_single" count="2"/>
                  <reg name="first" regnum="0" bitsize="64" type="shared"/>
                </feature>"""
                ),
                "second.xml": dedent(
                    """\
                <?xml version="1.0"?>
                <feature name="feature.second">
                  <vector id="shared" type="ieee_single" count="4"/>
                  <reg name="second" regnum="1" bitsize="128" type="shared"/>
                </feature>"""
                ),
                "third.xml": dedent(
                    """\
                <?xml version="1.0"?>
                <feature name="feature.third">
                  <reg name="unresolved" regnum="2" bitsize="64" type="shared"/>
                  <reg name="pc" bitsize="64"/>
                </feature>"""
                ),
            },
            "00" * 40,
        )

        self.assert_vector_info("first", 8, 2)
        self.assert_vector_info("second", 16, 4)
        self.assert_no_vector_info("unresolved")
