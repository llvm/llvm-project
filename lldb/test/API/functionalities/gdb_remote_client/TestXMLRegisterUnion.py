"""Test unions from GDB remote target description XML."""

from textwrap import dedent

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase
from lldbsuite.test.lldbtest import *


class TestXMLRegisterUnion(GDBRemoteTestBase):
    def setup_multidoc_test(self, docs, register_data):
        self.server.responder = MockGDBServerXMLResponder(docs, register_data)
        target = self.dbg.CreateTarget("")
        process = self.connect(target)
        lldbutil.expect_state_changes(
            self, self.dbg.GetListener(), process, [lldb.eStateStopped]
        )
        return process

    def setup_register_test(self, definitions, register_data):
        return self.setup_multidoc_test(
            {
                "target.xml": dedent(
                    """\
                <?xml version="1.0"?>
                <target version="1.0">
                  <architecture>aarch64</architecture>
                  <feature name="test.register.unions">
                    {}
                  </feature>
                </target>"""
                ).format(definitions)
            },
            register_data,
        )

    def assert_union_info(self, name, byte_size, members):
        self.expect(
            "register info {}".format(name),
            substrs=["Size: {} bytes".format(byte_size), "Union members:"] + members,
        )

    def assert_no_union_info(self, name):
        self.expect("register info {}".format(name), substrs=["Name: {}".format(name)])
        self.expect(
            "register info {}".format(name),
            matching=False,
            substrs=["Union members:"],
        )

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_direct_and_nested_union_metadata(self):
        self.setup_register_test(
            """\
            <vector id="v2f" type="ieee_single" count="2"/>
            <union id="views">
              <field name="f32" type="ieee_single"/>
              <field name="f64" type="ieee_double"/>
              <field name="lanes" type="v2f"/>
            </union>
            <union id="nested">
              <field name="view" type="views"/>
              <field name="raw" type="uint64"/>
            </union>
            <reg name="direct" regnum="0" bitsize="64" type="views"/>
            <reg name="nested" regnum="1" bitsize="64" type="nested"/>
            <reg name="larger" regnum="2" bitsize="128" type="views"/>
            <reg name="pc" bitsize="64"/>""",
            "00" * 40,
        )

        self.assert_union_info(
            "direct",
            8,
            [
                "f32 (ieee_single, 4 bytes)",
                "f64 (ieee_double, 8 bytes)",
                "lanes (v2f, 8 bytes)",
            ],
        )
        self.assert_union_info(
            "nested", 8, ["view (views, 8 bytes)", "raw (uint64, 8 bytes)"]
        )
        self.assert_union_info(
            "larger",
            16,
            [
                "f32 (ieee_single, 4 bytes)",
                "f64 (ieee_double, 8 bytes)",
                "lanes (v2f, 8 bytes)",
            ],
        )

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_vector_of_union_metadata(self):
        self.setup_register_test(
            """\
            <union id="scalar32_views">
              <field name="f32" type="ieee_single"/>
              <field name="u32" type="uint32"/>
            </union>
            <vector id="v4views" type="scalar32_views" count="4"/>
            <union id="pointer_view">
              <field name="pointer" type="data_ptr"/>
            </union>
            <vector id="v2p" type="pointer_view" count="2"/>
            <reg name="u0" regnum="0" bitsize="32" type="scalar32_views"/>
            <reg name="v0" regnum="1" bitsize="128" type="v4views"/>
            <reg name="p0" regnum="2" bitsize="128" type="v2p"/>
            <reg name="pc" bitsize="64"/>""",
            "00" * 44,
        )

        self.assert_union_info(
            "u0", 4, ["f32 (ieee_single, 4 bytes)", "u32 (uint32, 4 bytes)"]
        )
        self.expect(
            "register info v0",
            substrs=["Size: 16 bytes", "Vector elements: 4"],
        )
        self.expect(
            "register info p0",
            substrs=["Size: 16 bytes", "Vector elements: 2"],
        )

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_invalid_unions_are_ignored(self):
        self.setup_register_test(
            """\
            <enum id="enumeration" size="4">
              <evalue name="zero" value="0"/>
            </enum>
            <flags id="flags" size="4">
              <field name="bit" start="0" end="0"/>
            </flags>
            <union>
              <field name="value" type="uint64"/>
            </union>
            <union id="empty"/>
            <union id="missing_field_name">
              <field type="uint64"/>
            </union>
            <union id="missing_field_type">
              <field name="value"/>
            </union>
            <union id="unknown_field_type">
              <field name="value" type="not_defined"/>
            </union>
            <union id="partial_invalid_field">
              <field name="valid" type="uint64"/>
              <field name="invalid" type="not_defined"/>
            </union>
            <union id="forward_field_type">
              <field name="value" type="later"/>
            </union>
            <union id="enum_field_type">
              <field name="value" type="enumeration"/>
            </union>
            <union id="flags_field_type">
              <field name="value" type="flags"/>
            </union>
            <union id="later">
              <field name="value" type="uint64"/>
            </union>
            <union id="wide">
              <field name="value" type="uint64"/>
            </union>
            <reg name="empty" regnum="0" bitsize="64" type="empty"/>
            <reg name="missing_field_name" regnum="1" bitsize="64"
                 type="missing_field_name"/>
            <reg name="missing_field_type" regnum="2" bitsize="64"
                 type="missing_field_type"/>
            <reg name="unknown_field_type" regnum="3" bitsize="64"
                 type="unknown_field_type"/>
            <reg name="forward_field_type" regnum="4" bitsize="64"
                 type="forward_field_type"/>
            <reg name="enum_field_type" regnum="5" bitsize="32"
                 type="enum_field_type"/>
            <reg name="flags_field_type" regnum="6" bitsize="32"
                 type="flags_field_type"/>
            <reg name="later" regnum="7" bitsize="64" type="later"/>
            <reg name="wrong_size" regnum="8" bitsize="32" type="wide"/>
            <reg name="partial_invalid_field" regnum="9" bitsize="64"
                 type="partial_invalid_field"/>
            <reg name="pc" bitsize="64"/>""",
            "00" * 76,
        )

        for name in [
            "empty",
            "missing_field_name",
            "missing_field_type",
            "unknown_field_type",
            "partial_invalid_field",
            "forward_field_type",
            "enum_field_type",
            "flags_field_type",
            "wrong_size",
        ]:
            self.assert_no_union_info(name)
        self.assert_union_info("later", 8, ["value (uint64, 8 bytes)"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_duplicate_union_id_uses_first_definition(self):
        self.setup_register_test(
            """\
            <union id="shared">
              <field name="first" type="uint64"/>
            </union>
            <union id="shared">
              <field name="second" type="ieee_double"/>
            </union>
            <reg name="u0" regnum="0" bitsize="64" type="shared"/>
            <reg name="pc" bitsize="64"/>""",
            "00" * 16,
        )

        self.assert_union_info("u0", 8, ["first (uint64, 8 bytes)"])
        self.expect("register info u0", matching=False, substrs=["second"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_union_ids_are_scoped_to_included_feature(self):
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
                  <union id="shared">
                    <field name="first" type="uint64"/>
                  </union>
                  <reg name="first" regnum="0" bitsize="64" type="shared"/>
                </feature>"""
                ),
                "second.xml": dedent(
                    """\
                <?xml version="1.0"?>
                <feature name="feature.second">
                  <union id="shared">
                    <field name="second" type="uint32"/>
                  </union>
                  <reg name="second" regnum="1" bitsize="32" type="shared"/>
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
            "00" * 28,
        )

        self.assert_union_info("first", 8, ["first (uint64, 8 bytes)"])
        self.assert_union_info("second", 4, ["second (uint32, 4 bytes)"])
        self.assert_no_union_info("unresolved")
