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

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_direct_union_sb_api(self):
        process = self.setup_register_test(
            """\
            <vector id="v4f" type="ieee_single" count="4"/>
            <union id="views">
              <field name="scalar" type="ieee_single"/>
              <field name="lanes" type="v4f"/>
              <field name="raw" type="uint128"/>
            </union>
            <reg name="u0" regnum="0" bitsize="128" type="views"/>
            <reg name="pc" bitsize="64"/>""",
            "0000c03f000020400000604000009040" + "00" * 8,
        )

        frame = process.GetThreadAtIndex(0).GetFrameAtIndex(0)
        union = frame.FindRegister("u0")
        self.assertTrue(union.IsValid())
        self.assertTrue(union.GetType().IsValid())
        self.assertRegex(
            union.GetType().GetName(), r"^__lldb_register_union_[0-9]+_16$"
        )
        self.assertEqual(union.GetByteSize(), 16)
        self.assertEqual(union.GetNumChildren(), 3)
        self.assertEqual(
            [union.GetChildAtIndex(i).GetName() for i in range(3)],
            ["scalar", "lanes", "raw"],
        )
        self.assertAlmostEqual(
            union.GetChildMemberWithName("scalar").GetData().float[0], 1.5
        )
        self.assertAlmostEqual(
            union.GetValueForExpressionPath(".lanes[2]").GetData().float[0],
            3.5,
        )

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_union_smaller_than_register_sb_api(self):
        process = self.setup_register_test(
            """\
            <union id="small">
              <field name="value" type="uint32"/>
            </union>
            <reg name="u0" regnum="0" bitsize="128" type="small"/>
            <reg name="pc" bitsize="64"/>""",
            "2a000000" + "00" * 20,
        )

        union = process.GetThreadAtIndex(0).GetFrameAtIndex(0).FindRegister("u0")
        self.assertRegex(union.GetType().GetName(), r"^__lldb_register_union_[0-9]+_4$")
        self.assertEqual(union.GetByteSize(), 16)
        self.assertEqual(union.GetType().GetByteSize(), 4)
        self.assertEqual(union.GetChildMemberWithName("value").GetValueAsUnsigned(), 42)

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_nested_union_and_vector_sb_api(self):
        process = self.setup_register_test(
            """\
            <vector id="v4f" type="ieee_single" count="4"/>
            <union id="float_views">
              <field name="scalar" type="ieee_single"/>
              <field name="lanes" type="v4f"/>
            </union>
            <union id="nested">
              <field name="f32_view" type="float_views"/>
              <field name="raw" type="uint128"/>
            </union>
            <reg name="n0" regnum="0" bitsize="128" type="nested"/>
            <reg name="pc" bitsize="64"/>""",
            "0000c03f000020400000604000009040" + "00" * 8,
        )

        frame = process.GetThreadAtIndex(0).GetFrameAtIndex(0)
        union = frame.FindRegister("n0")
        self.assertEqual(union.GetNumChildren(), 2)
        lane = union.GetValueForExpressionPath(".f32_view.lanes[3]")
        self.assertTrue(lane.IsValid())
        self.assertAlmostEqual(lane.GetData().float[0], 4.5)

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_byte_view_and_vector_of_unions_sb_api(self):
        process = self.setup_register_test(
            """\
            <vector id="c32" type="uint8" count="32"/>
            <union id="bytes_view">
              <field name="c" type="c32"/>
            </union>
            <union id="scalar32_views">
              <field name="f32" type="ieee_single"/>
              <field name="u32" type="uint32"/>
            </union>
            <vector id="v4views" type="scalar32_views" count="4"/>
            <reg name="b0" regnum="0" bitsize="256" type="bytes_view"/>
            <reg name="vu0" regnum="1" bitsize="128" type="v4views"/>
            <reg name="pc" bitsize="64"/>""",
            bytes(range(32)).hex() + "0000c03f000020400000604000009040" + "00" * 8,
        )

        frame = process.GetThreadAtIndex(0).GetFrameAtIndex(0)
        byte_view = frame.FindRegister("b0").GetChildMemberWithName("c")
        self.assertEqual(byte_view.GetNumChildren(), 32)
        self.assertEqual(byte_view.GetChildAtIndex(16).GetValueAsUnsigned(), 16)

        vector = frame.FindRegister("vu0")
        self.assertEqual(vector.GetNumChildren(), 4)
        third = vector.GetChildAtIndex(2)
        self.assertEqual(
            [third.GetChildAtIndex(i).GetName() for i in range(2)],
            ["f32", "u32"],
        )
        self.assertAlmostEqual(
            third.GetChildMemberWithName("f32").GetData().float[0], 3.5
        )
        self.expect("register read vu0[2].f32", substrs=["vu0[2].f32 = 3.5"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_target_sized_union_vector_sb_api(self):
        process = self.setup_register_test(
            """\
            <union id="pointer_view">
              <field name="pointer" type="data_ptr"/>
            </union>
            <vector id="v2p" type="pointer_view" count="2"/>
            <reg name="v0" regnum="0" bitsize="128" type="v2p"/>
            <reg name="pc" bitsize="64"/>""",
            "34120000000000007856000000000000" + "00" * 8,
        )

        vector = process.GetThreadAtIndex(0).GetFrameAtIndex(0).FindRegister("v0")
        self.assertEqual(vector.GetNumChildren(), 2)
        self.assertEqual(
            vector.GetChildAtIndex(0)
            .GetChildMemberWithName("pointer")
            .GetValueAsUnsigned(),
            0x1234,
        )
        self.assertEqual(
            vector.GetChildAtIndex(1)
            .GetChildMemberWithName("pointer")
            .GetValueAsUnsigned(),
            0x5678,
        )

    @skipIfXmlSupportMissing
    @skipIfRemote
    @skipIfLLVMTargetMissing("SystemZ")
    def test_big_endian_union_sb_api(self):
        process = self.setup_multidoc_test(
            {
                "target.xml": dedent(
                    """\
                <?xml version="1.0"?>
                <target version="1.0">
                  <architecture>s390x</architecture>
                  <feature name="test.register.unions">
                    <vector id="v2f" type="ieee_single" count="2"/>
                    <union id="views">
                      <field name="scalar" type="ieee_single"/>
                      <field name="lanes" type="v2f"/>
                    </union>
                    <reg name="u0" regnum="0" bitsize="64" type="views"/>
                    <union id="small">
                      <field name="value" type="uint32"/>
                    </union>
                    <reg name="small" regnum="1" bitsize="128" type="small"/>
                    <reg name="pswa" regnum="2" bitsize="64"/>
                  </feature>
                </target>"""
                )
            },
            "3fc0000040200000" + "0000002a" + "00" * 20,
        )

        frame = process.GetThreadAtIndex(0).GetFrameAtIndex(0)
        union = frame.FindRegister("u0")
        self.assertAlmostEqual(
            union.GetChildMemberWithName("scalar").GetData().float[0], 1.5
        )
        lanes = union.GetChildMemberWithName("lanes")
        self.assertEqual(
            [lanes.GetChildAtIndex(i).GetData().float[0] for i in range(2)],
            [1.5, 2.5],
        )
        ull = process.GetTarget().GetBasicType(lldb.eBasicTypeUnsignedLongLong)
        self.assertEqual(union.Cast(ull).GetValueAsUnsigned(), 0x3FC0000040200000)

        small = frame.FindRegister("small")
        self.assertEqual(small.GetByteSize(), 16)
        self.assertEqual(small.GetType().GetByteSize(), 4)
        self.assertEqual(small.GetChildMemberWithName("value").GetValueAsUnsigned(), 42)

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_union_cli_summary_and_member_paths(self):
        process = self.setup_register_test(
            """\
            <vector id="v4f" type="ieee_single" count="4"/>
            <union id="views">
              <field name="f32" type="ieee_single"/>
              <field name="f64" type="ieee_double"/>
              <field name="u64" type="uint64"/>
            </union>
            <union id="vector_views">
              <field name="floats" type="v4f"/>
              <field name="raw" type="uint128"/>
            </union>
            <union id="nested">
              <field name="view" type="vector_views"/>
              <field name="raw" type="uint128"/>
            </union>
            <reg name="u0" altname="alt_u0" regnum="0" bitsize="64" type="views"/>
            <reg name="u1" regnum="1" bitsize="128" type="vector_views"/>
            <reg name="n0" regnum="2" bitsize="128" type="nested"/>
            <reg name="pc" bitsize="64"/>""",
            "0000c03fffffffff" + "0000c03f000020400000604000009040" * 2 + "00" * 8,
        )

        union = process.GetThreadAtIndex(0).GetFrameAtIndex(0).FindRegister("u0")
        self.assertEqual(
            union.GetSummary(),
            "(f32 = 1.5, f64 = NaN, u64 = 18446744070484131840)",
        )

        self.expect(
            "register read u0",
            substrs=[
                "u0 = 0xffffffff3fc00000",
                "     = (f32 = 1.5, f64 = NaN, u64 = 18446744070484131840)",
            ],
        )
        self.expect("register read u0.f32", substrs=["u0.f32 = 1.5"])
        self.expect(
            "register read u0.u64 --format X",
            substrs=["u0.u64 = 0xFFFFFFFF3FC00000"],
        )
        self.expect(
            "register read u0 --format X",
            substrs=["u0 = 0xFFFFFFFF3FC00000"],
        )
        self.expect("register read u0 --format X", matching=False, substrs=["f32 ="])
        self.expect("register read -A alt_u0.f32", substrs=["alt_u0.f32 = 1.5"])
        self.expect("register read $u0.f64", substrs=["u0.f64 = NaN"])
        self.expect("register read u1.floats[2]", substrs=["u1.floats[2] = 3.5"])
        self.expect(
            "register read n0.view.floats[3]",
            substrs=["n0.view.floats[3] = 4.5"],
        )

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_exact_dotted_register_name_takes_precedence(self):
        self.setup_register_test(
            """\
            <union id="views">
              <field name="f32" type="ieee_single"/>
              <field name="raw" type="uint64"/>
            </union>
            <reg name="u0" regnum="0" bitsize="64" type="views"/>
            <reg name="u0.f32" regnum="1" bitsize="32"/>
            <reg name="u0.view" regnum="2" bitsize="64" type="views"/>
            <reg name="pc" bitsize="64"/>""",
            "0000c03f00000000" + "2a000000" + "0000c03f00000000" + "00" * 8,
        )

        self.expect("register read u0.f32", substrs=["u0.f32 = 0x0000002a"])
        self.expect("register read u0.view.f32", substrs=["u0.view.f32 = 1.5"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_invalid_union_member_paths(self):
        self.setup_register_test(
            """\
            <union id="views">
              <field name="f32" type="ieee_single"/>
              <field name="raw" type="uint64"/>
            </union>
            <vector id="v2views" type="views" count="2"/>
            <reg name="u0" regnum="0" bitsize="64" type="views"/>
            <reg name="v0" regnum="1" bitsize="128" type="v2views"/>
            <reg name="pc" regnum="2" bitsize="64"/>""",
            "00" * 32,
        )

        invalid_paths = {
            "u0.missing": "No field path 'missing' in register 'u0'",
            "u0.": "No field path '' in register 'u0'",
            "u0..f32": "No field path '.f32' in register 'u0'",
            "u0[0]": "No field path '[0]' in register 'u0'",
            "v0[": "No field path '[' in register 'v0'",
            "v0[0": "No field path '[0' in register 'v0'",
            "v0[]": "No field path '[]' in register 'v0'",
            "v0[x]": "No field path '[x]' in register 'v0'",
            "v0[-1]": "No field path '[-1]' in register 'v0'",
            "v0[4294967296]": "No field path '[4294967296]' in register 'v0'",
            "v0[0]junk": "No field path '[0]junk' in register 'v0'",
            "v0[0].missing": "No field path '[0].missing' in register 'v0'",
        }
        for path, diagnostic in invalid_paths.items():
            self.expect(
                "register read " + path,
                error=True,
                substrs=[diagnostic],
            )
        self.expect(
            "register read v0[9].f32",
            error=True,
            substrs=["No field path '[9].f32'"],
        )
        self.expect(
            "register read pc.field",
            error=True,
            substrs=["Register 'pc' does not have a structured type"],
        )

    @skipIfXmlSupportMissing
    @skipIfRemote
    @skipIfLLVMTargetMissing("SystemZ")
    def test_big_endian_union_cli(self):
        self.setup_multidoc_test(
            {
                "target.xml": dedent(
                    """\
                <?xml version="1.0"?>
                <target version="1.0">
                  <architecture>s390x</architecture>
                  <feature name="test.register.unions">
                    <vector id="v2f" type="ieee_single" count="2"/>
                    <union id="views">
                      <field name="scalar" type="ieee_single"/>
                      <field name="lanes" type="v2f"/>
                    </union>
                    <reg name="u0" regnum="0" bitsize="64" type="views"/>
                    <reg name="pswa" regnum="1" bitsize="64"/>
                  </feature>
                </target>"""
                )
            },
            "3fc0000040200000" + "00" * 8,
        )

        self.expect(
            "register read u0",
            substrs=[
                "u0 = 0x3fc0000040200000",
                "     = (scalar = 1.5, lanes = (1.5, 2.5))",
            ],
        )
