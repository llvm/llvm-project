"""Test vectors from GDB remote target description XML."""

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
        return process

    def setup_register_test(self, definitions, register_data, architecture="aarch64"):
        return self.setup_multidoc_test(
            {
                "target.xml": dedent(
                    """\
                <?xml version="1.0"?>
                <target version="1.0">
                  <architecture>{}</architecture>
                  <feature name="test.register.vectors">
                    {}
                  </feature>
                </target>"""
                ).format(architecture, definitions)
            },
            register_data,
        )

    def assert_float_children(self, value, expected):
        self.assertTrue(value.IsValid())
        self.assertTrue(value.MightHaveChildren())
        self.assertEqual(value.GetNumChildren(), len(expected))
        for index, expected_value in enumerate(expected):
            child = value.GetChildAtIndex(index)
            self.assertTrue(child.IsValid())
            self.assertEqual(child.GetName(), "[{}]".format(index))
            self.assertEqual(child.GetTypeName(), "float")
            self.assertAlmostEqual(child.GetData().float[0], expected_value)
            self.assertAlmostEqual(float(child.GetValue()), expected_value)

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_direct_vector_cli_and_sb_api(self):
        process = self.setup_register_test(
            """\
            <vector id="v4f" type="ieee_single" count="4"/>
            <reg name="v0" regnum="0" bitsize="128" type="v4f"/>
            <reg name="pc" bitsize="64"/>""",
            "0000c03f000020400000604000009040" "0000000000000000",
        )

        self.expect(
            "register read v0",
            substrs=[
                "v0 = {0x00 0x00 0xc0 0x3f",
                "[0] = 1.5",
                "[1] = 2.5",
                "[2] = 3.5",
                "[3] = 4.5",
            ],
        )
        self.expect("register info v0", substrs=["Vector elements: 4"])

        frame = process.GetThreadAtIndex(0).GetFrameAtIndex(0)
        vector = frame.FindRegister("v0")
        self.assertEqual(vector.GetName(), "v0")
        self.assertEqual(vector.GetByteSize(), 16)
        self.assertTrue(vector.GetType().IsValid())
        self.assertEqual(vector.GetType().GetByteSize(), 16)
        self.assertIn("float", vector.GetTypeName())
        self.assert_float_children(vector, [1.5, 2.5, 3.5, 4.5])

        lane = vector.GetValueForExpressionPath("[2]")
        self.assertTrue(lane.IsValid())
        self.assertAlmostEqual(lane.GetData().float[0], 3.5)

        self.assertFalse(vector.IsDynamic())
        self.assertTrue(vector.IsSynthetic())
        self.assertTrue(vector.GetDynamicValue(lldb.eDynamicCanRunTarget).IsValid())
        self.assertTrue(vector.GetSyntheticValue().IsValid())
        self.assertTrue(vector.GetNonSyntheticValue().IsValid())
        self.assertFalse(vector.GetNonSyntheticValue().IsSynthetic())
        self.assertFalse(frame.FindRegister("v0[2]").IsValid())
        self.assertFalse(frame.FindRegister("does_not_exist").IsValid())

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_byte_vector_sb_api(self):
        process = self.setup_register_test(
            """\
            <vector id="bytes32" type="uint8" count="32"/>
            <reg name="bytes" regnum="0" bitsize="256" type="bytes32"/>
            <reg name="pc" bitsize="64"/>""",
            "".join("{:02x}".format(value) for value in range(32)) + "0000000000000000",
        )

        frame = process.GetThreadAtIndex(0).GetFrameAtIndex(0)
        vector = frame.FindRegister("bytes")
        self.assertEqual(vector.GetByteSize(), 32)
        self.assertEqual(vector.GetNumChildren(), 32)
        self.assertEqual(vector.GetData().uint8[:32], list(range(32)))

        lane = vector.GetValueForExpressionPath("[16]")
        self.assertTrue(lane.IsValid())
        error = lldb.SBError()
        self.assertEqual(lane.GetValueAsUnsigned(error, 0xDEADBEEF), 0x10)
        self.assertTrue(error.Success())

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_three_lane_vector_exact_layout(self):
        process = self.setup_register_test(
            """\
            <vector id="v3f" type="ieee_single" count="3"/>
            <reg name="v0" regnum="0" bitsize="96" type="v3f"/>
            <reg name="pc" bitsize="64"/>""",
            "0000c03f0000204000006040" "0000000000000000",
        )

        self.expect(
            "register read v0",
            substrs=[
                "v0 = {0x00 0x00 0xc0 0x3f",
                "[0] = 1.5",
                "[1] = 2.5",
                "[2] = 3.5",
            ],
        )

        vector = process.GetThreadAtIndex(0).GetFrameAtIndex(0).FindRegister("v0")
        self.assertEqual(vector.GetByteSize(), 12)
        self.assertEqual(vector.GetTypeName(), "float[3]")
        self.assert_float_children(vector, [1.5, 2.5, 3.5])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_nested_vectors(self):
        process = self.setup_register_test(
            """\
            <vector id="v2f" type="ieee_single" count="2"/>
            <vector id="v2v2f" type="v2f" count="2"/>
            <reg name="v0" regnum="0" bitsize="128" type="v2v2f"/>
            <reg name="pc" bitsize="64"/>""",
            "0000c03f000020400000604000009040" "0000000000000000",
        )

        self.expect(
            "register read v0",
            substrs=[
                "[0] = ([0] = 1.5, [1] = 2.5)",
                "[1] = ([0] = 3.5, [1] = 4.5)",
            ],
        )

        vector = process.GetThreadAtIndex(0).GetFrameAtIndex(0).FindRegister("v0")
        self.assertEqual(vector.GetNumChildren(), 2)
        self.assert_float_children(vector.GetChildAtIndex(0), [1.5, 2.5])
        self.assert_float_children(vector.GetChildAtIndex(1), [3.5, 4.5])
        lane = vector.GetValueForExpressionPath("[1][0]")
        self.assertTrue(lane.IsValid())
        self.assertAlmostEqual(lane.GetData().float[0], 3.5)

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_invalid_vectors_are_ignored(self):
        process = self.setup_register_test(
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
            <flags id="flags" size="8">
              <field name="bit" start="0" end="0"/>
            </flags>
            <vector id="flags_vector" type="flags" count="2"/>
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
            <reg name="flags_vector" regnum="9" bitsize="128"
                 type="flags_vector"/>
            <reg name="bad_size" regnum="10" bitsize="64" type="v4f"/>
            <reg name="pc" bitsize="64"/>""",
            "0000000000000000" * 8
            + "0000c03f00002040"
            + "00" * 16
            + "0000000000000000" * 2,
        )

        frame = process.GetThreadAtIndex(0).GetFrameAtIndex(0)
        for name in [
            "bad_count",
            "zero",
            "too_many",
            "unknown",
            "forward",
            "missing_type",
            "missing_count",
            "too_large",
            "flags_vector",
            "bad_size",
        ]:
            value = frame.FindRegister(name)
            self.assertTrue(value.IsValid())
            self.assertEqual(value.GetNumChildren(), 0)

        self.assert_float_children(frame.FindRegister("later"), [1.5, 2.5])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_duplicate_vector_id_uses_first_definition(self):
        process = self.setup_register_test(
            """\
            <vector id="shared" type="ieee_single" count="2"/>
            <vector id="shared" type="ieee_single" count="4"/>
            <reg name="v0" regnum="0" bitsize="64" type="shared"/>
            <reg name="pc" bitsize="64"/>""",
            "0000c03f00002040" "0000000000000000",
        )

        vector = process.GetThreadAtIndex(0).GetFrameAtIndex(0).FindRegister("v0")
        self.assert_float_children(vector, [1.5, 2.5])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_target_type_precedes_builtin_type(self):
        process = self.setup_register_test(
            """\
            <vector id="ieee_single" type="uint16" count="2"/>
            <vector id="nested" type="ieee_single" count="2"/>
            <reg name="v0" regnum="0" bitsize="64" type="nested"/>
            <reg name="pc" bitsize="64"/>""",
            "0100020003000400" "0000000000000000",
        )

        vector = process.GetThreadAtIndex(0).GetFrameAtIndex(0).FindRegister("v0")
        self.assertEqual(vector.GetNumChildren(), 2)
        self.assertEqual(
            [
                vector.GetChildAtIndex(outer)
                .GetChildAtIndex(inner)
                .GetValueAsUnsigned()
                for outer in range(2)
                for inner in range(2)
            ],
            [1, 2, 3, 4],
        )

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_bool_and_single_uint128_vectors(self):
        process = self.setup_register_test(
            """\
            <vector id="v4b" type="bool" count="4"/>
            <vector id="v1u128" type="uint128" count="1"/>
            <reg name="bools" regnum="0" bitsize="32" type="v4b"/>
            <reg name="wide" regnum="1" bitsize="128" type="v1u128"/>
            <reg name="pc" bitsize="64"/>""",
            "00010100" "01000000000000000000000000000000" "0000000000000000",
        )

        frame = process.GetThreadAtIndex(0).GetFrameAtIndex(0)
        bools = frame.FindRegister("bools")
        self.assertEqual(bools.GetTypeName(), "bool[4]")
        self.assertEqual(bools.GetNumChildren(), 4)
        self.assertEqual(
            [bools.GetChildAtIndex(i).GetValueAsUnsigned() for i in range(4)],
            [0, 1, 1, 0],
        )

        wide = frame.FindRegister("wide")
        self.assertEqual(wide.GetNumChildren(), 1)
        wide_child = wide.GetChildAtIndex(0)
        expected_bytes = [1] + [0] * 15
        self.assertEqual(wide.GetData().uint8[:16], expected_bytes)
        self.assertEqual(wide_child.GetData().uint8[:16], expected_bytes)
        self.assertEqual(wide_child.GetValue(), "1")

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_unsupported_direct_float_builtins_remain_readable(self):
        process = self.setup_register_test(
            """\
            <reg name="half" regnum="0" bitsize="16" type="ieee_half"/>
            <reg name="bfloat" regnum="1" bitsize="16" type="bfloat16"/>
            <reg name="x87" regnum="2" bitsize="80" type="i387_ext"/>
            <reg name="pc" bitsize="64"/>""",
            "0102" "0304" "05060708090a0b0c0d0e" "0000000000000000",
        )

        frame = process.GetThreadAtIndex(0).GetFrameAtIndex(0)
        self.assertEqual(frame.FindRegister("half").GetData().uint8[:2], [1, 2])
        self.assertEqual(frame.FindRegister("bfloat").GetData().uint8[:2], [3, 4])
        self.assertEqual(
            frame.FindRegister("x87").GetData().uint8[:10],
            list(range(5, 15)),
        )

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_vector_ids_are_scoped_to_included_feature(self):
        process = self.setup_multidoc_test(
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
            "0000c03f00002040"
            "0000c03f000020400000604000009040"
            "0100000000000000"
            "0000000000000000",
        )

        frame = process.GetThreadAtIndex(0).GetFrameAtIndex(0)
        self.assert_float_children(frame.FindRegister("first"), [1.5, 2.5])
        self.assert_float_children(frame.FindRegister("second"), [1.5, 2.5, 3.5, 4.5])
        self.assertEqual(frame.FindRegister("unresolved").GetNumChildren(), 0)

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_explicit_format_suppresses_typed_output(self):
        self.setup_register_test(
            """\
            <vector id="v4f" type="ieee_single" count="4"/>
            <reg name="v0" regnum="0" bitsize="128" type="v4f"/>
            <reg name="pc" bitsize="64"/>""",
            "0000c03f000020400000604000009040" "0000000000000000",
        )

        self.expect(
            "register read -f hex v0",
            substrs=["v0 = 0x4090000040600000402000003fc00000"],
        )
        self.expect("register read -f hex v0", matching=False, substrs=["[0] ="])

    @skipIfXmlSupportMissing
    @skipIfRemote
    @skipIfLLVMTargetMissing("SystemZ")
    def test_big_endian_vector(self):
        process = self.setup_register_test(
            """\
            <vector id="v2f" type="ieee_single" count="2"/>
            <reg name="v0" regnum="0" bitsize="64" type="v2f"/>
            <reg name="pswa" bitsize="64"/>""",
            "3fc0000040200000" "0000000000000000",
            architecture="s390x",
        )

        self.expect(
            "register read v0",
            substrs=[
                "v0 = {0x3f 0xc0 0x00 0x00 0x40 0x20 0x00 0x00}",
                "[0] = 1.5",
                "[1] = 2.5",
            ],
        )
        vector = process.GetThreadAtIndex(0).GetFrameAtIndex(0).FindRegister("v0")
        self.assert_float_children(vector, [1.5, 2.5])
