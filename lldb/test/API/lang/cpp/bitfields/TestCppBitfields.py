"""Show bitfields and check that they display correctly."""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class CppBitfieldsTestCase(TestBase):
    @no_debug_info_test
    def test_bitfields(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp", False)
        )

        # Accessing LargeBitsA.
        self.expect_expr(
            "lba",
            result_children=[
                ValueCheck(name="", type="int:32"),
                ValueCheck(name="a", type="unsigned int:20", value="2"),
            ],
        )
        self.expect_expr("lba.a", result_type="unsigned int", result_value="2")

        # Accessing LargeBitsB.
        self.expect_expr(
            "lbb",
            result_children=[
                ValueCheck(name="a", type="unsigned int:1", value="1"),
                ValueCheck(name="", type="int:31"),
                ValueCheck(name="b", type="unsigned int:20", value="3"),
            ],
        )
        self.expect_expr("lbb.b", result_type="unsigned int", result_value="3")

        # Accessing LargeBitsC.
        self.expect_expr(
            "lbc",
            result_children=[
                ValueCheck(name="", type="int:22"),
                ValueCheck(name="a", type="unsigned int:1", value="1"),
                ValueCheck(name="b", type="unsigned int:1", value="0"),
                ValueCheck(name="c", type="unsigned int:5", value="4"),
                ValueCheck(name="d", type="unsigned int:1", value="1"),
                ValueCheck(name="", type="int:2"),
                ValueCheck(name="e", type="unsigned int:20", value="20"),
            ],
        )
        self.expect_expr("lbc.c", result_type="unsigned int", result_value="4")

        # Accessing LargeBitsD.
        self.expect_expr(
            "lbd",
            result_children=[
                ValueCheck(name="arr", type="char[3]", summary='"ab"'),
                ValueCheck(name="", type="int:32"),
                ValueCheck(name="a", type="unsigned int:20", value="5"),
            ],
        )
        self.expect_expr("lbd.a", result_type="unsigned int", result_value="5")

        # Test BitfieldsInStructInUnion.
        # FIXME: This needs some more explanation for what it's actually testing.
        nested_struct_children = [
            ValueCheck(name="", type="int:22"),
            ValueCheck(name="a", type="uint64_t:1", value="1"),
            ValueCheck(name="b", type="uint64_t:1", value="0"),
            ValueCheck(name="c", type="uint64_t:1", value="1"),
            ValueCheck(name="d", type="uint64_t:1", value="0"),
            ValueCheck(name="e", type="uint64_t:1", value="1"),
            ValueCheck(name="f", type="uint64_t:1", value="0"),
            ValueCheck(name="g", type="uint64_t:1", value="1"),
            ValueCheck(name="h", type="uint64_t:1", value="0"),
            ValueCheck(name="i", type="uint64_t:1", value="1"),
            ValueCheck(name="j", type="uint64_t:1", value="0"),
            ValueCheck(name="k", type="uint64_t:1", value="1"),
        ]
        self.expect_expr(
            "bitfields_in_struct_in_union",
            result_type="BitfieldsInStructInUnion",
            result_children=[
                ValueCheck(
                    name="",
                    children=[ValueCheck(name="f", children=nested_struct_children)],
                )
            ],
        )
        self.expect_expr(
            "bitfields_in_struct_in_union.f.a", result_type="uint64_t", result_value="1"
        )

        # Unions with bitfields.
        self.expect_expr(
            "uwbf",
            result_type="UnionWithBitfields",
            result_children=[
                ValueCheck(name="a", value="255"),
                ValueCheck(name="b", value="65535"),
                ValueCheck(name="c", value="4294967295"),
                ValueCheck(name="x", value="4294967295"),
            ],
        )
        self.expect_expr(
            "uwubf",
            result_type="UnionWithUnnamedBitfield",
            result_children=[
                ValueCheck(name="a", value="16777215"),
                ValueCheck(name="x", value="4294967295"),
            ],
        )

        # Class with a base class and a bitfield.
        self.expect_expr(
            "derived",
            result_type="Derived",
            result_children=[
                ValueCheck(
                    name="Base",
                    children=[ValueCheck(name="b_a", value="2", type="uint32_t")],
                ),
                ValueCheck(name="d_a", value="1", type="uint32_t:1"),
            ],
        )

        # Struct with bool bitfields.
        self.expect_expr(
            "bb",
            result_type="",
            result_children=[
                ValueCheck(name="a", value="true", type="bool:1"),
                ValueCheck(name="b", value="false", type="bool:1"),
                ValueCheck(name="c", value="true", type="bool:2"),
                ValueCheck(name="d", value="true", type="bool:2"),
            ],
        )

        bb = self.frame().FindVariable("bb")
        self.assertSuccess(bb.GetError())

        bb_a = bb.GetChildAtIndex(0)
        self.assertSuccess(bb_a.GetError())
        self.assertEqual(bb_a.GetValueAsUnsigned(), 1)
        self.assertEqual(bb_a.GetValueAsSigned(), 1)

        bb_b = bb.GetChildAtIndex(1)
        self.assertSuccess(bb_b.GetError())
        self.assertEqual(bb_b.GetValueAsUnsigned(), 0)
        self.assertEqual(bb_b.GetValueAsSigned(), 0)

        bb_c = bb.GetChildAtIndex(2)
        self.assertSuccess(bb_c.GetError())
        self.assertEqual(bb_c.GetValueAsUnsigned(), 1)
        self.assertEqual(bb_c.GetValueAsSigned(), 1)

        bb_d = bb.GetChildAtIndex(3)
        self.assertSuccess(bb_d.GetError())
        self.assertEqual(bb_d.GetValueAsUnsigned(), 1)
        self.assertEqual(bb_d.GetValueAsSigned(), 1)

        # Test a class with a base class that has a vtable ptr. The derived
        # class has bitfields.
        base_with_vtable_children = [
            ValueCheck(name="a", type="unsigned int:4", value="5"),
            ValueCheck(name="b", type="unsigned int:4", value="0"),
            ValueCheck(name="c", type="unsigned int:4", value="5"),
        ]
        self.expect_expr("base_with_vtable", result_children=base_with_vtable_children)
        self.expect_var_path("base_with_vtable", children=base_with_vtable_children)

    @no_debug_info_test
    def test_bitfield_behind_vtable_ptr(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp", False)
        )

        # Test a class with a vtable ptr and bitfields.
        with_vtable_children = [
            ValueCheck(name="a", type="unsigned int:4", value="5"),
            ValueCheck(name="b", type="unsigned int:4", value="0"),
            ValueCheck(name="c", type="unsigned int:4", value="5"),
        ]
        self.expect_expr("with_vtable", result_children=with_vtable_children)
        self.expect_var_path("with_vtable", children=with_vtable_children)

        # Test a class with a vtable ptr and unnamed bitfield directly after.
        with_vtable_and_unnamed_children = [
            ValueCheck(name="", type="int:4", value="0"),
            ValueCheck(name="b", type="unsigned int:4", value="0"),
            ValueCheck(name="c", type="unsigned int:4", value="5"),
        ]
        self.expect_expr(
            "with_vtable_and_unnamed", result_children=with_vtable_and_unnamed_children
        )
        self.expect_var_path(
            "with_vtable_and_unnamed", children=with_vtable_and_unnamed_children
        )

        derived_with_vtable_children = [
            ValueCheck(
                name="Base",
                children=[ValueCheck(name="b_a", value="2", type="uint32_t")],
            ),
            ValueCheck(name="a", value="1", type="unsigned int:1"),
        ]
        self.expect_expr(
            "derived_with_vtable", result_children=derived_with_vtable_children
        )
        self.expect_var_path(
            "derived_with_vtable", children=derived_with_vtable_children
        )
