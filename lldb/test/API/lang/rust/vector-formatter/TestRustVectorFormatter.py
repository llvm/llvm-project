"""
Make sure Rust vector data formatter works.
"""

import os
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil


class TestRustVectorFormatter(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def load_rust_core(self):
        core_yaml_path = self.getSourcePath("main.core.yaml")
        main_yaml_path = self.getSourcePath("main.yaml")
        self.yaml2obj(core_yaml_path, self.getBuildArtifact("main.core"))
        self.yaml2obj(main_yaml_path, self.getBuildArtifact("main"))
        target = self.dbg.CreateTarget(self.getBuildArtifact("main"))
        process = target.LoadCore(self.getBuildArtifact("main.core"))
        self.assertTrue(process.IsValid())
        return target, process

    def test_vec_formatter(self):
        self.load_rust_core()

        self.expect_var_path(
            "vector",
            summary="size=3",
            children=[
                ValueCheck(name="[0]", value="1"),
                ValueCheck(name="[1]", value="2"),
                ValueCheck(name="[2]", value="3"),
            ],
        )

        self.expect_expr(
            "vector",
            result_summary="size=3",
            result_children=[
                ValueCheck(name="[0]", value="1"),
                ValueCheck(name="[1]", value="2"),
                ValueCheck(name="[2]", value="3"),
            ],
        )

        # Verify that pointers/references to a Vec are also formatted properly.
        self.expect(
            "frame variable --ptr-depth 1 &vector",
            substrs=["size=3", "[0] = 1", "[1] = 2", "[2] = 3"],
        )
        self.expect_expr(
            "&vector",
            result_summary="size=3",
        )

    def test_issue1_type_lookup(self):
        """'type lookup' should find find Rust structs."""
        self.load_rust_core()
        self.expect("type lookup UsizeNoHighBit", substrs=["struct UsizeNoHighBit"])

    def test_null_pointer_summary(self):
        """Null pointers to Rust structs need to print as 'NULL'."""
        target, _ = self.load_rust_core()
        vec_val = self.frame().FindVariable("vector").GetNonSyntheticValue()
        struct_ptr_type = (
            vec_val.GetChildMemberWithName("buf").GetType().GetPointerType()
        )
        null_data = lldb.SBData.CreateDataFromUInt64Array(
            target.GetByteOrder(), target.GetAddressByteSize(), [0]
        )
        null_ptr = target.CreateValueFromData("null_ptr", null_data, struct_ptr_type)
        self.assertTrue(str(null_ptr).endswith("= NULL"), str(null_ptr))

    def test_issue3_sbvalue_evaluate_expression(self):
        """SBValue::EvaluateExpression should work with a Rust struct."""
        self.load_rust_core()
        vec_val = self.frame().FindVariable("vector").GetNonSyntheticValue()
        len_val = vec_val.EvaluateExpression("len")
        self.assertTrue(len_val.GetError().Success(), str(len_val.GetError()))
        self.assertEqual(len_val.GetValueAsUnsigned(), 3)

    def test_rust_category_primitives(self):
        """A Rust-restricted formatter category should not fail to match Rust primitive types."""
        self.load_rust_core()
        rust_cat = self.dbg.CreateCategory("rust_custom")
        self.addTearDownHook(lambda: self.dbg.DeleteCategory("rust_custom"))
        rust_cat.AddLanguage(lldb.eLanguageTypeRust)
        rust_cat.AddTypeSummary(
            lldb.SBTypeNameSpecifier("int"),
            lldb.SBTypeSummary.CreateWithSummaryString("RUST_INT"),
        )
        rust_cat.SetEnabled(True)
        self.expect("frame variable vector[0]", substrs=["RUST_INT"])

    def test_cpp_category_fallback(self):
        """C++ formatter categories should still act as a fallback for Rust structs."""
        self.load_rust_core()
        self.runCmd("type category define -l c++ -e cpp_fallback")
        self.addTearDownHook(lambda: self.dbg.DeleteCategory("cpp_fallback"))
        self.runCmd(
            "type summary add --summary-string 'CPP_FALLBACK' -w cpp_fallback "
            "core::num::niche_types::UsizeNoHighBit"
        )
        self.expect("frame variable vector.buf.inner.cap", substrs=["CPP_FALLBACK"])
    def test_enum_variant_part_language(self):
        """Inner $Inner union and $Variant structs of Rust enums should not get C++ instead of Rust."""
        enum_yaml = os.path.join(
            self.getSourceDir(), "..", "enum-structs", "main.yaml"
        )
        obj_path = self.getBuildArtifact("enum.o")
        self.yaml2obj(enum_yaml, obj_path)
        target = self.dbg.CreateTarget(obj_path)
        enum_val = target.FindFirstGlobalVariable("ENUM_WITH_TUPLES_A")
        inner_val = enum_val.GetChildAtIndex(0)
        variant_val = inner_val.GetChildAtIndex(0)
        only_rust = self.dbg.CreateCategory("only_rust")
        self.addTearDownHook(lambda: self.dbg.DeleteCategory("only_rust"))
        only_rust.AddLanguage(lldb.eLanguageTypeRust)
        for val, label in [
            (enum_val, "OUTER_RUST"),
            (inner_val, "INNER_RUST"),
            (variant_val, "VARIANT_RUST"),
        ]:
            only_rust.AddTypeSummary(
                lldb.SBTypeNameSpecifier(val.GetTypeName()),
                lldb.SBTypeSummary.CreateWithSummaryString(label),
            )
        only_rust.SetEnabled(True)
        self.assertEqual(enum_val.GetSummary(), "OUTER_RUST")
        self.assertEqual(inner_val.GetSummary(), "INNER_RUST")
        self.assertEqual(variant_val.GetSummary(), "VARIANT_RUST")
