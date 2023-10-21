"""Test that lldb recognizes enum structs emitted by Rust compiler """
import logging

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from RustEnumValue import RustEnumValue


class TestRustEnumStructs(TestBase):
    def setUp(self):
        TestBase.setUp(self)
        src_dir = self.getSourceDir()
        yaml_path = os.path.join(src_dir, "main.yaml")
        obj_path = self.getBuildArtifact("main.o")
        self.yaml2obj(yaml_path, obj_path)
        self.dbg.CreateTarget(obj_path)

    def getFromGlobal(self, name):
        values = self.target().FindGlobalVariables(name, 1)
        self.assertEqual(values.GetSize(), 1)
        return RustEnumValue(values[0])

    def test_clike_enums_are_represented_correctly(self):
        # these type of enums are not using DW_TAG_variant_part.
        all_values = [
            self.target().FindFirstGlobalVariable("CLIKE_DEFAULT_A").GetValue(),
            self.target().FindFirstGlobalVariable("CLIKE_DEFAULT_B").GetValue(),
            self.target().FindFirstGlobalVariable("CLIKE_U8_A").GetValue(),
            self.target().FindFirstGlobalVariable("CLIKE_U8_C").GetValue(),
            self.target().FindFirstGlobalVariable("CLIKE_U32_A").GetValue(),
            self.target().FindFirstGlobalVariable("CLIKE_U32_B").GetValue(),
        ]
        self.assertEqual(
            all_values,
            [
                "A(2)",
                "B(10)",
                "VariantA(0)",
                "VariantC(2)",
                "VariantA(1)",
                "VariantB(2)",
            ],
        )

    def test_enum_with_tuples_has_all_variants(self):
        self.assertEqual(
            self.getFromGlobal("ENUM_WITH_TUPLES_A").getAllVariantTypes(),
            [
                "main::EnumWithTuples::A:8",
                "main::EnumWithTuples::B:8",
                "main::EnumWithTuples::C:8",
                "main::EnumWithTuples::D:8",
                "main::EnumWithTuples::AA:8",
                "main::EnumWithTuples::BB:8",
                "main::EnumWithTuples::BC:8",
                "main::EnumWithTuples::CC:8",
            ],
        )

    def test_enum_with_tuples_values_are_correct_a(self):
        # static ENUM_WITH_TUPLES_A: EnumWithTuples = EnumWithTuples::A(13);
        self.assertEqual(
            self.getFromGlobal("ENUM_WITH_TUPLES_A")
            .getCurrentValue()
            .GetChildAtIndex(0)
            .GetData()
            .GetUnsignedInt8(lldb.SBError(), 0),
            13,
        )

    def test_enum_with_tuples_values_are_correct_aa(self):
        # static ENUM_WITH_TUPLES_AA: EnumWithTuples = EnumWithTuples::AA(13, 37);
        value = self.getFromGlobal("ENUM_WITH_TUPLES_AA").getCurrentValue()
        self.assertEqual(
            (
                value.GetChildAtIndex(0).GetData().GetUnsignedInt8(lldb.SBError(), 0),
                value.GetChildAtIndex(1).GetData().GetUnsignedInt8(lldb.SBError(), 0),
            ),
            (13, 37),
        )

    def test_enum_with_tuples_values_are_correct_b(self):
        # static ENUM_WITH_TUPLES_B: EnumWithTuples = EnumWithTuples::B(37);
        self.assertEqual(
            self.getFromGlobal("ENUM_WITH_TUPLES_B")
            .getCurrentValue()
            .GetChildAtIndex(0)
            .GetData()
            .GetUnsignedInt16(lldb.SBError(), 0),
            37,
        )

    def test_enum_with_tuples_values_are_correct_bb(self):
        # static ENUM_WITH_TUPLES_BB: EnumWithTuples = EnumWithTuples::BB(37, 5535);
        value = self.getFromGlobal("ENUM_WITH_TUPLES_BB").getCurrentValue()
        self.assertEqual(
            (
                value.GetChildAtIndex(0).GetData().GetUnsignedInt16(lldb.SBError(), 0),
                value.GetChildAtIndex(1).GetData().GetUnsignedInt16(lldb.SBError(), 0),
            ),
            (37, 5535),
        )

    def test_enum_with_tuples_values_are_correct_bc(self):
        # static ENUM_WITH_TUPLES_BC: EnumWithTuples = EnumWithTuples::BC(65000, 165000);
        value = self.getFromGlobal("ENUM_WITH_TUPLES_BC").getCurrentValue()
        self.assertEqual(
            (
                value.GetChildAtIndex(0).GetData().GetUnsignedInt16(lldb.SBError(), 0),
                value.GetChildAtIndex(1).GetData().GetUnsignedInt32(lldb.SBError(), 0),
            ),
            (65000, 165000),
        )

    def test_enum_with_tuples_values_are_correct_c(self):
        # static ENUM_WITH_TUPLES_C: EnumWithTuples = EnumWithTuples::C(31337);
        self.assertEqual(
            self.getFromGlobal("ENUM_WITH_TUPLES_C")
            .getCurrentValue()
            .GetChildAtIndex(0)
            .GetData()
            .GetUnsignedInt32(lldb.SBError(), 0),
            31337,
        )

    def test_enum_with_tuples_values_are_correct_cc(self):
        # static ENUM_WITH_TUPLES_CC: EnumWithTuples = EnumWithTuples::CC(31337, 87236);
        value = self.getFromGlobal("ENUM_WITH_TUPLES_CC").getCurrentValue()
        self.assertEqual(
            (
                value.GetChildAtIndex(0).GetData().GetUnsignedInt32(lldb.SBError(), 0),
                value.GetChildAtIndex(1).GetData().GetUnsignedInt32(lldb.SBError(), 0),
            ),
            (31337, 87236),
        )

    def test_enum_with_tuples_values_are_correct_d(self):
        # static ENUM_WITH_TUPLES_D: EnumWithTuples = EnumWithTuples::D(123456789012345678);
        self.assertEqual(
            self.getFromGlobal("ENUM_WITH_TUPLES_D")
            .getCurrentValue()
            .GetChildAtIndex(0)
            .GetData()
            .GetUnsignedInt64(lldb.SBError(), 0),
            123456789012345678,
        )

    def test_mixed_enum_variants(self):
        # static MIXED_ENUM_A: MixedEnum1 = MixedEnum1::A;
        self.assertEqual(
            self.getFromGlobal("MIXED_ENUM_A").getAllVariantTypes(),
            [
                "main::MixedEnum::A:64",
                "main::MixedEnum::B:64",
                "main::MixedEnum::C:64",
                "main::MixedEnum::D:64",
                "main::MixedEnum::E:64",
            ],
        )

    def test_mixed_enum_a(self):
        # static MIXED_ENUM_A: MixedEnum = MixedEnum::A;
        value = self.getFromGlobal("MIXED_ENUM_A").getCurrentValue()
        self.assertEqual(value.GetType().GetDisplayTypeName(), "main::MixedEnum::A")
        self.assertEqual(value.GetValue(), None)

    def test_mixed_enum_c(self):
        # static MIXED_ENUM_C: MixedEnum = MixedEnum::C(254, -254);
        value = self.getFromGlobal("MIXED_ENUM_C").getCurrentValue()
        self.assertEqual(
            (
                value.GetChildAtIndex(0).GetData().GetUnsignedInt8(lldb.SBError(), 0),
                value.GetChildAtIndex(1).GetData().GetSignedInt32(lldb.SBError(), 0),
            ),
            (254, -254),
        )

    def test_mixed_enum_d_none(self):
        # static MIXED_ENUM_D_NONE: MixedEnum = MixedEnum::D(None);
        value = RustEnumValue(
            self.getFromGlobal("MIXED_ENUM_D_NONE").getCurrentValue().GetChildAtIndex(0)
        )
        self.assertEqual(
            value.getAllVariantTypes(),
            [
                "core::option::Option<main::Struct2>::None<main::Struct2>:32",
                "core::option::Option<main::Struct2>::Some<main::Struct2>:32",
            ],
        )
        self.assertEqual(value.getCurrentValue().GetValue(), None)
        self.assertEqual(
            value.getCurrentValue().GetType().GetDisplayTypeName(),
            "core::option::Option<main::Struct2>::None<main::Struct2>",
        )

    def test_mixed_enum_d_some(self):
        # static MIXED_ENUM_D_SOME: MixedEnum = MixedEnum::D(Some(Struct2 {
        #     field: 123456,
        #     inner: Struct1 { field: 123 },
        # }));
        variant_with_option = RustEnumValue(
            self.getFromGlobal("MIXED_ENUM_D_SOME").getCurrentValue().GetChildAtIndex(0)
        )

        value_inside_option = variant_with_option.getCurrentValue().GetChildAtIndex(0)
        self.assertEqual(
            value_inside_option.GetChildMemberWithName("field")
            .GetData()
            .GetUnsignedInt32(lldb.SBError(), 0),
            123456,
        )

        self.assertEqual(
            value_inside_option.GetChildMemberWithName("inner")
            .GetChildMemberWithName("field")
            .GetData()
            .GetSignedInt32(lldb.SBError(), 0),
            123,
        )
        self.assertEqual(
            value_inside_option.GetType().GetDisplayTypeName(), "main::Struct2"
        )

    def test_option_non_null_some_pointer(self):
        type = self.target().FindFirstType(
            "core::option::Option<core::ptr::non_null::NonNull<u64>>"
        )
        # this type is "optimized" by rust compiler so the discriminant isn't present on Some variant of option
        data = [1337]
        pointer_size = self.target().GetAddressByteSize()
        byte_order = self.target().GetByteOrder()
        value = RustEnumValue(
            self.target().CreateValueFromData(
                "adhoc_value",
                lldb.SBData.CreateDataFromUInt64Array(byte_order, pointer_size, data),
                type,
            )
        )
        self.assertEqual(value.getFields(), ["$variant$0", "$variant$"])
        self.assertEqual(
            value.getCurrentValue()
            .GetChildAtIndex(0)
            .GetChildMemberWithName("pointer")
            .GetValueAsUnsigned(),
            1337,
        )

    def test_option_non_null_none(self):
        type = self.target().FindFirstType(
            "core::option::Option<core::ptr::non_null::NonNull<u64>>"
        )
        # this type is "optimized" by rust compiler so the discriminant isn't present on Some variant of option
        # in this test case 0 is used to represent 'None'
        data = [0]
        pointer_size = self.target().GetAddressByteSize()
        byte_order = self.target().GetByteOrder()
        value = RustEnumValue(
            self.target().CreateValueFromData(
                "adhoc_value",
                lldb.SBData.CreateDataFromUInt64Array(byte_order, pointer_size, data),
                type,
            )
        )
        self.assertEqual(value.getFields(), ["$variant$0", "$variant$"])
        self.assertEqual(value.getCurrentValue().GetValue(), None)
        self.assertEqual(
            value.getCurrentValue().GetType().GetDisplayTypeName(),
            "core::option::Option<core::ptr::non_null::NonNull<u64>>::None<core::ptr::non_null::NonNull<unsigned long> >",
        )

    def test_niche_layout_with_fields_2(self):
        # static NICHE_W_FIELDS_2_A: NicheLayoutWithFields2 =
        #           NicheLayoutWithFields2::A(NonZeroU32::new(800).unwrap(), 900);
        value = self.getFromGlobal("NICHE_W_FIELDS_2_A").getCurrentValue()
        self.assertEqual(
            (
                value.GetChildAtIndex(0)
                .GetChildAtIndex(0)
                .GetData()
                .GetUnsignedInt32(lldb.SBError(), 0),
                value.GetChildAtIndex(1).GetData().GetUnsignedInt32(lldb.SBError(), 0),
            ),
            (800, 900),
        )

    def test_niche_layout_with_fields_3_a(self):
        # static NICHE_W_FIELDS_3_A: NicheLayoutWithFields3 = NicheLayoutWithFields3::A(137, true);
        value = self.getFromGlobal("NICHE_W_FIELDS_3_A").getCurrentValue()
        self.assertEqual(
            (
                value.GetChildAtIndex(0).GetData().GetUnsignedInt8(lldb.SBError(), 0),
                value.GetChildAtIndex(1).GetData().GetUnsignedInt8(lldb.SBError(), 0),
            ),
            (137, 1),
        )

    def test_niche_layout_with_fields_3_a(self):
        # static NICHE_W_FIELDS_3_C: NicheLayoutWithFields3 = NicheLayoutWithFields3::C(false);
        value = self.getFromGlobal("NICHE_W_FIELDS_3_C").getCurrentValue()
        self.assertEqual(
            value.GetChildAtIndex(0).GetData().GetUnsignedInt8(lldb.SBError(), 0), 0
        )
