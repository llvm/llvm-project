//===-- DWARFDIETest.cpp ----------------------------------------------=---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/SymbolFile/DWARF/DWARFDIE.h"
#include "Plugins/SymbolFile/DWARF/DWARFDebugInfo.h"
#include "Plugins/SymbolFile/DWARF/DWARFDeclContext.h"
#include "TestingSupport/Symbol/YAMLModuleTester.h"
#include "lldb/Core/dwarf.h"
#include "lldb/Symbol/Type.h"
#include "lldb/lldb-private-enumerations.h"
#include "llvm/ADT/STLExtras.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::plugin::dwarf;
using namespace lldb_private::dwarf;

TEST(DWARFDIETest, ChildIteration) {
  // Tests DWARFDIE::child_iterator.

  const char *yamldata = R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_386
DWARF:
  debug_abbrev:
    - Table:
        - Code:            0x00000001
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_language
              Form:            DW_FORM_data2
        - Code:            0x00000002
          Tag:             DW_TAG_base_type
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_encoding
              Form:            DW_FORM_data1
            - Attribute:       DW_AT_byte_size
              Form:            DW_FORM_data1
  debug_info:
    - Version:         4
      AddrSize:        8
      Entries:
        - AbbrCode:        0x00000001
          Values:
            - Value:           0x000000000000000C
        - AbbrCode:        0x00000002
          Values:
            - Value:           0x0000000000000007 # DW_ATE_unsigned
            - Value:           0x0000000000000004
        - AbbrCode:        0x00000002
          Values:
            - Value:           0x0000000000000007 # DW_ATE_unsigned
            - Value:           0x0000000000000008
        - AbbrCode:        0x00000002
          Values:
            - Value:           0x0000000000000005 # DW_ATE_signed
            - Value:           0x0000000000000008
        - AbbrCode:        0x00000000
)";

  YAMLModuleTester t(yamldata);
  ASSERT_TRUE((bool)t.GetDwarfUnit());

  DWARFUnit *unit = t.GetDwarfUnit();
  const DWARFDebugInfoEntry *die_first = unit->DIE().GetDIE();

  // Create a DWARFDIE that has three DW_TAG_base_type children.
  DWARFDIE top_die(unit, die_first);

  // Create the iterator range that has the three tags as elements.
  llvm::iterator_range<DWARFDIE::child_iterator> children = top_die.children();

  // Compare begin() to the first child DIE.
  DWARFDIE::child_iterator child_iter = children.begin();
  ASSERT_NE(child_iter, children.end());
  const DWARFDebugInfoEntry *die_child0 = die_first->GetFirstChild();
  EXPECT_EQ((*child_iter).GetDIE(), die_child0);

  // Step to the second child DIE.
  ++child_iter;
  ASSERT_NE(child_iter, children.end());
  const DWARFDebugInfoEntry *die_child1 = die_child0->GetSibling();
  EXPECT_EQ((*child_iter).GetDIE(), die_child1);

  // Step to the third child DIE.
  ++child_iter;
  ASSERT_NE(child_iter, children.end());
  const DWARFDebugInfoEntry *die_child2 = die_child1->GetSibling();
  EXPECT_EQ((*child_iter).GetDIE(), die_child2);

  // Step to the end of the range.
  ++child_iter;
  EXPECT_EQ(child_iter, children.end());

  // Take one of the DW_TAG_base_type DIEs (which has no children) and make
  // sure the children range is now empty.
  DWARFDIE no_children_die(unit, die_child0);
  EXPECT_TRUE(no_children_die.children().empty());
}

TEST(DWARFDIETest, PeekName) {
  const char *yamldata = R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_386
DWARF:
  debug_str:
    - 'NameType1'
    - 'NameType2'
  debug_abbrev:
    - Table:
        - Code:            0x00000001
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_language
              Form:            DW_FORM_data2
        - Code:            0x00000002
          Tag:             DW_TAG_base_type
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
        - Code:            0x00000003
          Tag:             DW_TAG_base_type
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_abstract_origin
              Form:            DW_FORM_ref1
        - Code:            0x00000004
          Tag:             DW_TAG_base_type
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_specification
              Form:            DW_FORM_ref1
  debug_info:
    - Version:         4
      AddrSize:        8
      Entries:
        - AbbrCode:        0x00000001
          Values:
            - Value:           0x000000000000000C
        - AbbrCode:        0x00000002
          Values:
            - Value:           0x0000000000000000 # Name = NameType1
        - AbbrCode:        0x00000002
          Values:
            - Value:           0x000000000000000a # Name = NameType2
        - AbbrCode:        0x00000003
          Values:
            - Value:           0x000000000000000e # Ref abstract origin to NameType1 DIE.
        - AbbrCode:        0x00000004
          Values:
            - Value:           0x0000000000000013 # Ref specification to NameType2 DIE.
        - AbbrCode:        0x00000000
)";

  YAMLModuleTester t(yamldata);
  auto *symbol_file =
      llvm::cast<SymbolFileDWARF>(t.GetModule()->GetSymbolFile());
  DWARFUnit *unit = symbol_file->DebugInfo().GetUnitAtIndex(0);

  dw_offset_t first_die_offset = 11;
  EXPECT_EQ(unit->PeekDIEName(first_die_offset), "");

  dw_offset_t second_die_offset = 14;
  EXPECT_EQ(unit->PeekDIEName(second_die_offset), "NameType1");

  dw_offset_t third_die_offset = 19;
  EXPECT_EQ(unit->PeekDIEName(third_die_offset), "NameType2");

  dw_offset_t fourth_die_offset = 24;
  EXPECT_EQ(unit->PeekDIEName(fourth_die_offset), "NameType1");

  dw_offset_t fifth_die_offset = 26;
  EXPECT_EQ(unit->PeekDIEName(fifth_die_offset), "NameType2");
}

TEST(DWARFDIETest, GetContext) {
  const char *yamldata = R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_386
DWARF:
  debug_abbrev:
    - ID:              0
      Table:
        - Code:            0x1
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_language
              Form:            DW_FORM_data2
        - Code:            0x2
          Tag:             DW_TAG_namespace
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_string
        - Code:            0x3
          Tag:             DW_TAG_structure_type
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_string
        - Code:            0x4
          Tag:             DW_TAG_namespace
          Children:        DW_CHILDREN_yes
  debug_info:
    - Version:         4
      AddrSize:        8
      Entries:
        - AbbrCode:        0x1
          Values:
            - Value:           0x000000000000000C
        - AbbrCode:        0x2
          Values:
            - CStr:            NAMESPACE
        - AbbrCode:        0x3
          Values:
            - CStr:            STRUCT
        - AbbrCode:        0x4
        - AbbrCode:        0x3
          Values:
            - CStr:            STRUCT
        - AbbrCode:        0x0
        - AbbrCode:        0x0
        - AbbrCode:        0x0
)";

  YAMLModuleTester t(yamldata);
  auto *symbol_file =
      llvm::cast<SymbolFileDWARF>(t.GetModule()->GetSymbolFile());
  DWARFUnit *unit = symbol_file->DebugInfo().GetUnitAtIndex(0);
  ASSERT_TRUE(unit);

  auto make_namespace = [](const char *name) {
    return CompilerContext(CompilerContextKind::Namespace, ConstString(name));
  };
  auto make_struct = [](const char *name) {
    return CompilerContext(CompilerContextKind::ClassOrStruct,
                           ConstString(name));
  };
  DWARFDIE struct_die = unit->DIE().GetFirstChild().GetFirstChild();
  ASSERT_TRUE(struct_die);
  DWARFDIE anon_struct_die = struct_die.GetSibling().GetFirstChild();
  ASSERT_TRUE(anon_struct_die);
  EXPECT_THAT(
      struct_die.GetDeclContext(),
      testing::ElementsAre(make_namespace("NAMESPACE"), make_struct("STRUCT")));
  EXPECT_THAT(
      struct_die.GetTypeLookupContext(),
      testing::ElementsAre(make_namespace("NAMESPACE"), make_struct("STRUCT")));
  EXPECT_THAT(struct_die.GetDWARFDeclContext(),
              DWARFDeclContext({{DW_TAG_structure_type, "STRUCT"},
                                {DW_TAG_namespace, "NAMESPACE"}}));
  EXPECT_THAT(anon_struct_die.GetDeclContext(),
              testing::ElementsAre(make_namespace("NAMESPACE"),
                                   make_namespace(nullptr),
                                   make_struct("STRUCT")));
  EXPECT_THAT(anon_struct_die.GetTypeLookupContext(),
              testing::ElementsAre(make_namespace("NAMESPACE"),
                                   make_namespace(nullptr),
                                   make_struct("STRUCT")));
  EXPECT_THAT(anon_struct_die.GetDWARFDeclContext(),
              DWARFDeclContext({{DW_TAG_structure_type, "STRUCT"},
                                {DW_TAG_namespace, nullptr},
                                {DW_TAG_namespace, "NAMESPACE"}}));
}

TEST(DWARFDIETest, GetContextInFunction) {
  // Make sure we get the right context fo each "struct_t" type. The first
  // should be "a::struct_t" and the one defined in the "foo" function should be
  // "struct_t". Previous DWARFDIE::GetTypeLookupContext() function calls would
  // have the "struct_t" in "foo" be "a::struct_t" because it would traverse the
  // entire die parent tree and ignore DW_TAG_subprogram and keep traversing the
  // parents.
  //
  // 0x0000000b: DW_TAG_compile_unit
  // 0x0000000c:   DW_TAG_namespace
  //                 DW_AT_name("a")
  // 0x0000000f:     DW_TAG_structure_type
  //                   DW_AT_name("struct_t")
  // 0x00000019:     DW_TAG_subprogram
  //                   DW_AT_name("foo")
  // 0x0000001e:       DW_TAG_structure_type
  //                     DW_AT_name("struct_t")
  // 0x00000028:       NULL
  // 0x00000029:     NULL
  // 0x0000002a:   NULL
  const char *yamldata = R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_386
DWARF:
  debug_str:
    - ''
  debug_abbrev:
    - ID:              0
      Table:
        - Code:            0x1
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
        - Code:            0x2
          Tag:             DW_TAG_namespace
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_string
        - Code:            0x3
          Tag:             DW_TAG_structure_type
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_string
        - Code:            0x4
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_string
  debug_info:
    - Length:          0x27
      Version:         4
      AbbrevTableID:   0
      AbbrOffset:      0x0
      AddrSize:        8
      Entries:
        - AbbrCode:        0x1
        - AbbrCode:        0x2
          Values:
            - Value:           0xDEADBEEFDEADBEEF
              CStr:            a
        - AbbrCode:        0x3
          Values:
            - Value:           0xDEADBEEFDEADBEEF
              CStr:            struct_t
        - AbbrCode:        0x4
          Values:
            - Value:           0xDEADBEEFDEADBEEF
              CStr:            foo
        - AbbrCode:        0x3
          Values:
            - Value:           0xDEADBEEFDEADBEEF
              CStr:            struct_t
        - AbbrCode:        0x0
        - AbbrCode:        0x0
        - AbbrCode:        0x0)";

  YAMLModuleTester t(yamldata);
  auto *symbol_file =
      llvm::cast<SymbolFileDWARF>(t.GetModule()->GetSymbolFile());
  DWARFUnit *unit = symbol_file->DebugInfo().GetUnitAtIndex(0);
  ASSERT_TRUE(unit);

  auto make_namespace = [](llvm::StringRef name) {
    return CompilerContext(CompilerContextKind::Namespace, ConstString(name));
  };
  auto make_struct = [](llvm::StringRef name) {
    return CompilerContext(CompilerContextKind::ClassOrStruct,
                           ConstString(name));
  };
  // Grab the "a::struct_t" type from the "a" namespace
  DWARFDIE a_struct_die = unit->DIE().GetFirstChild().GetFirstChild();
  ASSERT_TRUE(a_struct_die);
  EXPECT_THAT(
      a_struct_die.GetDeclContext(),
      testing::ElementsAre(make_namespace("a"), make_struct("struct_t")));
  // Grab the "struct_t" defined in the "foo" function.
  DWARFDIE foo_struct_die =
      unit->DIE().GetFirstChild().GetFirstChild().GetSibling().GetFirstChild();
  EXPECT_THAT(foo_struct_die.GetTypeLookupContext(),
              testing::ElementsAre(make_struct("struct_t")));
}

struct GetAttributesTestFixture : public testing::TestWithParam<dw_attr_t> {};

TEST_P(GetAttributesTestFixture, TestGetAttributes_IterationOrder) {
  // Tests that we accumulate all current DIE's attributes first
  // before checking the attributes of the specification.

  const char *yamldata = R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_AARCH64
DWARF:
  debug_str:
    - func
  debug_abbrev:
    - ID:              0
      Table:
        - Code:            0x1
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_language
              Form:            DW_FORM_data2
        - Code:            0x2
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_high_pc
              Form:            DW_FORM_data4
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_declaration
              Form:            DW_FORM_flag_present
            - Attribute:       DW_AT_external
              Form:            DW_FORM_flag_present
            - Attribute:       DW_AT_low_pc
              Form:            DW_FORM_data4
        - Code:            0x3
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_high_pc
              Form:            DW_FORM_data4
            - Attribute:       {0}
              Form:            DW_FORM_ref4
            - Attribute:       DW_AT_low_pc
              Form:            DW_FORM_data4
  debug_info:
     - Version:         5
       UnitType:        DW_UT_compile
       AddrSize:        8
       Entries:

# DW_TAG_compile_unit
#   DW_AT_language [DW_FORM_data2]    (DW_LANG_C_plus_plus)

        - AbbrCode:        0x1
          Values:
            - Value:           0x04

#     DW_TAG_subprogram
#       DW_AT_high_pc [DW_FORM_data4]
#       DW_AT_name [DW_FORM_strp] ("func")
#       DW_AT_low_pc [DW_FORM_data4]
        - AbbrCode:        0x2
          Values:
            - Value:           0xdeadbeef
            - Value:           0x0
            - Value:           0x1
            - Value:           0x1
            - Value:           0xdeadbeef

#     DW_TAG_subprogram
#       DW_AT_high_pc [DW_FORM_data4]
#       DW_AT_specification [DW_FORM_ref4] ("func")
#       DW_AT_low_pc [DW_FORM_data4]
        - AbbrCode:        0x3
          Values:
            - Value:           0xf00dcafe
            - Value:           0xf
            - Value:           0xf00dcafe

        - AbbrCode: 0x0
...
)";
  YAMLModuleTester t(llvm::formatv(yamldata, GetParam()).str());

  DWARFUnit *unit = t.GetDwarfUnit();
  ASSERT_NE(unit, nullptr);
  const DWARFDebugInfoEntry *cu_entry = unit->DIE().GetDIE();
  ASSERT_EQ(cu_entry->Tag(), DW_TAG_compile_unit);
  ASSERT_EQ(unit->GetDWARFLanguageType(), DW_LANG_C_plus_plus);
  DWARFDIE cu_die(unit, cu_entry);

  auto declaration = cu_die.GetFirstChild();
  ASSERT_TRUE(declaration.IsValid());
  ASSERT_EQ(declaration.Tag(), DW_TAG_subprogram);

  auto definition = declaration.GetSibling();
  ASSERT_TRUE(definition.IsValid());
  ASSERT_EQ(definition.Tag(), DW_TAG_subprogram);
  ASSERT_FALSE(definition.GetAttributeValueAsOptionalUnsigned(DW_AT_external));

  auto attrs = definition.GetAttributes(DWARFDebugInfoEntry::Recurse::yes);
  EXPECT_EQ(attrs.Size(), 7U);

  // Check that the attributes on the definition (that are also present
  // on the declaration) take precedence.
  for (auto attr : {DW_AT_low_pc, DW_AT_high_pc}) {
    auto idx = attrs.FindAttributeIndex(attr);
    EXPECT_NE(idx, UINT32_MAX);

    DWARFFormValue form_value;
    auto success = attrs.ExtractFormValueAtIndex(idx, form_value);
    EXPECT_TRUE(success);

    EXPECT_EQ(form_value.Unsigned(), 0xf00dcafe);
  }
}

TEST_P(GetAttributesTestFixture, TestGetAttributes_Cycle) {
  // Tests that GetAttributes can deal with cycles in
  // specifications/abstract origins.
  //
  // Contrived example:
  //
  // func1 -> func3
  //   ^       |
  //   |       v
  //   +------func2

  const char *yamldata = R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_AARCH64
DWARF:
  debug_abbrev:
    - ID:              0
      Table:
        - Code:            0x1
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_language
              Form:            DW_FORM_data2
        - Code:            0x2
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       {0}
              Form:            DW_FORM_ref4
  debug_info:
     - Version:         5
       UnitType:        DW_UT_compile
       AddrSize:        8
       Entries:

        - AbbrCode:        0x1
          Values:
            - Value:           0x04

        - AbbrCode:        0x2
          Values:
            - Value:           0x19

        - AbbrCode:        0x2
          Values:
            - Value:           0xf

        - AbbrCode:        0x2
          Values:
            - Value:           0x14

        - AbbrCode: 0x0
...
)";
  YAMLModuleTester t(llvm::formatv(yamldata, GetParam()).str());

  DWARFUnit *unit = t.GetDwarfUnit();
  ASSERT_NE(unit, nullptr);
  const DWARFDebugInfoEntry *cu_entry = unit->DIE().GetDIE();
  ASSERT_EQ(cu_entry->Tag(), DW_TAG_compile_unit);
  ASSERT_EQ(unit->GetDWARFLanguageType(), DW_LANG_C_plus_plus);
  DWARFDIE cu_die(unit, cu_entry);

  auto func1 = cu_die.GetFirstChild();
  ASSERT_TRUE(func1.IsValid());
  ASSERT_EQ(func1.Tag(), DW_TAG_subprogram);

  auto func2 = func1.GetSibling();
  ASSERT_TRUE(func2.IsValid());
  ASSERT_EQ(func2.Tag(), DW_TAG_subprogram);

  auto func3 = func2.GetSibling();
  ASSERT_TRUE(func3.IsValid());
  ASSERT_EQ(func3.Tag(), DW_TAG_subprogram);

  auto attrs = func1.GetAttributes(DWARFDebugInfoEntry::Recurse::yes);
  EXPECT_EQ(attrs.Size(), 3U);

  // Confirm that the specifications do form a cycle.
  {
    DWARFFormValue form_value;
    auto success = attrs.ExtractFormValueAtIndex(0, form_value);
    ASSERT_TRUE(success);

    EXPECT_EQ(form_value.Reference(), func3);
  }

  {
    DWARFFormValue form_value;
    auto success = attrs.ExtractFormValueAtIndex(1, form_value);
    ASSERT_TRUE(success);

    EXPECT_EQ(form_value.Reference(), func2);
  }

  {
    DWARFFormValue form_value;
    auto success = attrs.ExtractFormValueAtIndex(2, form_value);
    ASSERT_TRUE(success);

    EXPECT_EQ(form_value.Reference(), func1);
  }
}

TEST_P(GetAttributesTestFixture,
       TestGetAttributes_SkipNonApplicableAttributes) {
  // Tests that GetAttributes will omit attributes found through
  // specifications/abstract origins which are not applicable.

  const char *yamldata = R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_AARCH64
DWARF:
  debug_str:
    - func
  debug_abbrev:
    - ID:              0
      Table:
        - Code:            0x1
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_language
              Form:            DW_FORM_data2
        - Code:            0x2
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_declaration
              Form:            DW_FORM_flag_present
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_sibling
              Form:            DW_FORM_ref4
        - Code:            0x3
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_declaration
              Form:            DW_FORM_flag_present
            - Attribute:       {0}
              Form:            DW_FORM_ref4
            - Attribute:       DW_AT_sibling
              Form:            DW_FORM_ref4
  debug_info:
     - Version:         5
       UnitType:        DW_UT_compile
       AddrSize:        8
       Entries:

# DW_TAG_compile_unit
#   DW_AT_language [DW_FORM_data2]    (DW_LANG_C_plus_plus)

        - AbbrCode:        0x1
          Values:
            - Value:           0x04

#     DW_TAG_subprogram
#       DW_AT_declaration
#       DW_AT_name [DW_FORM_strp] ("func")
#       DW_AT_sibling
        - AbbrCode:        0x2
          Values:
            - Value:           0x1
            - Value:           0x0
            - Value:           0x18

#     DW_TAG_subprogram
#       DW_AT_declaration
#       DW_AT_specification [DW_FORM_ref4] ("func")
#       DW_AT_sibling
        - AbbrCode:        0x3
          Values:
            - Value:           0x1
            - Value:           0xf
            - Value:           0xdeadbeef

        - AbbrCode: 0x0
...
)";
  YAMLModuleTester t(llvm::formatv(yamldata, GetParam()).str());

  DWARFUnit *unit = t.GetDwarfUnit();
  ASSERT_NE(unit, nullptr);
  const DWARFDebugInfoEntry *cu_entry = unit->DIE().GetDIE();
  ASSERT_EQ(cu_entry->Tag(), DW_TAG_compile_unit);
  ASSERT_EQ(unit->GetDWARFLanguageType(), DW_LANG_C_plus_plus);
  DWARFDIE cu_die(unit, cu_entry);

  auto declaration = cu_die.GetFirstChild();
  ASSERT_TRUE(declaration.IsValid());
  ASSERT_EQ(declaration.Tag(), DW_TAG_subprogram);

  auto definition = declaration.GetSibling();
  ASSERT_TRUE(definition.IsValid());
  ASSERT_EQ(definition.Tag(), DW_TAG_subprogram);

  auto attrs = definition.GetAttributes(DWARFDebugInfoEntry::Recurse::yes);
  EXPECT_EQ(attrs.Size(), 4U);
  EXPECT_NE(attrs.FindAttributeIndex(DW_AT_name), UINT32_MAX);
  EXPECT_NE(attrs.FindAttributeIndex(GetParam()), UINT32_MAX);

  auto sibling_idx = attrs.FindAttributeIndex(DW_AT_sibling);
  EXPECT_NE(sibling_idx, UINT32_MAX);

  DWARFFormValue form_value;
  auto success = attrs.ExtractFormValueAtIndex(sibling_idx, form_value);
  ASSERT_TRUE(success);

  EXPECT_EQ(form_value.Unsigned(), 0xdeadbeef);
}

TEST_P(GetAttributesTestFixture, TestGetAttributes_NoRecurse) {
  // Tests that GetAttributes will not recurse if Recurse::No is passed to it.

  const char *yamldata = R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_AARCH64
DWARF:
  debug_str:
    - func
  debug_abbrev:
    - ID:              0
      Table:
        - Code:            0x1
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_language
              Form:            DW_FORM_data2
        - Code:            0x2
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
        - Code:            0x3
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_low_pc
              Form:            DW_FORM_data4
            - Attribute:       {0}
              Form:            DW_FORM_ref4
  debug_info:
     - Version:         5
       UnitType:        DW_UT_compile
       AddrSize:        8
       Entries:

# DW_TAG_compile_unit
#   DW_AT_language [DW_FORM_data2]    (DW_LANG_C_plus_plus)

        - AbbrCode:        0x1
          Values:
            - Value:           0x04

#     DW_TAG_subprogram
#       DW_AT_name [DW_FORM_strp] ("func")
        - AbbrCode:        0x2
          Values:
            - Value:           0x0

#     DW_TAG_subprogram
#       DW_AT_low_pc [DW_FORM_data4]
#       DW_AT_specification [DW_FORM_ref4]
        - AbbrCode:        0x3
          Values:
            - Value:           0xdeadbeef
            - Value:           0xf

        - AbbrCode: 0x0
...
)";
  YAMLModuleTester t(llvm::formatv(yamldata, GetParam()).str());

  DWARFUnit *unit = t.GetDwarfUnit();
  ASSERT_NE(unit, nullptr);
  const DWARFDebugInfoEntry *cu_entry = unit->DIE().GetDIE();
  ASSERT_EQ(cu_entry->Tag(), DW_TAG_compile_unit);
  ASSERT_EQ(unit->GetDWARFLanguageType(), DW_LANG_C_plus_plus);
  DWARFDIE cu_die(unit, cu_entry);

  auto declaration = cu_die.GetFirstChild();
  ASSERT_TRUE(declaration.IsValid());
  ASSERT_EQ(declaration.Tag(), DW_TAG_subprogram);

  auto definition = declaration.GetSibling();
  ASSERT_TRUE(definition.IsValid());
  ASSERT_EQ(definition.Tag(), DW_TAG_subprogram);

  auto attrs = definition.GetAttributes(DWARFDebugInfoEntry::Recurse::no);
  EXPECT_EQ(attrs.Size(), 2U);
  EXPECT_EQ(attrs.FindAttributeIndex(DW_AT_name), UINT32_MAX);
  EXPECT_NE(attrs.FindAttributeIndex(GetParam()), UINT32_MAX);
  EXPECT_NE(attrs.FindAttributeIndex(DW_AT_low_pc), UINT32_MAX);
}

TEST_P(GetAttributesTestFixture, TestGetAttributes_InvalidSpec) {
  // Test that GetAttributes doesn't try following invalid
  // specifications (but still add it to the list of attributes).

  const char *yamldata = R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_AARCH64
DWARF:
  debug_str:
    - func
  debug_abbrev:
    - ID:              0
      Table:
        - Code:            0x1
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_language
              Form:            DW_FORM_data2
        - Code:            0x2
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
        - Code:            0x3
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       {0}
              Form:            DW_FORM_ref4
  debug_info:
     - Version:         5
       UnitType:        DW_UT_compile
       AddrSize:        8
       Entries:

# DW_TAG_compile_unit
#   DW_AT_language [DW_FORM_data2]    (DW_LANG_C_plus_plus)

        - AbbrCode:        0x1
          Values:
            - Value:           0x04

#     DW_TAG_subprogram
#       DW_AT_name [DW_FORM_strp] ("func")
        - AbbrCode:        0x2
          Values:
            - Value:           0x0

#     DW_TAG_subprogram
#       DW_AT_specification [DW_FORM_ref4]
        - AbbrCode:        0x3
          Values:
            - Value:           0xdeadbeef

        - AbbrCode: 0x0
...
)";
  YAMLModuleTester t(llvm::formatv(yamldata, GetParam()).str());

  DWARFUnit *unit = t.GetDwarfUnit();
  ASSERT_NE(unit, nullptr);
  const DWARFDebugInfoEntry *cu_entry = unit->DIE().GetDIE();
  ASSERT_EQ(cu_entry->Tag(), DW_TAG_compile_unit);
  ASSERT_EQ(unit->GetDWARFLanguageType(), DW_LANG_C_plus_plus);
  DWARFDIE cu_die(unit, cu_entry);

  auto declaration = cu_die.GetFirstChild();
  ASSERT_TRUE(declaration.IsValid());
  ASSERT_EQ(declaration.Tag(), DW_TAG_subprogram);

  auto definition = declaration.GetSibling();
  ASSERT_TRUE(definition.IsValid());
  ASSERT_EQ(definition.Tag(), DW_TAG_subprogram);

  auto attrs = definition.GetAttributes(DWARFDebugInfoEntry::Recurse::yes);
  EXPECT_EQ(attrs.Size(), 1U);
  EXPECT_EQ(attrs.FindAttributeIndex(DW_AT_name), UINT32_MAX);
  EXPECT_NE(attrs.FindAttributeIndex(GetParam()), UINT32_MAX);
}

TEST(DWARFDIETest, TestGetAttributes_Worklist) {
  // Test that GetAttributes will follow both the abstract origin
  // and specification on a single DIE correctly (omitting non-applicable
  // attributes in the process).

  // Contrived example where
  // f1---> f2 --> f4
  //    `-> f3 `-> f5
  //
  const char *yamldata = R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_AARCH64
DWARF:
  debug_str:
    - foo
    - bar
  debug_abbrev:
    - ID:              0
      Table:
        - Code:            0x1
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_language
              Form:            DW_FORM_data2
        - Code:            0x2
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_specification
              Form:            DW_FORM_ref4
            - Attribute:       DW_AT_abstract_origin
              Form:            DW_FORM_ref4
        - Code:            0x3
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_declaration
              Form:            DW_FORM_flag_present
            - Attribute:       DW_AT_artificial
              Form:            DW_FORM_flag_present

  debug_info:
     - Version:         5
       UnitType:        DW_UT_compile
       AddrSize:        8
       Entries:

        - AbbrCode:        0x1
          Values:
            - Value:           0x04

#     DW_TAG_subprogram ("f1")
#       DW_AT_specification [DW_FORM_ref4] ("f2")
#       DW_AT_abstract_origin [DW_FORM_ref4] ("f3")
        - AbbrCode:        0x2
          Values:
            - Value:           0x18
            - Value:           0x21

#     DW_TAG_subprogram ("f2")
#       DW_AT_specification [DW_FORM_ref4] ("f4")
#       DW_AT_abstract_origin [DW_FORM_ref4] ("f5")
        - AbbrCode:        0x2
          Values:
            - Value:           0x22
            - Value:           0x23

#     DW_TAG_subprogram ("f3")
#       DW_AT_declaration [DW_FORM_flag_present]
#       DW_AT_artificial [DW_FORM_flag_present]
        - AbbrCode:        0x3
          Values:
            - Value:           0x1
            - Value:           0x1

#     DW_TAG_subprogram ("f4")
#       DW_AT_declaration [DW_FORM_flag_present]
#       DW_AT_artificial [DW_FORM_flag_present]
        - AbbrCode:        0x3
          Values:
            - Value:           0x1
            - Value:           0x1

#     DW_TAG_subprogram ("f5")
#       DW_AT_declaration [DW_FORM_flag_present]
#       DW_AT_artificial [DW_FORM_flag_present]
        - AbbrCode:        0x3
          Values:
            - Value:           0x1
            - Value:           0x1

        - AbbrCode: 0x0
...
)";
  YAMLModuleTester t(yamldata);

  DWARFUnit *unit = t.GetDwarfUnit();
  ASSERT_NE(unit, nullptr);
  const DWARFDebugInfoEntry *cu_entry = unit->DIE().GetDIE();
  ASSERT_EQ(cu_entry->Tag(), DW_TAG_compile_unit);
  ASSERT_EQ(unit->GetDWARFLanguageType(), DW_LANG_C_plus_plus);
  DWARFDIE cu_die(unit, cu_entry);

  auto f1 = cu_die.GetFirstChild();
  ASSERT_TRUE(f1.IsValid());
  ASSERT_EQ(f1.Tag(), DW_TAG_subprogram);

  auto attrs = f1.GetAttributes(DWARFDebugInfoEntry::Recurse::yes);
  EXPECT_EQ(attrs.Size(), 7U);
  EXPECT_EQ(attrs.FindAttributeIndex(DW_AT_declaration), UINT32_MAX);
}

INSTANTIATE_TEST_SUITE_P(GetAttributeTests, GetAttributesTestFixture,
                         testing::Values(DW_AT_specification,
                                         DW_AT_abstract_origin));
