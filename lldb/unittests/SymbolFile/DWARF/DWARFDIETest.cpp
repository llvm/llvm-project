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
