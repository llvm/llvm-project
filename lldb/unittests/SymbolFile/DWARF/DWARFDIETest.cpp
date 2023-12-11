//===-- DWARFDIETest.cpp ----------------------------------------------=---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/SymbolFile/DWARF/DWARFDIE.h"
#include "Plugins/SymbolFile/DWARF/DWARFDeclContext.h"
#include "TestingSupport/Symbol/YAMLModuleTester.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::plugin::dwarf;

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

TEST(DWARFDIETest, DeclContext) {
  const char *yamldata = R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_386
DWARF:
  debug_str:
    - 'mynamespace'
    - 'mystruct'
    - 'mytype'
  debug_abbrev:
    - Table:
        - Code:            0x00000001
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_language
              Form:            DW_FORM_data2
        - Code:            0x00000002
          Tag:             DW_TAG_structure_type
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
        - Code:            0x00000003
          Tag:             DW_TAG_base_type
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
        - Code:            0x00000004
          Tag:             DW_TAG_namespace
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
  debug_info:
    - Version:         4
      AddrSize:        8
      Entries:
        - AbbrCode:        0x00000001 # compile_unit
          Values:
            - Value:           0x000000000000000C
        - AbbrCode:        0x00000004 # namespace
          Values:
            - Value:           0x0000000000000000 # DW_ATE_strp
        - AbbrCode:        0x00000002 # structure_type
          Values:
            - Value:           0x000000000000000c # DW_ATE_strp
        - AbbrCode:        0x00000003 # base_type
          Values:
            - Value:           0x0000000000000015 # DW_ATE_strp
        - AbbrCode:        0x00000000
)";

  YAMLModuleTester t(yamldata);
  DWARFUnit *unit = t.GetDwarfUnit();
  ASSERT_TRUE(unit != nullptr);
  auto &ctx = unit->GetSymbolFileDWARF();

  auto top_level_die = unit->DIE();
  {
    ASSERT_TRUE(top_level_die);
    auto top_level_ctx = ctx.GetDWARFDeclContext(top_level_die);
    auto top_level_name = llvm::StringRef(top_level_ctx.GetQualifiedName());
    ASSERT_EQ(top_level_name, "");
  }

  auto namespace_die = top_level_die.GetFirstChild();
  {
    ASSERT_TRUE(namespace_die);
    auto namespace_ctx = ctx.GetDWARFDeclContext(namespace_die);
    auto namespace_name = llvm::StringRef(namespace_ctx.GetQualifiedName());
    ASSERT_EQ(namespace_name, "::mynamespace");
    auto namespace_names = namespace_ctx.GetQualifiedNameAsVector();
    ASSERT_EQ(namespace_names.size(), 1u);
    ASSERT_EQ(namespace_names.front(), "mynamespace");
  }

  auto struct_die = namespace_die.GetFirstChild();
  {
    ASSERT_TRUE(struct_die);
    auto struct_ctx = ctx.GetDWARFDeclContext(struct_die);
    auto struct_name = llvm::StringRef(struct_ctx.GetQualifiedName());
    ASSERT_EQ(struct_name, "mynamespace::mystruct");
    auto struct_names = struct_ctx.GetQualifiedNameAsVector();
    ASSERT_EQ(struct_names.size(), 2u);
    ASSERT_EQ(struct_names[0], "mystruct");
    ASSERT_EQ(struct_names[1], "mynamespace");
  }
  auto simple_type_die = struct_die.GetFirstChild();
  {
    ASSERT_TRUE(simple_type_die);
    auto simple_type_ctx = ctx.GetDWARFDeclContext(simple_type_die);
    auto simple_type_name = llvm::StringRef(simple_type_ctx.GetQualifiedName());
    ASSERT_EQ(simple_type_name, "mynamespace::mystruct::mytype");
    auto simple_type_names = simple_type_ctx.GetQualifiedNameAsVector();
    ASSERT_EQ(simple_type_names.size(), 3u);
    ASSERT_EQ(simple_type_names[0], "mytype");
    ASSERT_EQ(simple_type_names[1], "mystruct");
    ASSERT_EQ(simple_type_names[2], "mynamespace");
  }
}
