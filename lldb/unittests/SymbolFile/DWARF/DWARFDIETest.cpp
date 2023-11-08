//===-- DWARFDIETest.cpp ----------------------------------------------=---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/SymbolFile/DWARF/DWARFDIE.h"
#include "TestingSupport/Symbol/YAMLModuleTester.h"
#include "llvm/ADT/STLExtras.h"
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
