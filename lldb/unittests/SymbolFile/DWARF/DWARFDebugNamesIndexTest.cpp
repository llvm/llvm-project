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
#include "Plugins/SymbolFile/DWARF/DebugNamesDWARFIndex.h"
#include "TestingSupport/Symbol/YAMLModuleTester.h"
#include "llvm/ADT/STLExtras.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::plugin::dwarf;
using StringRef = llvm::StringRef;

static void
check_num_matches(DebugNamesDWARFIndex &index, int expected_num_matches,
                  llvm::ArrayRef<DWARFDeclContext::Entry> ctx_entries) {
  DWARFDeclContext ctx(ctx_entries);
  int num_matches = 0;

  index.GetFullyQualifiedType(ctx, [&](DWARFDIE die) {
    num_matches++;
    return true;
  });
  ASSERT_EQ(num_matches, expected_num_matches);
}

static DWARFDeclContext::Entry make_entry(const char *c) {
  return DWARFDeclContext::Entry(dwarf::DW_TAG_class_type, c);
}

TEST(DWARFDebugNamesIndexTest, FullyQualifiedQueryWithIDXParent) {
  const char *yamldata = R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_386
DWARF:
  debug_str:
    - '1'
    - '2'
    - '3'
  debug_abbrev:
    - Table:
        # We intentionally don't nest types in debug_info: if the nesting is not
        # inferred from debug_names, we want the test to fail.
        - Code:            0x1
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
        - Code:            0x2
          Tag:             DW_TAG_class_type
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
  debug_info:
    - Version:         4
      AddrSize:        8
      Entries:
        - AbbrCode:        0x1
        - AbbrCode:        0x2
          Values:
            - Value:       0x0 # Name "1"
        - AbbrCode:        0x2
          Values:
            - Value:       0x2 # Name "2"
        - AbbrCode:        0x2
          Values:
            - Value:       0x4 # Name "3"
        - AbbrCode:        0x0
  debug_names:
    Abbreviations:
    - Code:   0x11
      Tag: DW_TAG_class_type
      Indices:
        - Idx:   DW_IDX_parent
          Form:  DW_FORM_flag_present
        - Idx:   DW_IDX_die_offset
          Form:  DW_FORM_ref4
    - Code:   0x22
      Tag: DW_TAG_class_type
      Indices:
        - Idx:   DW_IDX_parent
          Form:  DW_FORM_ref4
        - Idx:   DW_IDX_die_offset
          Form:  DW_FORM_ref4
    Entries:
    - Name:   0x0  # strp to Name1
      Code:   0x11
      Values:
        - 0xc      # Die offset to entry named "1"
    - Name:   0x2  # strp to Name2
      Code:   0x22
      Values:
        - 0x0      # Parent = First entry ("1")
        - 0x11     # Die offset to entry named "1:2"
    - Name:   0x4  # strp to Name3
      Code:   0x22
      Values:
        - 0x6      # Parent = Second entry ("1::2")
        - 0x16     # Die offset to entry named "1::2::3"
    - Name:   0x4  # strp to Name3
      Code:   0x11
      Values:
        - 0x16     # Die offset to entry named "3"
)";

  YAMLModuleTester t(yamldata);
  auto *symbol_file =
      llvm::cast<SymbolFileDWARF>(t.GetModule()->GetSymbolFile());
  auto *index = static_cast<DebugNamesDWARFIndex *>(symbol_file->getIndex());
  ASSERT_NE(index, nullptr);

  check_num_matches(*index, 1, {make_entry("1")});
  check_num_matches(*index, 1, {make_entry("2"), make_entry("1")});
  check_num_matches(*index, 1,
                    {make_entry("3"), make_entry("2"), make_entry("1")});
  check_num_matches(*index, 0, {make_entry("2")});
  check_num_matches(*index, 1, {make_entry("3")});
}

TEST(DWARFDebugNamesIndexTest, FullyQualifiedQueryWithoutIDXParent) {
  const char *yamldata = R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_386
DWARF:
  debug_str:
    - '1'
    - '2'
  debug_abbrev:
    - Table:
        - Code:            0x1
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
        - Code:            0x2
          Tag:             DW_TAG_class_type
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
        - Code:            0x3
          Tag:             DW_TAG_class_type
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
  debug_info:
    - Version:         4
      AddrSize:        8
      Entries:
        - AbbrCode:        0x1
        - AbbrCode:        0x2
          Values:
            - Value:       0x0 # Name "1"
        - AbbrCode:        0x3
          Values:
            - Value:       0x2 # Name "2"
        - AbbrCode:        0x0
        - AbbrCode:        0x3
          Values:
            - Value:       0x2 # Name "2"
        - AbbrCode:        0x0
  debug_names:
    Abbreviations:
    - Code:   0x1
      Tag: DW_TAG_class_type
      Indices:
        - Idx:   DW_IDX_die_offset
          Form:  DW_FORM_ref4
    Entries:
    - Name:   0x0  # strp to Name1
      Code:   0x1
      Values:
        - 0xc      # Die offset to entry named "1"
    - Name:   0x2  # strp to Name2
      Code:   0x1
      Values:
        - 0x11     # Die offset to entry named "1::2"
    - Name:   0x2  # strp to Name2
      Code:   0x1
      Values:
        - 0x17     # Die offset to entry named "2"
)";

  YAMLModuleTester t(yamldata);
  auto *symbol_file =
      llvm::cast<SymbolFileDWARF>(t.GetModule()->GetSymbolFile());
  auto *index = static_cast<DebugNamesDWARFIndex *>(symbol_file->getIndex());
  ASSERT_NE(index, nullptr);

  check_num_matches(*index, 1, {make_entry("1")});
  check_num_matches(*index, 1, {make_entry("2"), make_entry("1")});
  check_num_matches(*index, 1, {make_entry("2")});
}
