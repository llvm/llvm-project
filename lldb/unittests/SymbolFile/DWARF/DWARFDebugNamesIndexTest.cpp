//===-- DWARFDebugNamesIndexTest.cpp
//----------------------------------------------=---===//
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
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/Symbol/YAMLModuleTester.h"
#include "lldb/Core/Debugger.h"
#include "lldb/lldb-private-enumerations.h"
#include "llvm/ADT/STLExtras.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::plugin::dwarf;
using StringRef = llvm::StringRef;

class DWARFDebugNamesIndexTest : public testing::Test {
public:
  void SetUp() override { Debugger::Initialize(nullptr); }

  void TearDown() override { Debugger::Terminate(); }
};

static void
check_num_matches(DebugNamesDWARFIndex &index, int expected_num_matches,
                  llvm::ArrayRef<DWARFDeclContext::Entry> ctx_entries) {
  DWARFDeclContext ctx(ctx_entries);
  int num_matches = 0;

  index.GetFullyQualifiedType(ctx, [&](DWARFDIE die) {
    num_matches++;
    return IterationAction::Continue;
  });
  ASSERT_EQ(num_matches, expected_num_matches);
}

static DWARFDeclContext::Entry make_entry(const char *c) {
  return DWARFDeclContext::Entry(llvm::dwarf::DW_TAG_class_type, c);
}

TEST_F(DWARFDebugNamesIndexTest, FullyQualifiedQueryWithIDXParent) {
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

TEST_F(DWARFDebugNamesIndexTest, FullyQualifiedQueryWithoutIDXParent) {
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

TEST_F(DWARFDebugNamesIndexTest, CaseInsesitiveQuery) {
  const char *yamldata = R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_X86_64
DWARF:
  debug_str:
    - 'num_int'
  debug_abbrev:
    - Table:
        - Code:            0x1
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_language
              Form:            DW_FORM_data2
            - Attribute:       DW_AT_identifier_case
              Form:            DW_FORM_data1
        - Code:            0x2
          Tag:             DW_TAG_variable
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_const_value    
              Form:            DW_FORM_udata        
  debug_info:
    - Version:         4
      AddrSize:        8
      Entries:
        - AbbrCode:        0x1
          Values:
            - Value:       0x0008 # DW_LANG_Fortran90
            - Value:       0x03   # DW_ID_case_insensitive (0x3)
        - AbbrCode:        0x2
          Values:
            - Value:       0x0    
            - Value:       0x2a                     
        - AbbrCode:        0x0
)";

  YAMLModuleTester t(yamldata);
  auto *symbol_file =
      llvm::cast<SymbolFileDWARF>(t.GetModule()->GetSymbolFile());
  auto *index = symbol_file->getIndex();
  int num_matches = 0;
  index->GetGlobalVariables(ConstString("NUM_INT"), [&](DWARFDIE die) {
    num_matches++;
    return IterationAction::Stop;
  });
  EXPECT_EQ(num_matches, 1);

  num_matches = 0;
  index->GetGlobalVariables(ConstString("num_int"), [&](DWARFDIE die) {
    num_matches++;
    return IterationAction::Stop;
  });
  EXPECT_EQ(num_matches, 1);

  num_matches = 0;
  index->GetGlobalVariables(ConstString("NuM_iNT"), [&](DWARFDIE die) {
    num_matches++;
    return IterationAction::Stop;
  });
  EXPECT_EQ(num_matches, 1);

  num_matches = 0;
  index->GetGlobalVariables(ConstString("num_in"), [&](DWARFDIE die) {
    num_matches++;
    return IterationAction::Stop;
  });
  EXPECT_EQ(num_matches, 0);
}

TEST_F(DWARFDebugNamesIndexTest, CasesSesitiveDefaultQuery) {
  const char *yamldata = R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_X86_64
DWARF:
  debug_abbrev:
    - Table:
        - Code:            0x1
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_language
              Form:            DW_FORM_data2
        - Code:            0x2
          Tag:             DW_TAG_variable
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_string
            - Attribute:       DW_AT_const_value    
              Form:            DW_FORM_udata        

    - Table:
        - Code:            0x1
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_language
              Form:            DW_FORM_data2
            - Attribute:       DW_AT_identifier_case
              Form:            DW_FORM_data1
        - Code:            0x2
          Tag:             DW_TAG_variable
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_string
            - Attribute:       DW_AT_const_value    
              Form:            DW_FORM_udata        

  debug_info:
    - Version:         4
      AddrSize:        8
      Entries:
        - AbbrCode:        0x1
          Values:
            - Value:       0x0004 # DW_LANG_C_plus_plus
        - AbbrCode:        0x2
          Values:
            - CStr:        'SensitiveVar'
            - Value:       0x2a                     
        - AbbrCode:        0x0

    - Version:         4
      AddrSize:        8
      Entries:
        - AbbrCode:        0x1
          Values:
            - Value:       0x0008 # DW_LANG_Fortran90
            - Value:       0x03   # DW_ID_case_insensitive
        - AbbrCode:        0x2
          Values:
            - CStr:        'InsensitiveVar'
            - Value:       0x2a                    
        - AbbrCode:        0x0
)";
  // If one Compile unit is case-insensitive and the other is case-sensitive we
  // should default to all compile units being case-sensitive.
  YAMLModuleTester t(yamldata);
  auto *symbol_file =
      llvm::cast<SymbolFileDWARF>(t.GetModule()->GetSymbolFile());
  auto *index = symbol_file->getIndex();

  int sens_exact = 0;
  index->GetGlobalVariables(ConstString("SensitiveVar"), [&](DWARFDIE die) {
    sens_exact++;
    return IterationAction::Stop;
  });
  EXPECT_EQ(sens_exact, 1);

  int sens_mismatch = 0;
  index->GetGlobalVariables(ConstString("sensitivevar"), [&](DWARFDIE die) {
    sens_mismatch++;
    return IterationAction::Stop;
  });
  EXPECT_EQ(sens_mismatch, 0);

  int insens_exact = 0;
  index->GetGlobalVariables(ConstString("InsensitiveVar"), [&](DWARFDIE die) {
    insens_exact++;
    return IterationAction::Stop;
  });
  EXPECT_EQ(insens_exact, 1);

  int insens_mismatch = 0;
  index->GetGlobalVariables(ConstString("insensitivevar"), [&](DWARFDIE die) {
    insens_mismatch++;
    return IterationAction::Stop;
  });
  EXPECT_EQ(insens_mismatch, 0);
}