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
#include "lldb/Symbol/Variable.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/lldb-private-enumerations.h"
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
    return IterationAction::Continue;
  });
  ASSERT_EQ(num_matches, expected_num_matches);
}

static DWARFDeclContext::Entry make_entry(const char *c) {
  return DWARFDeclContext::Entry(llvm::dwarf::DW_TAG_class_type, c);
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

TEST(DWARFDebugNamesIndexTest, FindGlobalVariablesStaticDataMember) {
  const char *yamldata = R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_X86_64
DWARF:
  debug_str:
    - 'int'
    - 'A'
    - 'int_val'
    - 'ns'
    - 'Foo'
    - 'mem'
  debug_abbrev:
    - Table:
        - Code:            0x1
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_language
              Form:            DW_FORM_data2
        - Code:            0x2
          Tag:             DW_TAG_base_type
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_encoding
              Form:            DW_FORM_data1
            - Attribute:       DW_AT_byte_size
              Form:            DW_FORM_data1
        - Code:            0x3
          Tag:             DW_TAG_structure_type
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_byte_size
              Form:            DW_FORM_data1
        - Code:            0x4
          Tag:             DW_TAG_variable
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_type
              Form:            DW_FORM_ref4
            - Attribute:       DW_AT_declaration
              Form:            DW_FORM_flag_present
            - Attribute:       DW_AT_const_value
              Form:            DW_FORM_data4
        - Code:            0x5
          Tag:             DW_TAG_namespace
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
        - Code:            0x6
          Tag:             DW_TAG_member
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_type
              Form:            DW_FORM_ref4
            - Attribute:       DW_AT_declaration
              Form:            DW_FORM_flag_present
            - Attribute:       DW_AT_const_value
              Form:            DW_FORM_data4
  debug_info:
    - Version:         5
      UnitType:        DW_UT_compile
      AddrSize:        8
      Entries:
        - AbbrCode:        0x1
          Values:
            - Value:       0x0021 # DW_LANG_C_plus_plus_14
        - AbbrCode:        0x2    # 0x0f: base_type "int"
          Values:
            - Value:       0x0    # "int"
            - Value:       0x5    # DW_ATE_signed
            - Value:       0x4
        - AbbrCode:        0x3    # 0x16: structure_type "A"
          Values:
            - Value:       0x4    # "A"
            - Value:       0x1
        - AbbrCode:        0x4    # 0x1c: variable "int_val"
          Values:
            - Value:       0x6    # "int_val"
            - Value:       0x0f   # -> int
            - Value:       0x1
            - Value:       0x2a   # 42
        - AbbrCode:        0x0    # 0x29: end A
        - AbbrCode:        0x5    # 0x2a: namespace "ns"
          Values:
            - Value:       0xe    # "ns"
        - AbbrCode:        0x3    # 0x2f: structure_type "Foo"
          Values:
            - Value:       0x11   # "Foo"
            - Value:       0x1
        - AbbrCode:        0x6    # 0x35: member "mem"
          Values:
            - Value:       0x15   # "mem"
            - Value:       0x0f   # -> int
            - Value:       0x1
            - Value:       0x0a   # 10
        - AbbrCode:        0x0    # 0x42: end Foo
        - AbbrCode:        0x0    # 0x43: end ns
        - AbbrCode:        0x0    # 0x44: end CU
  debug_names:
    Abbreviations:
    - Code:   0x1
      Tag:    DW_TAG_structure_type
      Indices:
        - Idx:   DW_IDX_die_offset
          Form:  DW_FORM_ref4
    - Code:   0x2
      Tag:    DW_TAG_namespace
      Indices:
        - Idx:   DW_IDX_die_offset
          Form:  DW_FORM_ref4
    Entries:
    - Name:   0x4   # "A"
      Code:   0x1
      Values:
        - 0x16
    - Name:   0xe   # "ns"
      Code:   0x2
      Values:
        - 0x2a
    - Name:   0x11  # "Foo"
      Code:   0x1
      Values:
        - 0x2f
)";

  YAMLModuleTester t(yamldata);
  auto *symbol_file =
      llvm::cast<SymbolFileDWARF>(t.GetModule()->GetSymbolFile());
  auto *index = static_cast<DebugNamesDWARFIndex *>(symbol_file->getIndex());
  ASSERT_NE(index, nullptr);

  VariableList vars;
  symbol_file->FindGlobalVariables(ConstString("A::int_val"),
                                   CompilerDeclContext(), UINT32_MAX, vars);
  ASSERT_EQ(vars.GetSize(), 1u);
  VariableSP var_sp = vars.GetVariableAtIndex(0);
  ASSERT_TRUE(var_sp);
  EXPECT_EQ(var_sp->GetName(), ConstString("A::int_val"));
  EXPECT_TRUE(var_sp->GetLocationIsConstantValueData());

  VariableList ns_vars;
  symbol_file->FindGlobalVariables(ConstString("ns::Foo::mem"),
                                   CompilerDeclContext(), UINT32_MAX, ns_vars);
  ASSERT_EQ(ns_vars.GetSize(), 1u);
  VariableSP ns_var_sp = ns_vars.GetVariableAtIndex(0);
  ASSERT_TRUE(ns_var_sp);
  EXPECT_EQ(ns_var_sp->GetName(), ConstString("ns::Foo::mem"));
  EXPECT_TRUE(ns_var_sp->GetLocationIsConstantValueData());
}
