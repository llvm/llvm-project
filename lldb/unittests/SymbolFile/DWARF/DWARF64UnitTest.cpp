//===-- DWARF64UnitTest.cpp------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/SymbolFile/DWARF/DWARFDebugInfo.h"
#include "Plugins/SymbolFile/DWARF/DWARFUnit.h"
#include "TestingSupport/Symbol/YAMLModuleTester.h"

using namespace lldb_private;
using namespace lldb_private::plugin::dwarf;
using namespace llvm::dwarf;

TEST(DWARF64UnitTest, DWARF64DebugInfoAndCU) {
  const char *yamldata = R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_PPC64
DWARF:
  debug_str:
    - 'clang version 18.1.8 (clang-18.1.8-1)'
    - 'main'
  debug_abbrev:
    - Table:
        - Code:            0x1
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_producer
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_language
              Form:            DW_FORM_data2
            - Attribute:       DW_AT_stmt_list 
              Form:            DW_FORM_sec_offset 
        - Code:            0x02
          Tag:             DW_TAG_subprogram 
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_name 
              Form:            DW_FORM_strp
  debug_info:
    - Format:          DWARF64
      Version:         4
      AbbrOffset:      0x0
      AddrSize:        8
      Entries:
        - AbbrCode:        0x1
          Values:
            - Value:           0x0
            - Value:           0x04
            - Value:           0x0
        - AbbrCode:        0x2
          Values:
            - Value:           0x1
        - AbbrCode:	   0x0
)";

  YAMLModuleTester t(yamldata);
  auto *symbol_file =
      llvm::cast<SymbolFileDWARF>(t.GetModule()->GetSymbolFile());
  DWARFUnit *unit = symbol_file->DebugInfo().GetUnitAtIndex(0);
  ASSERT_TRUE(unit);
  ASSERT_EQ(unit->GetFormParams().Format, DwarfFormat::DWARF64);
  ASSERT_EQ(unit->GetVersion(), 4);
  ASSERT_EQ(unit->GetAddressByteSize(), 8);

  DWARFFormValue form_value;
  const DWARFDebugInfoEntry *cu_entry = unit->DIE().GetDIE();
  ASSERT_EQ(cu_entry->Tag(), DW_TAG_compile_unit);
  ASSERT_EQ(unit->GetProducer(), eProducerClang);
  ASSERT_EQ(unit->GetDWARFLanguageType(), DW_LANG_C_plus_plus);
  auto attrs = cu_entry->GetAttributes(unit, DWARFDebugInfoEntry::Recurse::yes);
  attrs.ExtractFormValueAtIndex(2, form_value); // Validate DW_AT_stmt_list
  ASSERT_EQ(form_value.Unsigned(), 0UL);

  DWARFDIE cu_die(unit, cu_entry);
  auto declaration = cu_die.GetFirstChild();
  ASSERT_TRUE(declaration.IsValid());
  ASSERT_EQ(declaration.Tag(), DW_TAG_subprogram);
}

TEST(DWARF64UnitTest, DWARF5StrTable) {
  const char *yamldata = R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2MSB
  Type:    ET_EXEC
  Machine: EM_PPC64
DWARF:
  debug_str:
    - 'clang version 18.1.8 (clang-18.1.8-1)'
    - 'main.c'
    - 'foo'
    - 'main'
  debug_abbrev:
    - Table:
        - Code:            0x1
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_producer
              Form:            DW_FORM_strx1
            - Attribute:       DW_AT_language
              Form:            DW_FORM_data2
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strx1
            - Attribute:       DW_AT_str_offsets_base
              Form:            DW_FORM_sec_offset
        - Code:            0x2
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strx1
  debug_info:
    - Format:          DWARF64
      Version:         0x05
      UnitType:        DW_UT_compile
      AbbrOffset:      0x0
      AddrSize:        0x08
      Entries:
        - AbbrCode:        0x1
          Values:
            - Value:           0x0
            - Value:           0x04
            - Value:           0x1
            - Value:           0x00000010
        - AbbrCode:        0x2
          Values:
            - Value:           0x2
        - AbbrCode:        0x2
          Values:
            - Value:           0x3
        - AbbrCode:	   0x0

  debug_str_offsets:
    - Format:           DWARF64
      Version:          "0x05"
      Offsets:
          - 0x00000000
          - 0x00000026
          - 0x0000002d
          - 0x00000031
)";

  YAMLModuleTester t(yamldata);
  auto *symbol_file =
      llvm::cast<SymbolFileDWARF>(t.GetModule()->GetSymbolFile());
  DWARFUnit *unit = symbol_file->DebugInfo().GetUnitAtIndex(0);
  ASSERT_TRUE(unit);
  ASSERT_EQ(unit->GetFormParams().Format, DwarfFormat::DWARF64);
  ASSERT_EQ(unit->GetVersion(), 5);
  ASSERT_EQ(unit->GetAddressByteSize(), 8);

  DWARFFormValue form_value;
  const DWARFDebugInfoEntry *cu_entry = unit->DIE().GetDIE();
  ASSERT_EQ(cu_entry->Tag(), DW_TAG_compile_unit);
  ASSERT_EQ(unit->GetProducer(), eProducerClang);
  ASSERT_EQ(unit->GetDWARFLanguageType(), DW_LANG_C_plus_plus);
  auto attrs = cu_entry->GetAttributes(unit, DWARFDebugInfoEntry::Recurse::yes);
  attrs.ExtractFormValueAtIndex(3,
                                form_value); // Validate DW_AT_str_offsets_bae
  ASSERT_EQ(form_value.Unsigned(), 0x00000010UL);
  DWARFDIE cu_die(unit, cu_entry);
  ASSERT_EQ(ConstString(cu_die.GetName()), "main.c");

  auto func_foo = cu_die.GetFirstChild();
  ASSERT_TRUE(func_foo.IsValid());
  ASSERT_EQ(func_foo.Tag(), DW_TAG_subprogram);
  ASSERT_EQ(ConstString(func_foo.GetName()), "foo");

  auto func_main = func_foo.GetSibling();
  ASSERT_TRUE(func_main.IsValid());
  ASSERT_EQ(func_main.Tag(), DW_TAG_subprogram);
  ASSERT_EQ(ConstString(func_main.GetName()), "main");
}
