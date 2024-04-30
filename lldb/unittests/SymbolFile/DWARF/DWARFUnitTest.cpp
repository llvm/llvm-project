//===-- DWARFUnitTest.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/SymbolFile/DWARF/DWARFUnit.h"
#include "TestingSupport/Symbol/YAMLModuleTester.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::plugin::dwarf;

TEST(DWARFUnitTest, NullUnitDie) {
  // Make sure we don't crash parsing a null unit DIE.
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
  debug_info:
    - Version:         4
      AddrSize:        8
      Entries:
        - AbbrCode:        0x00000000
)";

  YAMLModuleTester t(yamldata);
  ASSERT_TRUE((bool)t.GetDwarfUnit());

  DWARFUnit *unit = t.GetDwarfUnit();
  const DWARFDebugInfoEntry *die_first = unit->DIE().GetDIE();
  ASSERT_NE(die_first, nullptr);
  EXPECT_TRUE(die_first->IsNULL());
}

TEST(DWARFUnitTest, MissingSentinel) {
  // Make sure we don't crash if the debug info is missing a null DIE sentinel.
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
  debug_info:
    - Version:         4
      AddrSize:        8
      Entries:
        - AbbrCode:        0x00000001
          Values:
            - Value:           0x000000000000000C
)";

  YAMLModuleTester t(yamldata);
  ASSERT_TRUE((bool)t.GetDwarfUnit());

  DWARFUnit *unit = t.GetDwarfUnit();
  const DWARFDebugInfoEntry *die_first = unit->DIE().GetDIE();
  ASSERT_NE(die_first, nullptr);
  EXPECT_EQ(die_first->GetFirstChild(), nullptr);
  EXPECT_EQ(die_first->GetSibling(), nullptr);
}

TEST(DWARFUnitTest, ClangProducer) {
  const char *yamldata = R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_386
DWARF:
  debug_str:
    - 'Apple clang version 13.0.0 (clang-1300.0.29.3)'
  debug_abbrev:
    - Table:
        - Code:            0x00000001
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_producer
              Form:            DW_FORM_strp
  debug_info:
    - Version:         4
      AddrSize:        8
      Entries:
        - AbbrCode:        0x1
          Values:
            - Value:           0x0
        - AbbrCode:        0x0
)";

  YAMLModuleTester t(yamldata);
  DWARFUnit *unit = t.GetDwarfUnit();
  ASSERT_TRUE((bool)unit);
  EXPECT_EQ(unit->GetProducer(), eProducerClang);
  EXPECT_EQ(unit->GetProducerVersion(), llvm::VersionTuple(1300, 0, 29, 3));
}

TEST(DWARFUnitTest, LLVMGCCProducer) {
  const char *yamldata = R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_386
DWARF:
  debug_str:
    - 'i686-apple-darwin11-llvm-gcc-4.2 (GCC) 4.2.1 (Based on Apple Inc. build 5658) (LLVM build 2336.11.00)'
  debug_abbrev:
    - Table:
        - Code:            0x00000001
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_producer
              Form:            DW_FORM_strp
  debug_info:
    - Version:         4
      AddrSize:        8
      Entries:
        - AbbrCode:        0x1
          Values:
            - Value:           0x0
        - AbbrCode:        0x0
)";

  YAMLModuleTester t(yamldata);
  DWARFUnit *unit = t.GetDwarfUnit();
  ASSERT_TRUE((bool)unit);
  EXPECT_EQ(unit->GetProducer(), eProducerLLVMGCC);
}

TEST(DWARFUnitTest, SwiftProducer) {
  const char *yamldata = R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_386
DWARF:
  debug_str:
    - 'Apple Swift version 5.5 (swiftlang-1300.0.31.1 clang-1300.0.29.1)'
  debug_abbrev:
    - Table:
        - Code:            0x00000001
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_producer
              Form:            DW_FORM_strp
  debug_info:
    - Version:         4
      AddrSize:        8
      Entries:
        - AbbrCode:        0x1
          Values:
            - Value:           0x0
        - AbbrCode:        0x0
)";

  YAMLModuleTester t(yamldata);
  DWARFUnit *unit = t.GetDwarfUnit();
  ASSERT_TRUE((bool)unit);
  EXPECT_EQ(unit->GetProducer(), eProducerSwift);
  EXPECT_EQ(unit->GetProducerVersion(), llvm::VersionTuple(1300, 0, 31, 1));
}
