//===- llvm/unittest/DebugInfo/DWARFDieTest.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/ObjectYAML/DWARFEmitter.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::dwarf;
using object::SectionedAddress;

namespace {

TEST(DWARFDie, getLocations) {
  const char *yamldata = R"(
    debug_abbrev:
      - Code:            0x00000001
        Tag:             DW_TAG_compile_unit
        Children:        DW_CHILDREN_no
        Attributes:
          - Attribute:       DW_AT_location
            Form:            DW_FORM_sec_offset
          - Attribute:       DW_AT_data_member_location
            Form:            DW_FORM_exprloc
          - Attribute:       DW_AT_vtable_elem_location
            Form:            DW_FORM_sec_offset
          - Attribute:       DW_AT_call_data_location
            Form:            DW_FORM_sec_offset
    debug_info:
      - Length:
          TotalLength:     0
        Version:         5
        UnitType:        DW_UT_compile
        AbbrOffset:      0
        AddrSize:        4
        Entries:
          - AbbrCode:        0x00000001
            Values:
              - Value:           12
              - Value:           0x0000000000000001
                BlockData:       [ 0x47 ]
              - Value:           20
              - Value:           25
  )";
  Expected<StringMap<std::unique_ptr<MemoryBuffer>>> Sections =
      DWARFYAML::EmitDebugSections(StringRef(yamldata), /*ApplyFixups=*/true,
                                   /*IsLittleEndian=*/true);
  ASSERT_THAT_EXPECTED(Sections, Succeeded());
  std::vector<uint8_t> Loclists{
      // Header
      0, 0, 0, 0, // Length
      5, 0,       // Version
      4,          // Address size
      0,          // Segment selector size
      0, 0, 0, 0, // Offset entry count
      // First location list.
      DW_LLE_start_length, // First entry
      1, 0, 0, 0,          // Start offset
      2,                   // Length
      0,                   // Expression length
      DW_LLE_end_of_list,
      // Second location list.
      DW_LLE_startx_length, // First entry
      1,                    // Start index
      2,                    // Length
      0,                    // Expression length
      DW_LLE_end_of_list,
      // Third location list.
      DW_LLE_start_length, // First entry
      1, 0, 0, 0,          // Start offset
      2,                   // Length
      0,                   // Expression length
                           // end_of_list intentionally missing
  };
  Loclists[0] = Loclists.size() - 4;
  Sections->try_emplace(
      "debug_loclists",
      MemoryBuffer::getMemBuffer(toStringRef(Loclists), "debug_loclists",
                                 /*RequiresNullTerminator=*/false));
  std::unique_ptr<DWARFContext> Ctx =
      DWARFContext::create(*Sections, 4, /*isLittleEndian=*/true);
  DWARFCompileUnit *CU = Ctx->getCompileUnitForOffset(0);
  ASSERT_NE(nullptr, CU);
  DWARFDie Die = CU->getUnitDIE();
  ASSERT_TRUE(Die.isValid());

  EXPECT_THAT_EXPECTED(Die.getLocations(DW_AT_location),
                       HasValue(testing::ElementsAre(DWARFLocationExpression{
                           DWARFAddressRange{1, 3}, {}})));

  EXPECT_THAT_EXPECTED(
      Die.getLocations(DW_AT_data_member_location),
      HasValue(testing::ElementsAre(DWARFLocationExpression{None, {0x47}})));

  EXPECT_THAT_EXPECTED(
      Die.getLocations(DW_AT_vtable_elem_location),
      Failed<ErrorInfoBase>(testing::Property(
          &ErrorInfoBase::message,
          "Unable to resolve indirect address 1 for: DW_LLE_startx_length")));

  EXPECT_THAT_EXPECTED(Die.getLocations(DW_AT_call_data_location),
                       Failed<ErrorInfoBase>(testing::Property(
                           &ErrorInfoBase::message, "unexpected end of data")));

  EXPECT_THAT_EXPECTED(
      Die.getLocations(DW_AT_call_data_value),
      Failed<ErrorInfoBase>(testing::Property(&ErrorInfoBase::message,
                                              "No DW_AT_call_data_value")));
}

} // end anonymous namespace
