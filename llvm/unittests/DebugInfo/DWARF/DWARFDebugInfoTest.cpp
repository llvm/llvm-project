//===- llvm/unittest/DebugInfo/DWARFDebugInfoTest.cpp ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DwarfGenerator.h"
#include "DwarfUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/DebugInfo/DWARF/DWARFCompileUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/DebugInfo/DWARF/DWARFVerifier.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/ObjectYAML/DWARFEmitter.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <string>

using namespace llvm;
using namespace dwarf;
using namespace utils;

namespace {

template <uint16_t Version, class AddrType, class RefAddrType>
void TestAllForms() {
  Triple Triple = getDefaultTargetTripleForAddrSize(sizeof(AddrType));
  if (!isConfigurationSupported(Triple))
    return;

  // Test that we can decode all DW_FORM values correctly.
  const AddrType AddrValue = (AddrType)0x0123456789abcdefULL;
  const uint8_t BlockData[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
  const uint32_t BlockSize = sizeof(BlockData);
  const RefAddrType RefAddr = 0x12345678;
  const uint8_t Data1 = 0x01U;
  const uint16_t Data2 = 0x2345U;
  const uint32_t Data4 = 0x6789abcdU;
  const uint64_t Data8 = 0x0011223344556677ULL;
  const uint64_t Data8_2 = 0xAABBCCDDEEFF0011ULL;
  const uint8_t Data16[16] = {1, 2,  3,  4,  5,  6,  7,  8,
                              9, 10, 11, 12, 13, 14, 15, 16};
  const int64_t SData = INT64_MIN;
  const int64_t ICSData = INT64_MAX; // DW_FORM_implicit_const SData
  const uint64_t UData[] = {UINT64_MAX - 1, UINT64_MAX - 2, UINT64_MAX - 3,
                            UINT64_MAX - 4, UINT64_MAX - 5, UINT64_MAX - 6,
                            UINT64_MAX - 7, UINT64_MAX - 8, UINT64_MAX - 9};
#define UDATA_1 18446744073709551614ULL
  const uint32_t Dwarf32Values[] = {1, 2, 3, 4, 5, 6, 7, 8};
  const char *StringValue = "Hello";
  const char *StrpValue = "World";
  const char *StrxValue = "Indexed";
  const char *Strx1Value = "Indexed1";
  const char *Strx2Value = "Indexed2";
  const char *Strx3Value = "Indexed3";
  const char *Strx4Value = "Indexed4";

  auto ExpectedDG = dwarfgen::Generator::create(Triple, Version);
  ASSERT_THAT_EXPECTED(ExpectedDG, Succeeded());
  dwarfgen::Generator *DG = ExpectedDG.get().get();
  dwarfgen::CompileUnit &CU = DG->addCompileUnit();
  dwarfgen::DIE CUDie = CU.getUnitDIE();

  if (Version >= 5)
    CUDie.addStrOffsetsBaseAttribute();

  uint16_t Attr = DW_AT_lo_user;

  //----------------------------------------------------------------------
  // Test address forms
  //----------------------------------------------------------------------
  const auto Attr_DW_FORM_addr = static_cast<dwarf::Attribute>(Attr++);
  CUDie.addAttribute(Attr_DW_FORM_addr, DW_FORM_addr, AddrValue);

  //----------------------------------------------------------------------
  // Test block forms
  //----------------------------------------------------------------------
  const auto Attr_DW_FORM_block = static_cast<dwarf::Attribute>(Attr++);
  CUDie.addAttribute(Attr_DW_FORM_block, DW_FORM_block, BlockData, BlockSize);

  const auto Attr_DW_FORM_block1 = static_cast<dwarf::Attribute>(Attr++);
  CUDie.addAttribute(Attr_DW_FORM_block1, DW_FORM_block1, BlockData, BlockSize);

  const auto Attr_DW_FORM_block2 = static_cast<dwarf::Attribute>(Attr++);
  CUDie.addAttribute(Attr_DW_FORM_block2, DW_FORM_block2, BlockData, BlockSize);

  const auto Attr_DW_FORM_block4 = static_cast<dwarf::Attribute>(Attr++);
  CUDie.addAttribute(Attr_DW_FORM_block4, DW_FORM_block4, BlockData, BlockSize);

  // We handle data16 as a block form.
  const auto Attr_DW_FORM_data16 = static_cast<dwarf::Attribute>(Attr++);
  if (Version >= 5)
    CUDie.addAttribute(Attr_DW_FORM_data16, DW_FORM_data16, Data16, 16);

  //----------------------------------------------------------------------
  // Test data forms
  //----------------------------------------------------------------------
  const auto Attr_DW_FORM_data1 = static_cast<dwarf::Attribute>(Attr++);
  CUDie.addAttribute(Attr_DW_FORM_data1, DW_FORM_data1, Data1);

  const auto Attr_DW_FORM_data2 = static_cast<dwarf::Attribute>(Attr++);
  CUDie.addAttribute(Attr_DW_FORM_data2, DW_FORM_data2, Data2);

  const auto Attr_DW_FORM_data4 = static_cast<dwarf::Attribute>(Attr++);
  CUDie.addAttribute(Attr_DW_FORM_data4, DW_FORM_data4, Data4);

  const auto Attr_DW_FORM_data8 = static_cast<dwarf::Attribute>(Attr++);
  CUDie.addAttribute(Attr_DW_FORM_data8, DW_FORM_data8, Data8);

  //----------------------------------------------------------------------
  // Test string forms
  //----------------------------------------------------------------------
  const auto Attr_DW_FORM_string = static_cast<dwarf::Attribute>(Attr++);
  CUDie.addAttribute(Attr_DW_FORM_string, DW_FORM_string, StringValue);

  const auto Attr_DW_FORM_strx = static_cast<dwarf::Attribute>(Attr++);
  const auto Attr_DW_FORM_strx1 = static_cast<dwarf::Attribute>(Attr++);
  const auto Attr_DW_FORM_strx2 = static_cast<dwarf::Attribute>(Attr++);
  const auto Attr_DW_FORM_strx3 = static_cast<dwarf::Attribute>(Attr++);
  const auto Attr_DW_FORM_strx4 = static_cast<dwarf::Attribute>(Attr++);
  if (Version >= 5) {
    CUDie.addAttribute(Attr_DW_FORM_strx, DW_FORM_strx, StrxValue);
    CUDie.addAttribute(Attr_DW_FORM_strx1, DW_FORM_strx1, Strx1Value);
    CUDie.addAttribute(Attr_DW_FORM_strx2, DW_FORM_strx2, Strx2Value);
    CUDie.addAttribute(Attr_DW_FORM_strx3, DW_FORM_strx3, Strx3Value);
    CUDie.addAttribute(Attr_DW_FORM_strx4, DW_FORM_strx4, Strx4Value);
  }

  const auto Attr_DW_FORM_strp = static_cast<dwarf::Attribute>(Attr++);
  CUDie.addAttribute(Attr_DW_FORM_strp, DW_FORM_strp, StrpValue);

  //----------------------------------------------------------------------
  // Test reference forms
  //----------------------------------------------------------------------
  const auto Attr_DW_FORM_ref_addr = static_cast<dwarf::Attribute>(Attr++);
  CUDie.addAttribute(Attr_DW_FORM_ref_addr, DW_FORM_ref_addr, RefAddr);

  const auto Attr_DW_FORM_ref1 = static_cast<dwarf::Attribute>(Attr++);
  CUDie.addAttribute(Attr_DW_FORM_ref1, DW_FORM_ref1, Data1);

  const auto Attr_DW_FORM_ref2 = static_cast<dwarf::Attribute>(Attr++);
  CUDie.addAttribute(Attr_DW_FORM_ref2, DW_FORM_ref2, Data2);

  const auto Attr_DW_FORM_ref4 = static_cast<dwarf::Attribute>(Attr++);
  CUDie.addAttribute(Attr_DW_FORM_ref4, DW_FORM_ref4, Data4);

  const auto Attr_DW_FORM_ref8 = static_cast<dwarf::Attribute>(Attr++);
  CUDie.addAttribute(Attr_DW_FORM_ref8, DW_FORM_ref8, Data8);

  const auto Attr_DW_FORM_ref_sig8 = static_cast<dwarf::Attribute>(Attr++);
  if (Version >= 4)
    CUDie.addAttribute(Attr_DW_FORM_ref_sig8, DW_FORM_ref_sig8, Data8_2);

  const auto Attr_DW_FORM_ref_udata = static_cast<dwarf::Attribute>(Attr++);
  CUDie.addAttribute(Attr_DW_FORM_ref_udata, DW_FORM_ref_udata, UData[0]);

  //----------------------------------------------------------------------
  // Test flag forms
  //----------------------------------------------------------------------
  const auto Attr_DW_FORM_flag_true = static_cast<dwarf::Attribute>(Attr++);
  CUDie.addAttribute(Attr_DW_FORM_flag_true, DW_FORM_flag, true);

  const auto Attr_DW_FORM_flag_false = static_cast<dwarf::Attribute>(Attr++);
  CUDie.addAttribute(Attr_DW_FORM_flag_false, DW_FORM_flag, false);

  const auto Attr_DW_FORM_flag_present = static_cast<dwarf::Attribute>(Attr++);
  if (Version >= 4)
    CUDie.addAttribute(Attr_DW_FORM_flag_present, DW_FORM_flag_present);

  //----------------------------------------------------------------------
  // Test SLEB128 based forms
  //----------------------------------------------------------------------
  const auto Attr_DW_FORM_sdata = static_cast<dwarf::Attribute>(Attr++);
  CUDie.addAttribute(Attr_DW_FORM_sdata, DW_FORM_sdata, SData);

  const auto Attr_DW_FORM_implicit_const =
    static_cast<dwarf::Attribute>(Attr++);
  if (Version >= 5)
    CUDie.addAttribute(Attr_DW_FORM_implicit_const, DW_FORM_implicit_const,
                       ICSData);

  //----------------------------------------------------------------------
  // Test ULEB128 based forms
  //----------------------------------------------------------------------
  const auto Attr_DW_FORM_udata = static_cast<dwarf::Attribute>(Attr++);
  CUDie.addAttribute(Attr_DW_FORM_udata, DW_FORM_udata, UData[0]);

  //----------------------------------------------------------------------
  // Test DWARF32/DWARF64 forms
  //----------------------------------------------------------------------
  const auto Attr_DW_FORM_GNU_ref_alt = static_cast<dwarf::Attribute>(Attr++);
  CUDie.addAttribute(Attr_DW_FORM_GNU_ref_alt, DW_FORM_GNU_ref_alt,
                     Dwarf32Values[0]);

  const auto Attr_DW_FORM_sec_offset = static_cast<dwarf::Attribute>(Attr++);
  if (Version >= 4)
    CUDie.addAttribute(Attr_DW_FORM_sec_offset, DW_FORM_sec_offset,
                       Dwarf32Values[1]);

  //----------------------------------------------------------------------
  // Add an address at the end to make sure we can decode this value
  //----------------------------------------------------------------------
  const auto Attr_Last = static_cast<dwarf::Attribute>(Attr++);
  CUDie.addAttribute(Attr_Last, DW_FORM_addr, AddrValue);

  //----------------------------------------------------------------------
  // Generate the DWARF
  //----------------------------------------------------------------------
  StringRef FileBytes = DG->generate();
  MemoryBufferRef FileBuffer(FileBytes, "dwarf");
  auto Obj = object::ObjectFile::createObjectFile(FileBuffer);
  EXPECT_TRUE((bool)Obj);
  std::unique_ptr<DWARFContext> DwarfContext = DWARFContext::create(**Obj);
  uint32_t NumCUs = DwarfContext->getNumCompileUnits();
  EXPECT_EQ(NumCUs, 1u);
  DWARFCompileUnit *U =
      cast<DWARFCompileUnit>(DwarfContext->getUnitAtIndex(0));
  auto DieDG = U->getUnitDIE(false);
  EXPECT_TRUE(DieDG.isValid());

  //----------------------------------------------------------------------
  // Test address forms
  //----------------------------------------------------------------------
  EXPECT_EQ(AddrValue, toAddress(DieDG.find(Attr_DW_FORM_addr), 0));

  //----------------------------------------------------------------------
  // Test block forms
  //----------------------------------------------------------------------
  Optional<DWARFFormValue> FormValue;
  ArrayRef<uint8_t> ExtractedBlockData;
  Optional<ArrayRef<uint8_t>> BlockDataOpt;

  FormValue = DieDG.find(Attr_DW_FORM_block);
  EXPECT_TRUE((bool)FormValue);
  BlockDataOpt = FormValue->getAsBlock();
  EXPECT_TRUE(BlockDataOpt.hasValue());
  ExtractedBlockData = BlockDataOpt.getValue();
  EXPECT_EQ(ExtractedBlockData.size(), BlockSize);
  EXPECT_TRUE(memcmp(ExtractedBlockData.data(), BlockData, BlockSize) == 0);

  FormValue = DieDG.find(Attr_DW_FORM_block1);
  EXPECT_TRUE((bool)FormValue);
  BlockDataOpt = FormValue->getAsBlock();
  EXPECT_TRUE(BlockDataOpt.hasValue());
  ExtractedBlockData = BlockDataOpt.getValue();
  EXPECT_EQ(ExtractedBlockData.size(), BlockSize);
  EXPECT_TRUE(memcmp(ExtractedBlockData.data(), BlockData, BlockSize) == 0);

  FormValue = DieDG.find(Attr_DW_FORM_block2);
  EXPECT_TRUE((bool)FormValue);
  BlockDataOpt = FormValue->getAsBlock();
  EXPECT_TRUE(BlockDataOpt.hasValue());
  ExtractedBlockData = BlockDataOpt.getValue();
  EXPECT_EQ(ExtractedBlockData.size(), BlockSize);
  EXPECT_TRUE(memcmp(ExtractedBlockData.data(), BlockData, BlockSize) == 0);

  FormValue = DieDG.find(Attr_DW_FORM_block4);
  EXPECT_TRUE((bool)FormValue);
  BlockDataOpt = FormValue->getAsBlock();
  EXPECT_TRUE(BlockDataOpt.hasValue());
  ExtractedBlockData = BlockDataOpt.getValue();
  EXPECT_EQ(ExtractedBlockData.size(), BlockSize);
  EXPECT_TRUE(memcmp(ExtractedBlockData.data(), BlockData, BlockSize) == 0);

  // Data16 is handled like a block.
  if (Version >= 5) {
    FormValue = DieDG.find(Attr_DW_FORM_data16);
    EXPECT_TRUE((bool)FormValue);
    BlockDataOpt = FormValue->getAsBlock();
    EXPECT_TRUE(BlockDataOpt.hasValue());
    ExtractedBlockData = BlockDataOpt.getValue();
    EXPECT_EQ(ExtractedBlockData.size(), 16u);
    EXPECT_TRUE(memcmp(ExtractedBlockData.data(), Data16, 16) == 0);
  }

  //----------------------------------------------------------------------
  // Test data forms
  //----------------------------------------------------------------------
  EXPECT_EQ(Data1, toUnsigned(DieDG.find(Attr_DW_FORM_data1), 0));
  EXPECT_EQ(Data2, toUnsigned(DieDG.find(Attr_DW_FORM_data2), 0));
  EXPECT_EQ(Data4, toUnsigned(DieDG.find(Attr_DW_FORM_data4), 0));
  EXPECT_EQ(Data8, toUnsigned(DieDG.find(Attr_DW_FORM_data8), 0));

  //----------------------------------------------------------------------
  // Test string forms
  //----------------------------------------------------------------------
  auto ExtractedStringValue = toString(DieDG.find(Attr_DW_FORM_string));
  EXPECT_TRUE((bool)ExtractedStringValue);
  EXPECT_STREQ(StringValue, *ExtractedStringValue);

  if (Version >= 5) {
    auto ExtractedStrxValue = toString(DieDG.find(Attr_DW_FORM_strx));
    EXPECT_TRUE((bool)ExtractedStrxValue);
    EXPECT_STREQ(StrxValue, *ExtractedStrxValue);

    auto ExtractedStrx1Value = toString(DieDG.find(Attr_DW_FORM_strx1));
    EXPECT_TRUE((bool)ExtractedStrx1Value);
    EXPECT_STREQ(Strx1Value, *ExtractedStrx1Value);

    auto ExtractedStrx2Value = toString(DieDG.find(Attr_DW_FORM_strx2));
    EXPECT_TRUE((bool)ExtractedStrx2Value);
    EXPECT_STREQ(Strx2Value, *ExtractedStrx2Value);

    auto ExtractedStrx3Value = toString(DieDG.find(Attr_DW_FORM_strx3));
    EXPECT_TRUE((bool)ExtractedStrx3Value);
    EXPECT_STREQ(Strx3Value, *ExtractedStrx3Value);

    auto ExtractedStrx4Value = toString(DieDG.find(Attr_DW_FORM_strx4));
    EXPECT_TRUE((bool)ExtractedStrx4Value);
    EXPECT_STREQ(Strx4Value, *ExtractedStrx4Value);
  }

  auto ExtractedStrpValue = toString(DieDG.find(Attr_DW_FORM_strp));
  EXPECT_TRUE((bool)ExtractedStrpValue);
  EXPECT_STREQ(StrpValue, *ExtractedStrpValue);

  //----------------------------------------------------------------------
  // Test reference forms
  //----------------------------------------------------------------------
  EXPECT_EQ(RefAddr, toReference(DieDG.find(Attr_DW_FORM_ref_addr), 0));
  EXPECT_EQ(Data1, toReference(DieDG.find(Attr_DW_FORM_ref1), 0));
  EXPECT_EQ(Data2, toReference(DieDG.find(Attr_DW_FORM_ref2), 0));
  EXPECT_EQ(Data4, toReference(DieDG.find(Attr_DW_FORM_ref4), 0));
  EXPECT_EQ(Data8, toReference(DieDG.find(Attr_DW_FORM_ref8), 0));
  if (Version >= 4) {
    EXPECT_EQ(Data8_2, toReference(DieDG.find(Attr_DW_FORM_ref_sig8), 0));
  }
  EXPECT_EQ(UData[0], toReference(DieDG.find(Attr_DW_FORM_ref_udata), 0));

  //----------------------------------------------------------------------
  // Test flag forms
  //----------------------------------------------------------------------
  EXPECT_EQ(1ULL, toUnsigned(DieDG.find(Attr_DW_FORM_flag_true), 0));
  EXPECT_EQ(0ULL, toUnsigned(DieDG.find(Attr_DW_FORM_flag_false), 1));
  if (Version >= 4) {
    EXPECT_EQ(1ULL, toUnsigned(DieDG.find(Attr_DW_FORM_flag_present), 0));
  }

  //----------------------------------------------------------------------
  // Test SLEB128 based forms
  //----------------------------------------------------------------------
  EXPECT_EQ(SData, toSigned(DieDG.find(Attr_DW_FORM_sdata), 0));
  if (Version >= 5) {
    EXPECT_EQ(ICSData, toSigned(DieDG.find(Attr_DW_FORM_implicit_const), 0));
  }

  //----------------------------------------------------------------------
  // Test ULEB128 based forms
  //----------------------------------------------------------------------
  EXPECT_EQ(UData[0], toUnsigned(DieDG.find(Attr_DW_FORM_udata), 0));

  //----------------------------------------------------------------------
  // Test DWARF32/DWARF64 forms
  //----------------------------------------------------------------------
  EXPECT_EQ(Dwarf32Values[0],
            toReference(DieDG.find(Attr_DW_FORM_GNU_ref_alt), 0));
  if (Version >= 4) {
    EXPECT_EQ(Dwarf32Values[1],
              toSectionOffset(DieDG.find(Attr_DW_FORM_sec_offset), 0));
  }

  //----------------------------------------------------------------------
  // Add an address at the end to make sure we can decode this value
  //----------------------------------------------------------------------
  EXPECT_EQ(AddrValue, toAddress(DieDG.find(Attr_Last), 0));
}

TEST(DWARFDebugInfo, TestDWARF32Version2Addr4AllForms) {
  // Test that we can decode all forms for DWARF32, version 2, with 4 byte
  // addresses.
  typedef uint32_t AddrType;
  // DW_FORM_ref_addr are the same as the address type in DWARF32 version 2.
  typedef AddrType RefAddrType;
  TestAllForms<2, AddrType, RefAddrType>();
}

TEST(DWARFDebugInfo, TestDWARF32Version2Addr8AllForms) {
  // Test that we can decode all forms for DWARF32, version 2, with 4 byte
  // addresses.
  typedef uint64_t AddrType;
  // DW_FORM_ref_addr are the same as the address type in DWARF32 version 2.
  typedef AddrType RefAddrType;
  TestAllForms<2, AddrType, RefAddrType>();
}

TEST(DWARFDebugInfo, TestDWARF32Version3Addr4AllForms) {
  // Test that we can decode all forms for DWARF32, version 3, with 4 byte
  // addresses.
  typedef uint32_t AddrType;
  // DW_FORM_ref_addr are 4 bytes in DWARF32 for version 3 and later.
  typedef uint32_t RefAddrType;
  TestAllForms<3, AddrType, RefAddrType>();
}

TEST(DWARFDebugInfo, TestDWARF32Version3Addr8AllForms) {
  // Test that we can decode all forms for DWARF32, version 3, with 8 byte
  // addresses.
  typedef uint64_t AddrType;
  // DW_FORM_ref_addr are 4 bytes in DWARF32 for version 3 and later
  typedef uint32_t RefAddrType;
  TestAllForms<3, AddrType, RefAddrType>();
}

TEST(DWARFDebugInfo, TestDWARF32Version4Addr4AllForms) {
  // Test that we can decode all forms for DWARF32, version 4, with 4 byte
  // addresses.
  typedef uint32_t AddrType;
  // DW_FORM_ref_addr are 4 bytes in DWARF32 for version 3 and later
  typedef uint32_t RefAddrType;
  TestAllForms<4, AddrType, RefAddrType>();
}

TEST(DWARFDebugInfo, TestDWARF32Version4Addr8AllForms) {
  // Test that we can decode all forms for DWARF32, version 4, with 8 byte
  // addresses.
  typedef uint64_t AddrType;
  // DW_FORM_ref_addr are 4 bytes in DWARF32 for version 3 and later
  typedef uint32_t RefAddrType;
  TestAllForms<4, AddrType, RefAddrType>();
}

TEST(DWARFDebugInfo, TestDWARF32Version5Addr4AllForms) {
  // Test that we can decode all forms for DWARF32, version 5, with 4 byte
  // addresses.
  typedef uint32_t AddrType;
  // DW_FORM_ref_addr are 4 bytes in DWARF32 for version 3 and later
  typedef uint32_t RefAddrType;
  TestAllForms<5, AddrType, RefAddrType>();
}

TEST(DWARFDebugInfo, TestDWARF32Version5Addr8AllForms) {
  // Test that we can decode all forms for DWARF32, version 5, with 8 byte
  // addresses.
  typedef uint64_t AddrType;
  // DW_FORM_ref_addr are 4 bytes in DWARF32 for version 3 and later
  typedef uint32_t RefAddrType;
  TestAllForms<5, AddrType, RefAddrType>();
}

template <uint16_t Version, class AddrType> void TestChildren() {
  Triple Triple = getDefaultTargetTripleForAddrSize(sizeof(AddrType));
  if (!isConfigurationSupported(Triple))
    return;

  // Test that we can decode DW_FORM_ref_addr values correctly in DWARF 2 with
  // 4 byte addresses. DW_FORM_ref_addr values should be 4 bytes when using
  // 8 byte addresses.

  auto ExpectedDG = dwarfgen::Generator::create(Triple, Version);
  ASSERT_THAT_EXPECTED(ExpectedDG, Succeeded());
  dwarfgen::Generator *DG = ExpectedDG.get().get();
  dwarfgen::CompileUnit &CU = DG->addCompileUnit();
  dwarfgen::DIE CUDie = CU.getUnitDIE();

  CUDie.addAttribute(DW_AT_name, DW_FORM_strp, "/tmp/main.c");
  CUDie.addAttribute(DW_AT_language, DW_FORM_data2, DW_LANG_C);

  dwarfgen::DIE SubprogramDie = CUDie.addChild(DW_TAG_subprogram);
  SubprogramDie.addAttribute(DW_AT_name, DW_FORM_strp, "main");
  SubprogramDie.addAttribute(DW_AT_low_pc, DW_FORM_addr, 0x1000U);
  SubprogramDie.addAttribute(DW_AT_high_pc, DW_FORM_addr, 0x2000U);

  dwarfgen::DIE IntDie = CUDie.addChild(DW_TAG_base_type);
  IntDie.addAttribute(DW_AT_name, DW_FORM_strp, "int");
  IntDie.addAttribute(DW_AT_encoding, DW_FORM_data1, DW_ATE_signed);
  IntDie.addAttribute(DW_AT_byte_size, DW_FORM_data1, 4);

  dwarfgen::DIE ArgcDie = SubprogramDie.addChild(DW_TAG_formal_parameter);
  ArgcDie.addAttribute(DW_AT_name, DW_FORM_strp, "argc");
  // ArgcDie.addAttribute(DW_AT_type, DW_FORM_ref4, IntDie);
  ArgcDie.addAttribute(DW_AT_type, DW_FORM_ref_addr, IntDie);

  StringRef FileBytes = DG->generate();
  MemoryBufferRef FileBuffer(FileBytes, "dwarf");
  auto Obj = object::ObjectFile::createObjectFile(FileBuffer);
  EXPECT_TRUE((bool)Obj);
  std::unique_ptr<DWARFContext> DwarfContext = DWARFContext::create(**Obj);

  // Verify the number of compile units is correct.
  uint32_t NumCUs = DwarfContext->getNumCompileUnits();
  EXPECT_EQ(NumCUs, 1u);
  DWARFCompileUnit *U =
      cast<DWARFCompileUnit>(DwarfContext->getUnitAtIndex(0));

  // Get the compile unit DIE is valid.
  auto DieDG = U->getUnitDIE(false);
  EXPECT_TRUE(DieDG.isValid());

  // Verify the first child of the compile unit DIE is our subprogram.
  auto SubprogramDieDG = DieDG.getFirstChild();
  EXPECT_TRUE(SubprogramDieDG.isValid());
  EXPECT_EQ(SubprogramDieDG.getTag(), DW_TAG_subprogram);

  // Verify the first child of the subprogram is our formal parameter.
  auto ArgcDieDG = SubprogramDieDG.getFirstChild();
  EXPECT_TRUE(ArgcDieDG.isValid());
  EXPECT_EQ(ArgcDieDG.getTag(), DW_TAG_formal_parameter);

  // Verify our formal parameter has a NULL tag sibling.
  auto NullDieDG = ArgcDieDG.getSibling();
  EXPECT_TRUE(NullDieDG.isValid());
  if (NullDieDG) {
    EXPECT_EQ(NullDieDG.getTag(), DW_TAG_null);
    EXPECT_TRUE(!NullDieDG.getSibling().isValid());
    EXPECT_TRUE(!NullDieDG.getFirstChild().isValid());
  }

  // Verify the sibling of our subprogram is our integer base type.
  auto IntDieDG = SubprogramDieDG.getSibling();
  EXPECT_TRUE(IntDieDG.isValid());
  EXPECT_EQ(IntDieDG.getTag(), DW_TAG_base_type);

  // Verify the sibling of our subprogram is our integer base is a NULL tag.
  NullDieDG = IntDieDG.getSibling();
  EXPECT_TRUE(NullDieDG.isValid());
  if (NullDieDG) {
    EXPECT_EQ(NullDieDG.getTag(), DW_TAG_null);
    EXPECT_TRUE(!NullDieDG.getSibling().isValid());
    EXPECT_TRUE(!NullDieDG.getFirstChild().isValid());
  }

  // Verify the previous sibling of our subprogram is our integer base type.
  IntDieDG = NullDieDG.getPreviousSibling();
  EXPECT_TRUE(IntDieDG.isValid());
  EXPECT_EQ(IntDieDG.getTag(), DW_TAG_base_type);
}

TEST(DWARFDebugInfo, TestDWARF32Version2Addr4Children) {
  // Test that we can decode all forms for DWARF32, version 2, with 4 byte
  // addresses.
  typedef uint32_t AddrType;
  TestChildren<2, AddrType>();
}

TEST(DWARFDebugInfo, TestDWARF32Version2Addr8Children) {
  // Test that we can decode all forms for DWARF32, version 2, with 8 byte
  // addresses.
  typedef uint64_t AddrType;
  TestChildren<2, AddrType>();
}

TEST(DWARFDebugInfo, TestDWARF32Version3Addr4Children) {
  // Test that we can decode all forms for DWARF32, version 3, with 4 byte
  // addresses.
  typedef uint32_t AddrType;
  TestChildren<3, AddrType>();
}

TEST(DWARFDebugInfo, TestDWARF32Version3Addr8Children) {
  // Test that we can decode all forms for DWARF32, version 3, with 8 byte
  // addresses.
  typedef uint64_t AddrType;
  TestChildren<3, AddrType>();
}

TEST(DWARFDebugInfo, TestDWARF32Version4Addr4Children) {
  // Test that we can decode all forms for DWARF32, version 4, with 4 byte
  // addresses.
  typedef uint32_t AddrType;
  TestChildren<4, AddrType>();
}

TEST(DWARFDebugInfo, TestDWARF32Version4Addr8Children) {
  // Test that we can decode all forms for DWARF32, version 4, with 8 byte
  // addresses.
  typedef uint64_t AddrType;
  TestChildren<4, AddrType>();
}

template <uint16_t Version, class AddrType> void TestReferences() {
  Triple Triple = getDefaultTargetTripleForAddrSize(sizeof(AddrType));
  if (!isConfigurationSupported(Triple))
    return;

  // Test that we can decode DW_FORM_refXXX values correctly in DWARF.
  auto ExpectedDG = dwarfgen::Generator::create(Triple, Version);
  ASSERT_THAT_EXPECTED(ExpectedDG, Succeeded());
  dwarfgen::Generator *DG = ExpectedDG.get().get();
  dwarfgen::CompileUnit &CU1 = DG->addCompileUnit();
  dwarfgen::CompileUnit &CU2 = DG->addCompileUnit();

  dwarfgen::DIE CU1Die = CU1.getUnitDIE();
  CU1Die.addAttribute(DW_AT_name, DW_FORM_strp, "/tmp/main.c");
  CU1Die.addAttribute(DW_AT_language, DW_FORM_data2, DW_LANG_C);

  dwarfgen::DIE CU1TypeDie = CU1Die.addChild(DW_TAG_base_type);
  CU1TypeDie.addAttribute(DW_AT_name, DW_FORM_strp, "int");
  CU1TypeDie.addAttribute(DW_AT_encoding, DW_FORM_data1, DW_ATE_signed);
  CU1TypeDie.addAttribute(DW_AT_byte_size, DW_FORM_data1, 4);

  dwarfgen::DIE CU1Ref1Die = CU1Die.addChild(DW_TAG_variable);
  CU1Ref1Die.addAttribute(DW_AT_name, DW_FORM_strp, "CU1Ref1");
  CU1Ref1Die.addAttribute(DW_AT_type, DW_FORM_ref1, CU1TypeDie);

  dwarfgen::DIE CU1Ref2Die = CU1Die.addChild(DW_TAG_variable);
  CU1Ref2Die.addAttribute(DW_AT_name, DW_FORM_strp, "CU1Ref2");
  CU1Ref2Die.addAttribute(DW_AT_type, DW_FORM_ref2, CU1TypeDie);

  dwarfgen::DIE CU1Ref4Die = CU1Die.addChild(DW_TAG_variable);
  CU1Ref4Die.addAttribute(DW_AT_name, DW_FORM_strp, "CU1Ref4");
  CU1Ref4Die.addAttribute(DW_AT_type, DW_FORM_ref4, CU1TypeDie);

  dwarfgen::DIE CU1Ref8Die = CU1Die.addChild(DW_TAG_variable);
  CU1Ref8Die.addAttribute(DW_AT_name, DW_FORM_strp, "CU1Ref8");
  CU1Ref8Die.addAttribute(DW_AT_type, DW_FORM_ref8, CU1TypeDie);

  dwarfgen::DIE CU1RefAddrDie = CU1Die.addChild(DW_TAG_variable);
  CU1RefAddrDie.addAttribute(DW_AT_name, DW_FORM_strp, "CU1RefAddr");
  CU1RefAddrDie.addAttribute(DW_AT_type, DW_FORM_ref_addr, CU1TypeDie);

  dwarfgen::DIE CU2Die = CU2.getUnitDIE();
  CU2Die.addAttribute(DW_AT_name, DW_FORM_strp, "/tmp/foo.c");
  CU2Die.addAttribute(DW_AT_language, DW_FORM_data2, DW_LANG_C);

  dwarfgen::DIE CU2TypeDie = CU2Die.addChild(DW_TAG_base_type);
  CU2TypeDie.addAttribute(DW_AT_name, DW_FORM_strp, "float");
  CU2TypeDie.addAttribute(DW_AT_encoding, DW_FORM_data1, DW_ATE_float);
  CU2TypeDie.addAttribute(DW_AT_byte_size, DW_FORM_data1, 4);

  dwarfgen::DIE CU2Ref1Die = CU2Die.addChild(DW_TAG_variable);
  CU2Ref1Die.addAttribute(DW_AT_name, DW_FORM_strp, "CU2Ref1");
  CU2Ref1Die.addAttribute(DW_AT_type, DW_FORM_ref1, CU2TypeDie);

  dwarfgen::DIE CU2Ref2Die = CU2Die.addChild(DW_TAG_variable);
  CU2Ref2Die.addAttribute(DW_AT_name, DW_FORM_strp, "CU2Ref2");
  CU2Ref2Die.addAttribute(DW_AT_type, DW_FORM_ref2, CU2TypeDie);

  dwarfgen::DIE CU2Ref4Die = CU2Die.addChild(DW_TAG_variable);
  CU2Ref4Die.addAttribute(DW_AT_name, DW_FORM_strp, "CU2Ref4");
  CU2Ref4Die.addAttribute(DW_AT_type, DW_FORM_ref4, CU2TypeDie);

  dwarfgen::DIE CU2Ref8Die = CU2Die.addChild(DW_TAG_variable);
  CU2Ref8Die.addAttribute(DW_AT_name, DW_FORM_strp, "CU2Ref8");
  CU2Ref8Die.addAttribute(DW_AT_type, DW_FORM_ref8, CU2TypeDie);

  dwarfgen::DIE CU2RefAddrDie = CU2Die.addChild(DW_TAG_variable);
  CU2RefAddrDie.addAttribute(DW_AT_name, DW_FORM_strp, "CU2RefAddr");
  CU2RefAddrDie.addAttribute(DW_AT_type, DW_FORM_ref_addr, CU2TypeDie);

  // Refer to a type in CU1 from CU2
  dwarfgen::DIE CU2ToCU1RefAddrDie = CU2Die.addChild(DW_TAG_variable);
  CU2ToCU1RefAddrDie.addAttribute(DW_AT_name, DW_FORM_strp, "CU2ToCU1RefAddr");
  CU2ToCU1RefAddrDie.addAttribute(DW_AT_type, DW_FORM_ref_addr, CU1TypeDie);

  // Refer to a type in CU2 from CU1
  dwarfgen::DIE CU1ToCU2RefAddrDie = CU1Die.addChild(DW_TAG_variable);
  CU1ToCU2RefAddrDie.addAttribute(DW_AT_name, DW_FORM_strp, "CU1ToCU2RefAddr");
  CU1ToCU2RefAddrDie.addAttribute(DW_AT_type, DW_FORM_ref_addr, CU2TypeDie);

  StringRef FileBytes = DG->generate();
  MemoryBufferRef FileBuffer(FileBytes, "dwarf");
  auto Obj = object::ObjectFile::createObjectFile(FileBuffer);
  EXPECT_TRUE((bool)Obj);
  std::unique_ptr<DWARFContext> DwarfContext = DWARFContext::create(**Obj);

  // Verify the number of compile units is correct.
  uint32_t NumCUs = DwarfContext->getNumCompileUnits();
  EXPECT_EQ(NumCUs, 2u);
  DWARFCompileUnit *U1 =
      cast<DWARFCompileUnit>(DwarfContext->getUnitAtIndex(0));
  DWARFCompileUnit *U2 =
      cast<DWARFCompileUnit>(DwarfContext->getUnitAtIndex(1));

  // Get the compile unit DIE is valid.
  auto Unit1DieDG = U1->getUnitDIE(false);
  EXPECT_TRUE(Unit1DieDG.isValid());

  auto Unit2DieDG = U2->getUnitDIE(false);
  EXPECT_TRUE(Unit2DieDG.isValid());

  // Verify the first child of the compile unit 1 DIE is our int base type.
  auto CU1TypeDieDG = Unit1DieDG.getFirstChild();
  EXPECT_TRUE(CU1TypeDieDG.isValid());
  EXPECT_EQ(CU1TypeDieDG.getTag(), DW_TAG_base_type);
  EXPECT_EQ(DW_ATE_signed, toUnsigned(CU1TypeDieDG.find(DW_AT_encoding), 0));

  // Verify the first child of the compile unit 2 DIE is our float base type.
  auto CU2TypeDieDG = Unit2DieDG.getFirstChild();
  EXPECT_TRUE(CU2TypeDieDG.isValid());
  EXPECT_EQ(CU2TypeDieDG.getTag(), DW_TAG_base_type);
  EXPECT_EQ(DW_ATE_float, toUnsigned(CU2TypeDieDG.find(DW_AT_encoding), 0));

  // Verify the sibling of the base type DIE is our Ref1 DIE and that its
  // DW_AT_type points to our base type DIE.
  auto CU1Ref1DieDG = CU1TypeDieDG.getSibling();
  EXPECT_TRUE(CU1Ref1DieDG.isValid());
  EXPECT_EQ(CU1Ref1DieDG.getTag(), DW_TAG_variable);
  EXPECT_EQ(CU1TypeDieDG.getOffset(),
            toReference(CU1Ref1DieDG.find(DW_AT_type), -1ULL));
  // Verify the sibling is our Ref2 DIE and that its DW_AT_type points to our
  // base type DIE in CU1.
  auto CU1Ref2DieDG = CU1Ref1DieDG.getSibling();
  EXPECT_TRUE(CU1Ref2DieDG.isValid());
  EXPECT_EQ(CU1Ref2DieDG.getTag(), DW_TAG_variable);
  EXPECT_EQ(CU1TypeDieDG.getOffset(),
            toReference(CU1Ref2DieDG.find(DW_AT_type), -1ULL));

  // Verify the sibling is our Ref4 DIE and that its DW_AT_type points to our
  // base type DIE in CU1.
  auto CU1Ref4DieDG = CU1Ref2DieDG.getSibling();
  EXPECT_TRUE(CU1Ref4DieDG.isValid());
  EXPECT_EQ(CU1Ref4DieDG.getTag(), DW_TAG_variable);
  EXPECT_EQ(CU1TypeDieDG.getOffset(),
            toReference(CU1Ref4DieDG.find(DW_AT_type), -1ULL));

  // Verify the sibling is our Ref8 DIE and that its DW_AT_type points to our
  // base type DIE in CU1.
  auto CU1Ref8DieDG = CU1Ref4DieDG.getSibling();
  EXPECT_TRUE(CU1Ref8DieDG.isValid());
  EXPECT_EQ(CU1Ref8DieDG.getTag(), DW_TAG_variable);
  EXPECT_EQ(CU1TypeDieDG.getOffset(),
            toReference(CU1Ref8DieDG.find(DW_AT_type), -1ULL));

  // Verify the sibling is our RefAddr DIE and that its DW_AT_type points to our
  // base type DIE in CU1.
  auto CU1RefAddrDieDG = CU1Ref8DieDG.getSibling();
  EXPECT_TRUE(CU1RefAddrDieDG.isValid());
  EXPECT_EQ(CU1RefAddrDieDG.getTag(), DW_TAG_variable);
  EXPECT_EQ(CU1TypeDieDG.getOffset(),
            toReference(CU1RefAddrDieDG.find(DW_AT_type), -1ULL));

  // Verify the sibling of the Ref4 DIE is our RefAddr DIE and that its
  // DW_AT_type points to our base type DIE.
  auto CU1ToCU2RefAddrDieDG = CU1RefAddrDieDG.getSibling();
  EXPECT_TRUE(CU1ToCU2RefAddrDieDG.isValid());
  EXPECT_EQ(CU1ToCU2RefAddrDieDG.getTag(), DW_TAG_variable);
  EXPECT_EQ(CU2TypeDieDG.getOffset(),
            toReference(CU1ToCU2RefAddrDieDG.find(DW_AT_type), -1ULL));

  // Verify the sibling of the base type DIE is our Ref1 DIE and that its
  // DW_AT_type points to our base type DIE.
  auto CU2Ref1DieDG = CU2TypeDieDG.getSibling();
  EXPECT_TRUE(CU2Ref1DieDG.isValid());
  EXPECT_EQ(CU2Ref1DieDG.getTag(), DW_TAG_variable);
  EXPECT_EQ(CU2TypeDieDG.getOffset(),
            toReference(CU2Ref1DieDG.find(DW_AT_type), -1ULL));
  // Verify the sibling is our Ref2 DIE and that its DW_AT_type points to our
  // base type DIE in CU2.
  auto CU2Ref2DieDG = CU2Ref1DieDG.getSibling();
  EXPECT_TRUE(CU2Ref2DieDG.isValid());
  EXPECT_EQ(CU2Ref2DieDG.getTag(), DW_TAG_variable);
  EXPECT_EQ(CU2TypeDieDG.getOffset(),
            toReference(CU2Ref2DieDG.find(DW_AT_type), -1ULL));

  // Verify the sibling is our Ref4 DIE and that its DW_AT_type points to our
  // base type DIE in CU2.
  auto CU2Ref4DieDG = CU2Ref2DieDG.getSibling();
  EXPECT_TRUE(CU2Ref4DieDG.isValid());
  EXPECT_EQ(CU2Ref4DieDG.getTag(), DW_TAG_variable);
  EXPECT_EQ(CU2TypeDieDG.getOffset(),
            toReference(CU2Ref4DieDG.find(DW_AT_type), -1ULL));

  // Verify the sibling is our Ref8 DIE and that its DW_AT_type points to our
  // base type DIE in CU2.
  auto CU2Ref8DieDG = CU2Ref4DieDG.getSibling();
  EXPECT_TRUE(CU2Ref8DieDG.isValid());
  EXPECT_EQ(CU2Ref8DieDG.getTag(), DW_TAG_variable);
  EXPECT_EQ(CU2TypeDieDG.getOffset(),
            toReference(CU2Ref8DieDG.find(DW_AT_type), -1ULL));

  // Verify the sibling is our RefAddr DIE and that its DW_AT_type points to our
  // base type DIE in CU2.
  auto CU2RefAddrDieDG = CU2Ref8DieDG.getSibling();
  EXPECT_TRUE(CU2RefAddrDieDG.isValid());
  EXPECT_EQ(CU2RefAddrDieDG.getTag(), DW_TAG_variable);
  EXPECT_EQ(CU2TypeDieDG.getOffset(),
            toReference(CU2RefAddrDieDG.find(DW_AT_type), -1ULL));

  // Verify the sibling of the Ref4 DIE is our RefAddr DIE and that its
  // DW_AT_type points to our base type DIE.
  auto CU2ToCU1RefAddrDieDG = CU2RefAddrDieDG.getSibling();
  EXPECT_TRUE(CU2ToCU1RefAddrDieDG.isValid());
  EXPECT_EQ(CU2ToCU1RefAddrDieDG.getTag(), DW_TAG_variable);
  EXPECT_EQ(CU1TypeDieDG.getOffset(),
            toReference(CU2ToCU1RefAddrDieDG.find(DW_AT_type), -1ULL));
}

TEST(DWARFDebugInfo, TestDWARF32Version2Addr4References) {
  // Test that we can decode all forms for DWARF32, version 2, with 4 byte
  // addresses.
  typedef uint32_t AddrType;
  TestReferences<2, AddrType>();
}

TEST(DWARFDebugInfo, TestDWARF32Version2Addr8References) {
  // Test that we can decode all forms for DWARF32, version 2, with 8 byte
  // addresses.
  typedef uint64_t AddrType;
  TestReferences<2, AddrType>();
}

TEST(DWARFDebugInfo, TestDWARF32Version3Addr4References) {
  // Test that we can decode all forms for DWARF32, version 3, with 4 byte
  // addresses.
  typedef uint32_t AddrType;
  TestReferences<3, AddrType>();
}

TEST(DWARFDebugInfo, TestDWARF32Version3Addr8References) {
  // Test that we can decode all forms for DWARF32, version 3, with 8 byte
  // addresses.
  typedef uint64_t AddrType;
  TestReferences<3, AddrType>();
}

TEST(DWARFDebugInfo, TestDWARF32Version4Addr4References) {
  // Test that we can decode all forms for DWARF32, version 4, with 4 byte
  // addresses.
  typedef uint32_t AddrType;
  TestReferences<4, AddrType>();
}

TEST(DWARFDebugInfo, TestDWARF32Version4Addr8References) {
  // Test that we can decode all forms for DWARF32, version 4, with 8 byte
  // addresses.
  typedef uint64_t AddrType;
  TestReferences<4, AddrType>();
}

template <uint16_t Version, class AddrType> void TestAddresses() {
  Triple Triple = getDefaultTargetTripleForAddrSize(sizeof(AddrType));
  if (!isConfigurationSupported(Triple))
    return;

  // Test the DWARF APIs related to accessing the DW_AT_low_pc and
  // DW_AT_high_pc.
  const bool SupportsHighPCAsOffset = Version >= 4;
  auto ExpectedDG = dwarfgen::Generator::create(Triple, Version);
  ASSERT_THAT_EXPECTED(ExpectedDG, Succeeded());
  dwarfgen::Generator *DG = ExpectedDG.get().get();
  dwarfgen::CompileUnit &CU = DG->addCompileUnit();
  dwarfgen::DIE CUDie = CU.getUnitDIE();

  CUDie.addAttribute(DW_AT_name, DW_FORM_strp, "/tmp/main.c");
  CUDie.addAttribute(DW_AT_language, DW_FORM_data2, DW_LANG_C);

  // Create a subprogram DIE with no low or high PC.
  dwarfgen::DIE SubprogramNoPC = CUDie.addChild(DW_TAG_subprogram);
  SubprogramNoPC.addAttribute(DW_AT_name, DW_FORM_strp, "no_pc");

  // Create a subprogram DIE with a low PC only.
  dwarfgen::DIE SubprogramLowPC = CUDie.addChild(DW_TAG_subprogram);
  SubprogramLowPC.addAttribute(DW_AT_name, DW_FORM_strp, "low_pc");
  const uint64_t ActualLowPC = 0x1000;
  const uint64_t ActualHighPC = 0x2000;
  const uint64_t ActualHighPCOffset = ActualHighPC - ActualLowPC;
  SubprogramLowPC.addAttribute(DW_AT_low_pc, DW_FORM_addr, ActualLowPC);

  // Create a subprogram DIE with a low and high PC.
  dwarfgen::DIE SubprogramLowHighPC = CUDie.addChild(DW_TAG_subprogram);
  SubprogramLowHighPC.addAttribute(DW_AT_name, DW_FORM_strp, "low_high_pc");
  SubprogramLowHighPC.addAttribute(DW_AT_low_pc, DW_FORM_addr, ActualLowPC);
  // Encode the high PC as an offset from the low PC if supported.
  if (SupportsHighPCAsOffset)
    SubprogramLowHighPC.addAttribute(DW_AT_high_pc, DW_FORM_data4,
                                     ActualHighPCOffset);
  else
    SubprogramLowHighPC.addAttribute(DW_AT_high_pc, DW_FORM_addr, ActualHighPC);

  StringRef FileBytes = DG->generate();
  MemoryBufferRef FileBuffer(FileBytes, "dwarf");
  auto Obj = object::ObjectFile::createObjectFile(FileBuffer);
  EXPECT_TRUE((bool)Obj);
  std::unique_ptr<DWARFContext> DwarfContext = DWARFContext::create(**Obj);

  // Verify the number of compile units is correct.
  uint32_t NumCUs = DwarfContext->getNumCompileUnits();
  EXPECT_EQ(NumCUs, 1u);
  DWARFCompileUnit *U =
      cast<DWARFCompileUnit>(DwarfContext->getUnitAtIndex(0));

  // Get the compile unit DIE is valid.
  auto DieDG = U->getUnitDIE(false);
  EXPECT_TRUE(DieDG.isValid());

  uint64_t LowPC, HighPC, SectionIndex;
  Optional<uint64_t> OptU64;
  // Verify the that our subprogram with no PC value fails appropriately when
  // asked for any PC values.
  auto SubprogramDieNoPC = DieDG.getFirstChild();
  EXPECT_TRUE(SubprogramDieNoPC.isValid());
  EXPECT_EQ(SubprogramDieNoPC.getTag(), DW_TAG_subprogram);
  OptU64 = toAddress(SubprogramDieNoPC.find(DW_AT_low_pc));
  EXPECT_FALSE((bool)OptU64);
  OptU64 = toAddress(SubprogramDieNoPC.find(DW_AT_high_pc));
  EXPECT_FALSE((bool)OptU64);
  EXPECT_FALSE(SubprogramDieNoPC.getLowAndHighPC(LowPC, HighPC, SectionIndex));
  OptU64 = toAddress(SubprogramDieNoPC.find(DW_AT_high_pc));
  EXPECT_FALSE((bool)OptU64);
  OptU64 = toUnsigned(SubprogramDieNoPC.find(DW_AT_high_pc));
  EXPECT_FALSE((bool)OptU64);
  OptU64 = SubprogramDieNoPC.getHighPC(ActualLowPC);
  EXPECT_FALSE((bool)OptU64);
  EXPECT_FALSE(SubprogramDieNoPC.getLowAndHighPC(LowPC, HighPC, SectionIndex));

  // Verify the that our subprogram with only a low PC value succeeds when
  // we ask for the Low PC, but fails appropriately when asked for the high PC
  // or both low and high PC values.
  auto SubprogramDieLowPC = SubprogramDieNoPC.getSibling();
  EXPECT_TRUE(SubprogramDieLowPC.isValid());
  EXPECT_EQ(SubprogramDieLowPC.getTag(), DW_TAG_subprogram);
  OptU64 = toAddress(SubprogramDieLowPC.find(DW_AT_low_pc));
  EXPECT_TRUE((bool)OptU64);
  EXPECT_EQ(OptU64.getValue(), ActualLowPC);
  OptU64 = toAddress(SubprogramDieLowPC.find(DW_AT_high_pc));
  EXPECT_FALSE((bool)OptU64);
  OptU64 = toUnsigned(SubprogramDieLowPC.find(DW_AT_high_pc));
  EXPECT_FALSE((bool)OptU64);
  OptU64 = SubprogramDieLowPC.getHighPC(ActualLowPC);
  EXPECT_FALSE((bool)OptU64);
  EXPECT_FALSE(SubprogramDieLowPC.getLowAndHighPC(LowPC, HighPC, SectionIndex));

  // Verify the that our subprogram with only a low PC value succeeds when
  // we ask for the Low PC, but fails appropriately when asked for the high PC
  // or both low and high PC values.
  auto SubprogramDieLowHighPC = SubprogramDieLowPC.getSibling();
  EXPECT_TRUE(SubprogramDieLowHighPC.isValid());
  EXPECT_EQ(SubprogramDieLowHighPC.getTag(), DW_TAG_subprogram);
  OptU64 = toAddress(SubprogramDieLowHighPC.find(DW_AT_low_pc));
  EXPECT_TRUE((bool)OptU64);
  EXPECT_EQ(OptU64.getValue(), ActualLowPC);
  // Get the high PC as an address. This should succeed if the high PC was
  // encoded as an address and fail if the high PC was encoded as an offset.
  OptU64 = toAddress(SubprogramDieLowHighPC.find(DW_AT_high_pc));
  if (SupportsHighPCAsOffset) {
    EXPECT_FALSE((bool)OptU64);
  } else {
    EXPECT_TRUE((bool)OptU64);
    EXPECT_EQ(OptU64.getValue(), ActualHighPC);
  }
  // Get the high PC as an unsigned constant. This should succeed if the high PC
  // was encoded as an offset and fail if the high PC was encoded as an address.
  OptU64 = toUnsigned(SubprogramDieLowHighPC.find(DW_AT_high_pc));
  if (SupportsHighPCAsOffset) {
    EXPECT_TRUE((bool)OptU64);
    EXPECT_EQ(OptU64.getValue(), ActualHighPCOffset);
  } else {
    EXPECT_FALSE((bool)OptU64);
  }

  OptU64 = SubprogramDieLowHighPC.getHighPC(ActualLowPC);
  EXPECT_TRUE((bool)OptU64);
  EXPECT_EQ(OptU64.getValue(), ActualHighPC);

  EXPECT_TRUE(SubprogramDieLowHighPC.getLowAndHighPC(LowPC, HighPC, SectionIndex));
  EXPECT_EQ(LowPC, ActualLowPC);
  EXPECT_EQ(HighPC, ActualHighPC);
}

TEST(DWARFDebugInfo, TestDWARF32Version2Addr4Addresses) {
  // Test that we can decode address values in DWARF32, version 2, with 4 byte
  // addresses.
  typedef uint32_t AddrType;
  TestAddresses<2, AddrType>();
}

TEST(DWARFDebugInfo, TestDWARF32Version2Addr8Addresses) {
  // Test that we can decode address values in DWARF32, version 2, with 8 byte
  // addresses.
  typedef uint64_t AddrType;
  TestAddresses<2, AddrType>();
}

TEST(DWARFDebugInfo, TestDWARF32Version3Addr4Addresses) {
  // Test that we can decode address values in DWARF32, version 3, with 4 byte
  // addresses.
  typedef uint32_t AddrType;
  TestAddresses<3, AddrType>();
}

TEST(DWARFDebugInfo, TestDWARF32Version3Addr8Addresses) {
  // Test that we can decode address values in DWARF32, version 3, with 8 byte
  // addresses.
  typedef uint64_t AddrType;
  TestAddresses<3, AddrType>();
}

TEST(DWARFDebugInfo, TestDWARF32Version4Addr4Addresses) {
  // Test that we can decode address values in DWARF32, version 4, with 4 byte
  // addresses.
  typedef uint32_t AddrType;
  TestAddresses<4, AddrType>();
}

TEST(DWARFDebugInfo, TestDWARF32Version4Addr8Addresses) {
  // Test that we can decode address values in DWARF32, version 4, with 8 byte
  // addresses.
  typedef uint64_t AddrType;
  TestAddresses<4, AddrType>();
}

TEST(DWARFDebugInfo, TestStringOffsets) {
  Triple Triple = getNormalizedDefaultTargetTriple();
  if (!isConfigurationSupported(Triple))
    return;

  const char *String1 = "Hello";
  const char *String2 = "World";

  auto ExpectedDG = dwarfgen::Generator::create(Triple, 5);
  ASSERT_THAT_EXPECTED(ExpectedDG, Succeeded());
  dwarfgen::Generator *DG = ExpectedDG.get().get();
  dwarfgen::CompileUnit &CU = DG->addCompileUnit();
  dwarfgen::DIE CUDie = CU.getUnitDIE();

  CUDie.addStrOffsetsBaseAttribute();

  uint16_t Attr = DW_AT_lo_user;

  // Create our strings. First we create a non-indexed reference to String1,
  // followed by an indexed String2. Finally, we add an indexed reference to
  // String1.
  const auto Attr1 = static_cast<dwarf::Attribute>(Attr++);
  CUDie.addAttribute(Attr1, DW_FORM_strp, String1);

  const auto Attr2 = static_cast<dwarf::Attribute>(Attr++);
  CUDie.addAttribute(Attr2, DW_FORM_strx, String2);

  const auto Attr3 = static_cast<dwarf::Attribute>(Attr++);
  CUDie.addAttribute(Attr3, DW_FORM_strx, String1);

  // Generate the DWARF
  StringRef FileBytes = DG->generate();
  MemoryBufferRef FileBuffer(FileBytes, "dwarf");
  auto Obj = object::ObjectFile::createObjectFile(FileBuffer);
  ASSERT_TRUE((bool)Obj);
  std::unique_ptr<DWARFContext> DwarfContext = DWARFContext::create(**Obj);
  uint32_t NumCUs = DwarfContext->getNumCompileUnits();
  ASSERT_EQ(NumCUs, 1u);
  DWARFUnit *U = DwarfContext->getUnitAtIndex(0);
  auto DieDG = U->getUnitDIE(false);
  ASSERT_TRUE(DieDG.isValid());

  // Now make sure the string offsets came out properly. Attr2 should have index
  // 0 (because it was the first indexed string) even though the string itself
  // was added eariler.
  auto Extracted1 = toString(DieDG.find(Attr1));
  ASSERT_TRUE((bool)Extracted1);
  EXPECT_STREQ(String1, *Extracted1);

  Optional<DWARFFormValue> Form2 = DieDG.find(Attr2);
  ASSERT_TRUE((bool)Form2);
  EXPECT_EQ(0u, Form2->getRawUValue());
  auto Extracted2 = toString(Form2);
  ASSERT_TRUE((bool)Extracted2);
  EXPECT_STREQ(String2, *Extracted2);

  Optional<DWARFFormValue> Form3 = DieDG.find(Attr3);
  ASSERT_TRUE((bool)Form3);
  EXPECT_EQ(1u, Form3->getRawUValue());
  auto Extracted3 = toString(Form3);
  ASSERT_TRUE((bool)Extracted3);
  EXPECT_STREQ(String1, *Extracted3);
}

TEST(DWARFDebugInfo, TestEmptyStringOffsets) {
  Triple Triple = getNormalizedDefaultTargetTriple();
  if (!isConfigurationSupported(Triple))
    return;

  const char *String1 = "Hello";

  auto ExpectedDG = dwarfgen::Generator::create(Triple, 5);
  ASSERT_THAT_EXPECTED(ExpectedDG, Succeeded());
  dwarfgen::Generator *DG = ExpectedDG.get().get();
  dwarfgen::CompileUnit &CU = DG->addCompileUnit();
  dwarfgen::DIE CUDie = CU.getUnitDIE();

  uint16_t Attr = DW_AT_lo_user;

  // We shall insert only one string. It will be referenced directly.
  const auto Attr1 = static_cast<dwarf::Attribute>(Attr++);
  CUDie.addAttribute(Attr1, DW_FORM_strp, String1);

  // Generate the DWARF
  StringRef FileBytes = DG->generate();
  MemoryBufferRef FileBuffer(FileBytes, "dwarf");
  auto Obj = object::ObjectFile::createObjectFile(FileBuffer);
  ASSERT_TRUE((bool)Obj);
  std::unique_ptr<DWARFContext> DwarfContext = DWARFContext::create(**Obj);
  EXPECT_TRUE(
      DwarfContext->getDWARFObj().getStrOffsetsSection().Data.empty());
}

TEST(DWARFDebugInfo, TestRelations) {
  Triple Triple = getNormalizedDefaultTargetTriple();
  if (!isConfigurationSupported(Triple))
    return;

  // Test the DWARF APIs related to accessing the DW_AT_low_pc and
  // DW_AT_high_pc.
  uint16_t Version = 4;
  auto ExpectedDG = dwarfgen::Generator::create(Triple, Version);
  ASSERT_THAT_EXPECTED(ExpectedDG, Succeeded());
  dwarfgen::Generator *DG = ExpectedDG.get().get();
  dwarfgen::CompileUnit &CU = DG->addCompileUnit();

  enum class Tag: uint16_t  {
    A = dwarf::DW_TAG_lo_user,
    B,
    C,
    C1,
    C2,
    D,
    D1
  };

  // Scope to allow us to re-use the same DIE names
  {
    // Create DWARF tree that looks like:
    //
    // CU
    //   A
    //     B
    //     C
    //       C1
    //       C2
    //     D
    //       D1
    dwarfgen::DIE CUDie = CU.getUnitDIE();
    dwarfgen::DIE A = CUDie.addChild((dwarf::Tag)Tag::A);
    A.addChild((dwarf::Tag)Tag::B);
    dwarfgen::DIE C = A.addChild((dwarf::Tag)Tag::C);
    dwarfgen::DIE D = A.addChild((dwarf::Tag)Tag::D);
    C.addChild((dwarf::Tag)Tag::C1);
    C.addChild((dwarf::Tag)Tag::C2);
    D.addChild((dwarf::Tag)Tag::D1);
  }

  MemoryBufferRef FileBuffer(DG->generate(), "dwarf");
  auto Obj = object::ObjectFile::createObjectFile(FileBuffer);
  EXPECT_TRUE((bool)Obj);
  std::unique_ptr<DWARFContext> DwarfContext = DWARFContext::create(**Obj);

  // Verify the number of compile units is correct.
  uint32_t NumCUs = DwarfContext->getNumCompileUnits();
  EXPECT_EQ(NumCUs, 1u);
  DWARFCompileUnit *U =
      cast<DWARFCompileUnit>(DwarfContext->getUnitAtIndex(0));

  // Get the compile unit DIE is valid.
  auto CUDie = U->getUnitDIE(false);
  EXPECT_TRUE(CUDie.isValid());

  // The compile unit doesn't have a parent or a sibling.
  auto ParentDie = CUDie.getParent();
  EXPECT_FALSE(ParentDie.isValid());
  auto SiblingDie = CUDie.getSibling();
  EXPECT_FALSE(SiblingDie.isValid());

  // Get the children of the compile unit
  auto A = CUDie.getFirstChild();
  auto B = A.getFirstChild();
  auto C = B.getSibling();
  auto D = C.getSibling();
  auto Null = D.getSibling();

  // Verify NULL Die is NULL and has no children or siblings
  EXPECT_TRUE(Null.isNULL());
  EXPECT_FALSE(Null.getSibling().isValid());
  EXPECT_FALSE(Null.getFirstChild().isValid());

  // Verify all children of the compile unit DIE are correct.
  EXPECT_EQ(A.getTag(), (dwarf::Tag)Tag::A);
  EXPECT_EQ(B.getTag(), (dwarf::Tag)Tag::B);
  EXPECT_EQ(C.getTag(), (dwarf::Tag)Tag::C);
  EXPECT_EQ(D.getTag(), (dwarf::Tag)Tag::D);

  // Verify who has children
  EXPECT_TRUE(A.hasChildren());
  EXPECT_FALSE(B.hasChildren());
  EXPECT_TRUE(C.hasChildren());
  EXPECT_TRUE(D.hasChildren());

  // Make sure the parent of all the children of the compile unit are the
  // compile unit.
  EXPECT_EQ(A.getParent(), CUDie);

  // Make sure the parent of all the children of A are the A.
  // B is the first child in A, so we need to verify we can get the previous
  // DIE as the parent.
  EXPECT_EQ(B.getParent(), A);
  // C is the second child in A, so we need to make sure we can backup across
  // other DIE (B) at the same level to get the correct parent.
  EXPECT_EQ(C.getParent(), A);
  // D is the third child of A. We need to verify we can backup across other DIE
  // (B and C) including DIE that have children (D) to get the correct parent.
  EXPECT_EQ(D.getParent(), A);

  // Verify that a DIE with no children returns an invalid DWARFDie.
  EXPECT_FALSE(B.getFirstChild().isValid());

  // Verify the children of the B DIE
  auto C1 = C.getFirstChild();
  auto C2 = C1.getSibling();
  EXPECT_TRUE(C2.getSibling().isNULL());

  // Verify all children of the B DIE correctly valid or invalid.
  EXPECT_EQ(C1.getTag(), (dwarf::Tag)Tag::C1);
  EXPECT_EQ(C2.getTag(), (dwarf::Tag)Tag::C2);

  // Make sure the parent of all the children of the B are the B.
  EXPECT_EQ(C1.getParent(), C);
  EXPECT_EQ(C2.getParent(), C);

  // Make sure iterators work as expected.
  EXPECT_THAT(std::vector<DWARFDie>(A.begin(), A.end()),
              testing::ElementsAre(B, C, D));
  EXPECT_THAT(std::vector<DWARFDie>(A.rbegin(), A.rend()),
              testing::ElementsAre(D, C, B));

  // Make sure conversion from reverse iterator works as expected.
  EXPECT_EQ(A.rbegin().base(), A.end());
  EXPECT_EQ(A.rend().base(), A.begin());

  // Make sure iterator is bidirectional.
  {
    auto Begin = A.begin();
    auto End = A.end();
    auto It = A.begin();

    EXPECT_EQ(It, Begin);
    EXPECT_EQ(*It, B);
    ++It;
    EXPECT_EQ(*It, C);
    ++It;
    EXPECT_EQ(*It, D);
    ++It;
    EXPECT_EQ(It, End);
    --It;
    EXPECT_EQ(*It, D);
    --It;
    EXPECT_EQ(*It, C);
    --It;
    EXPECT_EQ(*It, B);
    EXPECT_EQ(It, Begin);
  }

  // Make sure reverse iterator is bidirectional.
  {
    auto Begin = A.rbegin();
    auto End = A.rend();
    auto It = A.rbegin();

    EXPECT_EQ(It, Begin);
    EXPECT_EQ(*It, D);
    ++It;
    EXPECT_EQ(*It, C);
    ++It;
    EXPECT_EQ(*It, B);
    ++It;
    EXPECT_EQ(It, End);
    --It;
    EXPECT_EQ(*It, B);
    --It;
    EXPECT_EQ(*It, C);
    --It;
    EXPECT_EQ(*It, D);
    EXPECT_EQ(It, Begin);
  }
}

TEST(DWARFDebugInfo, TestDWARFDie) {
  // Make sure a default constructed DWARFDie doesn't have any parent, sibling
  // or child;
  DWARFDie DefaultDie;
  EXPECT_FALSE(DefaultDie.getParent().isValid());
  EXPECT_FALSE(DefaultDie.getFirstChild().isValid());
  EXPECT_FALSE(DefaultDie.getSibling().isValid());
}

TEST(DWARFDebugInfo, TestChildIterators) {
  Triple Triple = getNormalizedDefaultTargetTriple();
  if (!isConfigurationSupported(Triple))
    return;

  // Test the DWARF APIs related to iterating across the children of a DIE using
  // the DWARFDie::iterator class.
  uint16_t Version = 4;
  auto ExpectedDG = dwarfgen::Generator::create(Triple, Version);
  ASSERT_THAT_EXPECTED(ExpectedDG, Succeeded());
  dwarfgen::Generator *DG = ExpectedDG.get().get();
  dwarfgen::CompileUnit &CU = DG->addCompileUnit();

  enum class Tag: uint16_t  {
    A = dwarf::DW_TAG_lo_user,
    B,
  };

  // Scope to allow us to re-use the same DIE names
  {
    // Create DWARF tree that looks like:
    //
    // CU
    //   A
    //   B
    auto CUDie = CU.getUnitDIE();
    CUDie.addChild((dwarf::Tag)Tag::A);
    CUDie.addChild((dwarf::Tag)Tag::B);
  }

  MemoryBufferRef FileBuffer(DG->generate(), "dwarf");
  auto Obj = object::ObjectFile::createObjectFile(FileBuffer);
  EXPECT_TRUE((bool)Obj);
  std::unique_ptr<DWARFContext> DwarfContext = DWARFContext::create(**Obj);

  // Verify the number of compile units is correct.
  uint32_t NumCUs = DwarfContext->getNumCompileUnits();
  EXPECT_EQ(NumCUs, 1u);
  DWARFCompileUnit *U =
      cast<DWARFCompileUnit>(DwarfContext->getUnitAtIndex(0));

  // Get the compile unit DIE is valid.
  auto CUDie = U->getUnitDIE(false);
  EXPECT_TRUE(CUDie.isValid());
  uint32_t Index;
  DWARFDie A;
  DWARFDie B;

  // Verify the compile unit DIE's children.
  Index = 0;
  for (auto Die : CUDie.children()) {
    switch (Index++) {
      case 0: A = Die; break;
      case 1: B = Die; break;
    }
  }

  EXPECT_EQ(A.getTag(), (dwarf::Tag)Tag::A);
  EXPECT_EQ(B.getTag(), (dwarf::Tag)Tag::B);

  // Verify that A has no children by verifying that the begin and end contain
  // invalid DIEs and also that the iterators are equal.
  EXPECT_EQ(A.begin(), A.end());
}

TEST(DWARFDebugInfo, TestChildIteratorsOnInvalidDie) {
  // Verify that an invalid DIE has no children.
  DWARFDie Invalid;
  auto begin = Invalid.begin();
  auto end = Invalid.end();
  EXPECT_FALSE(begin->isValid());
  EXPECT_FALSE(end->isValid());
  EXPECT_EQ(begin, end);
}

TEST(DWARFDebugInfo, TestEmptyChildren) {
  const char *yamldata = "debug_abbrev:\n"
                         "  - Code:            0x00000001\n"
                         "    Tag:             DW_TAG_compile_unit\n"
                         "    Children:        DW_CHILDREN_yes\n"
                         "    Attributes:\n"
                         "debug_info:\n"
                         "  - Length:          0\n"
                         "    Version:         4\n"
                         "    AbbrOffset:      0\n"
                         "    AddrSize:        8\n"
                         "    Entries:\n"
                         "      - AbbrCode:        0x00000001\n"
                         "        Values:\n"
                         "      - AbbrCode:        0x00000000\n"
                         "        Values:\n";

  auto ErrOrSections = DWARFYAML::emitDebugSections(StringRef(yamldata), true);
  ASSERT_TRUE((bool)ErrOrSections);
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);

  // Verify the number of compile units is correct.
  uint32_t NumCUs = DwarfContext->getNumCompileUnits();
  EXPECT_EQ(NumCUs, 1u);
  DWARFCompileUnit *U =
      cast<DWARFCompileUnit>(DwarfContext->getUnitAtIndex(0));

  // Get the compile unit DIE is valid.
  auto CUDie = U->getUnitDIE(false);
  EXPECT_TRUE(CUDie.isValid());

  // Verify that the CU Die that says it has children, but doesn't, actually
  // has begin and end iterators that are equal. We want to make sure we don't
  // see the Null DIEs during iteration.
  EXPECT_EQ(CUDie.begin(), CUDie.end());
}

TEST(DWARFDebugInfo, TestAttributeIterators) {
  Triple Triple = getNormalizedDefaultTargetTriple();
  if (!isConfigurationSupported(Triple))
    return;

  // Test the DWARF APIs related to iterating across all attribute values in a
  // a DWARFDie.
  uint16_t Version = 4;
  auto ExpectedDG = dwarfgen::Generator::create(Triple, Version);
  ASSERT_THAT_EXPECTED(ExpectedDG, Succeeded());
  dwarfgen::Generator *DG = ExpectedDG.get().get();
  dwarfgen::CompileUnit &CU = DG->addCompileUnit();
  const uint64_t CULowPC = 0x1000;
  StringRef CUPath("/tmp/main.c");

  // Scope to allow us to re-use the same DIE names
  {
    auto CUDie = CU.getUnitDIE();
    // Encode an attribute value before an attribute with no data.
    CUDie.addAttribute(DW_AT_name, DW_FORM_strp, CUPath.data());
    // Encode an attribute value with no data in .debug_info/types to ensure
    // the iteration works correctly.
    CUDie.addAttribute(DW_AT_declaration, DW_FORM_flag_present);
    // Encode an attribute value after an attribute with no data.
    CUDie.addAttribute(DW_AT_low_pc, DW_FORM_addr, CULowPC);
  }

  MemoryBufferRef FileBuffer(DG->generate(), "dwarf");
  auto Obj = object::ObjectFile::createObjectFile(FileBuffer);
  EXPECT_TRUE((bool)Obj);
  std::unique_ptr<DWARFContext> DwarfContext = DWARFContext::create(**Obj);

  // Verify the number of compile units is correct.
  uint32_t NumCUs = DwarfContext->getNumCompileUnits();
  EXPECT_EQ(NumCUs, 1u);
  DWARFCompileUnit *U =
      cast<DWARFCompileUnit>(DwarfContext->getUnitAtIndex(0));

  // Get the compile unit DIE is valid.
  auto CUDie = U->getUnitDIE(false);
  EXPECT_TRUE(CUDie.isValid());

  auto R = CUDie.attributes();
  auto I = R.begin();
  auto E = R.end();

  ASSERT_NE(E, I);
  EXPECT_EQ(I->Attr, DW_AT_name);
  auto ActualCUPath = I->Value.getAsCString();
  EXPECT_EQ(CUPath, *ActualCUPath);

  ASSERT_NE(E, ++I);
  EXPECT_EQ(I->Attr, DW_AT_declaration);
  EXPECT_EQ(1ull, *I->Value.getAsUnsignedConstant());

  ASSERT_NE(E, ++I);
  EXPECT_EQ(I->Attr, DW_AT_low_pc);
  EXPECT_EQ(CULowPC, *I->Value.getAsAddress());

  EXPECT_EQ(E, ++I);
}

TEST(DWARFDebugInfo, TestFindRecurse) {
  Triple Triple = getNormalizedDefaultTargetTriple();
  if (!isConfigurationSupported(Triple))
    return;

  uint16_t Version = 4;
  auto ExpectedDG = dwarfgen::Generator::create(Triple, Version);
  ASSERT_THAT_EXPECTED(ExpectedDG, Succeeded());
  dwarfgen::Generator *DG = ExpectedDG.get().get();
  dwarfgen::CompileUnit &CU = DG->addCompileUnit();

  StringRef SpecDieName = "spec";
  StringRef SpecLinkageName = "spec_linkage";
  StringRef AbsDieName = "abs";
  // Scope to allow us to re-use the same DIE names
  {
    auto CUDie = CU.getUnitDIE();
    auto FuncSpecDie = CUDie.addChild(DW_TAG_subprogram);
    auto FuncAbsDie = CUDie.addChild(DW_TAG_subprogram);
    // Put the linkage name in a second abstract origin DIE to ensure we
    // recurse through more than just one DIE when looking for attributes.
    auto FuncAbsDie2 = CUDie.addChild(DW_TAG_subprogram);
    auto FuncDie = CUDie.addChild(DW_TAG_subprogram);
    auto VarAbsDie = CUDie.addChild(DW_TAG_variable);
    auto VarDie = CUDie.addChild(DW_TAG_variable);
    FuncSpecDie.addAttribute(DW_AT_name, DW_FORM_strp, SpecDieName);
    FuncAbsDie2.addAttribute(DW_AT_linkage_name, DW_FORM_strp, SpecLinkageName);
    FuncAbsDie.addAttribute(DW_AT_specification, DW_FORM_ref4, FuncSpecDie);
    FuncAbsDie.addAttribute(DW_AT_abstract_origin, DW_FORM_ref4, FuncAbsDie2);
    FuncDie.addAttribute(DW_AT_abstract_origin, DW_FORM_ref4, FuncAbsDie);
    VarAbsDie.addAttribute(DW_AT_name, DW_FORM_strp, AbsDieName);
    VarDie.addAttribute(DW_AT_abstract_origin, DW_FORM_ref4, VarAbsDie);
  }

  MemoryBufferRef FileBuffer(DG->generate(), "dwarf");
  auto Obj = object::ObjectFile::createObjectFile(FileBuffer);
  EXPECT_TRUE((bool)Obj);
  std::unique_ptr<DWARFContext> DwarfContext = DWARFContext::create(**Obj);

  // Verify the number of compile units is correct.
  uint32_t NumCUs = DwarfContext->getNumCompileUnits();
  EXPECT_EQ(NumCUs, 1u);
  DWARFCompileUnit *U =
      cast<DWARFCompileUnit>(DwarfContext->getUnitAtIndex(0));

  // Get the compile unit DIE is valid.
  auto CUDie = U->getUnitDIE(false);
  EXPECT_TRUE(CUDie.isValid());

  auto FuncSpecDie = CUDie.getFirstChild();
  auto FuncAbsDie = FuncSpecDie.getSibling();
  auto FuncAbsDie2 = FuncAbsDie.getSibling();
  auto FuncDie = FuncAbsDie2.getSibling();
  auto VarAbsDie = FuncDie.getSibling();
  auto VarDie = VarAbsDie.getSibling();

  // Make sure we can't extract the name from the specification die when using
  // DWARFDie::find() since it won't check the DW_AT_specification DIE.
  EXPECT_FALSE(FuncDie.find(DW_AT_name));

  // Make sure we can extract the name from the specification die when using
  // DWARFDie::findRecursively() since it should recurse through the
  // DW_AT_specification DIE.
  auto NameOpt = FuncDie.findRecursively(DW_AT_name);
  EXPECT_TRUE(NameOpt);
  // Test the dwarf::toString() helper function.
  auto StringOpt = toString(NameOpt);
  EXPECT_TRUE(StringOpt);
  EXPECT_EQ(SpecDieName, StringOpt.getValueOr(nullptr));
  // Test the dwarf::toString() helper function with a default value specified.
  EXPECT_EQ(SpecDieName, toString(NameOpt, nullptr));

  auto LinkageNameOpt = FuncDie.findRecursively(DW_AT_linkage_name);
  EXPECT_EQ(SpecLinkageName, toString(LinkageNameOpt).getValueOr(nullptr));

  // Make sure we can't extract the name from the abstract origin die when using
  // DWARFDie::find() since it won't check the DW_AT_abstract_origin DIE.
  EXPECT_FALSE(VarDie.find(DW_AT_name));

  // Make sure we can extract the name from the abstract origin die when using
  // DWARFDie::findRecursively() since it should recurse through the
  // DW_AT_abstract_origin DIE.
  NameOpt = VarDie.findRecursively(DW_AT_name);
  EXPECT_TRUE(NameOpt);
  // Test the dwarf::toString() helper function.
  StringOpt = toString(NameOpt);
  EXPECT_TRUE(StringOpt);
  EXPECT_EQ(AbsDieName, StringOpt.getValueOr(nullptr));
}

TEST(DWARFDebugInfo, TestDwarfToFunctions) {
  // Test all of the dwarf::toXXX functions that take a
  // Optional<DWARFFormValue> and extract the values from it.
  uint64_t InvalidU64 = 0xBADBADBADBADBADB;
  int64_t InvalidS64 = 0xBADBADBADBADBADB;

  // First test that we don't get valid values back when using an optional with
  // no value.
  Optional<DWARFFormValue> FormValOpt1 = DWARFFormValue();
  EXPECT_FALSE(toString(FormValOpt1).hasValue());
  EXPECT_FALSE(toUnsigned(FormValOpt1).hasValue());
  EXPECT_FALSE(toReference(FormValOpt1).hasValue());
  EXPECT_FALSE(toSigned(FormValOpt1).hasValue());
  EXPECT_FALSE(toAddress(FormValOpt1).hasValue());
  EXPECT_FALSE(toSectionOffset(FormValOpt1).hasValue());
  EXPECT_FALSE(toBlock(FormValOpt1).hasValue());
  EXPECT_EQ(nullptr, toString(FormValOpt1, nullptr));
  EXPECT_EQ(InvalidU64, toUnsigned(FormValOpt1, InvalidU64));
  EXPECT_EQ(InvalidU64, toReference(FormValOpt1, InvalidU64));
  EXPECT_EQ(InvalidU64, toAddress(FormValOpt1, InvalidU64));
  EXPECT_EQ(InvalidU64, toSectionOffset(FormValOpt1, InvalidU64));
  EXPECT_EQ(InvalidS64, toSigned(FormValOpt1, InvalidS64));

  // Test successful and unsuccessful address decoding.
  uint64_t Address = 0x100000000ULL;
  Optional<DWARFFormValue> FormValOpt2 =
      DWARFFormValue::createFromUValue(DW_FORM_addr, Address);

  EXPECT_FALSE(toString(FormValOpt2).hasValue());
  EXPECT_FALSE(toUnsigned(FormValOpt2).hasValue());
  EXPECT_FALSE(toReference(FormValOpt2).hasValue());
  EXPECT_FALSE(toSigned(FormValOpt2).hasValue());
  EXPECT_TRUE(toAddress(FormValOpt2).hasValue());
  EXPECT_FALSE(toSectionOffset(FormValOpt2).hasValue());
  EXPECT_FALSE(toBlock(FormValOpt2).hasValue());
  EXPECT_EQ(nullptr, toString(FormValOpt2, nullptr));
  EXPECT_EQ(InvalidU64, toUnsigned(FormValOpt2, InvalidU64));
  EXPECT_EQ(InvalidU64, toReference(FormValOpt2, InvalidU64));
  EXPECT_EQ(Address, toAddress(FormValOpt2, InvalidU64));
  EXPECT_EQ(InvalidU64, toSectionOffset(FormValOpt2, InvalidU64));
  EXPECT_EQ(InvalidS64, toSigned(FormValOpt2, InvalidU64));

  // Test successful and unsuccessful unsigned constant decoding.
  uint64_t UData8 = 0x1020304050607080ULL;
  Optional<DWARFFormValue> FormValOpt3 =
      DWARFFormValue::createFromUValue(DW_FORM_udata, UData8);

  EXPECT_FALSE(toString(FormValOpt3).hasValue());
  EXPECT_TRUE(toUnsigned(FormValOpt3).hasValue());
  EXPECT_FALSE(toReference(FormValOpt3).hasValue());
  EXPECT_TRUE(toSigned(FormValOpt3).hasValue());
  EXPECT_FALSE(toAddress(FormValOpt3).hasValue());
  EXPECT_FALSE(toSectionOffset(FormValOpt3).hasValue());
  EXPECT_FALSE(toBlock(FormValOpt3).hasValue());
  EXPECT_EQ(nullptr, toString(FormValOpt3, nullptr));
  EXPECT_EQ(UData8, toUnsigned(FormValOpt3, InvalidU64));
  EXPECT_EQ(InvalidU64, toReference(FormValOpt3, InvalidU64));
  EXPECT_EQ(InvalidU64, toAddress(FormValOpt3, InvalidU64));
  EXPECT_EQ(InvalidU64, toSectionOffset(FormValOpt3, InvalidU64));
  EXPECT_EQ((int64_t)UData8, toSigned(FormValOpt3, InvalidU64));

  // Test successful and unsuccessful reference decoding.
  uint32_t RefData = 0x11223344U;
  Optional<DWARFFormValue> FormValOpt4 =
      DWARFFormValue::createFromUValue(DW_FORM_ref_addr, RefData);

  EXPECT_FALSE(toString(FormValOpt4).hasValue());
  EXPECT_FALSE(toUnsigned(FormValOpt4).hasValue());
  EXPECT_TRUE(toReference(FormValOpt4).hasValue());
  EXPECT_FALSE(toSigned(FormValOpt4).hasValue());
  EXPECT_FALSE(toAddress(FormValOpt4).hasValue());
  EXPECT_FALSE(toSectionOffset(FormValOpt4).hasValue());
  EXPECT_FALSE(toBlock(FormValOpt4).hasValue());
  EXPECT_EQ(nullptr, toString(FormValOpt4, nullptr));
  EXPECT_EQ(InvalidU64, toUnsigned(FormValOpt4, InvalidU64));
  EXPECT_EQ(RefData, toReference(FormValOpt4, InvalidU64));
  EXPECT_EQ(InvalidU64, toAddress(FormValOpt4, InvalidU64));
  EXPECT_EQ(InvalidU64, toSectionOffset(FormValOpt4, InvalidU64));
  EXPECT_EQ(InvalidS64, toSigned(FormValOpt4, InvalidU64));

  // Test successful and unsuccessful signed constant decoding.
  int64_t SData8 = 0x1020304050607080ULL;
  Optional<DWARFFormValue> FormValOpt5 =
      DWARFFormValue::createFromSValue(DW_FORM_udata, SData8);

  EXPECT_FALSE(toString(FormValOpt5).hasValue());
  EXPECT_TRUE(toUnsigned(FormValOpt5).hasValue());
  EXPECT_FALSE(toReference(FormValOpt5).hasValue());
  EXPECT_TRUE(toSigned(FormValOpt5).hasValue());
  EXPECT_FALSE(toAddress(FormValOpt5).hasValue());
  EXPECT_FALSE(toSectionOffset(FormValOpt5).hasValue());
  EXPECT_FALSE(toBlock(FormValOpt5).hasValue());
  EXPECT_EQ(nullptr, toString(FormValOpt5, nullptr));
  EXPECT_EQ((uint64_t)SData8, toUnsigned(FormValOpt5, InvalidU64));
  EXPECT_EQ(InvalidU64, toReference(FormValOpt5, InvalidU64));
  EXPECT_EQ(InvalidU64, toAddress(FormValOpt5, InvalidU64));
  EXPECT_EQ(InvalidU64, toSectionOffset(FormValOpt5, InvalidU64));
  EXPECT_EQ(SData8, toSigned(FormValOpt5, InvalidU64));

  // Test successful and unsuccessful block decoding.
  uint8_t Data[] = { 2, 3, 4 };
  ArrayRef<uint8_t> Array(Data);
  Optional<DWARFFormValue> FormValOpt6 =
      DWARFFormValue::createFromBlockValue(DW_FORM_block1, Array);

  EXPECT_FALSE(toString(FormValOpt6).hasValue());
  EXPECT_FALSE(toUnsigned(FormValOpt6).hasValue());
  EXPECT_FALSE(toReference(FormValOpt6).hasValue());
  EXPECT_FALSE(toSigned(FormValOpt6).hasValue());
  EXPECT_FALSE(toAddress(FormValOpt6).hasValue());
  EXPECT_FALSE(toSectionOffset(FormValOpt6).hasValue());
  auto BlockOpt = toBlock(FormValOpt6);
  EXPECT_TRUE(BlockOpt.hasValue());
  EXPECT_EQ(*BlockOpt, Array);
  EXPECT_EQ(nullptr, toString(FormValOpt6, nullptr));
  EXPECT_EQ(InvalidU64, toUnsigned(FormValOpt6, InvalidU64));
  EXPECT_EQ(InvalidU64, toReference(FormValOpt6, InvalidU64));
  EXPECT_EQ(InvalidU64, toAddress(FormValOpt6, InvalidU64));
  EXPECT_EQ(InvalidU64, toSectionOffset(FormValOpt6, InvalidU64));
  EXPECT_EQ(InvalidS64, toSigned(FormValOpt6, InvalidU64));

  // Test
}

TEST(DWARFDebugInfo, TestFindAttrs) {
  Triple Triple = getNormalizedDefaultTargetTriple();
  if (!isConfigurationSupported(Triple))
    return;

  // Test the DWARFDie::find() and DWARFDie::findRecursively() that take an
  // ArrayRef<dwarf::Attribute> value to make sure they work correctly.
  uint16_t Version = 4;
  auto ExpectedDG = dwarfgen::Generator::create(Triple, Version);
  ASSERT_THAT_EXPECTED(ExpectedDG, Succeeded());
  dwarfgen::Generator *DG = ExpectedDG.get().get();
  dwarfgen::CompileUnit &CU = DG->addCompileUnit();

  StringRef DieMangled("_Z3fooi");
  // Scope to allow us to re-use the same DIE names
  {
    auto CUDie = CU.getUnitDIE();
    auto FuncSpecDie = CUDie.addChild(DW_TAG_subprogram);
    auto FuncDie = CUDie.addChild(DW_TAG_subprogram);
    FuncSpecDie.addAttribute(DW_AT_MIPS_linkage_name, DW_FORM_strp, DieMangled);
    FuncDie.addAttribute(DW_AT_specification, DW_FORM_ref4, FuncSpecDie);
  }

  MemoryBufferRef FileBuffer(DG->generate(), "dwarf");
  auto Obj = object::ObjectFile::createObjectFile(FileBuffer);
  EXPECT_TRUE((bool)Obj);
  std::unique_ptr<DWARFContext> DwarfContext = DWARFContext::create(**Obj);

  // Verify the number of compile units is correct.
  uint32_t NumCUs = DwarfContext->getNumCompileUnits();
  EXPECT_EQ(NumCUs, 1u);
  DWARFCompileUnit *U =
      cast<DWARFCompileUnit>(DwarfContext->getUnitAtIndex(0));

  // Get the compile unit DIE is valid.
  auto CUDie = U->getUnitDIE(false);
  EXPECT_TRUE(CUDie.isValid());

  auto FuncSpecDie = CUDie.getFirstChild();
  auto FuncDie = FuncSpecDie.getSibling();

  // Make sure that passing in an empty attribute list behave correctly.
  EXPECT_FALSE(FuncDie.find(ArrayRef<dwarf::Attribute>()).hasValue());

  // Make sure that passing in a list of attribute that are not contained
  // in the DIE returns nothing.
  EXPECT_FALSE(FuncDie.find({DW_AT_low_pc, DW_AT_entry_pc}).hasValue());

  const dwarf::Attribute Attrs[] = {DW_AT_linkage_name,
                                    DW_AT_MIPS_linkage_name};

  // Make sure we can't extract the linkage name attributes when using
  // DWARFDie::find() since it won't check the DW_AT_specification DIE.
  EXPECT_FALSE(FuncDie.find(Attrs).hasValue());

  // Make sure we can extract the name from the specification die when using
  // DWARFDie::findRecursively() since it should recurse through the
  // DW_AT_specification DIE.
  auto NameOpt = FuncDie.findRecursively(Attrs);
  EXPECT_TRUE(NameOpt.hasValue());
  EXPECT_EQ(DieMangled, toString(NameOpt, ""));
}

TEST(DWARFDebugInfo, TestImplicitConstAbbrevs) {
  Triple Triple = getNormalizedDefaultTargetTriple();
  if (!isConfigurationSupported(Triple))
    return;

  uint16_t Version = 5;
  auto ExpectedDG = dwarfgen::Generator::create(Triple, Version);
  ASSERT_THAT_EXPECTED(ExpectedDG, Succeeded());
  dwarfgen::Generator *DG = ExpectedDG.get().get();
  dwarfgen::CompileUnit &CU = DG->addCompileUnit();
  dwarfgen::DIE CUDie = CU.getUnitDIE();
  const dwarf::Attribute Attr = DW_AT_lo_user;
  const int64_t Val1 = 42;
  const int64_t Val2 = 43;

  auto FirstVal1DIE = CUDie.addChild(DW_TAG_class_type);
  FirstVal1DIE.addAttribute(Attr, DW_FORM_implicit_const, Val1);

  auto SecondVal1DIE = CUDie.addChild(DW_TAG_class_type);
  SecondVal1DIE.addAttribute(Attr, DW_FORM_implicit_const, Val1);

  auto Val2DIE = CUDie.addChild(DW_TAG_class_type);
  Val2DIE.addAttribute(Attr, DW_FORM_implicit_const, Val2);

  MemoryBufferRef FileBuffer(DG->generate(), "dwarf");
  auto Obj = object::ObjectFile::createObjectFile(FileBuffer);
  EXPECT_TRUE((bool)Obj);
  std::unique_ptr<DWARFContext> DwarfContext = DWARFContext::create(**Obj);
  DWARFCompileUnit *U =
      cast<DWARFCompileUnit>(DwarfContext->getUnitAtIndex(0));
  EXPECT_TRUE((bool)U);

  const auto *Abbrevs = U->getAbbreviations();
  EXPECT_TRUE((bool)Abbrevs);

  // Let's find implicit_const abbrevs and verify,
  // that there are exactly two of them and both of them
  // can be dumped correctly.
  typedef decltype(Abbrevs->begin()) AbbrevIt;
  AbbrevIt Val1Abbrev = Abbrevs->end();
  AbbrevIt Val2Abbrev = Abbrevs->end();
  for(auto it = Abbrevs->begin(); it != Abbrevs->end(); ++it) {
    if (it->getNumAttributes() == 0)
      continue; // root abbrev for DW_TAG_compile_unit

    auto A = it->getAttrByIndex(0);
    EXPECT_EQ(A, Attr);

    auto FormValue = it->getAttributeValue(/* offset */ 0, A, *U);
    EXPECT_TRUE((bool)FormValue);
    EXPECT_EQ(FormValue->getForm(), dwarf::DW_FORM_implicit_const);

    const auto V = FormValue->getAsSignedConstant();
    EXPECT_TRUE((bool)V);

    auto VerifyAbbrevDump = [&V](AbbrevIt it) {
      std::string S;
      llvm::raw_string_ostream OS(S);
      it->dump(OS);
      auto FormPos = OS.str().find("DW_FORM_implicit_const");
      EXPECT_NE(FormPos, std::string::npos);
      auto ValPos = S.find_first_of("-0123456789", FormPos);
      EXPECT_NE(ValPos, std::string::npos);
      int64_t Val = std::atoll(S.substr(ValPos).c_str());
      EXPECT_EQ(Val, *V);
    };

    switch(*V) {
    case Val1:
      EXPECT_EQ(Val1Abbrev, Abbrevs->end());
      Val1Abbrev = it;
      VerifyAbbrevDump(it);
      break;
    case Val2:
      EXPECT_EQ(Val2Abbrev, Abbrevs->end());
      Val2Abbrev = it;
      VerifyAbbrevDump(it);
      break;
    default:
      FAIL() << "Unexpected attribute value: " << *V;
    }
  }

  // Now let's make sure that two Val1-DIEs refer to the same abbrev,
  // and Val2-DIE refers to another one.
  auto DieDG = U->getUnitDIE(false);
  auto it = DieDG.begin();
  std::multimap<int64_t, decltype(it->getAbbreviationDeclarationPtr())> DIEs;
  const DWARFAbbreviationDeclaration *AbbrevPtrVal1 = nullptr;
  const DWARFAbbreviationDeclaration *AbbrevPtrVal2 = nullptr;
  for (; it != DieDG.end(); ++it) {
    const auto *AbbrevPtr = it->getAbbreviationDeclarationPtr();
    EXPECT_TRUE((bool)AbbrevPtr);
    auto FormValue = it->find(Attr);
    EXPECT_TRUE((bool)FormValue);
    const auto V = FormValue->getAsSignedConstant();
    EXPECT_TRUE((bool)V);
    switch(*V) {
    case Val1:
      AbbrevPtrVal1 = AbbrevPtr;
      break;
    case Val2:
      AbbrevPtrVal2 = AbbrevPtr;
      break;
    default:
      FAIL() << "Unexpected attribute value: " << *V;
    }
    DIEs.insert(std::make_pair(*V, AbbrevPtr));
  }
  EXPECT_EQ(DIEs.count(Val1), 2u);
  EXPECT_EQ(DIEs.count(Val2), 1u);
  auto Val1Range = DIEs.equal_range(Val1);
  for (auto it = Val1Range.first; it != Val1Range.second; ++it)
    EXPECT_EQ(it->second, AbbrevPtrVal1);
  EXPECT_EQ(DIEs.find(Val2)->second, AbbrevPtrVal2);
}

void VerifyWarning(DWARFContext &DwarfContext, StringRef Error) {
  SmallString<1024> Str;
  raw_svector_ostream Strm(Str);
  EXPECT_TRUE(DwarfContext.verify(Strm));
  EXPECT_TRUE(Str.str().contains(Error));
}

void VerifyError(DWARFContext &DwarfContext, StringRef Error) {
  SmallString<1024> Str;
  raw_svector_ostream Strm(Str);
  EXPECT_FALSE(DwarfContext.verify(Strm));
  EXPECT_TRUE(Str.str().contains(Error));
}

void VerifySuccess(DWARFContext &DwarfContext) {
  SmallString<1024> Str;
  raw_svector_ostream Strm(Str);
  EXPECT_TRUE(DwarfContext.verify(Strm));
}

TEST(DWARFDebugInfo, TestDwarfVerifyInvalidCURef) {
  // Create a single compile unit with a single function that has a DW_AT_type
  // that is CU relative. The CU offset is not valid because it is larger than
  // the compile unit itself.

  const char *yamldata = R"(
    debug_str:
      - ''
      - /tmp/main.c
      - main
    debug_abbrev:
      - Code:            0x00000001
        Tag:             DW_TAG_compile_unit
        Children:        DW_CHILDREN_yes
        Attributes:
          - Attribute:       DW_AT_name
            Form:            DW_FORM_strp
      - Code:            0x00000002
        Tag:             DW_TAG_subprogram
        Children:        DW_CHILDREN_no
        Attributes:
          - Attribute:       DW_AT_name
            Form:            DW_FORM_strp
          - Attribute:       DW_AT_type
            Form:            DW_FORM_ref4
    debug_info:
      - Length:          22
        Version:         4
        AbbrOffset:      0
        AddrSize:        8
        Entries:
          - AbbrCode:        0x00000001
            Values:
              - Value:           0x0000000000000001
          - AbbrCode:        0x00000002
            Values:
              - Value:           0x000000000000000D
              - Value:           0x0000000000001234
          - AbbrCode:        0x00000000
            Values:
  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(StringRef(yamldata));
  ASSERT_TRUE((bool)ErrOrSections);
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);
  VerifyError(*DwarfContext, "error: DW_FORM_ref4 CU offset 0x00001234 is "
                             "invalid (must be less than CU size of "
                             "0x0000001a):");
}

TEST(DWARFDebugInfo, TestDwarfVerifyInvalidRefAddr) {
  // Create a single compile unit with a single function that has an invalid
  // DW_AT_type with an invalid .debug_info offset in its DW_FORM_ref_addr.
  const char *yamldata = R"(
    debug_str:
      - ''
      - /tmp/main.c
      - main
    debug_abbrev:
      - Code:            0x00000001
        Tag:             DW_TAG_compile_unit
        Children:        DW_CHILDREN_yes
        Attributes:
          - Attribute:       DW_AT_name
            Form:            DW_FORM_strp
      - Code:            0x00000002
        Tag:             DW_TAG_subprogram
        Children:        DW_CHILDREN_no
        Attributes:
          - Attribute:       DW_AT_name
            Form:            DW_FORM_strp
          - Attribute:       DW_AT_type
            Form:            DW_FORM_ref_addr
    debug_info:
      - Length:          22
        Version:         4
        AbbrOffset:      0
        AddrSize:        8
        Entries:
          - AbbrCode:        0x00000001
            Values:
              - Value:           0x0000000000000001
          - AbbrCode:        0x00000002
            Values:
              - Value:           0x000000000000000D
              - Value:           0x0000000000001234
          - AbbrCode:        0x00000000
            Values:
  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(StringRef(yamldata));
  ASSERT_TRUE((bool)ErrOrSections);
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);
  VerifyError(*DwarfContext,
              "error: DW_FORM_ref_addr offset beyond .debug_info bounds:");
}

TEST(DWARFDebugInfo, TestDwarfVerifyInvalidRanges) {
  // Create a single compile unit with a DW_AT_ranges whose section offset
  // isn't valid.
  const char *yamldata = R"(
    debug_str:
      - ''
      - /tmp/main.c
    debug_abbrev:
      - Code:            0x00000001
        Tag:             DW_TAG_compile_unit
        Children:        DW_CHILDREN_no
        Attributes:
          - Attribute:       DW_AT_name
            Form:            DW_FORM_strp
          - Attribute:       DW_AT_ranges
            Form:            DW_FORM_sec_offset
    debug_info:
      - Length:          16
        Version:         4
        AbbrOffset:      0
        AddrSize:        8
        Entries:
          - AbbrCode:        0x00000001
            Values:
              - Value:           0x0000000000000001
              - Value:           0x0000000000001000

  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(StringRef(yamldata));
  ASSERT_TRUE((bool)ErrOrSections);
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);
  VerifyError(
      *DwarfContext,
      "error: DW_AT_ranges offset is beyond .debug_ranges bounds: 0x00001000");
}

TEST(DWARFDebugInfo, TestDwarfVerifyInvalidRnglists) {
  // Create a single compile unit with a DW_AT_ranges whose section offset
  // isn't valid.
  const char *yamldata = R"(
    debug_str:
      - ''
      - /tmp/main.c
    debug_abbrev:
      - Code:            0x00000001
        Tag:             DW_TAG_compile_unit
        Children:        DW_CHILDREN_no
        Attributes:
          - Attribute:       DW_AT_name
            Form:            DW_FORM_strp
          - Attribute:       DW_AT_ranges
            Form:            DW_FORM_sec_offset
    debug_info:
      - Length:          17
        Version:         5
        UnitType:        DW_UT_compile
        AbbrOffset:      0
        AddrSize:        8
        Entries:
          - AbbrCode:        0x00000001
            Values:
              - Value:           0x0000000000000001
              - Value:           0x0000000000001000

  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(StringRef(yamldata));
  ASSERT_TRUE((bool)ErrOrSections);
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);
  VerifyError(*DwarfContext, "error: DW_AT_ranges offset is beyond "
                             ".debug_rnglists bounds: 0x00001000");
}

TEST(DWARFDebugInfo, TestDwarfVerifyInvalidStmtList) {
  // Create a single compile unit with a DW_AT_stmt_list whose section offset
  // isn't valid.
  const char *yamldata = R"(
    debug_str:
      - ''
      - /tmp/main.c
    debug_abbrev:
      - Code:            0x00000001
        Tag:             DW_TAG_compile_unit
        Children:        DW_CHILDREN_no
        Attributes:
          - Attribute:       DW_AT_name
            Form:            DW_FORM_strp
          - Attribute:       DW_AT_stmt_list
            Form:            DW_FORM_sec_offset
    debug_info:
      - Length:          16
        Version:         4
        AbbrOffset:      0
        AddrSize:        8
        Entries:
          - AbbrCode:        0x00000001
            Values:
              - Value:           0x0000000000000001
              - Value:           0x0000000000001000

  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(StringRef(yamldata));
  ASSERT_TRUE((bool)ErrOrSections);
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);
  VerifyError(
      *DwarfContext,
      "error: DW_AT_stmt_list offset is beyond .debug_line bounds: 0x00001000");
}

TEST(DWARFDebugInfo, TestDwarfVerifyInvalidStrp) {
  // Create a single compile unit with a single function that has an invalid
  // DW_FORM_strp for the DW_AT_name.
  const char *yamldata = R"(
    debug_str:
      - ''
    debug_abbrev:
      - Code:            0x00000001
        Tag:             DW_TAG_compile_unit
        Children:        DW_CHILDREN_no
        Attributes:
          - Attribute:       DW_AT_name
            Form:            DW_FORM_strp
    debug_info:
      - Length:          12
        Version:         4
        AbbrOffset:      0
        AddrSize:        8
        Entries:
          - AbbrCode:        0x00000001
            Values:
              - Value:           0x0000000000001234
  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(StringRef(yamldata));
  ASSERT_TRUE((bool)ErrOrSections);
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);
  VerifyError(*DwarfContext,
              "error: DW_FORM_strp offset beyond .debug_str bounds:");
}

TEST(DWARFDebugInfo, TestDwarfVerifyInvalidRefAddrBetween) {
  // Create a single compile unit with a single function that has a DW_AT_type
  // with a valid .debug_info offset, but the offset is between two DIEs.
  const char *yamldata = R"(
    debug_str:
      - ''
      - /tmp/main.c
      - main
    debug_abbrev:
      - Code:            0x00000001
        Tag:             DW_TAG_compile_unit
        Children:        DW_CHILDREN_yes
        Attributes:
          - Attribute:       DW_AT_name
            Form:            DW_FORM_strp
      - Code:            0x00000002
        Tag:             DW_TAG_subprogram
        Children:        DW_CHILDREN_no
        Attributes:
          - Attribute:       DW_AT_name
            Form:            DW_FORM_strp
          - Attribute:       DW_AT_type
            Form:            DW_FORM_ref_addr
    debug_info:
      - Length:          22
        Version:         4
        AbbrOffset:      0
        AddrSize:        8
        Entries:
          - AbbrCode:        0x00000001
            Values:
              - Value:           0x0000000000000001
          - AbbrCode:        0x00000002
            Values:
              - Value:           0x000000000000000D
              - Value:           0x0000000000000011
          - AbbrCode:        0x00000000
            Values:
  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(StringRef(yamldata));
  ASSERT_TRUE((bool)ErrOrSections);
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);
  VerifyError(
      *DwarfContext,
      "error: invalid DIE reference 0x00000011. Offset is in between DIEs:");
}

TEST(DWARFDebugInfo, TestDwarfVerifyInvalidLineSequence) {
  // Create a single compile unit whose line table has a sequence in it where
  // the address decreases.
  StringRef yamldata = R"(
    debug_str:
      - ''
      - /tmp/main.c
    debug_abbrev:
      - Code:            0x00000001
        Tag:             DW_TAG_compile_unit
        Children:        DW_CHILDREN_no
        Attributes:
          - Attribute:       DW_AT_name
            Form:            DW_FORM_strp
          - Attribute:       DW_AT_stmt_list
            Form:            DW_FORM_sec_offset
    debug_info:
      - Length:          16
        Version:         4
        AbbrOffset:      0
        AddrSize:        8
        Entries:
          - AbbrCode:        0x00000001
            Values:
              - Value:           0x0000000000000001
              - Value:           0x0000000000000000
    debug_line:
      - Length:          68
        Version:         2
        PrologueLength:  34
        MinInstLength:   1
        DefaultIsStmt:   1
        LineBase:        251
        LineRange:       14
        OpcodeBase:      13
        StandardOpcodeLengths: [ 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1 ]
        IncludeDirs:
          - /tmp
        Files:
          - Name:            main.c
            DirIdx:          1
            ModTime:         0
            Length:          0
        Opcodes:
          - Opcode:          DW_LNS_extended_op
            ExtLen:          9
            SubOpcode:       DW_LNE_set_address
            Data:            4112
          - Opcode:          DW_LNS_advance_line
            SData:           9
            Data:            4112
          - Opcode:          DW_LNS_copy
            Data:            4112
          - Opcode:          DW_LNS_advance_pc
            Data:            18446744073709551600
          - Opcode:          DW_LNS_extended_op
            ExtLen:          1
            SubOpcode:       DW_LNE_end_sequence
            Data:            18446744073709551600
  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(yamldata);
  ASSERT_TRUE((bool)ErrOrSections);
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);
  VerifyError(*DwarfContext, "error: .debug_line[0x00000000] row[1] decreases "
                             "in address from previous row:");
}

TEST(DWARFDebugInfo, TestDwarfVerifyInvalidLineFileIndex) {
  // Create a single compile unit whose line table has a line table row with
  // an invalid file index.
  StringRef yamldata = R"(
    debug_str:
      - ''
      - /tmp/main.c
    debug_abbrev:
      - Code:            0x00000001
        Tag:             DW_TAG_compile_unit
        Children:        DW_CHILDREN_no
        Attributes:
          - Attribute:       DW_AT_name
            Form:            DW_FORM_strp
          - Attribute:       DW_AT_stmt_list
            Form:            DW_FORM_sec_offset
    debug_info:
      - Length:          16
        Version:         4
        AbbrOffset:      0
        AddrSize:        8
        Entries:
          - AbbrCode:        0x00000001
            Values:
              - Value:           0x0000000000000001
              - Value:           0x0000000000000000
    debug_line:
      - Length:          61
        Version:         2
        PrologueLength:  34
        MinInstLength:   1
        DefaultIsStmt:   1
        LineBase:        251
        LineRange:       14
        OpcodeBase:      13
        StandardOpcodeLengths: [ 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1 ]
        IncludeDirs:
          - /tmp
        Files:
          - Name:            main.c
            DirIdx:          1
            ModTime:         0
            Length:          0
        Opcodes:
          - Opcode:          DW_LNS_extended_op
            ExtLen:          9
            SubOpcode:       DW_LNE_set_address
            Data:            4096
          - Opcode:          DW_LNS_advance_line
            SData:           9
            Data:            4096
          - Opcode:          DW_LNS_copy
            Data:            4096
          - Opcode:          DW_LNS_advance_pc
            Data:            16
          - Opcode:          DW_LNS_set_file
            Data:            5
          - Opcode:          DW_LNS_extended_op
            ExtLen:          1
            SubOpcode:       DW_LNE_end_sequence
            Data:            5
  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(yamldata);
  ASSERT_TRUE((bool)ErrOrSections);
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);
  VerifyError(*DwarfContext, "error: .debug_line[0x00000000][1] has invalid "
                             "file index 5 (valid values are [1,1]):");
}

TEST(DWARFDebugInfo, TestDwarfVerifyInvalidLineTablePorlogueDirIndex) {
  // Create a single compile unit whose line table has a prologue with an
  // invalid dir index.
  StringRef yamldata = R"(
    debug_str:
      - ''
      - /tmp/main.c
    debug_abbrev:
      - Code:            0x00000001
        Tag:             DW_TAG_compile_unit
        Children:        DW_CHILDREN_no
        Attributes:
          - Attribute:       DW_AT_name
            Form:            DW_FORM_strp
          - Attribute:       DW_AT_stmt_list
            Form:            DW_FORM_sec_offset
    debug_info:
      - Length:          16
        Version:         4
        AbbrOffset:      0
        AddrSize:        8
        Entries:
          - AbbrCode:        0x00000001
            Values:
              - Value:           0x0000000000000001
              - Value:           0x0000000000000000
    debug_line:
      - Length:          61
        Version:         2
        PrologueLength:  34
        MinInstLength:   1
        DefaultIsStmt:   1
        LineBase:        251
        LineRange:       14
        OpcodeBase:      13
        StandardOpcodeLengths: [ 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1 ]
        IncludeDirs:
          - /tmp
        Files:
          - Name:            main.c
            DirIdx:          2
            ModTime:         0
            Length:          0
        Opcodes:
          - Opcode:          DW_LNS_extended_op
            ExtLen:          9
            SubOpcode:       DW_LNE_set_address
            Data:            4096
          - Opcode:          DW_LNS_advance_line
            SData:           9
            Data:            4096
          - Opcode:          DW_LNS_copy
            Data:            4096
          - Opcode:          DW_LNS_advance_pc
            Data:            16
          - Opcode:          DW_LNS_set_file
            Data:            1
          - Opcode:          DW_LNS_extended_op
            ExtLen:          1
            SubOpcode:       DW_LNE_end_sequence
            Data:            1
  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(yamldata);
  ASSERT_TRUE((bool)ErrOrSections);
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);
  VerifyError(*DwarfContext,
              "error: .debug_line[0x00000000].prologue."
              "file_names[1].dir_idx contains an invalid index: 2");
}

TEST(DWARFDebugInfo, TestDwarfVerifyDuplicateFileWarning) {
  // Create a single compile unit whose line table has a prologue with an
  // invalid dir index.
  StringRef yamldata = R"(
    debug_str:
      - ''
      - /tmp/main.c
    debug_abbrev:
      - Code:            0x00000001
        Tag:             DW_TAG_compile_unit
        Children:        DW_CHILDREN_no
        Attributes:
          - Attribute:       DW_AT_name
            Form:            DW_FORM_strp
          - Attribute:       DW_AT_stmt_list
            Form:            DW_FORM_sec_offset
    debug_info:
      - Length:          16
        Version:         4
        AbbrOffset:      0
        AddrSize:        8
        Entries:
          - AbbrCode:        0x00000001
            Values:
              - Value:           0x0000000000000001
              - Value:           0x0000000000000000
    debug_line:
      - Length:          71
        Version:         2
        PrologueLength:  44
        MinInstLength:   1
        DefaultIsStmt:   1
        LineBase:        251
        LineRange:       14
        OpcodeBase:      13
        StandardOpcodeLengths: [ 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1 ]
        IncludeDirs:
          - /tmp
        Files:
          - Name:            main.c
            DirIdx:          1
            ModTime:         0
            Length:          0
          - Name:            main.c
            DirIdx:          1
            ModTime:         0
            Length:          0
        Opcodes:
          - Opcode:          DW_LNS_extended_op
            ExtLen:          9
            SubOpcode:       DW_LNE_set_address
            Data:            4096
          - Opcode:          DW_LNS_advance_line
            SData:           9
            Data:            4096
          - Opcode:          DW_LNS_copy
            Data:            4096
          - Opcode:          DW_LNS_advance_pc
            Data:            16
          - Opcode:          DW_LNS_set_file
            Data:            1
          - Opcode:          DW_LNS_extended_op
            ExtLen:          1
            SubOpcode:       DW_LNE_end_sequence
            Data:            2
  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(yamldata);
  ASSERT_TRUE((bool)ErrOrSections);
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);
  VerifyWarning(*DwarfContext,
                "warning: .debug_line[0x00000000].prologue.file_names[2] is "
                "a duplicate of file_names[1]");
}

TEST(DWARFDebugInfo, TestDwarfVerifyCUDontShareLineTable) {
  // Create a two compile units where both compile units share the same
  // DW_AT_stmt_list value and verify we report the error correctly.
  StringRef yamldata = R"(
    debug_str:
      - ''
      - /tmp/main.c
      - /tmp/foo.c
    debug_abbrev:
      - Code:            0x00000001
        Tag:             DW_TAG_compile_unit
        Children:        DW_CHILDREN_no
        Attributes:
          - Attribute:       DW_AT_name
            Form:            DW_FORM_strp
          - Attribute:       DW_AT_stmt_list
            Form:            DW_FORM_sec_offset
    debug_info:
      - Length:          16
        Version:         4
        AbbrOffset:      0
        AddrSize:        8
        Entries:
          - AbbrCode:        0x00000001
            Values:
              - Value:           0x0000000000000001
              - Value:           0x0000000000000000
      - Length:          16
        Version:         4
        AbbrOffset:      0
        AddrSize:        8
        Entries:
          - AbbrCode:        0x00000001
            Values:
              - Value:           0x000000000000000D
              - Value:           0x0000000000000000
    debug_line:
      - Length:          60
        Version:         2
        PrologueLength:  34
        MinInstLength:   1
        DefaultIsStmt:   1
        LineBase:        251
        LineRange:       14
        OpcodeBase:      13
        StandardOpcodeLengths: [ 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1 ]
        IncludeDirs:
          - /tmp
        Files:
          - Name:            main.c
            DirIdx:          1
            ModTime:         0
            Length:          0
        Opcodes:
          - Opcode:          DW_LNS_extended_op
            ExtLen:          9
            SubOpcode:       DW_LNE_set_address
            Data:            4096
          - Opcode:          DW_LNS_advance_line
            SData:           9
            Data:            4096
          - Opcode:          DW_LNS_copy
            Data:            4096
          - Opcode:          DW_LNS_advance_pc
            Data:            256
          - Opcode:          DW_LNS_extended_op
            ExtLen:          1
            SubOpcode:       DW_LNE_end_sequence
            Data:            256
  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(yamldata);
  ASSERT_TRUE((bool)ErrOrSections);
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);
  VerifyError(*DwarfContext,
              "error: two compile unit DIEs, 0x0000000b and "
              "0x0000001f, have the same DW_AT_stmt_list section "
              "offset:");
}

TEST(DWARFDebugInfo, TestErrorReporting) {
  Triple Triple("x86_64-pc-linux");
  if (!isConfigurationSupported(Triple))
      return;

  auto ExpectedDG = dwarfgen::Generator::create(Triple, 4 /*DwarfVersion*/);
  ASSERT_THAT_EXPECTED(ExpectedDG, Succeeded());
  dwarfgen::Generator *DG = ExpectedDG.get().get();
  AsmPrinter *AP = DG->getAsmPrinter();
  MCContext *MC = DG->getMCContext();

  // Emit two compressed sections with broken headers.
  AP->OutStreamer->SwitchSection(
      MC->getELFSection(".zdebug_foo", 0 /*Type*/, 0 /*Flags*/));
  AP->OutStreamer->emitBytes("0");
  AP->OutStreamer->SwitchSection(
      MC->getELFSection(".zdebug_bar", 0 /*Type*/, 0 /*Flags*/));
  AP->OutStreamer->emitBytes("0");

  MemoryBufferRef FileBuffer(DG->generate(), "dwarf");
  auto Obj = object::ObjectFile::createObjectFile(FileBuffer);
  EXPECT_TRUE((bool)Obj);

  // DWARFContext parses whole file and finds the two errors we expect.
  int Errors = 0;
  std::unique_ptr<DWARFContext> Ctx1 =
      DWARFContext::create(**Obj, nullptr, "", [&](Error E) {
        ++Errors;
        consumeError(std::move(E));
      });
  EXPECT_TRUE(Errors == 2);
}

TEST(DWARFDebugInfo, TestDwarfVerifyCURangesIncomplete) {
  // Create a single compile unit with a single function. The compile
  // unit has a DW_AT_ranges attribute that doesn't fully contain the
  // address range of the function. The verification should fail due to
  // the CU ranges not containing all of the address ranges of all of the
  // functions.
  StringRef yamldata = R"(
    debug_str:
      - ''
      - /tmp/main.c
    debug_abbrev:
      - Code:            0x00000001
        Tag:             DW_TAG_compile_unit
        Children:        DW_CHILDREN_yes
        Attributes:
          - Attribute:       DW_AT_low_pc
            Form:            DW_FORM_addr
          - Attribute:       DW_AT_high_pc
            Form:            DW_FORM_addr
          - Attribute:       DW_AT_name
            Form:            DW_FORM_strp
      - Code:            0x00000002
        Tag:             DW_TAG_subprogram
        Children:        DW_CHILDREN_no
        Attributes:
          - Attribute:       DW_AT_low_pc
            Form:            DW_FORM_addr
          - Attribute:       DW_AT_high_pc
            Form:            DW_FORM_addr
    debug_info:
      - Length:          46
        Version:         4
        AbbrOffset:      0
        AddrSize:        8
        Entries:
          - AbbrCode:        0x00000001
            Values:
              - Value:           0x0000000000001000
              - Value:           0x0000000000001500
              - Value:           0x0000000000000001
          - AbbrCode:        0x00000002
            Values:
              - Value:           0x0000000000001000
              - Value:           0x0000000000002000
          - AbbrCode:        0x00000000
            Values:
  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(yamldata);
  ASSERT_TRUE((bool)ErrOrSections);
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);
  VerifyError(*DwarfContext, "error: DIE address ranges are not "
                             "contained in its parent's ranges:");
}

TEST(DWARFDebugInfo, TestDwarfVerifyLexicalBlockRanges) {
  // Create a single compile unit with a single function that has a lexical
  // block whose address range is not contained in the function address range.
  StringRef yamldata = R"(
    debug_str:
      - ''
      - /tmp/main.c
      - main
    debug_abbrev:
      - Code:            0x00000001
        Tag:             DW_TAG_compile_unit
        Children:        DW_CHILDREN_yes
        Attributes:
          - Attribute:       DW_AT_name
            Form:            DW_FORM_strp
      - Code:            0x00000002
        Tag:             DW_TAG_subprogram
        Children:        DW_CHILDREN_yes
        Attributes:
          - Attribute:       DW_AT_name
            Form:            DW_FORM_strp
          - Attribute:       DW_AT_low_pc
            Form:            DW_FORM_addr
          - Attribute:       DW_AT_high_pc
            Form:            DW_FORM_addr
      - Code:            0x00000003
        Tag:             DW_TAG_lexical_block
        Children:        DW_CHILDREN_no
        Attributes:
          - Attribute:       DW_AT_low_pc
            Form:            DW_FORM_addr
          - Attribute:       DW_AT_high_pc
            Form:            DW_FORM_addr
    debug_info:
      - Length:          52
        Version:         4
        AbbrOffset:      0
        AddrSize:        8
        Entries:
          - AbbrCode:        0x00000001
            Values:
              - Value:           0x0000000000000001
          - AbbrCode:        0x00000002
            Values:
              - Value:           0x000000000000000D
              - Value:           0x0000000000001000
              - Value:           0x0000000000002000
          - AbbrCode:        0x00000003
            Values:
              - Value:           0x0000000000001000
              - Value:           0x0000000000002001
          - AbbrCode:        0x00000000
            Values:
          - AbbrCode:        0x00000000
            Values:
  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(yamldata);
  ASSERT_TRUE((bool)ErrOrSections);
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);
  VerifyError(*DwarfContext, "error: DIE address ranges are not "
                             "contained in its parent's ranges:");
}

TEST(DWARFDebugInfo, TestDwarfVerifyOverlappingFunctionRanges) {
  // Create a single compile unit with a two functions that have overlapping
  // address ranges.
  StringRef yamldata = R"(
    debug_str:
      - ''
      - /tmp/main.c
      - main
      - foo
    debug_abbrev:
      - Code:            0x00000001
        Tag:             DW_TAG_compile_unit
        Children:        DW_CHILDREN_yes
        Attributes:
          - Attribute:       DW_AT_name
            Form:            DW_FORM_strp
      - Code:            0x00000002
        Tag:             DW_TAG_subprogram
        Children:        DW_CHILDREN_no
        Attributes:
          - Attribute:       DW_AT_name
            Form:            DW_FORM_strp
          - Attribute:       DW_AT_low_pc
            Form:            DW_FORM_addr
          - Attribute:       DW_AT_high_pc
            Form:            DW_FORM_addr
    debug_info:
      - Length:          55
        Version:         4
        AbbrOffset:      0
        AddrSize:        8
        Entries:
          - AbbrCode:        0x00000001
            Values:
              - Value:           0x0000000000000001
          - AbbrCode:        0x00000002
            Values:
              - Value:           0x000000000000000D
              - Value:           0x0000000000001000
              - Value:           0x0000000000002000
          - AbbrCode:        0x00000002
            Values:
              - Value:           0x0000000000000012
              - Value:           0x0000000000001FFF
              - Value:           0x0000000000002000
          - AbbrCode:        0x00000000
            Values:
  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(yamldata);
  ASSERT_TRUE((bool)ErrOrSections);
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);
  VerifyError(*DwarfContext, "error: DIEs have overlapping address ranges:");
}

TEST(DWARFDebugInfo, TestDwarfVerifyOverlappingLexicalBlockRanges) {
  // Create a single compile unit with a one function that has two lexical
  // blocks with overlapping address ranges.
  StringRef yamldata = R"(
    debug_str:
      - ''
      - /tmp/main.c
      - main
    debug_abbrev:
      - Code:            0x00000001
        Tag:             DW_TAG_compile_unit
        Children:        DW_CHILDREN_yes
        Attributes:
          - Attribute:       DW_AT_low_pc
            Form:            DW_FORM_addr
          - Attribute:       DW_AT_high_pc
            Form:            DW_FORM_addr
          - Attribute:       DW_AT_name
            Form:            DW_FORM_strp
      - Code:            0x00000002
        Tag:             DW_TAG_subprogram
        Children:        DW_CHILDREN_yes
        Attributes:
          - Attribute:       DW_AT_name
            Form:            DW_FORM_strp
          - Attribute:       DW_AT_low_pc
            Form:            DW_FORM_addr
          - Attribute:       DW_AT_high_pc
            Form:            DW_FORM_addr
      - Code:            0x00000003
        Tag:             DW_TAG_lexical_block
        Children:        DW_CHILDREN_no
        Attributes:
          - Attribute:       DW_AT_low_pc
            Form:            DW_FORM_addr
          - Attribute:       DW_AT_high_pc
            Form:            DW_FORM_addr
    debug_info:
      - Length:          85
        Version:         4
        AbbrOffset:      0
        AddrSize:        8
        Entries:
          - AbbrCode:        0x00000001
            Values:
              - Value:           0x0000000000001000
              - Value:           0x0000000000002000
              - Value:           0x0000000000000001
          - AbbrCode:        0x00000002
            Values:
              - Value:           0x000000000000000D
              - Value:           0x0000000000001000
              - Value:           0x0000000000002000
          - AbbrCode:        0x00000003
            Values:
              - Value:           0x0000000000001100
              - Value:           0x0000000000001300
          - AbbrCode:        0x00000003
            Values:
              - Value:           0x00000000000012FF
              - Value:           0x0000000000001300
          - AbbrCode:        0x00000000
            Values:
          - AbbrCode:        0x00000000
            Values:
  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(yamldata);
  ASSERT_TRUE((bool)ErrOrSections);
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);
  VerifyError(*DwarfContext, "error: DIEs have overlapping address ranges:");
}

TEST(DWARFDebugInfo, TestDwarfVerifyInvalidDIERange) {
  // Create a single compile unit with a single function that has an invalid
  // address range where the high PC is smaller than the low PC.
  StringRef yamldata = R"(
    debug_str:
      - ''
      - /tmp/main.c
      - main
    debug_abbrev:
      - Code:            0x00000001
        Tag:             DW_TAG_compile_unit
        Children:        DW_CHILDREN_yes
        Attributes:
          - Attribute:       DW_AT_name
            Form:            DW_FORM_strp
      - Code:            0x00000002
        Tag:             DW_TAG_subprogram
        Children:        DW_CHILDREN_no
        Attributes:
          - Attribute:       DW_AT_name
            Form:            DW_FORM_strp
          - Attribute:       DW_AT_low_pc
            Form:            DW_FORM_addr
          - Attribute:       DW_AT_high_pc
            Form:            DW_FORM_addr
    debug_info:
      - Length:          34
        Version:         4
        AbbrOffset:      0
        AddrSize:        8
        Entries:
          - AbbrCode:        0x00000001
            Values:
              - Value:           0x0000000000000001
          - AbbrCode:        0x00000002
            Values:
              - Value:           0x000000000000000D
              - Value:           0x0000000000001000
              - Value:           0x0000000000000900
          - AbbrCode:        0x00000000
            Values:
  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(yamldata);
  ASSERT_TRUE((bool)ErrOrSections);
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);
  VerifyError(*DwarfContext, "error: Invalid address range");
}

TEST(DWARFDebugInfo, TestDwarfVerifyElidedDoesntFail) {
  // Create a single compile unit with two functions: one that has a valid range
  // and one whose low and high PC are the same. When the low and high PC are
  // the same, this indicates the function was dead code stripped. We want to
  // ensure that verification succeeds.
  StringRef yamldata = R"(
    debug_str:
      - ''
      - /tmp/main.c
      - main
      - elided
    debug_abbrev:
      - Code:            0x00000001
        Tag:             DW_TAG_compile_unit
        Children:        DW_CHILDREN_yes
        Attributes:
          - Attribute:       DW_AT_low_pc
            Form:            DW_FORM_addr
          - Attribute:       DW_AT_high_pc
            Form:            DW_FORM_addr
          - Attribute:       DW_AT_name
            Form:            DW_FORM_strp
      - Code:            0x00000002
        Tag:             DW_TAG_subprogram
        Children:        DW_CHILDREN_no
        Attributes:
          - Attribute:       DW_AT_name
            Form:            DW_FORM_strp
          - Attribute:       DW_AT_low_pc
            Form:            DW_FORM_addr
          - Attribute:       DW_AT_high_pc
            Form:            DW_FORM_addr
    debug_info:
      - Length:          71
        Version:         4
        AbbrOffset:      0
        AddrSize:        8
        Entries:
          - AbbrCode:        0x00000001
            Values:
              - Value:           0x0000000000001000
              - Value:           0x0000000000002000
              - Value:           0x0000000000000001
          - AbbrCode:        0x00000002
            Values:
              - Value:           0x000000000000000D
              - Value:           0x0000000000001000
              - Value:           0x0000000000002000
          - AbbrCode:        0x00000002
            Values:
              - Value:           0x0000000000000012
              - Value:           0x0000000000002000
              - Value:           0x0000000000002000
          - AbbrCode:        0x00000000
            Values:
  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(yamldata);
  ASSERT_TRUE((bool)ErrOrSections);
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);
  VerifySuccess(*DwarfContext);
}

TEST(DWARFDebugInfo, TestDwarfVerifyNestedFunctions) {
  // Create a single compile unit with a nested function which is not contained
  // in its parent. Although LLVM doesn't generate this, it is valid accoridng
  // to the DWARF standard.
  StringRef yamldata = R"(
    debug_str:
      - ''
      - /tmp/main.c
      - main
      - nested
    debug_abbrev:
      - Code:            0x00000001
        Tag:             DW_TAG_compile_unit
        Children:        DW_CHILDREN_yes
        Attributes:
          - Attribute:       DW_AT_low_pc
            Form:            DW_FORM_addr
          - Attribute:       DW_AT_high_pc
            Form:            DW_FORM_addr
          - Attribute:       DW_AT_name
            Form:            DW_FORM_strp
      - Code:            0x00000002
        Tag:             DW_TAG_subprogram
        Children:        DW_CHILDREN_yes
        Attributes:
          - Attribute:       DW_AT_name
            Form:            DW_FORM_strp
          - Attribute:       DW_AT_low_pc
            Form:            DW_FORM_addr
          - Attribute:       DW_AT_high_pc
            Form:            DW_FORM_addr
    debug_info:
      - Length:          73
        Version:         4
        AbbrOffset:      0
        AddrSize:        8
        Entries:
          - AbbrCode:        0x00000001
            Values:
              - Value:           0x0000000000001000
              - Value:           0x0000000000002000
              - Value:           0x0000000000000001
          - AbbrCode:        0x00000002
            Values:
              - Value:           0x000000000000000D
              - Value:           0x0000000000001000
              - Value:           0x0000000000001500
          - AbbrCode:        0x00000002
            Values:
              - Value:           0x0000000000000012
              - Value:           0x0000000000001500
              - Value:           0x0000000000002000
          - AbbrCode:        0x00000000
            Values:
          - AbbrCode:        0x00000000
            Values:
          - AbbrCode:        0x00000000
            Values:
  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(yamldata);
  ASSERT_TRUE((bool)ErrOrSections);
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);
  VerifySuccess(*DwarfContext);
}

TEST(DWARFDebugInfo, TestDWARFDieRangeInfoContains) {
  DWARFVerifier::DieRangeInfo Empty;
  ASSERT_TRUE(Empty.contains(Empty));

  DWARFVerifier::DieRangeInfo Ranges(
      {{0x10, 0x20}, {0x30, 0x40}, {0x40, 0x50}});

  ASSERT_TRUE(Ranges.contains(Empty));
  ASSERT_FALSE(Ranges.contains({{{0x0f, 0x10}}}));
  ASSERT_FALSE(Ranges.contains({{{0x0f, 0x20}}}));
  ASSERT_FALSE(Ranges.contains({{{0x0f, 0x21}}}));

  // Test ranges that start at R's start address
  ASSERT_TRUE(Ranges.contains({{{0x10, 0x10}}}));
  ASSERT_TRUE(Ranges.contains({{{0x10, 0x11}}}));
  ASSERT_TRUE(Ranges.contains({{{0x10, 0x20}}}));
  ASSERT_FALSE(Ranges.contains({{{0x10, 0x21}}}));

  ASSERT_TRUE(Ranges.contains({{{0x11, 0x12}}}));

  // Test ranges that start at last bytes of Range
  ASSERT_TRUE(Ranges.contains({{{0x1f, 0x20}}}));
  ASSERT_FALSE(Ranges.contains({{{0x1f, 0x21}}}));

  // Test ranges that start after Range
  ASSERT_TRUE(Ranges.contains({{{0x20, 0x20}}}));
  ASSERT_FALSE(Ranges.contains({{{0x20, 0x21}}}));

  ASSERT_TRUE(Ranges.contains({{{0x31, 0x32}}}));
  ASSERT_TRUE(Ranges.contains({{{0x3f, 0x40}}}));
  ASSERT_TRUE(Ranges.contains({{{0x10, 0x20}, {0x30, 0x40}}}));
  ASSERT_TRUE(Ranges.contains({{{0x11, 0x12}, {0x31, 0x32}}}));
  ASSERT_TRUE(Ranges.contains(
      {{{0x11, 0x12}, {0x12, 0x13}, {0x31, 0x32}, {0x32, 0x33}}}));
  ASSERT_FALSE(Ranges.contains({{{0x11, 0x12},
                                 {0x12, 0x13},
                                 {0x20, 0x21},
                                 {0x31, 0x32},
                                 {0x32, 0x33}}}));
  ASSERT_FALSE(Ranges.contains(
      {{{0x11, 0x12}, {0x12, 0x13}, {0x31, 0x32}, {0x32, 0x51}}}));
  ASSERT_TRUE(Ranges.contains({{{0x11, 0x12}, {0x30, 0x50}}}));
  ASSERT_FALSE(Ranges.contains({{{0x30, 0x51}}}));
  ASSERT_FALSE(Ranges.contains({{{0x50, 0x51}}}));
}

namespace {

void AssertRangesIntersect(const DWARFAddressRange &LHS,
                           const DWARFAddressRange &RHS) {
  ASSERT_TRUE(LHS.intersects(RHS));
  ASSERT_TRUE(RHS.intersects(LHS));
}
void AssertRangesDontIntersect(const DWARFAddressRange &LHS,
                               const DWARFAddressRange &RHS) {
  ASSERT_FALSE(LHS.intersects(RHS));
  ASSERT_FALSE(RHS.intersects(LHS));
}

void AssertRangesIntersect(const DWARFVerifier::DieRangeInfo &LHS,
                           const DWARFAddressRangesVector &Ranges) {
  DWARFVerifier::DieRangeInfo RHS(Ranges);
  ASSERT_TRUE(LHS.intersects(RHS));
  ASSERT_TRUE(RHS.intersects(LHS));
}

void AssertRangesDontIntersect(const DWARFVerifier::DieRangeInfo &LHS,
                               const DWARFAddressRangesVector &Ranges) {
  DWARFVerifier::DieRangeInfo RHS(Ranges);
  ASSERT_FALSE(LHS.intersects(RHS));
  ASSERT_FALSE(RHS.intersects(LHS));
}

} // namespace
TEST(DWARFDebugInfo, TestDwarfRangesIntersect) {
  DWARFAddressRange R(0x10, 0x20);

  //----------------------------------------------------------------------
  // Test ranges that start before R...
  //----------------------------------------------------------------------
  // Other range ends before start of R
  AssertRangesDontIntersect(R, {0x00, 0x10});
  // Other range end address is start of a R
  AssertRangesIntersect(R, {0x00, 0x11});
  // Other range end address is in R
  AssertRangesIntersect(R, {0x00, 0x15});
  // Other range end address is at and of R
  AssertRangesIntersect(R, {0x00, 0x20});
  // Other range end address is past end of R
  AssertRangesIntersect(R, {0x00, 0x40});

  //----------------------------------------------------------------------
  // Test ranges that start at R's start address
  //----------------------------------------------------------------------
  // Ensure empty ranges doesn't match
  AssertRangesDontIntersect(R, {0x10, 0x10});
  // 1 byte of Range
  AssertRangesIntersect(R, {0x10, 0x11});
  // same as Range
  AssertRangesIntersect(R, {0x10, 0x20});
  // 1 byte past Range
  AssertRangesIntersect(R, {0x10, 0x21});

  //----------------------------------------------------------------------
  // Test ranges that start inside Range
  //----------------------------------------------------------------------
  // empty in range
  AssertRangesDontIntersect(R, {0x11, 0x11});
  // all in Range
  AssertRangesIntersect(R, {0x11, 0x1f});
  // ends at end of Range
  AssertRangesIntersect(R, {0x11, 0x20});
  // ends past Range
  AssertRangesIntersect(R, {0x11, 0x21});

  //----------------------------------------------------------------------
  // Test ranges that start at last bytes of Range
  //----------------------------------------------------------------------
  // ends at end of Range
  AssertRangesIntersect(R, {0x1f, 0x20});
  // ends past Range
  AssertRangesIntersect(R, {0x1f, 0x21});

  //----------------------------------------------------------------------
  // Test ranges that start after Range
  //----------------------------------------------------------------------
  // empty just past in Range
  AssertRangesDontIntersect(R, {0x20, 0x20});
  // valid past Range
  AssertRangesDontIntersect(R, {0x20, 0x21});
}

TEST(DWARFDebugInfo, TestDWARFDieRangeInfoIntersects) {

  DWARFVerifier::DieRangeInfo Ranges({{0x10, 0x20}, {0x30, 0x40}});

  // Test empty range
  AssertRangesDontIntersect(Ranges, {});
  // Test range that appears before all ranges in Ranges
  AssertRangesDontIntersect(Ranges, {{0x00, 0x10}});
  // Test range that appears between ranges in Ranges
  AssertRangesDontIntersect(Ranges, {{0x20, 0x30}});
  // Test range that appears after ranges in Ranges
  AssertRangesDontIntersect(Ranges, {{0x40, 0x50}});

  // Test range that start before first range
  AssertRangesIntersect(Ranges, {{0x00, 0x11}});
  // Test range that start at first range
  AssertRangesIntersect(Ranges, {{0x10, 0x11}});
  // Test range that start in first range
  AssertRangesIntersect(Ranges, {{0x11, 0x12}});
  // Test range that start at end of first range
  AssertRangesIntersect(Ranges, {{0x1f, 0x20}});
  // Test range that starts at end of first range
  AssertRangesDontIntersect(Ranges, {{0x20, 0x21}});
  // Test range that starts at end of first range
  AssertRangesIntersect(Ranges, {{0x20, 0x31}});

  // Test range that start before second range and ends before second
  AssertRangesDontIntersect(Ranges, {{0x2f, 0x30}});
  // Test range that start before second range and ends in second
  AssertRangesIntersect(Ranges, {{0x2f, 0x31}});
  // Test range that start at second range
  AssertRangesIntersect(Ranges, {{0x30, 0x31}});
  // Test range that start in second range
  AssertRangesIntersect(Ranges, {{0x31, 0x32}});
  // Test range that start at end of second range
  AssertRangesIntersect(Ranges, {{0x3f, 0x40}});
  // Test range that starts at end of second range
  AssertRangesDontIntersect(Ranges, {{0x40, 0x41}});

  AssertRangesDontIntersect(Ranges, {{0x20, 0x21}, {0x2f, 0x30}});
  AssertRangesIntersect(Ranges, {{0x20, 0x21}, {0x2f, 0x31}});
}

TEST(DWARFDebugInfo, TestDWARF64UnitLength) {
  static const char DebugInfoSecRaw[] =
      "\xff\xff\xff\xff"                 // DWARF64 mark
      "\x88\x77\x66\x55\x44\x33\x22\x11" // Length
      "\x05\x00"                         // Version
      "\x01"                             // DW_UT_compile
      "\x04"                             // Address size
      "\0\0\0\0\0\0\0\0";                // Offset Into Abbrev. Sec.
  StringMap<std::unique_ptr<MemoryBuffer>> Sections;
  Sections.insert(std::make_pair(
      "debug_info", MemoryBuffer::getMemBuffer(StringRef(
                        DebugInfoSecRaw, sizeof(DebugInfoSecRaw) - 1))));
  auto Context = DWARFContext::create(Sections, /* AddrSize = */ 4,
                                      /* isLittleEndian = */ true);
  const auto &Obj = Context->getDWARFObj();
  Obj.forEachInfoSections([&](const DWARFSection &Sec) {
    DWARFUnitHeader Header;
    DWARFDataExtractor Data(Obj, Sec, /* IsLittleEndian = */ true,
                            /* AddressSize = */ 4);
    uint64_t Offset = 0;
    EXPECT_FALSE(Header.extract(*Context, Data, &Offset, DW_SECT_INFO));
    // Header.extract() returns false because there is not enough space
    // in the section for the declared length. Anyway, we can check that
    // the properties are read correctly.
    ASSERT_EQ(DwarfFormat::DWARF64, Header.getFormat());
    ASSERT_EQ(0x1122334455667788ULL, Header.getLength());
    ASSERT_EQ(5, Header.getVersion());
    ASSERT_EQ(DW_UT_compile, Header.getUnitType());
    ASSERT_EQ(4, Header.getAddressByteSize());

    // Check that the length can be correctly read in the unit class.
    DWARFUnitVector DummyUnitVector;
    DWARFSection DummySec;
    DWARFCompileUnit CU(*Context, Sec, Header, /* DA = */ 0, /* RS = */ 0,
                        /* LocSection = */ 0, /* SS = */ StringRef(),
                        /* SOS = */ DummySec, /* AOS = */ 0,
                        /* LS = */ DummySec, /* LE = */ true,
                        /* isDWO= */ false, DummyUnitVector);
    ASSERT_EQ(0x1122334455667788ULL, CU.getLength());
  });
}

} // end anonymous namespace
