//===- llvm/unittest/DebugInfo/DWARFFormValueTest.cpp ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../lib/CodeGen/DwarfGenerator.h"
#include "llvm/DebugInfo/DWARF/DWARFAbbreviationDeclaration.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugInfoEntry.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetSelect.h"
#include "gtest/gtest.h"
#include <climits>

using namespace llvm;
using namespace dwarf;

namespace {

void initLLVMIfNeeded() {
  static bool gInitialized = false;
  if (!gInitialized) {
    gInitialized = true;
    InitializeAllTargets();
    InitializeAllTargetMCs();
    InitializeAllAsmPrinters();
    InitializeAllAsmParsers();
  }
}

Triple getHostTripleForAddrSize(uint8_t AddrSize) {
  Triple PT(Triple::normalize(LLVM_HOST_TRIPLE));

  if (AddrSize == 8 && PT.isArch32Bit())
    return PT.get64BitArchVariant();
  if (AddrSize == 4 && PT.isArch64Bit())
    return PT.get32BitArchVariant();
  return PT;
}

/// Take any llvm::Expected and check and handle any errors.
///
/// \param Expected a llvm::Excepted instance to check.
/// \returns true if there were errors, false otherwise.
template <typename T>
static bool HandleExpectedError(T &Expected) {
  std::string ErrorMsg;
  handleAllErrors(Expected.takeError(), [&](const llvm::ErrorInfoBase &EI) {
    ErrorMsg = EI.message();
  });
  if (!ErrorMsg.empty()) {
    ::testing::AssertionFailure() << "error: " << ErrorMsg;
    return true;
  }
  return false;
}

template <uint16_t Version, class AddrType, class RefAddrType>
void TestAllForms() {
  // Test that we can decode all DW_FORM values correctly.

  const uint8_t AddrSize = sizeof(AddrType);
  const AddrType AddrValue = (AddrType)0x0123456789abcdefULL;
  const uint8_t BlockData[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
  const uint32_t BlockSize = sizeof(BlockData);
  const RefAddrType RefAddr = 0x12345678;
  const uint8_t Data1 = 0x01U;
  const uint16_t Data2 = 0x2345U;
  const uint32_t Data4 = 0x6789abcdU;
  const uint64_t Data8 = 0x0011223344556677ULL;
  const uint64_t Data8_2 = 0xAABBCCDDEEFF0011ULL;
  const int64_t SData = INT64_MIN;
  const uint64_t UData[] = {UINT64_MAX - 1, UINT64_MAX - 2, UINT64_MAX - 3,
                            UINT64_MAX - 4, UINT64_MAX - 5, UINT64_MAX - 6,
                            UINT64_MAX - 7, UINT64_MAX - 8, UINT64_MAX - 9};
#define UDATA_1 18446744073709551614ULL
  const uint32_t Dwarf32Values[] = {1, 2, 3, 4, 5, 6, 7, 8};
  const char *StringValue = "Hello";
  const char *StrpValue = "World";
  initLLVMIfNeeded();
  Triple Triple = getHostTripleForAddrSize(AddrSize);
  auto ExpectedDG = dwarfgen::Generator::create(Triple, Version);
  if (HandleExpectedError(ExpectedDG))
    return;
  dwarfgen::Generator *DG = ExpectedDG.get().get();
  dwarfgen::CompileUnit &CU = DG->addCompileUnit();
  dwarfgen::DIE CUDie = CU.getUnitDIE();
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
  CUDie.addAttribute(Attr_DW_FORM_flag_present, DW_FORM_flag_present);

  //----------------------------------------------------------------------
  // Test SLEB128 based forms
  //----------------------------------------------------------------------
  const auto Attr_DW_FORM_sdata = static_cast<dwarf::Attribute>(Attr++);
  CUDie.addAttribute(Attr_DW_FORM_sdata, DW_FORM_sdata, SData);

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
  DWARFContextInMemory DwarfContext(*Obj.get());
  uint32_t NumCUs = DwarfContext.getNumCompileUnits();
  EXPECT_EQ(NumCUs, 1u);
  DWARFCompileUnit *U = DwarfContext.getCompileUnitAtIndex(0);
  auto DiePtr = U->getUnitDIE(false);
  EXPECT_TRUE(DiePtr != nullptr);

  //----------------------------------------------------------------------
  // Test address forms
  //----------------------------------------------------------------------
  EXPECT_EQ(DiePtr->getAttributeValueAsAddress(U, Attr_DW_FORM_addr, 0),
            AddrValue);

  //----------------------------------------------------------------------
  // Test block forms
  //----------------------------------------------------------------------
  DWARFFormValue FormValue;
  ArrayRef<uint8_t> ExtractedBlockData;
  Optional<ArrayRef<uint8_t>> BlockDataOpt;

  EXPECT_TRUE(DiePtr->getAttributeValue(U, Attr_DW_FORM_block, FormValue));
  BlockDataOpt = FormValue.getAsBlock();
  EXPECT_TRUE(BlockDataOpt.hasValue());
  ExtractedBlockData = BlockDataOpt.getValue();
  EXPECT_EQ(ExtractedBlockData.size(), BlockSize);
  EXPECT_TRUE(memcmp(ExtractedBlockData.data(), BlockData, BlockSize) == 0);

  EXPECT_TRUE(DiePtr->getAttributeValue(U, Attr_DW_FORM_block1, FormValue));
  BlockDataOpt = FormValue.getAsBlock();
  EXPECT_TRUE(BlockDataOpt.hasValue());
  ExtractedBlockData = BlockDataOpt.getValue();
  EXPECT_EQ(ExtractedBlockData.size(), BlockSize);
  EXPECT_TRUE(memcmp(ExtractedBlockData.data(), BlockData, BlockSize) == 0);

  EXPECT_TRUE(DiePtr->getAttributeValue(U, Attr_DW_FORM_block2, FormValue));
  BlockDataOpt = FormValue.getAsBlock();
  EXPECT_TRUE(BlockDataOpt.hasValue());
  ExtractedBlockData = BlockDataOpt.getValue();
  EXPECT_EQ(ExtractedBlockData.size(), BlockSize);
  EXPECT_TRUE(memcmp(ExtractedBlockData.data(), BlockData, BlockSize) == 0);

  EXPECT_TRUE(DiePtr->getAttributeValue(U, Attr_DW_FORM_block4, FormValue));
  BlockDataOpt = FormValue.getAsBlock();
  EXPECT_TRUE(BlockDataOpt.hasValue());
  ExtractedBlockData = BlockDataOpt.getValue();
  EXPECT_EQ(ExtractedBlockData.size(), BlockSize);
  EXPECT_TRUE(memcmp(ExtractedBlockData.data(), BlockData, BlockSize) == 0);

  //----------------------------------------------------------------------
  // Test data forms
  //----------------------------------------------------------------------
  EXPECT_EQ(
      DiePtr->getAttributeValueAsUnsignedConstant(U, Attr_DW_FORM_data1, 0),
      Data1);
  EXPECT_EQ(
      DiePtr->getAttributeValueAsUnsignedConstant(U, Attr_DW_FORM_data2, 0),
      Data2);
  EXPECT_EQ(
      DiePtr->getAttributeValueAsUnsignedConstant(U, Attr_DW_FORM_data4, 0),
      Data4);
  EXPECT_EQ(
      DiePtr->getAttributeValueAsUnsignedConstant(U, Attr_DW_FORM_data8, 0),
      Data8);

  //----------------------------------------------------------------------
  // Test string forms
  //----------------------------------------------------------------------
  const char *ExtractedStringValue =
      DiePtr->getAttributeValueAsString(U, Attr_DW_FORM_string, nullptr);
  EXPECT_TRUE(ExtractedStringValue != nullptr);
  EXPECT_TRUE(strcmp(StringValue, ExtractedStringValue) == 0);

  const char *ExtractedStrpValue =
      DiePtr->getAttributeValueAsString(U, Attr_DW_FORM_strp, nullptr);
  EXPECT_TRUE(ExtractedStrpValue != nullptr);
  EXPECT_TRUE(strcmp(StrpValue, ExtractedStrpValue) == 0);

  //----------------------------------------------------------------------
  // Test reference forms
  //----------------------------------------------------------------------
  EXPECT_EQ(DiePtr->getAttributeValueAsReference(U, Attr_DW_FORM_ref_addr, 0),
            RefAddr);
  EXPECT_EQ(DiePtr->getAttributeValueAsReference(U, Attr_DW_FORM_ref1, 0),
            Data1);
  EXPECT_EQ(DiePtr->getAttributeValueAsReference(U, Attr_DW_FORM_ref2, 0),
            Data2);
  EXPECT_EQ(DiePtr->getAttributeValueAsReference(U, Attr_DW_FORM_ref4, 0),
            Data4);
  EXPECT_EQ(DiePtr->getAttributeValueAsReference(U, Attr_DW_FORM_ref8, 0),
            Data8);
  EXPECT_EQ(DiePtr->getAttributeValueAsReference(U, Attr_DW_FORM_ref_sig8, 0),
            Data8_2);
  EXPECT_EQ(DiePtr->getAttributeValueAsReference(U, Attr_DW_FORM_ref_udata, 0),
            UData[0]);

  //----------------------------------------------------------------------
  // Test flag forms
  //----------------------------------------------------------------------
  EXPECT_EQ(DiePtr->getAttributeValueAsUnsignedConstant(
                U, Attr_DW_FORM_flag_true, 0ULL),
            1ULL);
  EXPECT_EQ(DiePtr->getAttributeValueAsUnsignedConstant(
                U, Attr_DW_FORM_flag_false, 1ULL),
            0ULL);
  EXPECT_EQ(DiePtr->getAttributeValueAsUnsignedConstant(
                U, Attr_DW_FORM_flag_present, 0ULL),
            1ULL);

  // TODO: test Attr_DW_FORM_implicit_const extraction

  //----------------------------------------------------------------------
  // Test SLEB128 based forms
  //----------------------------------------------------------------------
  EXPECT_EQ(DiePtr->getAttributeValueAsSignedConstant(U, Attr_DW_FORM_sdata, 0),
            SData);

  //----------------------------------------------------------------------
  // Test ULEB128 based forms
  //----------------------------------------------------------------------
  EXPECT_EQ(
      DiePtr->getAttributeValueAsUnsignedConstant(U, Attr_DW_FORM_udata, 0),
      UData[0]);

  //----------------------------------------------------------------------
  // Test DWARF32/DWARF64 forms
  //----------------------------------------------------------------------
  EXPECT_EQ(
      DiePtr->getAttributeValueAsReference(U, Attr_DW_FORM_GNU_ref_alt, 0),
      Dwarf32Values[0]);
  EXPECT_EQ(
      DiePtr->getAttributeValueAsSectionOffset(U, Attr_DW_FORM_sec_offset, 0),
      Dwarf32Values[1]);

  //----------------------------------------------------------------------
  // Add an address at the end to make sure we can decode this value
  //----------------------------------------------------------------------
  EXPECT_EQ(DiePtr->getAttributeValueAsAddress(U, Attr_Last, 0), AddrValue);
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

template <uint16_t Version, class AddrType> void TestChildren() {
  // Test that we can decode DW_FORM_ref_addr values correctly in DWARF 2 with
  // 4 byte addresses. DW_FORM_ref_addr values should be 4 bytes when using
  // 8 byte addresses.

  const uint8_t AddrSize = sizeof(AddrType);
  initLLVMIfNeeded();
  Triple Triple = getHostTripleForAddrSize(AddrSize);
  auto ExpectedDG = dwarfgen::Generator::create(Triple, Version);
  if (HandleExpectedError(ExpectedDG))
    return;
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
  DWARFContextInMemory DwarfContext(*Obj.get());

  // Verify the number of compile units is correct.
  uint32_t NumCUs = DwarfContext.getNumCompileUnits();
  EXPECT_EQ(NumCUs, 1u);
  DWARFCompileUnit *U = DwarfContext.getCompileUnitAtIndex(0);

  // Get the compile unit DIE is valid.
  auto DiePtr = U->getUnitDIE(false);
  EXPECT_TRUE(DiePtr != nullptr);
  // DiePtr->dump(llvm::outs(), U, UINT32_MAX);

  // Verify the first child of the compile unit DIE is our subprogram.
  auto SubprogramDiePtr = DiePtr->getFirstChild();
  EXPECT_TRUE(SubprogramDiePtr != nullptr);
  EXPECT_EQ(SubprogramDiePtr->getTag(), DW_TAG_subprogram);

  // Verify the first child of the subprogram is our formal parameter.
  auto ArgcDiePtr = SubprogramDiePtr->getFirstChild();
  EXPECT_TRUE(ArgcDiePtr != nullptr);
  EXPECT_EQ(ArgcDiePtr->getTag(), DW_TAG_formal_parameter);

  // Verify our formal parameter has a NULL tag sibling.
  auto NullDiePtr = ArgcDiePtr->getSibling();
  EXPECT_TRUE(NullDiePtr != nullptr);
  if (NullDiePtr) {
    EXPECT_EQ(NullDiePtr->getTag(), DW_TAG_null);
    EXPECT_TRUE(NullDiePtr->getSibling() == nullptr);
    EXPECT_TRUE(NullDiePtr->getFirstChild() == nullptr);
  }

  // Verify the sibling of our subprogram is our integer base type.
  auto IntDiePtr = SubprogramDiePtr->getSibling();
  EXPECT_TRUE(IntDiePtr != nullptr);
  EXPECT_EQ(IntDiePtr->getTag(), DW_TAG_base_type);

  // Verify the sibling of our subprogram is our integer base is a NULL tag.
  NullDiePtr = IntDiePtr->getSibling();
  EXPECT_TRUE(NullDiePtr != nullptr);
  if (NullDiePtr) {
    EXPECT_EQ(NullDiePtr->getTag(), DW_TAG_null);
    EXPECT_TRUE(NullDiePtr->getSibling() == nullptr);
    EXPECT_TRUE(NullDiePtr->getFirstChild() == nullptr);
  }
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
  // Test that we can decode DW_FORM_refXXX values correctly in DWARF.

  const uint8_t AddrSize = sizeof(AddrType);
  initLLVMIfNeeded();
  Triple Triple = getHostTripleForAddrSize(AddrSize);
  auto ExpectedDG = dwarfgen::Generator::create(Triple, Version);
  if (HandleExpectedError(ExpectedDG))
    return;
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
  DWARFContextInMemory DwarfContext(*Obj.get());

  // Verify the number of compile units is correct.
  uint32_t NumCUs = DwarfContext.getNumCompileUnits();
  EXPECT_EQ(NumCUs, 2u);
  DWARFCompileUnit *U1 = DwarfContext.getCompileUnitAtIndex(0);
  DWARFCompileUnit *U2 = DwarfContext.getCompileUnitAtIndex(1);

  // Get the compile unit DIE is valid.
  auto Unit1DiePtr = U1->getUnitDIE(false);
  EXPECT_TRUE(Unit1DiePtr != nullptr);
  // Unit1DiePtr->dump(llvm::outs(), U1, UINT32_MAX);

  auto Unit2DiePtr = U2->getUnitDIE(false);
  EXPECT_TRUE(Unit2DiePtr != nullptr);
  // Unit2DiePtr->dump(llvm::outs(), U2, UINT32_MAX);

  // Verify the first child of the compile unit 1 DIE is our int base type.
  auto CU1TypeDiePtr = Unit1DiePtr->getFirstChild();
  EXPECT_TRUE(CU1TypeDiePtr != nullptr);
  EXPECT_EQ(CU1TypeDiePtr->getTag(), DW_TAG_base_type);
  EXPECT_EQ(
      CU1TypeDiePtr->getAttributeValueAsUnsignedConstant(U1, DW_AT_encoding, 0),
      DW_ATE_signed);

  // Verify the first child of the compile unit 2 DIE is our float base type.
  auto CU2TypeDiePtr = Unit2DiePtr->getFirstChild();
  EXPECT_TRUE(CU2TypeDiePtr != nullptr);
  EXPECT_EQ(CU2TypeDiePtr->getTag(), DW_TAG_base_type);
  EXPECT_EQ(
      CU2TypeDiePtr->getAttributeValueAsUnsignedConstant(U2, DW_AT_encoding, 0),
      DW_ATE_float);

  // Verify the sibling of the base type DIE is our Ref1 DIE and that its
  // DW_AT_type points to our base type DIE.
  auto CU1Ref1DiePtr = CU1TypeDiePtr->getSibling();
  EXPECT_TRUE(CU1Ref1DiePtr != nullptr);
  EXPECT_EQ(CU1Ref1DiePtr->getTag(), DW_TAG_variable);
  EXPECT_EQ(CU1Ref1DiePtr->getAttributeValueAsReference(U1, DW_AT_type, -1ULL),
            CU1TypeDiePtr->getOffset());
  // Verify the sibling is our Ref2 DIE and that its DW_AT_type points to our
  // base type DIE in CU1.
  auto CU1Ref2DiePtr = CU1Ref1DiePtr->getSibling();
  EXPECT_TRUE(CU1Ref2DiePtr != nullptr);
  EXPECT_EQ(CU1Ref2DiePtr->getTag(), DW_TAG_variable);
  EXPECT_EQ(CU1Ref2DiePtr->getAttributeValueAsReference(U1, DW_AT_type, -1ULL),
            CU1TypeDiePtr->getOffset());

  // Verify the sibling is our Ref4 DIE and that its DW_AT_type points to our
  // base type DIE in CU1.
  auto CU1Ref4DiePtr = CU1Ref2DiePtr->getSibling();
  EXPECT_TRUE(CU1Ref4DiePtr != nullptr);
  EXPECT_EQ(CU1Ref4DiePtr->getTag(), DW_TAG_variable);
  EXPECT_EQ(CU1Ref4DiePtr->getAttributeValueAsReference(U1, DW_AT_type, -1ULL),
            CU1TypeDiePtr->getOffset());

  // Verify the sibling is our Ref8 DIE and that its DW_AT_type points to our
  // base type DIE in CU1.
  auto CU1Ref8DiePtr = CU1Ref4DiePtr->getSibling();
  EXPECT_TRUE(CU1Ref8DiePtr != nullptr);
  EXPECT_EQ(CU1Ref8DiePtr->getTag(), DW_TAG_variable);
  EXPECT_EQ(CU1Ref8DiePtr->getAttributeValueAsReference(U1, DW_AT_type, -1ULL),
            CU1TypeDiePtr->getOffset());

  // Verify the sibling is our RefAddr DIE and that its DW_AT_type points to our
  // base type DIE in CU1.
  auto CU1RefAddrDiePtr = CU1Ref8DiePtr->getSibling();
  EXPECT_TRUE(CU1RefAddrDiePtr != nullptr);
  EXPECT_EQ(CU1RefAddrDiePtr->getTag(), DW_TAG_variable);
  EXPECT_EQ(
      CU1RefAddrDiePtr->getAttributeValueAsReference(U1, DW_AT_type, -1ULL),
      CU1TypeDiePtr->getOffset());

  // Verify the sibling of the Ref4 DIE is our RefAddr DIE and that its
  // DW_AT_type points to our base type DIE.
  auto CU1ToCU2RefAddrDiePtr = CU1RefAddrDiePtr->getSibling();
  EXPECT_TRUE(CU1ToCU2RefAddrDiePtr != nullptr);
  EXPECT_EQ(CU1ToCU2RefAddrDiePtr->getTag(), DW_TAG_variable);
  EXPECT_EQ(CU1ToCU2RefAddrDiePtr->getAttributeValueAsReference(U1, DW_AT_type,
                                                                -1ULL),
            CU2TypeDiePtr->getOffset());

  // Verify the sibling of the base type DIE is our Ref1 DIE and that its
  // DW_AT_type points to our base type DIE.
  auto CU2Ref1DiePtr = CU2TypeDiePtr->getSibling();
  EXPECT_TRUE(CU2Ref1DiePtr != nullptr);
  EXPECT_EQ(CU2Ref1DiePtr->getTag(), DW_TAG_variable);
  EXPECT_EQ(CU2Ref1DiePtr->getAttributeValueAsReference(U2, DW_AT_type, -1ULL),
            CU2TypeDiePtr->getOffset());
  // Verify the sibling is our Ref2 DIE and that its DW_AT_type points to our
  // base type DIE in CU2.
  auto CU2Ref2DiePtr = CU2Ref1DiePtr->getSibling();
  EXPECT_TRUE(CU2Ref2DiePtr != nullptr);
  EXPECT_EQ(CU2Ref2DiePtr->getTag(), DW_TAG_variable);
  EXPECT_EQ(CU2Ref2DiePtr->getAttributeValueAsReference(U2, DW_AT_type, -1ULL),
            CU2TypeDiePtr->getOffset());

  // Verify the sibling is our Ref4 DIE and that its DW_AT_type points to our
  // base type DIE in CU2.
  auto CU2Ref4DiePtr = CU2Ref2DiePtr->getSibling();
  EXPECT_TRUE(CU2Ref4DiePtr != nullptr);
  EXPECT_EQ(CU2Ref4DiePtr->getTag(), DW_TAG_variable);
  EXPECT_EQ(CU2Ref4DiePtr->getAttributeValueAsReference(U2, DW_AT_type, -1ULL),
            CU2TypeDiePtr->getOffset());

  // Verify the sibling is our Ref8 DIE and that its DW_AT_type points to our
  // base type DIE in CU2.
  auto CU2Ref8DiePtr = CU2Ref4DiePtr->getSibling();
  EXPECT_TRUE(CU2Ref8DiePtr != nullptr);
  EXPECT_EQ(CU2Ref8DiePtr->getTag(), DW_TAG_variable);
  EXPECT_EQ(CU2Ref8DiePtr->getAttributeValueAsReference(U2, DW_AT_type, -1ULL),
            CU2TypeDiePtr->getOffset());

  // Verify the sibling is our RefAddr DIE and that its DW_AT_type points to our
  // base type DIE in CU2.
  auto CU2RefAddrDiePtr = CU2Ref8DiePtr->getSibling();
  EXPECT_TRUE(CU2RefAddrDiePtr != nullptr);
  EXPECT_EQ(CU2RefAddrDiePtr->getTag(), DW_TAG_variable);
  EXPECT_EQ(
      CU2RefAddrDiePtr->getAttributeValueAsReference(U2, DW_AT_type, -1ULL),
      CU2TypeDiePtr->getOffset());

  // Verify the sibling of the Ref4 DIE is our RefAddr DIE and that its
  // DW_AT_type points to our base type DIE.
  auto CU2ToCU1RefAddrDiePtr = CU2RefAddrDiePtr->getSibling();
  EXPECT_TRUE(CU2ToCU1RefAddrDiePtr != nullptr);
  EXPECT_EQ(CU2ToCU1RefAddrDiePtr->getTag(), DW_TAG_variable);
  EXPECT_EQ(CU2ToCU1RefAddrDiePtr->getAttributeValueAsReference(U2, DW_AT_type,
                                                                -1ULL),
            CU1TypeDiePtr->getOffset());
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

} // end anonymous namespace
