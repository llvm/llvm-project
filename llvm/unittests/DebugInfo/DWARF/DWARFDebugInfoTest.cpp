//===- llvm/unittest/DebugInfo/DWARFFormValueTest.cpp ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DwarfGenerator.h"
#include "llvm/DebugInfo/DWARF/DWARFAbbreviationDeclaration.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
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
  const int64_t ICSData = INT64_MAX; // DW_FORM_implicit_const SData
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
  auto DieDG = U->getUnitDIE(false);
  EXPECT_TRUE(DieDG.isValid());

  //----------------------------------------------------------------------
  // Test address forms
  //----------------------------------------------------------------------
  EXPECT_EQ(DieDG.getAttributeValueAsAddress(Attr_DW_FORM_addr).getValueOr(0),
            AddrValue);

  //----------------------------------------------------------------------
  // Test block forms
  //----------------------------------------------------------------------
  Optional<DWARFFormValue> FormValue;
  ArrayRef<uint8_t> ExtractedBlockData;
  Optional<ArrayRef<uint8_t>> BlockDataOpt;

  FormValue = DieDG.getAttributeValue(Attr_DW_FORM_block);
  EXPECT_TRUE((bool)FormValue);
  BlockDataOpt = FormValue->getAsBlock();
  EXPECT_TRUE(BlockDataOpt.hasValue());
  ExtractedBlockData = BlockDataOpt.getValue();
  EXPECT_EQ(ExtractedBlockData.size(), BlockSize);
  EXPECT_TRUE(memcmp(ExtractedBlockData.data(), BlockData, BlockSize) == 0);

  FormValue = DieDG.getAttributeValue(Attr_DW_FORM_block1);
  EXPECT_TRUE((bool)FormValue);
  BlockDataOpt = FormValue->getAsBlock();
  EXPECT_TRUE(BlockDataOpt.hasValue());
  ExtractedBlockData = BlockDataOpt.getValue();
  EXPECT_EQ(ExtractedBlockData.size(), BlockSize);
  EXPECT_TRUE(memcmp(ExtractedBlockData.data(), BlockData, BlockSize) == 0);

  FormValue = DieDG.getAttributeValue(Attr_DW_FORM_block2);
  EXPECT_TRUE((bool)FormValue);
  BlockDataOpt = FormValue->getAsBlock();
  EXPECT_TRUE(BlockDataOpt.hasValue());
  ExtractedBlockData = BlockDataOpt.getValue();
  EXPECT_EQ(ExtractedBlockData.size(), BlockSize);
  EXPECT_TRUE(memcmp(ExtractedBlockData.data(), BlockData, BlockSize) == 0);

  FormValue = DieDG.getAttributeValue(Attr_DW_FORM_block4);
  EXPECT_TRUE((bool)FormValue);
  BlockDataOpt = FormValue->getAsBlock();
  EXPECT_TRUE(BlockDataOpt.hasValue());
  ExtractedBlockData = BlockDataOpt.getValue();
  EXPECT_EQ(ExtractedBlockData.size(), BlockSize);
  EXPECT_TRUE(memcmp(ExtractedBlockData.data(), BlockData, BlockSize) == 0);

  //----------------------------------------------------------------------
  // Test data forms
  //----------------------------------------------------------------------
  EXPECT_EQ(DieDG.getAttributeValueAsUnsignedConstant(Attr_DW_FORM_data1)
                .getValueOr(0),
            Data1);
  EXPECT_EQ(DieDG.getAttributeValueAsUnsignedConstant(Attr_DW_FORM_data2)
                .getValueOr(0),
            Data2);
  EXPECT_EQ(DieDG.getAttributeValueAsUnsignedConstant(Attr_DW_FORM_data4)
                .getValueOr(0),
            Data4);
  EXPECT_EQ(DieDG.getAttributeValueAsUnsignedConstant(Attr_DW_FORM_data8)
                .getValueOr(0),
            Data8);

  //----------------------------------------------------------------------
  // Test string forms
  //----------------------------------------------------------------------
  const char *ExtractedStringValue =
      DieDG.getAttributeValueAsString(Attr_DW_FORM_string, nullptr);
  EXPECT_TRUE(ExtractedStringValue != nullptr);
  EXPECT_TRUE(strcmp(StringValue, ExtractedStringValue) == 0);

  const char *ExtractedStrpValue =
      DieDG.getAttributeValueAsString(Attr_DW_FORM_strp, nullptr);
  EXPECT_TRUE(ExtractedStrpValue != nullptr);
  EXPECT_TRUE(strcmp(StrpValue, ExtractedStrpValue) == 0);

  //----------------------------------------------------------------------
  // Test reference forms
  //----------------------------------------------------------------------
  EXPECT_EQ(
      DieDG.getAttributeValueAsReference(Attr_DW_FORM_ref_addr).getValueOr(0),
      RefAddr);
  EXPECT_EQ(DieDG.getAttributeValueAsReference(Attr_DW_FORM_ref1).getValueOr(0),
            Data1);
  EXPECT_EQ(DieDG.getAttributeValueAsReference(Attr_DW_FORM_ref2).getValueOr(0),
            Data2);
  EXPECT_EQ(DieDG.getAttributeValueAsReference(Attr_DW_FORM_ref4).getValueOr(0),
            Data4);
  EXPECT_EQ(DieDG.getAttributeValueAsReference(Attr_DW_FORM_ref8).getValueOr(0),
            Data8);
  EXPECT_EQ(
      DieDG.getAttributeValueAsReference(Attr_DW_FORM_ref_sig8).getValueOr(0),
      Data8_2);
  EXPECT_EQ(
      DieDG.getAttributeValueAsReference(Attr_DW_FORM_ref_udata).getValueOr(0),
      UData[0]);

  //----------------------------------------------------------------------
  // Test flag forms
  //----------------------------------------------------------------------
  EXPECT_EQ(DieDG.getAttributeValueAsUnsignedConstant(Attr_DW_FORM_flag_true)
                .getValueOr(0),
            1ULL);
  EXPECT_EQ(DieDG.getAttributeValueAsUnsignedConstant(Attr_DW_FORM_flag_false)
                .getValueOr(1),
            0ULL);
  EXPECT_EQ(DieDG.getAttributeValueAsUnsignedConstant(Attr_DW_FORM_flag_present)
                .getValueOr(0ULL),
            1ULL);

  //----------------------------------------------------------------------
  // Test SLEB128 based forms
  //----------------------------------------------------------------------
  EXPECT_EQ(
      DieDG.getAttributeValueAsSignedConstant(Attr_DW_FORM_sdata).getValueOr(0),
      SData);
  if (Version >= 5)
    EXPECT_EQ(
        DieDG.getAttributeValueAsSignedConstant(Attr_DW_FORM_implicit_const)
            .getValueOr(0),
        ICSData);

  //----------------------------------------------------------------------
  // Test ULEB128 based forms
  //----------------------------------------------------------------------
  EXPECT_EQ(DieDG.getAttributeValueAsUnsignedConstant(Attr_DW_FORM_udata)
                .getValueOr(0),
            UData[0]);

  //----------------------------------------------------------------------
  // Test DWARF32/DWARF64 forms
  //----------------------------------------------------------------------
  EXPECT_EQ(DieDG.getAttributeValueAsReference(Attr_DW_FORM_GNU_ref_alt)
                .getValueOr(0),
            Dwarf32Values[0]);
  EXPECT_EQ(DieDG.getAttributeValueAsSectionOffset(Attr_DW_FORM_sec_offset)
                .getValueOr(0),
            Dwarf32Values[1]);

  //----------------------------------------------------------------------
  // Add an address at the end to make sure we can decode this value
  //----------------------------------------------------------------------
  EXPECT_EQ(DieDG.getAttributeValueAsAddress(Attr_Last).getValueOr(0),
            AddrValue);
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
  auto DieDG = U->getUnitDIE(false);
  EXPECT_TRUE(DieDG.isValid());
  // DieDG.dump(llvm::outs(), U, UINT32_MAX);

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
  auto Unit1DieDG = U1->getUnitDIE(false);
  EXPECT_TRUE(Unit1DieDG.isValid());
  // Unit1DieDG.dump(llvm::outs(), UINT32_MAX);

  auto Unit2DieDG = U2->getUnitDIE(false);
  EXPECT_TRUE(Unit2DieDG.isValid());
  // Unit2DieDG.dump(llvm::outs(), UINT32_MAX);

  // Verify the first child of the compile unit 1 DIE is our int base type.
  auto CU1TypeDieDG = Unit1DieDG.getFirstChild();
  EXPECT_TRUE(CU1TypeDieDG.isValid());
  EXPECT_EQ(CU1TypeDieDG.getTag(), DW_TAG_base_type);
  EXPECT_EQ(CU1TypeDieDG.getAttributeValueAsUnsignedConstant(DW_AT_encoding)
                .getValueOr(0),
            DW_ATE_signed);

  // Verify the first child of the compile unit 2 DIE is our float base type.
  auto CU2TypeDieDG = Unit2DieDG.getFirstChild();
  EXPECT_TRUE(CU2TypeDieDG.isValid());
  EXPECT_EQ(CU2TypeDieDG.getTag(), DW_TAG_base_type);
  EXPECT_EQ(CU2TypeDieDG.getAttributeValueAsUnsignedConstant(DW_AT_encoding)
                .getValueOr(0),
            DW_ATE_float);

  // Verify the sibling of the base type DIE is our Ref1 DIE and that its
  // DW_AT_type points to our base type DIE.
  auto CU1Ref1DieDG = CU1TypeDieDG.getSibling();
  EXPECT_TRUE(CU1Ref1DieDG.isValid());
  EXPECT_EQ(CU1Ref1DieDG.getTag(), DW_TAG_variable);
  EXPECT_EQ(
      CU1Ref1DieDG.getAttributeValueAsReference(DW_AT_type).getValueOr(-1ULL),
      CU1TypeDieDG.getOffset());
  // Verify the sibling is our Ref2 DIE and that its DW_AT_type points to our
  // base type DIE in CU1.
  auto CU1Ref2DieDG = CU1Ref1DieDG.getSibling();
  EXPECT_TRUE(CU1Ref2DieDG.isValid());
  EXPECT_EQ(CU1Ref2DieDG.getTag(), DW_TAG_variable);
  EXPECT_EQ(
      CU1Ref2DieDG.getAttributeValueAsReference(DW_AT_type).getValueOr(-1ULL),
      CU1TypeDieDG.getOffset());

  // Verify the sibling is our Ref4 DIE and that its DW_AT_type points to our
  // base type DIE in CU1.
  auto CU1Ref4DieDG = CU1Ref2DieDG.getSibling();
  EXPECT_TRUE(CU1Ref4DieDG.isValid());
  EXPECT_EQ(CU1Ref4DieDG.getTag(), DW_TAG_variable);
  EXPECT_EQ(
      CU1Ref4DieDG.getAttributeValueAsReference(DW_AT_type).getValueOr(-1ULL),
      CU1TypeDieDG.getOffset());

  // Verify the sibling is our Ref8 DIE and that its DW_AT_type points to our
  // base type DIE in CU1.
  auto CU1Ref8DieDG = CU1Ref4DieDG.getSibling();
  EXPECT_TRUE(CU1Ref8DieDG.isValid());
  EXPECT_EQ(CU1Ref8DieDG.getTag(), DW_TAG_variable);
  EXPECT_EQ(
      CU1Ref8DieDG.getAttributeValueAsReference(DW_AT_type).getValueOr(-1ULL),
      CU1TypeDieDG.getOffset());

  // Verify the sibling is our RefAddr DIE and that its DW_AT_type points to our
  // base type DIE in CU1.
  auto CU1RefAddrDieDG = CU1Ref8DieDG.getSibling();
  EXPECT_TRUE(CU1RefAddrDieDG.isValid());
  EXPECT_EQ(CU1RefAddrDieDG.getTag(), DW_TAG_variable);
  EXPECT_EQ(CU1RefAddrDieDG.getAttributeValueAsReference(DW_AT_type)
                .getValueOr(-1ULL),
            CU1TypeDieDG.getOffset());

  // Verify the sibling of the Ref4 DIE is our RefAddr DIE and that its
  // DW_AT_type points to our base type DIE.
  auto CU1ToCU2RefAddrDieDG = CU1RefAddrDieDG.getSibling();
  EXPECT_TRUE(CU1ToCU2RefAddrDieDG.isValid());
  EXPECT_EQ(CU1ToCU2RefAddrDieDG.getTag(), DW_TAG_variable);
  EXPECT_EQ(CU1ToCU2RefAddrDieDG.getAttributeValueAsReference(DW_AT_type)
                .getValueOr(-1ULL),
            CU2TypeDieDG.getOffset());

  // Verify the sibling of the base type DIE is our Ref1 DIE and that its
  // DW_AT_type points to our base type DIE.
  auto CU2Ref1DieDG = CU2TypeDieDG.getSibling();
  EXPECT_TRUE(CU2Ref1DieDG.isValid());
  EXPECT_EQ(CU2Ref1DieDG.getTag(), DW_TAG_variable);
  EXPECT_EQ(
      CU2Ref1DieDG.getAttributeValueAsReference(DW_AT_type).getValueOr(-1ULL),
      CU2TypeDieDG.getOffset());
  // Verify the sibling is our Ref2 DIE and that its DW_AT_type points to our
  // base type DIE in CU2.
  auto CU2Ref2DieDG = CU2Ref1DieDG.getSibling();
  EXPECT_TRUE(CU2Ref2DieDG.isValid());
  EXPECT_EQ(CU2Ref2DieDG.getTag(), DW_TAG_variable);
  EXPECT_EQ(
      CU2Ref2DieDG.getAttributeValueAsReference(DW_AT_type).getValueOr(-1ULL),
      CU2TypeDieDG.getOffset());

  // Verify the sibling is our Ref4 DIE and that its DW_AT_type points to our
  // base type DIE in CU2.
  auto CU2Ref4DieDG = CU2Ref2DieDG.getSibling();
  EXPECT_TRUE(CU2Ref4DieDG.isValid());
  EXPECT_EQ(CU2Ref4DieDG.getTag(), DW_TAG_variable);
  EXPECT_EQ(
      CU2Ref4DieDG.getAttributeValueAsReference(DW_AT_type).getValueOr(-1ULL),
      CU2TypeDieDG.getOffset());

  // Verify the sibling is our Ref8 DIE and that its DW_AT_type points to our
  // base type DIE in CU2.
  auto CU2Ref8DieDG = CU2Ref4DieDG.getSibling();
  EXPECT_TRUE(CU2Ref8DieDG.isValid());
  EXPECT_EQ(CU2Ref8DieDG.getTag(), DW_TAG_variable);
  EXPECT_EQ(
      CU2Ref8DieDG.getAttributeValueAsReference(DW_AT_type).getValueOr(-1ULL),
      CU2TypeDieDG.getOffset());

  // Verify the sibling is our RefAddr DIE and that its DW_AT_type points to our
  // base type DIE in CU2.
  auto CU2RefAddrDieDG = CU2Ref8DieDG.getSibling();
  EXPECT_TRUE(CU2RefAddrDieDG.isValid());
  EXPECT_EQ(CU2RefAddrDieDG.getTag(), DW_TAG_variable);
  EXPECT_EQ(CU2RefAddrDieDG.getAttributeValueAsReference(DW_AT_type)
                .getValueOr(-1ULL),
            CU2TypeDieDG.getOffset());

  // Verify the sibling of the Ref4 DIE is our RefAddr DIE and that its
  // DW_AT_type points to our base type DIE.
  auto CU2ToCU1RefAddrDieDG = CU2RefAddrDieDG.getSibling();
  EXPECT_TRUE(CU2ToCU1RefAddrDieDG.isValid());
  EXPECT_EQ(CU2ToCU1RefAddrDieDG.getTag(), DW_TAG_variable);
  EXPECT_EQ(CU2ToCU1RefAddrDieDG.getAttributeValueAsReference(DW_AT_type)
                .getValueOr(-1ULL),
            CU1TypeDieDG.getOffset());
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
  // Test the DWARF APIs related to accessing the DW_AT_low_pc and
  // DW_AT_high_pc.
  const uint8_t AddrSize = sizeof(AddrType);
  const bool SupportsHighPCAsOffset = Version >= 4;
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
  DWARFContextInMemory DwarfContext(*Obj.get());
  
  // Verify the number of compile units is correct.
  uint32_t NumCUs = DwarfContext.getNumCompileUnits();
  EXPECT_EQ(NumCUs, 1u);
  DWARFCompileUnit *U = DwarfContext.getCompileUnitAtIndex(0);
  
  // Get the compile unit DIE is valid.
  auto DieDG = U->getUnitDIE(false);
  EXPECT_TRUE(DieDG.isValid());
  // DieDG.dump(llvm::outs(), U, UINT32_MAX);
  
  uint64_t LowPC, HighPC;
  Optional<uint64_t> OptU64;
  // Verify the that our subprogram with no PC value fails appropriately when
  // asked for any PC values.
  auto SubprogramDieNoPC = DieDG.getFirstChild();
  EXPECT_TRUE(SubprogramDieNoPC.isValid());
  EXPECT_EQ(SubprogramDieNoPC.getTag(), DW_TAG_subprogram);
  OptU64 = SubprogramDieNoPC.getAttributeValueAsAddress(DW_AT_low_pc);
  EXPECT_FALSE((bool)OptU64);
  OptU64 = SubprogramDieNoPC.getAttributeValueAsAddress(DW_AT_high_pc);
  EXPECT_FALSE((bool)OptU64);
  EXPECT_FALSE(SubprogramDieNoPC.getLowAndHighPC(LowPC, HighPC));
  OptU64 = SubprogramDieNoPC.getAttributeValueAsAddress(DW_AT_high_pc);
  EXPECT_FALSE((bool)OptU64);
  OptU64 = SubprogramDieNoPC.getAttributeValueAsUnsignedConstant(DW_AT_high_pc);
  EXPECT_FALSE((bool)OptU64);
  OptU64 = SubprogramDieNoPC.getHighPC(ActualLowPC);
  EXPECT_FALSE((bool)OptU64);
  EXPECT_FALSE(SubprogramDieNoPC.getLowAndHighPC(LowPC, HighPC));
  
  
  // Verify the that our subprogram with only a low PC value succeeds when
  // we ask for the Low PC, but fails appropriately when asked for the high PC
  // or both low and high PC values.
  auto SubprogramDieLowPC = SubprogramDieNoPC.getSibling();
  EXPECT_TRUE(SubprogramDieLowPC.isValid());
  EXPECT_EQ(SubprogramDieLowPC.getTag(), DW_TAG_subprogram);
  OptU64 = SubprogramDieLowPC.getAttributeValueAsAddress(DW_AT_low_pc);
  EXPECT_TRUE((bool)OptU64);
  EXPECT_EQ(OptU64.getValue(), ActualLowPC);
  OptU64 = SubprogramDieLowPC.getAttributeValueAsAddress(DW_AT_high_pc);
  EXPECT_FALSE((bool)OptU64);
  OptU64 = SubprogramDieLowPC.getAttributeValueAsUnsignedConstant(DW_AT_high_pc);
  EXPECT_FALSE((bool)OptU64);
  OptU64 = SubprogramDieLowPC.getHighPC(ActualLowPC);
  EXPECT_FALSE((bool)OptU64);
  EXPECT_FALSE(SubprogramDieLowPC.getLowAndHighPC(LowPC, HighPC));

  
  // Verify the that our subprogram with only a low PC value succeeds when
  // we ask for the Low PC, but fails appropriately when asked for the high PC
  // or both low and high PC values.
  auto SubprogramDieLowHighPC = SubprogramDieLowPC.getSibling();
  EXPECT_TRUE(SubprogramDieLowHighPC.isValid());
  EXPECT_EQ(SubprogramDieLowHighPC.getTag(), DW_TAG_subprogram);
  OptU64 = SubprogramDieLowHighPC.getAttributeValueAsAddress(DW_AT_low_pc);
  EXPECT_TRUE((bool)OptU64);
  EXPECT_EQ(OptU64.getValue(), ActualLowPC);
  // Get the high PC as an address. This should succeed if the high PC was
  // encoded as an address and fail if the high PC was encoded as an offset.
  OptU64 = SubprogramDieLowHighPC.getAttributeValueAsAddress(DW_AT_high_pc);
  if (SupportsHighPCAsOffset) {
    EXPECT_FALSE((bool)OptU64);
  } else {
    EXPECT_TRUE((bool)OptU64);
    EXPECT_EQ(OptU64.getValue(), ActualHighPC);
  }
  // Get the high PC as an unsigned constant. This should succeed if the high PC
  // was encoded as an offset and fail if the high PC was encoded as an address.
  OptU64 = SubprogramDieLowHighPC.getAttributeValueAsUnsignedConstant(
      DW_AT_high_pc);
  if (SupportsHighPCAsOffset) {
    EXPECT_TRUE((bool)OptU64);
    EXPECT_EQ(OptU64.getValue(), ActualHighPCOffset);
  } else {
    EXPECT_FALSE((bool)OptU64);
  }

  OptU64 = SubprogramDieLowHighPC.getHighPC(ActualLowPC);
  EXPECT_TRUE((bool)OptU64);
  EXPECT_EQ(OptU64.getValue(), ActualHighPC);

  EXPECT_TRUE(SubprogramDieLowHighPC.getLowAndHighPC(LowPC, HighPC));
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

TEST(DWARFDebugInfo, TestRelations) {
  // Test the DWARF APIs related to accessing the DW_AT_low_pc and
  // DW_AT_high_pc.
  uint16_t Version = 4;
  
  const uint8_t AddrSize = sizeof(void *);
  initLLVMIfNeeded();
  Triple Triple = getHostTripleForAddrSize(AddrSize);
  auto ExpectedDG = dwarfgen::Generator::create(Triple, Version);
  if (HandleExpectedError(ExpectedDG))
    return;
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
  DWARFContextInMemory DwarfContext(*Obj.get());
  
  // Verify the number of compile units is correct.
  uint32_t NumCUs = DwarfContext.getNumCompileUnits();
  EXPECT_EQ(NumCUs, 1u);
  DWARFCompileUnit *U = DwarfContext.getCompileUnitAtIndex(0);
  
  // Get the compile unit DIE is valid.
  auto CUDie = U->getUnitDIE(false);
  EXPECT_TRUE(CUDie.isValid());
  // CUDie.dump(llvm::outs(), UINT32_MAX);
  
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
  // Test the DWARF APIs related to iterating across the children of a DIE using
  // the DWARFDie::iterator class.
  uint16_t Version = 4;
  
  const uint8_t AddrSize = sizeof(void *);
  initLLVMIfNeeded();
  Triple Triple = getHostTripleForAddrSize(AddrSize);
  auto ExpectedDG = dwarfgen::Generator::create(Triple, Version);
  if (HandleExpectedError(ExpectedDG))
    return;
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
  DWARFContextInMemory DwarfContext(*Obj.get());
  
  // Verify the number of compile units is correct.
  uint32_t NumCUs = DwarfContext.getNumCompileUnits();
  EXPECT_EQ(NumCUs, 1u);
  DWARFCompileUnit *U = DwarfContext.getCompileUnitAtIndex(0);
  
  // Get the compile unit DIE is valid.
  auto CUDie = U->getUnitDIE(false);
  EXPECT_TRUE(CUDie.isValid());
  // CUDie.dump(llvm::outs(), UINT32_MAX);
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
  // Test a DIE that says it has children in the abbreviation, but actually
  // doesn't have any attributes, will not return anything during iteration.
  // We do this by making sure the begin and end iterators are equal.
  uint16_t Version = 4;
  
  const uint8_t AddrSize = sizeof(void *);
  initLLVMIfNeeded();
  Triple Triple = getHostTripleForAddrSize(AddrSize);
  auto ExpectedDG = dwarfgen::Generator::create(Triple, Version);
  if (HandleExpectedError(ExpectedDG))
    return;
  dwarfgen::Generator *DG = ExpectedDG.get().get();
  dwarfgen::CompileUnit &CU = DG->addCompileUnit();
  
  // Scope to allow us to re-use the same DIE names
  {
    // Create a compile unit DIE that has an abbreviation that says it has
    // children, but doesn't have any actual attributes. This helps us test
    // a DIE that has only one child: a NULL DIE.
    auto CUDie = CU.getUnitDIE();
    CUDie.setForceChildren();
  }
  
  MemoryBufferRef FileBuffer(DG->generate(), "dwarf");
  auto Obj = object::ObjectFile::createObjectFile(FileBuffer);
  EXPECT_TRUE((bool)Obj);
  DWARFContextInMemory DwarfContext(*Obj.get());
  
  // Verify the number of compile units is correct.
  uint32_t NumCUs = DwarfContext.getNumCompileUnits();
  EXPECT_EQ(NumCUs, 1u);
  DWARFCompileUnit *U = DwarfContext.getCompileUnitAtIndex(0);
  
  // Get the compile unit DIE is valid.
  auto CUDie = U->getUnitDIE(false);
  EXPECT_TRUE(CUDie.isValid());
  CUDie.dump(llvm::outs(), UINT32_MAX);
  
  // Verify that the CU Die that says it has children, but doesn't, actually
  // has begin and end iterators that are equal. We want to make sure we don't
  // see the Null DIEs during iteration.
  EXPECT_EQ(CUDie.begin(), CUDie.end());
}

} // end anonymous namespace
