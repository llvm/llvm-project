//===- llvm/unittest/DebugInfo/DWARFFormValueTest.cpp ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DwarfGenerator.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/DebugInfo/DWARF/DWARFCompileUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/ObjectYAML/DWARFEmitter.h"
#include "llvm/ObjectYAML/DWARFYAML.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "gtest/gtest.h"
#include <climits>
#include <cstdint>
#include <cstring>
#include <string>

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
  handleAllErrors(Expected.takeError(), [&](const ErrorInfoBase &EI) {
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
  DWARFContextInMemory DwarfContext(*Obj.get());
  uint32_t NumCUs = DwarfContext.getNumCompileUnits();
  EXPECT_EQ(NumCUs, 1u);
  DWARFCompileUnit *U = DwarfContext.getCompileUnitAtIndex(0);
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
  EXPECT_TRUE(strcmp(StringValue, *ExtractedStringValue) == 0);

  auto ExtractedStrpValue = toString(DieDG.find(Attr_DW_FORM_strp));
  EXPECT_TRUE((bool)ExtractedStrpValue);
  EXPECT_TRUE(strcmp(StrpValue, *ExtractedStrpValue) == 0);

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
  
  uint64_t LowPC, HighPC;
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
  EXPECT_FALSE(SubprogramDieNoPC.getLowAndHighPC(LowPC, HighPC));
  OptU64 = toAddress(SubprogramDieNoPC.find(DW_AT_high_pc));
  EXPECT_FALSE((bool)OptU64);
  OptU64 = toUnsigned(SubprogramDieNoPC.find(DW_AT_high_pc));
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
  OptU64 = toAddress(SubprogramDieLowPC.find(DW_AT_low_pc));
  EXPECT_TRUE((bool)OptU64);
  EXPECT_EQ(OptU64.getValue(), ActualLowPC);
  OptU64 = toAddress(SubprogramDieLowPC.find(DW_AT_high_pc));
  EXPECT_FALSE((bool)OptU64);
  OptU64 = toUnsigned(SubprogramDieLowPC.find(DW_AT_high_pc));
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
                         "  - Length:\n"
                         "      TotalLength:          9\n"
                         "    Version:         4\n"
                         "    AbbrOffset:      0\n"
                         "    AddrSize:        8\n"
                         "    Entries:\n"
                         "      - AbbrCode:        0x00000001\n"
                         "        Values:\n"
                         "      - AbbrCode:        0x00000000\n"
                         "        Values:\n";

  auto ErrOrSections = DWARFYAML::EmitDebugSections(StringRef(yamldata));
  ASSERT_TRUE((bool)ErrOrSections);
  DWARFContextInMemory DwarfContext(*ErrOrSections, 8);

  // Verify the number of compile units is correct.
  uint32_t NumCUs = DwarfContext.getNumCompileUnits();
  EXPECT_EQ(NumCUs, 1u);
  DWARFCompileUnit *U = DwarfContext.getCompileUnitAtIndex(0);

  // Get the compile unit DIE is valid.
  auto CUDie = U->getUnitDIE(false);
  EXPECT_TRUE(CUDie.isValid());
  
  // Verify that the CU Die that says it has children, but doesn't, actually
  // has begin and end iterators that are equal. We want to make sure we don't
  // see the Null DIEs during iteration.
  EXPECT_EQ(CUDie.begin(), CUDie.end());
}

TEST(DWARFDebugInfo, TestAttributeIterators) {
  // Test the DWARF APIs related to iterating across all attribute values in a
  // a DWARFDie.
  uint16_t Version = 4;
  
  const uint8_t AddrSize = sizeof(void *);
  initLLVMIfNeeded();
  Triple Triple = getHostTripleForAddrSize(AddrSize);
  auto ExpectedDG = dwarfgen::Generator::create(Triple, Version);
  if (HandleExpectedError(ExpectedDG))
    return;
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
  DWARFContextInMemory DwarfContext(*Obj.get());
  
  // Verify the number of compile units is correct.
  uint32_t NumCUs = DwarfContext.getNumCompileUnits();
  EXPECT_EQ(NumCUs, 1u);
  DWARFCompileUnit *U = DwarfContext.getCompileUnitAtIndex(0);
  
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
  uint16_t Version = 4;
  
  const uint8_t AddrSize = sizeof(void *);
  initLLVMIfNeeded();
  Triple Triple = getHostTripleForAddrSize(AddrSize);
  auto ExpectedDG = dwarfgen::Generator::create(Triple, Version);
  if (HandleExpectedError(ExpectedDG))
    return;
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
    auto FuncDie = CUDie.addChild(DW_TAG_subprogram);
    auto VarAbsDie = CUDie.addChild(DW_TAG_variable);
    auto VarDie = CUDie.addChild(DW_TAG_variable);
    FuncSpecDie.addAttribute(DW_AT_name, DW_FORM_strp, SpecDieName);
    FuncAbsDie.addAttribute(DW_AT_linkage_name, DW_FORM_strp, SpecLinkageName);
    FuncAbsDie.addAttribute(DW_AT_specification, DW_FORM_ref4, FuncSpecDie);
    FuncDie.addAttribute(DW_AT_abstract_origin, DW_FORM_ref4, FuncAbsDie);
    VarAbsDie.addAttribute(DW_AT_name, DW_FORM_strp, AbsDieName);
    VarDie.addAttribute(DW_AT_abstract_origin, DW_FORM_ref4, VarAbsDie);
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
  
  auto FuncSpecDie = CUDie.getFirstChild();
  auto FuncAbsDie = FuncSpecDie.getSibling();
  auto FuncDie = FuncAbsDie.getSibling();
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
  DWARFFormValue FormVal;
  uint64_t InvalidU64 = 0xBADBADBADBADBADB;
  int64_t InvalidS64 = 0xBADBADBADBADBADB;
  // First test that we don't get valid values back when using an optional with
  // no value.
  Optional<DWARFFormValue> FormValOpt;
  EXPECT_FALSE(toString(FormValOpt).hasValue());
  EXPECT_FALSE(toUnsigned(FormValOpt).hasValue());
  EXPECT_FALSE(toReference(FormValOpt).hasValue());
  EXPECT_FALSE(toSigned(FormValOpt).hasValue());
  EXPECT_FALSE(toAddress(FormValOpt).hasValue());
  EXPECT_FALSE(toSectionOffset(FormValOpt).hasValue());
  EXPECT_FALSE(toBlock(FormValOpt).hasValue());
  EXPECT_EQ(nullptr, toString(FormValOpt, nullptr));
  EXPECT_EQ(InvalidU64, toUnsigned(FormValOpt, InvalidU64));
  EXPECT_EQ(InvalidU64, toReference(FormValOpt, InvalidU64));
  EXPECT_EQ(InvalidU64, toAddress(FormValOpt, InvalidU64));
  EXPECT_EQ(InvalidU64, toSectionOffset(FormValOpt, InvalidU64));
  EXPECT_EQ(InvalidS64, toSigned(FormValOpt, InvalidS64));

  // Test successful and unsuccessful address decoding.
  uint64_t Address = 0x100000000ULL;
  FormVal.setForm(DW_FORM_addr);
  FormVal.setUValue(Address);
  FormValOpt = FormVal;

  EXPECT_FALSE(toString(FormValOpt).hasValue());
  EXPECT_FALSE(toUnsigned(FormValOpt).hasValue());
  EXPECT_FALSE(toReference(FormValOpt).hasValue());
  EXPECT_FALSE(toSigned(FormValOpt).hasValue());
  EXPECT_TRUE(toAddress(FormValOpt).hasValue());
  EXPECT_FALSE(toSectionOffset(FormValOpt).hasValue());
  EXPECT_FALSE(toBlock(FormValOpt).hasValue());
  EXPECT_EQ(nullptr, toString(FormValOpt, nullptr));
  EXPECT_EQ(InvalidU64, toUnsigned(FormValOpt, InvalidU64));
  EXPECT_EQ(InvalidU64, toReference(FormValOpt, InvalidU64));
  EXPECT_EQ(Address, toAddress(FormValOpt, InvalidU64));
  EXPECT_EQ(InvalidU64, toSectionOffset(FormValOpt, InvalidU64));
  EXPECT_EQ(InvalidS64, toSigned(FormValOpt, InvalidU64));

  // Test successful and unsuccessful unsigned constant decoding.
  uint64_t UData8 = 0x1020304050607080ULL;
  FormVal.setForm(DW_FORM_udata);
  FormVal.setUValue(UData8);
  FormValOpt = FormVal;
  
  EXPECT_FALSE(toString(FormValOpt).hasValue());
  EXPECT_TRUE(toUnsigned(FormValOpt).hasValue());
  EXPECT_FALSE(toReference(FormValOpt).hasValue());
  EXPECT_TRUE(toSigned(FormValOpt).hasValue());
  EXPECT_FALSE(toAddress(FormValOpt).hasValue());
  EXPECT_FALSE(toSectionOffset(FormValOpt).hasValue());
  EXPECT_FALSE(toBlock(FormValOpt).hasValue());
  EXPECT_EQ(nullptr, toString(FormValOpt, nullptr));
  EXPECT_EQ(UData8, toUnsigned(FormValOpt, InvalidU64));
  EXPECT_EQ(InvalidU64, toReference(FormValOpt, InvalidU64));
  EXPECT_EQ(InvalidU64, toAddress(FormValOpt, InvalidU64));
  EXPECT_EQ(InvalidU64, toSectionOffset(FormValOpt, InvalidU64));
  EXPECT_EQ((int64_t)UData8, toSigned(FormValOpt, InvalidU64));

  // Test successful and unsuccessful reference decoding.
  uint32_t RefData = 0x11223344U;
  FormVal.setForm(DW_FORM_ref_addr);
  FormVal.setUValue(RefData);
  FormValOpt = FormVal;
  
  EXPECT_FALSE(toString(FormValOpt).hasValue());
  EXPECT_FALSE(toUnsigned(FormValOpt).hasValue());
  EXPECT_TRUE(toReference(FormValOpt).hasValue());
  EXPECT_FALSE(toSigned(FormValOpt).hasValue());
  EXPECT_FALSE(toAddress(FormValOpt).hasValue());
  EXPECT_FALSE(toSectionOffset(FormValOpt).hasValue());
  EXPECT_FALSE(toBlock(FormValOpt).hasValue());
  EXPECT_EQ(nullptr, toString(FormValOpt, nullptr));
  EXPECT_EQ(InvalidU64, toUnsigned(FormValOpt, InvalidU64));
  EXPECT_EQ(RefData, toReference(FormValOpt, InvalidU64));
  EXPECT_EQ(InvalidU64, toAddress(FormValOpt, InvalidU64));
  EXPECT_EQ(InvalidU64, toSectionOffset(FormValOpt, InvalidU64));
  EXPECT_EQ(InvalidS64, toSigned(FormValOpt, InvalidU64));

  // Test successful and unsuccessful signed constant decoding.
  int64_t SData8 = 0x1020304050607080ULL;
  FormVal.setForm(DW_FORM_udata);
  FormVal.setSValue(SData8);
  FormValOpt = FormVal;
  
  EXPECT_FALSE(toString(FormValOpt).hasValue());
  EXPECT_TRUE(toUnsigned(FormValOpt).hasValue());
  EXPECT_FALSE(toReference(FormValOpt).hasValue());
  EXPECT_TRUE(toSigned(FormValOpt).hasValue());
  EXPECT_FALSE(toAddress(FormValOpt).hasValue());
  EXPECT_FALSE(toSectionOffset(FormValOpt).hasValue());
  EXPECT_FALSE(toBlock(FormValOpt).hasValue());
  EXPECT_EQ(nullptr, toString(FormValOpt, nullptr));
  EXPECT_EQ((uint64_t)SData8, toUnsigned(FormValOpt, InvalidU64));
  EXPECT_EQ(InvalidU64, toReference(FormValOpt, InvalidU64));
  EXPECT_EQ(InvalidU64, toAddress(FormValOpt, InvalidU64));
  EXPECT_EQ(InvalidU64, toSectionOffset(FormValOpt, InvalidU64));
  EXPECT_EQ(SData8, toSigned(FormValOpt, InvalidU64));

  // Test successful and unsuccessful block decoding.
  uint8_t Data[] = { 2, 3, 4 };
  ArrayRef<uint8_t> Array(Data);
  FormVal.setForm(DW_FORM_block1);
  FormVal.setBlockValue(Array);
  FormValOpt = FormVal;
  
  EXPECT_FALSE(toString(FormValOpt).hasValue());
  EXPECT_FALSE(toUnsigned(FormValOpt).hasValue());
  EXPECT_FALSE(toReference(FormValOpt).hasValue());
  EXPECT_FALSE(toSigned(FormValOpt).hasValue());
  EXPECT_FALSE(toAddress(FormValOpt).hasValue());
  EXPECT_FALSE(toSectionOffset(FormValOpt).hasValue());
  auto BlockOpt = toBlock(FormValOpt);
  EXPECT_TRUE(BlockOpt.hasValue());
  EXPECT_EQ(*BlockOpt, Array);
  EXPECT_EQ(nullptr, toString(FormValOpt, nullptr));
  EXPECT_EQ(InvalidU64, toUnsigned(FormValOpt, InvalidU64));
  EXPECT_EQ(InvalidU64, toReference(FormValOpt, InvalidU64));
  EXPECT_EQ(InvalidU64, toAddress(FormValOpt, InvalidU64));
  EXPECT_EQ(InvalidU64, toSectionOffset(FormValOpt, InvalidU64));
  EXPECT_EQ(InvalidS64, toSigned(FormValOpt, InvalidU64));

  // Test
}

TEST(DWARFDebugInfo, TestFindAttrs) {
  // Test the DWARFDie::find() and DWARFDie::findRecursively() that take an
  // ArrayRef<dwarf::Attribute> value to make sure they work correctly.
  uint16_t Version = 4;
  
  const uint8_t AddrSize = sizeof(void *);
  initLLVMIfNeeded();
  Triple Triple = getHostTripleForAddrSize(AddrSize);
  auto ExpectedDG = dwarfgen::Generator::create(Triple, Version);
  if (HandleExpectedError(ExpectedDG))
    return;
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
  DWARFContextInMemory DwarfContext(*Obj.get());
  
  // Verify the number of compile units is correct.
  uint32_t NumCUs = DwarfContext.getNumCompileUnits();
  EXPECT_EQ(NumCUs, 1u);
  DWARFCompileUnit *U = DwarfContext.getCompileUnitAtIndex(0);
  
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
  uint16_t Version = 5;

  const uint8_t AddrSize = sizeof(void *);
  initLLVMIfNeeded();
  Triple Triple = getHostTripleForAddrSize(AddrSize);
  auto ExpectedDG = dwarfgen::Generator::create(Triple, Version);
  if (HandleExpectedError(ExpectedDG))
    return;
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
  DWARFContextInMemory DwarfContext(*Obj.get());
  DWARFCompileUnit *U = DwarfContext.getCompileUnitAtIndex(0);
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

void VerifyError(DWARFContext &DwarfContext, StringRef Error) {
  SmallString<1024> Str;
  raw_svector_ostream Strm(Str);
  EXPECT_FALSE(DwarfContext.verify(Strm, DIDT_All));
  EXPECT_TRUE(Str.str().contains(Error));
}

TEST(DWARFDebugInfo, TestDwarfVerifyInvalidCURef) {
  // Create a single compile unit with a single function that has a DW_AT_type
  // that is CU relative. The CU offset is not valid becuase it is larger than
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
      - Length:
          TotalLength:     22
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
  auto ErrOrSections = DWARFYAML::EmitDebugSections(StringRef(yamldata));
  ASSERT_TRUE((bool)ErrOrSections);
  DWARFContextInMemory DwarfContext(*ErrOrSections, 8);
  VerifyError(DwarfContext, "error: DW_FORM_ref4 CU offset 0x00001234 is "
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
      - Length:
          TotalLength:     22
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
  auto ErrOrSections = DWARFYAML::EmitDebugSections(StringRef(yamldata));
  ASSERT_TRUE((bool)ErrOrSections);
  DWARFContextInMemory DwarfContext(*ErrOrSections, 8);
  VerifyError(DwarfContext,
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
      - Length:
          TotalLength:     16
        Version:         4
        AbbrOffset:      0
        AddrSize:        8
        Entries:
          - AbbrCode:        0x00000001
            Values:
              - Value:           0x0000000000000001
              - Value:           0x0000000000001000

  )";
  auto ErrOrSections = DWARFYAML::EmitDebugSections(StringRef(yamldata));
  ASSERT_TRUE((bool)ErrOrSections);
  DWARFContextInMemory DwarfContext(*ErrOrSections, 8);
  VerifyError(DwarfContext,
              "error: DW_AT_ranges offset is beyond .debug_ranges bounds:");
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
      - Length:
          TotalLength:     16
        Version:         4
        AbbrOffset:      0
        AddrSize:        8
        Entries:
          - AbbrCode:        0x00000001
            Values:
              - Value:           0x0000000000000001
              - Value:           0x0000000000001000

  )";
  auto ErrOrSections = DWARFYAML::EmitDebugSections(StringRef(yamldata));
  ASSERT_TRUE((bool)ErrOrSections);
  DWARFContextInMemory DwarfContext(*ErrOrSections, 8);
  VerifyError(
      DwarfContext,
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
      - Length:
          TotalLength:     12
        Version:         4
        AbbrOffset:      0
        AddrSize:        8
        Entries:
          - AbbrCode:        0x00000001
            Values:
              - Value:           0x0000000000001234
  )";
  auto ErrOrSections = DWARFYAML::EmitDebugSections(StringRef(yamldata));
  ASSERT_TRUE((bool)ErrOrSections);
  DWARFContextInMemory DwarfContext(*ErrOrSections, 8);
  VerifyError(DwarfContext,
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
      - Length:
          TotalLength:     22
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
  auto ErrOrSections = DWARFYAML::EmitDebugSections(StringRef(yamldata));
  ASSERT_TRUE((bool)ErrOrSections);
  DWARFContextInMemory DwarfContext(*ErrOrSections, 8);
  VerifyError(
      DwarfContext,
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
      - Length:
          TotalLength:     16
        Version:         4
        AbbrOffset:      0
        AddrSize:        8
        Entries:
          - AbbrCode:        0x00000001
            Values:
              - Value:           0x0000000000000001
              - Value:           0x0000000000000000
    debug_line:
      - Length:
          TotalLength:     68
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
  auto ErrOrSections = DWARFYAML::EmitDebugSections(yamldata);
  ASSERT_TRUE((bool)ErrOrSections);
  DWARFContextInMemory DwarfContext(*ErrOrSections, 8);
  VerifyError(DwarfContext, "error: .debug_line[0x00000000] row[1] decreases "
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
      - Length:
          TotalLength:     16
        Version:         4
        AbbrOffset:      0
        AddrSize:        8
        Entries:
          - AbbrCode:        0x00000001
            Values:
              - Value:           0x0000000000000001
              - Value:           0x0000000000000000
    debug_line:
      - Length:
          TotalLength:     61
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
  auto ErrOrSections = DWARFYAML::EmitDebugSections(yamldata);
  ASSERT_TRUE((bool)ErrOrSections);
  DWARFContextInMemory DwarfContext(*ErrOrSections, 8);
  VerifyError(DwarfContext, "error: .debug_line[0x00000000][1] has invalid "
                            "file index 5 (valid values are [1,1]):");
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
      - Length:          
          TotalLength:     16
        Version:         4
        AbbrOffset:      0
        AddrSize:        8
        Entries:         
          - AbbrCode:        0x00000001
            Values:          
              - Value:           0x0000000000000001
              - Value:           0x0000000000000000
      - Length:          
          TotalLength:     16
        Version:         4
        AbbrOffset:      0
        AddrSize:        8
        Entries:         
          - AbbrCode:        0x00000001
            Values:          
              - Value:           0x000000000000000D
              - Value:           0x0000000000000000
    debug_line:      
      - Length:          
          TotalLength:     60
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
  auto ErrOrSections = DWARFYAML::EmitDebugSections(yamldata);
  ASSERT_TRUE((bool)ErrOrSections);
  DWARFContextInMemory DwarfContext(*ErrOrSections, 8);
  VerifyError(DwarfContext, "error: two compile unit DIEs, 0x0000000b and "
                            "0x0000001f, have the same DW_AT_stmt_list section "
                            "offset:");
}

} // end anonymous namespace
