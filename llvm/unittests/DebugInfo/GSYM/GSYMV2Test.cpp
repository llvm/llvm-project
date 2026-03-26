//===- llvm/unittest/DebugInfo/GSYMV2Test.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/DebugInfo/GSYM/FileWriter.h"
#include "llvm/DebugInfo/GSYM/FunctionInfo.h"
#include "llvm/DebugInfo/GSYM/GlobalData.h"
#include "llvm/DebugInfo/GSYM/GsymCreatorV2.h"
#include "llvm/DebugInfo/GSYM/HeaderV2.h"
#include "llvm/DebugInfo/GSYM/OutputAggregator.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Testing/Support/Error.h"

#include "gtest/gtest.h"
#include <string>

using namespace llvm;
using namespace gsym;

static void checkError(std::string ExpectedMsg, Error Err) {
  ASSERT_TRUE(bool(Err));
  handleAllErrors(std::move(Err), [&](const ErrorInfoBase &Actual) {
    EXPECT_EQ(Actual.message(), ExpectedMsg);
  });
}

/// Helper to encode a GsymCreatorV2 and return the raw bytes.
static Expected<SmallString<512>> encodeV2(const GsymCreatorV2 &GC,
                                           llvm::endianness ByteOrder) {
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  FileWriter FW(OutStrm, ByteOrder);
  if (auto Err = GC.encode(FW))
    return std::move(Err);
  return Str;
}

/// Helper to decode the HeaderV2 from raw bytes.
static Expected<HeaderV2> decodeHeaderV2(StringRef Data,
                                         llvm::endianness ByteOrder) {
  DataExtractor DE(Data, ByteOrder == llvm::endianness::little, 8);
  return HeaderV2::decode(DE);
}

/// Helper to decode a GlobalData entry at a given offset.
static GlobalData decodeGlobalDataEntry(StringRef Data, uint64_t &Offset,
                                        llvm::endianness ByteOrder) {
  DataExtractor DE(Data, ByteOrder == llvm::endianness::little, 8);
  GlobalData GD;
  GD.Type = static_cast<GlobalInfoType>(DE.getU32(&Offset));
  GD.Padding = DE.getU32(&Offset);
  GD.FileOffset = DE.getU64(&Offset);
  GD.FileSize = DE.getU64(&Offset);
  return GD;
}

//===----------------------------------------------------------------------===//
// Encode error tests
//===----------------------------------------------------------------------===//

TEST(GSYMV2Test, TestEncodeErrorNoFunctions) {
  GsymCreatorV2 GC;
  auto Result = encodeV2(GC, llvm::endianness::little);
  checkError("no functions to encode", Result.takeError());
}

TEST(GSYMV2Test, TestEncodeErrorNotFinalized) {
  GsymCreatorV2 GC;
  const uint32_t Name = GC.insertString("foo");
  GC.addFunctionInfo(FunctionInfo(0x1000, 0x100, Name));
  auto Result = encodeV2(GC, llvm::endianness::little);
  checkError("GsymCreatorV2 wasn't finalized prior to encoding",
             Result.takeError());
}

TEST(GSYMV2Test, TestDoubleFinalize) {
  GsymCreatorV2 GC;
  const uint32_t Name = GC.insertString("foo");
  GC.addFunctionInfo(FunctionInfo(0x1000, 0x100, Name));
  OutputAggregator Null(nullptr);
  Error Err = GC.finalize(Null);
  ASSERT_FALSE(bool(Err));
  Err = GC.finalize(Null);
  ASSERT_TRUE(bool(Err));
  checkError("already finalized", std::move(Err));
}

//===----------------------------------------------------------------------===//
// Header and GlobalData structure tests
//===----------------------------------------------------------------------===//

/// Encode a V2 GSYM and verify the header fields and GlobalData layout.
static void TestV2HeaderAndGlobalData(llvm::endianness ByteOrder,
                                      uint64_t BaseAddr,
                                      uint8_t ExpectedAddrOffSize,
                                      uint32_t ExpectedNumAddresses,
                                      bool HasUUID) {
  GsymCreatorV2 GC;
  const uint32_t Func1Name = GC.insertString("foo");
  const uint32_t Func2Name = GC.insertString("bar");
  GC.addFunctionInfo(FunctionInfo(BaseAddr + 0x00, 0x10, Func1Name));
  GC.addFunctionInfo(FunctionInfo(BaseAddr + 0x20, 0x10, Func2Name));

  uint8_t UUID[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  if (HasUUID)
    GC.setUUID(UUID);

  OutputAggregator Null(nullptr);
  Error Err = GC.finalize(Null);
  ASSERT_FALSE(bool(Err));

  auto Result = encodeV2(GC, ByteOrder);
  ASSERT_THAT_EXPECTED(Result, Succeeded());
  StringRef Data = *Result;

  // Verify header.
  auto HdrOrErr = decodeHeaderV2(Data, ByteOrder);
  ASSERT_THAT_EXPECTED(HdrOrErr, Succeeded());
  const HeaderV2 &Hdr = *HdrOrErr;
  EXPECT_EQ(Hdr.Magic, GSYM_MAGIC);
  EXPECT_EQ(Hdr.Version, GSYM_VERSION_2);
  EXPECT_EQ(Hdr.Padding, 0u);
  EXPECT_EQ(Hdr.BaseAddress, BaseAddr);
  EXPECT_EQ(Hdr.NumAddresses, ExpectedNumAddresses);
  EXPECT_EQ(Hdr.AddrOffSize, ExpectedAddrOffSize);
  EXPECT_EQ(Hdr.AddrInfoOffSize, 4u); // Small file, should be 4 bytes.
  EXPECT_EQ(Hdr.StrpSize, 4u);        // Small string table, should be 4 bytes.
  EXPECT_EQ(Hdr.Padding2, 0u);

  // Decode GlobalData entries starting at offset 24 (after fixed header).
  uint64_t Offset = 24;
  bool FoundAddrOffsets = false;
  bool FoundAddrInfoOffsets = false;
  bool FoundStringTable = false;
  bool FoundFileTable = false;
  bool FoundFunctionInfo = false;
  bool FoundUUID = false;
  bool FoundEndOfList = false;

  while (Offset < Data.size()) {
    GlobalData GD = decodeGlobalDataEntry(Data, Offset, ByteOrder);
    EXPECT_EQ(GD.Padding, 0u);

    switch (GD.Type) {
    case GlobalInfoType::EndOfList:
      EXPECT_EQ(GD.FileOffset, 0u);
      EXPECT_EQ(GD.FileSize, 0u);
      FoundEndOfList = true;
      break;
    case GlobalInfoType::AddrOffsets:
      EXPECT_EQ(GD.FileSize,
                ExpectedNumAddresses * (uint64_t)ExpectedAddrOffSize);
      EXPECT_GT(GD.FileOffset, 0u);
      FoundAddrOffsets = true;
      break;
    case GlobalInfoType::AddrInfoOffsets:
      EXPECT_EQ(GD.FileSize, ExpectedNumAddresses * 4u); // AddrInfoOffSize=4
      EXPECT_GT(GD.FileOffset, 0u);
      FoundAddrInfoOffsets = true;
      break;
    case GlobalInfoType::StringTable:
      EXPECT_GT(GD.FileSize, 0u);
      EXPECT_GT(GD.FileOffset, 0u);
      FoundStringTable = true;
      break;
    case GlobalInfoType::FileTable:
      EXPECT_GT(GD.FileSize, 0u);
      EXPECT_GT(GD.FileOffset, 0u);
      FoundFileTable = true;
      break;
    case GlobalInfoType::FunctionInfo:
      EXPECT_GT(GD.FileSize, 0u);
      EXPECT_GT(GD.FileOffset, 0u);
      FoundFunctionInfo = true;
      break;
    case GlobalInfoType::UUID:
      EXPECT_EQ(GD.FileSize, sizeof(UUID));
      EXPECT_GT(GD.FileOffset, 0u);
      FoundUUID = true;
      break;
    }
    if (FoundEndOfList)
      break;
  }

  EXPECT_TRUE(FoundAddrOffsets);
  EXPECT_TRUE(FoundAddrInfoOffsets);
  EXPECT_TRUE(FoundStringTable);
  EXPECT_TRUE(FoundFileTable);
  EXPECT_TRUE(FoundFunctionInfo);
  EXPECT_TRUE(FoundEndOfList);
  EXPECT_EQ(FoundUUID, HasUUID);

  // Verify that all section data fits within the encoded buffer.
  Offset = 24;
  while (Offset < Data.size()) {
    GlobalData GD = decodeGlobalDataEntry(Data, Offset, ByteOrder);
    if (GD.Type == GlobalInfoType::EndOfList)
      break;
    EXPECT_LE(GD.FileOffset + GD.FileSize, Data.size())
        << "Section type " << static_cast<uint32_t>(GD.Type)
        << " extends beyond buffer";
  }
}

TEST(GSYMV2Test, TestHeaderAndGlobalDataLittle) {
  TestV2HeaderAndGlobalData(llvm::endianness::little, 0x1000,
                            /*ExpectedAddrOffSize=*/1,
                            /*ExpectedNumAddresses=*/2,
                            /*HasUUID=*/true);
}

TEST(GSYMV2Test, TestHeaderAndGlobalDataBig) {
  TestV2HeaderAndGlobalData(llvm::endianness::big, 0x1000,
                            /*ExpectedAddrOffSize=*/1,
                            /*ExpectedNumAddresses=*/2,
                            /*HasUUID=*/true);
}

TEST(GSYMV2Test, TestHeaderAndGlobalDataNoUUID) {
  TestV2HeaderAndGlobalData(llvm::endianness::little, 0x1000,
                            /*ExpectedAddrOffSize=*/1,
                            /*ExpectedNumAddresses=*/2,
                            /*HasUUID=*/false);
}

//===----------------------------------------------------------------------===//
// Address offset size tests
//===----------------------------------------------------------------------===//

static void TestV2AddrOffSize(uint64_t BaseAddr, uint64_t Func2Offset,
                              uint8_t ExpectedAddrOffSize) {
  GsymCreatorV2 GC;
  const uint32_t Func1Name = GC.insertString("foo");
  const uint32_t Func2Name = GC.insertString("bar");
  GC.addFunctionInfo(FunctionInfo(BaseAddr, 0x10, Func1Name));
  GC.addFunctionInfo(
      FunctionInfo(BaseAddr + Func2Offset, 0x10, Func2Name));
  OutputAggregator Null(nullptr);
  Error Err = GC.finalize(Null);
  ASSERT_FALSE(bool(Err));

  auto Result = encodeV2(GC, llvm::endianness::little);
  ASSERT_THAT_EXPECTED(Result, Succeeded());

  auto HdrOrErr = decodeHeaderV2(*Result, llvm::endianness::little);
  ASSERT_THAT_EXPECTED(HdrOrErr, Succeeded());
  EXPECT_EQ(HdrOrErr->AddrOffSize, ExpectedAddrOffSize);
}

TEST(GSYMV2Test, TestAddrOffSize1Byte) {
  TestV2AddrOffSize(0x1000, 0x20, 1);
}

TEST(GSYMV2Test, TestAddrOffSize2Byte) {
  TestV2AddrOffSize(0x1000, 0x200, 2);
}

TEST(GSYMV2Test, TestAddrOffSize4Byte) {
  TestV2AddrOffSize(0x1000, 0x20000, 4);
}

TEST(GSYMV2Test, TestAddrOffSize8Byte) {
  TestV2AddrOffSize(0x1000, 0x100000000ULL, 8);
}

//===----------------------------------------------------------------------===//
// AddrInfoOffsets verification
//===----------------------------------------------------------------------===//

TEST(GSYMV2Test, TestAddrInfoOffsetsPointToFunctionInfo) {
  // Verify that each AddrInfoOffset entry points to a valid location within
  // the FunctionInfo section.
  GsymCreatorV2 GC;
  const uint32_t Func1Name = GC.insertString("func_a");
  const uint32_t Func2Name = GC.insertString("func_b");
  const uint32_t Func3Name = GC.insertString("func_c");
  GC.addFunctionInfo(FunctionInfo(0x1000, 0x100, Func1Name));
  GC.addFunctionInfo(FunctionInfo(0x1100, 0x100, Func2Name));
  GC.addFunctionInfo(FunctionInfo(0x1200, 0x100, Func3Name));
  OutputAggregator Null(nullptr);
  Error Err = GC.finalize(Null);
  ASSERT_FALSE(bool(Err));

  auto Result = encodeV2(GC, llvm::endianness::little);
  ASSERT_THAT_EXPECTED(Result, Succeeded());
  StringRef Data = *Result;

  // Find the AddrInfoOffsets and FunctionInfo sections from GlobalData.
  uint64_t Offset = 24;
  uint64_t AIOffsetsOffset = 0, AIOffsetsSize = 0;
  uint64_t FIOffset = 0, FISize = 0;
  while (Offset < Data.size()) {
    GlobalData GD = decodeGlobalDataEntry(Data, Offset, llvm::endianness::little);
    if (GD.Type == GlobalInfoType::AddrInfoOffsets) {
      AIOffsetsOffset = GD.FileOffset;
      AIOffsetsSize = GD.FileSize;
    } else if (GD.Type == GlobalInfoType::FunctionInfo) {
      FIOffset = GD.FileOffset;
      FISize = GD.FileSize;
    } else if (GD.Type == GlobalInfoType::EndOfList) {
      break;
    }
  }
  ASSERT_GT(AIOffsetsOffset, 0u);
  ASSERT_GT(FIOffset, 0u);

  // Each AddrInfoOffset should point within [FIOffset, FIOffset + FISize).
  DataExtractor DE(Data, /*IsLittleEndian=*/true, 8);
  uint64_t AIOffset = AIOffsetsOffset;
  for (uint32_t I = 0; I < 3; ++I) {
    uint32_t FuncOffset = DE.getU32(&AIOffset);
    EXPECT_GE(FuncOffset, FIOffset)
        << "AddrInfoOffset[" << I << "] before FunctionInfo section";
    EXPECT_LT(FuncOffset, FIOffset + FISize)
        << "AddrInfoOffset[" << I << "] beyond FunctionInfo section";
  }

  // Offsets should be strictly increasing (sorted functions, no overlap).
  AIOffset = AIOffsetsOffset;
  uint32_t PrevOffset = DE.getU32(&AIOffset);
  for (uint32_t I = 1; I < 3; ++I) {
    uint32_t CurOffset = DE.getU32(&AIOffset);
    EXPECT_GT(CurOffset, PrevOffset)
        << "AddrInfoOffset[" << I << "] not strictly increasing";
    PrevOffset = CurOffset;
  }
}

//===----------------------------------------------------------------------===//
// UUID section verification
//===----------------------------------------------------------------------===//

TEST(GSYMV2Test, TestUUIDSection) {
  GsymCreatorV2 GC;
  const uint32_t Name = GC.insertString("main");
  GC.addFunctionInfo(FunctionInfo(0x1000, 0x100, Name));
  uint8_t UUID[] = {0xAA, 0xBB, 0xCC, 0xDD, 0x11, 0x22, 0x33, 0x44};
  GC.setUUID(UUID);
  OutputAggregator Null(nullptr);
  Error Err = GC.finalize(Null);
  ASSERT_FALSE(bool(Err));

  auto Result = encodeV2(GC, llvm::endianness::little);
  ASSERT_THAT_EXPECTED(Result, Succeeded());
  StringRef Data = *Result;

  // Find UUID section.
  uint64_t Offset = 24;
  uint64_t UUIDOffset = 0, UUIDSize = 0;
  while (Offset < Data.size()) {
    GlobalData GD = decodeGlobalDataEntry(Data, Offset, llvm::endianness::little);
    if (GD.Type == GlobalInfoType::UUID) {
      UUIDOffset = GD.FileOffset;
      UUIDSize = GD.FileSize;
    } else if (GD.Type == GlobalInfoType::EndOfList) {
      break;
    }
  }
  ASSERT_EQ(UUIDSize, sizeof(UUID));
  ASSERT_GT(UUIDOffset, 0u);

  // Verify the UUID bytes match.
  StringRef UUIDData = Data.substr(UUIDOffset, UUIDSize);
  EXPECT_EQ(UUIDData, StringRef(reinterpret_cast<const char *>(UUID),
                                sizeof(UUID)));
}
