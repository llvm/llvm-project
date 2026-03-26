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
#include "llvm/DebugInfo/GSYM/GsymReaderV2.h"
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

//===----------------------------------------------------------------------===//
// Reader V2 tests (without creator — hand-crafted binary)
//===----------------------------------------------------------------------===//

/// Helper to build a minimal valid V2 GSYM binary in native endianness.
/// Creates one function "main" at BaseAddr with size FuncSize.
static SmallString<512> buildMinimalV2Binary(uint64_t BaseAddr,
                                             uint32_t FuncSize) {
  SmallString<512> Str;
  raw_svector_ostream OS(Str);
  FileWriter FW(OS, llvm::endianness::native);

  // We'll build: header (24) + GlobalData entries (6 entries * 24 = 144) +
  // sections. Total GlobalData entries: AddrOffsets, AddrInfoOffsets,
  // StringTable, FileTable, FunctionInfo, EndOfList = 6.
  constexpr uint64_t HeaderSize = 24;
  constexpr uint64_t NumGlobalEntries = 6;
  constexpr uint64_t GlobalDataSize = NumGlobalEntries * 24;
  constexpr uint8_t AddrOffSize = 1;
  constexpr uint8_t AddrInfoOffSize = 4;
  constexpr uint32_t NumAddresses = 1;

  // Layout sections sequentially after header + GlobalData.
  uint64_t CurOffset = HeaderSize + GlobalDataSize;

  // AddrOffsets: 1 address * 1 byte.
  const uint64_t AddrOffsetsOff = CurOffset;
  const uint64_t AddrOffsetsSize = NumAddresses * AddrOffSize;
  CurOffset += AddrOffsetsSize;

  // Pad to 4-byte alignment for AddrInfoOffsets.
  CurOffset = llvm::alignTo(CurOffset, 4);
  const uint64_t AddrInfoOffsetsOff = CurOffset;
  const uint64_t AddrInfoOffsetsSize = NumAddresses * AddrInfoOffSize;
  CurOffset += AddrInfoOffsetsSize;

  // FileTable: 4 bytes (NumFiles=1) + 1 FileEntry (8 bytes) = 12.
  CurOffset = llvm::alignTo(CurOffset, 4);
  const uint64_t FileTableOff = CurOffset;
  const uint64_t FileTableSize = 4 + 8; // 1 file entry (dir=0, base=0).
  CurOffset += FileTableSize;

  // StringTable: "\0main\0" = 6 bytes.
  const uint64_t StringTableOff = CurOffset;
  const char StrTabData[] = "\0main";
  const uint64_t StringTableSize = sizeof(StrTabData); // includes trailing \0
  CurOffset += StringTableSize;

  // FunctionInfo: encode a minimal FunctionInfo.
  CurOffset = llvm::alignTo(CurOffset, 4);
  const uint64_t FuncInfoOff = CurOffset;
  // FunctionInfo encoding: uint32_t Size, uint32_t Name (strp offset).
  // "main" is at offset 1 in the string table.
  // Minimal FI: size (4 bytes) + name (4 bytes) = 8 bytes, no line table or
  // inline info (InfoType::EndOfList = 0 follows).
  // Actually FunctionInfo::encode writes: size, name, then info types.
  // Let's pre-encode a FunctionInfo to get exact bytes.
  SmallString<64> FIBuf;
  {
    raw_svector_ostream FIOS(FIBuf);
    FileWriter FIFW(FIOS, llvm::endianness::native);
    FunctionInfo FI(BaseAddr, FuncSize, /*Name=*/1); // "main" at strtab offset 1
    auto OffOrErr = FI.encode(FIFW);
    assert(OffOrErr && "FunctionInfo encode failed");
    (void)OffOrErr;
  }
  const uint64_t FuncInfoSize = FIBuf.size();
  const uint64_t AddrInfoOffset = FuncInfoOff; // Points to start of FI section.

  // Write header.
  FW.writeU32(GSYM_MAGIC);           // Magic
  FW.writeU16(GSYM_VERSION_2);       // Version
  FW.writeU16(0);                    // Padding
  FW.writeU64(BaseAddr);             // BaseAddress
  FW.writeU32(NumAddresses);         // NumAddresses
  FW.writeU8(AddrOffSize);           // AddrOffSize
  FW.writeU8(AddrInfoOffSize);       // AddrInfoOffSize
  FW.writeU8(4);                     // StrpSize
  FW.writeU8(0);                     // Padding2

  // GlobalData entries.
  auto writeGD = [&](GlobalInfoType Type, uint64_t Off, uint64_t Size) {
    FW.writeU32(static_cast<uint32_t>(Type));
    FW.writeU32(0); // Padding
    FW.writeU64(Off);
    FW.writeU64(Size);
  };
  writeGD(GlobalInfoType::AddrOffsets, AddrOffsetsOff, AddrOffsetsSize);
  writeGD(GlobalInfoType::AddrInfoOffsets, AddrInfoOffsetsOff, AddrInfoOffsetsSize);
  writeGD(GlobalInfoType::StringTable, StringTableOff, StringTableSize);
  writeGD(GlobalInfoType::FileTable, FileTableOff, FileTableSize);
  writeGD(GlobalInfoType::FunctionInfo, FuncInfoOff, FuncInfoSize);
  writeGD(GlobalInfoType::EndOfList, 0, 0);

  // AddrOffsets section.
  assert(FW.tell() == AddrOffsetsOff);
  FW.writeU8(0); // Offset from BaseAddr = 0 for first function.

  // Pad to AddrInfoOffsets.
  FW.alignTo(4);
  assert(FW.tell() == AddrInfoOffsetsOff);
  FW.writeU32(static_cast<uint32_t>(AddrInfoOffset));

  // FileTable.
  FW.alignTo(4);
  assert(FW.tell() == FileTableOff);
  FW.writeU32(1);  // NumFiles = 1
  FW.writeU32(0);  // File[0].Dir = 0
  FW.writeU32(0);  // File[0].Base = 0

  // StringTable.
  assert(FW.tell() == StringTableOff);
  FW.writeData(
      ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(StrTabData),
                        StringTableSize));

  // FunctionInfo.
  FW.alignTo(4);
  assert(FW.tell() == FuncInfoOff);
  FW.writeData(
      ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(FIBuf.data()),
                        FIBuf.size()));

  return Str;
}

TEST(GSYMV2Test, TestReaderV2ParseHandCrafted) {
  // Build a minimal V2 binary by hand and verify the reader can parse it.
  auto Bytes = buildMinimalV2Binary(0x1000, 0x100);
  auto GR = GsymReaderV2::copyBuffer(StringRef(Bytes.data(), Bytes.size()));
  ASSERT_THAT_EXPECTED(GR, Succeeded());

  const HeaderV2 &Hdr = GR->getHeader();
  EXPECT_EQ(Hdr.Magic, GSYM_MAGIC);
  EXPECT_EQ(Hdr.Version, GSYM_VERSION_2);
  EXPECT_EQ(Hdr.BaseAddress, 0x1000u);
  EXPECT_EQ(Hdr.NumAddresses, 1u);
  EXPECT_EQ(Hdr.AddrOffSize, 1u);
  EXPECT_EQ(Hdr.AddrInfoOffSize, 4u);
  EXPECT_EQ(GR->getNumAddresses(), 1u);

  // Verify address lookup.
  auto Addr = GR->getAddress(0);
  ASSERT_TRUE(Addr.has_value());
  EXPECT_EQ(*Addr, 0x1000u);

  // Verify getString.
  EXPECT_EQ(GR->getString(1), "main");

  // Verify getFile (index 0 is the empty file entry).
  auto FE = GR->getFile(0);
  ASSERT_TRUE(FE.has_value());
  EXPECT_EQ(FE->Dir, 0u);
  EXPECT_EQ(FE->Base, 0u);
}

TEST(GSYMV2Test, TestReaderV2GetFunctionInfoHandCrafted) {
  auto Bytes = buildMinimalV2Binary(0x1000, 0x100);
  auto GR = GsymReaderV2::copyBuffer(StringRef(Bytes.data(), Bytes.size()));
  ASSERT_THAT_EXPECTED(GR, Succeeded());

  // getFunctionInfo should decode the function at 0x1000.
  auto FI = GR->getFunctionInfo(0x1000);
  ASSERT_THAT_EXPECTED(FI, Succeeded());
  EXPECT_EQ(FI->Range, AddressRange(0x1000, 0x1100));
  EXPECT_EQ(FI->Name, 1u); // "main" at strtab offset 1
  EXPECT_EQ(GR->getString(FI->Name), "main");

  // Address within the function range should also work.
  auto FI2 = GR->getFunctionInfo(0x1050);
  ASSERT_THAT_EXPECTED(FI2, Succeeded());
  EXPECT_EQ(FI2->Range, AddressRange(0x1000, 0x1100));

  // Address outside range should fail.
  auto FI3 = GR->getFunctionInfo(0x2000);
  EXPECT_THAT_EXPECTED(FI3, Failed());
}

TEST(GSYMV2Test, TestReaderV2LookupHandCrafted) {
  auto Bytes = buildMinimalV2Binary(0x1000, 0x100);
  auto GR = GsymReaderV2::copyBuffer(StringRef(Bytes.data(), Bytes.size()));
  ASSERT_THAT_EXPECTED(GR, Succeeded());

  // lookup should return a LookupResult.
  auto LR = GR->lookup(0x1000);
  ASSERT_THAT_EXPECTED(LR, Succeeded());
  EXPECT_EQ(LR->FuncName, "main");
  EXPECT_EQ(LR->FuncRange, AddressRange(0x1000, 0x1100));

  // lookup within range.
  auto LR2 = GR->lookup(0x1080);
  ASSERT_THAT_EXPECTED(LR2, Succeeded());
  EXPECT_EQ(LR2->FuncName, "main");

  // lookup outside range.
  auto LR3 = GR->lookup(0x2000);
  EXPECT_THAT_EXPECTED(LR3, Failed());
}

TEST(GSYMV2Test, TestReaderV2InvalidMagic) {
  // Create a buffer with invalid magic.
  char Buf[24] = {};
  Buf[0] = 'X'; // Bad magic.
  auto GR = GsymReaderV2::copyBuffer(StringRef(Buf, sizeof(Buf)));
  EXPECT_THAT_EXPECTED(GR, Failed());
}

TEST(GSYMV2Test, TestReaderV2TooSmall) {
  // Buffer smaller than header.
  char Buf[10] = {};
  auto GR = GsymReaderV2::copyBuffer(StringRef(Buf, sizeof(Buf)));
  EXPECT_THAT_EXPECTED(GR, Failed());
}

//===----------------------------------------------------------------------===//
// Round-trip tests: Creator V2 -> Reader V2
//===----------------------------------------------------------------------===//

/// Helper to create, finalize, encode with GsymCreatorV2, then decode with
/// GsymReaderV2 and return the reader.
static Expected<GsymReaderV2> createAndReadV2(GsymCreatorV2 &GC) {
  OutputAggregator Null(nullptr);
  if (auto Err = GC.finalize(Null))
    return std::move(Err);

  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  FileWriter FW(OutStrm, llvm::endianness::native);
  if (auto Err = GC.encode(FW))
    return std::move(Err);

  return GsymReaderV2::copyBuffer(OutStrm.str());
}

TEST(GSYMV2Test, TestRoundTripSingleFunction) {
  GsymCreatorV2 GC;
  const uint32_t Name = GC.insertString("hello");
  GC.addFunctionInfo(FunctionInfo(0x2000, 0x200, Name));

  auto GR = createAndReadV2(GC);
  ASSERT_THAT_EXPECTED(GR, Succeeded());

  EXPECT_EQ(GR->getNumAddresses(), 1u);
  EXPECT_EQ(GR->getHeader().BaseAddress, 0x2000u);

  auto FI = GR->getFunctionInfo(0x2000);
  ASSERT_THAT_EXPECTED(FI, Succeeded());
  EXPECT_EQ(FI->Range, AddressRange(0x2000, 0x2200));
  EXPECT_EQ(GR->getString(FI->Name), "hello");
}

TEST(GSYMV2Test, TestRoundTripMultipleFunctions) {
  GsymCreatorV2 GC;
  const uint32_t Name1 = GC.insertString("alpha");
  const uint32_t Name2 = GC.insertString("beta");
  const uint32_t Name3 = GC.insertString("gamma");
  GC.addFunctionInfo(FunctionInfo(0x1000, 0x100, Name1));
  GC.addFunctionInfo(FunctionInfo(0x1100, 0x100, Name2));
  GC.addFunctionInfo(FunctionInfo(0x1200, 0x100, Name3));

  auto GR = createAndReadV2(GC);
  ASSERT_THAT_EXPECTED(GR, Succeeded());

  EXPECT_EQ(GR->getNumAddresses(), 3u);
  EXPECT_EQ(GR->getHeader().BaseAddress, 0x1000u);

  // Verify each function can be looked up by address.
  auto FI1 = GR->getFunctionInfo(0x1000);
  ASSERT_THAT_EXPECTED(FI1, Succeeded());
  EXPECT_EQ(GR->getString(FI1->Name), "alpha");
  EXPECT_EQ(FI1->Range, AddressRange(0x1000, 0x1100));

  auto FI2 = GR->getFunctionInfo(0x1100);
  ASSERT_THAT_EXPECTED(FI2, Succeeded());
  EXPECT_EQ(GR->getString(FI2->Name), "beta");
  EXPECT_EQ(FI2->Range, AddressRange(0x1100, 0x1200));

  auto FI3 = GR->getFunctionInfo(0x1200);
  ASSERT_THAT_EXPECTED(FI3, Succeeded());
  EXPECT_EQ(GR->getString(FI3->Name), "gamma");
  EXPECT_EQ(FI3->Range, AddressRange(0x1200, 0x1300));

  // Lookup in the middle of a function.
  auto FI2Mid = GR->getFunctionInfo(0x1150);
  ASSERT_THAT_EXPECTED(FI2Mid, Succeeded());
  EXPECT_EQ(GR->getString(FI2Mid->Name), "beta");
}

TEST(GSYMV2Test, TestRoundTripLookup) {
  GsymCreatorV2 GC;
  const uint32_t Name1 = GC.insertString("start");
  const uint32_t Name2 = GC.insertString("end");
  GC.addFunctionInfo(FunctionInfo(0x5000, 0x500, Name1));
  GC.addFunctionInfo(FunctionInfo(0x5500, 0x500, Name2));

  auto GR = createAndReadV2(GC);
  ASSERT_THAT_EXPECTED(GR, Succeeded());

  // Lookup first function.
  auto LR1 = GR->lookup(0x5000);
  ASSERT_THAT_EXPECTED(LR1, Succeeded());
  EXPECT_EQ(LR1->FuncName, "start");
  EXPECT_EQ(LR1->FuncRange, AddressRange(0x5000, 0x5500));

  // Lookup second function.
  auto LR2 = GR->lookup(0x5500);
  ASSERT_THAT_EXPECTED(LR2, Succeeded());
  EXPECT_EQ(LR2->FuncName, "end");

  // Lookup within first function.
  auto LR3 = GR->lookup(0x5100);
  ASSERT_THAT_EXPECTED(LR3, Succeeded());
  EXPECT_EQ(LR3->FuncName, "start");

  // Lookup outside all functions.
  auto LR4 = GR->lookup(0x6000);
  EXPECT_THAT_EXPECTED(LR4, Failed());
}

TEST(GSYMV2Test, TestRoundTripGetFunctionInfoAtIndex) {
  GsymCreatorV2 GC;
  const uint32_t Name1 = GC.insertString("func_x");
  const uint32_t Name2 = GC.insertString("func_y");
  GC.addFunctionInfo(FunctionInfo(0x3000, 0x100, Name1));
  GC.addFunctionInfo(FunctionInfo(0x3100, 0x100, Name2));

  auto GR = createAndReadV2(GC);
  ASSERT_THAT_EXPECTED(GR, Succeeded());

  // Access by index.
  auto FI0 = GR->getFunctionInfoAtIndex(0);
  ASSERT_THAT_EXPECTED(FI0, Succeeded());
  EXPECT_EQ(GR->getString(FI0->Name), "func_x");

  auto FI1 = GR->getFunctionInfoAtIndex(1);
  ASSERT_THAT_EXPECTED(FI1, Succeeded());
  EXPECT_EQ(GR->getString(FI1->Name), "func_y");

  // Out of bounds index.
  auto FI2 = GR->getFunctionInfoAtIndex(2);
  EXPECT_THAT_EXPECTED(FI2, Failed());
}

TEST(GSYMV2Test, TestRoundTripAddressTable) {
  GsymCreatorV2 GC;
  const uint32_t N1 = GC.insertString("a");
  const uint32_t N2 = GC.insertString("b");
  const uint32_t N3 = GC.insertString("c");
  GC.addFunctionInfo(FunctionInfo(0x8000, 0x10, N1));
  GC.addFunctionInfo(FunctionInfo(0x8020, 0x10, N2));
  GC.addFunctionInfo(FunctionInfo(0x8040, 0x10, N3));

  auto GR = createAndReadV2(GC);
  ASSERT_THAT_EXPECTED(GR, Succeeded());

  // Verify addresses via getAddress.
  EXPECT_EQ(GR->getAddress(0), std::optional<uint64_t>(0x8000u));
  EXPECT_EQ(GR->getAddress(1), std::optional<uint64_t>(0x8020u));
  EXPECT_EQ(GR->getAddress(2), std::optional<uint64_t>(0x8040u));
  EXPECT_EQ(GR->getAddress(3), std::nullopt); // Out of bounds.
}

TEST(GSYMV2Test, TestRoundTripLargeAddressOffsets) {
  // Test with address offsets that require 4-byte AddrOffSize.
  GsymCreatorV2 GC;
  const uint32_t N1 = GC.insertString("near");
  const uint32_t N2 = GC.insertString("far");
  GC.addFunctionInfo(FunctionInfo(0x1000, 0x10, N1));
  GC.addFunctionInfo(FunctionInfo(0x1000 + 0x20000, 0x10, N2));

  auto GR = createAndReadV2(GC);
  ASSERT_THAT_EXPECTED(GR, Succeeded());

  EXPECT_EQ(GR->getHeader().AddrOffSize, 4u);
  EXPECT_EQ(GR->getNumAddresses(), 2u);

  auto FI1 = GR->getFunctionInfo(0x1000);
  ASSERT_THAT_EXPECTED(FI1, Succeeded());
  EXPECT_EQ(GR->getString(FI1->Name), "near");

  auto FI2 = GR->getFunctionInfo(0x1000 + 0x20000);
  ASSERT_THAT_EXPECTED(FI2, Succeeded());
  EXPECT_EQ(GR->getString(FI2->Name), "far");
}
