//===- llvm/unittest/DebugInfo/GSYMTest.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/GSYM/DwarfTransformer.h"
#include "llvm/DebugInfo/GSYM/ExtractRanges.h"
#include "llvm/DebugInfo/GSYM/FileEntry.h"
#include "llvm/DebugInfo/GSYM/FileWriter.h"
#include "llvm/DebugInfo/GSYM/FunctionInfo.h"
#include "llvm/DebugInfo/GSYM/GsymCreatorV1.h"
#include "llvm/DebugInfo/GSYM/GsymCreatorV2.h"
#include "llvm/DebugInfo/GSYM/GsymDataExtractor.h"
#include "llvm/DebugInfo/GSYM/GsymReaderV1.h"
#include "llvm/DebugInfo/GSYM/GsymReaderV2.h"
#include "llvm/DebugInfo/GSYM/Header.h"
#include "llvm/DebugInfo/GSYM/HeaderV2.h"
#include "llvm/DebugInfo/GSYM/InlineInfo.h"
#include "llvm/DebugInfo/GSYM/OutputAggregator.h"
#include "llvm/DebugInfo/GSYM/StringTable.h"
#include "llvm/ObjectYAML/DWARFEmitter.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Testing/Support/Error.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <string>

using namespace llvm;
using namespace gsym;

void checkError(ArrayRef<std::string> ExpectedMsgs, Error Err) {
  ASSERT_TRUE(bool(Err));
  size_t WhichMsg = 0;
  Error Remaining =
      handleErrors(std::move(Err), [&](const ErrorInfoBase &Actual) {
        ASSERT_LT(WhichMsg, ExpectedMsgs.size());
        // Use .str(), because googletest doesn't visualise a StringRef
        // properly.
        EXPECT_EQ(Actual.message(), ExpectedMsgs[WhichMsg++]);
      });
  EXPECT_EQ(WhichMsg, ExpectedMsgs.size());
  EXPECT_FALSE(Remaining);
}

void checkError(std::string ExpectedMsg, Error Err) {
  checkError(ArrayRef<std::string>{ExpectedMsg}, std::move(Err));
}
TEST(GSYMTest, TestFileEntry) {
  // Make sure default constructed GSYM FileEntry has zeroes in the
  // directory and basename string table indexes.
  FileEntry empty1;
  FileEntry empty2;
  EXPECT_EQ(empty1.Dir, 0u);
  EXPECT_EQ(empty1.Base, 0u);
  // Verify 64-bit values can be stored and retrieved
  FileEntry large(0x1'0000'0001, 0x2'0000'0002);
  EXPECT_EQ(large.Dir, 0x1'0000'0001ull);
  EXPECT_EQ(large.Base, 0x2'0000'0002ull);
  // Verify equality operator works
  FileEntry a1(10, 30);
  FileEntry a2(10, 30);
  FileEntry b(10, 40);
  EXPECT_EQ(empty1, empty2);
  EXPECT_EQ(a1, a2);
  EXPECT_NE(a1, b);
  EXPECT_NE(a1, empty1);
  // Test we can use llvm::gsym::FileEntry in llvm::DenseMap.
  DenseMap<FileEntry, uint32_t> EntryToIndex;
  constexpr uint32_t Index1 = 1;
  constexpr uint32_t Index2 = 1;
  auto R = EntryToIndex.insert(std::make_pair(a1, Index1));
  EXPECT_TRUE(R.second);
  EXPECT_EQ(R.first->second, Index1);
  R = EntryToIndex.insert(std::make_pair(a1, Index1));
  EXPECT_FALSE(R.second);
  EXPECT_EQ(R.first->second, Index1);
  R = EntryToIndex.insert(std::make_pair(b, Index2));
  EXPECT_TRUE(R.second);
  EXPECT_EQ(R.first->second, Index2);
  R = EntryToIndex.insert(std::make_pair(a1, Index2));
  EXPECT_FALSE(R.second);
  EXPECT_EQ(R.first->second, Index2);
}

template <typename StrpT> static void TestFunctionInfoImpl(StrpT NameOffset) {
  // Test GSYM FunctionInfo structs and functionality.
  FunctionInfo invalid;
  EXPECT_FALSE(invalid.isValid());
  EXPECT_FALSE(invalid.hasRichInfo());
  const uint64_t StartAddr = 0x1000;
  const uint64_t EndAddr = 0x1100;
  const uint64_t Size = EndAddr - StartAddr;
  FunctionInfo FI(StartAddr, Size, NameOffset);
  EXPECT_TRUE(FI.isValid());
  EXPECT_FALSE(FI.hasRichInfo());
  EXPECT_EQ(FI.startAddress(), StartAddr);
  EXPECT_EQ(FI.endAddress(), EndAddr);
  EXPECT_EQ(FI.size(), Size);
  EXPECT_EQ(FI.Name, NameOffset);
  const uint32_t FileIdx = 1;
  const uint32_t Line = 12;
  FI.OptLineTable = LineTable();
  FI.OptLineTable->push(LineEntry(StartAddr,FileIdx,Line));
  EXPECT_TRUE(FI.hasRichInfo());
  FI.clear();
  EXPECT_FALSE(FI.isValid());
  EXPECT_FALSE(FI.hasRichInfo());

  FunctionInfo A1(0x1000, 0x100, NameOffset);
  FunctionInfo A2(0x1000, 0x100, NameOffset);
  FunctionInfo B;
  // Check == operator
  EXPECT_EQ(A1, A2);
  // Make sure things are not equal if they only differ by start address.
  B = A2;
  B.Range = {0x1001, B.endAddress()};
  EXPECT_NE(B, A2);
  // Make sure things are not equal if they only differ by size.
  B = A2;
  B.Range = {B.startAddress(), B.startAddress() + 0x101};
  EXPECT_NE(B, A2);
  // Make sure things are not equal if they only differ by name.
  B = A2;
  B.Name = NameOffset + 30;
  EXPECT_NE(B, A2);
  // Check < operator.
  // Check less than where address differs.
  B = A2;
  B.Range = {A2.startAddress() + 0x1000, A2.endAddress() + 0x1000};
  EXPECT_LT(A1, B);

  // We use the < operator to take a variety of different FunctionInfo
  // structs from a variety of sources: symtab, debug info, runtime info
  // and we sort them and want the sorting to allow us to quickly get the
  // best version of a function info.
  FunctionInfo FISymtab(StartAddr, Size, NameOffset);
  FunctionInfo FIWithLines(StartAddr, Size, NameOffset);
  FIWithLines.OptLineTable = LineTable();
  FIWithLines.OptLineTable->push(LineEntry(StartAddr,FileIdx,Line));
  // Test that a FunctionInfo with just a name and size is less than one
  // that has name, size and any number of line table entries
  EXPECT_LT(FISymtab, FIWithLines);

  // Test that if we have a function info without inline info and one with
  // that the one without inline info is less than the one with.
  FunctionInfo FIWithInlines = FISymtab;
  FIWithInlines.Inline = InlineInfo();
  FIWithInlines.Inline->Ranges.insert(
      AddressRange(StartAddr, StartAddr + 0x10));
  EXPECT_LT(FISymtab, FIWithInlines);

  // Test that if we have a function info with inline entries and one more
  // inline entries that the one with fewer inline functins is less than the
  // one with more.
  FunctionInfo FIWithMoreInlines = FIWithInlines;
  FIWithMoreInlines.Inline->Children.push_back(InlineInfo());
  EXPECT_LT(FIWithInlines, FIWithMoreInlines);

  FunctionInfo FIWithLinesAndInline = FIWithLines;
  FIWithLinesAndInline.Inline = InlineInfo();
  FIWithLinesAndInline.Inline->Ranges.insert(
      AddressRange(StartAddr, StartAddr + 0x10));
  // Test that a FunctionInfo with name, size, and line entries is less than
  // the same one with valid inline info
  EXPECT_LT(FIWithLines, FIWithLinesAndInline);

  // Test if we have an entry with lines and one with more lines for the same
  // range, the ones with more lines is greater than the one with less.
  FunctionInfo FIWithMoreLines = FIWithLines;
  FIWithMoreLines.OptLineTable->push(LineEntry(StartAddr,FileIdx,Line+5));
  EXPECT_LT(FIWithLines, FIWithMoreLines);

  // Test that if we have the same number of lines we compare the line entries
  // in the FunctionInfo.OptLineTable.Lines vector.
  FunctionInfo FIWithLinesWithHigherAddress = FIWithLines;
  FIWithLinesWithHigherAddress.OptLineTable->get(0).Addr += 0x10;
  EXPECT_LT(FIWithLines, FIWithLinesWithHigherAddress);
}

TEST(GSYMTest, TestFunctionInfo) {
  TestFunctionInfoImpl<uint32_t>(30);
  TestFunctionInfoImpl<uint64_t>(0x100000030);
}

template <typename StrpT>
static void TestFunctionInfoDecodeError(llvm::endianness ByteOrder,
                                        StringRef Bytes,
                                        const uint64_t BaseAddr,
                                        std::string ExpectedErrorMsg) {
  GsymDataExtractor Data(Bytes, ByteOrder == llvm::endianness::little,
                         sizeof(StrpT));
  llvm::Expected<FunctionInfo> Decoded = FunctionInfo::decode(Data, BaseAddr);
  // Make sure decoding fails.
  ASSERT_FALSE((bool)Decoded);
  // Make sure decoded object is the same as the one we encoded.
  checkError(ExpectedErrorMsg, Decoded.takeError());
}

template <typename StrpT> static void TestFunctionInfoDecodeErrors() {
  // Test decoding FunctionInfo objects that ensure we report an appropriate
  // error message.
  const llvm::endianness ByteOrder = llvm::endianness::little;
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  FileWriter FW(OutStrm, ByteOrder);
  FW.setStringOffsetSize(sizeof(StrpT));
  const uint64_t BaseAddr = 0x100;
  // FunctionInfo layout: Size(4) + Name(sizeof(StrpT)) + InfoType(4) + ...
  constexpr uint64_t NameOff = 4;
  constexpr uint64_t PostNameOff = NameOff + sizeof(StrpT);
  constexpr uint64_t PostInfoTypeOff = PostNameOff + 4;
  auto Hex = [](uint64_t V) { return llvm::formatv("0x{0:x-8}", V).str(); };
  TestFunctionInfoDecodeError<StrpT>(ByteOrder, OutStrm.str(), BaseAddr,
                                     "0x00000000: missing FunctionInfo Size");
  FW.writeU32(0x100); // Function size.
  TestFunctionInfoDecodeError<StrpT>(ByteOrder, OutStrm.str(), BaseAddr,
                                     Hex(NameOff) +
                                         ": missing FunctionInfo Name");
  // Write out an invalid Name string table offset of zero.
  switch (sizeof(StrpT)) {
  case 4:
    FW.writeU32(0);
    break;
  case 8:
    FW.writeU64(0);
    break;
  default:
    FAIL() << "unsupported StrpT size: " << sizeof(StrpT);
  }
  TestFunctionInfoDecodeError<StrpT>(
      ByteOrder, OutStrm.str(), BaseAddr,
      Hex(NameOff) + ": invalid FunctionInfo Name value 0x0");
  // Modify the Name to be 0x00000001, which is a valid value.
  // fixup32 works for both sizes in little-endian (sets low 4 bytes).
  FW.fixup32(0x00000001, NameOff);
  TestFunctionInfoDecodeError<StrpT>(
      ByteOrder, OutStrm.str(), BaseAddr,
      Hex(PostNameOff) + ": missing FunctionInfo InfoType value");
  auto FixupOffset = FW.tell();
  FW.writeU32(1); // InfoType::LineTableInfo.
  TestFunctionInfoDecodeError<StrpT>(
      ByteOrder, OutStrm.str(), BaseAddr,
      Hex(PostInfoTypeOff) + ": missing FunctionInfo InfoType length");
  FW.fixup32(7, FixupOffset); // Write an invalid InfoType enumeration value
  FW.writeU32(0); // LineTableInfo InfoType data length.
  TestFunctionInfoDecodeError<StrpT>(ByteOrder, OutStrm.str(), BaseAddr,
                                     Hex(PostNameOff) +
                                         ": unsupported InfoType 7");
}

TEST(GSYMTest, TestFunctionInfoDecodeErrors) {
  TestFunctionInfoDecodeErrors<uint32_t>();
  TestFunctionInfoDecodeErrors<uint64_t>();
}

template <typename StrpT>
static void TestFunctionInfoEncodeError(llvm::endianness ByteOrder,
                                        const FunctionInfo &FI,
                                        std::string ExpectedErrorMsg) {
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  FileWriter FW(OutStrm, ByteOrder);
  FW.setStringOffsetSize(sizeof(StrpT));
  Expected<uint64_t> ExpectedOffset = FI.encode(FW);
  ASSERT_FALSE(ExpectedOffset);
  checkError(ExpectedErrorMsg, ExpectedOffset.takeError());
}

template <typename StrpT> static void TestFunctionInfoEncodeErrors() {
  const uint64_t FuncAddr = 0x1000;
  const uint64_t FuncSize = 0x100;
  const StrpT InvalidName = 0;
  const StrpT ValidName = 1;
  FunctionInfo InvalidNameFI(FuncAddr, FuncSize, InvalidName);
  TestFunctionInfoEncodeError<StrpT>(
      llvm::endianness::little, InvalidNameFI,
      "attempted to encode invalid FunctionInfo object");

  FunctionInfo InvalidLineTableFI(FuncAddr, FuncSize, ValidName);
  // Empty line tables are not valid. Verify if the encoding of anything
  // in our line table fails, that we see get the error propagated.
  InvalidLineTableFI.OptLineTable = LineTable();
  TestFunctionInfoEncodeError<StrpT>(
      llvm::endianness::little, InvalidLineTableFI,
      "attempted to encode invalid LineTable object");

  FunctionInfo InvalidInlineInfoFI(FuncAddr, FuncSize, ValidName);
  // Empty line tables are not valid. Verify if the encoding of anything
  // in our line table fails, that we see get the error propagated.
  InvalidInlineInfoFI.Inline = InlineInfo();
  TestFunctionInfoEncodeError<StrpT>(
      llvm::endianness::little, InvalidInlineInfoFI,
      "attempted to encode invalid InlineInfo object");
}

TEST(GSYMTest, TestFunctionInfoEncodeErrors) {
  TestFunctionInfoEncodeErrors<uint32_t>();
  TestFunctionInfoEncodeErrors<uint64_t>();
}

template <typename StrpT>
static void TestFunctionInfoEncodeDecode(llvm::endianness ByteOrder,
                                         const FunctionInfo &FI) {
  // Test encoding and decoding FunctionInfo objects.
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  FileWriter FW(OutStrm, ByteOrder);
  FW.setStringOffsetSize(sizeof(StrpT));
  llvm::Expected<uint64_t> ExpectedOffset = FI.encode(FW);
  ASSERT_TRUE(bool(ExpectedOffset));
  // Verify we got the encoded offset back from the encode function.
  ASSERT_EQ(ExpectedOffset.get(), 0ULL);
  std::string Bytes(OutStrm.str());
  GsymDataExtractor Data(Bytes, ByteOrder == llvm::endianness::little,
                         sizeof(StrpT));
  llvm::Expected<FunctionInfo> Decoded =
      FunctionInfo::decode(Data, FI.Range.start());
  // Make sure decoding succeeded.
  ASSERT_TRUE((bool)Decoded);
  // Make sure decoded object is the same as the one we encoded.
  EXPECT_EQ(FI, Decoded.get());
}

static void AddLines(uint64_t FuncAddr, uint32_t FileIdx, FunctionInfo &FI) {
    FI.OptLineTable = LineTable();
    LineEntry Line0(FuncAddr + 0x000, FileIdx, 10);
    LineEntry Line1(FuncAddr + 0x010, FileIdx, 11);
    LineEntry Line2(FuncAddr + 0x100, FileIdx, 1000);
    FI.OptLineTable->push(Line0);
    FI.OptLineTable->push(Line1);
    FI.OptLineTable->push(Line2);
}

template <typename StrpT>
static void AddInline(uint64_t FuncAddr, uint64_t FuncSize, FunctionInfo &FI,
                      StrpT InlineName) {
  FI.Inline = InlineInfo();
  FI.Inline->Ranges.insert(AddressRange(FuncAddr, FuncAddr + FuncSize));
  InlineInfo Inline1;
  Inline1.Ranges.insert(AddressRange(FuncAddr + 0x10, FuncAddr + 0x30));
  Inline1.Name = InlineName;
  Inline1.CallFile = 1;
  Inline1.CallLine = 11;
  FI.Inline->Children.push_back(Inline1);
}

template <typename StrpT>
static void TestFunctionInfoEncoding(StrpT FuncName, StrpT InlineName) {
  constexpr uint64_t FuncAddr = 0x1000;
  constexpr uint64_t FuncSize = 0x100;
  constexpr uint32_t FileIdx = 1;
  // Make sure that we can encode and decode a FunctionInfo with no line table
  // or inline info.
  FunctionInfo FI(FuncAddr, FuncSize, FuncName);
  TestFunctionInfoEncodeDecode<StrpT>(llvm::endianness::little, FI);
  TestFunctionInfoEncodeDecode<StrpT>(llvm::endianness::big, FI);

  // Make sure that we can encode and decode a FunctionInfo with a line table
  // and no inline info.
  FunctionInfo FILines(FuncAddr, FuncSize, FuncName);
  AddLines(FuncAddr, FileIdx, FILines);
  TestFunctionInfoEncodeDecode<StrpT>(llvm::endianness::little, FILines);
  TestFunctionInfoEncodeDecode<StrpT>(llvm::endianness::big, FILines);

  // Make sure that we can encode and decode a FunctionInfo with no line table
  // and with inline info.
  FunctionInfo FIInline(FuncAddr, FuncSize, FuncName);
  AddInline(FuncAddr, FuncSize, FIInline, InlineName);
  TestFunctionInfoEncodeDecode<StrpT>(llvm::endianness::little, FIInline);
  TestFunctionInfoEncodeDecode<StrpT>(llvm::endianness::big, FIInline);

  // Make sure that we can encode and decode a FunctionInfo with no line table
  // and with inline info.
  FunctionInfo FIBoth(FuncAddr, FuncSize, FuncName);
  AddLines(FuncAddr, FileIdx, FIBoth);
  AddInline(FuncAddr, FuncSize, FIBoth, InlineName);
  TestFunctionInfoEncodeDecode<StrpT>(llvm::endianness::little, FIBoth);
  TestFunctionInfoEncodeDecode<StrpT>(llvm::endianness::big, FIBoth);
}

TEST(GSYMTest, TestFunctionInfoEncoding) {
  TestFunctionInfoEncoding<uint32_t>(1, 1);
  TestFunctionInfoEncoding<uint64_t>(0x1'0000'0001, 0x1'0000'0002);
}

template <typename StrpT>
static void TestInlineInfoEncodeDecode(llvm::endianness ByteOrder,
                                       const InlineInfo &Inline) {
  // Test encoding and decoding InlineInfo objects
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  FileWriter FW(OutStrm, ByteOrder);
  FW.setStringOffsetSize(sizeof(StrpT));
  const uint64_t BaseAddr = Inline.Ranges[0].start();
  llvm::Error Err = Inline.encode(FW, BaseAddr);
  ASSERT_FALSE(Err);
  std::string Bytes(OutStrm.str());
  GsymDataExtractor Data(Bytes, ByteOrder == llvm::endianness::little,
                         sizeof(StrpT));
  llvm::Expected<InlineInfo> Decoded = InlineInfo::decode(Data, BaseAddr);
  // Make sure decoding succeeded.
  ASSERT_TRUE((bool)Decoded);
  // Make sure decoded object is the same as the one we encoded.
  EXPECT_EQ(Inline, Decoded.get());
}

template <typename StrpT>
static void TestInlineInfoDecodeError(llvm::endianness ByteOrder,
                                      StringRef Bytes, const uint64_t BaseAddr,
                                      std::string ExpectedErrorMsg) {
  GsymDataExtractor Data(Bytes, ByteOrder == llvm::endianness::little,
                         sizeof(StrpT));
  llvm::Expected<InlineInfo> Decoded = InlineInfo::decode(Data, BaseAddr);
  // Make sure decoding fails.
  ASSERT_FALSE((bool)Decoded);
  // Make sure decoded object is the same as the one we encoded.
  checkError(ExpectedErrorMsg, Decoded.takeError());
}

template <typename StrpT>
static void TestInlineInfoEncodeError(llvm::endianness ByteOrder,
                                      const InlineInfo &Inline,
                                      std::string ExpectedErrorMsg) {
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  FileWriter FW(OutStrm, ByteOrder);
  FW.setStringOffsetSize(sizeof(StrpT));
  const uint64_t BaseAddr =
      Inline.Ranges.empty() ? 0 : Inline.Ranges[0].start();
  llvm::Error Err = Inline.encode(FW, BaseAddr);
  checkError(ExpectedErrorMsg, std::move(Err));
}

template <typename StrpT>
static void TestInlineInfoImpl(StrpT Name1, StrpT Name2, StrpT Name3) {
  // Test InlineInfo structs.
  InlineInfo II;
  EXPECT_FALSE(II.isValid());
  II.Ranges.insert(AddressRange(0x1000, 0x2000));
  // Make sure InlineInfo in valid with just an address range since
  // top level InlineInfo objects have ranges with no name, call file
  // or call line
  EXPECT_TRUE(II.isValid());
  // Make sure InlineInfo isn't after being cleared.
  II.clear();
  EXPECT_FALSE(II.isValid());

  // Create an InlineInfo that contains the following data. The
  // indentation of the address range indicates the parent child
  // relationships of the InlineInfo objects:
  //
  // Variable    Range and values
  // =========== ====================================================
  // Root        [0x100-0x200) (no name, file, or line)
  // Inline1       [0x150-0x160) Name = Name1, File = 1, Line = 11
  // Inline1Sub1     [0x152-0x155) Name = Name2, File = 2, Line = 22
  // Inline1Sub2     [0x157-0x158) Name = Name3, File = 3, Line = 33
  InlineInfo Root;
  Root.Ranges.insert(AddressRange(0x100, 0x200));
  InlineInfo Inline1;
  Inline1.Ranges.insert(AddressRange(0x150, 0x160));
  Inline1.Name = Name1;
  Inline1.CallFile = 1;
  Inline1.CallLine = 11;
  InlineInfo Inline1Sub1;
  Inline1Sub1.Ranges.insert(AddressRange(0x152, 0x155));
  Inline1Sub1.Name = Name2;
  Inline1Sub1.CallFile = 2;
  Inline1Sub1.CallLine = 22;
  InlineInfo Inline1Sub2;
  Inline1Sub2.Ranges.insert(AddressRange(0x157, 0x158));
  Inline1Sub2.Name = Name3;
  Inline1Sub2.CallFile = 3;
  Inline1Sub2.CallLine = 33;
  Inline1.Children.push_back(Inline1Sub1);
  Inline1.Children.push_back(Inline1Sub2);
  Root.Children.push_back(Inline1);

  // Make sure an address that is out of range won't match
  EXPECT_FALSE(Root.getInlineStack(0x50));

  // Verify that we get no inline stacks for addresses out of [0x100-0x200)
  EXPECT_FALSE(Root.getInlineStack(Root.Ranges[0].start() - 1));
  EXPECT_FALSE(Root.getInlineStack(Root.Ranges[0].end()));

  // Verify we get no inline stack entries for addresses that are in
  // [0x100-0x200) but not in [0x150-0x160)
  EXPECT_FALSE(Root.getInlineStack(Inline1.Ranges[0].start() - 1));
  EXPECT_FALSE(Root.getInlineStack(Inline1.Ranges[0].end()));

  // Verify we get one inline stack entry for addresses that are in
  // [[0x150-0x160)) but not in [0x152-0x155) or [0x157-0x158)
  auto InlineInfos = Root.getInlineStack(Inline1.Ranges[0].start());
  ASSERT_TRUE(InlineInfos);
  ASSERT_EQ(InlineInfos->size(), 1u);
  ASSERT_EQ(*InlineInfos->at(0), Inline1);
  InlineInfos = Root.getInlineStack(Inline1.Ranges[0].end() - 1);
  EXPECT_TRUE(InlineInfos);
  ASSERT_EQ(InlineInfos->size(), 1u);
  ASSERT_EQ(*InlineInfos->at(0), Inline1);

  // Verify we get two inline stack entries for addresses that are in
  // [0x152-0x155)
  InlineInfos = Root.getInlineStack(Inline1Sub1.Ranges[0].start());
  EXPECT_TRUE(InlineInfos);
  ASSERT_EQ(InlineInfos->size(), 2u);
  ASSERT_EQ(*InlineInfos->at(0), Inline1Sub1);
  ASSERT_EQ(*InlineInfos->at(1), Inline1);
  InlineInfos = Root.getInlineStack(Inline1Sub1.Ranges[0].end() - 1);
  EXPECT_TRUE(InlineInfos);
  ASSERT_EQ(InlineInfos->size(), 2u);
  ASSERT_EQ(*InlineInfos->at(0), Inline1Sub1);
  ASSERT_EQ(*InlineInfos->at(1), Inline1);

  // Verify we get two inline stack entries for addresses that are in
  // [0x157-0x158)
  InlineInfos = Root.getInlineStack(Inline1Sub2.Ranges[0].start());
  EXPECT_TRUE(InlineInfos);
  ASSERT_EQ(InlineInfos->size(), 2u);
  ASSERT_EQ(*InlineInfos->at(0), Inline1Sub2);
  ASSERT_EQ(*InlineInfos->at(1), Inline1);
  InlineInfos = Root.getInlineStack(Inline1Sub2.Ranges[0].end() - 1);
  EXPECT_TRUE(InlineInfos);
  ASSERT_EQ(InlineInfos->size(), 2u);
  ASSERT_EQ(*InlineInfos->at(0), Inline1Sub2);
  ASSERT_EQ(*InlineInfos->at(1), Inline1);

  // Test encoding and decoding InlineInfo objects
  TestInlineInfoEncodeDecode<StrpT>(llvm::endianness::little, Root);
  TestInlineInfoEncodeDecode<StrpT>(llvm::endianness::big, Root);
}

TEST(GSYMTest, TestInlineInfo) {
  TestInlineInfoImpl<uint32_t>(1, 2, 3);
  TestInlineInfoImpl<uint64_t>(0x1'0000'0001, 0x1'0000'0002, 0x1'0000'0003);
}

template <typename StrpT> static void TestInlineInfoEncodeErrors() {
  // Test InlineInfo encoding errors.

  // Test that we get an error when trying to encode an InlineInfo object
  // that has no ranges.
  InlineInfo Empty;
  std::string EmptyErr("attempted to encode invalid InlineInfo object");
  TestInlineInfoEncodeError<StrpT>(llvm::endianness::little, Empty, EmptyErr);
  TestInlineInfoEncodeError<StrpT>(llvm::endianness::big, Empty, EmptyErr);

  // Verify that we get an error trying to encode an InlineInfo object that has
  // a child InlineInfo that has no ranges.
  InlineInfo ContainsEmpty;
  ContainsEmpty.Ranges.insert({0x100, 0x200});
  ContainsEmpty.Children.push_back(Empty);
  TestInlineInfoEncodeError<StrpT>(llvm::endianness::little, ContainsEmpty,
                                   EmptyErr);
  TestInlineInfoEncodeError<StrpT>(llvm::endianness::big, ContainsEmpty,
                                   EmptyErr);

  // Verify that we get an error trying to encode an InlineInfo object that has
  // a child whose address range is not contained in the parent address range.
  InlineInfo ChildNotContained;
  std::string ChildNotContainedErr("child range not contained in parent");
  ChildNotContained.Ranges.insert({0x100, 0x200});
  InlineInfo ChildNotContainedChild;
  ChildNotContainedChild.Ranges.insert({0x200, 0x300});
  ChildNotContained.Children.push_back(ChildNotContainedChild);
  TestInlineInfoEncodeError<StrpT>(llvm::endianness::little, ChildNotContained,
                                   ChildNotContainedErr);
  TestInlineInfoEncodeError<StrpT>(llvm::endianness::big, ChildNotContained,
                                   ChildNotContainedErr);
}

TEST(GSYMTest, TestInlineInfoEncodeErrors) {
  TestInlineInfoEncodeErrors<uint32_t>();
  TestInlineInfoEncodeErrors<uint64_t>();
}

template <typename StrpT> static void TestInlineInfoDecodeErrors() {
  // Test decoding InlineInfo objects that ensure we report an appropriate
  // error message.
  const llvm::endianness ByteOrder = llvm::endianness::little;
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  FileWriter FW(OutStrm, ByteOrder);
  FW.setStringOffsetSize(sizeof(StrpT));
  const uint64_t BaseAddr = 0x100;
  // InlineInfo layout: Ranges(4) + HasChildren(1) + Name(sizeof(StrpT)) + ...
  constexpr uint64_t ChildrenOff = 4; // After encoded ranges
  constexpr uint64_t NameOff = ChildrenOff + 1;
  constexpr uint64_t PostNameOff = NameOff + sizeof(StrpT);
  constexpr uint64_t PostCallFileOff = PostNameOff + 1; // CallFile is ULEB(0)
  auto Hex = [](uint64_t V) { return llvm::formatv("0x{0:x-8}", V).str(); };
  TestInlineInfoDecodeError<StrpT>(
      ByteOrder, OutStrm.str(), BaseAddr,
      "0x00000000: missing InlineInfo address ranges data");
  AddressRanges Ranges;
  Ranges.insert({BaseAddr, BaseAddr+0x100});
  encodeRanges(Ranges, FW, BaseAddr);
  TestInlineInfoDecodeError<StrpT>(
      ByteOrder, OutStrm.str(), BaseAddr,
      Hex(ChildrenOff) + ": missing InlineInfo uint8_t indicating children");
  FW.writeU8(0);
  TestInlineInfoDecodeError<StrpT>(ByteOrder, OutStrm.str(), BaseAddr,
                                   Hex(NameOff) + ": missing InlineInfo name");
  switch (sizeof(StrpT)) {
  case 4:
    FW.writeU32(0);
    break;
  case 8:
    FW.writeU64(0);
    break;
  default:
    FAIL() << "unsupported StrpT size: " << sizeof(StrpT);
  }
  TestInlineInfoDecodeError<StrpT>(
      ByteOrder, OutStrm.str(), BaseAddr,
      Hex(PostNameOff) + ": missing ULEB128 for InlineInfo call file");
  FW.writeU8(0);
  TestInlineInfoDecodeError<StrpT>(
      ByteOrder, OutStrm.str(), BaseAddr,
      Hex(PostCallFileOff) + ": missing ULEB128 for InlineInfo call line");
}

TEST(GSYMTest, TestInlineInfoDecodeErrors) {
  TestInlineInfoDecodeErrors<uint32_t>();
  TestInlineInfoDecodeErrors<uint64_t>();
}

TEST(GSYMTest, TestLineEntry) {
  // test llvm::gsym::LineEntry structs.
  const uint64_t ValidAddr = 0x1000;
  const uint64_t InvalidFileIdx = 0;
  const uint32_t ValidFileIdx = 1;
  const uint32_t ValidLine = 5;

  LineEntry Invalid;
  EXPECT_FALSE(Invalid.isValid());
  // Make sure that an entry is invalid if it has a bad file index.
  LineEntry BadFile(ValidAddr, InvalidFileIdx, ValidLine);
  EXPECT_FALSE(BadFile.isValid());
  // Test operators
  LineEntry E1(ValidAddr, ValidFileIdx, ValidLine);
  LineEntry E2(ValidAddr, ValidFileIdx, ValidLine);
  LineEntry DifferentAddr(ValidAddr + 1, ValidFileIdx, ValidLine);
  LineEntry DifferentFile(ValidAddr, ValidFileIdx + 1, ValidLine);
  LineEntry DifferentLine(ValidAddr, ValidFileIdx, ValidLine + 1);
  EXPECT_TRUE(E1.isValid());
  EXPECT_EQ(E1, E2);
  EXPECT_NE(E1, DifferentAddr);
  EXPECT_NE(E1, DifferentFile);
  EXPECT_NE(E1, DifferentLine);
  EXPECT_LT(E1, DifferentAddr);
}

TEST(GSYMTest, TestStringTable) {
  StringTable StrTab(StringRef("\0Hello\0World\0", 13));
  // Test extracting strings from a string table.
  EXPECT_EQ(StrTab.getString(0), "");
  EXPECT_EQ(StrTab.getString(1), "Hello");
  EXPECT_EQ(StrTab.getString(7), "World");
  EXPECT_EQ(StrTab.getString(8), "orld");
  // Test pointing to last NULL terminator gets empty string.
  EXPECT_EQ(StrTab.getString(12), "");
  // Test pointing to past end gets empty string.
  EXPECT_EQ(StrTab.getString(13), "");
}

static void TestFileWriterHelper(llvm::endianness ByteOrder) {
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  FileWriter FW(OutStrm, ByteOrder);
  const int64_t MinSLEB = INT64_MIN;
  const int64_t MaxSLEB = INT64_MAX;
  const uint64_t MinULEB = 0;
  const uint64_t MaxULEB = UINT64_MAX;
  const uint8_t U8 = 0x10;
  const uint16_t U16 = 0x1122;
  const uint32_t U32 = 0x12345678;
  const uint64_t U64 = 0x33445566778899aa;
  const char *Hello = "hello";
  FW.writeU8(U8);
  FW.writeU16(U16);
  FW.writeU32(U32);
  FW.writeU64(U64);
  FW.alignTo(16);
  const off_t FixupOffset = FW.tell();
  FW.writeU32(0);
  FW.writeSLEB(MinSLEB);
  FW.writeSLEB(MaxSLEB);
  FW.writeULEB(MinULEB);
  FW.writeULEB(MaxULEB);
  FW.writeNullTerminated(Hello);
  // Test Seek, Tell using Fixup32.
  FW.fixup32(U32, FixupOffset);

  std::string Bytes(OutStrm.str());
  GsymDataExtractor Data(Bytes, ByteOrder == llvm::endianness::little);
  uint64_t Offset = 0;
  EXPECT_EQ(Data.getU8(&Offset), U8);
  EXPECT_EQ(Data.getU16(&Offset), U16);
  EXPECT_EQ(Data.getU32(&Offset), U32);
  EXPECT_EQ(Data.getU64(&Offset), U64);
  Offset = alignTo(Offset, 16);
  EXPECT_EQ(Data.getU32(&Offset), U32);
  EXPECT_EQ(Data.getSLEB128(&Offset), MinSLEB);
  EXPECT_EQ(Data.getSLEB128(&Offset), MaxSLEB);
  EXPECT_EQ(Data.getULEB128(&Offset), MinULEB);
  EXPECT_EQ(Data.getULEB128(&Offset), MaxULEB);
  EXPECT_EQ(Data.getCStrRef(&Offset), StringRef(Hello));
}

TEST(GSYMTest, TestFileWriter) {
  TestFileWriterHelper(llvm::endianness::little);
  TestFileWriterHelper(llvm::endianness::big);
}

static void TestWriteUnsignedHelper(llvm::endianness ByteOrder) {
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  FileWriter FW(OutStrm, ByteOrder);
  const bool IsLittleEndian = ByteOrder == llvm::endianness::little;

  // Write values of sizes 1 through 8.
  FW.writeUnsigned(0x01, 1);
  FW.writeUnsigned(0x0102, 2);
  FW.writeUnsigned(0x010203, 3);
  FW.writeUnsigned(0x01020304, 4);
  FW.writeUnsigned(0x0102030405, 5);
  FW.writeUnsigned(0x010203040506, 6);
  FW.writeUnsigned(0x01020304050607, 7);
  FW.writeUnsigned(0x0102030405060708, 8);

  std::string Bytes(OutStrm.str());
  GsymDataExtractor Data(Bytes, IsLittleEndian);
  uint64_t Offset = 0;

  EXPECT_EQ(0x01U, Data.getUnsigned(&Offset, 1));
  EXPECT_EQ(0x0102U, Data.getUnsigned(&Offset, 2));
  EXPECT_EQ(0x010203U, Data.getUnsigned(&Offset, 3));
  EXPECT_EQ(0x01020304U, Data.getUnsigned(&Offset, 4));
  EXPECT_EQ(0x0102030405U, Data.getUnsigned(&Offset, 5));
  EXPECT_EQ(0x010203040506U, Data.getUnsigned(&Offset, 6));
  EXPECT_EQ(0x01020304050607U, Data.getUnsigned(&Offset, 7));
  EXPECT_EQ(0x0102030405060708U, Data.getUnsigned(&Offset, 8));
  EXPECT_EQ(Offset, Str.size());
}

TEST(GSYMTest, TestWriteUnsigned) {
  TestWriteUnsignedHelper(llvm::endianness::little);
  TestWriteUnsignedHelper(llvm::endianness::big);
  TestWriteUnsignedHelper(llvm::endianness::native);
}

TEST(GSYMTest, TestAddressRangeEncodeDecode) {
  // Test encoding and decoding AddressRange objects. AddressRange objects
  // are always stored as offsets from the a base address. The base address
  // is the FunctionInfo's base address for function level ranges, and is
  // the base address of the parent range for subranges.
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  const auto ByteOrder = llvm::endianness::native;
  FileWriter FW(OutStrm, ByteOrder);
  const uint64_t BaseAddr = 0x1000;
  const AddressRange Range1(0x1000, 0x1010);
  const AddressRange Range2(0x1020, 0x1030);
  encodeRange(Range1, FW, BaseAddr);
  encodeRange(Range2, FW, BaseAddr);
  std::string Bytes(OutStrm.str());
  GsymDataExtractor Data(Bytes, ByteOrder == llvm::endianness::little);

  AddressRange DecodedRange1, DecodedRange2;
  uint64_t Offset = 0;
  DecodedRange1 = decodeRange(Data, BaseAddr, Offset);
  DecodedRange2 = decodeRange(Data, BaseAddr, Offset);
  EXPECT_EQ(Range1, DecodedRange1);
  EXPECT_EQ(Range2, DecodedRange2);
}

static void TestAddressRangeEncodeDecodeHelper(const AddressRanges &Ranges,
                                               const uint64_t BaseAddr) {
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  const auto ByteOrder = llvm::endianness::native;
  FileWriter FW(OutStrm, ByteOrder);
  encodeRanges(Ranges, FW, BaseAddr);

  std::string Bytes(OutStrm.str());
  GsymDataExtractor Data(Bytes, ByteOrder == llvm::endianness::little);

  AddressRanges DecodedRanges;
  uint64_t Offset = 0;
  decodeRanges(DecodedRanges, Data, BaseAddr, Offset);
  EXPECT_EQ(Ranges, DecodedRanges);
}

TEST(GSYMTest, TestAddressRangesEncodeDecode) {
  // Test encoding and decoding AddressRanges. AddressRanges objects contain
  // ranges that are stored as offsets from the a base address. The base address
  // is the FunctionInfo's base address for function level ranges, and is the
  // base address of the parent range for subranges.
  const uint64_t BaseAddr = 0x1000;

  // Test encoding and decoding with no ranges.
  AddressRanges Ranges;
  TestAddressRangeEncodeDecodeHelper(Ranges, BaseAddr);

  // Test encoding and decoding with 1 range.
  Ranges.insert(AddressRange(0x1000, 0x1010));
  TestAddressRangeEncodeDecodeHelper(Ranges, BaseAddr);

  // Test encoding and decoding with multiple ranges.
  Ranges.insert(AddressRange(0x1020, 0x1030));
  Ranges.insert(AddressRange(0x1050, 0x1070));
  TestAddressRangeEncodeDecodeHelper(Ranges, BaseAddr);
}

static void TestLineTableHelper(llvm::endianness ByteOrder,
                                const LineTable &LT) {
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  FileWriter FW(OutStrm, ByteOrder);
  const uint64_t BaseAddr = LT[0].Addr;
  llvm::Error Err = LT.encode(FW, BaseAddr);
  ASSERT_FALSE(Err);
  std::string Bytes(OutStrm.str());
  GsymDataExtractor Data(Bytes, ByteOrder == llvm::endianness::little);
  llvm::Expected<LineTable> Decoded = LineTable::decode(Data, BaseAddr);
  // Make sure decoding succeeded.
  ASSERT_TRUE((bool)Decoded);
  // Make sure decoded object is the same as the one we encoded.
  EXPECT_EQ(LT, Decoded.get());
}

TEST(GSYMTest, TestLineTable) {
  const uint64_t StartAddr = 0x1000;
  const uint32_t FileIdx = 1;
  LineTable LT;
  LineEntry Line0(StartAddr+0x000, FileIdx, 10);
  LineEntry Line1(StartAddr+0x010, FileIdx, 11);
  LineEntry Line2(StartAddr+0x100, FileIdx, 1000);
  ASSERT_TRUE(LT.empty());
  ASSERT_EQ(LT.size(), (size_t)0);
  LT.push(Line0);
  ASSERT_EQ(LT.size(), (size_t)1);
  LT.push(Line1);
  LT.push(Line2);
  LT.push(LineEntry(StartAddr+0x120, FileIdx, 900));
  LT.push(LineEntry(StartAddr+0x120, FileIdx, 2000));
  LT.push(LineEntry(StartAddr+0x121, FileIdx, 2001));
  LT.push(LineEntry(StartAddr+0x122, FileIdx, 2002));
  LT.push(LineEntry(StartAddr+0x123, FileIdx, 2003));
  ASSERT_FALSE(LT.empty());
  ASSERT_EQ(LT.size(), (size_t)8);
  // Test operator[].
  ASSERT_EQ(LT[0], Line0);
  ASSERT_EQ(LT[1], Line1);
  ASSERT_EQ(LT[2], Line2);

  // Test encoding and decoding line tables.
  TestLineTableHelper(llvm::endianness::little, LT);
  TestLineTableHelper(llvm::endianness::big, LT);

  // Verify the clear method works as expected.
  LT.clear();
  ASSERT_TRUE(LT.empty());
  ASSERT_EQ(LT.size(), (size_t)0);

  LineTable LT1;
  LineTable LT2;

  // Test that two empty line tables are equal and neither are less than
  // each other.
  ASSERT_EQ(LT1, LT2);
  ASSERT_FALSE(LT1 < LT1);
  ASSERT_FALSE(LT1 < LT2);
  ASSERT_FALSE(LT2 < LT1);
  ASSERT_FALSE(LT2 < LT2);

  // Test that a line table with less number of line entries is less than a
  // line table with more line entries and that they are not equal.
  LT2.push(Line0);
  ASSERT_LT(LT1, LT2);
  ASSERT_NE(LT1, LT2);

  // Test that two line tables with the same entries are equal.
  LT1.push(Line0);
  ASSERT_EQ(LT1, LT2);
  ASSERT_FALSE(LT1 < LT2);
  ASSERT_FALSE(LT2 < LT2);
}

static void TestLineTableDecodeError(llvm::endianness ByteOrder,
                                     StringRef Bytes, const uint64_t BaseAddr,
                                     std::string ExpectedErrorMsg) {
  GsymDataExtractor Data(Bytes, ByteOrder == llvm::endianness::little);
  llvm::Expected<LineTable> Decoded = LineTable::decode(Data, BaseAddr);
  // Make sure decoding fails.
  ASSERT_FALSE((bool)Decoded);
  // Make sure decoded object is the same as the one we encoded.
  checkError(ExpectedErrorMsg, Decoded.takeError());
}

TEST(GSYMTest, TestLineTableDecodeErrors) {
  // Test decoding InlineInfo objects that ensure we report an appropriate
  // error message.
  const llvm::endianness ByteOrder = llvm::endianness::little;
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  FileWriter FW(OutStrm, ByteOrder);
  const uint64_t BaseAddr = 0x100;
  TestLineTableDecodeError(ByteOrder, OutStrm.str(), BaseAddr,
      "0x00000000: missing LineTable MinDelta");
  FW.writeU8(1); // MinDelta (ULEB)
  TestLineTableDecodeError(ByteOrder, OutStrm.str(), BaseAddr,
      "0x00000001: missing LineTable MaxDelta");
  FW.writeU8(10); // MaxDelta (ULEB)
  TestLineTableDecodeError(ByteOrder, OutStrm.str(), BaseAddr,
      "0x00000002: missing LineTable FirstLine");
  FW.writeU8(20); // FirstLine (ULEB)
  TestLineTableDecodeError(ByteOrder, OutStrm.str(), BaseAddr,
      "0x00000003: EOF found before EndSequence");
  // Test a SetFile with the argument missing from the stream
  FW.writeU8(1); // SetFile opcode (uint8_t)
  TestLineTableDecodeError(ByteOrder, OutStrm.str(), BaseAddr,
      "0x00000004: EOF found before SetFile value");
  FW.writeU8(5); // SetFile value as index (ULEB)
  // Test a AdvancePC with the argument missing from the stream
  FW.writeU8(2); // AdvancePC opcode (uint8_t)
  TestLineTableDecodeError(ByteOrder, OutStrm.str(), BaseAddr,
      "0x00000006: EOF found before AdvancePC value");
  FW.writeU8(20); // AdvancePC value as offset (ULEB)
  // Test a AdvancePC with the argument missing from the stream
  FW.writeU8(3); // AdvanceLine opcode (uint8_t)
  TestLineTableDecodeError(ByteOrder, OutStrm.str(), BaseAddr,
      "0x00000008: EOF found before AdvanceLine value");
  FW.writeU8(20); // AdvanceLine value as offset (LLEB)
}

TEST(GSYMTest, TestLineTableEncodeErrors) {
  const uint64_t BaseAddr = 0x1000;
  const uint32_t FileIdx = 1;
  const llvm::endianness ByteOrder = llvm::endianness::little;
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  FileWriter FW(OutStrm, ByteOrder);
  LineTable LT;
  checkError("attempted to encode invalid LineTable object",
             LT.encode(FW, BaseAddr));

  // Try to encode a line table where a line entry has an address that is less
  // than BaseAddr and verify we get an appropriate error.
  LineEntry Line0(BaseAddr+0x000, FileIdx, 10);
  LineEntry Line1(BaseAddr+0x010, FileIdx, 11);
  LT.push(Line0);
  LT.push(Line1);
  checkError("LineEntry has address 0x1000 which is less than the function "
             "start address 0x1010", LT.encode(FW, BaseAddr+0x10));
  LT.clear();

  // Try to encode a line table where a line entries  has an address that is less
  // than BaseAddr and verify we get an appropriate error.
  LT.push(Line1);
  LT.push(Line0);
  checkError("LineEntry in LineTable not in ascending order",
             LT.encode(FW, BaseAddr));
  LT.clear();
}

template <typename HeaderT>
static void TestHeaderEncodeError(const HeaderT &H,
                                  std::string ExpectedErrorMsg) {
  const llvm::endianness ByteOrder = llvm::endianness::little;
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  FileWriter FW(OutStrm, ByteOrder);
  llvm::Error Err = H.encode(FW);
  checkError(ExpectedErrorMsg, std::move(Err));
}

template <typename HeaderT>
static void TestHeaderDecodeError(StringRef Bytes,
                                  std::string ExpectedErrorMsg) {
  GsymDataExtractor Data(Bytes, /*IsLittleEndian=*/true);
  llvm::Expected<HeaderT> Decoded = HeaderT::decode(Data);
  ASSERT_FALSE((bool)Decoded);
  checkError(ExpectedErrorMsg, Decoded.takeError());
}

// Populate a GSYM V1 header with valid values.
static void InitHeader(Header &H) {
  H.Magic = GSYM_MAGIC;
  H.Version = Header::getVersion();
  H.AddrOffSize = 4;
  H.UUIDSize = 16;
  H.BaseAddress = 0x1000;
  H.NumAddresses = 1;
  H.StrtabOffset= 0x2000;
  H.StrtabSize = 0x1000;
  for (size_t i=0; i<GSYM_MAX_UUID_SIZE; ++i) {
    if (i < H.UUIDSize)
      H.UUID[i] = i;
    else
      H.UUID[i] = 0;
  }
}

// Populate a GSYM V2 header with valid values.
static void InitHeaderV2(HeaderV2 &H) {
  H.Magic = GSYM_MAGIC;
  H.Version = HeaderV2::getVersion();
  H.AddrOffSize = 4;
  H.StrTableEncoding = StringTableEncoding::Default;
  H.BaseAddress = 0x1000;
  H.NumAddresses = 1;
}

TEST(GSYMTest, TestHeaderEncodeErrors) {
  Header H;
  InitHeader(H);
  H.Magic = 12;
  TestHeaderEncodeError(H, "invalid GSYM magic 0x0000000c");
  InitHeader(H);
  H.Version = 12;
  TestHeaderEncodeError(H, "unsupported GSYM version 12");
  InitHeader(H);
  H.AddrOffSize = 12;
  TestHeaderEncodeError(H, "invalid address offset size 12");
  InitHeader(H);
  H.UUIDSize = 128;
  TestHeaderEncodeError(H, "invalid UUID size 128");
}

TEST(GSYMTest, TestHeaderDecodeErrors) {
  const llvm::endianness ByteOrder = llvm::endianness::little;
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  FileWriter FW(OutStrm, ByteOrder);
  Header H;
  InitHeader(H);
  llvm::Error Err = H.encode(FW);
  ASSERT_FALSE(Err);
  FW.fixup32(12, offsetof(Header, Magic));
  TestHeaderDecodeError<Header>(OutStrm.str(), "invalid GSYM magic 0x0000000c");
  FW.fixup32(GSYM_MAGIC, offsetof(Header, Magic));
  FW.fixup32(12, offsetof(Header, Version));
  TestHeaderDecodeError<Header>(OutStrm.str(), "unsupported GSYM version 12");
  FW.fixup32(Header::getVersion(), offsetof(Header, Version));
  FW.fixup32(12, offsetof(Header, AddrOffSize));
  TestHeaderDecodeError<Header>(OutStrm.str(),
                                "invalid address offset size 12");
  FW.fixup32(4, offsetof(Header, AddrOffSize));
  FW.fixup32(128, offsetof(Header, UUIDSize));
  TestHeaderDecodeError<Header>(OutStrm.str(), "invalid UUID size 128");
}

TEST(GSYMTest, TestHeaderV2EncodeErrors) {
  HeaderV2 H;
  InitHeaderV2(H);
  H.Magic = 12;
  TestHeaderEncodeError(H, "invalid GSYM magic 0x0000000c");
  InitHeaderV2(H);
  H.Version = 12;
  TestHeaderEncodeError(H, "unsupported GSYM version 12");
  InitHeaderV2(H);
  H.AddrOffSize = 12;
  TestHeaderEncodeError(H, "invalid address offset size 12");
  InitHeaderV2(H);
  H.StrTableEncoding = static_cast<StringTableEncoding>(12);
  TestHeaderEncodeError(H, "unsupported string table encoding 12");
}

TEST(GSYMTest, TestHeaderV2DecodeErrors) {
  const llvm::endianness ByteOrder = llvm::endianness::little;
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  FileWriter FW(OutStrm, ByteOrder);
  HeaderV2 H;
  InitHeaderV2(H);
  llvm::Error Err = H.encode(FW);
  ASSERT_FALSE(Err);
  // bad magic
  FW.fixup32(12, offsetof(HeaderV2, Magic));
  TestHeaderDecodeError<HeaderV2>(OutStrm.str(),
                                  "invalid GSYM magic 0x0000000c");
  FW.fixup32(GSYM_MAGIC, 0);
  // bad version
  FW.fixup32(12, offsetof(HeaderV2, Version));
  TestHeaderDecodeError<HeaderV2>(OutStrm.str(), "unsupported GSYM version 12");
  FW.fixup32(HeaderV2::getVersion(), offsetof(HeaderV2, Version));
  // bad address offset size
  FW.fixup32(12, offsetof(HeaderV2, AddrOffSize));
  TestHeaderDecodeError<HeaderV2>(OutStrm.str(),
                                  "invalid address offset size 12");
  FW.fixup32(8, offsetof(HeaderV2, AddrOffSize));
  // bad string table encoding
  FW.fixup32(12, offsetof(HeaderV2, StrTableEncoding));
  TestHeaderDecodeError<HeaderV2>(OutStrm.str(),
                                  "unsupported string table encoding 12");
  FW.fixup32(1, offsetof(HeaderV2, StrTableEncoding));
  // truncated header
  TestHeaderDecodeError<HeaderV2>(StringRef("short"),
                                  "not enough data for a gsym::HeaderV2");
}

template <typename HeaderT>
static void TestHeaderEncodeDecode(const HeaderT &H,
                                   llvm::endianness ByteOrder) {
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  FileWriter FW(OutStrm, ByteOrder);
  llvm::Error Err = H.encode(FW);
  ASSERT_FALSE(Err);
  std::string Bytes(OutStrm.str());
  GsymDataExtractor Data(Bytes, ByteOrder == llvm::endianness::little);
  llvm::Expected<HeaderT> Decoded = HeaderT::decode(Data);
  // Make sure decoding succeeded.
  ASSERT_TRUE((bool)Decoded);
  EXPECT_EQ(H, Decoded.get());
}
TEST(GSYMTest, TestHeaderEncodeDecode) {
  Header H;
  InitHeader(H);
  TestHeaderEncodeDecode(H, llvm::endianness::little);
  TestHeaderEncodeDecode(H, llvm::endianness::big);
}
TEST(GSYMTest, TestHeaderV2EncodeDecode) {
  HeaderV2 H;
  InitHeaderV2(H);
  TestHeaderEncodeDecode(H, llvm::endianness::little);
  TestHeaderEncodeDecode(H, llvm::endianness::big);
}

template <typename CreatorT>
static void TestGsymCreatorEncodeErrorHelper(llvm::endianness ByteOrder,
                                             const CreatorT &GC,
                                             std::string ExpectedErrorMsg) {
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  FileWriter FW(OutStrm, ByteOrder);
  FW.setStringOffsetSize(GC.getStringOffsetSize());
  llvm::Error Err = GC.encode(FW);
  ASSERT_TRUE(bool(Err));
  checkError(ExpectedErrorMsg, std::move(Err));
}

template <typename CreatorT> static void TestGsymCreatorEncodeErrors() {
  const uint8_t ValidUUID[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                               14, 15, 16};
  // Verify we get an error when trying to encode a GsymCreator with no
  // function infos. We shouldn't be saving a GSYM file in this case since
  // there is nothing inside of it.
  CreatorT GC;
  TestGsymCreatorEncodeErrorHelper(llvm::endianness::little, GC,
                                   "no functions to encode");
  const uint64_t FuncAddr = 0x1000;
  const uint64_t FuncSize = 0x100;
  const uint32_t FuncName = GC.insertString("foo");
  // Verify we get an error trying to encode a GsymCreator that isn't
  // finalized.
  GC.addFunctionInfo(FunctionInfo(FuncAddr, FuncSize, FuncName));
  TestGsymCreatorEncodeErrorHelper(
      llvm::endianness::little, GC,
      "GsymCreator wasn't finalized prior to encoding");
  std::string finalizeIssues;
  raw_string_ostream OS(finalizeIssues);
  OutputAggregator Agg(&OS);
  llvm::Error finalizeErr = GC.finalize(Agg);
  ASSERT_FALSE(bool(finalizeErr));
  finalizeErr = GC.finalize(Agg);
  ASSERT_TRUE(bool(finalizeErr));
  checkError("already finalized", std::move(finalizeErr));
  GC.setUUID(ValidUUID);
  // Verify errors are propagated when we try to encoding an invalid line
  // table.
  GC.forEachFunctionInfo([](FunctionInfo &FI) -> bool {
    FI.OptLineTable = LineTable(); // Invalid line table.
    return false; // Stop iterating
  });
  TestGsymCreatorEncodeErrorHelper(
      llvm::endianness::little, GC,
      "attempted to encode invalid LineTable object");
  // Verify errors are propagated when we try to encoding an invalid inline
  // info.
  GC.forEachFunctionInfo([](FunctionInfo &FI) -> bool {
    FI.OptLineTable = std::nullopt;
    FI.Inline = InlineInfo(); // Invalid InlineInfo.
    return false; // Stop iterating
  });
  TestGsymCreatorEncodeErrorHelper(
      llvm::endianness::little, GC,
      "attempted to encode invalid InlineInfo object");
}

TEST(GSYMTest, TestGsymCreatorV1EncodeErrors) {
  TestGsymCreatorEncodeErrors<GsymCreatorV1>();

  // V1-specific: verify we get an error with a UUID that is too long.
  // V2 stores UUID as a GlobalData entry with no size limit.
  const uint8_t UUIDTooLong[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
  GsymCreatorV1 GC;
  GC.addFunctionInfo(FunctionInfo(0x1000, 0x100, GC.insertString("foo")));
  OutputAggregator Null(nullptr);
  llvm::Error Err = GC.finalize(Null);
  ASSERT_FALSE(bool(Err));
  GC.setUUID(UUIDTooLong);
  TestGsymCreatorEncodeErrorHelper(llvm::endianness::little, GC,
                                   "invalid UUID size 21");
}

TEST(GSYMTest, TestGsymCreatorV2EncodeErrors) {
  TestGsymCreatorEncodeErrors<GsymCreatorV2>();
}

static void Compare(const GsymCreator &GC, const GsymReader &GR) {
  // Verify that all of the data in a GsymCreator is correctly decoded from
  // a GsymReader. To do this, we iterate over
  GC.forEachFunctionInfo([&](const FunctionInfo &FI) -> bool {
    auto DecodedFI = GR.getFunctionInfo(FI.Range.start());
    EXPECT_TRUE(bool(DecodedFI));
    EXPECT_EQ(FI, *DecodedFI);
    return true; // Keep iterating over all FunctionInfo objects.
  });
}

static void TestEncodeDecode(const GsymCreator &GC, llvm::endianness ByteOrder,
                             uint64_t ExpBaseAddress, uint32_t ExpNumAddresses,
                             uint8_t ExpAddrOffSize, uint8_t ExpAddrInfoOffSize,
                             uint8_t ExpStrOffSize) {
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  FileWriter FW(OutStrm, ByteOrder);
  FW.setStringOffsetSize(GC.getStringOffsetSize());
  llvm::Error Err = GC.encode(FW);
  ASSERT_FALSE((bool)Err);
  auto GR = GsymReader::copyBuffer(OutStrm.str());
  ASSERT_TRUE(bool(GR));
  EXPECT_EQ((*GR)->getBaseAddress(), ExpBaseAddress);
  EXPECT_EQ((*GR)->getNumAddresses(), ExpNumAddresses);
  EXPECT_EQ((*GR)->getAddressOffsetSize(), ExpAddrOffSize);
  EXPECT_EQ((*GR)->getAddressInfoOffsetSize(), ExpAddrInfoOffSize);
  EXPECT_EQ((*GR)->getStringOffsetSize(), ExpStrOffSize);
  Compare(GC, **GR);
}

/// Test address offset encoding with a given target byte size.
/// Uses an address delta of (1 << ((TargetByteSize-1) * 8)) to force the
/// creator to use at least TargetByteSize bytes for address offsets.
template <typename CreatorT>
static void TestGsymCreatorAddrOffsetsImpl(uint8_t TargetByteSize,
                                           uint8_t ExpAddrOffSize,
                                           uint8_t ExpAddrInfoOffSize,
                                           uint8_t ExpStrOffSize) {
  uint8_t UUID[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  CreatorT GC;
  GC.setUUID(UUID);
  constexpr uint64_t BaseAddr = 0x1000;
  const uint64_t AddrDelta =
      (TargetByteSize == 1) ? 0x20ULL : (1ULL << ((TargetByteSize - 1) * 8));
  const uint32_t Func1Name = GC.insertString("foo");
  const uint32_t Func2Name = GC.insertString("bar");
  GC.addFunctionInfo(FunctionInfo(BaseAddr, 0x10, Func1Name));
  GC.addFunctionInfo(FunctionInfo(BaseAddr + AddrDelta, 0x10, Func2Name));
  OutputAggregator Null(nullptr);
  Error Err = GC.finalize(Null);
  ASSERT_FALSE(Err);
  TestEncodeDecode(GC, llvm::endianness::little, BaseAddr, 2, ExpAddrOffSize,
                   ExpAddrInfoOffSize, ExpStrOffSize);
  TestEncodeDecode(GC, llvm::endianness::big, BaseAddr, 2, ExpAddrOffSize,
                   ExpAddrInfoOffSize, ExpStrOffSize);
}

// V1 tests (power-of-two AddrOffSize: 1, 2, 4, 8)
TEST(GSYMTest, TestGsymCreatorV11ByteAddrOffsets) {
  TestGsymCreatorAddrOffsetsImpl<GsymCreatorV1>(1, 1, 4, 4);
}
TEST(GSYMTest, TestGsymCreatorV12ByteAddrOffsets) {
  TestGsymCreatorAddrOffsetsImpl<GsymCreatorV1>(2, 2, 4, 4);
}
TEST(GSYMTest, TestGsymCreatorV13ByteAddrOffsets) {
  TestGsymCreatorAddrOffsetsImpl<GsymCreatorV1>(3, 4, 4, 4);
}
TEST(GSYMTest, TestGsymCreatorV14ByteAddrOffsets) {
  TestGsymCreatorAddrOffsetsImpl<GsymCreatorV1>(4, 4, 4, 4);
}
TEST(GSYMTest, TestGsymCreatorV15ByteAddrOffsets) {
  TestGsymCreatorAddrOffsetsImpl<GsymCreatorV1>(5, 8, 4, 4);
}
TEST(GSYMTest, TestGsymCreatorV16ByteAddrOffsets) {
  TestGsymCreatorAddrOffsetsImpl<GsymCreatorV1>(6, 8, 4, 4);
}
TEST(GSYMTest, TestGsymCreatorV17ByteAddrOffsets) {
  TestGsymCreatorAddrOffsetsImpl<GsymCreatorV1>(7, 8, 4, 4);
}
TEST(GSYMTest, TestGsymCreatorV18ByteAddrOffsets) {
  TestGsymCreatorAddrOffsetsImpl<GsymCreatorV1>(8, 8, 4, 4);
}

// V2 tests (AddrOffSize rounds to power-of-2, fixed AddrInfoOffSize=8,
// StrOffSize=8)
TEST(GSYMTest, TestGsymCreatorV21ByteAddrOffsets) {
  TestGsymCreatorAddrOffsetsImpl<GsymCreatorV2>(1, 1, 8, 8);
}
TEST(GSYMTest, TestGsymCreatorV22ByteAddrOffsets) {
  TestGsymCreatorAddrOffsetsImpl<GsymCreatorV2>(2, 2, 8, 8);
}
TEST(GSYMTest, TestGsymCreatorV23ByteAddrOffsets) {
  TestGsymCreatorAddrOffsetsImpl<GsymCreatorV2>(3, 4, 8, 8);
}
TEST(GSYMTest, TestGsymCreatorV24ByteAddrOffsets) {
  TestGsymCreatorAddrOffsetsImpl<GsymCreatorV2>(4, 4, 8, 8);
}
TEST(GSYMTest, TestGsymCreatorV25ByteAddrOffsets) {
  TestGsymCreatorAddrOffsetsImpl<GsymCreatorV2>(5, 8, 8, 8);
}
TEST(GSYMTest, TestGsymCreatorV26ByteAddrOffsets) {
  TestGsymCreatorAddrOffsetsImpl<GsymCreatorV2>(6, 8, 8, 8);
}
TEST(GSYMTest, TestGsymCreatorV27ByteAddrOffsets) {
  TestGsymCreatorAddrOffsetsImpl<GsymCreatorV2>(7, 8, 8, 8);
}
TEST(GSYMTest, TestGsymCreatorV28ByteAddrOffsets) {
  TestGsymCreatorAddrOffsetsImpl<GsymCreatorV2>(8, 8, 8, 8);
}

static void VerifyFunctionInfo(const GsymReader &GR, uint64_t Addr,
                               const FunctionInfo &FI) {
  auto ExpFI = GR.getFunctionInfo(Addr);
  ASSERT_THAT_EXPECTED(ExpFI, Succeeded());
  ASSERT_EQ(FI, ExpFI.get());
}

static void VerifyFunctionInfoError(const GsymReader &GR, uint64_t Addr,
                                    std::string ErrMessage) {
  auto ExpFI = GR.getFunctionInfo(Addr);
  ASSERT_FALSE(bool(ExpFI));
  checkError(ErrMessage, ExpFI.takeError());
}

template <typename CreatorT>
static void TestGsymReader(llvm::endianness ByteOrder) {
  uint8_t UUID[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  CreatorT GC;
  GC.setUUID(UUID);
  constexpr uint64_t BaseAddr = 0x1000;
  constexpr uint64_t Func1Addr = BaseAddr;
  constexpr uint64_t Func2Addr = BaseAddr+0x20;
  constexpr uint64_t FuncSize = 0x10;
  const uint32_t Func1Name = GC.insertString("foo");
  const uint32_t Func2Name = GC.insertString("bar");
  GC.addFunctionInfo(FunctionInfo(Func1Addr, FuncSize, Func1Name));
  GC.addFunctionInfo(FunctionInfo(Func2Addr, FuncSize, Func2Name));
  OutputAggregator Null(nullptr);
  Error FinalizeErr = GC.finalize(Null);
  ASSERT_FALSE(FinalizeErr);
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  FileWriter FW(OutStrm, ByteOrder);
  FW.setStringOffsetSize(GC.getStringOffsetSize());
  llvm::Error Err = GC.encode(FW);
  ASSERT_FALSE((bool)Err);
  if (auto ExpectedGR = GsymReader::copyBuffer(OutStrm.str())) {
    const GsymReader &GR = **ExpectedGR;
    VerifyFunctionInfoError(GR, Func1Addr-1, "address 0xfff is not in GSYM");

    FunctionInfo Func1(Func1Addr, FuncSize, Func1Name);
    VerifyFunctionInfo(GR, Func1Addr, Func1);
    VerifyFunctionInfo(GR, Func1Addr+1, Func1);
    VerifyFunctionInfo(GR, Func1Addr+FuncSize-1, Func1);
    VerifyFunctionInfoError(GR, Func1Addr+FuncSize,
                            "address 0x1010 is not in GSYM");
    VerifyFunctionInfoError(GR, Func2Addr-1, "address 0x101f is not in GSYM");
    FunctionInfo Func2(Func2Addr, FuncSize, Func2Name);
    VerifyFunctionInfo(GR, Func2Addr, Func2);
    VerifyFunctionInfo(GR, Func2Addr+1, Func2);
    VerifyFunctionInfo(GR, Func2Addr+FuncSize-1, Func2);
    VerifyFunctionInfoError(GR, Func2Addr+FuncSize,
                            "address 0x1030 is not in GSYM");
  }
}

TEST(GSYMTest, TestGsymReaderV1) {
  TestGsymReader<GsymCreatorV1>(llvm::endianness::little);
  TestGsymReader<GsymCreatorV1>(llvm::endianness::big);
}
TEST(GSYMTest, TestGsymReaderV2) {
  TestGsymReader<GsymCreatorV2>(llvm::endianness::little);
  TestGsymReader<GsymCreatorV2>(llvm::endianness::big);
}

template <typename CreatorT>
static void TestGsymLookups(llvm::endianness ByteOrder) {
  // Test creating a GSYM file with a function that has a inline information.
  // Verify that lookups work correctly. Lookups do not decode the entire
  // FunctionInfo or InlineInfo, they only extract information needed for the
  // lookup to happen which avoids allocations which can slow down
  // symbolication.
  CreatorT GC;
  FunctionInfo FI(0x1000, 0x100, GC.insertString("main"));
  FI.OptLineTable = LineTable();
  const uint32_t MainFileIndex = GC.insertFile("/tmp/main.c");
  const uint32_t FooFileIndex = GC.insertFile("/tmp/foo.h");
  FI.OptLineTable->push(LineEntry(0x1000, MainFileIndex, 5));
  FI.OptLineTable->push(LineEntry(0x1010, FooFileIndex, 10));
  FI.OptLineTable->push(LineEntry(0x1012, FooFileIndex, 20));
  FI.OptLineTable->push(LineEntry(0x1014, FooFileIndex, 11));
  FI.OptLineTable->push(LineEntry(0x1016, FooFileIndex, 30));
  FI.OptLineTable->push(LineEntry(0x1018, FooFileIndex, 12));
  FI.OptLineTable->push(LineEntry(0x1020, MainFileIndex, 8));
  FI.Inline = InlineInfo();

  FI.Inline->Name = GC.insertString("inline1");
  FI.Inline->CallFile = MainFileIndex;
  FI.Inline->CallLine = 6;
  FI.Inline->Ranges.insert(AddressRange(0x1010, 0x1020));
  InlineInfo Inline2;
  Inline2.Name = GC.insertString("inline2");
  Inline2.CallFile = FooFileIndex;
  Inline2.CallLine = 33;
  Inline2.Ranges.insert(AddressRange(0x1012, 0x1014));
  FI.Inline->Children.emplace_back(Inline2);
  InlineInfo Inline3;
  Inline3.Name = GC.insertString("inline3");
  Inline3.CallFile = FooFileIndex;
  Inline3.CallLine = 35;
  Inline3.Ranges.insert(AddressRange(0x1016, 0x1018));
  FI.Inline->Children.emplace_back(Inline3);
  GC.addFunctionInfo(std::move(FI));
  OutputAggregator Null(nullptr);
  Error FinalizeErr = GC.finalize(Null);
  ASSERT_FALSE(FinalizeErr);
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  FileWriter FW(OutStrm, ByteOrder);
  FW.setStringOffsetSize(GC.getStringOffsetSize());
  llvm::Error Err = GC.encode(FW);
  ASSERT_FALSE((bool)Err);
  auto GROrErr = GsymReader::copyBuffer(OutStrm.str());
  ASSERT_TRUE(bool(GROrErr));
  const std::unique_ptr<GsymReader> &GR = *GROrErr;

  // Verify inline info is correct when doing lookups.
  auto LR = GR->lookup(0x1000);
  ASSERT_THAT_EXPECTED(LR, Succeeded());
  EXPECT_THAT(LR->Locations,
    testing::ElementsAre(SourceLocation{"main", "/tmp", "main.c", 5}));
  LR = GR->lookup(0x100F);
  ASSERT_THAT_EXPECTED(LR, Succeeded());
  EXPECT_THAT(LR->Locations,
    testing::ElementsAre(SourceLocation{"main", "/tmp", "main.c", 5, 15}));

  LR = GR->lookup(0x1010);
  ASSERT_THAT_EXPECTED(LR, Succeeded());

  EXPECT_THAT(LR->Locations,
    testing::ElementsAre(SourceLocation{"inline1", "/tmp", "foo.h", 10},
                         SourceLocation{"main", "/tmp", "main.c", 6, 16}));

  LR = GR->lookup(0x1012);
  ASSERT_THAT_EXPECTED(LR, Succeeded());
  EXPECT_THAT(LR->Locations,
    testing::ElementsAre(SourceLocation{"inline2", "/tmp", "foo.h", 20},
                         SourceLocation{"inline1", "/tmp", "foo.h", 33, 2},
                         SourceLocation{"main", "/tmp", "main.c", 6, 18}));

  LR = GR->lookup(0x1014);
  ASSERT_THAT_EXPECTED(LR, Succeeded());
  EXPECT_THAT(LR->Locations,
    testing::ElementsAre(SourceLocation{"inline1", "/tmp", "foo.h", 11, 4},
                         SourceLocation{"main", "/tmp", "main.c", 6, 20}));

  LR = GR->lookup(0x1016);
  ASSERT_THAT_EXPECTED(LR, Succeeded());
  EXPECT_THAT(LR->Locations,
    testing::ElementsAre(SourceLocation{"inline3", "/tmp", "foo.h", 30},
                         SourceLocation{"inline1", "/tmp", "foo.h", 35, 6},
                         SourceLocation{"main", "/tmp", "main.c", 6, 22}));

  LR = GR->lookup(0x1018);
  ASSERT_THAT_EXPECTED(LR, Succeeded());
  EXPECT_THAT(LR->Locations,
    testing::ElementsAre(SourceLocation{"inline1", "/tmp", "foo.h", 12, 8},
                         SourceLocation{"main", "/tmp", "main.c", 6, 24}));

  LR = GR->lookup(0x1020);
  ASSERT_THAT_EXPECTED(LR, Succeeded());
  EXPECT_THAT(LR->Locations,
    testing::ElementsAre(SourceLocation{"main", "/tmp", "main.c", 8, 32}));
}

TEST(GSYMTest, TestGsymLookups) {
  TestGsymLookups<GsymCreatorV1>(llvm::endianness::little);
  TestGsymLookups<GsymCreatorV1>(llvm::endianness::big);
}
TEST(GSYMTest, TestGsymLookupsV2) {
  TestGsymLookups<GsymCreatorV2>(llvm::endianness::little);
  TestGsymLookups<GsymCreatorV2>(llvm::endianness::big);
}

template <typename CreatorT> static void TestDWARFFunctionWithAddresses() {
  // Create a single compile unit with a single function and make sure it gets
  // converted to DWARF correctly. The function's address range is in where
  // DW_AT_low_pc and DW_AT_high_pc are both addresses.
  StringRef yamldata = R"(
  debug_str:
    - ''
    - /tmp/main.c
    - main
  debug_abbrev:
    - Table:
        - Code:            0x00000001
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_low_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_high_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_language
              Form:            DW_FORM_data2
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
    - Version:         4
      AddrSize:        8
      Entries:
        - AbbrCode:        0x00000001
          Values:
            - Value:           0x0000000000000001
            - Value:           0x0000000000001000
            - Value:           0x0000000000002000
            - Value:           0x0000000000000004
        - AbbrCode:        0x00000002
          Values:
            - Value:           0x000000000000000D
            - Value:           0x0000000000001000
            - Value:           0x0000000000002000
        - AbbrCode:        0x00000000
  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(yamldata);
  ASSERT_THAT_EXPECTED(ErrOrSections, Succeeded());
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);
  ASSERT_TRUE(DwarfContext.get() != nullptr);
  auto &OS = llvm::nulls();
  OutputAggregator OSAgg(&OS);
  CreatorT GC;
  DwarfTransformer DT(*DwarfContext, GC);
  const uint32_t ThreadCount = 1;
  ASSERT_THAT_ERROR(DT.convert(ThreadCount, OSAgg), Succeeded());
  ASSERT_THAT_ERROR(GC.finalize(OSAgg), Succeeded());
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  const auto ByteOrder = llvm::endianness::native;
  FileWriter FW(OutStrm, ByteOrder);
  FW.setStringOffsetSize(GC.getStringOffsetSize());
  ASSERT_THAT_ERROR(GC.encode(FW), Succeeded());
  auto GROrErr = GsymReader::copyBuffer(OutStrm.str());
  ASSERT_THAT_EXPECTED(GROrErr, Succeeded());
  const std::unique_ptr<GsymReader> &GR = *GROrErr;
  // There should only be one function in our GSYM.
  EXPECT_EQ(GR->getNumAddresses(), 1u);
  auto ExpFI = GR->getFunctionInfo(0x1000);
  ASSERT_THAT_EXPECTED(ExpFI, Succeeded());
  ASSERT_EQ(ExpFI->Range, AddressRange(0x1000, 0x2000));
  EXPECT_FALSE(ExpFI->OptLineTable.has_value());
  EXPECT_FALSE(ExpFI->Inline.has_value());
}

TEST(GSYMTest, TestDWARFFunctionWithAddresses) {
  TestDWARFFunctionWithAddresses<GsymCreatorV1>();
}
TEST(GSYMTest, TestDWARFFunctionWithAddressesV2) {
  TestDWARFFunctionWithAddresses<GsymCreatorV2>();
}

template <typename CreatorT>
static void TestDWARFFunctionWithAddressAndOffset() {
  // Create a single compile unit with a single function and make sure it gets
  // converted to DWARF correctly. The function's address range is in where
  // DW_AT_low_pc is an address and the DW_AT_high_pc is an offset.
  StringRef yamldata = R"(
  debug_str:
    - ''
    - /tmp/main.c
    - main
  debug_abbrev:
    - Table:
        - Code:            0x00000001
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_low_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_high_pc
              Form:            DW_FORM_data4
            - Attribute:       DW_AT_language
              Form:            DW_FORM_data2
        - Code:            0x00000002
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_low_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_high_pc
              Form:            DW_FORM_data4
  debug_info:
    - Version:         4
      AddrSize:        8
      Entries:
        - AbbrCode:        0x00000001
          Values:
            - Value:           0x0000000000000001
            - Value:           0x0000000000001000
            - Value:           0x0000000000001000
            - Value:           0x0000000000000004
        - AbbrCode:        0x00000002
          Values:
            - Value:           0x000000000000000D
            - Value:           0x0000000000001000
            - Value:           0x0000000000001000
        - AbbrCode:        0x00000000
  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(yamldata);
  ASSERT_THAT_EXPECTED(ErrOrSections, Succeeded());
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);
  ASSERT_TRUE(DwarfContext.get() != nullptr);
  auto &OS = llvm::nulls();
  OutputAggregator OSAgg(&OS);
  CreatorT GC;
  DwarfTransformer DT(*DwarfContext, GC);
  const uint32_t ThreadCount = 1;
  ASSERT_THAT_ERROR(DT.convert(ThreadCount, OSAgg), Succeeded());
  ASSERT_THAT_ERROR(GC.finalize(OSAgg), Succeeded());
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  const auto ByteOrder = llvm::endianness::native;
  FileWriter FW(OutStrm, ByteOrder);
  FW.setStringOffsetSize(GC.getStringOffsetSize());
  ASSERT_THAT_ERROR(GC.encode(FW), Succeeded());
  auto GROrErr = GsymReader::copyBuffer(OutStrm.str());
  ASSERT_THAT_EXPECTED(GROrErr, Succeeded());
  const std::unique_ptr<GsymReader> &GR = *GROrErr;
  // There should only be one function in our GSYM.
  EXPECT_EQ(GR->getNumAddresses(), 1u);
  auto ExpFI = GR->getFunctionInfo(0x1000);
  ASSERT_THAT_EXPECTED(ExpFI, Succeeded());
  ASSERT_EQ(ExpFI->Range, AddressRange(0x1000, 0x2000));
  EXPECT_FALSE(ExpFI->OptLineTable.has_value());
  EXPECT_FALSE(ExpFI->Inline.has_value());
}

TEST(GSYMTest, TestDWARFFunctionWithAddressAndOffset) {
  TestDWARFFunctionWithAddressAndOffset<GsymCreatorV1>();
}
TEST(GSYMTest, TestDWARFFunctionWithAddressAndOffsetV2) {
  TestDWARFFunctionWithAddressAndOffset<GsymCreatorV2>();
}

template <typename CreatorT> static void TestDWARFStructMethodNoMangled() {
  // Sometimes the compiler will omit the mangled name in the DWARF for static
  // and member functions of classes and structs. This test verifies that the
  // fully qualified name of the method is computed and used as the string for
  // the function in the GSYM in these cases. Otherwise we might just get a
  // function name like "erase" instead of "std::vector<int>::erase".
  StringRef yamldata = R"(
  debug_str:
    - ''
    - /tmp/main.c
    - Foo
    - dump
    - this
  debug_abbrev:
    - Table:
        - Code:            0x00000001
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_low_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_high_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_language
              Form:            DW_FORM_data2
        - Code:            0x00000002
          Tag:             DW_TAG_structure_type
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
        - Code:            0x00000003
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_low_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_high_pc
              Form:            DW_FORM_addr
        - Code:            0x00000004
          Tag:             DW_TAG_formal_parameter
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_type
              Form:            DW_FORM_ref4
            - Attribute:       DW_AT_artificial
              Form:            DW_FORM_flag_present
  debug_info:
    - Version:         4
      AddrSize:        8
      Entries:
        - AbbrCode:        0x00000001
          Values:
            - Value:           0x0000000000000001
            - Value:           0x0000000000001000
            - Value:           0x0000000000002000
            - Value:           0x0000000000000004
        - AbbrCode:        0x00000002
          Values:
            - Value:           0x000000000000000D
        - AbbrCode:        0x00000003
          Values:
            - Value:           0x0000000000000011
            - Value:           0x0000000000001000
            - Value:           0x0000000000002000
        - AbbrCode:        0x00000004
          Values:
            - Value:           0x0000000000000016
            - Value:           0x0000000000000022
            - Value:           0x0000000000000001
        - AbbrCode:        0x00000000
        - AbbrCode:        0x00000000
        - AbbrCode:        0x00000000
  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(yamldata);
  ASSERT_THAT_EXPECTED(ErrOrSections, Succeeded());
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);
  ASSERT_TRUE(DwarfContext.get() != nullptr);
  auto &OS = llvm::nulls();
  OutputAggregator OSAgg(&OS);
  CreatorT GC;
  DwarfTransformer DT(*DwarfContext, GC);
  const uint32_t ThreadCount = 1;
  ASSERT_THAT_ERROR(DT.convert(ThreadCount, OSAgg), Succeeded());
  ASSERT_THAT_ERROR(GC.finalize(OSAgg), Succeeded());
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  const auto ByteOrder = llvm::endianness::native;
  FileWriter FW(OutStrm, ByteOrder);
  FW.setStringOffsetSize(GC.getStringOffsetSize());
  ASSERT_THAT_ERROR(GC.encode(FW), Succeeded());
  auto GROrErr = GsymReader::copyBuffer(OutStrm.str());
  ASSERT_THAT_EXPECTED(GROrErr, Succeeded());
  const std::unique_ptr<GsymReader> &GR = *GROrErr;
  // There should only be one function in our GSYM.
  EXPECT_EQ(GR->getNumAddresses(), 1u);
  auto ExpFI = GR->getFunctionInfo(0x1000);
  ASSERT_THAT_EXPECTED(ExpFI, Succeeded());
  ASSERT_EQ(ExpFI->Range, AddressRange(0x1000, 0x2000));
  EXPECT_FALSE(ExpFI->OptLineTable.has_value());
  EXPECT_FALSE(ExpFI->Inline.has_value());
  StringRef MethodName = GR->getString(ExpFI->Name);
  EXPECT_EQ(MethodName, "Foo::dump");
}

TEST(GSYMTest, TestDWARFStructMethodNoMangled) {
  TestDWARFStructMethodNoMangled<GsymCreatorV1>();
}
TEST(GSYMTest, TestDWARFStructMethodNoMangledV2) {
  TestDWARFStructMethodNoMangled<GsymCreatorV2>();
}

template <typename CreatorT> static void TestDWARFTextRanges() {
  // Linkers don't understand DWARF, they just like to concatenate and
  // relocate data within the DWARF sections. This means that if a function
  // gets dead stripped, and if those functions use an offset as the
  // DW_AT_high_pc, we can end up with many functions at address zero. The
  // DwarfTransformer allows clients to specify valid .text address ranges
  // and any addresses of any functions must fall within those ranges if any
  // have been specified. This means that an object file can calcuate the
  // address ranges within the binary where code lives and set these ranges
  // as constraints in the DwarfTransformer. ObjectFile instances can
  // add a address ranges of sections that have executable permissions. This
  // keeps bad information from being added to a GSYM file and causing issues
  // when symbolicating.
  StringRef yamldata = R"(
  debug_str:
    - ''
    - /tmp/main.c
    - main
    - dead_stripped
    - dead_stripped2
  debug_abbrev:
    - Table:
        - Code:            0x00000001
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_low_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_high_pc
              Form:            DW_FORM_data4
            - Attribute:       DW_AT_language
              Form:            DW_FORM_data2
        - Code:            0x00000002
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_low_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_high_pc
              Form:            DW_FORM_data4
  debug_info:
    - Version:         4
      AddrSize:        8
      Entries:
        - AbbrCode:        0x00000001
          Values:
            - Value:           0x0000000000000001
            - Value:           0x0000000000001000
            - Value:           0x0000000000001000
            - Value:           0x0000000000000004
        - AbbrCode:        0x00000002
          Values:
            - Value:           0x000000000000000D
            - Value:           0x0000000000001000
            - Value:           0x0000000000001000
        - AbbrCode:        0x00000002
          Values:
            - Value:           0x0000000000000012
            - Value:           0x0000000000000000
            - Value:           0x0000000000000100
        - AbbrCode:        0x00000002
          Values:
            - Value:           0x0000000000000020
            - Value:           0x0000000000000000
            - Value:           0x0000000000000040
        - AbbrCode:        0x00000000
  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(yamldata);
  ASSERT_THAT_EXPECTED(ErrOrSections, Succeeded());
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);
  ASSERT_TRUE(DwarfContext.get() != nullptr);
  auto &OS = llvm::nulls();
  OutputAggregator OSAgg(&OS);
  CreatorT GC;
  DwarfTransformer DT(*DwarfContext, GC);
  // Only allow addresses between [0x1000 - 0x2000) to be linked into the
  // GSYM.
  AddressRanges TextRanges;
  TextRanges.insert(AddressRange(0x1000, 0x2000));
  GC.SetValidTextRanges(TextRanges);
  const uint32_t ThreadCount = 1;
  ASSERT_THAT_ERROR(DT.convert(ThreadCount, OSAgg), Succeeded());
  ASSERT_THAT_ERROR(GC.finalize(OSAgg), Succeeded());
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  const auto ByteOrder = llvm::endianness::native;
  FileWriter FW(OutStrm, ByteOrder);
  FW.setStringOffsetSize(GC.getStringOffsetSize());
  ASSERT_THAT_ERROR(GC.encode(FW), Succeeded());
  auto GROrErr = GsymReader::copyBuffer(OutStrm.str());
  ASSERT_THAT_EXPECTED(GROrErr, Succeeded());
  const std::unique_ptr<GsymReader> &GR = *GROrErr;
  // There should only be one function in our GSYM.
  EXPECT_EQ(GR->getNumAddresses(), 1u);
  auto ExpFI = GR->getFunctionInfo(0x1000);
  ASSERT_THAT_EXPECTED(ExpFI, Succeeded());
  ASSERT_EQ(ExpFI->Range, AddressRange(0x1000, 0x2000));
  EXPECT_FALSE(ExpFI->OptLineTable.has_value());
  EXPECT_FALSE(ExpFI->Inline.has_value());
  StringRef MethodName = GR->getString(ExpFI->Name);
  EXPECT_EQ(MethodName, "main");
}

TEST(GSYMTest, TestDWARFTextRanges) { TestDWARFTextRanges<GsymCreatorV1>(); }
TEST(GSYMTest, TestDWARFTextRangesV2) { TestDWARFTextRanges<GsymCreatorV2>(); }

template <typename CreatorT>
static void TestEmptySymbolEndAddressOfTextRanges() {
  // Test that if we have valid text ranges and we have a symbol with no size
  // as the last FunctionInfo entry that the size of the symbol gets set to the
  // end address of the text range.
  CreatorT GC;
  AddressRanges TextRanges;
  TextRanges.insert(AddressRange(0x1000, 0x2000));
  GC.SetValidTextRanges(TextRanges);
  GC.addFunctionInfo(FunctionInfo(0x1500, 0, GC.insertString("symbol")));
  OutputAggregator Null(nullptr);
  ASSERT_THAT_ERROR(GC.finalize(Null), Succeeded());
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  const auto ByteOrder = llvm::endianness::native;
  FileWriter FW(OutStrm, ByteOrder);
  FW.setStringOffsetSize(GC.getStringOffsetSize());
  ASSERT_THAT_ERROR(GC.encode(FW), Succeeded());
  auto GROrErr = GsymReader::copyBuffer(OutStrm.str());
  ASSERT_THAT_EXPECTED(GROrErr, Succeeded());
  const std::unique_ptr<GsymReader> &GR = *GROrErr;
  // There should only be one function in our GSYM.
  EXPECT_EQ(GR->getNumAddresses(), 1u);
  auto ExpFI = GR->getFunctionInfo(0x1500);
  ASSERT_THAT_EXPECTED(ExpFI, Succeeded());
  ASSERT_EQ(ExpFI->Range, AddressRange(0x1500, 0x2000));
  EXPECT_FALSE(ExpFI->OptLineTable.has_value());
  EXPECT_FALSE(ExpFI->Inline.has_value());
  StringRef MethodName = GR->getString(ExpFI->Name);
  EXPECT_EQ(MethodName, "symbol");
}

TEST(GSYMTest, TestEmptySymbolEndAddressOfTextRanges) {
  TestEmptySymbolEndAddressOfTextRanges<GsymCreatorV1>();
}
TEST(GSYMTest, TestEmptySymbolEndAddressOfTextRangesV2) {
  TestEmptySymbolEndAddressOfTextRanges<GsymCreatorV2>();
}

template <typename CreatorT> static void TestDWARFInlineInfo() {
  // Make sure we parse the line table and inline information correctly from
  // DWARF.
  StringRef yamldata = R"(
  debug_str:
    - ''
    - /tmp/main.c
    - main
    - inline1
  debug_abbrev:
    - Table:
        - Code:            0x00000001
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_low_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_high_pc
              Form:            DW_FORM_data4
            - Attribute:       DW_AT_language
              Form:            DW_FORM_data2
            - Attribute:       DW_AT_stmt_list
              Form:            DW_FORM_sec_offset
        - Code:            0x00000002
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_low_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_high_pc
              Form:            DW_FORM_data4
        - Code:            0x00000003
          Tag:             DW_TAG_inlined_subroutine
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_low_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_high_pc
              Form:            DW_FORM_data4
            - Attribute:       DW_AT_call_file
              Form:            DW_FORM_data4
            - Attribute:       DW_AT_call_line
              Form:            DW_FORM_data4
  debug_info:
    - Version:         4
      AddrSize:        8
      Entries:
        - AbbrCode:        0x00000001
          Values:
            - Value:           0x0000000000000001
            - Value:           0x0000000000001000
            - Value:           0x0000000000001000
            - Value:           0x0000000000000004
            - Value:           0x0000000000000000
        - AbbrCode:        0x00000002
          Values:
            - Value:           0x000000000000000D
            - Value:           0x0000000000001000
            - Value:           0x0000000000001000
        - AbbrCode:        0x00000003
          Values:
            - Value:           0x0000000000000012
            - Value:           0x0000000000001100
            - Value:           0x0000000000000100
            - Value:           0x0000000000000001
            - Value:           0x000000000000000A
        - AbbrCode:        0x00000000
        - AbbrCode:        0x00000000
  debug_line:
    - Length:          96
      Version:         2
      PrologueLength:  46
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
        - Name:            inline.h
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
        - Opcode:          DW_LNS_set_file
          Data:            2
        - Opcode:          DW_LNS_advance_line
          SData:           10
          Data:            2
        - Opcode:          DW_LNS_copy
          Data:            2
        - Opcode:          DW_LNS_advance_pc
          Data:            128
        - Opcode:          DW_LNS_advance_line
          SData:           1
          Data:            128
        - Opcode:          DW_LNS_copy
          Data:            128
        - Opcode:          DW_LNS_advance_pc
          Data:            128
        - Opcode:          DW_LNS_set_file
          Data:            1
        - Opcode:          DW_LNS_advance_line
          SData:           -10
          Data:            1
        - Opcode:          DW_LNS_copy
          Data:            1
        - Opcode:          DW_LNS_advance_pc
          Data:            3584
        - Opcode:          DW_LNS_advance_line
          SData:           1
          Data:            3584
        - Opcode:          DW_LNS_extended_op
          ExtLen:          1
          SubOpcode:       DW_LNE_end_sequence
          Data:            3584
  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(yamldata);
  ASSERT_THAT_EXPECTED(ErrOrSections, Succeeded());
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);
  ASSERT_TRUE(DwarfContext.get() != nullptr);
  auto &OS = llvm::nulls();
  OutputAggregator OSAgg(&OS);
  CreatorT GC;
  DwarfTransformer DT(*DwarfContext, GC);
  const uint32_t ThreadCount = 1;
  ASSERT_THAT_ERROR(DT.convert(ThreadCount, OSAgg), Succeeded());
  ASSERT_THAT_ERROR(GC.finalize(OSAgg), Succeeded());
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  const auto ByteOrder = llvm::endianness::native;
  FileWriter FW(OutStrm, ByteOrder);
  FW.setStringOffsetSize(GC.getStringOffsetSize());
  ASSERT_THAT_ERROR(GC.encode(FW), Succeeded());
  auto GROrErr = GsymReader::copyBuffer(OutStrm.str());
  ASSERT_THAT_EXPECTED(GROrErr, Succeeded());
  const std::unique_ptr<GsymReader> &GR = *GROrErr;
  // There should only be one function in our GSYM.
  EXPECT_EQ(GR->getNumAddresses(), 1u);
  auto ExpFI = GR->getFunctionInfo(0x1000);
  ASSERT_THAT_EXPECTED(ExpFI, Succeeded());
  ASSERT_EQ(ExpFI->Range, AddressRange(0x1000, 0x2000));
  EXPECT_TRUE(ExpFI->OptLineTable.has_value());
  EXPECT_TRUE(ExpFI->Inline.has_value());
  StringRef MethodName = GR->getString(ExpFI->Name);
  EXPECT_EQ(MethodName, "main");

    // Verify inline info is correct when doing lookups.
  auto LR = GR->lookup(0x1000);
  ASSERT_THAT_EXPECTED(LR, Succeeded());
  EXPECT_THAT(LR->Locations,
    testing::ElementsAre(SourceLocation{"main", "/tmp", "main.c", 10}));
  LR = GR->lookup(0x1100-1);
  ASSERT_THAT_EXPECTED(LR, Succeeded());
  EXPECT_THAT(LR->Locations,
    testing::ElementsAre(SourceLocation{"main", "/tmp", "main.c", 10, 255}));

  LR = GR->lookup(0x1100);
  ASSERT_THAT_EXPECTED(LR, Succeeded());
  EXPECT_THAT(LR->Locations,
    testing::ElementsAre(SourceLocation{"inline1", "/tmp", "inline.h", 20},
                         SourceLocation{"main", "/tmp", "main.c", 10, 256}));
  LR = GR->lookup(0x1180-1);
  ASSERT_THAT_EXPECTED(LR, Succeeded());
  EXPECT_THAT(LR->Locations,
    testing::ElementsAre(SourceLocation{"inline1", "/tmp", "inline.h", 20, 127},
                         SourceLocation{"main", "/tmp", "main.c", 10, 383}));
  LR = GR->lookup(0x1180);
  ASSERT_THAT_EXPECTED(LR, Succeeded());
  EXPECT_THAT(LR->Locations,
    testing::ElementsAre(SourceLocation{"inline1", "/tmp", "inline.h", 21, 128},
                         SourceLocation{"main", "/tmp", "main.c", 10, 384}));
  LR = GR->lookup(0x1200-1);
  ASSERT_THAT_EXPECTED(LR, Succeeded());
  EXPECT_THAT(LR->Locations,
    testing::ElementsAre(SourceLocation{"inline1", "/tmp", "inline.h", 21, 255},
                         SourceLocation{"main", "/tmp", "main.c", 10, 511}));
  LR = GR->lookup(0x1200);
  ASSERT_THAT_EXPECTED(LR, Succeeded());
  EXPECT_THAT(LR->Locations,
    testing::ElementsAre(SourceLocation{"main", "/tmp", "main.c", 11, 512}));
}

TEST(GSYMTest, TestDWARFInlineInfo) { TestDWARFInlineInfo<GsymCreatorV1>(); }
TEST(GSYMTest, TestDWARFInlineInfoV2) { TestDWARFInlineInfo<GsymCreatorV2>(); }

template <typename CreatorT> static void TestDWARFNoLines() {
  // Check that if a DW_TAG_subprogram doesn't have line table entries that
  // we fall back and use the DW_AT_decl_file and DW_AT_decl_line to at least
  // point to the function definition. This DWARF file has 4 functions:
  //  "lines_no_decl": has line table entries, no DW_AT_decl_file/line attrs.
  //  "lines_with_decl": has line table entries and has DW_AT_decl_file/line,
  //                     make sure we don't use DW_AT_decl_file/line and make
  //                     sure there is a line table.
  //  "no_lines_no_decl": no line table entries and no DW_AT_decl_file/line,
  //                      make sure there is no line table for this function.
  //  "no_lines_with_decl": no line table and has DW_AT_decl_file/line, make
  //                        sure we have one line table entry that starts at
  //                        the function start address and the decl file and
  //                        line.
  //
  // 0x0000000b: DW_TAG_compile_unit
  //               DW_AT_name	("/tmp/main.c")
  //               DW_AT_low_pc	(0x0000000000001000)
  //               DW_AT_high_pc	(0x0000000000002000)
  //               DW_AT_language	(DW_LANG_C_plus_plus)
  //               DW_AT_stmt_list	(0x00000000)
  //
  // 0x00000022:   DW_TAG_subprogram
  //                 DW_AT_name	("lines_no_decl")
  //                 DW_AT_low_pc	(0x0000000000001000)
  //                 DW_AT_high_pc	(0x0000000000002000)
  //
  // 0x00000033:   DW_TAG_subprogram
  //                 DW_AT_name	("lines_with_decl")
  //                 DW_AT_low_pc	(0x0000000000002000)
  //                 DW_AT_high_pc	(0x0000000000003000)
  //                 DW_AT_decl_file	("/tmp/main.c")
  //                 DW_AT_decl_line	(20)
  //
  // 0x00000046:   DW_TAG_subprogram
  //                 DW_AT_name	("no_lines_no_decl")
  //                 DW_AT_low_pc	(0x0000000000003000)
  //                 DW_AT_high_pc	(0x0000000000004000)
  //
  // 0x00000057:   DW_TAG_subprogram
  //                 DW_AT_name	("no_lines_with_decl")
  //                 DW_AT_low_pc	(0x0000000000004000)
  //                 DW_AT_high_pc	(0x0000000000005000)
  //                 DW_AT_decl_file	("/tmp/main.c")
  //                 DW_AT_decl_line	(40)
  //
  // 0x0000006a:   NULL

  StringRef yamldata = R"(
  debug_str:
    - ''
    - '/tmp/main.c'
    - lines_no_decl
    - lines_with_decl
    - no_lines_no_decl
    - no_lines_with_decl
  debug_abbrev:
    - Table:
        - Code:            0x00000001
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_low_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_high_pc
              Form:            DW_FORM_data4
            - Attribute:       DW_AT_language
              Form:            DW_FORM_data2
            - Attribute:       DW_AT_stmt_list
              Form:            DW_FORM_sec_offset
        - Code:            0x00000002
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_low_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_high_pc
              Form:            DW_FORM_data4
        - Code:            0x00000003
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_low_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_high_pc
              Form:            DW_FORM_data4
            - Attribute:       DW_AT_decl_file
              Form:            DW_FORM_data1
            - Attribute:       DW_AT_decl_line
              Form:            DW_FORM_data1
  debug_info:
    - Version:         4
      AddrSize:        8
      Entries:
        - AbbrCode:        0x00000001
          Values:
            - Value:           0x0000000000000001
            - Value:           0x0000000000001000
            - Value:           0x0000000000001000
            - Value:           0x0000000000000004
            - Value:           0x0000000000000000
        - AbbrCode:        0x00000002
          Values:
            - Value:           0x000000000000000D
            - Value:           0x0000000000001000
            - Value:           0x0000000000001000
        - AbbrCode:        0x00000003
          Values:
            - Value:           0x000000000000001B
            - Value:           0x0000000000002000
            - Value:           0x0000000000001000
            - Value:           0x0000000000000001
            - Value:           0x0000000000000014
        - AbbrCode:        0x00000002
          Values:
            - Value:           0x000000000000002B
            - Value:           0x0000000000003000
            - Value:           0x0000000000001000
        - AbbrCode:        0x00000003
          Values:
            - Value:           0x000000000000003C
            - Value:           0x0000000000004000
            - Value:           0x0000000000001000
            - Value:           0x0000000000000001
            - Value:           0x0000000000000028
        - AbbrCode:        0x00000000
  debug_line:
    - Length:          92
      Version:         2
      PrologueLength:  34
      MinInstLength:   1
      DefaultIsStmt:   1
      LineBase:        251
      LineRange:       14
      OpcodeBase:      13
      StandardOpcodeLengths: [ 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1 ]
      IncludeDirs:
        - '/tmp'
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
          SData:           10
          Data:            0
        - Opcode:          DW_LNS_copy
          Data:            0
        - Opcode:          DW_LNS_advance_pc
          Data:            512
        - Opcode:          DW_LNS_advance_line
          SData:           1
          Data:            0
        - Opcode:          DW_LNS_copy
          Data:            0
        - Opcode:          DW_LNS_advance_pc
          Data:            3584
        - Opcode:          DW_LNS_extended_op
          ExtLen:          1
          SubOpcode:       DW_LNE_end_sequence
          Data:            0
        - Opcode:          DW_LNS_extended_op
          ExtLen:          9
          SubOpcode:       DW_LNE_set_address
          Data:            8192
        - Opcode:          DW_LNS_advance_line
          SData:           20
          Data:            0
        - Opcode:          DW_LNS_copy
          Data:            0
        - Opcode:          DW_LNS_advance_pc
          Data:            512
        - Opcode:          DW_LNS_advance_line
          SData:           1
          Data:            0
        - Opcode:          DW_LNS_copy
          Data:            0
        - Opcode:          DW_LNS_advance_pc
          Data:            3584
        - Opcode:          DW_LNS_extended_op
          ExtLen:          1
          SubOpcode:       DW_LNE_end_sequence
          Data:            0
  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(yamldata);
  ASSERT_THAT_EXPECTED(ErrOrSections, Succeeded());
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);
  ASSERT_TRUE(DwarfContext.get() != nullptr);
  auto &OS = llvm::nulls();
  OutputAggregator OSAgg(&OS);
  CreatorT GC;
  DwarfTransformer DT(*DwarfContext, GC);
  const uint32_t ThreadCount = 1;
  ASSERT_THAT_ERROR(DT.convert(ThreadCount, OSAgg), Succeeded());
  ASSERT_THAT_ERROR(GC.finalize(OSAgg), Succeeded());
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  const auto ByteOrder = llvm::endianness::native;
  FileWriter FW(OutStrm, ByteOrder);
  FW.setStringOffsetSize(GC.getStringOffsetSize());
  ASSERT_THAT_ERROR(GC.encode(FW), Succeeded());
  auto GROrErr = GsymReader::copyBuffer(OutStrm.str());
  ASSERT_THAT_EXPECTED(GROrErr, Succeeded());
  const std::unique_ptr<GsymReader> &GR = *GROrErr;

  EXPECT_EQ(GR->getNumAddresses(), 4u);

  auto ExpFI = GR->getFunctionInfo(0x1000);
  ASSERT_THAT_EXPECTED(ExpFI, Succeeded());
  ASSERT_EQ(ExpFI->Range, AddressRange(0x1000, 0x2000));
  EXPECT_TRUE(ExpFI->OptLineTable);
  StringRef MethodName = GR->getString(ExpFI->Name);
  EXPECT_EQ(MethodName, "lines_no_decl");
  // Make sure have two line table entries and that get the first line entry
  // correct.
  EXPECT_EQ(ExpFI->OptLineTable->size(), 2u);
  EXPECT_EQ(ExpFI->OptLineTable->first()->Addr, 0x1000u);
  EXPECT_EQ(ExpFI->OptLineTable->first()->Line, 11u);

  ExpFI = GR->getFunctionInfo(0x2000);
  ASSERT_THAT_EXPECTED(ExpFI, Succeeded());
  ASSERT_EQ(ExpFI->Range, AddressRange(0x2000, 0x3000));
  EXPECT_TRUE(ExpFI->OptLineTable);
  MethodName = GR->getString(ExpFI->Name);
  EXPECT_EQ(MethodName, "lines_with_decl");
  // Make sure have two line table entries and that we don't use line 20
  // from the DW_AT_decl_file/line as a line table entry.
  EXPECT_EQ(ExpFI->OptLineTable->size(), 2u);
  EXPECT_EQ(ExpFI->OptLineTable->first()->Addr, 0x2000u);
  EXPECT_EQ(ExpFI->OptLineTable->first()->Line, 21u);

  ExpFI = GR->getFunctionInfo(0x3000);
  ASSERT_THAT_EXPECTED(ExpFI, Succeeded());
  ASSERT_EQ(ExpFI->Range, AddressRange(0x3000, 0x4000));
  // Make sure we have no line table.
  EXPECT_FALSE(ExpFI->OptLineTable.has_value());
  MethodName = GR->getString(ExpFI->Name);
  EXPECT_EQ(MethodName, "no_lines_no_decl");

  ExpFI = GR->getFunctionInfo(0x4000);
  ASSERT_THAT_EXPECTED(ExpFI, Succeeded());
  ASSERT_EQ(ExpFI->Range, AddressRange(0x4000, 0x5000));
  EXPECT_TRUE(ExpFI->OptLineTable.has_value());
  MethodName = GR->getString(ExpFI->Name);
  EXPECT_EQ(MethodName, "no_lines_with_decl");
  // Make sure we have one line table entry that uses the DW_AT_decl_file/line
  // as the one and only line entry.
  EXPECT_EQ(ExpFI->OptLineTable->size(), 1u);
  EXPECT_EQ(ExpFI->OptLineTable->first()->Addr, 0x4000u);
  EXPECT_EQ(ExpFI->OptLineTable->first()->Line, 40u);
}

TEST(GSYMTest, TestDWARFNoLines) { TestDWARFNoLines<GsymCreatorV1>(); }
TEST(GSYMTest, TestDWARFNoLinesV2) { TestDWARFNoLines<GsymCreatorV2>(); }

template <typename CreatorT> static void TestDWARFDeadStripAddr4() {
  // Check that various techniques that compilers use for dead code stripping
  // work for 4 byte addresses. Make sure we keep the good functions and
  // strip any functions whose name starts with "stripped".
  //
  // 1 - Compilers might set the low PC to -1 (UINT32_MAX) for compile unit
  //     with 4 byte addresses ("stripped1")
  // 2 - Set the low and high PC to the same value ("stripped2")
  // 3 - Have the high PC lower than the low PC ("stripped3")
  //
  // 0x0000000b: DW_TAG_compile_unit
  //               DW_AT_name	("/tmp/main.c")
  //               DW_AT_low_pc	(0x0000000000001000)
  //               DW_AT_high_pc	(0x0000000000002000)
  //               DW_AT_language	(DW_LANG_C_plus_plus)
  //
  // 0x0000001a:   DW_TAG_subprogram
  //                 DW_AT_name	("main")
  //                 DW_AT_low_pc	(0x0000000000001000)
  //                 DW_AT_high_pc	(0x0000000000002000)
  //
  // 0x00000027:   DW_TAG_subprogram
  //                 DW_AT_name	("stripped1")
  //                 DW_AT_low_pc	(0x00000000ffffffff)
  //                 DW_AT_high_pc	(0x0000000100000000)
  //
  // 0x00000034:   DW_TAG_subprogram
  //                 DW_AT_name	("stripped2")
  //                 DW_AT_low_pc	(0x0000000000003000)
  //                 DW_AT_high_pc	(0x0000000000003000)
  //
  // 0x00000041:   DW_TAG_subprogram
  //                 DW_AT_name	("stripped3")
  //                 DW_AT_low_pc	(0x0000000000004000)
  //                 DW_AT_high_pc	(0x0000000000003fff)
  //
  // 0x0000004e:   NULL

  StringRef yamldata = R"(
  debug_str:
    - ''
    - '/tmp/main.c'
    - main
    - stripped1
    - stripped2
    - stripped3
  debug_abbrev:
    - Table:
        - Code:            0x00000001
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_low_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_high_pc
              Form:            DW_FORM_data4
            - Attribute:       DW_AT_language
              Form:            DW_FORM_data2
        - Code:            0x00000002
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_low_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_high_pc
              Form:            DW_FORM_data4
        - Code:            0x00000003
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
    - Version:         4
      AddrSize:        4
      Entries:
        - AbbrCode:        0x00000001
          Values:
            - Value:           0x0000000000000001
            - Value:           0x0000000000001000
            - Value:           0x0000000000001000
            - Value:           0x0000000000000004
        - AbbrCode:        0x00000002
          Values:
            - Value:           0x000000000000000D
            - Value:           0x0000000000001000
            - Value:           0x0000000000001000
        - AbbrCode:        0x00000002
          Values:
            - Value:           0x0000000000000012
            - Value:           0x00000000FFFFFFFF
            - Value:           0x0000000000000001
        - AbbrCode:        0x00000003
          Values:
            - Value:           0x000000000000001C
            - Value:           0x0000000000003000
            - Value:           0x0000000000003000
        - AbbrCode:        0x00000003
          Values:
            - Value:           0x0000000000000026
            - Value:           0x0000000000004000
            - Value:           0x0000000000003FFF
        - AbbrCode:        0x00000000
  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(yamldata);
  ASSERT_THAT_EXPECTED(ErrOrSections, Succeeded());
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 4);
  ASSERT_TRUE(DwarfContext.get() != nullptr);
  auto &OS = llvm::nulls();
  OutputAggregator OSAgg(&OS);
  CreatorT GC;
  DwarfTransformer DT(*DwarfContext, GC);
  const uint32_t ThreadCount = 1;
  ASSERT_THAT_ERROR(DT.convert(ThreadCount, OSAgg), Succeeded());
  ASSERT_THAT_ERROR(GC.finalize(OSAgg), Succeeded());
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  const auto ByteOrder = llvm::endianness::native;
  FileWriter FW(OutStrm, ByteOrder);
  FW.setStringOffsetSize(GC.getStringOffsetSize());
  ASSERT_THAT_ERROR(GC.encode(FW), Succeeded());
  auto GROrErr = GsymReader::copyBuffer(OutStrm.str());
  ASSERT_THAT_EXPECTED(GROrErr, Succeeded());
  const std::unique_ptr<GsymReader> &GR = *GROrErr;

  // Test that the only function that made it was the "main" function.
  EXPECT_EQ(GR->getNumAddresses(), 1u);
  auto ExpFI = GR->getFunctionInfo(0x1000);
  ASSERT_THAT_EXPECTED(ExpFI, Succeeded());
  ASSERT_EQ(ExpFI->Range, AddressRange(0x1000, 0x2000));
  StringRef MethodName = GR->getString(ExpFI->Name);
  EXPECT_EQ(MethodName, "main");
}

TEST(GSYMTest, TestDWARFDeadStripAddr4) {
  TestDWARFDeadStripAddr4<GsymCreatorV1>();
}
TEST(GSYMTest, TestDWARFDeadStripAddr4V2) {
  TestDWARFDeadStripAddr4<GsymCreatorV2>();
}

template <typename CreatorT> static void TestDWARFDeadStripAddr8() {
  // Check that various techniques that compilers use for dead code stripping
  // work for 4 byte addresses. Make sure we keep the good functions and
  // strip any functions whose name starts with "stripped".
  //
  // 1 - Compilers might set the low PC to -1 (UINT64_MAX) for compile unit
  //     with 8 byte addresses ("stripped1")
  // 2 - Set the low and high PC to the same value ("stripped2")
  // 3 - Have the high PC lower than the low PC ("stripped3")
  //
  // 0x0000000b: DW_TAG_compile_unit
  //               DW_AT_name	("/tmp/main.c")
  //               DW_AT_low_pc	(0x0000000000001000)
  //               DW_AT_high_pc	(0x0000000000002000)
  //               DW_AT_language	(DW_LANG_C_plus_plus)
  //
  // 0x0000001e:   DW_TAG_subprogram
  //                 DW_AT_name	("main")
  //                 DW_AT_low_pc	(0x0000000000001000)
  //                 DW_AT_high_pc	(0x0000000000002000)
  //
  // 0x0000002f:   DW_TAG_subprogram
  //                 DW_AT_name	("stripped1")
  //                 DW_AT_low_pc	(0xffffffffffffffff)
  //                 DW_AT_high_pc	(0x0000000000000000)
  //
  // 0x00000040:   DW_TAG_subprogram
  //                 DW_AT_name	("stripped2")
  //                 DW_AT_low_pc	(0x0000000000003000)
  //                 DW_AT_high_pc	(0x0000000000003000)
  //
  // 0x00000055:   DW_TAG_subprogram
  //                 DW_AT_name	("stripped3")
  //                 DW_AT_low_pc	(0x0000000000004000)
  //                 DW_AT_high_pc	(0x0000000000003fff)
  //
  // 0x0000006a:   NULL

  StringRef yamldata = R"(
  debug_str:
    - ''
    - '/tmp/main.c'
    - main
    - stripped1
    - stripped2
    - stripped3
  debug_abbrev:
    - Table:
        - Code:            0x00000001
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_low_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_high_pc
              Form:            DW_FORM_data4
            - Attribute:       DW_AT_language
              Form:            DW_FORM_data2
        - Code:            0x00000002
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_low_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_high_pc
              Form:            DW_FORM_data4
        - Code:            0x00000003
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
    - Version:         4
      AddrSize:        8
      Entries:
        - AbbrCode:        0x00000001
          Values:
            - Value:           0x0000000000000001
            - Value:           0x0000000000001000
            - Value:           0x0000000000001000
            - Value:           0x0000000000000004
        - AbbrCode:        0x00000002
          Values:
            - Value:           0x000000000000000D
            - Value:           0x0000000000001000
            - Value:           0x0000000000001000
        - AbbrCode:        0x00000002
          Values:
            - Value:           0x0000000000000012
            - Value:           0xFFFFFFFFFFFFFFFF
            - Value:           0x0000000000000001
        - AbbrCode:        0x00000003
          Values:
            - Value:           0x000000000000001C
            - Value:           0x0000000000003000
            - Value:           0x0000000000003000
        - AbbrCode:        0x00000003
          Values:
            - Value:           0x0000000000000026
            - Value:           0x0000000000004000
            - Value:           0x0000000000003FFF
        - AbbrCode:        0x00000000
  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(yamldata);
  ASSERT_THAT_EXPECTED(ErrOrSections, Succeeded());
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);
  ASSERT_TRUE(DwarfContext.get() != nullptr);
  auto &OS = llvm::nulls();
  OutputAggregator OSAgg(&OS);
  CreatorT GC;
  DwarfTransformer DT(*DwarfContext, GC);
  const uint32_t ThreadCount = 1;
  ASSERT_THAT_ERROR(DT.convert(ThreadCount, OSAgg), Succeeded());
  ASSERT_THAT_ERROR(GC.finalize(OSAgg), Succeeded());
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  const auto ByteOrder = llvm::endianness::native;
  FileWriter FW(OutStrm, ByteOrder);
  FW.setStringOffsetSize(GC.getStringOffsetSize());
  ASSERT_THAT_ERROR(GC.encode(FW), Succeeded());
  auto GROrErr = GsymReader::copyBuffer(OutStrm.str());
  ASSERT_THAT_EXPECTED(GROrErr, Succeeded());
  const std::unique_ptr<GsymReader> &GR = *GROrErr;

  // Test that the only function that made it was the "main" function.
  EXPECT_EQ(GR->getNumAddresses(), 1u);
  auto ExpFI = GR->getFunctionInfo(0x1000);
  ASSERT_THAT_EXPECTED(ExpFI, Succeeded());
  ASSERT_EQ(ExpFI->Range, AddressRange(0x1000, 0x2000));
  StringRef MethodName = GR->getString(ExpFI->Name);
  EXPECT_EQ(MethodName, "main");
}

TEST(GSYMTest, TestDWARFDeadStripAddr8) {
  TestDWARFDeadStripAddr8<GsymCreatorV1>();
}
TEST(GSYMTest, TestDWARFDeadStripAddr8V2) {
  TestDWARFDeadStripAddr8<GsymCreatorV2>();
}

template <typename CreatorT>
static void TestGsymCreatorMultipleSymbolsWithNoSizeImpl(
    uint8_t ExpAddrOffSize, uint8_t ExpAddrInfoOffSize, uint8_t ExpStrOffSize) {
  // Multiple symbols at the same address with zero size were being emitted
  // instead of being combined into a single entry. This function tests to make
  // sure we only get one symbol.
  uint8_t UUID[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  CreatorT GC;
  GC.setUUID(UUID);
  constexpr uint64_t BaseAddr = 0x1000;
  const uint32_t Func1Name = GC.insertString("foo");
  const uint32_t Func2Name = GC.insertString("bar");
  GC.addFunctionInfo(FunctionInfo(BaseAddr, 0, Func1Name));
  GC.addFunctionInfo(FunctionInfo(BaseAddr, 0, Func2Name));
  OutputAggregator Null(nullptr);
  Error Err = GC.finalize(Null);
  ASSERT_FALSE(Err);
  TestEncodeDecode(GC, llvm::endianness::little, BaseAddr, 1, ExpAddrOffSize,
                   ExpAddrInfoOffSize, ExpStrOffSize);
  TestEncodeDecode(GC, llvm::endianness::big, BaseAddr, 1, ExpAddrOffSize,
                   ExpAddrInfoOffSize, ExpStrOffSize);
}

TEST(GSYMTest, TestGsymCreatorV1MultipleSymbolsWithNoSize) {
  TestGsymCreatorMultipleSymbolsWithNoSizeImpl<GsymCreatorV1>(1, 4, 4);
}
TEST(GSYMTest, TestGsymCreatorV2MultipleSymbolsWithNoSize) {
  TestGsymCreatorMultipleSymbolsWithNoSizeImpl<GsymCreatorV2>(1, 8, 8);
}

// Helper function to quickly create a FunctionInfo in a GsymCreator for testing.
static void AddFunctionInfo(GsymCreator &GC, const char *FuncName,
                            uint64_t FuncAddr, const char *SourcePath,
                            const char *HeaderPath) {
  FunctionInfo FI(FuncAddr, 0x30, GC.insertString(FuncName));
  FI.OptLineTable = LineTable();
  const uint32_t SourceFileIdx = GC.insertFile(SourcePath);
  const uint32_t HeaderFileIdx = GC.insertFile(HeaderPath);
  FI.OptLineTable->push(LineEntry(FuncAddr+0x00, SourceFileIdx, 5));
  FI.OptLineTable->push(LineEntry(FuncAddr+0x10, HeaderFileIdx, 10));
  FI.OptLineTable->push(LineEntry(FuncAddr+0x12, HeaderFileIdx, 20));
  FI.OptLineTable->push(LineEntry(FuncAddr+0x14, HeaderFileIdx, 11));
  FI.OptLineTable->push(LineEntry(FuncAddr+0x16, HeaderFileIdx, 30));
  FI.OptLineTable->push(LineEntry(FuncAddr+0x18, HeaderFileIdx, 12));
  FI.OptLineTable->push(LineEntry(FuncAddr+0x20, SourceFileIdx, 8));
  FI.Inline = InlineInfo();

  std::string InlineName1(FuncName); InlineName1.append("1");
  std::string InlineName2(FuncName); InlineName2.append("2");
  std::string InlineName3(FuncName); InlineName3.append("3");

  FI.Inline->Name = GC.insertString(InlineName1);
  FI.Inline->CallFile = SourceFileIdx;
  FI.Inline->CallLine = 6;
  FI.Inline->Ranges.insert(AddressRange(FuncAddr + 0x10, FuncAddr + 0x20));
  InlineInfo Inline2;
  Inline2.Name = GC.insertString(InlineName2);
  Inline2.CallFile = HeaderFileIdx;
  Inline2.CallLine = 33;
  Inline2.Ranges.insert(AddressRange(FuncAddr + 0x12, FuncAddr + 0x14));
  FI.Inline->Children.emplace_back(Inline2);
  InlineInfo Inline3;
  Inline3.Name = GC.insertString(InlineName3);
  Inline3.CallFile = HeaderFileIdx;
  Inline3.CallLine = 35;
  Inline3.Ranges.insert(AddressRange(FuncAddr + 0x16, FuncAddr + 0x18));
  FI.Inline->Children.emplace_back(Inline3);
  GC.addFunctionInfo(std::move(FI));
}

// Finalize a GsymCreator, encode it and decode it and return the error or
// GsymReader that was successfully decoded.
static Expected<std::unique_ptr<GsymReader>>
FinalizeEncodeAndDecode(GsymCreator &GC) {
  OutputAggregator Null(nullptr);
  Error FinalizeErr = GC.finalize(Null);
  if (FinalizeErr)
    return std::move(FinalizeErr);
  SmallString<1024> Str;
  raw_svector_ostream OutStrm(Str);
  const auto ByteOrder = llvm::endianness::native;
  FileWriter FW(OutStrm, ByteOrder);
  FW.setStringOffsetSize(GC.getStringOffsetSize());
  llvm::Error Err = GC.encode(FW);
  if (Err)
    return std::move(Err);
  return GsymReader::copyBuffer(OutStrm.str());
}

template <typename CreatorT>
static void TestGsymSegmenting(uint64_t SegmentSize) {
  // Test creating a GSYM file with function infos and segment the information.
  // We verify segmenting is working by creating a full GSYM and also by
  // encoding multiple segments, then we verify that we get the same information
  // when doing lookups on the full GSYM that was decoded from encoding the
  // entire GSYM and also by decoding information from the segments themselves.
  CreatorT GC;
  GC.setBaseAddress(0);
  AddFunctionInfo(GC, "main", 0x1000, "/tmp/main.c", "/tmp/main.h");
  AddFunctionInfo(GC, "foo", 0x2000, "/tmp/foo.c", "/tmp/foo.h");
  AddFunctionInfo(GC, "bar", 0x3000, "/tmp/bar.c", "/tmp/bar.h");
  AddFunctionInfo(GC, "baz", 0x4000, "/tmp/baz.c", "/tmp/baz.h");
  auto GROrErr = FinalizeEncodeAndDecode(GC);
  ASSERT_THAT_EXPECTED(GROrErr, Succeeded());
  const std::unique_ptr<GsymReader> &GR = *GROrErr;

  // Create segmented GSYM files where each file contains 1 function. We will
  // then test doing lookups on the "GR", or the full GSYM file and then test
  // doing lookups on the GsymReader objects for each segment to ensure we get
  // the exact same information. So after all of the code below we will have
  // GsymReader objects that each contain one function. We name the creators
  // and readers to match the one and only address they contain.
  // GC1000 and GR1000 are for [0x1000-0x1030)
  // GC2000 and GR2000 are for [0x2000-0x2030)
  // GC3000 and GR3000 are for [0x3000-0x3030)
  // GC4000 and GR4000 are for [0x4000-0x4030)

  // Create the segments and verify that FuncIdx, an in/out parameter, gets
  // updated as expected.
  size_t FuncIdx = 0;
  // Make sure we get an error if the segment size is too small to encode a
  // single function info.
  llvm::Expected<std::unique_ptr<GsymCreator>> GCError =
      GC.createSegment(57, FuncIdx);
  ASSERT_FALSE((bool)GCError);
  checkError("a segment size of 57 is to small to fit any function infos, "
             "specify a larger value", GCError.takeError());
  // Make sure that the function index didn't get incremented when we didn't
  // encode any values into the segmented GsymCreator.
  ASSERT_EQ(FuncIdx, (size_t)0);

  llvm::Expected<std::unique_ptr<GsymCreator>> GC1000 =
      GC.createSegment(SegmentSize, FuncIdx);
  ASSERT_THAT_EXPECTED(GC1000, Succeeded());
  ASSERT_EQ(FuncIdx, (size_t)1);
  llvm::Expected<std::unique_ptr<GsymCreator>> GC2000 =
      GC.createSegment(SegmentSize, FuncIdx);
  ASSERT_THAT_EXPECTED(GC2000, Succeeded());
  ASSERT_EQ(FuncIdx, (size_t)2);
  llvm::Expected<std::unique_ptr<GsymCreator>> GC3000 =
      GC.createSegment(SegmentSize, FuncIdx);
  ASSERT_THAT_EXPECTED(GC3000, Succeeded());
  ASSERT_EQ(FuncIdx, (size_t)3);
  llvm::Expected<std::unique_ptr<GsymCreator>> GC4000 =
      GC.createSegment(SegmentSize, FuncIdx);
  ASSERT_THAT_EXPECTED(GC4000, Succeeded());
  ASSERT_EQ(FuncIdx, (size_t)4);
  // When there are no function infos left to encode we expect to get  no error
  // and get a NULL GsymCreator in the return value from createSegment.
  llvm::Expected<std::unique_ptr<GsymCreator>> GCNull =
      GC.createSegment(SegmentSize, FuncIdx);
  ASSERT_THAT_EXPECTED(GCNull, Succeeded());
  ASSERT_TRUE(GC1000.get() != nullptr);
  ASSERT_TRUE(GC2000.get() != nullptr);
  ASSERT_TRUE(GC3000.get() != nullptr);
  ASSERT_TRUE(GC4000.get() != nullptr);
  ASSERT_TRUE(GCNull.get() == nullptr);
  // Encode and decode the GsymReader for each segment and verify they succeed.
  Expected<std::unique_ptr<GsymReader>> GR1000 =
      FinalizeEncodeAndDecode(*GC1000.get());
  ASSERT_THAT_EXPECTED(GR1000, Succeeded());
  Expected<std::unique_ptr<GsymReader>> GR2000 =
      FinalizeEncodeAndDecode(*GC2000.get());
  ASSERT_THAT_EXPECTED(GR2000, Succeeded());
  Expected<std::unique_ptr<GsymReader>> GR3000 =
      FinalizeEncodeAndDecode(*GC3000.get());
  ASSERT_THAT_EXPECTED(GR3000, Succeeded());
  Expected<std::unique_ptr<GsymReader>> GR4000 =
      FinalizeEncodeAndDecode(*GC4000.get());
  ASSERT_THAT_EXPECTED(GR4000, Succeeded());

  // Verify that all lookups match the range [0x1000-0x1030) when doing lookups
  // in the GsymReader that contains all functions and from the segmented
  // GsymReader in GR1000.
  for (uint64_t Addr = 0x1000; Addr < 0x1030; ++Addr) {
    // Lookup in the main GsymReader that contains all function infos
    auto MainLR = GR->lookup(Addr);
    ASSERT_THAT_EXPECTED(MainLR, Succeeded());
    auto SegmentLR = (*GR1000)->lookup(Addr);
    ASSERT_THAT_EXPECTED(SegmentLR, Succeeded());
    // Make sure the lookup results match.
    EXPECT_EQ(MainLR.get(), SegmentLR.get());
    // Make sure that the lookups on the functions that are not in the segment
    // fail as expected.
    ASSERT_THAT_EXPECTED((*GR1000)->lookup(0x2000), Failed());
    ASSERT_THAT_EXPECTED((*GR1000)->lookup(0x3000), Failed());
    ASSERT_THAT_EXPECTED((*GR1000)->lookup(0x4000), Failed());
  }

  // Verify that all lookups match the range [0x2000-0x2030) when doing lookups
  // in the GsymReader that contains all functions and from the segmented
  // GsymReader in GR2000.
  for (uint64_t Addr = 0x2000; Addr < 0x2030; ++Addr) {
    // Lookup in the main GsymReader that contains all function infos
    auto MainLR = GR->lookup(Addr);
    ASSERT_THAT_EXPECTED(MainLR, Succeeded());
    auto SegmentLR = (*GR2000)->lookup(Addr);
    ASSERT_THAT_EXPECTED(SegmentLR, Succeeded());
    // Make sure the lookup results match.
    EXPECT_EQ(MainLR.get(), SegmentLR.get());
    // Make sure that the lookups on the functions that are not in the segment
    // fail as expected.
    ASSERT_THAT_EXPECTED((*GR2000)->lookup(0x1000), Failed());
    ASSERT_THAT_EXPECTED((*GR2000)->lookup(0x3000), Failed());
    ASSERT_THAT_EXPECTED((*GR2000)->lookup(0x4000), Failed());
  }

  // Verify that all lookups match the range [0x3000-0x3030) when doing lookups
  // in the GsymReader that contains all functions and from the segmented
  // GsymReader in GR3000.
  for (uint64_t Addr = 0x3000; Addr < 0x3030; ++Addr) {
    // Lookup in the main GsymReader that contains all function infos
    auto MainLR = GR->lookup(Addr);
    ASSERT_THAT_EXPECTED(MainLR, Succeeded());
    auto SegmentLR = (*GR3000)->lookup(Addr);
    ASSERT_THAT_EXPECTED(SegmentLR, Succeeded());
    // Make sure the lookup results match.
    EXPECT_EQ(MainLR.get(), SegmentLR.get());
    // Make sure that the lookups on the functions that are not in the segment
    // fail as expected.
    ASSERT_THAT_EXPECTED((*GR3000)->lookup(0x1000), Failed());
    ASSERT_THAT_EXPECTED((*GR3000)->lookup(0x2000), Failed());
    ASSERT_THAT_EXPECTED((*GR3000)->lookup(0x4000), Failed());
  }

  // Verify that all lookups match the range [0x4000-0x4030) when doing lookups
  // in the GsymReader that contains all functions and from the segmented
  // GsymReader in GR4000.
  for (uint64_t Addr = 0x4000; Addr < 0x4030; ++Addr) {
    // Lookup in the main GsymReader that contains all function infos
    auto MainLR = GR->lookup(Addr);
    ASSERT_THAT_EXPECTED(MainLR, Succeeded());
    // Lookup in the GsymReader for that contains 0x4000
    auto SegmentLR = (*GR4000)->lookup(Addr);
    ASSERT_THAT_EXPECTED(SegmentLR, Succeeded());
    // Make sure the lookup results match.
    EXPECT_EQ(MainLR.get(), SegmentLR.get());
    // Make sure that the lookups on the functions that are not in the segment
    // fail as expected.
    ASSERT_THAT_EXPECTED((*GR4000)->lookup(0x1000), Failed());
    ASSERT_THAT_EXPECTED((*GR4000)->lookup(0x2000), Failed());
    ASSERT_THAT_EXPECTED((*GR4000)->lookup(0x3000), Failed());
  }
}

TEST(GSYMTest, TestGsymSegmenting) { TestGsymSegmenting<GsymCreatorV1>(128); }
TEST(GSYMTest, TestGsymSegmentingV2) { TestGsymSegmenting<GsymCreatorV2>(256); }

template <typename CreatorT>
static void TestGsymSegmentingNoBase(uint64_t SegmentSize) {
  // Test creating a GSYM file with function infos and segment the information.
  // We verify segmenting is working by creating a full GSYM and also by
  // encoding multiple segments, then we verify that we get the same information
  // when doing lookups on the full GSYM that was decoded from encoding the
  // entire GSYM and also by decoding information from the segments themselves.
  CreatorT GC;
  AddFunctionInfo(GC, "main", 0x1000, "/tmp/main.c", "/tmp/main.h");
  AddFunctionInfo(GC, "foo", 0x2000, "/tmp/foo.c", "/tmp/foo.h");
  AddFunctionInfo(GC, "bar", 0x3000, "/tmp/bar.c", "/tmp/bar.h");
  AddFunctionInfo(GC, "baz", 0x4000, "/tmp/baz.c", "/tmp/baz.h");
  auto GROrErr = FinalizeEncodeAndDecode(GC);
  ASSERT_THAT_EXPECTED(GROrErr, Succeeded());
  const std::unique_ptr<GsymReader> &GR = *GROrErr;

  // Create segmented GSYM files where each file contains 1 function. We will
  // then test doing lookups on the "GR", or the full GSYM file and then test
  // doing lookups on the GsymReader objects for each segment to ensure we get
  // the exact same information. So after all of the code below we will have
  // GsymReader objects that each contain one function. We name the creators
  // and readers to match the one and only address they contain.
  // GC1000 and GR1000 are for [0x1000-0x1030)
  // GC2000 and GR2000 are for [0x2000-0x2030)
  // GC3000 and GR3000 are for [0x3000-0x3030)
  // GC4000 and GR4000 are for [0x4000-0x4030)

  // Create the segments and verify that FuncIdx, an in/out parameter, gets
  // updated as expected.
  size_t FuncIdx = 0;
  // Make sure we get an error if the segment size is too small to encode a
  // single function info.
  llvm::Expected<std::unique_ptr<GsymCreator>> GCError =
      GC.createSegment(57, FuncIdx);
  ASSERT_FALSE((bool)GCError);
  checkError("a segment size of 57 is to small to fit any function infos, "
             "specify a larger value", GCError.takeError());
  // Make sure that the function index didn't get incremented when we didn't
  // encode any values into the segmented GsymCreator.
  ASSERT_EQ(FuncIdx, (size_t)0);

  llvm::Expected<std::unique_ptr<GsymCreator>> GC1000 =
      GC.createSegment(SegmentSize, FuncIdx);
  ASSERT_THAT_EXPECTED(GC1000, Succeeded());
  ASSERT_EQ(FuncIdx, (size_t)1);
  llvm::Expected<std::unique_ptr<GsymCreator>> GC2000 =
      GC.createSegment(SegmentSize, FuncIdx);
  ASSERT_THAT_EXPECTED(GC2000, Succeeded());
  ASSERT_EQ(FuncIdx, (size_t)2);
  llvm::Expected<std::unique_ptr<GsymCreator>> GC3000 =
      GC.createSegment(SegmentSize, FuncIdx);
  ASSERT_THAT_EXPECTED(GC3000, Succeeded());
  ASSERT_EQ(FuncIdx, (size_t)3);
  llvm::Expected<std::unique_ptr<GsymCreator>> GC4000 =
      GC.createSegment(SegmentSize, FuncIdx);
  ASSERT_THAT_EXPECTED(GC4000, Succeeded());
  ASSERT_EQ(FuncIdx, (size_t)4);
  // When there are no function infos left to encode we expect to get  no error
  // and get a NULL GsymCreator in the return value from createSegment.
  llvm::Expected<std::unique_ptr<GsymCreator>> GCNull =
      GC.createSegment(SegmentSize, FuncIdx);
  ASSERT_THAT_EXPECTED(GCNull, Succeeded());
  ASSERT_TRUE(GC1000.get() != nullptr);
  ASSERT_TRUE(GC2000.get() != nullptr);
  ASSERT_TRUE(GC3000.get() != nullptr);
  ASSERT_TRUE(GC4000.get() != nullptr);
  ASSERT_TRUE(GCNull.get() == nullptr);
  // Encode and decode the GsymReader for each segment and verify they succeed.
  Expected<std::unique_ptr<GsymReader>> GR1000 =
      FinalizeEncodeAndDecode(*GC1000.get());
  ASSERT_THAT_EXPECTED(GR1000, Succeeded());
  Expected<std::unique_ptr<GsymReader>> GR2000 =
      FinalizeEncodeAndDecode(*GC2000.get());
  ASSERT_THAT_EXPECTED(GR2000, Succeeded());
  Expected<std::unique_ptr<GsymReader>> GR3000 =
      FinalizeEncodeAndDecode(*GC3000.get());
  ASSERT_THAT_EXPECTED(GR3000, Succeeded());
  Expected<std::unique_ptr<GsymReader>> GR4000 =
      FinalizeEncodeAndDecode(*GC4000.get());
  ASSERT_THAT_EXPECTED(GR4000, Succeeded());

  // Verify that all lookups match the range [0x1000-0x1030) when doing lookups
  // in the GsymReader that contains all functions and from the segmented
  // GsymReader in GR1000.
  for (uint64_t Addr = 0x1000; Addr < 0x1030; ++Addr) {
    // Lookup in the main GsymReader that contains all function infos
    auto MainLR = GR->lookup(Addr);
    ASSERT_THAT_EXPECTED(MainLR, Succeeded());
    auto SegmentLR = (*GR1000)->lookup(Addr);
    ASSERT_THAT_EXPECTED(SegmentLR, Succeeded());
    // Make sure the lookup results match.
    EXPECT_EQ(MainLR.get(), SegmentLR.get());
    // Make sure that the lookups on the functions that are not in the segment
    // fail as expected.
    ASSERT_THAT_EXPECTED((*GR1000)->lookup(0x2000), Failed());
    ASSERT_THAT_EXPECTED((*GR1000)->lookup(0x3000), Failed());
    ASSERT_THAT_EXPECTED((*GR1000)->lookup(0x4000), Failed());
  }

  // Verify that all lookups match the range [0x2000-0x2030) when doing lookups
  // in the GsymReader that contains all functions and from the segmented
  // GsymReader in GR2000.
  for (uint64_t Addr = 0x2000; Addr < 0x2030; ++Addr) {
    // Lookup in the main GsymReader that contains all function infos
    auto MainLR = GR->lookup(Addr);
    ASSERT_THAT_EXPECTED(MainLR, Succeeded());
    auto SegmentLR = (*GR2000)->lookup(Addr);
    ASSERT_THAT_EXPECTED(SegmentLR, Succeeded());
    // Make sure the lookup results match.
    EXPECT_EQ(MainLR.get(), SegmentLR.get());
    // Make sure that the lookups on the functions that are not in the segment
    // fail as expected.
    ASSERT_THAT_EXPECTED((*GR2000)->lookup(0x1000), Failed());
    ASSERT_THAT_EXPECTED((*GR2000)->lookup(0x3000), Failed());
    ASSERT_THAT_EXPECTED((*GR2000)->lookup(0x4000), Failed());
  }

  // Verify that all lookups match the range [0x3000-0x3030) when doing lookups
  // in the GsymReader that contains all functions and from the segmented
  // GsymReader in GR3000.
  for (uint64_t Addr = 0x3000; Addr < 0x3030; ++Addr) {
    // Lookup in the main GsymReader that contains all function infos
    auto MainLR = GR->lookup(Addr);
    ASSERT_THAT_EXPECTED(MainLR, Succeeded());
    auto SegmentLR = (*GR3000)->lookup(Addr);
    ASSERT_THAT_EXPECTED(SegmentLR, Succeeded());
    // Make sure the lookup results match.
    EXPECT_EQ(MainLR.get(), SegmentLR.get());
    // Make sure that the lookups on the functions that are not in the segment
    // fail as expected.
    ASSERT_THAT_EXPECTED((*GR3000)->lookup(0x1000), Failed());
    ASSERT_THAT_EXPECTED((*GR3000)->lookup(0x2000), Failed());
    ASSERT_THAT_EXPECTED((*GR3000)->lookup(0x4000), Failed());
  }

  // Verify that all lookups match the range [0x4000-0x4030) when doing lookups
  // in the GsymReader that contains all functions and from the segmented
  // GsymReader in GR4000.
  for (uint64_t Addr = 0x4000; Addr < 0x4030; ++Addr) {
    // Lookup in the main GsymReader that contains all function infos
    auto MainLR = GR->lookup(Addr);
    ASSERT_THAT_EXPECTED(MainLR, Succeeded());
    // Lookup in the GsymReader for that contains 0x4000
    auto SegmentLR = (*GR4000)->lookup(Addr);
    ASSERT_THAT_EXPECTED(SegmentLR, Succeeded());
    // Make sure the lookup results match.
    EXPECT_EQ(MainLR.get(), SegmentLR.get());
    // Make sure that the lookups on the functions that are not in the segment
    // fail as expected.
    ASSERT_THAT_EXPECTED((*GR4000)->lookup(0x1000), Failed());
    ASSERT_THAT_EXPECTED((*GR4000)->lookup(0x2000), Failed());
    ASSERT_THAT_EXPECTED((*GR4000)->lookup(0x3000), Failed());
  }
}

TEST(GSYMTest, TestGsymSegmentingNoBase) {
  TestGsymSegmentingNoBase<GsymCreatorV1>(128);
}
TEST(GSYMTest, TestGsymSegmentingNoBaseV2) {
  TestGsymSegmentingNoBase<GsymCreatorV2>(256);
}

template <typename CreatorT> static void TestDWARFInlineRangeScopes() {
  // Test cases where inlined functions address ranges are not contained in the
  // parent ranges and that we can successfully remove them and emit error
  // messages. The DWARF for this looks like the dump below. The inlined
  // functions named "invalid1" and "invalid2" are expected to be removed and
  // an appropriate error message will be emitted.
  //
  // 0x0000000b: DW_TAG_compile_unit
  //               DW_AT_name	("/tmp/main.cpp")
  //               DW_AT_language	(DW_LANG_C)
  //               DW_AT_stmt_list	(0x00000000)
  //
  // 0x00000015:   DW_TAG_subprogram
  //                 DW_AT_name	("foo")
  //                 DW_AT_low_pc	(0x0000000000001000)
  //                 DW_AT_high_pc	(0x0000000000002000)
  //
  // 0x0000002a:     DW_TAG_inlined_subroutine
  //                   DW_AT_name	("invalid1")
  //                   DW_AT_low_pc	(0x0000000000000fff)
  //                   DW_AT_high_pc	(0x0000000000001001)
  //                   DW_AT_call_file	("/tmp/main.cpp")
  //                   DW_AT_call_line	(10)
  //
  // 0x00000041:     DW_TAG_inlined_subroutine
  //                   DW_AT_name	("valid1")
  //                   DW_AT_low_pc	(0x0000000000001010)
  //                   DW_AT_high_pc	(0x0000000000001100)
  //                   DW_AT_call_file	("/tmp/main.cpp")
  //                   DW_AT_call_line	(11)
  //
  // 0x00000058:       DW_TAG_inlined_subroutine
  //                     DW_AT_name	("invalid2")
  //                     DW_AT_low_pc	(0x0000000000001000)
  //                     DW_AT_high_pc	(0x0000000000001100)
  //                     DW_AT_call_file	("/tmp/main.cpp")
  //                     DW_AT_call_line	(12)
  //
  // 0x0000006f:       DW_TAG_inlined_subroutine
  //                     DW_AT_name	("valid2")
  //                     DW_AT_low_pc	(0x0000000000001020)
  //                     DW_AT_high_pc	(0x0000000000001030)
  //                     DW_AT_call_file	("/tmp/main.cpp")
  //                     DW_AT_call_line	(13)
  //
  // 0x00000086:       NULL
  //
  // 0x00000087:     NULL
  //
  // 0x00000088:   NULL

  StringRef yamldata = R"(
  debug_str:
    - ''
    - '/tmp/main.cpp'
    - foo
    - invalid1
    - valid1
    - invalid2
    - valid2
  debug_abbrev:
    - ID:              0
      Table:
        - Code:            0x1
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_language
              Form:            DW_FORM_udata
            - Attribute:       DW_AT_stmt_list
              Form:            DW_FORM_sec_offset
        - Code:            0x2
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_low_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_high_pc
              Form:            DW_FORM_addr
        - Code:            0x3
          Tag:             DW_TAG_inlined_subroutine
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_low_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_high_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_call_file
              Form:            DW_FORM_data1
            - Attribute:       DW_AT_call_line
              Form:            DW_FORM_data1
        - Code:            0x4
          Tag:             DW_TAG_inlined_subroutine
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_low_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_high_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_call_file
              Form:            DW_FORM_data1
            - Attribute:       DW_AT_call_line
              Form:            DW_FORM_data1
  debug_info:
    - Length:          0x85
      Version:         4
      AbbrevTableID:   0
      AbbrOffset:      0x0
      AddrSize:        8
      Entries:
        - AbbrCode:        0x1
          Values:
            - Value:           0x1
            - Value:           0x2
            - Value:           0x0
        - AbbrCode:        0x2
          Values:
            - Value:           0xF
            - Value:           0x1000
            - Value:           0x2000
        - AbbrCode:        0x3
          Values:
            - Value:           0x13
            - Value:           0xFFF
            - Value:           0x1001
            - Value:           0x1
            - Value:           0xA
        - AbbrCode:        0x4
          Values:
            - Value:           0x1C
            - Value:           0x1010
            - Value:           0x1100
            - Value:           0x1
            - Value:           0xB
        - AbbrCode:        0x3
          Values:
            - Value:           0x23
            - Value:           0x1000
            - Value:           0x1100
            - Value:           0x1
            - Value:           0xC
        - AbbrCode:        0x3
          Values:
            - Value:           0x2C
            - Value:           0x1020
            - Value:           0x1030
            - Value:           0x1
            - Value:           0xD
        - AbbrCode:        0x0
        - AbbrCode:        0x0
        - AbbrCode:        0x0
  debug_line:
    - Length:          84
      Version:         2
      PrologueLength:  36
      MinInstLength:   1
      DefaultIsStmt:   1
      LineBase:        251
      LineRange:       14
      OpcodeBase:      13
      StandardOpcodeLengths: [ 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1 ]
      IncludeDirs:
        - '/tmp'
      Files:
        - Name:            main.cpp
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
          Data:            0
        - Opcode:          DW_LNS_copy
          Data:            0
        - Opcode:          DW_LNS_advance_pc
          Data:            16
        - Opcode:          DW_LNS_advance_line
          SData:           1
          Data:            0
        - Opcode:          DW_LNS_copy
          Data:            0
        - Opcode:          DW_LNS_advance_pc
          Data:            16
        - Opcode:          DW_LNS_advance_line
          SData:           1
          Data:            0
        - Opcode:          DW_LNS_copy
          Data:            0
        - Opcode:          DW_LNS_advance_pc
          Data:            16
        - Opcode:          DW_LNS_advance_line
          SData:           1
          Data:            0
        - Opcode:          DW_LNS_copy
          Data:            0
        - Opcode:          DW_LNS_advance_pc
          Data:            4048
        - Opcode:          DW_LNS_advance_line
          SData:           1
          Data:            0
        - Opcode:          DW_LNS_copy
          Data:            0
        - Opcode:          DW_LNS_advance_pc
          Data:            16
        - Opcode:          DW_LNS_advance_line
          SData:           -1
          Data:            0
        - Opcode:          DW_LNS_extended_op
          ExtLen:          1
          SubOpcode:       DW_LNE_end_sequence
          Data:            0
  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(yamldata);
  ASSERT_THAT_EXPECTED(ErrOrSections, Succeeded());
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);
  ASSERT_TRUE(DwarfContext.get() != nullptr);
  std::string errors;
  raw_string_ostream OS(errors);
  OutputAggregator OSAgg(&OS);
  CreatorT GC;
  DwarfTransformer DT(*DwarfContext, GC);
  const uint32_t ThreadCount = 1;
  ASSERT_THAT_ERROR(DT.convert(ThreadCount, OSAgg), Succeeded());
  ASSERT_THAT_ERROR(GC.finalize(OSAgg), Succeeded());
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  const auto ByteOrder = llvm::endianness::native;
  FileWriter FW(OutStrm, ByteOrder);
  FW.setStringOffsetSize(GC.getStringOffsetSize());
  ASSERT_THAT_ERROR(GC.encode(FW), Succeeded());
  auto GROrErr = GsymReader::copyBuffer(OutStrm.str());
  ASSERT_THAT_EXPECTED(GROrErr, Succeeded());
  const std::unique_ptr<GsymReader> &GR = *GROrErr;
  // There should only be one function in our GSYM.
  EXPECT_EQ(GR->getNumAddresses(), 1u);
  auto ExpFI = GR->getFunctionInfo(0x1000);
  ASSERT_THAT_EXPECTED(ExpFI, Succeeded());
  ASSERT_EQ(ExpFI->Range, AddressRange(0x1000, 0x2000));
  EXPECT_TRUE(ExpFI->OptLineTable.has_value());
  EXPECT_TRUE(ExpFI->Inline.has_value());
  StringRef FuncName = GR->getString(ExpFI->Name);
  EXPECT_EQ(FuncName, "foo");
  std::vector<std::string> ExpectedLogErrors = {
    "error: inlined function DIE at 0x0000002a has a range [0x0000000000000fff "
    "- 0x0000000000001001) that isn't contained in any parent address ranges, "
    "this inline range will be removed.",
    "error: inlined function DIE at 0x00000058 has a range [0x0000000000001000 "
    "- 0x0000000000001100) that isn't contained in any parent address ranges, "
    "this inline range will be removed."
  };
  // Make sure all expected errors are in the error stream for the two invalid
  // inlined functions that we removed due to invalid range scoping.
  for (const auto &Error: ExpectedLogErrors) {
    EXPECT_TRUE(OS.str().find(Error) != std::string::npos);
  }
  // The top level inline info is for the function "foo" itself. Verify that
  // we have only 1 inline function inside of this, even though the DWARF
  // contains two. One of the inline functions in "foo" is invalid, so we must
  // only end up with 1.
  StringRef InlineFuncName = GR->getString(ExpFI->Inline->Name);
  EXPECT_EQ(InlineFuncName, "foo");
  EXPECT_EQ(ExpFI->Inline->CallFile, 0u);
  EXPECT_EQ(ExpFI->Inline->CallLine, 0u);
  EXPECT_EQ(ExpFI->Inline->Children.size(), 1u);


  // The first inline function "valid1" contains two inline functions in the
  // DWARF, but one has an address range which isn't contained in any ranges
  // from "foo", so only 1 inline function be parsed.
  InlineInfo &Inline1 = ExpFI->Inline->Children[0];
  StringRef Inline1Name = GR->getString(Inline1.Name);
  EXPECT_EQ(Inline1Name, "valid1");
  EXPECT_EQ(Inline1.CallFile, 1u);
  EXPECT_EQ(Inline1.CallLine, 11u);
  EXPECT_EQ(Inline1.Children.size(), 1u);


  // The second inline function "valid2" contains two inline functions in the
  // DWARF, but one has an address range which isn't contained in any ranges
  // from "valid1", so only 1 inline function be parsed.
  InlineInfo &Inline2 = Inline1.Children[0];
  StringRef Inline2Name = GR->getString(Inline2.Name);
  EXPECT_EQ(Inline2Name, "valid2");
  EXPECT_EQ(Inline2.CallFile, 1u);
  EXPECT_EQ(Inline2.CallLine, 13u);
  EXPECT_EQ(Inline2.Children.size(), 0u);
}

TEST(GSYMTest, TestDWARFInlineRangeScopes) {
  TestDWARFInlineRangeScopes<GsymCreatorV1>();
}
TEST(GSYMTest, TestDWARFInlineRangeScopesV2) {
  TestDWARFInlineRangeScopes<GsymCreatorV2>();
}

template <typename CreatorT> static void TestDWARFEmptyInline() {
  // Test cases where we have inline function information in the DWARF that
  // results in us trying to parse the inline info, but since the inline
  // info ends up not adding any valid inline functions due to ranges
  // not being correct, we end up not encoding any inline information. This
  // tests that if we end up creating an empty inline info struct, we end up
  // not encoding it into the GSYM file.
  //
  // 0x0000000b: DW_TAG_compile_unit
  //               DW_AT_name	("/tmp/main.cpp")
  //               DW_AT_language	(DW_LANG_C)
  //               DW_AT_stmt_list	(0x00000000)
  //
  // 0x00000015:   DW_TAG_subprogram
  //                 DW_AT_name	("foo")
  //                 DW_AT_low_pc	(0x0000000000001000)
  //                 DW_AT_high_pc	(0x0000000000001050)
  //
  // 0x0000002a:     DW_TAG_inlined_subroutine
  //                   DW_AT_name	("inlineWithInvalidRange")
  //                   DW_AT_low_pc	(0x0000000000001100)
  //                   DW_AT_high_pc	(0x0000000000001200)
  //                   DW_AT_call_file	("/tmp/main.cpp")
  //                   DW_AT_call_line	(11)
  //
  // 0x00000047:     NULL
  //
  // 0x00000048:   NULL

  StringRef yamldata = R"(
  debug_str:
    - ''
    - '/tmp/main.cpp'
    - foo
    - inlineWithInvalidRange
  debug_abbrev:
    - ID:              0
      Table:
        - Code:            0x1
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_language
              Form:            DW_FORM_udata
            - Attribute:       DW_AT_stmt_list
              Form:            DW_FORM_sec_offset
        - Code:            0x2
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_low_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_high_pc
              Form:            DW_FORM_addr
        - Code:            0x3
          Tag:             DW_TAG_inlined_subroutine
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_low_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_high_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_call_file
              Form:            DW_FORM_data4
            - Attribute:       DW_AT_call_line
              Form:            DW_FORM_data4
  debug_info:
    - Length:          0x45
      Version:         4
      AbbrevTableID:   0
      AbbrOffset:      0x0
      AddrSize:        8
      Entries:
        - AbbrCode:        0x1
          Values:
            - Value:           0x1
            - Value:           0x2
            - Value:           0x0
        - AbbrCode:        0x2
          Values:
            - Value:           0xF
            - Value:           0x1000
            - Value:           0x1050
        - AbbrCode:        0x3
          Values:
            - Value:           0x13
            - Value:           0x1100
            - Value:           0x1200
            - Value:           0x1
            - Value:           0xB
        - AbbrCode:        0x0
        - AbbrCode:        0x0
  debug_line:
    - Length:          76
      Version:         2
      PrologueLength:  36
      MinInstLength:   1
      DefaultIsStmt:   1
      LineBase:        251
      LineRange:       14
      OpcodeBase:      13
      StandardOpcodeLengths: [ 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1 ]
      IncludeDirs:
        - '/tmp'
      Files:
        - Name:            main.cpp
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
          Data:            0
        - Opcode:          DW_LNS_copy
          Data:            0
        - Opcode:          DW_LNS_advance_pc
          Data:            16
        - Opcode:          DW_LNS_advance_line
          SData:           1
          Data:            0
        - Opcode:          DW_LNS_copy
          Data:            0
        - Opcode:          DW_LNS_advance_pc
          Data:            16
        - Opcode:          DW_LNS_advance_line
          SData:           1
          Data:            0
        - Opcode:          DW_LNS_copy
          Data:            0
        - Opcode:          DW_LNS_advance_pc
          Data:            16
        - Opcode:          DW_LNS_advance_line
          SData:           1
          Data:            0
        - Opcode:          DW_LNS_copy
          Data:            0
        - Opcode:          DW_LNS_advance_pc
          Data:            32
        - Opcode:          DW_LNS_extended_op
          ExtLen:          1
          SubOpcode:       DW_LNE_end_sequence
          Data:            0
  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(yamldata);
  ASSERT_THAT_EXPECTED(ErrOrSections, Succeeded());
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);
  ASSERT_TRUE(DwarfContext.get() != nullptr);
  std::string errors;
  raw_string_ostream OS(errors);
  OutputAggregator OSAgg(&OS);
  CreatorT GC;
  DwarfTransformer DT(*DwarfContext, GC);
  const uint32_t ThreadCount = 1;
  ASSERT_THAT_ERROR(DT.convert(ThreadCount, OSAgg), Succeeded());
  ASSERT_THAT_ERROR(GC.finalize(OSAgg), Succeeded());
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  const auto ByteOrder = llvm::endianness::native;
  FileWriter FW(OutStrm, ByteOrder);
  FW.setStringOffsetSize(GC.getStringOffsetSize());
  ASSERT_THAT_ERROR(GC.encode(FW), Succeeded());
  auto GROrErr = GsymReader::copyBuffer(OutStrm.str());
  ASSERT_THAT_EXPECTED(GROrErr, Succeeded());
  const std::unique_ptr<GsymReader> &GR = *GROrErr;
  // There should only be one function in our GSYM.
  EXPECT_EQ(GR->getNumAddresses(), 1u);
  auto ExpFI = GR->getFunctionInfo(0x1000);
  ASSERT_THAT_EXPECTED(ExpFI, Succeeded());
  ASSERT_EQ(ExpFI->Range, AddressRange(0x1000, 0x1050));
  EXPECT_TRUE(ExpFI->OptLineTable.has_value());
  EXPECT_FALSE(ExpFI->Inline.has_value());
  StringRef FuncName = GR->getString(ExpFI->Name);
  EXPECT_EQ(FuncName, "foo");
  std::vector<std::string> ExpectedLogErrors = {
    "error: inlined function DIE at 0x0000002a has a range [0x0000000000001100"
    " - 0x0000000000001200) that isn't contained in any parent address ranges,"
    " this inline range will be removed.",
    "warning: DIE contains inline function information that has no valid "
    "ranges, removing inline information:",
  };
  // Make sure all expected errors are in the error stream for the two invalid
  // inlined functions that we removed due to invalid range scoping.
  for (const auto &Error: ExpectedLogErrors) {
    EXPECT_TRUE(OS.str().find(Error) != std::string::npos);
  }
}

TEST(GSYMTest, TestDWARFEmptyInline) { TestDWARFEmptyInline<GsymCreatorV1>(); }
TEST(GSYMTest, TestDWARFEmptyInlineV2) {
  TestDWARFEmptyInline<GsymCreatorV2>();
}

template <typename CreatorT> static void TestFinalizeForLineTables() {
  // This example has two compile units:
  // - one contains a function "foo" with line table entries and "bar" without
  // - one contains a function "bar" with line table entries and "foo" without
  // This test ensures that no matter what order information gets processed,
  // we want to make sure that we prioritize the entries with the most debug
  // info.
  //
  // The DWARF is the same for the functions, but the first compile unit has
  // lines entries for "foo" and the second one doesn't. And the first compile
  // unit has no line entries for "bar", but the second one does. We expect the
  // resulting gsym file to have a "foo" and "bar" that both have line entries.
  //
  // 0x0000000b: DW_TAG_compile_unit
  //               DW_AT_name	("/tmp/main.cpp")
  //               DW_AT_language	(DW_LANG_C)
  //               DW_AT_stmt_list	(0x00000000)
  //
  // 0x00000015:   DW_TAG_subprogram
  //                 DW_AT_name	("foo")
  //                 DW_AT_low_pc	(0x0000000000001000)
  //                 DW_AT_high_pc	(0x0000000000001050)
  //
  // 0x0000002a:   DW_TAG_subprogram
  //                 DW_AT_name	("bar")
  //                 DW_AT_low_pc	(0x0000000000002000)
  //                 DW_AT_high_pc	(0x0000000000002050)
  //
  // 0x0000003f:   NULL
  // 0x00000040: Compile Unit: length = 0x0000003c, format = DWARF32, version = 0x0004, abbr_offset = 0x0000, addr_size = 0x08 (next unit at 0x00000080)
  //
  // 0x0000004b: DW_TAG_compile_unit
  //               DW_AT_name	("/tmp/main.cpp")
  //               DW_AT_language	(DW_LANG_C)
  //               DW_AT_stmt_list	(0x00000043)
  //
  // 0x00000055:   DW_TAG_subprogram
  //                 DW_AT_name	("foo")
  //                 DW_AT_low_pc	(0x0000000000001000)
  //                 DW_AT_high_pc	(0x0000000000001050)
  //
  // 0x0000006a:   DW_TAG_subprogram
  //                 DW_AT_name	("bar")
  //                 DW_AT_low_pc	(0x0000000000002000)
  //                 DW_AT_high_pc	(0x0000000000002050)
  //
  // 0x0000007f:   NULL

  StringRef yamldata = R"(
  debug_str:
    - ''
    - '/tmp/main.cpp'
    - foo
    - bar
  debug_abbrev:
    - ID:              0
      Table:
        - Code:            0x1
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_language
              Form:            DW_FORM_udata
            - Attribute:       DW_AT_stmt_list
              Form:            DW_FORM_sec_offset
        - Code:            0x2
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
    - Length:          0x3C
      Version:         4
      AbbrevTableID:   0
      AbbrOffset:      0x0
      AddrSize:        8
      Entries:
        - AbbrCode:        0x1
          Values:
            - Value:           0x1
            - Value:           0x2
            - Value:           0x0
        - AbbrCode:        0x2
          Values:
            - Value:           0xF
            - Value:           0x1000
            - Value:           0x1050
        - AbbrCode:        0x2
          Values:
            - Value:           0x13
            - Value:           0x2000
            - Value:           0x2050
        - AbbrCode:        0x0
    - Length:          0x3C
      Version:         4
      AbbrevTableID:   0
      AbbrOffset:      0x0
      AddrSize:        8
      Entries:
        - AbbrCode:        0x1
          Values:
            - Value:           0x1
            - Value:           0x2
            - Value:           0x43
        - AbbrCode:        0x2
          Values:
            - Value:           0xF
            - Value:           0x1000
            - Value:           0x1050
        - AbbrCode:        0x2
          Values:
            - Value:           0x13
            - Value:           0x2000
            - Value:           0x2050
        - AbbrCode:        0x0
  debug_line:
    - Length:          63
      Version:         2
      PrologueLength:  36
      MinInstLength:   1
      DefaultIsStmt:   1
      LineBase:        251
      LineRange:       14
      OpcodeBase:      13
      StandardOpcodeLengths: [ 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1 ]
      IncludeDirs:
        - '/tmp'
      Files:
        - Name:            main.cpp
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
          Data:            0
        - Opcode:          DW_LNS_copy
          Data:            0
        - Opcode:          DW_LNS_advance_pc
          Data:            80
        - Opcode:          DW_LNS_advance_line
          SData:           1
          Data:            0
        - Opcode:          DW_LNS_extended_op
          ExtLen:          1
          SubOpcode:       DW_LNE_end_sequence
          Data:            0
    - Length:          63
      Version:         2
      PrologueLength:  36
      MinInstLength:   1
      DefaultIsStmt:   1
      LineBase:        251
      LineRange:       14
      OpcodeBase:      13
      StandardOpcodeLengths: [ 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1 ]
      IncludeDirs:
        - '/tmp'
      Files:
        - Name:            main.cpp
          DirIdx:          1
          ModTime:         0
          Length:          0
      Opcodes:
        - Opcode:          DW_LNS_extended_op
          ExtLen:          9
          SubOpcode:       DW_LNE_set_address
          Data:            8192
        - Opcode:          DW_LNS_advance_line
          SData:           19
          Data:            0
        - Opcode:          DW_LNS_copy
          Data:            0
        - Opcode:          DW_LNS_advance_pc
          Data:            80
        - Opcode:          DW_LNS_advance_line
          SData:           1
          Data:            0
        - Opcode:          DW_LNS_extended_op
          ExtLen:          1
          SubOpcode:       DW_LNE_end_sequence
          Data:            0
  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(yamldata);
  ASSERT_THAT_EXPECTED(ErrOrSections, Succeeded());
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);
  ASSERT_TRUE(DwarfContext.get() != nullptr);
  std::string errors;
  raw_string_ostream OS(errors);
  OutputAggregator OSAgg(&OS);
  CreatorT GC;
  DwarfTransformer DT(*DwarfContext, GC);
  const uint32_t ThreadCount = 1;
  ASSERT_THAT_ERROR(DT.convert(ThreadCount, OSAgg), Succeeded());
  ASSERT_THAT_ERROR(GC.finalize(OSAgg), Succeeded());
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  const auto ByteOrder = llvm::endianness::native;
  FileWriter FW(OutStrm, ByteOrder);
  FW.setStringOffsetSize(GC.getStringOffsetSize());
  ASSERT_THAT_ERROR(GC.encode(FW), Succeeded());
  auto GROrErr = GsymReader::copyBuffer(OutStrm.str());
  ASSERT_THAT_EXPECTED(GROrErr, Succeeded());
  const std::unique_ptr<GsymReader> &GR = *GROrErr;
  // There should only be two functions in our GSYM.
  EXPECT_EQ(GR->getNumAddresses(), 2u);
  // Verify "foo" is present and has a line table
  auto ExpFI = GR->getFunctionInfo(0x1000);
  ASSERT_THAT_EXPECTED(ExpFI, Succeeded());
  ASSERT_EQ(ExpFI->Range, AddressRange(0x1000, 0x1050));
  EXPECT_TRUE(ExpFI->OptLineTable.has_value());
  EXPECT_FALSE(ExpFI->Inline.has_value());
  StringRef FuncName = GR->getString(ExpFI->Name);
  EXPECT_EQ(FuncName, "foo");

  // Verify "foo" is present and has a line table
  auto ExpFI2 = GR->getFunctionInfo(0x2000);
  ASSERT_THAT_EXPECTED(ExpFI2, Succeeded());
  ASSERT_EQ(ExpFI2->Range, AddressRange(0x2000, 0x2050));
  EXPECT_TRUE(ExpFI2->OptLineTable.has_value());
  EXPECT_FALSE(ExpFI2->Inline.has_value());
  StringRef FuncName2 = GR->getString(ExpFI2->Name);
  EXPECT_EQ(FuncName2, "bar");
}

TEST(GSYMTest, TestFinalizeForLineTables) {
  TestFinalizeForLineTables<GsymCreatorV1>();
}
TEST(GSYMTest, TestFinalizeForLineTablesV2) {
  TestFinalizeForLineTables<GsymCreatorV2>();
}

template <typename CreatorT> static void TestRangeWarnings() {
  // This example has a single compile unit that has a DW_TAG_subprogram that
  // has two discontiguous ranges. We will create two FunctionInfo objects for
  // each range in the function that only contains info for each range. We also
  // want to verify that we only emit errors and warnings for ranges that
  // aren't contained in any parent address ranges if this is true. Prior to
  // this fix we would create two FunctionInfo objects and as each one was
  // being created we would end up warning about all of the ranges that weren't
  // in the current FunctionInfo's range even though the DWARF was well formed.
  // Now we don't incorrectly emit errors when there are none.
  //
  // 0x0000000b: DW_TAG_compile_unit
  //               DW_AT_name	("/tmp/main.cpp")
  //               DW_AT_language	(DW_LANG_C)
  //               DW_AT_stmt_list	(0x00000000)
  //
  // 0x00000015:   DW_TAG_subprogram
  //                 DW_AT_name	("foo")
  //                 DW_AT_ranges	(0x00000000
  //                    [0x0000000000001000, 0x0000000000001050)
  //                    [0x0000000000002000, 0x0000000000002050))
  //
  // 0x0000001e:     DW_TAG_inlined_subroutine
  //                   DW_AT_name	("inline1")
  //                   DW_AT_ranges	(0x00000030
  //                      [0x0000000000001010, 0x0000000000001040)
  //                      [0x0000000000002010, 0x0000000000002040))
  //                   DW_AT_call_file	("/tmp/main.cpp")
  //                   DW_AT_call_line	(11)
  //
  // 0x0000002f:       DW_TAG_inlined_subroutine
  //                     DW_AT_name	("inline2")
  //                     DW_AT_ranges	(0x00000060
  //                        [0x0000000000001015, 0x0000000000001020)
  //                        [0x0000000000002015, 0x0000000000002020))
  //                     DW_AT_call_file	("/tmp/inline.h")
  //                     DW_AT_call_line	(21)
  //
  // 0x00000040:       NULL
  //
  // 0x00000041:     NULL
  //
  // 0x00000042:   NULL

  StringRef yamldata = R"(
  debug_str:
    - ''
    - '/tmp/main.cpp'
    - foo
    - inline1
    - inline2
  debug_abbrev:
    - ID:              0
      Table:
        - Code:            0x1
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_language
              Form:            DW_FORM_udata
            - Attribute:       DW_AT_stmt_list
              Form:            DW_FORM_sec_offset
        - Code:            0x2
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_ranges
              Form:            DW_FORM_sec_offset
        - Code:            0x3
          Tag:             DW_TAG_inlined_subroutine
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_ranges
              Form:            DW_FORM_sec_offset
            - Attribute:       DW_AT_call_file
              Form:            DW_FORM_data4
            - Attribute:       DW_AT_call_line
              Form:            DW_FORM_data4
        - Code:            0x4
          Tag:             DW_TAG_inlined_subroutine
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_ranges
              Form:            DW_FORM_sec_offset
            - Attribute:       DW_AT_call_file
              Form:            DW_FORM_data4
            - Attribute:       DW_AT_call_line
              Form:            DW_FORM_data4
  debug_ranges:
    - Offset:          0x0
      AddrSize:        0x8
      Entries:
        - LowOffset:       0x1000
          HighOffset:      0x1050
        - LowOffset:       0x2000
          HighOffset:      0x2050
    - Offset:          0x30
      AddrSize:        0x8
      Entries:
        - LowOffset:       0x1010
          HighOffset:      0x1040
        - LowOffset:       0x2010
          HighOffset:      0x2040
    - Offset:          0x60
      AddrSize:        0x8
      Entries:
        - LowOffset:       0x1015
          HighOffset:      0x1020
        - LowOffset:       0x2015
          HighOffset:      0x2020
  debug_info:
    - Length:          0x3F
      Version:         4
      AbbrevTableID:   0
      AbbrOffset:      0x0
      AddrSize:        8
      Entries:
        - AbbrCode:        0x1
          Values:
            - Value:           0x1
            - Value:           0x2
            - Value:           0x0
        - AbbrCode:        0x2
          Values:
            - Value:           0xF
            - Value:           0x0
        - AbbrCode:        0x3
          Values:
            - Value:           0x13
            - Value:           0x30
            - Value:           0x1
            - Value:           0xB
        - AbbrCode:        0x4
          Values:
            - Value:           0x1B
            - Value:           0x60
            - Value:           0x2
            - Value:           0x15
        - AbbrCode:        0x0
        - AbbrCode:        0x0
        - AbbrCode:        0x0
  debug_line:
    - Length:          120
      Version:         2
      PrologueLength:  48
      MinInstLength:   1
      DefaultIsStmt:   1
      LineBase:        251
      LineRange:       14
      OpcodeBase:      13
      StandardOpcodeLengths: [ 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1 ]
      IncludeDirs:
        - '/tmp'
      Files:
        - Name:            main.cpp
          DirIdx:          1
          ModTime:         0
          Length:          0
        - Name:            inline.h
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
          Data:            0
        - Opcode:          DW_LNS_copy
          Data:            0
        - Opcode:          DW_LNS_advance_pc
          Data:            16
        - Opcode:          DW_LNS_set_file
          Data:            2
        - Opcode:          DW_LNS_advance_line
          SData:           10
          Data:            0
        - Opcode:          DW_LNS_copy
          Data:            0
        - Opcode:          DW_LNS_advance_pc
          Data:            16
        - Opcode:          DW_LNS_advance_line
          SData:           1
          Data:            0
        - Opcode:          DW_LNS_copy
          Data:            0
        - Opcode:          DW_LNS_advance_pc
          Data:            48
        - Opcode:          DW_LNS_set_file
          Data:            1
        - Opcode:          DW_LNS_advance_line
          SData:           -10
          Data:            0
        - Opcode:          DW_LNS_extended_op
          ExtLen:          1
          SubOpcode:       DW_LNE_end_sequence
          Data:            0
        - Opcode:          DW_LNS_extended_op
          ExtLen:          9
          SubOpcode:       DW_LNE_set_address
          Data:            8192
        - Opcode:          DW_LNS_advance_line
          SData:           19
          Data:            0
        - Opcode:          DW_LNS_copy
          Data:            0
        - Opcode:          DW_LNS_advance_pc
          Data:            16
        - Opcode:          DW_LNS_set_file
          Data:            2
        - Opcode:          DW_LNS_copy
          Data:            0
        - Opcode:          DW_LNS_advance_pc
          Data:            16
        - Opcode:          DW_LNS_advance_line
          SData:           1
          Data:            0
        - Opcode:          DW_LNS_copy
          Data:            0
        - Opcode:          DW_LNS_advance_pc
          Data:            48
        - Opcode:          DW_LNS_set_file
          Data:            1
        - Opcode:          DW_LNS_extended_op
          ExtLen:          1
          SubOpcode:       DW_LNE_end_sequence
          Data:            0
  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(yamldata);
  ASSERT_THAT_EXPECTED(ErrOrSections, Succeeded());
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);
  ASSERT_TRUE(DwarfContext.get() != nullptr);
  std::string errors;
  raw_string_ostream OS(errors);
  OutputAggregator OSAgg(&OS);
  CreatorT GC;
  DwarfTransformer DT(*DwarfContext, GC);
  const uint32_t ThreadCount = 1;
  ASSERT_THAT_ERROR(DT.convert(ThreadCount, OSAgg), Succeeded());
  ASSERT_THAT_ERROR(GC.finalize(OSAgg), Succeeded());
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  const auto ByteOrder = llvm::endianness::native;
  FileWriter FW(OutStrm, ByteOrder);
  FW.setStringOffsetSize(GC.getStringOffsetSize());
  ASSERT_THAT_ERROR(GC.encode(FW), Succeeded());
  auto GROrErr = GsymReader::copyBuffer(OutStrm.str());
  ASSERT_THAT_EXPECTED(GROrErr, Succeeded());
  const std::unique_ptr<GsymReader> &GR = *GROrErr;
  // There should be two functions in our GSYM.
  EXPECT_EQ(GR->getNumAddresses(), 2u);
  // Verify "foo" is present and has a line table
  auto ExpFI = GR->getFunctionInfo(0x1000);
  ASSERT_THAT_EXPECTED(ExpFI, Succeeded());
  ASSERT_EQ(ExpFI->Range, AddressRange(0x1000, 0x1050));
  EXPECT_TRUE(ExpFI->OptLineTable.has_value());
  EXPECT_TRUE(ExpFI->Inline.has_value());
  StringRef FuncName = GR->getString(ExpFI->Name);
  EXPECT_EQ(FuncName, "foo");

  // Verify "foo" is present and has a line table
  auto ExpFI2 = GR->getFunctionInfo(0x2000);
  ASSERT_THAT_EXPECTED(ExpFI2, Succeeded());
  ASSERT_EQ(ExpFI2->Range, AddressRange(0x2000, 0x2050));
  EXPECT_TRUE(ExpFI2->OptLineTable.has_value());
  EXPECT_TRUE(ExpFI2->Inline.has_value());
  StringRef FuncName2 = GR->getString(ExpFI2->Name);
  EXPECT_EQ(FuncName2, "foo");

  // Make sure we don't see spurious errors in the output:
  EXPECT_TRUE(errors.find("error:") == std::string::npos);
}

TEST(GSYMTest, TestRangeWarnings) { TestRangeWarnings<GsymCreatorV1>(); }
TEST(GSYMTest, TestRangeWarningsV2) { TestRangeWarnings<GsymCreatorV2>(); }

template <typename CreatorT> static void TestEmptyRangeWarnings() {
  // This example has a single compile unit that has a DW_TAG_subprogram that
  // has a function that contains an inlined function that has an empty range.
  // We want to make sure that if we run into only empty inline functions
  // inside of a real function, that we don't end up with inline information
  // in the GSYM and we don't warn about the inline function's range not being
  // contined in the parent ranges since it is ok for inline functions to be
  // elided.
  //
  // 0x0000000b: DW_TAG_compile_unit
  //               DW_AT_name	("/tmp/main.cpp")
  //               DW_AT_language	(DW_LANG_C)
  //               DW_AT_stmt_list	(0x00000000)
  //
  // 0x00000015:   DW_TAG_subprogram
  //                 DW_AT_name	("foo")
  //                 DW_AT_low_pc	(0x0000000000001000)
  //                 DW_AT_high_pc	(0x0000000000001050)
  //
  // 0x0000002a:     DW_TAG_inlined_subroutine
  //                   DW_AT_name	("inline1")
  //                   DW_AT_low_pc	(0x0000000000001010)
  //                   DW_AT_high_pc	(0x0000000000001010)
  //                   DW_AT_call_file	("/tmp/main.cpp")
  //                   DW_AT_call_line	(11)
  //
  // 0x00000047:     NULL
  //
  // 0x00000048:   NULL

  StringRef yamldata = R"(
  debug_str:
    - ''
    - '/tmp/main.cpp'
    - foo
    - inline1
  debug_abbrev:
    - ID:              0
      Table:
        - Code:            0x1
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_language
              Form:            DW_FORM_udata
            - Attribute:       DW_AT_stmt_list
              Form:            DW_FORM_sec_offset
        - Code:            0x2
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_low_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_high_pc
              Form:            DW_FORM_addr
        - Code:            0x3
          Tag:             DW_TAG_inlined_subroutine
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_low_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_high_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_call_file
              Form:            DW_FORM_data4
            - Attribute:       DW_AT_call_line
              Form:            DW_FORM_data4
  debug_info:
    - Length:          0x45
      Version:         4
      AbbrevTableID:   0
      AbbrOffset:      0x0
      AddrSize:        8
      Entries:
        - AbbrCode:        0x1
          Values:
            - Value:           0x1
            - Value:           0x2
            - Value:           0x0
        - AbbrCode:        0x2
          Values:
            - Value:           0xF
            - Value:           0x1000
            - Value:           0x1050
        - AbbrCode:        0x3
          Values:
            - Value:           0x13
            - Value:           0x1010
            - Value:           0x1010
            - Value:           0x1
            - Value:           0xB
        - AbbrCode:        0x0
        - AbbrCode:        0x0
  debug_line:
    - Length:          89
      Version:         2
      PrologueLength:  48
      MinInstLength:   1
      DefaultIsStmt:   1
      LineBase:        251
      LineRange:       14
      OpcodeBase:      13
      StandardOpcodeLengths: [ 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1 ]
      IncludeDirs:
        - '/tmp'
      Files:
        - Name:            main.cpp
          DirIdx:          1
          ModTime:         0
          Length:          0
        - Name:            inline.h
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
          Data:            0
        - Opcode:          DW_LNS_copy
          Data:            0
        - Opcode:          DW_LNS_advance_pc
          Data:            16
        - Opcode:          DW_LNS_set_file
          Data:            2
        - Opcode:          DW_LNS_advance_line
          SData:           10
          Data:            0
        - Opcode:          DW_LNS_copy
          Data:            0
        - Opcode:          DW_LNS_advance_pc
          Data:            16
        - Opcode:          DW_LNS_advance_line
          SData:           1
          Data:            0
        - Opcode:          DW_LNS_copy
          Data:            0
        - Opcode:          DW_LNS_advance_pc
          Data:            48
        - Opcode:          DW_LNS_set_file
          Data:            1
        - Opcode:          DW_LNS_advance_line
          SData:           -10
          Data:            0
        - Opcode:          DW_LNS_extended_op
          ExtLen:          1
          SubOpcode:       DW_LNE_end_sequence
          Data:            0
  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(yamldata);
  ASSERT_THAT_EXPECTED(ErrOrSections, Succeeded());
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);
  ASSERT_TRUE(DwarfContext.get() != nullptr);
  std::string errors;
  raw_string_ostream OS(errors);
  OutputAggregator OSAgg(&OS);
  CreatorT GC;
  DwarfTransformer DT(*DwarfContext, GC);
  const uint32_t ThreadCount = 1;
  ASSERT_THAT_ERROR(DT.convert(ThreadCount, OSAgg), Succeeded());
  ASSERT_THAT_ERROR(GC.finalize(OSAgg), Succeeded());
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  const auto ByteOrder = llvm::endianness::native;
  FileWriter FW(OutStrm, ByteOrder);
  FW.setStringOffsetSize(GC.getStringOffsetSize());
  ASSERT_THAT_ERROR(GC.encode(FW), Succeeded());
  auto GROrErr = GsymReader::copyBuffer(OutStrm.str());
  ASSERT_THAT_EXPECTED(GROrErr, Succeeded());
  const std::unique_ptr<GsymReader> &GR = *GROrErr;
  // There should be one function in our GSYM.
  EXPECT_EQ(GR->getNumAddresses(), 1u);
  // Verify "foo" is present and has a line table and no inline info.
  auto ExpFI = GR->getFunctionInfo(0x1000);
  ASSERT_THAT_EXPECTED(ExpFI, Succeeded());
  ASSERT_EQ(ExpFI->Range, AddressRange(0x1000, 0x1050));
  EXPECT_TRUE(ExpFI->OptLineTable.has_value());
  EXPECT_FALSE(ExpFI->Inline.has_value());
  StringRef FuncName = GR->getString(ExpFI->Name);
  EXPECT_EQ(FuncName, "foo");

  // Make sure we don't see spurious errors in the output:
  EXPECT_TRUE(errors.find("error:") == std::string::npos);
}

TEST(GSYMTest, TestEmptyRangeWarnings) {
  TestEmptyRangeWarnings<GsymCreatorV1>();
}
TEST(GSYMTest, TestEmptyRangeWarningsV2) {
  TestEmptyRangeWarnings<GsymCreatorV2>();
}

template <typename CreatorT> static void TestEmptyLinkageName() {
  // This example has a single compile unit that has a DW_TAG_subprogram that
  // has a function that has an empty linkage name and a valid normal name.
  // Previously this would cause an encoding error:
  //
  // DWARF conversion failed: attempted to encode invalid FunctionInfo object
  //
  // This was because we would get a valid but empty linkage name and we would
  // try to use this in the GSYM FunctionInfo and that would cause the error
  // as the name was empty.
  //
  // 0x0000000b: DW_TAG_compile_unit
  //               DW_AT_name        ("/tmp/main.cpp")
  //               DW_AT_language    (DW_LANG_C)
  //               DW_AT_stmt_list   (0x00000000)
  //
  // 0x00000015:   DW_TAG_subprogram
  //                 DW_AT_name      ("foo")
  //                 DW_AT_linkage_name      ("")
  //                 DW_AT_low_pc    (0x0000000000001000)
  //                 DW_AT_high_pc   (0x0000000000001050)
  //
  // 0x0000002e:   NULL


  StringRef yamldata = R"(
  debug_str:
    - ''
    - '/tmp/main.cpp'
    - foo
    - ''
  debug_abbrev:
    - ID:              0
      Table:
        - Code:            0x1
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_language
              Form:            DW_FORM_udata
            - Attribute:       DW_AT_stmt_list
              Form:            DW_FORM_sec_offset
        - Code:            0x2
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_linkage_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_low_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_high_pc
              Form:            DW_FORM_addr
  debug_info:
    - Length:          0x2B
      Version:         4
      AbbrevTableID:   0
      AbbrOffset:      0x0
      AddrSize:        8
      Entries:
        - AbbrCode:        0x1
          Values:
            - Value:           0x1
            - Value:           0x2
            - Value:           0x0
        - AbbrCode:        0x2
          Values:
            - Value:           0xF
            - Value:           0x13
            - Value:           0x1000
            - Value:           0x1050
        - AbbrCode:        0x0
  debug_line:
    - Length:          68
      Version:         2
      PrologueLength:  36
      MinInstLength:   1
      DefaultIsStmt:   1
      LineBase:        251
      LineRange:       14
      OpcodeBase:      13
      StandardOpcodeLengths: [ 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1 ]
      IncludeDirs:
        - '/tmp'
      Files:
        - Name:            main.cpp
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
          Data:            0
        - Opcode:          DW_LNS_copy
          Data:            0
        - Opcode:          DW_LNS_advance_pc
          Data:            256
        - Opcode:          DW_LNS_advance_line
          SData:           1
          Data:            0
        - Opcode:          DW_LNS_copy
          Data:            0
        - Opcode:          DW_LNS_advance_pc
          Data:            256
        - Opcode:          DW_LNS_extended_op
          ExtLen:          1
          SubOpcode:       DW_LNE_end_sequence
          Data:            0
  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(yamldata);
  ASSERT_THAT_EXPECTED(ErrOrSections, Succeeded());
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);
  ASSERT_TRUE(DwarfContext.get() != nullptr);
  std::string errors;
  raw_string_ostream OS(errors);
  OutputAggregator OSAgg(&OS);
  CreatorT GC;
  DwarfTransformer DT(*DwarfContext, GC);
  const uint32_t ThreadCount = 1;
  ASSERT_THAT_ERROR(DT.convert(ThreadCount, OSAgg), Succeeded());
  ASSERT_THAT_ERROR(GC.finalize(OSAgg), Succeeded());
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  const auto ByteOrder = llvm::endianness::native;
  FileWriter FW(OutStrm, ByteOrder);
  FW.setStringOffsetSize(GC.getStringOffsetSize());
  ASSERT_THAT_ERROR(GC.encode(FW), Succeeded());
  auto GROrErr = GsymReader::copyBuffer(OutStrm.str());
  ASSERT_THAT_EXPECTED(GROrErr, Succeeded());
  const std::unique_ptr<GsymReader> &GR = *GROrErr;
  // There should be one function in our GSYM.
  EXPECT_EQ(GR->getNumAddresses(), 1u);
  // Verify "foo" is present and has a line table and no inline info.
  auto ExpFI = GR->getFunctionInfo(0x1000);
  ASSERT_THAT_EXPECTED(ExpFI, Succeeded());
  ASSERT_EQ(ExpFI->Range, AddressRange(0x1000, 0x1050));
  EXPECT_TRUE(ExpFI->OptLineTable.has_value());
  EXPECT_FALSE(ExpFI->Inline.has_value());
  StringRef FuncName = GR->getString(ExpFI->Name);
  EXPECT_EQ(FuncName, "foo");

  // Make sure we don't see spurious errors in the output:
  EXPECT_TRUE(errors.find("error:") == std::string::npos);
}

TEST(GSYMTest, TestEmptyLinkageName) { TestEmptyLinkageName<GsymCreatorV1>(); }
TEST(GSYMTest, TestEmptyLinkageNameV2) {
  TestEmptyLinkageName<GsymCreatorV2>();
}

template <typename CreatorT> static void TestLineTablesWithEmptyRanges() {
  // Test that lookups find the right line table entry when there are multiple
  // line entries with the same address. When we have multiple line table
  // entries with the same address, we need to pick the last one in the line
  // table. We do this because a line entry's start address in the defined by
  // the line table entry's address and the size is determined by the
  // subtracting the next line table's address. If the current line table
  // entry's address is the same as the next one, then there is no code
  // assiciated with the current line table entry and it should be ignored.
  //
  // 0x0000000b: DW_TAG_compile_unit
  //               DW_AT_name        ("/tmp/main.cpp")
  //               DW_AT_language    (DW_LANG_C)
  //               DW_AT_stmt_list   (0x00000000)
  //
  // 0x00000015:   DW_TAG_subprogram
  //                 DW_AT_name      ("foo")
  //                 DW_AT_low_pc    (0x0000000000001000)
  //                 DW_AT_high_pc   (0x0000000000001050)
  //
  // 0x0000002a:   NULL
  //
  // The line table has a duplicate entry at 0x1010:
  //
  // Address    Line   Column File   ISA Discriminator Flags
  // ---------- ------ ------ ------ --- ------------- -------------
  // 0x00001000     10      0      1   0             0  is_stmt
  // 0x00001010     11      0      1   0             0  is_stmt
  // 0x00001010     12      0      1   0             0  is_stmt
  // 0x00001050     13      0      1   0             0  is_stmt end_sequence

  StringRef yamldata = R"(
  debug_str:
    - ''
    - '/tmp/main.cpp'
    - foo
  debug_abbrev:
    - ID:              0
      Table:
        - Code:            0x1
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_language
              Form:            DW_FORM_udata
            - Attribute:       DW_AT_stmt_list
              Form:            DW_FORM_sec_offset
        - Code:            0x2
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
    - Length:          0x27
      Version:         4
      AbbrevTableID:   0
      AbbrOffset:      0x0
      AddrSize:        8
      Entries:
        - AbbrCode:        0x1
          Values:
            - Value:           0x1
            - Value:           0x2
            - Value:           0x0
        - AbbrCode:        0x2
          Values:
            - Value:           0xF
            - Value:           0x1000
            - Value:           0x1050
        - AbbrCode:        0x0
  debug_line:
    - Length:          71
      Version:         2
      PrologueLength:  36
      MinInstLength:   1
      DefaultIsStmt:   1
      LineBase:        251
      LineRange:       14
      OpcodeBase:      13
      StandardOpcodeLengths: [ 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1 ]
      IncludeDirs:
        - '/tmp'
      Files:
        - Name:            main.cpp
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
          Data:            0
        - Opcode:          DW_LNS_copy
          Data:            0
        - Opcode:          DW_LNS_advance_pc
          Data:            16
        - Opcode:          DW_LNS_advance_line
          SData:           1
          Data:            0
        - Opcode:          DW_LNS_copy
          Data:            0
        - Opcode:          DW_LNS_advance_line
          SData:           1
          Data:            0
        - Opcode:          DW_LNS_copy
          Data:            0
        - Opcode:          DW_LNS_advance_pc
          Data:            64
        - Opcode:          DW_LNS_advance_line
          SData:           1
          Data:            0
        - Opcode:          DW_LNS_extended_op
          ExtLen:          1
          SubOpcode:       DW_LNE_end_sequence
          Data:            0
  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(yamldata);
  ASSERT_THAT_EXPECTED(ErrOrSections, Succeeded());
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);
  ASSERT_TRUE(DwarfContext.get() != nullptr);
  std::string errors;
  raw_string_ostream OS(errors);
  OutputAggregator OSAgg(&OS);
  CreatorT GC;
  DwarfTransformer DT(*DwarfContext, GC);
  const uint32_t ThreadCount = 1;
  ASSERT_THAT_ERROR(DT.convert(ThreadCount, OSAgg), Succeeded());
  ASSERT_THAT_ERROR(GC.finalize(OSAgg), Succeeded());
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  const auto ByteOrder = llvm::endianness::native;
  FileWriter FW(OutStrm, ByteOrder);
  FW.setStringOffsetSize(GC.getStringOffsetSize());
  ASSERT_THAT_ERROR(GC.encode(FW), Succeeded());
  auto GROrErr = GsymReader::copyBuffer(OutStrm.str());
  ASSERT_THAT_EXPECTED(GROrErr, Succeeded());
  const std::unique_ptr<GsymReader> &GR = *GROrErr;
  // There should be one function in our GSYM.
  EXPECT_EQ(GR->getNumAddresses(), 1u);
  // Verify "foo" is present and has a line table and no inline info.
  auto ExpFI = GR->getFunctionInfo(0x1000);
  ASSERT_THAT_EXPECTED(ExpFI, Succeeded());
  ASSERT_EQ(ExpFI->Range, AddressRange(0x1000, 0x1050));
  EXPECT_TRUE(ExpFI->OptLineTable.has_value());
  EXPECT_FALSE(ExpFI->Inline.has_value());
  StringRef FuncName = GR->getString(ExpFI->Name);
  EXPECT_EQ(FuncName, "foo");

  // Make sure we don't see spurious errors in the output:
  EXPECT_TRUE(errors.find("error:") == std::string::npos);

  // Make sure that when we lookup address 0x1010, that we get the entry that
  // matches line 12, the second line entry that also has the address of
  // 0x1010.
  auto LR = GR->lookup(0x1010);
  ASSERT_THAT_EXPECTED(LR, Succeeded());
  SourceLocation src_loc = {"foo", "/tmp", "main.cpp", 12, 16};
  EXPECT_THAT(LR->Locations, testing::ElementsAre(src_loc));
}

TEST(GSYMTest, TestLineTablesWithEmptyRanges) {
  TestLineTablesWithEmptyRanges<GsymCreatorV1>();
}
TEST(GSYMTest, TestLineTablesWithEmptyRangesV2) {
  TestLineTablesWithEmptyRanges<GsymCreatorV2>();
}

template <typename CreatorT> static void TestHandlingOfInvalidFileIndexes() {
  // Test that llvm-gsymutil can handle invalid file indexes in the following
  // cases:
  //  - In line entries in the line table
  //  - When parsing inline entries that have a DW_AT_call_file
  //  - When parsing function dies with no line table entries and it tries to
  //    use the DW_AT_decl_file
  //
  //
  // 0x0000000b: DW_TAG_compile_unit
  //               DW_AT_name        ("/tmp/main.cpp")
  //               DW_AT_language    (DW_LANG_C)
  //               DW_AT_stmt_list   (0x00000000)
  //
  // 0x00000015:   DW_TAG_subprogram
  //                 DW_AT_name      ("foo")
  //                 DW_AT_low_pc    (0x0000000000001000)
  //                 DW_AT_high_pc   (0x0000000000001050)
  //
  // 0x0000002a:     DW_TAG_inlined_subroutine
  //                   DW_AT_name    ("inline_with_invalid_call_file")
  //                   DW_AT_low_pc  (0x0000000000001010)
  //                   DW_AT_high_pc (0x0000000000001020)
  //                   DW_AT_call_file       (0x0000000a)
  //                   DW_AT_call_line       (11)
  //
  // 0x00000047:       DW_TAG_inlined_subroutine
  //                     DW_AT_name
  //                     ("inline_inside_parent_with_invalid_call_file")
  //                     DW_AT_low_pc        (0x0000000000001010)
  //                     DW_AT_high_pc       (0x0000000000001015)
  //                     DW_AT_call_file     ("/tmp/main.cpp")
  //                     DW_AT_call_line     (12)
  //
  // 0x00000064:       NULL
  //
  // 0x00000065:     DW_TAG_inlined_subroutine
  //                   DW_AT_name    ("inline_with_valid_call_file")
  //                   DW_AT_low_pc  (0x0000000000001020)
  //                   DW_AT_high_pc (0x0000000000001030)
  //                   DW_AT_call_file       ("/tmp/main.cpp")
  //                   DW_AT_call_line       (13)
  //
  // 0x00000082:       DW_TAG_inlined_subroutine
  //                     DW_AT_name
  //                     ("inline_inside_parent_with_valid_call_file")
  //                     DW_AT_low_pc        (0x0000000000001020)
  //                     DW_AT_high_pc       (0x0000000000001025)
  //                     DW_AT_call_file     ("/tmp/main.cpp")
  //                     DW_AT_call_line     (14)
  //
  // 0x0000009f:       NULL
  //
  // 0x000000a0:     NULL
  //
  // 0x000000a1:   DW_TAG_subprogram
  //                 DW_AT_name      ("func_with_valid_decl_file")
  //                 DW_AT_decl_file ("/tmp/main.cpp")
  //                 DW_AT_decl_line (20)
  //                 DW_AT_low_pc    (0x0000000000002000)
  //                 DW_AT_high_pc   (0x0000000000002050)
  //
  // 0x000000b8:   DW_TAG_subprogram
  //                 DW_AT_name      ("func_with_invalid_decl_file")
  //                 DW_AT_decl_file (0x0a)
  //                 DW_AT_decl_line (20)
  //                 DW_AT_low_pc    (0x0000000000003000)
  //                 DW_AT_high_pc   (0x0000000000003050)
  //
  // 0x000000cf:   NULL
  //
  // The table looks has an entry at address 0x0000000000001010 that has an
  // invalid file index that needs to be removed.
  //
  // Address            Line   Column File   ISA Discriminator Flags
  // ---------- ------ ------ ------ --- ------------- -------------
  // 0x00001000     10      0      1   0             0  is_stmt
  // 0x00001010     11      0     10   0             0  is_stmt
  // 0x00001020     11      0      1   0             0  is_stmt
  // 0x00001030     12      0      1   0             0  is_stmt
  // 0x00001050     12      0      1   0             0  is_stmt end_sequence

  StringRef yamldata = R"(
  debug_str:
    - ''
    - '/tmp/main.cpp'
    - foo
    - inline_with_invalid_call_file
    - inline_inside_parent_with_invalid_call_file
    - inline_with_valid_call_file
    - inline_inside_parent_with_valid_call_file
    - func_with_valid_decl_file
    - func_with_invalid_decl_file
  debug_abbrev:
    - ID:              0
      Table:
        - Code:            0x1
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_language
              Form:            DW_FORM_udata
            - Attribute:       DW_AT_stmt_list
              Form:            DW_FORM_sec_offset
        - Code:            0x2
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_low_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_high_pc
              Form:            DW_FORM_addr
        - Code:            0x3
          Tag:             DW_TAG_inlined_subroutine
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_low_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_high_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_call_file
              Form:            DW_FORM_data4
            - Attribute:       DW_AT_call_line
              Form:            DW_FORM_data4
        - Code:            0x4
          Tag:             DW_TAG_inlined_subroutine
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_low_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_high_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_call_file
              Form:            DW_FORM_data4
            - Attribute:       DW_AT_call_line
              Form:            DW_FORM_data4
        - Code:            0x5
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_decl_file
              Form:            DW_FORM_data1
            - Attribute:       DW_AT_decl_line
              Form:            DW_FORM_data1
            - Attribute:       DW_AT_low_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_high_pc
              Form:            DW_FORM_addr
  debug_info:
    - Length:          0xCC
      Version:         4
      AbbrevTableID:   0
      AbbrOffset:      0x0
      AddrSize:        8
      Entries:
        - AbbrCode:        0x1
          Values:
            - Value:           0x1
            - Value:           0x2
            - Value:           0x0
        - AbbrCode:        0x2
          Values:
            - Value:           0xF
            - Value:           0x1000
            - Value:           0x1050
        - AbbrCode:        0x3
          Values:
            - Value:           0x13
            - Value:           0x1010
            - Value:           0x1020
            - Value:           0xA
            - Value:           0xB
        - AbbrCode:        0x4
          Values:
            - Value:           0x31
            - Value:           0x1010
            - Value:           0x1015
            - Value:           0x1
            - Value:           0xC
        - AbbrCode:        0x0
        - AbbrCode:        0x3
          Values:
            - Value:           0x5D
            - Value:           0x1020
            - Value:           0x1030
            - Value:           0x1
            - Value:           0xD
        - AbbrCode:        0x4
          Values:
            - Value:           0x79
            - Value:           0x1020
            - Value:           0x1025
            - Value:           0x1
            - Value:           0xE
        - AbbrCode:        0x0
        - AbbrCode:        0x0
        - AbbrCode:        0x5
          Values:
            - Value:           0xA3
            - Value:           0x1
            - Value:           0x14
            - Value:           0x2000
            - Value:           0x2050
        - AbbrCode:        0x5
          Values:
            - Value:           0xBD
            - Value:           0xA
            - Value:           0x14
            - Value:           0x3000
            - Value:           0x3050
        - AbbrCode:        0x0
  debug_line:
    - Length:          78
      Version:         2
      PrologueLength:  36
      MinInstLength:   1
      DefaultIsStmt:   1
      LineBase:        251
      LineRange:       14
      OpcodeBase:      13
      StandardOpcodeLengths: [ 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1 ]
      IncludeDirs:
        - '/tmp'
      Files:
        - Name:            main.cpp
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
          Data:            0
        - Opcode:          DW_LNS_copy
          Data:            0
        - Opcode:          DW_LNS_advance_pc
          Data:            16
        - Opcode:          DW_LNS_set_file
          Data:            10
        - Opcode:          DW_LNS_advance_line
          SData:           1
          Data:            0
        - Opcode:          DW_LNS_copy
          Data:            0
        - Opcode:          DW_LNS_advance_pc
          Data:            16
        - Opcode:          DW_LNS_set_file
          Data:            1
        - Opcode:          DW_LNS_copy
          Data:            0
        - Opcode:          DW_LNS_advance_pc
          Data:            16
        - Opcode:          DW_LNS_advance_line
          SData:           1
          Data:            0
        - Opcode:          DW_LNS_copy
          Data:            0
        - Opcode:          DW_LNS_advance_pc
          Data:            32
        - Opcode:          DW_LNS_extended_op
          ExtLen:          1
          SubOpcode:       DW_LNE_end_sequence
          Data:            0
  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(yamldata);
  ASSERT_THAT_EXPECTED(ErrOrSections, Succeeded());
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);
  ASSERT_TRUE(DwarfContext.get() != nullptr);
  std::string errors;
  raw_string_ostream OS(errors);
  OutputAggregator OSAgg(&OS);
  CreatorT GC;
  DwarfTransformer DT(*DwarfContext, GC);
  const uint32_t ThreadCount = 1;
  ASSERT_THAT_ERROR(DT.convert(ThreadCount, OSAgg), Succeeded());
  ASSERT_THAT_ERROR(GC.finalize(OSAgg), Succeeded());
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  const auto ByteOrder = llvm::endianness::native;
  FileWriter FW(OutStrm, ByteOrder);
  FW.setStringOffsetSize(GC.getStringOffsetSize());
  ASSERT_THAT_ERROR(GC.encode(FW), Succeeded());
  auto GROrErr = GsymReader::copyBuffer(OutStrm.str());
  ASSERT_THAT_EXPECTED(GROrErr, Succeeded());
  const std::unique_ptr<GsymReader> &GR = *GROrErr;
  // There should be one function in our GSYM.
  EXPECT_EQ(GR->getNumAddresses(), 3u);
  // Verify "foo" is present and has a line table and no inline info.
  auto ExpFI = GR->getFunctionInfo(0x1000);
  ASSERT_THAT_EXPECTED(ExpFI, Succeeded());
  ASSERT_EQ(ExpFI->Range, AddressRange(0x1000, 0x1050));
  StringRef FuncName = GR->getString(ExpFI->Name);
  EXPECT_EQ(FuncName, "foo");

  EXPECT_TRUE(ExpFI->OptLineTable.has_value());
  // Make sure we only have 3 entries to show we removed the line entry with
  // the invalid file index whose address is 0x0000000000001010.
  ASSERT_EQ(ExpFI->OptLineTable->size(), 3u);
  EXPECT_TRUE(ExpFI->Inline.has_value());

  // Make sure that we only have one inline function, not two. We remove one of
  // the inline functions because it has an invalid DW_AT_call_file attribute.
  ASSERT_EQ(ExpFI->Inline->Children.size(), 1u);
  StringRef InlineName = GR->getString(ExpFI->Inline->Children[0].Name);
  EXPECT_EQ(InlineName, "inline_with_valid_call_file");

  ExpFI = GR->getFunctionInfo(0x0000000000002000);
  ASSERT_THAT_EXPECTED(ExpFI, Succeeded());
  ASSERT_EQ(ExpFI->Range, AddressRange(0x2000, 0x2050));
  FuncName = GR->getString(ExpFI->Name);
  EXPECT_EQ(FuncName, "func_with_valid_decl_file");
  EXPECT_FALSE(ExpFI->Inline.has_value());
  // Make sure we only have 1 entry in the line table which indicates we were
  // able to parse the DW_AT_decl_file/DW_AT_decl_line correctly.
  EXPECT_TRUE(ExpFI->OptLineTable.has_value());
  ASSERT_EQ(ExpFI->OptLineTable->size(), 1u);

  ExpFI = GR->getFunctionInfo(0x0000000000003000);
  ASSERT_THAT_EXPECTED(ExpFI, Succeeded());
  ASSERT_EQ(ExpFI->Range, AddressRange(0x3000, 0x3050));
  FuncName = GR->getString(ExpFI->Name);
  EXPECT_EQ(FuncName, "func_with_invalid_decl_file");
  EXPECT_FALSE(ExpFI->Inline.has_value());
  // Make sure we only no line table because there are no line entries in the
  // line table and the DW_AT_decl_file attribute was invalid so we were not
  // able to parse the DW_AT_decl_file/DW_AT_decl_line correctly.
  EXPECT_FALSE(ExpFI->OptLineTable.has_value());

  // Make sure we don't see spurious errors in the output:
  std::vector<std::string> ExpectedLogErrors = {
      "error: function DIE at 0x00000015 has a line entry with invalid DWARF "
      "file index, this entry will be removed:",
      "error: inlined function DIE at 0x0000002a has an invalid file index 10 "
      "in its DW_AT_call_file attribute, this inline entry and all children "
      "will be removed.",
      "error: function DIE at 0x000000b8 has an invalid file index 10 in its "
      "DW_AT_decl_file attribute, unable to create a single line entry from "
      "the DW_AT_decl_file/DW_AT_decl_line attributes."};
  // Make sure all expected errors are in the error stream for the two invalid
  // inlined functions that we removed due to invalid range scoping.
  for (const auto &Error : ExpectedLogErrors)
    EXPECT_TRUE(errors.find(Error) != std::string::npos);
}

TEST(GSYMTest, TestHandlingOfInvalidFileIndexes) {
  TestHandlingOfInvalidFileIndexes<GsymCreatorV1>();
}
TEST(GSYMTest, TestHandlingOfInvalidFileIndexesV2) {
  TestHandlingOfInvalidFileIndexes<GsymCreatorV2>();
}

template <typename CreatorT>
static void TestLookupsOfOverlappingAndUnequalRanges() {
  // Test that llvm-gsymutil lookup the correct funtion info when address
  // ranges overlap. When functions overlap we always want to pick the first
  // function info when symbolicating if there are multiple entries with the
  // same address. Previous to this fix we would just binary search the address
  // table and pick the first function info that matched the address. After
  // this fix we now always select the first matching entry whose address range
  // contains the lookup address to ensure we have the most debug info. We have
  // seen case where the debug info would contain a small range and a symbol
  // would have the same start address but the range was larger and sometimes,
  // depending on how the binary search of the address table happened, we would
  // pick these latter entries. We want the first entries because they always
  // have the most debug info.
  //
  // To repro this case, we just make some simple DWARF that has two
  // overlapping ranges and ensure that any lookups between 0x1000 and 0x104f
  // match "foo", and any ranges between 0x1050 and 0x1fff match "bar".
  //
  // 0x0000000b: DW_TAG_compile_unit
  //               DW_AT_name	("/tmp/main.cpp")
  //               DW_AT_language	(DW_LANG_C)
  //               DW_AT_stmt_list	(0x00000000)
  //
  // 0x00000015:   DW_TAG_subprogram
  //                 DW_AT_name	("foo")
  //                 DW_AT_low_pc	(0x0000000000001000)
  //                 DW_AT_high_pc	(0x0000000000001050)
  //
  // 0x0000002a:   DW_TAG_subprogram
  //                 DW_AT_name	("bar")
  //                 DW_AT_low_pc	(0x0000000000001000)
  //                 DW_AT_high_pc	(0x0000000000001100)
  //
  // 0x0000003f:   NULL

  StringRef yamldata = R"(
  debug_str:
    - ''
    - '/tmp/main.cpp'
    - foo
    - bar
  debug_abbrev:
    - ID:              0
      Table:
        - Code:            0x1
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_language
              Form:            DW_FORM_udata
            - Attribute:       DW_AT_stmt_list
              Form:            DW_FORM_sec_offset
        - Code:            0x2
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
    - Length:          0x3C
      Version:         4
      AbbrevTableID:   0
      AbbrOffset:      0x0
      AddrSize:        8
      Entries:
        - AbbrCode:        0x1
          Values:
            - Value:           0x1
            - Value:           0x2
            - Value:           0x0
        - AbbrCode:        0x2
          Values:
            - Value:           0xF
            - Value:           0x1000
            - Value:           0x1050
        - AbbrCode:        0x2
          Values:
            - Value:           0x13
            - Value:           0x1000
            - Value:           0x1100
        - AbbrCode:        0x0
  debug_line:
    - Length:          71
      Version:         2
      PrologueLength:  36
      MinInstLength:   1
      DefaultIsStmt:   1
      LineBase:        251
      LineRange:       14
      OpcodeBase:      13
      StandardOpcodeLengths: [ 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1 ]
      IncludeDirs:
        - '/tmp'
      Files:
        - Name:            main.cpp
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
          Data:            0
        - Opcode:          DW_LNS_copy
          Data:            0
        - Opcode:          DW_LNS_advance_pc
          Data:            16
        - Opcode:          DW_LNS_advance_line
          SData:           1
          Data:            0
        - Opcode:          DW_LNS_copy
          Data:            0
        - Opcode:          DW_LNS_advance_line
          SData:           1
          Data:            0
        - Opcode:          DW_LNS_copy
          Data:            0
        - Opcode:          DW_LNS_advance_pc
          Data:            64
        - Opcode:          DW_LNS_advance_line
          SData:           1
          Data:            0
        - Opcode:          DW_LNS_extended_op
          ExtLen:          1
          SubOpcode:       DW_LNE_end_sequence
          Data:            0
  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(yamldata);
  ASSERT_THAT_EXPECTED(ErrOrSections, Succeeded());
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);
  ASSERT_TRUE(DwarfContext.get() != nullptr);
  std::string errors;
  raw_string_ostream OS(errors);
  OutputAggregator OSAgg(&OS);
  CreatorT GC;
  DwarfTransformer DT(*DwarfContext, GC);
  const uint32_t ThreadCount = 1;
  ASSERT_THAT_ERROR(DT.convert(ThreadCount, OSAgg), Succeeded());
  ASSERT_THAT_ERROR(GC.finalize(OSAgg), Succeeded());
  SmallString<512> Str;
  raw_svector_ostream OutStrm(Str);
  const auto ByteOrder = llvm::endianness::native;
  FileWriter FW(OutStrm, ByteOrder);
  FW.setStringOffsetSize(GC.getStringOffsetSize());
  ASSERT_THAT_ERROR(GC.encode(FW), Succeeded());
  auto GROrErr = GsymReader::copyBuffer(OutStrm.str());
  ASSERT_THAT_EXPECTED(GROrErr, Succeeded());
  std::unique_ptr<GsymReader> &GR = *GROrErr;
  // There should be two functions in our GSYM.
  EXPECT_EQ(GR->getNumAddresses(), 2u);
  // Verify "foo" is correctly looked up for each of its addresses.
  for (uint64_t Addr = 0x1000; Addr < 0x1050; ++Addr) {
    auto ExpFI = GR->getFunctionInfo(Addr);
    ASSERT_THAT_EXPECTED(ExpFI, Succeeded());
    ASSERT_EQ(ExpFI->Range, AddressRange(0x1000, 0x1050));
    StringRef FuncName = GR->getString(ExpFI->Name);
    EXPECT_EQ(FuncName, "foo");
  }

  // Verify "bar" is correctly looked up for each of its addresses.
  for (uint64_t Addr = 0x1050; Addr < 0x1100; ++Addr) {
    auto ExpFI = GR->getFunctionInfo(Addr);
    ASSERT_THAT_EXPECTED(ExpFI, Succeeded());
    ASSERT_EQ(ExpFI->Range, AddressRange(0x1000, 0x1100));
    StringRef FuncName = GR->getString(ExpFI->Name);
    EXPECT_EQ(FuncName, "bar");
  }

  // Prior to the fix for this issue when we dumped an entire GSYM file, we
  // were using a function that would extract a FunctionInfo object for a
  // given address which caused us to always dump the first FunctionInfo
  // entry for a given address. We now dump it correctly using an address
  // index. Below we verify that we dump the right FunctionInfo gets dumped.

  SmallString<512> DumpStr;
  raw_svector_ostream DumpStrm(DumpStr);
  GR->dump(DumpStrm);

  // Make sure we see both "foo" and "bar" in the output of an entire GSYM
  // dump. Prior to this fix we would two "foo" entries.
  std::vector<std::string> ExpectedDumpLines = {
      "[0x0000000000001000 - 0x0000000000001050) \"foo\"",
      "[0x0000000000001000 - 0x0000000000001100) \"bar\""};
  // Make sure all expected errors are in the error stream for the two invalid
  // inlined functions that we removed due to invalid range scoping.
  for (const auto &Line : ExpectedDumpLines)
    EXPECT_TRUE(DumpStr.find(Line) != std::string::npos);
}

TEST(GSYMTest, TestLookupsOfOverlappingAndUnequalRanges) {
  TestLookupsOfOverlappingAndUnequalRanges<GsymCreatorV1>();
}
TEST(GSYMTest, TestLookupsOfOverlappingAndUnequalRangesV2) {
  TestLookupsOfOverlappingAndUnequalRanges<GsymCreatorV2>();
}

template <typename CreatorT> static void TestUnableToLocateDWO() {
  // Test that llvm-gsymutil will not produce "uanble to locate DWO file" for
  // Apple binaries. Apple uses DW_AT_GNU_dwo_id for non split DWARF purposes
  // and this makes llvm-gsymutil create warnings and errors.
  //
  // 0x0000000b: DW_TAG_compile_unit
  //               DW_AT_name        ("main.cpp")
  //               DW_AT_language    (DW_LANG_C)
  //               DW_AT_GNU_dwo_id  (0xfffffffe)
  StringRef yamldata = R"(
  debug_str:
    - ''
    - main.cpp
  debug_abbrev:
    - ID:              0
      Table:
        - Code:            0x1
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_language
              Form:            DW_FORM_udata
            - Attribute:       DW_AT_GNU_dwo_id
              Form:            DW_FORM_data4
  debug_info:
    - Length:          0x11
      Version:         4
      AbbrevTableID:   0
      AbbrOffset:      0x0
      AddrSize:        8
      Entries:
        - AbbrCode:        0x1
          Values:
            - Value:           0x1
            - Value:           0x2
            - Value:           0xFFFFFFFE
  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(yamldata);
  ASSERT_THAT_EXPECTED(ErrOrSections, Succeeded());
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);
  ASSERT_TRUE(DwarfContext.get() != nullptr);
  std::string errors;
  raw_string_ostream OS(errors);
  OutputAggregator OSAgg(&OS);
  CreatorT GC;
  // Make a DWARF transformer that is MachO (Apple) to avoid warnings about
  // not finding DWO files.
  DwarfTransformer DT(*DwarfContext, GC, /*LDCS=*/false, /*MachO*/ true);
  const uint32_t ThreadCount = 1;
  ASSERT_THAT_ERROR(DT.convert(ThreadCount, OSAgg), Succeeded());
  ASSERT_THAT_ERROR(GC.finalize(OSAgg), Succeeded());

  // Make sure this warning is not in the binary
  std::string warn("warning: Unable to retrieve DWO .debug_info section for");
  EXPECT_TRUE(errors.find(warn) == std::string::npos);
}

TEST(GSYMTest, TestUnableToLocateDWO) {
  TestUnableToLocateDWO<GsymCreatorV1>();
}
TEST(GSYMTest, TestUnableToLocateDWOV2) {
  TestUnableToLocateDWO<GsymCreatorV2>();
}

template <typename CreatorT>
static void TestDWARFTransformNoErrorForMissingFileDecl() {
  // Test that if llvm-gsymutil finds a line table for a compile unit and if
  // there are no matching entries for a function in that compile unit, that
  // it doesn't print out a error saying that a DIE has an invalid file index
  // if there is no DW_AT_decl_file attribute.
  //
  // 0x0000000b: DW_TAG_compile_unit
  //               DW_AT_name        ("main.cpp")
  //               DW_AT_language    (DW_LANG_C)
  //               DW_AT_stmt_list   (0x00000000)
  //
  // 0x00000015:   DW_TAG_subprogram
  //                 DW_AT_name      ("foo")
  //                 DW_AT_low_pc    (0x0000000000001000)
  //                 DW_AT_high_pc   (0x0000000000001050)
  //
  // 0x0000002a:   NULL
  //
  // Line table that has entries, but none that match "foo":
  //
  // Address            Line   Column File   ISA Discriminator OpIndex Flags
  // ------------------ ------ ------ ------ --- ------------- ------- -----
  // 0x0000000000002000     10      0      1   0             0       0 is_stmt
  // 0x0000000000002050     13      0      1   0             0       0 is_stmt

  StringRef yamldata = R"(
  debug_str:
    - ''
    - main.cpp
  debug_abbrev:
    - ID:              0
      Table:
        - Code:            0x1
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_language
              Form:            DW_FORM_udata
            - Attribute:       DW_AT_stmt_list
              Form:            DW_FORM_sec_offset
        - Code:            0x2
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_string
            - Attribute:       DW_AT_low_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_high_pc
              Form:            DW_FORM_addr
  debug_info:
    - Length:          0x27
      Version:         4
      AbbrevTableID:   0
      AbbrOffset:      0x0
      AddrSize:        8
      Entries:
        - AbbrCode:        0x1
          Values:
            - Value:           0x1
            - Value:           0x2
            - Value:           0x0
        - AbbrCode:        0x2
          Values:
            - Value:           0xDEADBEEFDEADBEEF
              CStr:            foo
            - Value:           0x1000
            - Value:           0x1050
        - AbbrCode:        0x0
  debug_line:
    - Length:          58
      Version:         2
      PrologueLength:  31
      MinInstLength:   1
      DefaultIsStmt:   1
      LineBase:        251
      LineRange:       14
      OpcodeBase:      13
      StandardOpcodeLengths: [ 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1 ]
      Files:
        - Name:            main.cpp
          DirIdx:          0
          ModTime:         0
          Length:          0
      Opcodes:
        - Opcode:          DW_LNS_extended_op
          ExtLen:          9
          SubOpcode:       DW_LNE_set_address
          Data:            8192
        - Opcode:          DW_LNS_advance_line
          SData:           9
          Data:            0
        - Opcode:          DW_LNS_copy
          Data:            0
        - Opcode:          DW_LNS_advance_pc
          Data:            80
        - Opcode:          DW_LNS_advance_line
          SData:           3
          Data:            0
        - Opcode:          DW_LNS_extended_op
          ExtLen:          1
          SubOpcode:       DW_LNE_end_sequence
          Data:            0
  )";
  auto ErrOrSections = DWARFYAML::emitDebugSections(yamldata);
  ASSERT_THAT_EXPECTED(ErrOrSections, Succeeded());
  std::unique_ptr<DWARFContext> DwarfContext =
      DWARFContext::create(*ErrOrSections, 8);
  ASSERT_TRUE(DwarfContext.get() != nullptr);
  std::string errors;
  raw_string_ostream OS(errors);
  OutputAggregator OSAgg(&OS);
  CreatorT GC;
  DwarfTransformer DT(*DwarfContext, GC);
  const uint32_t ThreadCount = 1;
  ASSERT_THAT_ERROR(DT.convert(ThreadCount, OSAgg), Succeeded());
  ASSERT_THAT_ERROR(GC.finalize(OSAgg), Succeeded());

  // Make sure this warning is not in the binary
  std::string error_str("error: function DIE at 0x00000015 has an invalid file "
                        "index 4294967295 in its DW_AT_decl_file attribute");
  EXPECT_TRUE(errors.find(error_str) == std::string::npos);
}

TEST(GSYMTest, TestDWARFTransformNoErrorForMissingFileDecl) {
  TestDWARFTransformNoErrorForMissingFileDecl<GsymCreatorV1>();
}
TEST(GSYMTest, TestDWARFTransformNoErrorForMissingFileDeclV2) {
  TestDWARFTransformNoErrorForMissingFileDecl<GsymCreatorV2>();
}

//===----------------------------------------------------------------------===//
// 8-byte string offset (gsym_strp_t) unit tests
//
// These tests verify that values > UINT32_MAX survive encoding/decoding
// without truncation when using 8-byte string offsets (V2 format).
//===----------------------------------------------------------------------===//

/// Test FunctionInfo encoding/decoding with Name > UINT32_MAX.
TEST(GSYMTest, TestFunctionInfoLargeNameOffset) {
  const gsym_strp_t LargeName = 0x1'2345'6789;
  const uint64_t BaseAddr = 0x5000;
  const uint32_t FuncSize = 0x100;

  // Encode.
  SmallString<128> Buf;
  raw_svector_ostream OS(Buf);
  FileWriter FW(OS, llvm::endianness::little);
  FW.setStringOffsetSize(8);

  FunctionInfo FI(BaseAddr, FuncSize, LargeName);
  auto EncResult = FI.encode(FW);
  ASSERT_THAT_EXPECTED(EncResult, Succeeded());

  // Decode.
  GsymDataExtractor Data(StringRef(Buf.data(), Buf.size()),
                         /*IsLittleEndian=*/true, 8);
  auto DecResult = FunctionInfo::decode(Data, BaseAddr);
  ASSERT_THAT_EXPECTED(DecResult, Succeeded());

  EXPECT_EQ(DecResult->Name, LargeName);
  EXPECT_EQ(DecResult->Range, AddressRange(BaseAddr, BaseAddr + FuncSize));
}

/// Test InlineInfo encoding/decoding with Name > UINT32_MAX.
TEST(GSYMTest, TestInlineInfoLargeNameOffset) {
  const gsym_strp_t LargeName = 0xAB'CDEF'0123;
  const uint64_t BaseAddr = 0x1000;

  InlineInfo II;
  II.Name = LargeName;
  II.CallFile = 1;
  II.CallLine = 42;
  II.Ranges.insert(AddressRange(0x1000, 0x1100));

  // Encode.
  SmallString<128> Buf;
  raw_svector_ostream OS(Buf);
  FileWriter FW(OS, llvm::endianness::little);
  FW.setStringOffsetSize(8);
  llvm::Error EncErr = II.encode(FW, BaseAddr);
  ASSERT_FALSE(bool(EncErr));

  // Decode.
  GsymDataExtractor Data(StringRef(Buf.data(), Buf.size()),
                         /*IsLittleEndian=*/true, 8);
  auto DecResult = InlineInfo::decode(Data, BaseAddr);
  ASSERT_THAT_EXPECTED(DecResult, Succeeded());

  EXPECT_EQ(DecResult->Name, LargeName);
  EXPECT_EQ(DecResult->CallFile, 1u);
  EXPECT_EQ(DecResult->CallLine, 42u);
  EXPECT_EQ(DecResult->Ranges.size(), 1u);
}

/// Test CallSiteInfo encoding/decoding with MatchRegex entries > UINT32_MAX.
TEST(GSYMTest, TestCallSiteInfoLargeMatchRegex) {
  const gsym_strp_t LargeRegex1 = 0x1'0000'0001;
  const gsym_strp_t LargeRegex2 = 0xDEAD'BEEF'CAFE'BABEull;

  CallSiteInfo CSI;
  CSI.ReturnOffset = 0x10;
  CSI.Flags = CallSiteInfo::InternalCall;
  CSI.MatchRegex.push_back(LargeRegex1);
  CSI.MatchRegex.push_back(LargeRegex2);

  // Encode.
  SmallString<128> Buf;
  raw_svector_ostream OS(Buf);
  FileWriter FW(OS, llvm::endianness::little);
  FW.setStringOffsetSize(8);
  llvm::Error EncErr = CSI.encode(FW);
  ASSERT_FALSE(bool(EncErr));

  // Decode.
  GsymDataExtractor Data(StringRef(Buf.data(), Buf.size()),
                         /*IsLittleEndian=*/true, 8);
  uint64_t Offset = 0;
  auto DecResult = CallSiteInfo::decode(Data, Offset);
  ASSERT_THAT_EXPECTED(DecResult, Succeeded());

  EXPECT_EQ(DecResult->ReturnOffset, 0x10u);
  EXPECT_EQ(DecResult->Flags, CallSiteInfo::InternalCall);
  ASSERT_EQ(DecResult->MatchRegex.size(), 2u);
  EXPECT_EQ(DecResult->MatchRegex[0], LargeRegex1);
  EXPECT_EQ(DecResult->MatchRegex[1], LargeRegex2);
}

/// Test CallSiteInfoCollection encoding/decoding with large string offsets.
TEST(GSYMTest, TestCallSiteInfoCollectionLargeMatchRegex) {
  const gsym_strp_t LargeRegex = 0xFF'FFFF'FFFF;

  CallSiteInfoCollection CSIC;
  CallSiteInfo CSI;
  CSI.ReturnOffset = 0x20;
  CSI.MatchRegex.push_back(LargeRegex);
  CSIC.CallSites.push_back(CSI);

  // Encode.
  SmallString<128> Buf;
  raw_svector_ostream OS(Buf);
  FileWriter FW(OS, llvm::endianness::little);
  FW.setStringOffsetSize(8);
  llvm::Error EncErr = CSIC.encode(FW);
  ASSERT_FALSE(bool(EncErr));

  // Decode.
  GsymDataExtractor Data(StringRef(Buf.data(), Buf.size()),
                         /*IsLittleEndian=*/true, 8);
  auto DecResult = CallSiteInfoCollection::decode(Data);
  ASSERT_THAT_EXPECTED(DecResult, Succeeded());

  ASSERT_EQ(DecResult->CallSites.size(), 1u);
  ASSERT_EQ(DecResult->CallSites[0].MatchRegex.size(), 1u);
  EXPECT_EQ(DecResult->CallSites[0].MatchRegex[0], LargeRegex);
}

/// Test FunctionInfo with all optional data containing large string offsets.
TEST(GSYMTest, TestFunctionInfoAllFieldsLargeOffsets) {
  const gsym_strp_t LargeFuncName = 0x2'0000'0000;
  const gsym_strp_t LargeInlineName = 0x3'0000'0000;
  const gsym_strp_t LargeRegex = 0x4'0000'0000;
  const uint64_t BaseAddr = 0x8000;

  FunctionInfo FI(BaseAddr, 0x200, LargeFuncName);

  // Add InlineInfo with large name.
  FI.Inline = InlineInfo();
  FI.Inline->Name = LargeInlineName;
  FI.Inline->CallFile = 1;
  FI.Inline->CallLine = 10;
  FI.Inline->Ranges.insert(AddressRange(0x8010, 0x8100));

  // Add CallSiteInfo with large regex offset.
  FI.CallSites = CallSiteInfoCollection();
  CallSiteInfo CSI;
  CSI.ReturnOffset = 0x30;
  CSI.MatchRegex.push_back(LargeRegex);
  FI.CallSites->CallSites.push_back(CSI);

  // Encode.
  SmallString<256> Buf;
  raw_svector_ostream OS(Buf);
  FileWriter FW(OS, llvm::endianness::little);
  FW.setStringOffsetSize(8);
  auto EncResult = FI.encode(FW);
  ASSERT_THAT_EXPECTED(EncResult, Succeeded());

  // Decode.
  GsymDataExtractor Data(StringRef(Buf.data(), Buf.size()),
                         /*IsLittleEndian=*/true, 8);
  auto DecResult = FunctionInfo::decode(Data, BaseAddr);
  ASSERT_THAT_EXPECTED(DecResult, Succeeded());

  EXPECT_EQ(DecResult->Name, LargeFuncName);
  ASSERT_TRUE(DecResult->Inline.has_value());
  EXPECT_EQ(DecResult->Inline->Name, LargeInlineName);
  ASSERT_TRUE(DecResult->CallSites.has_value());
  ASSERT_EQ(DecResult->CallSites->CallSites.size(), 1u);
  ASSERT_EQ(DecResult->CallSites->CallSites[0].MatchRegex.size(), 1u);
  EXPECT_EQ(DecResult->CallSites->CallSites[0].MatchRegex[0], LargeRegex);
}

/// Test MergedFunctionsInfo encoding/decoding with large string offsets.
TEST(GSYMTest, TestMergedFunctionsInfoLargeOffsets) {
  const gsym_strp_t LargeName1 = 0x5'0000'0001;
  const gsym_strp_t LargeName2 = 0x6'0000'0002;
  const uint64_t BaseAddr = 0x9000;

  MergedFunctionsInfo MFI;
  MFI.MergedFunctions.push_back(FunctionInfo(BaseAddr, 0x100, LargeName1));
  MFI.MergedFunctions.push_back(FunctionInfo(BaseAddr, 0x100, LargeName2));

  // Encode.
  SmallString<256> Buf;
  raw_svector_ostream OS(Buf);
  FileWriter FW(OS, llvm::endianness::little);
  FW.setStringOffsetSize(8);
  llvm::Error EncErr = MFI.encode(FW);
  ASSERT_FALSE(bool(EncErr));

  // Decode.
  GsymDataExtractor Data(StringRef(Buf.data(), Buf.size()),
                         /*IsLittleEndian=*/true, 8);
  auto DecResult = MergedFunctionsInfo::decode(Data, BaseAddr);
  ASSERT_THAT_EXPECTED(DecResult, Succeeded());

  ASSERT_EQ(DecResult->MergedFunctions.size(), 2u);
  EXPECT_EQ(DecResult->MergedFunctions[0].Name, LargeName1);
  EXPECT_EQ(DecResult->MergedFunctions[1].Name, LargeName2);
}
