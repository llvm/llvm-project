//===- llvm/unittest/DebugInfo/GSYMTest.cpp -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/DebugInfo/GSYM/FileEntry.h"
#include "llvm/DebugInfo/GSYM/FunctionInfo.h"
#include "llvm/DebugInfo/GSYM/InlineInfo.h"
#include "llvm/DebugInfo/GSYM/Range.h"
#include "llvm/DebugInfo/GSYM/StringTable.h"
#include "llvm/Testing/Support/Error.h"

#include "gtest/gtest.h"
#include <string>

using namespace llvm;
using namespace gsym;

TEST(GSYMTest, TestFileEntry) {
  // Make sure default constructed GSYM FileEntry has zeroes in the
  // directory and basename string table indexes.
  FileEntry empty1;
  FileEntry empty2;
  EXPECT_EQ(empty1.Dir, 0u);
  EXPECT_EQ(empty1.Base, 0u);
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

TEST(GSYMTest, TestFunctionInfo) {
  // Test GSYM FunctionInfo structs and functionality.
  FunctionInfo invalid;
  EXPECT_FALSE(invalid.isValid());
  EXPECT_FALSE(invalid.hasRichInfo());
  const uint64_t StartAddr = 0x1000;
  const uint64_t EndAddr = 0x1100;
  const uint64_t Size = EndAddr - StartAddr;
  const uint32_t NameOffset = 30;
  FunctionInfo FI(StartAddr, Size, NameOffset);
  EXPECT_TRUE(FI.isValid());
  EXPECT_FALSE(FI.hasRichInfo());
  EXPECT_EQ(FI.startAddress(), StartAddr);
  EXPECT_EQ(FI.endAddress(), EndAddr);
  EXPECT_EQ(FI.size(), Size);
  const uint32_t FileIdx = 1;
  const uint32_t Line = 12;
  FI.Lines.push_back(LineEntry(StartAddr, FileIdx, Line));
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
  B.setStartAddress(0x2000);
  EXPECT_NE(B, A2);
  // Make sure things are not equal if they only differ by size.
  B = A2;
  B.setSize(0x101);
  EXPECT_NE(B, A2);
  // Make sure things are not equal if they only differ by name.
  B = A2;
  B.Name = 60;
  EXPECT_NE(B, A2);
  // Check < operator.
  // Check less than where address differs.
  B = A2;
  B.setStartAddress(A2.startAddress() + 0x1000);
  EXPECT_LT(A1, B);

  // We use the < operator to take a variety of different FunctionInfo
  // structs from a variety of sources: symtab, debug info, runtime info
  // and we sort them and want the sorting to allow us to quickly get the
  // best version of a function info.
  FunctionInfo FISymtab(StartAddr, Size, NameOffset);
  FunctionInfo FIWithLines(StartAddr, Size, NameOffset);
  FIWithLines.Lines.push_back(LineEntry(StartAddr, FileIdx, Line));
  // Test that a FunctionInfo with just a name and size is less than one
  // that has name, size and any number of line table entries
  EXPECT_LT(FISymtab, FIWithLines);

  FunctionInfo FIWithLinesAndInline = FIWithLines;
  FIWithLinesAndInline.Inline.Ranges.insert(
      AddressRange(StartAddr, StartAddr + 0x10));
  // Test that a FunctionInfo with name, size, and line entries is less than
  // the same one with valid inline info
  EXPECT_LT(FIWithLines, FIWithLinesAndInline);

  // Test if we have an entry with lines and one with more lines for the same
  // range, the ones with more lines is greater than the one with less.
  FunctionInfo FIWithMoreLines = FIWithLines;
  FIWithMoreLines.Lines.push_back(LineEntry(StartAddr, FileIdx, Line + 5));
  EXPECT_LT(FIWithLines, FIWithMoreLines);

  // Test that if we have the same number of lines we compare the line entries
  // in the FunctionInfo.Lines vector.
  FunctionInfo FIWithLinesWithHigherAddress = FIWithLines;
  FIWithLinesWithHigherAddress.Lines[0].Addr += 0x10;
  EXPECT_LT(FIWithLines, FIWithLinesWithHigherAddress);
}

TEST(GSYMTest, TestInlineInfo) {
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
  // Inline1       [0x150-0x160) Name = 1, File = 1, Line = 11
  // Inline1Sub1     [0x152-0x155) Name = 2, File = 2, Line = 22
  // Inline1Sub2     [0x157-0x158) Name = 3, File = 3, Line = 33
  InlineInfo Root;
  Root.Ranges.insert(AddressRange(0x100, 0x200));
  InlineInfo Inline1;
  Inline1.Ranges.insert(AddressRange(0x150, 0x160));
  Inline1.Name = 1;
  Inline1.CallFile = 1;
  Inline1.CallLine = 11;
  InlineInfo Inline1Sub1;
  Inline1Sub1.Ranges.insert(AddressRange(0x152, 0x155));
  Inline1Sub1.Name = 2;
  Inline1Sub1.CallFile = 2;
  Inline1Sub1.CallLine = 22;
  InlineInfo Inline1Sub2;
  Inline1Sub2.Ranges.insert(AddressRange(0x157, 0x158));
  Inline1Sub2.Name = 3;
  Inline1Sub2.CallFile = 3;
  Inline1Sub2.CallLine = 33;
  Inline1.Children.push_back(Inline1Sub1);
  Inline1.Children.push_back(Inline1Sub2);
  Root.Children.push_back(Inline1);

  // Make sure an address that is out of range won't match
  EXPECT_FALSE(Root.getInlineStack(0x50));

  // Verify that we get no inline stacks for addresses out of [0x100-0x200)
  EXPECT_FALSE(Root.getInlineStack(Root.Ranges[0].Start - 1));
  EXPECT_FALSE(Root.getInlineStack(Root.Ranges[0].End));

  // Verify we get no inline stack entries for addresses that are in
  // [0x100-0x200) but not in [0x150-0x160)
  EXPECT_FALSE(Root.getInlineStack(Inline1.Ranges[0].Start - 1));
  EXPECT_FALSE(Root.getInlineStack(Inline1.Ranges[0].End));

  // Verify we get one inline stack entry for addresses that are in
  // [[0x150-0x160)) but not in [0x152-0x155) or [0x157-0x158)
  auto InlineInfos = Root.getInlineStack(Inline1.Ranges[0].Start);
  ASSERT_TRUE(InlineInfos);
  ASSERT_EQ(InlineInfos->size(), 1u);
  ASSERT_EQ(*InlineInfos->at(0), Inline1);
  InlineInfos = Root.getInlineStack(Inline1.Ranges[0].End - 1);
  EXPECT_TRUE(InlineInfos);
  ASSERT_EQ(InlineInfos->size(), 1u);
  ASSERT_EQ(*InlineInfos->at(0), Inline1);

  // Verify we get two inline stack entries for addresses that are in
  // [0x152-0x155)
  InlineInfos = Root.getInlineStack(Inline1Sub1.Ranges[0].Start);
  EXPECT_TRUE(InlineInfos);
  ASSERT_EQ(InlineInfos->size(), 2u);
  ASSERT_EQ(*InlineInfos->at(0), Inline1Sub1);
  ASSERT_EQ(*InlineInfos->at(1), Inline1);
  InlineInfos = Root.getInlineStack(Inline1Sub1.Ranges[0].End - 1);
  EXPECT_TRUE(InlineInfos);
  ASSERT_EQ(InlineInfos->size(), 2u);
  ASSERT_EQ(*InlineInfos->at(0), Inline1Sub1);
  ASSERT_EQ(*InlineInfos->at(1), Inline1);

  // Verify we get two inline stack entries for addresses that are in
  // [0x157-0x158)
  InlineInfos = Root.getInlineStack(Inline1Sub2.Ranges[0].Start);
  EXPECT_TRUE(InlineInfos);
  ASSERT_EQ(InlineInfos->size(), 2u);
  ASSERT_EQ(*InlineInfos->at(0), Inline1Sub2);
  ASSERT_EQ(*InlineInfos->at(1), Inline1);
  InlineInfos = Root.getInlineStack(Inline1Sub2.Ranges[0].End - 1);
  EXPECT_TRUE(InlineInfos);
  ASSERT_EQ(InlineInfos->size(), 2u);
  ASSERT_EQ(*InlineInfos->at(0), Inline1Sub2);
  ASSERT_EQ(*InlineInfos->at(1), Inline1);
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

TEST(GSYMTest, TestRanges) {
  // test llvm::gsym::AddressRange.
  const uint64_t StartAddr = 0x1000;
  const uint64_t EndAddr = 0x2000;
  // Verify constructor and API to ensure it takes start and end address.
  const AddressRange Range(StartAddr, EndAddr);
  EXPECT_EQ(Range.size(), EndAddr - StartAddr);

  // Verify llvm::gsym::AddressRange::contains().
  EXPECT_FALSE(Range.contains(0));
  EXPECT_FALSE(Range.contains(StartAddr - 1));
  EXPECT_TRUE(Range.contains(StartAddr));
  EXPECT_TRUE(Range.contains(EndAddr - 1));
  EXPECT_FALSE(Range.contains(EndAddr));
  EXPECT_FALSE(Range.contains(UINT64_MAX));

  const AddressRange RangeSame(StartAddr, EndAddr);
  const AddressRange RangeDifferentStart(StartAddr + 1, EndAddr);
  const AddressRange RangeDifferentEnd(StartAddr, EndAddr + 1);
  const AddressRange RangeDifferentStartEnd(StartAddr + 1, EndAddr + 1);
  // Test == and != with values that are the same
  EXPECT_EQ(Range, RangeSame);
  EXPECT_FALSE(Range != RangeSame);
  // Test == and != with values that are the different
  EXPECT_NE(Range, RangeDifferentStart);
  EXPECT_NE(Range, RangeDifferentEnd);
  EXPECT_NE(Range, RangeDifferentStartEnd);
  EXPECT_FALSE(Range == RangeDifferentStart);
  EXPECT_FALSE(Range == RangeDifferentEnd);
  EXPECT_FALSE(Range == RangeDifferentStartEnd);

  // Test "bool operator<(const AddressRange &, const AddressRange &)".
  EXPECT_FALSE(Range < RangeSame);
  EXPECT_FALSE(RangeSame < Range);
  EXPECT_LT(Range, RangeDifferentStart);
  EXPECT_LT(Range, RangeDifferentEnd);
  EXPECT_LT(Range, RangeDifferentStartEnd);
  // Test "bool operator<(const AddressRange &, uint64_t)"
  EXPECT_LT(Range.Start, StartAddr + 1);
  // Test "bool operator<(uint64_t, const AddressRange &)"
  EXPECT_LT(StartAddr - 1, Range.Start);

  // Verify llvm::gsym::AddressRange::isContiguousWith() and
  // llvm::gsym::AddressRange::intersects().
  const AddressRange EndsBeforeRangeStart(0, StartAddr - 1);
  const AddressRange EndsAtRangeStart(0, StartAddr);
  const AddressRange OverlapsRangeStart(StartAddr - 1, StartAddr + 1);
  const AddressRange InsideRange(StartAddr + 1, EndAddr - 1);
  const AddressRange OverlapsRangeEnd(EndAddr - 1, EndAddr + 1);
  const AddressRange StartsAtRangeEnd(EndAddr, EndAddr + 0x100);
  const AddressRange StartsAfterRangeEnd(EndAddr + 1, EndAddr + 0x100);

  EXPECT_FALSE(Range.intersects(EndsBeforeRangeStart));
  EXPECT_FALSE(Range.intersects(EndsAtRangeStart));
  EXPECT_TRUE(Range.intersects(OverlapsRangeStart));
  EXPECT_TRUE(Range.intersects(InsideRange));
  EXPECT_TRUE(Range.intersects(OverlapsRangeEnd));
  EXPECT_FALSE(Range.intersects(StartsAtRangeEnd));
  EXPECT_FALSE(Range.intersects(StartsAfterRangeEnd));

  // Test the functions that maintain GSYM address ranges:
  //  "bool AddressRange::contains(uint64_t Addr) const;"
  //  "void AddressRanges::insert(const AddressRange &R);"
  AddressRanges Ranges;
  Ranges.insert(AddressRange(0x1000, 0x2000));
  Ranges.insert(AddressRange(0x2000, 0x3000));
  Ranges.insert(AddressRange(0x4000, 0x5000));

  EXPECT_FALSE(Ranges.contains(0));
  EXPECT_FALSE(Ranges.contains(0x1000 - 1));
  EXPECT_TRUE(Ranges.contains(0x1000));
  EXPECT_TRUE(Ranges.contains(0x2000));
  EXPECT_TRUE(Ranges.contains(0x4000));
  EXPECT_TRUE(Ranges.contains(0x2000 - 1));
  EXPECT_TRUE(Ranges.contains(0x3000 - 1));
  EXPECT_FALSE(Ranges.contains(0x3000 + 1));
  EXPECT_TRUE(Ranges.contains(0x5000 - 1));
  EXPECT_FALSE(Ranges.contains(0x5000 + 1));
  EXPECT_FALSE(Ranges.contains(UINT64_MAX));

  // Verify that intersecting ranges get combined
  Ranges.clear();
  Ranges.insert(AddressRange(0x1100, 0x1F00));
  // Verify a wholy contained range that is added doesn't do anything.
  Ranges.insert(AddressRange(0x1500, 0x1F00));
  EXPECT_EQ(Ranges.size(), 1u);
  EXPECT_EQ(Ranges[0], AddressRange(0x1100, 0x1F00));

  // Verify a range that starts before and intersects gets combined.
  Ranges.insert(AddressRange(0x1000, Ranges[0].Start + 1));
  EXPECT_EQ(Ranges.size(), 1u);
  EXPECT_EQ(Ranges[0], AddressRange(0x1000, 0x1F00));

  // Verify a range that starts inside and extends ranges gets combined.
  Ranges.insert(AddressRange(Ranges[0].End - 1, 0x2000));
  EXPECT_EQ(Ranges.size(), 1u);
  EXPECT_EQ(Ranges[0], AddressRange(0x1000, 0x2000));

  // Verify that adjacent ranges don't get combined
  Ranges.insert(AddressRange(0x2000, 0x3000));
  EXPECT_EQ(Ranges.size(), 2u);
  EXPECT_EQ(Ranges[0], AddressRange(0x1000, 0x2000));
  EXPECT_EQ(Ranges[1], AddressRange(0x2000, 0x3000));
  // Verify if we add an address range that intersects two ranges
  // that they get combined
  Ranges.insert(AddressRange(Ranges[0].End - 1, Ranges[1].Start + 1));
  EXPECT_EQ(Ranges.size(), 1u);
  EXPECT_EQ(Ranges[0], AddressRange(0x1000, 0x3000));

  Ranges.insert(AddressRange(0x3000, 0x4000));
  Ranges.insert(AddressRange(0x4000, 0x5000));
  Ranges.insert(AddressRange(0x2000, 0x4500));
  EXPECT_EQ(Ranges.size(), 1u);
  EXPECT_EQ(Ranges[0], AddressRange(0x1000, 0x5000));
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
