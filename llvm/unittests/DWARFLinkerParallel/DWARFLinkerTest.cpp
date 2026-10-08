//===- llvm/unittest/DWARFLinkerParallel/DWARFLinkerTest.cpp --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/DWARFLinker/Utils.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLine.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace dwarf_linker;

#define DEVELOPER_DIR "/Applications/Xcode.app/Contents/Developer"

namespace {

TEST(DWARFLinker, PathTest) {
  EXPECT_EQ(guessDeveloperDir("/Foo"), "");
  EXPECT_EQ(guessDeveloperDir("Foo.sdk"), "");
  EXPECT_EQ(guessDeveloperDir(
                DEVELOPER_DIR
                "/Platforms/MacOSX.platform/Developer/SDKs/MacOSX14.4.sdk"),
            DEVELOPER_DIR);
  EXPECT_EQ(guessDeveloperDir(DEVELOPER_DIR "/SDKs/MacOSX.sdk"), DEVELOPER_DIR);
  EXPECT_TRUE(
      isInToolchainDir("/Library/Developer/Toolchains/"
                       "swift-DEVELOPMENT-SNAPSHOT-2024-05-15-a.xctoolchain/"
                       "usr/lib/swift/macosx/_StringProcessing.swiftmodule/"
                       "arm64-apple-macos.private.swiftinterface"));
  EXPECT_FALSE(isInToolchainDir("/Foo/not-an.xctoolchain/Bar/Baz"));
}

// Helpers for building DWARFDebugLine::LineTable fixtures. Only the fields
// that buildStmtSeqOffsetToFirstRowIndex inspects are populated; everything
// else defaults to Row's/Sequence's own defaults.
namespace {
void addRows(DWARFDebugLine::LineTable &LT, unsigned Count,
             ArrayRef<unsigned> EndSequenceIndices) {
  for (unsigned I = 0; I < Count; ++I)
    LT.Rows.emplace_back();
  for (unsigned I : EndSequenceIndices)
    LT.Rows[I].EndSequence = 1;
}

void addParsedSequence(DWARFDebugLine::LineTable &LT, uint64_t StmtSeqOffset,
                       unsigned FirstRowIndex) {
  DWARFDebugLine::Sequence S;
  S.StmtSeqOffset = StmtSeqOffset;
  S.FirstRowIndex = FirstRowIndex;
  LT.Sequences.push_back(S);
}
} // namespace

TEST(DWARFLinker, BuildStmtSeqOffsetToFirstRowIndex_ParserOnly) {
  // Single parser-registered sequence covering rows [0..3). The attribute
  // at offset 0x10 resolves straight from LT.Sequences.
  DWARFDebugLine::LineTable LT;
  addRows(LT, /*Count=*/3, /*EndSequenceIndices=*/{2});
  addParsedSequence(LT, /*StmtSeqOffset=*/0x10, /*FirstRowIndex=*/0);

  DenseMap<uint64_t, uint64_t> Result;
  buildStmtSeqOffsetToFirstRowIndex(LT, /*SortedStmtSeqOffsets=*/{0x10},
                                    Result);

  EXPECT_EQ(Result.lookup(0x10), 0u);
  EXPECT_EQ(Result.size(), 1u);
}

TEST(DWARFLinker, BuildStmtSeqOffsetToFirstRowIndex_ParserMissFallback) {
  // Two real sequences in the table (rows [0..3) and [3..5)), but the
  // parser only registered the second one. The fallback must recover
  // the first from the end_sequence boundary at row 2.
  DWARFDebugLine::LineTable LT;
  addRows(LT, /*Count=*/5, /*EndSequenceIndices=*/{2, 4});
  addParsedSequence(LT, /*StmtSeqOffset=*/0x20, /*FirstRowIndex=*/3);

  DenseMap<uint64_t, uint64_t> Result;
  buildStmtSeqOffsetToFirstRowIndex(LT, /*SortedStmtSeqOffsets=*/{0x10, 0x20},
                                    Result);

  EXPECT_EQ(Result.lookup(0x10), 0u); // recovered from row boundaries
  EXPECT_EQ(Result.lookup(0x20), 3u); // parser-registered ground truth
}

TEST(DWARFLinker, BuildStmtSeqOffsetToFirstRowIndex_RealignIgnoresStale) {
  // SeqStartRows is seeded with row 0 and then every row immediately
  // following an end_sequence marker. For 16 rows with end_sequence at
  // {2, 4, 8, 10, 14} that's {0, 3, 5, 9, 11, 15}.
  //
  //   StmtAttrs    {0x04, 0x08, 0x10, 0x12, 0x14}
  //   SeqStartRows {0,    3,    5,    9,    11,   15}
  //   Parser       {0x08 -> 9, 0x14 -> 15}
  //
  // Walk:
  //   anchor (0x08, 9): (0x04, 0) < anchor  -> pair  0x04 -> 0
  //                     0x08 not < 0x08     -> exit while
  //                     drop seq-starts < 9 -> {9, 11, 15}
  //                     ground truth        ->      0x08 -> 9
  //   anchor (0x14, 15): (0x10, 11) < anchor -> pair 0x10 -> 11
  //                      (0x12, 15) not < 15 -> exit while
  //                      drop stmt-attrs<0x14-> {0x14}
  //                      ground truth        ->      0x14 -> 15
  //
  // 0x12 is dropped because no safe row to pair it with remains.
  DWARFDebugLine::LineTable LT;
  addRows(LT, /*Count=*/16, /*EndSequenceIndices=*/{2, 4, 8, 10, 14});
  addParsedSequence(LT, 0x08, 9);
  addParsedSequence(LT, 0x14, 15);

  DenseMap<uint64_t, uint64_t> Result;
  buildStmtSeqOffsetToFirstRowIndex(LT, {0x04, 0x08, 0x10, 0x12, 0x14}, Result);

  EXPECT_EQ(Result.lookup(0x04), 0u);
  EXPECT_EQ(Result.lookup(0x08), 9u);
  EXPECT_EQ(Result.lookup(0x10), 11u);
  EXPECT_FALSE(Result.contains(0x12));
  EXPECT_EQ(Result.lookup(0x14), 15u);
}

TEST(DWARFLinker, BuildStmtSeqOffsetToFirstRowIndex_EmptyRows) {
  // When LT.Rows is empty the builder just copies the parser-registered
  // sequences over and returns — no fallback work possible.
  DWARFDebugLine::LineTable LT;
  addParsedSequence(LT, 0x10, 0);
  addParsedSequence(LT, 0x20, 42);

  DenseMap<uint64_t, uint64_t> Result;
  buildStmtSeqOffsetToFirstRowIndex(LT, {0x10, 0x20, 0x30}, Result);

  EXPECT_EQ(Result.lookup(0x10), 0u);
  EXPECT_EQ(Result.lookup(0x20), 42u);
  EXPECT_FALSE(Result.contains(0x30)); // no way to recover without rows
}

TEST(DWARFLinker, BuildStmtSeqOffsetToFirstRowIndex_NoAttributes) {
  // No stmt-seq attributes to resolve: the builder still seeds the map
  // from parser-registered sequences (harmless, helps shared callers).
  DWARFDebugLine::LineTable LT;
  addRows(LT, /*Count=*/3, /*EndSequenceIndices=*/{2});
  addParsedSequence(LT, 0x10, 0);

  DenseMap<uint64_t, uint64_t> Result;
  buildStmtSeqOffsetToFirstRowIndex(LT, /*SortedStmtSeqOffsets=*/{}, Result);

  EXPECT_EQ(Result.lookup(0x10), 0u);
  EXPECT_EQ(Result.size(), 1u);
}

} // anonymous namespace
