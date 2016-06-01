//===- llvm/unittest/DebugInfo/PDB/MappedBlockStreamTest.cpp --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <unordered_map>

#include "llvm/DebugInfo/CodeView/StreamReader.h"
#include "llvm/DebugInfo/CodeView/StreamRef.h"
#include "llvm/DebugInfo/PDB/Raw/IPDBFile.h"
#include "llvm/DebugInfo/PDB/Raw/MappedBlockStream.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::pdb;

namespace {

#define EXPECT_NO_ERROR(Err)                                                   \
  {                                                                            \
    auto E = std::move(Err);                                                   \
    EXPECT_FALSE(static_cast<bool>(E));                                        \
    if (E)                                                                     \
      consumeError(std::move(E));                                              \
  }

#define EXPECT_ERROR(Err)                                                      \
  {                                                                            \
    auto E = std::move(Err);                                                   \
    EXPECT_TRUE(static_cast<bool>(E));                                         \
    if (E)                                                                     \
      consumeError(std::move(E));                                              \
  }

class DiscontiguousFile : public IPDBFile {
public:
  DiscontiguousFile()
      : Blocks{0, 1, 2, 5, 4, 3, 6, 7, 8, 9},
        Data{'A', 'B', 'C', 'F', 'E', 'D', 'G', 'H', 'I', 'J'} {}

  virtual uint32_t getBlockSize() const override { return 1; }
  virtual uint32_t getBlockCount() const override { return 10; }
  virtual uint32_t getNumStreams() const override { return 1; }
  virtual uint32_t getStreamByteSize(uint32_t StreamIndex) const override {
    return getBlockCount() * getBlockSize();
  }
  virtual ArrayRef<uint32_t>
  getStreamBlockList(uint32_t StreamIndex) const override {
    if (StreamIndex != 0)
      return ArrayRef<uint32_t>();
    return Blocks;
  }
  virtual StringRef getBlockData(uint32_t BlockIndex,
                                 uint32_t NumBytes) const override {
    return StringRef(&Data[BlockIndex], NumBytes);
  }

private:
  std::vector<uint32_t> Blocks;
  std::vector<char> Data;
};

// Tests that a read which is entirely contained within a single block works
// and does not allocate.
TEST(MappedBlockStreamTest, ZeroCopyReadNoBreak) {
  DiscontiguousFile F;
  MappedBlockStream S(0, F);
  StreamReader R(S);
  StringRef Str;
  EXPECT_NO_ERROR(R.readFixedString(Str, 1));
  EXPECT_EQ(Str, StringRef("A"));
  EXPECT_EQ(0, S.getNumBytesCopied());
}

// Tests that a read which outputs into a full destination buffer works and
// does not fail due to the length of the output buffer.
TEST(MappedBlockStreamTest, ReadOntoNonEmptyBuffer) {
  DiscontiguousFile F;
  MappedBlockStream S(0, F);
  StreamReader R(S);
  StringRef Str = "ZYXWVUTSRQPONMLKJIHGFEDCBA";
  EXPECT_NO_ERROR(R.readFixedString(Str, 1));
  EXPECT_EQ(Str, StringRef("A"));
  EXPECT_EQ(0, S.getNumBytesCopied());
}

// Tests that a read which crosses a block boundary, but where the subsequent
// blocks are still contiguous in memory to the previous block works and does
// not allocate memory.
TEST(MappedBlockStreamTest, ZeroCopyReadContiguousBreak) {
  DiscontiguousFile F;
  MappedBlockStream S(0, F);
  StreamReader R(S);
  StringRef Str;
  EXPECT_NO_ERROR(R.readFixedString(Str, 2));
  EXPECT_EQ(Str, StringRef("AB"));
  EXPECT_EQ(0, S.getNumBytesCopied());

  R.setOffset(6);
  EXPECT_NO_ERROR(R.readFixedString(Str, 4));
  EXPECT_EQ(Str, StringRef("GHIJ"));
  EXPECT_EQ(0, S.getNumBytesCopied());
}

// Tests that a read which crosses a block boundary and cannot be referenced
// contiguously works and allocates only the precise amount of bytes
// requested.
TEST(MappedBlockStreamTest, CopyReadNonContiguousBreak) {
  DiscontiguousFile F;
  MappedBlockStream S(0, F);
  StreamReader R(S);
  StringRef Str;
  EXPECT_NO_ERROR(R.readFixedString(Str, 10));
  EXPECT_EQ(Str, StringRef("ABCDEFGHIJ"));
  EXPECT_EQ(10, S.getNumBytesCopied());
}

// Test that an out of bounds read which doesn't cross a block boundary
// fails and allocates no memory.
TEST(MappedBlockStreamTest, InvalidReadSizeNoBreak) {
  DiscontiguousFile F;
  MappedBlockStream S(0, F);
  StreamReader R(S);
  StringRef Str;

  R.setOffset(10);
  EXPECT_ERROR(R.readFixedString(Str, 1));
  EXPECT_EQ(0, S.getNumBytesCopied());
}

// Test that an out of bounds read which crosses a contiguous block boundary
// fails and allocates no memory.
TEST(MappedBlockStreamTest, InvalidReadSizeContiguousBreak) {
  DiscontiguousFile F;
  MappedBlockStream S(0, F);
  StreamReader R(S);
  StringRef Str;

  R.setOffset(6);
  EXPECT_ERROR(R.readFixedString(Str, 5));
  EXPECT_EQ(0, S.getNumBytesCopied());
}

// Test that an out of bounds read which crosses a discontiguous block
// boundary fails and allocates no memory.
TEST(MappedBlockStreamTest, InvalidReadSizeNonContiguousBreak) {
  DiscontiguousFile F;
  MappedBlockStream S(0, F);
  StreamReader R(S);
  StringRef Str;

  EXPECT_ERROR(R.readFixedString(Str, 11));
  EXPECT_EQ(0, S.getNumBytesCopied());
}

} // end anonymous namespace
