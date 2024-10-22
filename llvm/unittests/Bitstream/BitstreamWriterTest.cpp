//===- BitstreamWriterTest.cpp - Tests for BitstreamWriter ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Bitstream/BitstreamWriter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Bitstream/BitCodeEnums.h"
#include "llvm/Bitstream/BitstreamReader.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(BitstreamWriterTest, emitBlob) {
  SmallString<64> Buffer;
  BitstreamWriter W(Buffer);
  W.emitBlob("str", /* ShouldEmitSize */ false);
  EXPECT_EQ(StringRef("str\0", 4), Buffer);
}

TEST(BitstreamWriterTest, emitBlobWithSize) {
  SmallString<64> Buffer;
  {
    BitstreamWriter W(Buffer);
    W.emitBlob("str");
  }
  SmallString<64> Expected;
  {
    BitstreamWriter W(Expected);
    W.EmitVBR(3, 6);
    W.FlushToWord();
    W.Emit('s', 8);
    W.Emit('t', 8);
    W.Emit('r', 8);
    W.Emit(0, 8);
  }
  EXPECT_EQ(Expected.str(), Buffer);
}

TEST(BitstreamWriterTest, emitBlobEmpty) {
  SmallString<64> Buffer;
  BitstreamWriter W(Buffer);
  W.emitBlob("", /* ShouldEmitSize */ false);
  EXPECT_EQ(StringRef(""), Buffer);
}

TEST(BitstreamWriterTest, emitBlob4ByteAligned) {
  SmallString<64> Buffer;
  BitstreamWriter W(Buffer);
  W.emitBlob("str0", /* ShouldEmitSize */ false);
  EXPECT_EQ(StringRef("str0"), Buffer);
}

class BitstreamWriterFlushTest : public ::testing::TestWithParam<int> {
protected:
  // Any value after bitc::FIRST_APPLICATION_BLOCKID is good, but let's pick a
  // distinctive one.
  const unsigned BlkID = bitc::FIRST_APPLICATION_BLOCKID + 17;

  void write(StringRef TestFilePath, int FlushThreshold,
             llvm::function_ref<void(BitstreamWriter &)> Action) {
    std::error_code EC;
    raw_fd_stream Out(TestFilePath, EC);
    ASSERT_FALSE(EC);
    BitstreamWriter W(Out, FlushThreshold);
    Action(W);
  }
};

TEST_P(BitstreamWriterFlushTest, simpleExample) {
  llvm::unittest::TempFile TestFile("bitstream", "", "",
                                    /*Unique*/ true);
  write(TestFile.path(), GetParam(),
        [&](BitstreamWriter &W) { W.EmitVBR(42, 2); });

  ErrorOr<std::unique_ptr<MemoryBuffer>> MB =
      MemoryBuffer::getFile(TestFile.path());
  ASSERT_TRUE(!!MB);
  ASSERT_NE(*MB, nullptr);
  BitstreamCursor Cursor((*MB)->getBuffer());
  auto V = Cursor.ReadVBR(2);
  EXPECT_TRUE(!!V);
  EXPECT_EQ(*V, 42U);
}

TEST_P(BitstreamWriterFlushTest, subBlock) {
  llvm::unittest::TempFile TestFile("bitstream", "", "",
                                    /*Unique*/ true);
  write(TestFile.path(), GetParam(), [&](BitstreamWriter &W) {
    W.EnterSubblock(BlkID, 2);
    W.EmitVBR(42, 2);
    W.ExitBlock();
  });
  ErrorOr<std::unique_ptr<MemoryBuffer>> MB =
      MemoryBuffer::getFile(TestFile.path());
  ASSERT_TRUE(!!MB);
  ASSERT_NE(*MB, nullptr);
  BitstreamCursor Cursor((*MB)->getBuffer());
  auto Blk = Cursor.advance(BitstreamCursor::AF_DontAutoprocessAbbrevs);
  ASSERT_TRUE(!!Blk);
  EXPECT_EQ(Blk->Kind, BitstreamEntry::SubBlock);
  EXPECT_EQ(Blk->ID, BlkID);
  EXPECT_FALSE(Cursor.EnterSubBlock(BlkID));
  auto V = Cursor.ReadVBR(2);
  EXPECT_TRUE(!!V);
  EXPECT_EQ(*V, 42U);
  // ReadBlockEnd() returns false if it actually read the block end.
  EXPECT_FALSE(Cursor.ReadBlockEnd());
  EXPECT_TRUE(Cursor.AtEndOfStream());
}

TEST_P(BitstreamWriterFlushTest, blobRawRead) {
  llvm::unittest::TempFile TestFile("bitstream", "", "",
                                    /*Unique*/ true);
  write(TestFile.path(), GetParam(), [&](BitstreamWriter &W) {
    W.emitBlob("str", /* ShouldEmitSize */ false);
  });

  ErrorOr<std::unique_ptr<MemoryBuffer>> MB =
      MemoryBuffer::getFile(TestFile.path());
  ASSERT_TRUE(!!MB);
  ASSERT_NE(*MB, nullptr);
  EXPECT_EQ(StringRef("str\0", 4), (*MB)->getBuffer());
}

INSTANTIATE_TEST_SUITE_P(BitstreamWriterFlushCases, BitstreamWriterFlushTest,
                         ::testing::Values(0, 1 /*MB*/));
} // end namespace
