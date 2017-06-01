//===- BitstreamReaderTest.cpp - Tests for BitstreamReader ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/Bitcode/BitstreamReader.h"
#include "llvm/Bitcode/BitstreamWriter.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(BitstreamReaderTest, AtEndOfStream) {
  uint8_t Bytes[4] = {
    0x00, 0x01, 0x02, 0x03
  };
  BitstreamCursor Cursor(Bytes);

  EXPECT_FALSE(Cursor.AtEndOfStream());
  (void)Cursor.Read(8);
  EXPECT_FALSE(Cursor.AtEndOfStream());
  (void)Cursor.Read(24);
  EXPECT_TRUE(Cursor.AtEndOfStream());

  Cursor.JumpToBit(0);
  EXPECT_FALSE(Cursor.AtEndOfStream());

  Cursor.JumpToBit(32);
  EXPECT_TRUE(Cursor.AtEndOfStream());
}

TEST(BitstreamReaderTest, AtEndOfStreamJump) {
  uint8_t Bytes[4] = {
    0x00, 0x01, 0x02, 0x03
  };
  BitstreamCursor Cursor(Bytes);

  Cursor.JumpToBit(32);
  EXPECT_TRUE(Cursor.AtEndOfStream());
}

TEST(BitstreamReaderTest, AtEndOfStreamEmpty) {
  BitstreamCursor Cursor(ArrayRef<uint8_t>{});

  EXPECT_TRUE(Cursor.AtEndOfStream());
}

TEST(BitstreamReaderTest, getCurrentByteNo) {
  uint8_t Bytes[] = {0x00, 0x01, 0x02, 0x03};
  SimpleBitstreamCursor Cursor(Bytes);

  for (unsigned I = 0, E = 32; I != E; ++I) {
    EXPECT_EQ(I / 8, Cursor.getCurrentByteNo());
    (void)Cursor.Read(1);
  }
  EXPECT_EQ(4u, Cursor.getCurrentByteNo());
}

TEST(BitstreamReaderTest, getPointerToByte) {
  uint8_t Bytes[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07};
  SimpleBitstreamCursor Cursor(Bytes);

  for (unsigned I = 0, E = 8; I != E; ++I) {
    EXPECT_EQ(Bytes + I, Cursor.getPointerToByte(I, 1));
  }
}

TEST(BitstreamReaderTest, getPointerToBit) {
  uint8_t Bytes[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07};
  SimpleBitstreamCursor Cursor(Bytes);

  for (unsigned I = 0, E = 8; I != E; ++I) {
    EXPECT_EQ(Bytes + I, Cursor.getPointerToBit(I * 8, 1));
  }
}

TEST(BitstreamReaderTest, readRecordWithBlobWhileStreaming) {
  SmallVector<uint8_t, 1> BlobData;
  for (unsigned I = 0, E = 1024; I != E; ++I)
    BlobData.push_back(I);

  // Try a bunch of different sizes.
  const unsigned Magic = 0x12345678;
  const unsigned BlockID = bitc::FIRST_APPLICATION_BLOCKID;
  const unsigned RecordID = 1;
  for (unsigned I = 0, BlobSize = 0, E = BlobData.size(); BlobSize < E;
       BlobSize += ++I) {
    StringRef BlobIn((const char *)BlobData.begin(), BlobSize);

    // Write the bitcode.
    SmallVector<char, 1> Buffer;
    unsigned AbbrevID;
    {
      BitstreamWriter Stream(Buffer);
      Stream.Emit(Magic, 32);
      Stream.EnterSubblock(BlockID, 3);

      auto Abbrev = std::make_shared<BitCodeAbbrev>();
      Abbrev->Add(BitCodeAbbrevOp(RecordID));
      Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));
      AbbrevID = Stream.EmitAbbrev(std::move(Abbrev));
      unsigned Record[] = {RecordID};
      Stream.EmitRecordWithBlob(AbbrevID, makeArrayRef(Record), BlobIn);

      Stream.ExitBlock();
    }

    // Stream the buffer into the reader.
    BitstreamCursor Stream(
        ArrayRef<uint8_t>((const uint8_t *)Buffer.begin(), Buffer.size()));

    // Header.  Included in test so that we can run llvm-bcanalyzer to debug
    // when there are problems.
    ASSERT_EQ(Magic, Stream.Read(32));

    // Block.
    BitstreamEntry Entry =
        Stream.advance(BitstreamCursor::AF_DontAutoprocessAbbrevs);
    ASSERT_EQ(BitstreamEntry::SubBlock, Entry.Kind);
    ASSERT_EQ(BlockID, Entry.ID);
    ASSERT_FALSE(Stream.EnterSubBlock(BlockID));

    // Abbreviation.
    Entry = Stream.advance();
    ASSERT_EQ(BitstreamEntry::Record, Entry.Kind);
    ASSERT_EQ(AbbrevID, Entry.ID);

    // Record.
    StringRef BlobOut;
    SmallVector<uint64_t, 1> Record;
    ASSERT_EQ(RecordID, Stream.readRecord(Entry.ID, Record, &BlobOut));
    EXPECT_TRUE(Record.empty());
    EXPECT_EQ(BlobIn, BlobOut);
  }
}

TEST(BitstreamReaderTest, shortRead) {
  uint8_t Bytes[] = {8, 7, 6, 5, 4, 3, 2, 1};
  for (unsigned I = 1; I != 8; ++I) {
    SimpleBitstreamCursor Cursor(ArrayRef<uint8_t>(Bytes, I));
    EXPECT_EQ(8ull, Cursor.Read(8));
  }
}

} // end anonymous namespace
