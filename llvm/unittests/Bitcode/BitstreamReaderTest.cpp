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
#include "llvm/Support/StreamingMemoryObject.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class BufferStreamer : public DataStreamer {
  StringRef Buffer;

public:
  BufferStreamer(StringRef Buffer) : Buffer(Buffer) {}
  size_t GetBytes(unsigned char *OutBuffer, size_t Length) override {
    if (Length >= Buffer.size())
      Length = Buffer.size();

    std::copy(Buffer.begin(), Buffer.begin() + Length, OutBuffer);
    Buffer = Buffer.drop_front(Length);
    return Length;
  }
};

TEST(BitstreamReaderTest, AtEndOfStream) {
  uint8_t Bytes[4] = {
    0x00, 0x01, 0x02, 0x03
  };
  BitstreamReader Reader(std::begin(Bytes), std::end(Bytes));
  BitstreamCursor Cursor(Reader);

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
  BitstreamReader Reader(std::begin(Bytes), std::end(Bytes));
  BitstreamCursor Cursor(Reader);

  Cursor.JumpToBit(32);
  EXPECT_TRUE(Cursor.AtEndOfStream());
}

TEST(BitstreamReaderTest, AtEndOfStreamEmpty) {
  uint8_t Dummy = 0xFF;
  BitstreamReader Reader(&Dummy, &Dummy);
  BitstreamCursor Cursor(Reader);

  EXPECT_TRUE(Cursor.AtEndOfStream());
}

TEST(BitstreamReaderTest, getCurrentByteNo) {
  uint8_t Bytes[] = {0x00, 0x01, 0x02, 0x03};
  BitstreamReader Reader(std::begin(Bytes), std::end(Bytes));
  SimpleBitstreamCursor Cursor(Reader);

  for (unsigned I = 0, E = 33; I != E; ++I) {
    EXPECT_EQ(I / 8, Cursor.getCurrentByteNo());
    (void)Cursor.Read(1);
  }
  EXPECT_EQ(4u, Cursor.getCurrentByteNo());
}

TEST(BitstreamReaderTest, getPointerToByte) {
  uint8_t Bytes[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07};
  BitstreamReader Reader(std::begin(Bytes), std::end(Bytes));
  SimpleBitstreamCursor Cursor(Reader);

  for (unsigned I = 0, E = 8; I != E; ++I) {
    EXPECT_EQ(Bytes + I, Cursor.getPointerToByte(I, 1));
  }
}

TEST(BitstreamReaderTest, getPointerToBit) {
  uint8_t Bytes[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07};
  BitstreamReader Reader(std::begin(Bytes), std::end(Bytes));
  SimpleBitstreamCursor Cursor(Reader);

  for (unsigned I = 0, E = 8; I != E; ++I) {
    EXPECT_EQ(Bytes + I, Cursor.getPointerToBit(I * 8, 1));
  }
}

TEST(BitstreamReaderTest, jumpToPointer) {
  uint8_t Bytes[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07};
  BitstreamReader Reader(std::begin(Bytes), std::end(Bytes));
  SimpleBitstreamCursor Cursor(Reader);

  for (unsigned I : {0, 6, 2, 7}) {
    Cursor.jumpToPointer(Bytes + I);
    EXPECT_EQ(I, Cursor.getCurrentByteNo());
  }
}

TEST(BitstreamReaderTest, setArtificialByteLimit) {
  uint8_t Bytes[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                     0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f};
  BitstreamReader Reader(std::begin(Bytes), std::end(Bytes));
  SimpleBitstreamCursor Cursor(Reader);

  Cursor.setArtificialByteLimit(8);
  EXPECT_EQ(8u, Cursor.getSizeIfKnown());
  while (!Cursor.AtEndOfStream())
    (void)Cursor.Read(1);

  EXPECT_EQ(8u, Cursor.getCurrentByteNo());
}

TEST(BitstreamReaderTest, setArtificialByteLimitNotWordBoundary) {
  uint8_t Bytes[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                     0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f};
  BitstreamReader Reader(std::begin(Bytes), std::end(Bytes));
  SimpleBitstreamCursor Cursor(Reader);

  Cursor.setArtificialByteLimit(5);
  EXPECT_EQ(8u, Cursor.getSizeIfKnown());
  while (!Cursor.AtEndOfStream())
    (void)Cursor.Read(1);

  EXPECT_EQ(8u, Cursor.getCurrentByteNo());
}

TEST(BitstreamReaderTest, setArtificialByteLimitPastTheEnd) {
  uint8_t Bytes[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                     0x08, 0x09, 0x0a, 0x0b};
  BitstreamReader Reader(std::begin(Bytes), std::end(Bytes));
  SimpleBitstreamCursor Cursor(Reader);

  // The size of the memory object isn't known yet.  Set it too high and
  // confirm that we don't read too far.
  Cursor.setArtificialByteLimit(24);
  EXPECT_EQ(24u, Cursor.getSizeIfKnown());
  while (!Cursor.AtEndOfStream())
    (void)Cursor.Read(1);

  EXPECT_EQ(12u, Cursor.getCurrentByteNo());
  EXPECT_EQ(12u, Cursor.getSizeIfKnown());
}

TEST(BitstreamReaderTest, setArtificialByteLimitPastTheEndKnown) {
  uint8_t Bytes[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                     0x08, 0x09, 0x0a, 0x0b};
  BitstreamReader Reader(std::begin(Bytes), std::end(Bytes));
  SimpleBitstreamCursor Cursor(Reader);

  // Save the size of the memory object in the cursor.
  while (!Cursor.AtEndOfStream())
    (void)Cursor.Read(1);
  EXPECT_EQ(12u, Cursor.getCurrentByteNo());
  EXPECT_EQ(12u, Cursor.getSizeIfKnown());

  Cursor.setArtificialByteLimit(20);
  EXPECT_TRUE(Cursor.AtEndOfStream());
  EXPECT_EQ(12u, Cursor.getSizeIfKnown());
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

      BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
      Abbrev->Add(BitCodeAbbrevOp(RecordID));
      Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));
      AbbrevID = Stream.EmitAbbrev(Abbrev);
      unsigned Record[] = {RecordID};
      Stream.EmitRecordWithBlob(AbbrevID, makeArrayRef(Record), BlobIn);

      Stream.ExitBlock();
    }

    // Stream the buffer into the reader.
    BitstreamReader R(llvm::make_unique<StreamingMemoryObject>(
        llvm::make_unique<BufferStreamer>(
            StringRef(Buffer.begin(), Buffer.size()))));
    BitstreamCursor Stream(R);

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

} // end anonymous namespace
