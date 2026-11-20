//===- BitstreamReaderTest.cpp - Tests for BitstreamReader ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Bitstream/BitstreamReader.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Bitstream/BitstreamWriter.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(BitstreamReaderTest, AtEndOfStream) {
  uint8_t Bytes[4] = {
    0x00, 0x01, 0x02, 0x03
  };
  BitstreamCursor Cursor(Bytes);

  EXPECT_FALSE(Cursor.AtEndOfStream());
  Expected<SimpleBitstreamCursor::word_t> MaybeRead = Cursor.Read(8);
  EXPECT_TRUE((bool)MaybeRead);
  EXPECT_FALSE(Cursor.AtEndOfStream());
  MaybeRead = Cursor.Read(24);
  EXPECT_TRUE((bool)MaybeRead);
  EXPECT_TRUE(Cursor.AtEndOfStream());

  EXPECT_FALSE(Cursor.JumpToBit(0));
  EXPECT_FALSE(Cursor.AtEndOfStream());

  EXPECT_FALSE(Cursor.JumpToBit(32));
  EXPECT_TRUE(Cursor.AtEndOfStream());
}

TEST(BitstreamReaderTest, AtEndOfStreamJump) {
  uint8_t Bytes[4] = {
    0x00, 0x01, 0x02, 0x03
  };
  BitstreamCursor Cursor(Bytes);

  EXPECT_FALSE(Cursor.JumpToBit(32));
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
    Expected<SimpleBitstreamCursor::word_t> MaybeRead = Cursor.Read(1);
    EXPECT_TRUE((bool)MaybeRead);
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
      Stream.EmitRecordWithBlob(AbbrevID, ArrayRef(Record), BlobIn);

      Stream.ExitBlock();
    }

    // Stream the buffer into the reader.
    BitstreamCursor Stream(
        ArrayRef<uint8_t>((const uint8_t *)Buffer.begin(), Buffer.size()));

    // Header.  Included in test so that we can run llvm-bcanalyzer to debug
    // when there are problems.
    Expected<SimpleBitstreamCursor::word_t> MaybeRead = Stream.Read(32);
    ASSERT_TRUE((bool)MaybeRead);
    ASSERT_EQ(Magic, MaybeRead.get());

    // Block.
    Expected<BitstreamEntry> MaybeEntry =
        Stream.advance(BitstreamCursor::AF_DontAutoprocessAbbrevs);
    ASSERT_TRUE((bool)MaybeEntry);
    BitstreamEntry Entry = MaybeEntry.get();
    ASSERT_EQ(BitstreamEntry::SubBlock, Entry.Kind);
    ASSERT_EQ(BlockID, Entry.ID);
    ASSERT_FALSE(Stream.EnterSubBlock(BlockID));

    // Abbreviation.
    MaybeEntry = Stream.advance();
    ASSERT_TRUE((bool)MaybeEntry);
    Entry = MaybeEntry.get();
    ASSERT_EQ(BitstreamEntry::Record, Entry.Kind);
    ASSERT_EQ(AbbrevID, Entry.ID);

    // Record.
    StringRef BlobOut;
    SmallVector<uint64_t, 1> Record;
    Expected<unsigned> MaybeRecord =
        Stream.readRecord(Entry.ID, Record, &BlobOut);
    ASSERT_TRUE((bool)MaybeRecord);
    ASSERT_EQ(RecordID, MaybeRecord.get());
    EXPECT_TRUE(Record.empty());
    EXPECT_EQ(BlobIn, BlobOut);
  }
}

TEST(BitstreamReaderTest, readAbbrevWithManyOperands) {
  // Abbreviations of various sizes, crossing the inline capacities of the
  // operand lists.
  const unsigned BlockID = bitc::FIRST_APPLICATION_BLOCKID;
  const unsigned RecordID = 1;
  for (unsigned NumOps : {1, 2, 3, 4, 7, 8, 32, 33, 40}) {
    // The record code, then fixed-width, VBR and literal operands.
    SmallVector<BitCodeAbbrevOp> Ops;
    SmallVector<uint64_t> Values;
    Ops.push_back(BitCodeAbbrevOp(RecordID));
    for (unsigned I = 1; I != NumOps; ++I) {
      Values.push_back(I * 3);
      if (I % 3 == 0)
        Ops.push_back(BitCodeAbbrevOp(I * 3));
      else
        Ops.push_back(BitCodeAbbrevOp(
            I % 3 == 1 ? BitCodeAbbrevOp::Fixed : BitCodeAbbrevOp::VBR, 7));
    }

    SmallVector<char, 1> Buffer;
    unsigned AbbrevID;
    {
      BitstreamWriter Stream(Buffer);
      Stream.EnterSubblock(BlockID, 3);
      auto Abbrev = std::make_shared<BitCodeAbbrev>();
      for (const BitCodeAbbrevOp &Op : Ops)
        Abbrev->Add(Op);
      AbbrevID = Stream.EmitAbbrev(std::move(Abbrev));
      Stream.EmitRecord(RecordID, Values, AbbrevID);
      Stream.ExitBlock();
    }

    BitstreamCursor Stream(
        ArrayRef<uint8_t>((const uint8_t *)Buffer.begin(), Buffer.size()));
    Expected<BitstreamEntry> MaybeEntry =
        Stream.advance(BitstreamCursor::AF_DontAutoprocessAbbrevs);
    ASSERT_TRUE((bool)MaybeEntry);
    ASSERT_EQ(BitstreamEntry::SubBlock, MaybeEntry->Kind);
    ASSERT_FALSE(Stream.EnterSubBlock(BlockID));

    // The abbreviation is read automatically, then the record.
    MaybeEntry = Stream.advance();
    ASSERT_TRUE((bool)MaybeEntry);
    ASSERT_EQ(BitstreamEntry::Record, MaybeEntry->Kind);
    ASSERT_EQ(AbbrevID, MaybeEntry->ID);

    Expected<const BitCodeAbbrev *> MaybeAbbrev = Stream.getAbbrev(AbbrevID);
    ASSERT_TRUE((bool)MaybeAbbrev);
    const BitCodeAbbrev *Abbrev = *MaybeAbbrev;
    ASSERT_EQ(NumOps, Abbrev->getNumOperandInfos());
    for (unsigned I = 0; I != NumOps; ++I) {
      const BitCodeAbbrevOp &Want = Ops[I];
      const BitCodeAbbrevOp &Got = Abbrev->getOperandInfo(I);
      ASSERT_EQ(Want.isLiteral(), Got.isLiteral());
      if (Want.isLiteral()) {
        EXPECT_EQ(Want.getLiteralValue(), Got.getLiteralValue());
      } else {
        EXPECT_EQ(Want.getEncoding(), Got.getEncoding());
        EXPECT_EQ(Want.getEncodingData(), Got.getEncodingData());
      }
    }

    SmallVector<uint64_t> Record;
    Expected<unsigned> MaybeRecord = Stream.readRecord(AbbrevID, Record);
    ASSERT_TRUE((bool)MaybeRecord);
    EXPECT_EQ(RecordID, *MaybeRecord);
    EXPECT_EQ(Values, Record);
  }
}

TEST(BitstreamReaderTest, deeplyNestedBlocks) {
  // Nest more blocks than the cursor's block scope stores inline and check
  // that leaving each block restores the parent's abbreviation width and
  // abbreviations, also in a copy of the cursor.
  const unsigned Depth = 6;
  SmallVector<char, 1> Buffer;
  {
    BitstreamWriter Stream(Buffer);
    SmallVector<unsigned, Depth> AbbrevIDs;
    for (unsigned I = 0; I != Depth; ++I) {
      Stream.EnterSubblock(bitc::FIRST_APPLICATION_BLOCKID + I, 3 + I);
      auto Abbrev = std::make_shared<BitCodeAbbrev>();
      Abbrev->Add(BitCodeAbbrevOp(I + 1));
      Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 8));
      AbbrevIDs.push_back(Stream.EmitAbbrev(std::move(Abbrev)));
      unsigned Record[] = {I + 1, I};
      Stream.EmitRecordWithAbbrev(AbbrevIDs.back(), Record);
    }
    for (unsigned I = Depth; I-- != 0;) {
      unsigned Record[] = {I + 1, 100 + I};
      Stream.EmitRecordWithAbbrev(AbbrevIDs[I], Record);
      Stream.ExitBlock();
    }
  }

  BitstreamCursor Cursor(
      ArrayRef<uint8_t>((const uint8_t *)Buffer.begin(), Buffer.size()));
  auto ExpectRecord = [](BitstreamCursor &Cursor, unsigned Code,
                         uint64_t Value) {
    Expected<BitstreamEntry> MaybeEntry = Cursor.advance();
    ASSERT_TRUE((bool)MaybeEntry);
    ASSERT_EQ(BitstreamEntry::Record, MaybeEntry->Kind);
    SmallVector<uint64_t, 1> Record;
    Expected<unsigned> MaybeCode = Cursor.readRecord(MaybeEntry->ID, Record);
    ASSERT_TRUE((bool)MaybeCode);
    EXPECT_EQ(Code, *MaybeCode);
    EXPECT_EQ(ArrayRef<uint64_t>(Value), ArrayRef(Record));
  };
  for (unsigned I = 0; I != Depth; ++I) {
    Expected<BitstreamEntry> MaybeEntry = Cursor.advance();
    ASSERT_TRUE((bool)MaybeEntry);
    ASSERT_EQ(BitstreamEntry::SubBlock, MaybeEntry->Kind);
    ASSERT_EQ(bitc::FIRST_APPLICATION_BLOCKID + I, MaybeEntry->ID);
    ASSERT_FALSE(Cursor.EnterSubBlock(MaybeEntry->ID));
    EXPECT_EQ(3 + I, Cursor.getAbbrevIDWidth());
    ExpectRecord(Cursor, I + 1, I);
  }

  BitstreamCursor Copy = Cursor;
  for (BitstreamCursor *C : {&Cursor, &Copy}) {
    for (unsigned I = Depth; I-- != 0;) {
      EXPECT_EQ(3 + I, C->getAbbrevIDWidth());
      ExpectRecord(*C, I + 1, 100 + I);
      Expected<BitstreamEntry> MaybeEntry = C->advance();
      ASSERT_TRUE((bool)MaybeEntry);
      ASSERT_EQ(BitstreamEntry::EndBlock, MaybeEntry->Kind);
    }
    EXPECT_EQ(2u, C->getAbbrevIDWidth());
    EXPECT_TRUE(C->AtEndOfStream());
  }
}

TEST(BitstreamReaderTest, shortRead) {
  uint8_t Bytes[] = {8, 7, 6, 5, 4, 3, 2, 1};
  for (unsigned I = 1; I != 8; ++I) {
    SimpleBitstreamCursor Cursor(ArrayRef<uint8_t>(Bytes, I));
    Expected<SimpleBitstreamCursor::word_t> MaybeRead = Cursor.Read(8);
    ASSERT_TRUE((bool)MaybeRead);
    EXPECT_EQ(8ull, MaybeRead.get());
  }
}

static_assert(std::is_trivially_copyable_v<BitCodeAbbrevOp>,
              "trivially copyable");

} // end anonymous namespace
