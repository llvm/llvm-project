//===- BitstreamReaderTest.cpp - Tests for BitstreamReader ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bitcode/BitstreamReader.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

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
  while (!Cursor.AtEndOfStream())
    (void)Cursor.Read(1);

  EXPECT_EQ(8u, Cursor.getCurrentByteNo());
}

TEST(BitstreamReaderTest, setArtificialByteLimitNot4ByteBoundary) {
  uint8_t Bytes[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                     0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f};
  BitstreamReader Reader(std::begin(Bytes), std::end(Bytes));
  SimpleBitstreamCursor Cursor(Reader);

  Cursor.setArtificialByteLimit(5);
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
  Cursor.setArtificialByteLimit(20);
  while (!Cursor.AtEndOfStream())
    (void)Cursor.Read(1);

  EXPECT_EQ(12u, Cursor.getCurrentByteNo());
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

  Cursor.setArtificialByteLimit(20);
  EXPECT_TRUE(Cursor.AtEndOfStream());
}

} // end anonymous namespace
