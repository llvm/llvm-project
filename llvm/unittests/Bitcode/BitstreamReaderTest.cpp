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

} // end anonymous namespace
