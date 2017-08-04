//===- BitstreamWriterTest.cpp - Tests for BitstreamWriter ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bitcode/BitstreamWriter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
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
  EXPECT_EQ(StringRef(Expected), Buffer);
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

} // end namespace
