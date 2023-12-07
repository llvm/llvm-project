//===- llvm/unittest/Support/formatted_raw_ostream_test.cpp ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(formatted_raw_ostreamTest, Test_Tell) {
  // Check offset when underlying stream has buffer contents.
  SmallString<128> A;
  raw_svector_ostream B(A);
  formatted_raw_ostream C(B);
  char tmp[100] = "";

  for (unsigned i = 0; i != 3; ++i) {
    C.write(tmp, 100);

    EXPECT_EQ(100*(i+1), (unsigned) C.tell());
  }
}

TEST(formatted_raw_ostreamTest, Test_LineColumn) {
  // Test tracking of line and column numbers in a stream.
  SmallString<128> A;
  raw_svector_ostream B(A);
  formatted_raw_ostream C(B);

  EXPECT_EQ(0U, C.getLine());
  EXPECT_EQ(0U, C.getColumn());

  C << "a";
  EXPECT_EQ(0U, C.getLine());
  EXPECT_EQ(1U, C.getColumn());

  C << "bcdef";
  EXPECT_EQ(0U, C.getLine());
  EXPECT_EQ(6U, C.getColumn());

  // '\n' increments line number, sets column to zero.
  C << "\n";
  EXPECT_EQ(1U, C.getLine());
  EXPECT_EQ(0U, C.getColumn());

  // '\r sets column to zero without changing line number
  C << "foo\r";
  EXPECT_EQ(1U, C.getLine());
  EXPECT_EQ(0U, C.getColumn());

  // '\t' advances column to the next multiple of 8.
  // FIXME: If the column number is already a multiple of 8 this will do
  // nothing, is this behaviour correct?
  C << "1\t";
  EXPECT_EQ(8U, C.getColumn());
  C << "\t";
  EXPECT_EQ(8U, C.getColumn());
  C << "1234567\t";
  EXPECT_EQ(16U, C.getColumn());
  EXPECT_EQ(1U, C.getLine());
}

TEST(formatted_raw_ostreamTest, Test_Flush) {
  // Flushing the buffer causes the characters in the buffer to be scanned
  // before the buffer is emptied, so line and column numbers will still be
  // tracked properly.
  SmallString<128> A;
  raw_svector_ostream B(A);
  B.SetBufferSize(32);
  formatted_raw_ostream C(B);

  C << "\nabc";
  EXPECT_EQ(4U, C.GetNumBytesInBuffer());
  C.flush();
  EXPECT_EQ(1U, C.getLine());
  EXPECT_EQ(3U, C.getColumn());
  EXPECT_EQ(0U, C.GetNumBytesInBuffer());
}

TEST(formatted_raw_ostreamTest, Test_UTF8) {
  SmallString<128> A;
  raw_svector_ostream B(A);
  B.SetBufferSize(32);
  formatted_raw_ostream C(B);

  // U+00A0 Non-breaking space: encoded as two bytes, but only one column wide.
  C << "\xc2\xa0";
  EXPECT_EQ(0U, C.getLine());
  EXPECT_EQ(1U, C.getColumn());
  EXPECT_EQ(2U, C.GetNumBytesInBuffer());

  // U+2468 CIRCLED DIGIT NINE: encoded as three bytes, but only one column
  // wide.
  C << reinterpret_cast<const char *>(u8"\u2468");
  EXPECT_EQ(0U, C.getLine());
  EXPECT_EQ(2U, C.getColumn());
  EXPECT_EQ(5U, C.GetNumBytesInBuffer());

  // U+00010000 LINEAR B SYLLABLE B008 A: encoded as four bytes, but only one
  // column wide.
  C << reinterpret_cast<const char *>(u8"\U00010000");
  EXPECT_EQ(0U, C.getLine());
  EXPECT_EQ(3U, C.getColumn());
  EXPECT_EQ(9U, C.GetNumBytesInBuffer());

  // U+55B5, CJK character, encodes as three bytes, takes up two columns.
  C << reinterpret_cast<const char *>(u8"\u55b5");
  EXPECT_EQ(0U, C.getLine());
  EXPECT_EQ(5U, C.getColumn());
  EXPECT_EQ(12U, C.GetNumBytesInBuffer());

  // U+200B, zero-width space, encoded as three bytes but has no effect on the
  // column or line number.
  C << reinterpret_cast<const char *>(u8"\u200b");
  EXPECT_EQ(0U, C.getLine());
  EXPECT_EQ(5U, C.getColumn());
  EXPECT_EQ(15U, C.GetNumBytesInBuffer());
}

TEST(formatted_raw_ostreamTest, Test_UTF8Buffered) {
  SmallString<128> A;
  raw_svector_ostream B(A);
  B.SetBufferSize(4);
  formatted_raw_ostream C(B);

  // U+2468 encodes as three bytes, so will cause the buffer to be flushed after
  // the first byte (4 byte buffer, 3 bytes already written). We need to save
  // the first part of the UTF-8 encoding until after the buffer is cleared and
  // the remaining two bytes are written, at which point we can check the
  // display width. In this case the display width is 1, so we end at column 4,
  // with 6 bytes written into total, 2 of which are in the buffer.
  C << reinterpret_cast<const char *>(u8"123\u2468");
  EXPECT_EQ(0U, C.getLine());
  EXPECT_EQ(4U, C.getColumn());
  EXPECT_EQ(2U, C.GetNumBytesInBuffer());
  C.flush();
  EXPECT_EQ(6U, A.size());

  // Same as above, but with a CJK character which displays as two columns.
  C << reinterpret_cast<const char *>(u8"123\u55b5");
  EXPECT_EQ(0U, C.getLine());
  EXPECT_EQ(9U, C.getColumn());
  EXPECT_EQ(2U, C.GetNumBytesInBuffer());
  C.flush();
  EXPECT_EQ(12U, A.size());
}

TEST(formatted_raw_ostreamTest, Test_UTF8TinyBuffer) {
  SmallString<128> A;
  raw_svector_ostream B(A);
  B.SetBufferSize(1);
  formatted_raw_ostream C(B);

  // The stream has a one-byte buffer, so it gets flushed multiple times while
  // printing a single Unicode character.
  C << "\xe2\x91\xa8";
  EXPECT_EQ(0U, C.getLine());
  EXPECT_EQ(1U, C.getColumn());
  EXPECT_EQ(0U, C.GetNumBytesInBuffer());
  C.flush();
  EXPECT_EQ(3U, A.size());
}
}
