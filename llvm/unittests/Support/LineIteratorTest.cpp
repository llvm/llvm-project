//===- LineIterator.cpp - Unit tests --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::sys;

namespace {

TEST(LineIteratorTest, Basic) {
  std::unique_ptr<MemoryBuffer> Buffer = MemoryBuffer::getMemBuffer("line 1\n"
                                                                    "line 2\n"
                                                                    "line 3");

  line_iterator I = line_iterator(*Buffer), E;

  EXPECT_FALSE(I.is_at_eof());
  EXPECT_NE(E, I);

  EXPECT_EQ("line 1", *I);
  EXPECT_EQ(1, I.line_number());
  ++I;
  EXPECT_EQ("line 2", *I);
  EXPECT_EQ(2, I.line_number());
  ++I;
  EXPECT_EQ("line 3", *I);
  EXPECT_EQ(3, I.line_number());
  ++I;

  EXPECT_TRUE(I.is_at_eof());
  EXPECT_EQ(E, I);
}

TEST(LineIteratorTest, CommentAndBlankSkipping) {
  std::unique_ptr<MemoryBuffer> Buffer(
      MemoryBuffer::getMemBuffer("line 1\n"
                                 "line 2\n"
                                 "# Comment 1\n"
                                 "\n"
                                 "line 5\n"
                                 "\n"
                                 "# Comment 2"));

  line_iterator I = line_iterator(*Buffer, true, '#'), E;

  EXPECT_FALSE(I.is_at_eof());
  EXPECT_NE(E, I);

  EXPECT_EQ("line 1", *I);
  EXPECT_EQ(1, I.line_number());
  ++I;
  EXPECT_EQ("line 2", *I);
  EXPECT_EQ(2, I.line_number());
  ++I;
  EXPECT_EQ("line 5", *I);
  EXPECT_EQ(5, I.line_number());
  ++I;

  EXPECT_TRUE(I.is_at_eof());
  EXPECT_EQ(E, I);
}

TEST(LineIteratorTest, CommentSkippingKeepBlanks) {
  std::unique_ptr<MemoryBuffer> Buffer(
      MemoryBuffer::getMemBuffer("line 1\n"
                                 "line 2\n"
                                 "# Comment 1\n"
                                 "# Comment 2\n"
                                 "\n"
                                 "line 6\n"
                                 "\n"
                                 "# Comment 3"));

  line_iterator I = line_iterator(*Buffer, false, '#'), E;

  EXPECT_FALSE(I.is_at_eof());
  EXPECT_NE(E, I);

  EXPECT_EQ("line 1", *I);
  EXPECT_EQ(1, I.line_number());
  ++I;
  EXPECT_EQ("line 2", *I);
  EXPECT_EQ(2, I.line_number());
  ++I;
  EXPECT_EQ("", *I);
  EXPECT_EQ(5, I.line_number());
  ++I;
  EXPECT_EQ("line 6", *I);
  EXPECT_EQ(6, I.line_number());
  ++I;
  EXPECT_EQ("", *I);
  EXPECT_EQ(7, I.line_number());
  ++I;

  EXPECT_TRUE(I.is_at_eof());
  EXPECT_EQ(E, I);
}


TEST(LineIteratorTest, BlankSkipping) {
  std::unique_ptr<MemoryBuffer> Buffer = MemoryBuffer::getMemBuffer("\n\n\n"
                                                                    "line 1\n"
                                                                    "\n\n\n"
                                                                    "line 2\n"
                                                                    "\n\n\n");

  line_iterator I = line_iterator(*Buffer), E;

  EXPECT_FALSE(I.is_at_eof());
  EXPECT_NE(E, I);

  EXPECT_EQ("line 1", *I);
  EXPECT_EQ(4, I.line_number());
  ++I;
  EXPECT_EQ("line 2", *I);
  EXPECT_EQ(8, I.line_number());
  ++I;

  EXPECT_TRUE(I.is_at_eof());
  EXPECT_EQ(E, I);
}

TEST(LineIteratorTest, BlankKeeping) {
  std::unique_ptr<MemoryBuffer> Buffer = MemoryBuffer::getMemBuffer("\n\n"
                                                                    "line 3\n"
                                                                    "\n"
                                                                    "line 5\n"
                                                                    "\n\n");
  line_iterator I = line_iterator(*Buffer, false), E;

  EXPECT_FALSE(I.is_at_eof());
  EXPECT_NE(E, I);

  EXPECT_EQ("", *I);
  EXPECT_EQ(1, I.line_number());
  ++I;
  EXPECT_EQ("", *I);
  EXPECT_EQ(2, I.line_number());
  ++I;
  EXPECT_EQ("line 3", *I);
  EXPECT_EQ(3, I.line_number());
  ++I;
  EXPECT_EQ("", *I);
  EXPECT_EQ(4, I.line_number());
  ++I;
  EXPECT_EQ("line 5", *I);
  EXPECT_EQ(5, I.line_number());
  ++I;
  EXPECT_EQ("", *I);
  EXPECT_EQ(6, I.line_number());
  ++I;
  EXPECT_EQ("", *I);
  EXPECT_EQ(7, I.line_number());
  ++I;

  EXPECT_TRUE(I.is_at_eof());
  EXPECT_EQ(E, I);
}

TEST(LineIteratorTest, EmptyBuffers) {
  std::unique_ptr<MemoryBuffer> Buffer = MemoryBuffer::getMemBuffer("");
  EXPECT_TRUE(line_iterator(*Buffer).is_at_eof());
  EXPECT_EQ(line_iterator(), line_iterator(*Buffer));
  EXPECT_TRUE(line_iterator(*Buffer, false).is_at_eof());
  EXPECT_EQ(line_iterator(), line_iterator(*Buffer, false));

  Buffer = MemoryBuffer::getMemBuffer("\n\n\n");
  EXPECT_TRUE(line_iterator(*Buffer).is_at_eof());
  EXPECT_EQ(line_iterator(), line_iterator(*Buffer));

  Buffer = MemoryBuffer::getMemBuffer("# foo\n"
                                      "\n"
                                      "# bar");
  EXPECT_TRUE(line_iterator(*Buffer, true, '#').is_at_eof());
  EXPECT_EQ(line_iterator(), line_iterator(*Buffer, true, '#'));

  Buffer = MemoryBuffer::getMemBuffer("\n"
                                      "# baz\n"
                                      "\n");
  EXPECT_TRUE(line_iterator(*Buffer, true, '#').is_at_eof());
  EXPECT_EQ(line_iterator(), line_iterator(*Buffer, true, '#'));
}

} // anonymous namespace
