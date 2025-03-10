//===-- StatuslineTest.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Statusline.h"
#include "gtest/gtest.h"

using namespace lldb_private;

class TestStatusline : public Statusline {
public:
  using Statusline::TrimAndPad;
};

TEST(StatuslineTest, TestTrimAndPad) {
  // Test basic ASCII.
  EXPECT_EQ("     ", TestStatusline::TrimAndPad("", 5));
  EXPECT_EQ("foo  ", TestStatusline::TrimAndPad("foo", 5));
  EXPECT_EQ("fooba", TestStatusline::TrimAndPad("fooba", 5));
  EXPECT_EQ("fooba", TestStatusline::TrimAndPad("foobar", 5));

  // Simple test that ANSI escape codes don't contribute to the visible width.
  EXPECT_EQ("\x1B[30m     ", TestStatusline::TrimAndPad("\x1B[30m", 5));
  EXPECT_EQ("\x1B[30mfoo  ", TestStatusline::TrimAndPad("\x1B[30mfoo", 5));
  EXPECT_EQ("\x1B[30mfooba", TestStatusline::TrimAndPad("\x1B[30mfooba", 5));
  EXPECT_EQ("\x1B[30mfooba", TestStatusline::TrimAndPad("\x1B[30mfoobar", 5));

  // Test that we include as many escape codes as we can.
  EXPECT_EQ("fooba\x1B[30m", TestStatusline::TrimAndPad("fooba\x1B[30m", 5));
  EXPECT_EQ("fooba\x1B[30m\x1B[34m",
            TestStatusline::TrimAndPad("fooba\x1B[30m\x1B[34m", 5));
  EXPECT_EQ("fooba\x1B[30m\x1B[34m",
            TestStatusline::TrimAndPad("fooba\x1B[30m\x1B[34mr", 5));

  // Test Unicode.
  EXPECT_EQ("❤️    ", TestStatusline::TrimAndPad("❤️", 5));
  EXPECT_EQ("    ❤️", TestStatusline::TrimAndPad("    ❤️", 5));
}
