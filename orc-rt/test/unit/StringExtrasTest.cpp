//===- StringExtrasTest.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for string utility APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/StringExtras.h"
#include "gtest/gtest.h"

using namespace orc_rt;

TEST(StringExtrasTest, JoinSingleString) { EXPECT_EQ(join({"x"}, "-"), "x"); }

TEST(StringExtrasTest, JoinMultipleString) {
  EXPECT_EQ(join({"a", "b", "c"}, "-"), "a-b-c");
}

TEST(StringExtrasTest, JoinMantainsEmptyString) {
  EXPECT_EQ(join({"a", "", "c"}, "-"), "a--c");
  EXPECT_EQ(join({"", "b"}, "-"), "-b");
  EXPECT_EQ(join({"a", ""}, "-"), "a-");
}

TEST(StringExtrasTest, JoinEmptySeparator) {
  EXPECT_EQ(join({"a", "b", "c"}, ""), "abc");
}

TEST(StringExtrasTest, JoinMultiSeparator) {
  EXPECT_EQ(join({"a", "b", "c"}, ", "), "a, b, c");
}
