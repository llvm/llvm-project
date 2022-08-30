//===-- UUIDTest.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Utility/UUID.h"

using namespace lldb_private;

TEST(UUIDTest, RelationalOperators) {
  UUID empty;
  UUID a16 = UUID("1234567890123456", 16);
  UUID b16 = UUID("1234567890123457", 16);
  UUID a20 = UUID("12345678901234567890", 20);
  UUID b20 = UUID("12345678900987654321", 20);

  EXPECT_EQ(empty, empty);
  EXPECT_EQ(a16, a16);
  EXPECT_EQ(a20, a20);

  EXPECT_NE(a16, b16);
  EXPECT_NE(a20, b20);
  EXPECT_NE(a16, a20);
  EXPECT_NE(empty, a16);

  EXPECT_LT(empty, a16);
  EXPECT_LT(a16, a20);
  EXPECT_LT(a16, b16);
  EXPECT_GT(a20, b20);
}

TEST(UUIDTest, Validity) {
  UUID empty;
  std::vector<uint8_t> zeroes(20, 0);
  UUID a16 = UUID(zeroes.data(), 16);
  UUID a20 = UUID(zeroes.data(), 20);
  UUID from_str;
  from_str.SetFromStringRef("00000000-0000-0000-0000-000000000000");

  EXPECT_FALSE(empty);
  EXPECT_FALSE(a16);
  EXPECT_FALSE(a20);
  EXPECT_FALSE(from_str);
}

TEST(UUIDTest, SetFromStringRef) {
  UUID u;
  EXPECT_TRUE(u.SetFromStringRef("404142434445464748494a4b4c4d4e4f"));
  EXPECT_EQ(UUID("@ABCDEFGHIJKLMNO", 16), u);

  EXPECT_TRUE(u.SetFromStringRef("40-41-42-43-4445464748494a4b4c4d4e4f"));
  EXPECT_EQ(UUID("@ABCDEFGHIJKLMNO", 16), u);

  EXPECT_TRUE(
      u.SetFromStringRef("40-41-42-43-4445464748494a4b4c4d4e4f-50515253"));
  EXPECT_EQ(UUID("@ABCDEFGHIJKLMNOPQRS", 20), u);

  EXPECT_TRUE(u.SetFromStringRef("40-41-42-43-4445464748494a4b4c4d4e4f"));

  EXPECT_FALSE(u.SetFromStringRef("40xxxxx"));
  EXPECT_FALSE(u.SetFromStringRef(""));
  EXPECT_EQ(UUID("@ABCDEFGHIJKLMNO", 16), u)
      << "uuid was changed by failed parse calls";

  EXPECT_TRUE(u.SetFromStringRef("404142434445464748494a4b4c4d4e4f-50515253"));
  EXPECT_EQ(UUID("@ABCDEFGHIJKLMNOPQRS", 20), u);

  EXPECT_TRUE(u.SetFromStringRef("40414243"));
  EXPECT_EQ(UUID("@ABCD", 4), u);

  EXPECT_FALSE(u.SetFromStringRef("4"));
}

TEST(UUIDTest, StringConverion) {
  EXPECT_EQ("40414243", UUID("@ABC", 4).GetAsString());
  EXPECT_EQ("40414243-4445-4647", UUID("@ABCDEFG", 8).GetAsString());
  EXPECT_EQ("40414243-4445-4647-4849-4A4B",
            UUID("@ABCDEFGHIJK", 12).GetAsString());
  EXPECT_EQ("40414243-4445-4647-4849-4A4B4C4D4E4F",
            UUID("@ABCDEFGHIJKLMNO", 16).GetAsString());
  EXPECT_EQ("40414243-4445-4647-4849-4A4B4C4D4E4F-50515253",
            UUID("@ABCDEFGHIJKLMNOPQRS", 20).GetAsString());
}
