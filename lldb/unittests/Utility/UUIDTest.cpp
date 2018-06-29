//===-- UUIDTest.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Utility/UUID.h"

using namespace lldb_private;

TEST(UUIDTest, RelationalOperators) {
  UUID empty;
  UUID a16 = UUID::fromData("1234567890123456", 16);
  UUID b16 = UUID::fromData("1234567890123457", 16);
  UUID a20 = UUID::fromData("12345678901234567890", 20);
  UUID b20 = UUID::fromData("12345678900987654321", 20);

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
  UUID a16 = UUID::fromData(zeroes.data(), 16);
  UUID a20 = UUID::fromData(zeroes.data(), 20);
  UUID a16_0 = UUID::fromOptionalData(zeroes.data(), 16);
  UUID a20_0 = UUID::fromOptionalData(zeroes.data(), 20);
  EXPECT_FALSE(empty);
  EXPECT_TRUE(a16);
  EXPECT_TRUE(a20);
  EXPECT_FALSE(a16_0);
  EXPECT_FALSE(a20_0);
}

TEST(UUIDTest, SetFromStringRef) {
  UUID u;
  EXPECT_EQ(32u, u.SetFromStringRef("404142434445464748494a4b4c4d4e4f"));
  EXPECT_EQ(UUID::fromData("@ABCDEFGHIJKLMNO", 16), u);

  EXPECT_EQ(36u, u.SetFromStringRef("40-41-42-43-4445464748494a4b4c4d4e4f"));
  EXPECT_EQ(UUID::fromData("@ABCDEFGHIJKLMNO", 16), u);

  EXPECT_EQ(45u, u.SetFromStringRef(
                     "40-41-42-43-4445464748494a4b4c4d4e4f-50515253", 20));
  EXPECT_EQ(UUID::fromData("@ABCDEFGHIJKLMNOPQRS", 20), u);

  EXPECT_EQ(0u, u.SetFromStringRef("40-41-42-43-4445464748494a4b4c4d4e4f", 20));
  EXPECT_EQ(0u, u.SetFromStringRef("40xxxxx"));
  EXPECT_EQ(0u, u.SetFromStringRef(""));
  EXPECT_EQ(UUID::fromData("@ABCDEFGHIJKLMNOPQRS", 20), u)
      << "uuid was changed by failed parse calls";

  EXPECT_EQ(
      32u, u.SetFromStringRef("404142434445464748494a4b4c4d4e4f-50515253", 16));
  EXPECT_EQ(UUID::fromData("@ABCDEFGHIJKLMNO", 16), u);
}

TEST(UUIDTest, StringConverion) {
  EXPECT_EQ("40414243", UUID::fromData("@ABC", 4).GetAsString());
  EXPECT_EQ("40414243-4445-4647", UUID::fromData("@ABCDEFG", 8).GetAsString());
  EXPECT_EQ("40414243-4445-4647-4849-4A4B",
            UUID::fromData("@ABCDEFGHIJK", 12).GetAsString());
  EXPECT_EQ("40414243-4445-4647-4849-4A4B4C4D4E4F",
            UUID::fromData("@ABCDEFGHIJKLMNO", 16).GetAsString());
  EXPECT_EQ("40414243-4445-4647-4849-4A4B4C4D4E4F-50515253",
            UUID::fromData("@ABCDEFGHIJKLMNOPQRS", 20).GetAsString());
}
