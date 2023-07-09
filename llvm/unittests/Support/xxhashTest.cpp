//===- llvm/unittest/Support/xxhashTest.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/xxhash.h"
#include "gtest/gtest.h"

using namespace llvm;

const uint32_t PRIME32_1 = 0x9E3779B1U;
const uint64_t PRIME64 = 0x9e3779b185ebca8d;

TEST(xxhashTest, Basic) {
  //EXPECT_EQ(0xef46db3751d8e999U, xxHash64(StringRef()));
  //EXPECT_EQ(0x33bf00a859c4ba3fU, xxHash64("foo"));
  //EXPECT_EQ(0x48a37c90ad27a659U, xxHash64("bar"));
  //EXPECT_EQ(0x69196c1b3af0bff9U,
  //          xxHash64("0123456789abcdefghijklmnopqrstuvwxyz"));
}

TEST(xxhashTest, xxh3) {
  constexpr int TEST_DATA_SIZE = 2243;
  uint8_t a[TEST_DATA_SIZE];
  uint64_t byte_gen = PRIME32_1;
  for (size_t i = 0; i < TEST_DATA_SIZE; ++i) {
    a[i] = (uint8_t)(byte_gen >> 56);
    byte_gen *= PRIME64;
  }

  // clang-format off
  EXPECT_EQ(0x776EDDFB6BFD9195ULL, xxh3_64bits(a, 0));    // empty string
  EXPECT_EQ(0xB936EBAE24CB01C5ULL, xxh3_64bits(a, 1));    //  1 -  3
  EXPECT_EQ(0x27B56A84CD2D7325ULL, xxh3_64bits(a, 6));    //  4 -  8
  EXPECT_EQ(0xA713DAF0DFBB77E7ULL, xxh3_64bits(a, 12));   //  9 - 16
  EXPECT_EQ(0xA3FE70BF9D3510EBULL, xxh3_64bits(a, 24));   // 17 - 32
  EXPECT_EQ(0x397DA259ECBA1F11ULL, xxh3_64bits(a, 48));   // 33 - 64
  EXPECT_EQ(0xBCDEFBBB2C47C90AULL, xxh3_64bits(a, 80));   // 65 - 96
  EXPECT_EQ(0xCD94217EE362EC3AULL, xxh3_64bits(a, 195));  // 129-240

  EXPECT_EQ(0x1B2AFF3B46C74648ULL, xxh3_64bits(a, 403));  // one block, last stripe is overlapping
  EXPECT_EQ(0x43E368661808A9E8ULL, xxh3_64bits(a, 512));  // one block, finishing at stripe boundary
  EXPECT_EQ(0xC7169244BBDA8BD4ULL, xxh3_64bits(a, 2048)); // 2 blocks, finishing at block boundary
  EXPECT_EQ(0x30FEB637E114C0C7ULL, xxh3_64bits(a, 2240)); // 3 blocks, finishing at stripe boundary
  EXPECT_EQ(0x62C631454648A193ULL, xxh3_64bits(a, 2243)); // 3 blocks, last stripe is overlapping
  // clang-format on
}
