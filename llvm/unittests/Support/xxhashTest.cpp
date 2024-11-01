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

TEST(xxhashTest, Basic) {
  EXPECT_EQ(0xef46db3751d8e999U, xxHash64(StringRef()));
  EXPECT_EQ(0x33bf00a859c4ba3fU, xxHash64("foo"));
  EXPECT_EQ(0x48a37c90ad27a659U, xxHash64("bar"));
  EXPECT_EQ(0x69196c1b3af0bff9U,
            xxHash64("0123456789abcdefghijklmnopqrstuvwxyz"));
}

TEST(xxhashTest, xxh3) {
  constexpr size_t size = 2243;
  uint8_t a[size];
  uint64_t x = 1;
  for (size_t i = 0; i < size; ++i) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    a[i] = uint8_t(x);
  }

#define F(len, expected)                                                       \
  EXPECT_EQ(uint64_t(expected), xxh3_64bits(ArrayRef(a, size_t(len))))
  F(0, 0x2d06800538d394c2);
  F(1, 0xd0d496e05c553485);
  F(2, 0x84d625edb7055eac);
  F(3, 0x6ea2d59aca5c3778);
  F(4, 0xbf65290914e80242);
  F(5, 0xc01fd099ad4fc8e4);
  F(6, 0x9e3ea8187399caa5);
  F(7, 0x9da8b60540644f5a);
  F(8, 0xabc1413da6cd0209);
  F(9, 0x8bc89400bfed51f6);
  F(16, 0x7e46916754d7c9b8);
  F(17, 0xed4be912ba5f836d);
  F(32, 0xf59b59b58c304fd1);
  F(33, 0x9013fb74ca603e0c);
  F(64, 0xfa5271fcce0db1c3);
  F(65, 0x79c42431727f1012);
  F(96, 0x591ee0ddf9c9ccd1);
  F(97, 0x8ffc6a3111fe19da);
  F(128, 0x06a146ee9a2da378);
  F(129, 0xbc7138129bf065da);
  F(403, 0xcefeb3ffa532ad8c);
  F(512, 0xcdfa6b6268e3650f);
  F(513, 0x4bb5d42742f9765f);
  F(2048, 0x330ce110cbb79eae);
  F(2049, 0x3ba6afa0249fef9a);
  F(2240, 0xd61d4d2a94e926a8);
  F(2243, 0x0979f786a24edde7);
#undef F
}
