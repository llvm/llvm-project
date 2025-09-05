//===- bit-test.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for orc-rt's bit.h APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/bit.h"
#include "gtest/gtest.h"

#include <cstdint>

#if defined(BYTE_ORDER) && defined(BIG_ENDIAN) && BYTE_ORDER == BIG_ENDIAN
#define IS_BIG_ENDIAN
#elif defined(BYTE_ORDER) && defined(LITTLE_ENDIAN) &&                         \
    BYTE_ORDER == LITTLE_ENDIAN
#define IS_LITTLE_ENDIAN
#endif

using namespace orc_rt;

TEST(BitTest, endian) {
#if defined(IS_BIG_ENDIAN)
  EXPECT_EQ(endian::native, endian::big);
#elif defined(IS_LITTLE_ENDIAN)
  EXPECT_EQ(endian::native, endian::little);
#else
  ADD_FAILURE() << "BYTE_ORDER is neither BIG_ENDIAN nor LITTLE_ENDIAN.";
#endif
}

TEST(BitTest, byte_swap_32) {
  unsigned char Seq[] = {0x01, 0x23, 0x45, 0x67};
  uint32_t X = 0;
  memcpy(&X, Seq, sizeof(X));
  auto Y = byteswap(X);

  static_assert(sizeof(Seq) == sizeof(X),
                "sizeof(char[4]) != sizeof(uint32_t) ?");
  static_assert(std::is_same_v<decltype(X), decltype(Y)>,
                "byte_swap return type doesn't match input");

#if defined(IS_BIG_ENDIAN)
  EXPECT_EQ(X, uint32_t(0x01234567));
  EXPECT_EQ(Y, uint32_t(0x67452301));
#elif defined(IS_LITTLE_ENDIAN)
  EXPECT_EQ(X, uint32_t(0x67452301));
  EXPECT_EQ(Y, uint32_t(0x01234567));
#else
  ADD_FAILURE() << "BYTE_ORDER is neither BIG_ENDIAN nor LITTLE_ENDIAN.";
#endif
}

TEST(BitTest, byte_swap_64) {
  unsigned char Seq[] = {0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF};
  uint64_t X = 0;
  memcpy(&X, Seq, sizeof(X));
  auto Y = byteswap(X);

  static_assert(sizeof(Seq) == sizeof(X),
                "sizeof(char[8]) != sizeof(uint64_t) ?");
  static_assert(std::is_same_v<decltype(X), decltype(Y)>,
                "byte_swap return type doesn't match input");

#if defined(IS_BIG_ENDIAN)
  EXPECT_EQ(X, uint64_t(0x0123456789ABCDEF));
  EXPECT_EQ(Y, uint64_t(0xEFCDAB8967452301));
#elif defined(IS_LITTLE_ENDIAN)
  EXPECT_EQ(X, uint64_t(0xEFCDAB8967452301));
  EXPECT_EQ(Y, uint64_t(0x0123456789ABCDEF));
#else
  ADD_FAILURE() << "BYTE_ORDER is neither BIG_ENDIAN nor LITTLE_ENDIAN.";
#endif
}
