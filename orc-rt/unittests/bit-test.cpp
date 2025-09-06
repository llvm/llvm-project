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

TEST(BitTest, CountlZero) {
  uint8_t Z8 = 0;
  uint16_t Z16 = 0;
  uint32_t Z32 = 0;
  uint64_t Z64 = 0;
  EXPECT_EQ(8, llvm::countl_zero(Z8));
  EXPECT_EQ(16, llvm::countl_zero(Z16));
  EXPECT_EQ(32, llvm::countl_zero(Z32));
  EXPECT_EQ(64, llvm::countl_zero(Z64));

  uint8_t NZ8 = 42;
  uint16_t NZ16 = 42;
  uint32_t NZ32 = 42;
  uint64_t NZ64 = 42;
  EXPECT_EQ(2, llvm::countl_zero(NZ8));
  EXPECT_EQ(10, llvm::countl_zero(NZ16));
  EXPECT_EQ(26, llvm::countl_zero(NZ32));
  EXPECT_EQ(58, llvm::countl_zero(NZ64));

  EXPECT_EQ(8, llvm::countl_zero(0x00F000FFu));
  EXPECT_EQ(8, llvm::countl_zero(0x00F12345u));
  for (unsigned i = 0; i <= 30; ++i) {
    EXPECT_EQ(int(31 - i), llvm::countl_zero(1u << i));
  }

  EXPECT_EQ(8, llvm::countl_zero(0x00F1234500F12345ULL));
  EXPECT_EQ(1, llvm::countl_zero(1ULL << 62));
  for (unsigned i = 0; i <= 62; ++i) {
    EXPECT_EQ(int(63 - i), llvm::countl_zero(1ULL << i));
  }
}

TEST(BitTest, BitWidth) {
  EXPECT_EQ(0, llvm::bit_width(uint8_t(0)));
  EXPECT_EQ(0, llvm::bit_width(uint16_t(0)));
  EXPECT_EQ(0, llvm::bit_width(uint32_t(0)));
  EXPECT_EQ(0, llvm::bit_width(uint64_t(0)));

  EXPECT_EQ(1, llvm::bit_width(uint8_t(1)));
  EXPECT_EQ(1, llvm::bit_width(uint16_t(1)));
  EXPECT_EQ(1, llvm::bit_width(uint32_t(1)));
  EXPECT_EQ(1, llvm::bit_width(uint64_t(1)));

  EXPECT_EQ(2, llvm::bit_width(uint8_t(2)));
  EXPECT_EQ(2, llvm::bit_width(uint16_t(2)));
  EXPECT_EQ(2, llvm::bit_width(uint32_t(2)));
  EXPECT_EQ(2, llvm::bit_width(uint64_t(2)));

  EXPECT_EQ(2, llvm::bit_width(uint8_t(3)));
  EXPECT_EQ(2, llvm::bit_width(uint16_t(3)));
  EXPECT_EQ(2, llvm::bit_width(uint32_t(3)));
  EXPECT_EQ(2, llvm::bit_width(uint64_t(3)));

  EXPECT_EQ(3, llvm::bit_width(uint8_t(4)));
  EXPECT_EQ(3, llvm::bit_width(uint16_t(4)));
  EXPECT_EQ(3, llvm::bit_width(uint32_t(4)));
  EXPECT_EQ(3, llvm::bit_width(uint64_t(4)));

  EXPECT_EQ(7, llvm::bit_width(uint8_t(0x7f)));
  EXPECT_EQ(15, llvm::bit_width(uint16_t(0x7fff)));
  EXPECT_EQ(31, llvm::bit_width(uint32_t(0x7fffffffu)));
  EXPECT_EQ(63, llvm::bit_width(uint64_t(0x7fffffffffffffffull)));

  EXPECT_EQ(8, llvm::bit_width(uint8_t(0x80)));
  EXPECT_EQ(16, llvm::bit_width(uint16_t(0x8000)));
  EXPECT_EQ(32, llvm::bit_width(uint32_t(0x80000000u)));
  EXPECT_EQ(64, llvm::bit_width(uint64_t(0x8000000000000000ull)));

  EXPECT_EQ(8, llvm::bit_width(uint8_t(0xff)));
  EXPECT_EQ(16, llvm::bit_width(uint16_t(0xffff)));
  EXPECT_EQ(32, llvm::bit_width(uint32_t(0xffffffffu)));
  EXPECT_EQ(64, llvm::bit_width(uint64_t(0xffffffffffffffffull)));
}
