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
  EXPECT_EQ(8, orc_rt::countl_zero(Z8));
  EXPECT_EQ(16, orc_rt::countl_zero(Z16));
  EXPECT_EQ(32, orc_rt::countl_zero(Z32));
  EXPECT_EQ(64, orc_rt::countl_zero(Z64));

  uint8_t NZ8 = 42;
  uint16_t NZ16 = 42;
  uint32_t NZ32 = 42;
  uint64_t NZ64 = 42;
  EXPECT_EQ(2, orc_rt::countl_zero(NZ8));
  EXPECT_EQ(10, orc_rt::countl_zero(NZ16));
  EXPECT_EQ(26, orc_rt::countl_zero(NZ32));
  EXPECT_EQ(58, orc_rt::countl_zero(NZ64));

  EXPECT_EQ(8, orc_rt::countl_zero(0x00F000FFu));
  EXPECT_EQ(8, orc_rt::countl_zero(0x00F12345u));
  for (unsigned i = 0; i <= 30; ++i) {
    EXPECT_EQ(int(31 - i), orc_rt::countl_zero(1u << i));
  }

  EXPECT_EQ(8, orc_rt::countl_zero(0x00F1234500F12345ULL));
  EXPECT_EQ(1, orc_rt::countl_zero(1ULL << 62));
  for (unsigned i = 0; i <= 62; ++i) {
    EXPECT_EQ(int(63 - i), orc_rt::countl_zero(1ULL << i));
  }
}

TEST(BitTest, BitWidth) {
  EXPECT_EQ(0, orc_rt::bit_width(uint8_t(0)));
  EXPECT_EQ(0, orc_rt::bit_width(uint16_t(0)));
  EXPECT_EQ(0, orc_rt::bit_width(uint32_t(0)));
  EXPECT_EQ(0, orc_rt::bit_width(uint64_t(0)));

  EXPECT_EQ(1, orc_rt::bit_width(uint8_t(1)));
  EXPECT_EQ(1, orc_rt::bit_width(uint16_t(1)));
  EXPECT_EQ(1, orc_rt::bit_width(uint32_t(1)));
  EXPECT_EQ(1, orc_rt::bit_width(uint64_t(1)));

  EXPECT_EQ(2, orc_rt::bit_width(uint8_t(2)));
  EXPECT_EQ(2, orc_rt::bit_width(uint16_t(2)));
  EXPECT_EQ(2, orc_rt::bit_width(uint32_t(2)));
  EXPECT_EQ(2, orc_rt::bit_width(uint64_t(2)));

  EXPECT_EQ(2, orc_rt::bit_width(uint8_t(3)));
  EXPECT_EQ(2, orc_rt::bit_width(uint16_t(3)));
  EXPECT_EQ(2, orc_rt::bit_width(uint32_t(3)));
  EXPECT_EQ(2, orc_rt::bit_width(uint64_t(3)));

  EXPECT_EQ(3, orc_rt::bit_width(uint8_t(4)));
  EXPECT_EQ(3, orc_rt::bit_width(uint16_t(4)));
  EXPECT_EQ(3, orc_rt::bit_width(uint32_t(4)));
  EXPECT_EQ(3, orc_rt::bit_width(uint64_t(4)));

  EXPECT_EQ(7, orc_rt::bit_width(uint8_t(0x7f)));
  EXPECT_EQ(15, orc_rt::bit_width(uint16_t(0x7fff)));
  EXPECT_EQ(31, orc_rt::bit_width(uint32_t(0x7fffffffu)));
  EXPECT_EQ(63, orc_rt::bit_width(uint64_t(0x7fffffffffffffffull)));

  EXPECT_EQ(8, orc_rt::bit_width(uint8_t(0x80)));
  EXPECT_EQ(16, orc_rt::bit_width(uint16_t(0x8000)));
  EXPECT_EQ(32, orc_rt::bit_width(uint32_t(0x80000000u)));
  EXPECT_EQ(64, orc_rt::bit_width(uint64_t(0x8000000000000000ull)));

  EXPECT_EQ(8, orc_rt::bit_width(uint8_t(0xff)));
  EXPECT_EQ(16, orc_rt::bit_width(uint16_t(0xffff)));
  EXPECT_EQ(32, orc_rt::bit_width(uint32_t(0xffffffffu)));
  EXPECT_EQ(64, orc_rt::bit_width(uint64_t(0xffffffffffffffffull)));
}

TEST(BitTest, HasSingleBit) {
  EXPECT_FALSE(orc_rt::has_single_bit(0U));
  EXPECT_FALSE(orc_rt::has_single_bit(0ULL));

  EXPECT_FALSE(orc_rt::has_single_bit(~0U));
  EXPECT_FALSE(orc_rt::has_single_bit(~0ULL));

  EXPECT_TRUE(orc_rt::has_single_bit(1U));
  EXPECT_TRUE(orc_rt::has_single_bit(1ULL));

  static const int8_t kValueS8 = -128;
  EXPECT_TRUE(orc_rt::has_single_bit(static_cast<uint8_t>(kValueS8)));

  static const int16_t kValueS16 = -32768;
  EXPECT_TRUE(orc_rt::has_single_bit(static_cast<uint16_t>(kValueS16)));
}

TEST(BitTest, Rotl) {
  EXPECT_EQ(0x53U, orc_rt::rotl<uint8_t>(0x53, 0));
  EXPECT_EQ(0x4dU, orc_rt::rotl<uint8_t>(0x53, 2));
  EXPECT_EQ(0xa6U, orc_rt::rotl<uint8_t>(0x53, 9));
  EXPECT_EQ(0x9aU, orc_rt::rotl<uint8_t>(0x53, -5));

  EXPECT_EQ(0xabcdU, orc_rt::rotl<uint16_t>(0xabcd, 0));
  EXPECT_EQ(0xf36aU, orc_rt::rotl<uint16_t>(0xabcd, 6));
  EXPECT_EQ(0xaf36U, orc_rt::rotl<uint16_t>(0xabcd, 18));
  EXPECT_EQ(0xf36aU, orc_rt::rotl<uint16_t>(0xabcd, -10));

  EXPECT_EQ(0xdeadbeefU, orc_rt::rotl<uint32_t>(0xdeadbeef, 0));
  EXPECT_EQ(0x7ddfbd5bU, orc_rt::rotl<uint32_t>(0xdeadbeef, 17));
  EXPECT_EQ(0x5b7ddfbdU, orc_rt::rotl<uint32_t>(0xdeadbeef, 41));
  EXPECT_EQ(0xb6fbbf7aU, orc_rt::rotl<uint32_t>(0xdeadbeef, -22));

  EXPECT_EQ(0x12345678deadbeefULL,
            orc_rt::rotl<uint64_t>(0x12345678deadbeefULL, 0));
  EXPECT_EQ(0xf56df77891a2b3c6ULL,
            orc_rt::rotl<uint64_t>(0x12345678deadbeefULL, 35));
  EXPECT_EQ(0x8d159e37ab6fbbc4ULL,
            orc_rt::rotl<uint64_t>(0x12345678deadbeefULL, 70));
  EXPECT_EQ(0xb7dde2468acf1bd5ULL,
            orc_rt::rotl<uint64_t>(0x12345678deadbeefULL, -19));
}

TEST(BitTest, Rotr) {
  EXPECT_EQ(0x53U, orc_rt::rotr<uint8_t>(0x53, 0));
  EXPECT_EQ(0xd4U, orc_rt::rotr<uint8_t>(0x53, 2));
  EXPECT_EQ(0xa9U, orc_rt::rotr<uint8_t>(0x53, 9));
  EXPECT_EQ(0x6aU, orc_rt::rotr<uint8_t>(0x53, -5));

  EXPECT_EQ(0xabcdU, orc_rt::rotr<uint16_t>(0xabcd, 0));
  EXPECT_EQ(0x36afU, orc_rt::rotr<uint16_t>(0xabcd, 6));
  EXPECT_EQ(0x6af3U, orc_rt::rotr<uint16_t>(0xabcd, 18));
  EXPECT_EQ(0x36afU, orc_rt::rotr<uint16_t>(0xabcd, -10));

  EXPECT_EQ(0xdeadbeefU, orc_rt::rotr<uint32_t>(0xdeadbeef, 0));
  EXPECT_EQ(0xdf77ef56U, orc_rt::rotr<uint32_t>(0xdeadbeef, 17));
  EXPECT_EQ(0x77ef56dfU, orc_rt::rotr<uint32_t>(0xdeadbeef, 41));
  EXPECT_EQ(0xbbf7ab6fU, orc_rt::rotr<uint32_t>(0xdeadbeef, -22));

  EXPECT_EQ(0x12345678deadbeefULL,
            orc_rt::rotr<uint64_t>(0x12345678deadbeefULL, 0));
  EXPECT_EQ(0x1bd5b7dde2468acfULL,
            orc_rt::rotr<uint64_t>(0x12345678deadbeefULL, 35));
  EXPECT_EQ(0xbc48d159e37ab6fbULL,
            orc_rt::rotr<uint64_t>(0x12345678deadbeefULL, 70));
  EXPECT_EQ(0xb3c6f56df77891a2ULL,
            orc_rt::rotr<uint64_t>(0x12345678deadbeefULL, -19));
}
