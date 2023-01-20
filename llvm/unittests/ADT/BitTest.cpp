//===- llvm/unittests/ADT/BitTest.cpp - <bit> tests ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/bit.h"
#include "gtest/gtest.h"
#include <cstdint>
#include <cstdlib>

using namespace llvm;

namespace {

TEST(BitTest, BitCast) {
  static const uint8_t kValueU8 = 0x80;
  EXPECT_TRUE(llvm::bit_cast<int8_t>(kValueU8) < 0);

  static const uint16_t kValueU16 = 0x8000;
  EXPECT_TRUE(llvm::bit_cast<int16_t>(kValueU16) < 0);

  static const float kValueF32 = 5632.34f;
  EXPECT_FLOAT_EQ(kValueF32,
                  llvm::bit_cast<float>(llvm::bit_cast<uint32_t>(kValueF32)));

  static const double kValueF64 = 87987234.983498;
  EXPECT_DOUBLE_EQ(kValueF64,
                   llvm::bit_cast<double>(llvm::bit_cast<uint64_t>(kValueF64)));
}

TEST(BitTest, HasSingleBit) {
  EXPECT_FALSE(llvm::has_single_bit(0U));
  EXPECT_FALSE(llvm::has_single_bit(0ULL));

  EXPECT_FALSE(llvm::has_single_bit(~0U));
  EXPECT_FALSE(llvm::has_single_bit(~0ULL));

  EXPECT_TRUE(llvm::has_single_bit(1U));
  EXPECT_TRUE(llvm::has_single_bit(1ULL));

  static const int8_t kValueS8 = -128;
  EXPECT_TRUE(llvm::has_single_bit(static_cast<uint8_t>(kValueS8)));

  static const int16_t kValueS16 = -32768;
  EXPECT_TRUE(llvm::has_single_bit(static_cast<uint16_t>(kValueS16)));
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

TEST(BitTest, CountrZero) {
  uint8_t Z8 = 0;
  uint16_t Z16 = 0;
  uint32_t Z32 = 0;
  uint64_t Z64 = 0;
  EXPECT_EQ(8, llvm::countr_zero(Z8));
  EXPECT_EQ(16, llvm::countr_zero(Z16));
  EXPECT_EQ(32, llvm::countr_zero(Z32));
  EXPECT_EQ(64, llvm::countr_zero(Z64));

  uint8_t NZ8 = 42;
  uint16_t NZ16 = 42;
  uint32_t NZ32 = 42;
  uint64_t NZ64 = 42;
  EXPECT_EQ(1, llvm::countr_zero(NZ8));
  EXPECT_EQ(1, llvm::countr_zero(NZ16));
  EXPECT_EQ(1, llvm::countr_zero(NZ32));
  EXPECT_EQ(1, llvm::countr_zero(NZ64));
}

TEST(BitTest, CountlOne) {
  for (int i = 30; i >= 0; --i) {
    // Start with all ones and unset some bit.
    EXPECT_EQ(31 - i, llvm::countl_one(0xFFFFFFFF ^ (1 << i)));
  }
  for (int i = 62; i >= 0; --i) {
    // Start with all ones and unset some bit.
    EXPECT_EQ(63 - i, llvm::countl_one(0xFFFFFFFFFFFFFFFFULL ^ (1LL << i)));
  }
  for (int i = 30; i >= 0; --i) {
    // Start with all ones and unset some bit.
    EXPECT_EQ(31 - i, llvm::countl_one(0xFFFFFFFF ^ (1 << i)));
  }
}

TEST(BitTest, CountrOne) {
  uint8_t AllOnes8 = ~(uint8_t)0;
  uint16_t AllOnes16 = ~(uint16_t)0;
  uint32_t AllOnes32 = ~(uint32_t)0;
  uint64_t AllOnes64 = ~(uint64_t)0;
  EXPECT_EQ(8, llvm::countr_one(AllOnes8));
  EXPECT_EQ(16, llvm::countr_one(AllOnes16));
  EXPECT_EQ(32, llvm::countr_one(AllOnes32));
  EXPECT_EQ(64, llvm::countr_one(AllOnes64));

  uint8_t X8 = 6;
  uint16_t X16 = 6;
  uint32_t X32 = 6;
  uint64_t X64 = 6;
  EXPECT_EQ(0, llvm::countr_one(X8));
  EXPECT_EQ(0, llvm::countr_one(X16));
  EXPECT_EQ(0, llvm::countr_one(X32));
  EXPECT_EQ(0, llvm::countr_one(X64));

  uint8_t Y8 = 23;
  uint16_t Y16 = 23;
  uint32_t Y32 = 23;
  uint64_t Y64 = 23;
  EXPECT_EQ(3, llvm::countr_one(Y8));
  EXPECT_EQ(3, llvm::countr_one(Y16));
  EXPECT_EQ(3, llvm::countr_one(Y32));
  EXPECT_EQ(3, llvm::countr_one(Y64));
}

TEST(BitTest, PopCount) {
  EXPECT_EQ(0, llvm::popcount(0U));
  EXPECT_EQ(0, llvm::popcount(0ULL));

  EXPECT_EQ(32, llvm::popcount(~0U));
  EXPECT_EQ(64, llvm::popcount(~0ULL));

  for (int I = 0; I != 32; ++I)
    EXPECT_EQ(1, llvm::popcount(1U << I));
}

} // anonymous namespace
