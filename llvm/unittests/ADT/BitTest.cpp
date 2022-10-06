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

TEST(BitTest, PopCount) {
  EXPECT_EQ(0, llvm::popcount(0U));
  EXPECT_EQ(0, llvm::popcount(0ULL));

  EXPECT_EQ(32, llvm::popcount(~0U));
  EXPECT_EQ(64, llvm::popcount(~0ULL));

  for (int I = 0; I != 32; ++I)
    EXPECT_EQ(1, llvm::popcount(1U << I));
}

} // anonymous namespace
