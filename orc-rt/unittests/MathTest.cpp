//===- MathTest.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for orc-rt's Math.h APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/Math.h"
#include "gtest/gtest.h"

using namespace orc_rt;

TEST(STLExtrasTest, isPowerOf2) {
  // Test [0..16]
  EXPECT_FALSE(isPowerOf2(0x00));
  EXPECT_TRUE(isPowerOf2(0x01));
  EXPECT_TRUE(isPowerOf2(0x02));
  EXPECT_FALSE(isPowerOf2(0x03));
  EXPECT_TRUE(isPowerOf2(0x04));
  EXPECT_FALSE(isPowerOf2(0x05));
  EXPECT_FALSE(isPowerOf2(0x06));
  EXPECT_FALSE(isPowerOf2(0x07));
  EXPECT_TRUE(isPowerOf2(0x08));
  EXPECT_FALSE(isPowerOf2(0x09));
  EXPECT_FALSE(isPowerOf2(0x0A));
  EXPECT_FALSE(isPowerOf2(0x0B));
  EXPECT_FALSE(isPowerOf2(0x0C));
  EXPECT_FALSE(isPowerOf2(0x0D));
  EXPECT_FALSE(isPowerOf2(0x0E));
  EXPECT_FALSE(isPowerOf2(0x0F));
  EXPECT_TRUE(isPowerOf2(0x10));

  // Test some higher powers of two and their adjacent values.
  EXPECT_FALSE(isPowerOf2(0x1F));
  EXPECT_TRUE(isPowerOf2(0x20));
  EXPECT_FALSE(isPowerOf2(0x21));

  EXPECT_FALSE(isPowerOf2(0x3F));
  EXPECT_TRUE(isPowerOf2(0x40));
  EXPECT_FALSE(isPowerOf2(0x41));

  EXPECT_FALSE(isPowerOf2(0x7F));
  EXPECT_TRUE(isPowerOf2(0x80));
  EXPECT_FALSE(isPowerOf2(0x81));

  // Test larger values.
  EXPECT_FALSE(isPowerOf2(0x3fffffff));
  EXPECT_TRUE(isPowerOf2(0x40000000));
  EXPECT_FALSE(isPowerOf2(0x40000001));

  // Test negatives.
  EXPECT_FALSE(isPowerOf2(-1));
}

TEST(STLExtrasTest, nextPowerOf2) {
  EXPECT_EQ(nextPowerOf2(0x00), (1 << 0));
  EXPECT_EQ(nextPowerOf2(0x01), (1 << 1));
  EXPECT_EQ(nextPowerOf2(0x02), (1 << 2));
  EXPECT_EQ(nextPowerOf2(0x03), (1 << 2));
  EXPECT_EQ(nextPowerOf2(0x04), (1 << 3));
  EXPECT_EQ(nextPowerOf2(0x05), (1 << 3));
  EXPECT_EQ(nextPowerOf2(0x06), (1 << 3));
  EXPECT_EQ(nextPowerOf2(0x07), (1 << 3));
  EXPECT_EQ(nextPowerOf2(0x08), (1 << 4));
  EXPECT_EQ(nextPowerOf2(0x09), (1 << 4));
  EXPECT_EQ(nextPowerOf2(0x0a), (1 << 4));
  EXPECT_EQ(nextPowerOf2(0x0b), (1 << 4));
  EXPECT_EQ(nextPowerOf2(0x0c), (1 << 4));
  EXPECT_EQ(nextPowerOf2(0x0d), (1 << 4));
  EXPECT_EQ(nextPowerOf2(0x0e), (1 << 4));
  EXPECT_EQ(nextPowerOf2(0x0f), (1 << 4));
  EXPECT_EQ(nextPowerOf2(0x10), (1 << 5));
}
