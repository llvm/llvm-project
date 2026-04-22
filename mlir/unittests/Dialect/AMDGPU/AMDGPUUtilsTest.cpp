//===- AMDGPUUtilsTest.cpp - Unit tests for AMDGPU dialect utils ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "gtest/gtest.h"

namespace mlir::amdgpu {
namespace {

TEST(ChipsetTest, Parsing) {
  FailureOr<Chipset> chipset = Chipset::parse("gfx90a");
  ASSERT_TRUE(succeeded(chipset));
  EXPECT_EQ(chipset->majorVersion, 9u);
  EXPECT_EQ(chipset->minorVersion, 0u);
  EXPECT_EQ(chipset->steppingVersion, 0xau);

  chipset = Chipset::parse("gfx942");
  ASSERT_TRUE(succeeded(chipset));
  EXPECT_EQ(chipset->majorVersion, 9u);
  EXPECT_EQ(chipset->minorVersion, 4u);
  EXPECT_EQ(chipset->steppingVersion, 2u);

  chipset = Chipset::parse("gfx1103");
  ASSERT_TRUE(succeeded(chipset));
  EXPECT_EQ(chipset->majorVersion, 11u);
  EXPECT_EQ(chipset->minorVersion, 0u);
  EXPECT_EQ(chipset->steppingVersion, 3u);
}

TEST(ChipsetTest, ParsingInvalid) {
  EXPECT_TRUE(failed(Chipset::parse("navi33")));
  EXPECT_TRUE(failed(Chipset::parse("rdna2")));
  EXPECT_TRUE(failed(Chipset::parse("sm_80")));
  EXPECT_TRUE(failed(Chipset::parse("GFX942")));
  EXPECT_TRUE(failed(Chipset::parse("Gfx942")));
  EXPECT_TRUE(failed(Chipset::parse("gfx9")));
  EXPECT_TRUE(failed(Chipset::parse("gfx_942")));
  EXPECT_TRUE(failed(Chipset::parse("gfx942_")));
  EXPECT_TRUE(failed(Chipset::parse("gfxmeow")));
  EXPECT_TRUE(failed(Chipset::parse("gfx1fff")));
}

TEST(ChipsetTest, Comparison) {
  EXPECT_EQ(Chipset(9, 4, 2), Chipset(9, 4, 2));
  EXPECT_NE(Chipset(9, 0, 0), Chipset(10, 0, 0));

  EXPECT_LT(Chipset(9, 0, 0), Chipset(10, 0, 0));
  EXPECT_LT(Chipset(9, 0, 0), Chipset(9, 4, 2));
  EXPECT_FALSE(Chipset(9, 4, 2) < Chipset(9, 4, 2));

  EXPECT_GT(Chipset(9, 0, 0xa), Chipset(9, 0, 8));
  EXPECT_GE(Chipset(9, 0, 0xa), Chipset(9, 0, 0xa));
  EXPECT_FALSE(Chipset(9, 0, 0xa) >= Chipset(9, 4, 2));
}

TEST(ChipsetTest, HasDot1Insts) {
  // gfx9: enabled from gfx906 onward.
  EXPECT_FALSE(hasDot1Insts(Chipset(9, 0, 0)));
  EXPECT_TRUE(hasDot1Insts(Chipset(9, 0, 6)));
  EXPECT_TRUE(hasDot1Insts(Chipset(9, 0, 8)));
  EXPECT_TRUE(hasDot1Insts(Chipset(9, 0, 0xa)));
  EXPECT_TRUE(hasDot1Insts(Chipset(9, 4, 2)));
  EXPECT_TRUE(hasDot1Insts(Chipset(9, 5, 0)));

  // gfx10: only gfx10.1.1, gfx10.1.2, and gfx10.3+ enable Dot1.
  EXPECT_FALSE(hasDot1Insts(Chipset(10, 1, 0))); // gfx1010
  EXPECT_TRUE(hasDot1Insts(Chipset(10, 1, 1)));  // gfx1011
  EXPECT_TRUE(hasDot1Insts(Chipset(10, 1, 2)));  // gfx1012
  EXPECT_FALSE(hasDot1Insts(Chipset(10, 1, 3))); // gfx1013
  EXPECT_TRUE(hasDot1Insts(Chipset(10, 3, 0)));  // gfx1030

  // Not on gfx11+/gfx12+/gfx13+.
  EXPECT_FALSE(hasDot1Insts(Chipset(11, 0, 0)));
  EXPECT_FALSE(hasDot1Insts(Chipset(12, 0, 0)));
  EXPECT_FALSE(hasDot1Insts(Chipset(12, 5, 0)));
  EXPECT_FALSE(hasDot1Insts(Chipset(13, 0, 0)));
}

TEST(ChipsetTest, HasDot7Insts) {
  // Same as Dot1 plus all of gfx11+/gfx12+/gfx13+.
  EXPECT_FALSE(hasDot7Insts(Chipset(9, 0, 0)));
  EXPECT_TRUE(hasDot7Insts(Chipset(9, 0, 6)));
  EXPECT_FALSE(hasDot7Insts(Chipset(10, 1, 0)));
  EXPECT_TRUE(hasDot7Insts(Chipset(11, 0, 0)));
  EXPECT_TRUE(hasDot7Insts(Chipset(12, 0, 0)));
  EXPECT_TRUE(hasDot7Insts(Chipset(12, 5, 0))); // gfx1250 still has Dot7.
  EXPECT_TRUE(hasDot7Insts(Chipset(13, 0, 0)));
}

TEST(ChipsetTest, HasDot8Insts) {
  // gfx11+ only.
  EXPECT_FALSE(hasDot8Insts(Chipset(9, 4, 2)));
  EXPECT_FALSE(hasDot8Insts(Chipset(10, 3, 0)));
  EXPECT_TRUE(hasDot8Insts(Chipset(11, 0, 0)));
  EXPECT_TRUE(hasDot8Insts(Chipset(12, 5, 0))); // gfx1250 has Dot8.
}

TEST(ChipsetTest, HasDot9Insts) {
  // gfx11.x and gfx12.0 only.
  EXPECT_FALSE(hasDot9Insts(Chipset(9, 4, 2)));
  EXPECT_FALSE(hasDot9Insts(Chipset(10, 3, 0)));
  EXPECT_TRUE(hasDot9Insts(Chipset(11, 0, 0)));
  EXPECT_TRUE(hasDot9Insts(Chipset(11, 7, 0)));
  EXPECT_TRUE(hasDot9Insts(Chipset(12, 0, 0)));
  EXPECT_FALSE(hasDot9Insts(Chipset(12, 5, 0))); // gfx1250 lacks Dot9.
  EXPECT_FALSE(hasDot9Insts(Chipset(13, 0, 0)));
}

TEST(ChipsetTest, HasDot10Insts) {
  // Dot1's set plus gfx11.x and gfx12.0 (excludes gfx12.5+/gfx13+).
  EXPECT_TRUE(hasDot10Insts(Chipset(9, 0, 6)));
  EXPECT_FALSE(hasDot10Insts(Chipset(10, 1, 0)));
  EXPECT_TRUE(hasDot10Insts(Chipset(10, 3, 0)));
  EXPECT_TRUE(hasDot10Insts(Chipset(11, 0, 0)));
  EXPECT_TRUE(hasDot10Insts(Chipset(12, 0, 0)));
  EXPECT_FALSE(hasDot10Insts(Chipset(12, 5, 0)));
  EXPECT_FALSE(hasDot10Insts(Chipset(13, 0, 0)));
}

TEST(ChipsetTest, HasDot11Insts) {
  // Only gfx11.7 and gfx12.0.
  EXPECT_FALSE(hasDot11Insts(Chipset(9, 5, 0)));
  EXPECT_FALSE(hasDot11Insts(Chipset(11, 0, 0)));
  EXPECT_FALSE(hasDot11Insts(Chipset(11, 5, 0)));
  EXPECT_TRUE(hasDot11Insts(Chipset(11, 7, 0)));
  EXPECT_TRUE(hasDot11Insts(Chipset(12, 0, 0)));
  EXPECT_FALSE(hasDot11Insts(Chipset(12, 5, 0)));
}

TEST(ChipsetTest, HasDot12Insts) {
  // gfx9.5.0, gfx11.x, and gfx12.0.
  EXPECT_FALSE(hasDot12Insts(Chipset(9, 0, 6)));
  EXPECT_FALSE(hasDot12Insts(Chipset(9, 4, 2)));
  EXPECT_TRUE(hasDot12Insts(Chipset(9, 5, 0)));
  EXPECT_TRUE(hasDot12Insts(Chipset(11, 0, 0)));
  EXPECT_TRUE(hasDot12Insts(Chipset(12, 0, 0)));
  EXPECT_FALSE(hasDot12Insts(Chipset(12, 5, 0)));
}

} // namespace
} // namespace mlir::amdgpu
