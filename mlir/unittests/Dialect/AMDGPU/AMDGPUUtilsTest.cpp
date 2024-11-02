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

  chipset = Chipset::parse("gfx940");
  ASSERT_TRUE(succeeded(chipset));
  EXPECT_EQ(chipset->majorVersion, 9u);
  EXPECT_EQ(chipset->minorVersion, 4u);
  EXPECT_EQ(chipset->steppingVersion, 0u);

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
  EXPECT_TRUE(failed(Chipset::parse("GFX940")));
  EXPECT_TRUE(failed(Chipset::parse("Gfx940")));
  EXPECT_TRUE(failed(Chipset::parse("gfx9")));
  EXPECT_TRUE(failed(Chipset::parse("gfx_940")));
  EXPECT_TRUE(failed(Chipset::parse("gfx940_")));
  EXPECT_TRUE(failed(Chipset::parse("gfxmeow")));
  EXPECT_TRUE(failed(Chipset::parse("gfx1fff")));
}

TEST(ChipsetTest, Comparison) {
  EXPECT_EQ(Chipset(9, 4, 0), Chipset(9, 4, 0));
  EXPECT_NE(Chipset(9, 4, 0), Chipset(9, 4, 2));
  EXPECT_NE(Chipset(9, 0, 0), Chipset(10, 0, 0));

  EXPECT_LT(Chipset(9, 0, 0), Chipset(10, 0, 0));
  EXPECT_LT(Chipset(9, 0, 0), Chipset(9, 4, 2));
  EXPECT_LE(Chipset(9, 4, 1), Chipset(9, 4, 1));
  EXPECT_FALSE(Chipset(9, 4, 2) < Chipset(9, 4, 2));
  EXPECT_FALSE(Chipset(9, 4, 2) < Chipset(9, 4, 0));

  EXPECT_GT(Chipset(9, 0, 0xa), Chipset(9, 0, 8));
  EXPECT_GE(Chipset(9, 0, 0xa), Chipset(9, 0, 0xa));
  EXPECT_FALSE(Chipset(9, 4, 1) >= Chipset(9, 4, 2));
  EXPECT_FALSE(Chipset(9, 0, 0xa) >= Chipset(9, 4, 0));
}

} // namespace
} // namespace mlir::amdgpu
