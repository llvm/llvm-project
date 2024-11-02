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
  EXPECT_EQ(chipset->minorVersion, 0x0au);

  chipset = Chipset::parse("gfx940");
  ASSERT_TRUE(succeeded(chipset));
  EXPECT_EQ(chipset->majorVersion, 9u);
  EXPECT_EQ(chipset->minorVersion, 0x40u);

  chipset = Chipset::parse("gfx1103");
  ASSERT_TRUE(succeeded(chipset));
  EXPECT_EQ(chipset->majorVersion, 11u);
  EXPECT_EQ(chipset->minorVersion, 0x03u);
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
  EXPECT_EQ(Chipset(9, 0x40), Chipset(9, 0x40));
  EXPECT_NE(Chipset(9, 0x40), Chipset(9, 0x42));
  EXPECT_NE(Chipset(9, 0x00), Chipset(10, 0x00));

  EXPECT_LT(Chipset(9, 0x00), Chipset(10, 0x00));
  EXPECT_LT(Chipset(9, 0x0a), Chipset(9, 0x42));
  EXPECT_FALSE(Chipset(9, 0x42) < Chipset(9, 0x42));
  EXPECT_FALSE(Chipset(9, 0x42) < Chipset(9, 0x40));
}

} // namespace
} // namespace mlir::amdgpu
