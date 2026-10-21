//===- ChipInfoTest.cpp - AGX chip capability tests -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AGX/Utils/ChipInfo.h"
#include "gtest/gtest.h"

using namespace mlir::agx;

TEST(AGXChipInfoTest, ParsesEveryProcessor) {
  for (const char *chip :
       {"generic", "agx-g13g", "agx-g13s", "agx-g13c", "agx-g13d", "agx-g14g",
        "agx-g14s", "agx-g14c", "agx-g14d", "agx-g15g", "agx-g15c",
        "agx-g16g", "agx-g16c", "agx-g17p", "agx-g17g"})
    EXPECT_TRUE(succeeded(ChipInfo::get(chip))) << chip;
  for (const char *chip : {"", "agx", "agx-g12g", "agx-g18g", "gfx942",
                           "agx-g14", "agx-g14cc"})
    EXPECT_TRUE(failed(ChipInfo::get(chip))) << chip;
}

TEST(AGXChipInfoTest, Capabilities) {
  ChipInfo m2max = *ChipInfo::get("agx-g14c");
  EXPECT_EQ(m2max.generation, 14u);
  EXPECT_EQ(m2max.variant, 'c');
  EXPECT_EQ(m2max.simdWidth, 32u);
  EXPECT_EQ(m2max.threadgroupMemoryBytes, 65536u);
  EXPECT_FALSE(m2max.dynamicCaching);
  // Native forms stay off until their coverage is recorded.
  EXPECT_FALSE(m2max.laneShiftFMA);
  EXPECT_TRUE(m2max.blockBarrier);
  EXPECT_FALSE(ChipInfo::get("agx-g14c", "+agx-strict")->blockBarrier);
  EXPECT_FALSE(ChipInfo::get("agx-g13g")->blockBarrier);
  EXPECT_EQ(m2max.matrixUnit, MatrixUnit::Emulated);

  ChipInfo m5 = *ChipInfo::get("agx-g17g");
  EXPECT_TRUE(m5.dynamicCaching);
  EXPECT_EQ(m5.maxGPRs, 126u);
  EXPECT_EQ(ChipInfo::get("agx-g17p")->maxGPRs, 96u);
  EXPECT_EQ(ChipInfo::get("generic")->generation, 13u);
}
