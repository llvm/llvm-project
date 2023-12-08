//===- MathExtrasTest.cpp - MathExtras Tests ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/MathExtras.h"
#include "gmock/gmock.h"

using namespace mlir;
using ::testing::Eq;

TEST(MathExtrasTest, CeilDivTest) {
  EXPECT_THAT(ceilDiv(14, 3), Eq(5));
  EXPECT_THAT(ceilDiv(14, -3), Eq(-4));
  EXPECT_THAT(ceilDiv(-14, -3), Eq(5));
  EXPECT_THAT(ceilDiv(-14, 3), Eq(-4));
  EXPECT_THAT(ceilDiv(0, 3), Eq(0));
  EXPECT_THAT(ceilDiv(0, -3), Eq(0));
}

TEST(MathExtrasTest, FloorDivTest) {
  EXPECT_THAT(floorDiv(14, 3), Eq(4));
  EXPECT_THAT(floorDiv(14, -3), Eq(-5));
  EXPECT_THAT(floorDiv(-14, -3), Eq(4));
  EXPECT_THAT(floorDiv(-14, 3), Eq(-5));
  EXPECT_THAT(floorDiv(0, 3), Eq(0));
  EXPECT_THAT(floorDiv(0, -3), Eq(0));
}
