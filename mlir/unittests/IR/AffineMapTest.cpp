//===- AffineMapTest.cpp - unit tests for affine map API ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "gtest/gtest.h"

using namespace mlir;

// Test AffineMap replace API for the zero result case.
TEST(AffineMapTest, inferMapFromAffineExprs) {
  MLIRContext ctx;
  OpBuilder b(&ctx);
  AffineMap map = b.getEmptyAffineMap();
  DenseMap<AffineExpr, AffineExpr> replacements;
  map.replace(replacements);
  EXPECT_EQ(map, map);
}

TEST(AffineMapTest, isProjectedPermutation) {
  MLIRContext ctx;
  OpBuilder b(&ctx);

  // 1. Empty map - a projected permutation.
  AffineMap map1 = b.getEmptyAffineMap();
  EXPECT_TRUE(map1.isProjectedPermutation());

  // 2. Contains a symbol - not a projected permutation.
  AffineMap map2 = AffineMap::get(0, 1, &ctx);
  EXPECT_FALSE(map2.isProjectedPermutation());

  // 3. The result map is {0} - since zero results are _allowed_, this _is_ a
  // projected permutation.
  auto zero = b.getAffineConstantExpr(0);
  AffineMap map3 = AffineMap::get(1, 0, {zero}, &ctx);
  EXPECT_TRUE(map3.isProjectedPermutation(/*allowZeroInResults=*/true));

  // 4. The result map is {0} - since zero results are _not allowed_, this _is
  // not_ a projected permutation.
  AffineMap map4 = AffineMap::get(1, 0, {zero}, &ctx);
  EXPECT_FALSE(map4.isProjectedPermutation(/*allowZeroInResults=*/false));

  // 5. The number of results > inputs, not a projected permutation.
  AffineMap map5 = AffineMap::get(1, 0, {zero, zero}, &ctx);
  EXPECT_FALSE(map5.isProjectedPermutation(/*allowZeroInResults=*/true));

  // 6. A constant result that's not a {0} - not a projected permutation.
  auto one = b.getAffineConstantExpr(1);
  AffineMap map6 = AffineMap::get(1, 0, {one}, &ctx);
  EXPECT_FALSE(map6.isProjectedPermutation(/*allowZeroInResults=*/true));

  // 7. Not a dim expression - not a projected permutation.
  auto d0 = b.getAffineDimExpr(0);
  auto d1 = b.getAffineDimExpr(1);

  auto sum = d0 + d1;
  AffineMap map7 = AffineMap::get(2, 0, {sum}, &ctx);
  EXPECT_FALSE(map7.isProjectedPermutation());
}
