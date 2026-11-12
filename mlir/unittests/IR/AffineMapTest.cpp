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

  // 1. Empty map
  AffineMap map1 = b.getEmptyAffineMap();
  EXPECT_TRUE(map1.isProjectedPermutation());

  // 2. Map with a symbol
  AffineMap map2 = AffineMap::get(0, 1, &ctx);
  EXPECT_FALSE(map2.isProjectedPermutation());

  // 3. The result map is {0} and zero results are _allowed_.
  auto zero = b.getAffineConstantExpr(0);
  AffineMap map3 = AffineMap::get(1, 0, {zero}, &ctx);
  EXPECT_TRUE(map3.isProjectedPermutation(/*allowZeroInResults=*/true));

  // 4. The result map is {0} and zero results are _not allowed_
  AffineMap map4 = AffineMap::get(1, 0, {zero}, &ctx);
  EXPECT_FALSE(map4.isProjectedPermutation(/*allowZeroInResults=*/false));

  // 5. The number of results > inputs
  AffineMap map5 = AffineMap::get(1, 0, {zero, zero}, &ctx);
  EXPECT_FALSE(map5.isProjectedPermutation(/*allowZeroInResults=*/true));

  // 6. A constant result that's not a {0}
  auto one = b.getAffineConstantExpr(1);
  AffineMap map6 = AffineMap::get(1, 0, {one}, &ctx);
  EXPECT_FALSE(map6.isProjectedPermutation(/*allowZeroInResults=*/true));

  // 7. Not a dim expression
  auto d0 = b.getAffineDimExpr(0);
  auto d1 = b.getAffineDimExpr(1);

  auto sum = d0 + d1;
  AffineMap map7 = AffineMap::get(2, 0, {sum}, &ctx);
  EXPECT_FALSE(map7.isProjectedPermutation());

  // 8. (d0, d1, d2, d3, d4, d5) ->(d5, d3, d0, d1, d2, d4)
  auto d2 = b.getAffineDimExpr(2);
  auto d3 = b.getAffineDimExpr(3);
  auto d4 = b.getAffineDimExpr(4);
  auto d5 = b.getAffineDimExpr(5);
  AffineMap map8 = AffineMap::get(6, 0, {d5, d3, d0, d1, d2, d4}, &ctx);
  EXPECT_TRUE(map8.isProjectedPermutation());

  // 9. (d0, d1, d2, d3, d4, d5) ->(d5, d3, d0 + d1, d2, d4)
  AffineMap map9 = AffineMap::get(6, 0, {d5, d3, sum, d2, d4}, &ctx);
  EXPECT_FALSE(map9.isProjectedPermutation());

  // 10. (d0, d1, d2, d3, d4, d5) ->(d5, d3, d2, d4)
  AffineMap map10 = AffineMap::get(6, 0, {d5, d3, d2, d4}, &ctx);
  EXPECT_TRUE(map10.isProjectedPermutation());
}

TEST(AffineMapTest, getInversePermutation) {
  MLIRContext ctx;
  OpBuilder b(&ctx);

  // 0. Empty map
  AffineMap map0 = AffineMap::get(0, 0, {}, &ctx);
  AffineMap inverseMap0 = inversePermutation(map0);
  EXPECT_TRUE(inverseMap0.isEmpty());

  auto d0 = b.getAffineDimExpr(0);
  auto d1 = b.getAffineDimExpr(1);
  auto d2 = b.getAffineDimExpr(2);

  // 1.   (d0, d1, d2) -> (d1, d1, d0, d2, d1, d2, d1, d0)
  AffineMap map1 = AffineMap::get(3, 0, {d1, d1, d0, d2, d1, d2, d1, d0}, &ctx);
  //      (d0, d1, d2, d3, d4, d5, d6, d7) -> (d2, d0, d3)
  AffineMap inverseMap1 = inversePermutation(map1);
  auto resultsInv1 = inverseMap1.getResults();
  EXPECT_EQ(resultsInv1.size(), 3UL);

  // Expect (d2, d0, d3)
  SmallVector<unsigned> expected = {2, 0, 3};
  for (auto [idx, res] : llvm::enumerate(resultsInv1)) {
    AffineDimExpr expr = llvm::dyn_cast<AffineDimExpr>(res);
    EXPECT_TRUE(expr && expr.getPosition() == expected[idx]);
  }

  // 2.   (d0, d1, d2) -> (d1, d0 + d1, d0, d2, d1, d2, d1, d0)
  auto sum = d0 + d1;
  AffineMap map2 =
      AffineMap::get(3, 0, {d1, sum, d0, d2, d1, d2, d1, d0}, &ctx);
  //      (d0, d1, d2, d3, d4, d5, d6, d7) -> (d2, d0, d3)
  AffineMap inverseMap2 = inversePermutation(map2);
  auto resultsInv2 = inverseMap2.getResults();
  EXPECT_EQ(resultsInv2.size(), 3UL);

  // Expect (d2, d0, d3)
  expected = {2, 0, 3};
  for (auto [idx, res] : llvm::enumerate(resultsInv2)) {
    AffineDimExpr expr = llvm::dyn_cast<AffineDimExpr>(res);
    EXPECT_TRUE(expr && expr.getPosition() == expected[idx]);
  }
}
