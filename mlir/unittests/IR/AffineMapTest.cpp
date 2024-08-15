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
