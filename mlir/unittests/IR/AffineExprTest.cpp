//===- AffineExprTest.cpp - unit tests for affine expression API ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "gtest/gtest.h"

using namespace mlir;

// Test creating AffineExprs using the overloaded binary operators.
TEST(AffineExprTest, constructFromBinaryOperators) {
  MLIRContext ctx;
  OpBuilder b(&ctx);

  auto d0 = b.getAffineDimExpr(0);
  auto d1 = b.getAffineDimExpr(1);

  auto sum = d0 + d1;
  auto difference = d0 - d1;
  auto product = d0 * d1;
  auto remainder = d0 % d1;

  ASSERT_EQ(sum.getKind(), AffineExprKind::Add);
  ASSERT_EQ(difference.getKind(), AffineExprKind::Add);
  ASSERT_EQ(product.getKind(), AffineExprKind::Mul);
  ASSERT_EQ(remainder.getKind(), AffineExprKind::Mod);
}
