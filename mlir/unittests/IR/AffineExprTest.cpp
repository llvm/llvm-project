//===- AffineExprTest.cpp - unit tests for affine expression API ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <limits>

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "gtest/gtest.h"

using namespace mlir;

static std::string toString(AffineExpr expr) {
  std::string s;
  llvm::raw_string_ostream ss(s);
  ss << expr;
  return s;
}

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

TEST(AffineExprTest, constantFolding) {
  MLIRContext ctx;
  OpBuilder b(&ctx);
  auto cn1 = b.getAffineConstantExpr(-1);
  auto c0 = b.getAffineConstantExpr(0);
  auto c1 = b.getAffineConstantExpr(1);
  auto c2 = b.getAffineConstantExpr(2);
  auto c3 = b.getAffineConstantExpr(3);
  auto c6 = b.getAffineConstantExpr(6);
  auto cmax = b.getAffineConstantExpr(std::numeric_limits<int64_t>::max());
  auto cmin = b.getAffineConstantExpr(std::numeric_limits<int64_t>::min());

  ASSERT_EQ(getAffineBinaryOpExpr(AffineExprKind::Add, c1, c2), c3);
  ASSERT_EQ(getAffineBinaryOpExpr(AffineExprKind::Mul, c2, c3), c6);
  ASSERT_EQ(getAffineBinaryOpExpr(AffineExprKind::FloorDiv, c3, c2), c1);
  ASSERT_EQ(getAffineBinaryOpExpr(AffineExprKind::CeilDiv, c3, c2), c2);

  // Test division by zero:
  auto c3ceildivc0 = getAffineBinaryOpExpr(AffineExprKind::CeilDiv, c3, c0);
  ASSERT_EQ(c3ceildivc0.getKind(), AffineExprKind::CeilDiv);

  auto c3floordivc0 = getAffineBinaryOpExpr(AffineExprKind::FloorDiv, c3, c0);
  ASSERT_EQ(c3floordivc0.getKind(), AffineExprKind::FloorDiv);

  auto c3modc0 = getAffineBinaryOpExpr(AffineExprKind::Mod, c3, c0);
  ASSERT_EQ(c3modc0.getKind(), AffineExprKind::Mod);

  // Test overflow:
  auto cmaxplusc1 = getAffineBinaryOpExpr(AffineExprKind::Add, cmax, c1);
  ASSERT_EQ(cmaxplusc1.getKind(), AffineExprKind::Add);

  auto cmaxtimesc2 = getAffineBinaryOpExpr(AffineExprKind::Mul, cmax, c2);
  ASSERT_EQ(cmaxtimesc2.getKind(), AffineExprKind::Mul);

  auto cminceildivcn1 =
      getAffineBinaryOpExpr(AffineExprKind::CeilDiv, cmin, cn1);
  ASSERT_EQ(cminceildivcn1.getKind(), AffineExprKind::CeilDiv);

  auto cminfloordivcn1 =
      getAffineBinaryOpExpr(AffineExprKind::FloorDiv, cmin, cn1);
  ASSERT_EQ(cminfloordivcn1.getKind(), AffineExprKind::FloorDiv);
}

TEST(AffineExprTest, divisionSimplification) {
  MLIRContext ctx;
  OpBuilder b(&ctx);
  auto cn6 = b.getAffineConstantExpr(-6);
  auto c6 = b.getAffineConstantExpr(6);
  auto d0 = b.getAffineDimExpr(0);
  auto d1 = b.getAffineDimExpr(1);

  ASSERT_EQ(c6.floorDiv(-1), cn6);
  ASSERT_EQ((d0 * 6).floorDiv(2), d0 * 3);
  ASSERT_EQ((d0 * 6).floorDiv(4).getKind(), AffineExprKind::FloorDiv);
  ASSERT_EQ((d0 * 6).floorDiv(-2), d0 * -3);
  ASSERT_EQ((d0 * 6 + d1).floorDiv(2), d0 * 3 + d1.floorDiv(2));
  ASSERT_EQ((d0 * 6 + d1).floorDiv(-2), d0 * -3 + d1.floorDiv(-2));
  ASSERT_EQ((d0 * 6 + d1).floorDiv(4).getKind(), AffineExprKind::FloorDiv);

  ASSERT_EQ(c6.ceilDiv(-1), cn6);
  ASSERT_EQ((d0 * 6).ceilDiv(2), d0 * 3);
  ASSERT_EQ((d0 * 6).ceilDiv(4).getKind(), AffineExprKind::CeilDiv);
  ASSERT_EQ((d0 * 6).ceilDiv(-2), d0 * -3);
}

TEST(AffineExprTest, modSimplificationRegression) {
  MLIRContext ctx;
  OpBuilder b(&ctx);
  auto d0 = b.getAffineDimExpr(0);
  auto sum = d0 + d0.floorDiv(3).floorDiv(-3);
  ASSERT_EQ(sum.getKind(), AffineExprKind::Add);
}

TEST(AffineExprTest, divisorOfNegativeFloorDiv) {
  MLIRContext ctx;
  OpBuilder b(&ctx);
  ASSERT_EQ(b.getAffineDimExpr(0).floorDiv(-1).getLargestKnownDivisor(), 1);
}

TEST(AffineExprTest, d0PlusD0FloorDivNeg2) {
  // Regression test for a bug where this was rewritten to d0 mod -2. We do not
  // support a negative RHS for mod in LowerAffinePass.
  MLIRContext ctx;
  OpBuilder b(&ctx);
  auto d0 = b.getAffineDimExpr(0);
  auto sum = d0 + d0.floorDiv(-2) * 2;
  ASSERT_EQ(toString(sum), "d0 + (d0 floordiv -2) * 2");
}

TEST(AffineExprTest, simpleAffineExprFlattenerRegression) {

  // Regression test for a bug where mod simplification was not handled
  // properly when `lhs % rhs` was happened to have the property that `lhs
  // floordiv rhs = lhs`.
  MLIRContext ctx;
  OpBuilder b(&ctx);

  auto d0 = b.getAffineDimExpr(0);

  // Manually replace variables by constants to avoid constant folding.
  AffineExpr expr = (d0 - (d0 + 2)).floorDiv(8) % 8;
  AffineExpr result = mlir::simplifyAffineExpr(expr, 1, 0);

  ASSERT_TRUE(isa<AffineConstantExpr>(result));
  ASSERT_EQ(cast<AffineConstantExpr>(result).getValue(), 7);
}
