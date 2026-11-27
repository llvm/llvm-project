//===- FlatLinearValueConstraintsTest.cpp - Tests for FlatLinearConstraints -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/FlatLinearValueConstraints.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"

#include <gtest/gtest.h>
#include <optional>

using namespace mlir;

// A dimension without a constant upper bound has no constant bound on its
// size. The output maps must be left untouched in that case rather than being
// built from a null AffineExpr.
TEST(FlatLinearConstraintsTest, getConstantBoundOnDimSizeUnboundedAbove) {
  MLIRContext context;

  // x >= 0.
  FlatLinearConstraints cst(/*numDims=*/1);
  cst.addBound(presburger::BoundType::LB, 0, 0);

  AffineMap lb, ub;
  unsigned minLbPos = 0, minUbPos = 0;
  std::optional<int64_t> size = cst.getConstantBoundOnDimSize(
      &context, /*pos=*/0, &lb, &ub, &minLbPos, &minUbPos);
  EXPECT_FALSE(size.has_value());
  EXPECT_FALSE(lb);
  EXPECT_FALSE(ub);

  // Requesting only the upper bound must behave the same way.
  size =
      cst.getConstantBoundOnDimSize(&context, /*pos=*/0, /*lb=*/nullptr, &ub);
  EXPECT_FALSE(size.has_value());
  EXPECT_FALSE(ub);
}

// Likewise when the dimension is entirely unconstrained.
TEST(FlatLinearConstraintsTest, getConstantBoundOnDimSizeUnconstrained) {
  MLIRContext context;

  FlatLinearConstraints cst(/*numDims=*/1);

  AffineMap ub;
  std::optional<int64_t> size =
      cst.getConstantBoundOnDimSize(&context, /*pos=*/0, /*lb=*/nullptr, &ub);
  EXPECT_FALSE(size.has_value());
  EXPECT_FALSE(ub);
}

// Sanity check that a bounded dimension still reports both bound maps.
TEST(FlatLinearConstraintsTest, getConstantBoundOnDimSizeBounded) {
  MLIRContext context;

  // 2 <= x <= 5.
  FlatLinearConstraints cst(/*numDims=*/1);
  cst.addBound(presburger::BoundType::LB, 0, 2);
  cst.addBound(presburger::BoundType::UB, 0, 5);

  AffineMap lb, ub;
  std::optional<int64_t> size =
      cst.getConstantBoundOnDimSize(&context, /*pos=*/0, &lb, &ub);
  ASSERT_TRUE(size.has_value());
  EXPECT_EQ(*size, 4);
  ASSERT_TRUE(lb);
  ASSERT_TRUE(ub);
  EXPECT_EQ(lb.getSingleConstantResult(), 2);
  EXPECT_EQ(ub.getSingleConstantResult(), 5);
}
