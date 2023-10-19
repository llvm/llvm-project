//===- LoopLikeSCFOpsTest.cpp - SCF LoopLikeOpInterface Tests -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::scf;

//===----------------------------------------------------------------------===//
// Test Fixture
//===----------------------------------------------------------------------===//

class SCFLoopLikeTest : public ::testing::Test {
protected:
  SCFLoopLikeTest() : b(&context), loc(UnknownLoc::get(&context)) {
    context.loadDialect<arith::ArithDialect, scf::SCFDialect>();
  }

  void checkUnidimensional(LoopLikeOpInterface loopLikeOp) {
    std::optional<OpFoldResult> maybeLb = loopLikeOp.getSingleLowerBound();
    EXPECT_TRUE(maybeLb.has_value());
    std::optional<OpFoldResult> maybeUb = loopLikeOp.getSingleUpperBound();
    EXPECT_TRUE(maybeUb.has_value());
    std::optional<OpFoldResult> maybeStep = loopLikeOp.getSingleStep();
    EXPECT_TRUE(maybeStep.has_value());
    std::optional<OpFoldResult> maybeIndVar =
        loopLikeOp.getSingleInductionVar();
    EXPECT_TRUE(maybeIndVar.has_value());
  }

  void checkMultidimensional(LoopLikeOpInterface loopLikeOp) {
    std::optional<OpFoldResult> maybeLb = loopLikeOp.getSingleLowerBound();
    EXPECT_FALSE(maybeLb.has_value());
    std::optional<OpFoldResult> maybeUb = loopLikeOp.getSingleUpperBound();
    EXPECT_FALSE(maybeUb.has_value());
    std::optional<OpFoldResult> maybeStep = loopLikeOp.getSingleStep();
    EXPECT_FALSE(maybeStep.has_value());
    std::optional<OpFoldResult> maybeIndVar =
        loopLikeOp.getSingleInductionVar();
    EXPECT_FALSE(maybeIndVar.has_value());
  }

  MLIRContext context;
  OpBuilder b;
  Location loc;
};

TEST_F(SCFLoopLikeTest, queryUnidimensionalLooplikes) {
  Value lb = b.create<arith::ConstantIndexOp>(loc, 0);
  Value ub = b.create<arith::ConstantIndexOp>(loc, 10);
  Value step = b.create<arith::ConstantIndexOp>(loc, 2);

  auto forOp = b.create<scf::ForOp>(loc, lb, ub, step);
  checkUnidimensional(forOp);

  auto forallOp = b.create<scf::ForallOp>(
      loc, ArrayRef<OpFoldResult>(lb), ArrayRef<OpFoldResult>(ub),
      ArrayRef<OpFoldResult>(step), ValueRange(), std::nullopt);
  checkUnidimensional(forallOp);

  auto parallelOp = b.create<scf::ParallelOp>(
      loc, ValueRange(lb), ValueRange(ub), ValueRange(step), ValueRange());
  checkUnidimensional(parallelOp);
}

TEST_F(SCFLoopLikeTest, queryMultidimensionalLooplikes) {
  Value lb = b.create<arith::ConstantIndexOp>(loc, 0);
  Value ub = b.create<arith::ConstantIndexOp>(loc, 10);
  Value step = b.create<arith::ConstantIndexOp>(loc, 2);

  auto forallOp = b.create<scf::ForallOp>(
      loc, ArrayRef<OpFoldResult>({lb, lb}), ArrayRef<OpFoldResult>({ub, ub}),
      ArrayRef<OpFoldResult>({step, step}), ValueRange(), std::nullopt);
  checkMultidimensional(forallOp);

  auto parallelOp = b.create<scf::ParallelOp>(
      loc, ValueRange({lb, lb}), ValueRange({ub, ub}), ValueRange({step, step}), ValueRange());
  checkMultidimensional(parallelOp);
}