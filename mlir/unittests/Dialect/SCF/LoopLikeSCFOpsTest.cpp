//===- LoopLikeSCFOpsTest.cpp - SCF LoopLikeOpInterface Tests -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
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
    std::optional<OpFoldResult> maybeSingleLb =
        loopLikeOp.getSingleLowerBound();
    EXPECT_TRUE(maybeSingleLb.has_value());
    std::optional<OpFoldResult> maybeSingleUb =
        loopLikeOp.getSingleUpperBound();
    EXPECT_TRUE(maybeSingleUb.has_value());
    std::optional<OpFoldResult> maybeSingleStep = loopLikeOp.getSingleStep();
    EXPECT_TRUE(maybeSingleStep.has_value());
    std::optional<OpFoldResult> maybeSingleIndVar =
        loopLikeOp.getSingleInductionVar();
    EXPECT_TRUE(maybeSingleIndVar.has_value());

    std::optional<SmallVector<OpFoldResult>> maybeLb =
        loopLikeOp.getLoopLowerBounds();
    ASSERT_TRUE(maybeLb.has_value());
    EXPECT_EQ((*maybeLb).size(), 1u);
    std::optional<SmallVector<OpFoldResult>> maybeUb =
        loopLikeOp.getLoopUpperBounds();
    ASSERT_TRUE(maybeUb.has_value());
    EXPECT_EQ((*maybeUb).size(), 1u);
    std::optional<SmallVector<OpFoldResult>> maybeStep =
        loopLikeOp.getLoopSteps();
    ASSERT_TRUE(maybeStep.has_value());
    EXPECT_EQ((*maybeStep).size(), 1u);
    std::optional<SmallVector<Value>> maybeInductionVars =
        loopLikeOp.getLoopInductionVars();
    ASSERT_TRUE(maybeInductionVars.has_value());
    EXPECT_EQ((*maybeInductionVars).size(), 1u);
  }

  void checkMultidimensional(LoopLikeOpInterface loopLikeOp) {
    std::optional<OpFoldResult> maybeSingleLb =
        loopLikeOp.getSingleLowerBound();
    EXPECT_FALSE(maybeSingleLb.has_value());
    std::optional<OpFoldResult> maybeSingleUb =
        loopLikeOp.getSingleUpperBound();
    EXPECT_FALSE(maybeSingleUb.has_value());
    std::optional<OpFoldResult> maybeSingleStep = loopLikeOp.getSingleStep();
    EXPECT_FALSE(maybeSingleStep.has_value());
    std::optional<OpFoldResult> maybeSingleIndVar =
        loopLikeOp.getSingleInductionVar();
    EXPECT_FALSE(maybeSingleIndVar.has_value());

    std::optional<SmallVector<OpFoldResult>> maybeLb =
        loopLikeOp.getLoopLowerBounds();
    ASSERT_TRUE(maybeLb.has_value());
    EXPECT_EQ((*maybeLb).size(), 2u);
    std::optional<SmallVector<OpFoldResult>> maybeUb =
        loopLikeOp.getLoopUpperBounds();
    ASSERT_TRUE(maybeUb.has_value());
    EXPECT_EQ((*maybeUb).size(), 2u);
    std::optional<SmallVector<OpFoldResult>> maybeStep =
        loopLikeOp.getLoopSteps();
    ASSERT_TRUE(maybeStep.has_value());
    EXPECT_EQ((*maybeStep).size(), 2u);
    std::optional<SmallVector<Value>> maybeInductionVars =
        loopLikeOp.getLoopInductionVars();
    ASSERT_TRUE(maybeInductionVars.has_value());
    EXPECT_EQ((*maybeInductionVars).size(), 2u);
  }

  MLIRContext context;
  OpBuilder b;
  Location loc;
};

TEST_F(SCFLoopLikeTest, queryUnidimensionalLooplikes) {
  OwningOpRef<arith::ConstantIndexOp> lb =
      b.create<arith::ConstantIndexOp>(loc, 0);
  OwningOpRef<arith::ConstantIndexOp> ub =
      b.create<arith::ConstantIndexOp>(loc, 10);
  OwningOpRef<arith::ConstantIndexOp> step =
      b.create<arith::ConstantIndexOp>(loc, 2);

  OwningOpRef<scf::ForOp> forOp =
      b.create<scf::ForOp>(loc, lb.get(), ub.get(), step.get());
  checkUnidimensional(forOp.get());

  OwningOpRef<scf::ForallOp> forallOp = b.create<scf::ForallOp>(
      loc, ArrayRef<OpFoldResult>(lb->getResult()),
      ArrayRef<OpFoldResult>(ub->getResult()),
      ArrayRef<OpFoldResult>(step->getResult()), ValueRange(), std::nullopt);
  checkUnidimensional(forallOp.get());

  OwningOpRef<scf::ParallelOp> parallelOp = b.create<scf::ParallelOp>(
      loc, ValueRange(lb->getResult()), ValueRange(ub->getResult()),
      ValueRange(step->getResult()), ValueRange());
  checkUnidimensional(parallelOp.get());
}

TEST_F(SCFLoopLikeTest, queryMultidimensionalLooplikes) {
  OwningOpRef<arith::ConstantIndexOp> lb =
      b.create<arith::ConstantIndexOp>(loc, 0);
  OwningOpRef<arith::ConstantIndexOp> ub =
      b.create<arith::ConstantIndexOp>(loc, 10);
  OwningOpRef<arith::ConstantIndexOp> step =
      b.create<arith::ConstantIndexOp>(loc, 2);

  OwningOpRef<scf::ForallOp> forallOp = b.create<scf::ForallOp>(
      loc, ArrayRef<OpFoldResult>({lb->getResult(), lb->getResult()}),
      ArrayRef<OpFoldResult>({ub->getResult(), ub->getResult()}),
      ArrayRef<OpFoldResult>({step->getResult(), step->getResult()}),
      ValueRange(), std::nullopt);
  checkMultidimensional(forallOp.get());

  OwningOpRef<scf::ParallelOp> parallelOp = b.create<scf::ParallelOp>(
      loc, ValueRange({lb->getResult(), lb->getResult()}),
      ValueRange({ub->getResult(), ub->getResult()}),
      ValueRange({step->getResult(), step->getResult()}), ValueRange());
  checkMultidimensional(parallelOp.get());
}
