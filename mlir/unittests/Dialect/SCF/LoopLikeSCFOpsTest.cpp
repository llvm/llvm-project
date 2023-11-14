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
