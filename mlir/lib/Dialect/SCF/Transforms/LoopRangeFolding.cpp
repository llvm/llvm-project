//===- LoopRangeFolding.cpp - Code to perform loop range folding-----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements loop range folding.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_SCFFORLOOPRANGEFOLDING
#include "mlir/Dialect/SCF/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::scf;

namespace {
struct ForLoopRangeFoldingPattern : public OpRewritePattern<ForOp> {
  using Base::Base;

  void initialize() { setHasBoundedRewriteRecursion(); }

  LogicalResult matchAndRewrite(ForOp op,
                                PatternRewriter &rewriter) const override {
    Value indVar = op.getInductionVar();

    auto canBeFolded = [&](Value value) {
      return op.isDefinedOutsideOfLoop(value) || value == indVar;
    };

    // If the induction variable is used by more than one operation, we can't
    // fold its arith ops into the loop range. Note that a single operation
    // may use the induction variable several times (e.g. `arith.addi %i,
    // %i`), so check for a single user rather than a single use.
    if (indVar.use_empty())
      return failure();

    Operation *user = *indVar.getUsers().begin();
    if (!llvm::all_of(indVar.getUsers(),
                      [&](Operation *u) { return u == user; }))
      return failure();

    if (!isa<arith::AddIOp, arith::MulIOp>(user))
      return failure();

    if (!llvm::all_of(user->getOperands(), canBeFolded))
      return failure();

    if (auto mulOp = dyn_cast<arith::MulIOp>(user)) {
      // Only fold if the multiplier is a known strictly positive constant.
      // Multiplying by zero or a negative value would produce an invalid
      // step (scf.for requires a strictly positive step).
      Value multiplier =
          (mulOp.getLhs() == indVar) ? mulOp.getRhs() : mulOp.getLhs();
      std::optional<int64_t> multiplierVal = getConstantIntValue(multiplier);
      if (!multiplierVal || *multiplierVal <= 0)
        return failure();
    }

    rewriter.setInsertionPoint(op);
    IRMapping lbMap;
    lbMap.map(indVar, op.getLowerBound());
    IRMapping ubMap;
    ubMap.map(indVar, op.getUpperBound());
    IRMapping stepMap;
    stepMap.map(indVar, op.getStep());

    Operation *lbFold = rewriter.clone(*user, lbMap);
    Operation *ubFold = rewriter.clone(*user, ubMap);
    Operation *stepFold = nullptr;
    if (auto addOp = dyn_cast<arith::AddIOp>(user)) {
      // `arith.addi %i, %i` is `2 * %i`, so the step has to be doubled too.
      if (addOp.getLhs() == indVar && addOp.getRhs() == indVar)
        stepFold = rewriter.clone(*user, stepMap);
    } else {
      stepFold = rewriter.clone(*user, stepMap);
    }

    rewriter.modifyOpInPlace(op, [&]() {
      op.setLowerBound(lbFold->getResult(0));
      op.setUpperBound(ubFold->getResult(0));
      if (stepFold)
        op.setStep(stepFold->getResult(0));
    });
    rewriter.replaceOp(user, indVar);
    return success();
  }
};

struct ForLoopRangeFolding
    : public impl::SCFForLoopRangeFoldingBase<ForLoopRangeFolding> {
  void runOnOperation() override;
};
} // namespace

void ForLoopRangeFolding::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<ForLoopRangeFoldingPattern>(&getContext());
  GreedyRewriteConfig config;
  config.setStrictness(GreedyRewriteStrictness::ExistingOps)
      .enableFolding(false)
      .enableConstantCSE(false)
      .setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
  if (failed(
          applyPatternsGreedily(getOperation(), std::move(patterns), config)))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::createForLoopRangeFoldingPass() {
  return std::make_unique<ForLoopRangeFolding>();
}
