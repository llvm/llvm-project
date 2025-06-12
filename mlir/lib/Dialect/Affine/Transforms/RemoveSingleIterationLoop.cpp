//===- RemoveSingleIterationLoop.cpp --- remove single iteration loop ---*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_REMOVESINGLEITERATIONLOOP
#include "mlir/Dialect/Affine/Passes.h.inc"
} // namespace affine
} // namespace mlir

#define DEBUG_TYPE "affine-remove-single-iteration"

using namespace mlir;
using namespace affine;

/// Replaces the given op with the contents of the given single-block region,
/// using the operands of the block terminator to replace operation results.
static void replaceOpWithRegion(PatternRewriter &rewriter, Operation *op,
                                Region &region, ValueRange blockArgs = {}) {
  assert(llvm::hasSingleElement(region) && "expected single-region block");
  Block *block = &region.front();
  Operation *terminator = block->getTerminator();
  ValueRange results = terminator->getOperands();
  rewriter.inlineBlockBefore(block, op, blockArgs);
  rewriter.replaceOp(op, results);
  rewriter.eraseOp(terminator);
}

/// Return true if we can prove that the we always run at least the first
/// iteration of the ForOp.
static bool alwaysRunsFirstIteration(AffineForOp op) {
  // Can't perform the analysis if the loops's bounds aren't index-typed.
  if (!op.getInductionVar().getType().isIndex())
    return false;
  SmallVector<Value> lowerMapOperands = op.getLowerBoundOperands();
  SmallVector<Value> upperMapOperands = op.getUpperBoundOperands();
  ValueBoundsConstraintSet::Variable lower(op.getLowerBoundMap(),
                                           lowerMapOperands);
  ValueBoundsConstraintSet::Variable upper(op.getUpperBoundMap(),
                                           upperMapOperands);
  FailureOr<bool> isLb = ValueBoundsConstraintSet::compare(
      lower, ValueBoundsConstraintSet::LT, upper);
  return isLb.value_or(false);
}

/// Return true if we can prove that the we never run more than one iteration of
/// the ForOp.
static bool neverRunsSecondIteration(AffineForOp op) {
  // Can't perform the analysis if the loops's bounds aren't index-typed.
  if (!op.getInductionVar().getType().isIndex())
    return false;

  // The loop will only loop once if the inducation variable for the next time
  // in the loop is greater than or equal to upper.
  MLIRContext *context = op.getContext();
  SmallVector<Value> lowerMapOperands = op.getLowerBoundOperands();
  SmallVector<Value> upperMapOperands = op.getUpperBoundOperands();
  SmallVector<AffineExpr> results;
  AffineMap lowerMap = op.getLowerBoundMap();
  for (AffineExpr expr : lowerMap.getResults()) {
    results.push_back(expr + op.getStep().getSExtValue());
  }
  AffineMap nextItMap = AffineMap::get(
      lowerMap.getNumDims(), lowerMap.getNumSymbols(), results, context);
  ValueBoundsConstraintSet::Variable nextItVar(nextItMap, lowerMapOperands);
  ValueBoundsConstraintSet::Variable upperVar(op.getUpperBoundMap(),
                                              upperMapOperands);
  FailureOr<bool> isUpperUnderNextIter = ValueBoundsConstraintSet::compare(
      nextItVar, ValueBoundsConstraintSet::LE, upperVar);
  return isUpperUnderNextIter.value_or(false);
}

namespace {

/// Rewriting pattern that replaces single-iteration loops with their bodies.
struct SimplifyTrivialLoops : public OpRewritePattern<AffineForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineForOp op,
                                PatternRewriter &rewriter) const override {
    if (!(alwaysRunsFirstIteration(op) && neverRunsSecondIteration(op))) {
      return failure();
    }

    // The first iteration is always run and the second iteration is never run
    // so the loop always have 1 iteration. Inline its body and remove the loop.
    SmallVector<Value> blockArgs;
    blockArgs.reserve(op.getInits().size() + 1);
    rewriter.setInsertionPointToStart(op.getBody());
    Value lower = rewriter.create<AffineApplyOp>(
        op.getLoc(), op.getLowerBoundMap(), op.getLowerBoundOperands());
    op.getInductionVar().replaceAllUsesWith(lower);
    blockArgs.push_back(lower);
    llvm::append_range(blockArgs, op.getInits());
    replaceOpWithRegion(rewriter, op, op.getRegion(), blockArgs);
    return success();
  }
};

struct RemoveSingleIterationLoop
    : public affine::impl::RemoveSingleIterationLoopBase<
          RemoveSingleIterationLoop> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    RewritePatternSet patterns(funcOp.getContext());
    populateRemoveSingleIterationLoopPattern(patterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

void mlir::affine::populateRemoveSingleIterationLoopPattern(
    RewritePatternSet &patterns) {
  patterns.add<SimplifyTrivialLoops>(patterns.getContext());
}

std::unique_ptr<InterfacePass<FunctionOpInterface>>
mlir::affine::createRemoveSingleIterationLoopPass() {
  return std::make_unique<RemoveSingleIterationLoop>();
}
