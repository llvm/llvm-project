//===- IntRangeOptimizations.cpp - Optimizations based on integer ranges --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::arith {
#define GEN_PASS_DEF_ARITHINTRANGEOPTS
#include "mlir/Dialect/Arith/Transforms/Passes.h.inc"
} // namespace mlir::arith

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::dataflow;

static std::optional<APInt> getMaybeConstantValue(DataFlowSolver &solver,
                                                  Value value) {
  auto *maybeInferredRange =
      solver.lookupState<IntegerValueRangeLattice>(value);
  if (!maybeInferredRange || maybeInferredRange->getValue().isUninitialized())
    return std::nullopt;
  const ConstantIntRanges &inferredRange =
      maybeInferredRange->getValue().getValue();
  return inferredRange.getConstantValue();
}

/// Patterned after SCCP
static LogicalResult maybeReplaceWithConstant(DataFlowSolver &solver,
                                              PatternRewriter &rewriter,
                                              Value value) {
  if (value.use_empty())
    return failure();
  std::optional<APInt> maybeConstValue = getMaybeConstantValue(solver, value);
  if (!maybeConstValue.has_value())
    return failure();

  Operation *maybeDefiningOp = value.getDefiningOp();
  Dialect *valueDialect =
      maybeDefiningOp ? maybeDefiningOp->getDialect()
                      : value.getParentRegion()->getParentOp()->getDialect();
  Attribute constAttr =
      rewriter.getIntegerAttr(value.getType(), *maybeConstValue);
  Operation *constOp = valueDialect->materializeConstant(
      rewriter, constAttr, value.getType(), value.getLoc());
  // Fall back to arith.constant if the dialect materializer doesn't know what
  // to do with an integer constant.
  if (!constOp)
    constOp = rewriter.getContext()
                  ->getLoadedDialect<ArithDialect>()
                  ->materializeConstant(rewriter, constAttr, value.getType(),
                                        value.getLoc());
  if (!constOp)
    return failure();

  rewriter.replaceAllUsesWith(value, constOp->getResult(0));
  return success();
}

namespace {
class DataFlowListener : public RewriterBase::Listener {
public:
  DataFlowListener(DataFlowSolver &s) : s(s) {}

protected:
  void notifyOperationErased(Operation *op) override {
    s.eraseState(op);
    for (Value res : op->getResults())
      s.eraseState(res);
  }

  DataFlowSolver &s;
};

/// Rewrite any results of `op` that were inferred to be constant integers to
/// and replace their uses with that constant. Return success() if all results
/// where thus replaced and the operation is erased. Also replace any block
/// arguments with their constant values.
struct MaterializeKnownConstantValues : public RewritePattern {
  MaterializeKnownConstantValues(MLIRContext *context, DataFlowSolver &s)
      : RewritePattern(Pattern::MatchAnyOpTypeTag(), /*benefit=*/1, context),
        solver(s) {}

  LogicalResult match(Operation *op) const override {
    if (matchPattern(op, m_Constant()))
      return failure();

    auto needsReplacing = [&](Value v) {
      return getMaybeConstantValue(solver, v).has_value() && !v.use_empty();
    };
    bool hasConstantResults = llvm::any_of(op->getResults(), needsReplacing);
    if (op->getNumRegions() == 0)
      return success(hasConstantResults);
    bool hasConstantRegionArgs = false;
    for (Region &region : op->getRegions()) {
      for (Block &block : region.getBlocks()) {
        hasConstantRegionArgs |=
            llvm::any_of(block.getArguments(), needsReplacing);
      }
    }
    return success(hasConstantResults || hasConstantRegionArgs);
  }

  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
    bool replacedAll = (op->getNumResults() != 0);
    for (Value v : op->getResults())
      replacedAll &=
          (succeeded(maybeReplaceWithConstant(solver, rewriter, v)) ||
           v.use_empty());
    if (replacedAll && isOpTriviallyDead(op)) {
      rewriter.eraseOp(op);
      return;
    }

    PatternRewriter::InsertionGuard guard(rewriter);
    for (Region &region : op->getRegions()) {
      for (Block &block : region.getBlocks()) {
        rewriter.setInsertionPointToStart(&block);
        for (BlockArgument &arg : block.getArguments()) {
          (void)maybeReplaceWithConstant(solver, rewriter, arg);
        }
      }
    }
  }

private:
  DataFlowSolver &solver;
};

template <typename RemOp>
struct DeleteTrivialRem : public OpRewritePattern<RemOp> {
  DeleteTrivialRem(MLIRContext *context, DataFlowSolver &s)
      : OpRewritePattern<RemOp>(context), solver(s) {}

  LogicalResult matchAndRewrite(RemOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getOperand(0);
    Value rhs = op.getOperand(1);
    auto maybeModulus = getConstantIntValue(rhs);
    if (!maybeModulus.has_value())
      return failure();
    int64_t modulus = *maybeModulus;
    if (modulus <= 0)
      return failure();
    auto *maybeLhsRange = solver.lookupState<IntegerValueRangeLattice>(lhs);
    if (!maybeLhsRange || maybeLhsRange->getValue().isUninitialized())
      return failure();
    const ConstantIntRanges &lhsRange = maybeLhsRange->getValue().getValue();
    const APInt &min = isa<RemUIOp>(op) ? lhsRange.umin() : lhsRange.smin();
    const APInt &max = isa<RemUIOp>(op) ? lhsRange.umax() : lhsRange.smax();
    // The minima and maxima here are given as closed ranges, we must be
    // strictly less than the modulus.
    if (min.isNegative() || min.uge(modulus))
      return failure();
    if (max.isNegative() || max.uge(modulus))
      return failure();
    if (!min.ule(max))
      return failure();

    // With all those conditions out of the way, we know thas this invocation of
    // a remainder is a noop because the input is strictly within the range
    // [0, modulus), so get rid of it.
    rewriter.replaceOp(op, ValueRange{lhs});
    return success();
  }

private:
  DataFlowSolver &solver;
};

struct IntRangeOptimizationsPass
    : public arith::impl::ArithIntRangeOptsBase<IntRangeOptimizationsPass> {

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = op->getContext();
    DataFlowSolver solver;
    solver.load<DeadCodeAnalysis>();
    solver.load<IntegerRangeAnalysis>();
    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();

    DataFlowListener listener(solver);

    RewritePatternSet patterns(ctx);
    populateIntRangeOptimizationsPatterns(patterns, solver);

    GreedyRewriteConfig config;
    config.listener = &listener;

    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns), config)))
      signalPassFailure();
  }
};
} // namespace

void mlir::arith::populateIntRangeOptimizationsPatterns(
    RewritePatternSet &patterns, DataFlowSolver &solver) {
  patterns.add<MaterializeKnownConstantValues, DeleteTrivialRem<RemSIOp>,
               DeleteTrivialRem<RemUIOp>>(patterns.getContext(), solver);
}

std::unique_ptr<Pass> mlir::arith::createIntRangeOptimizationsPass() {
  return std::make_unique<IntRangeOptimizationsPass>();
}
