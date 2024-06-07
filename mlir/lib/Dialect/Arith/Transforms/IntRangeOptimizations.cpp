//===- IntRangeOptimizations.cpp - Optimizations based on integer ranges --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "mlir/Dialect/Arith/Transforms/Passes.h"

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/FoldUtils.h"

namespace mlir::arith {
#define GEN_PASS_DEF_ARITHINTRANGEOPTS
#include "mlir/Dialect/Arith/Transforms/Passes.h.inc"
} // namespace mlir::arith

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::dataflow;

/// Patterned after SCCP
static LogicalResult replaceWithConstant(DataFlowSolver &solver,
                                         RewriterBase &rewriter,
                                         OperationFolder &folder, Value value) {
  auto *maybeInferredRange =
      solver.lookupState<IntegerValueRangeLattice>(value);
  if (!maybeInferredRange || maybeInferredRange->getValue().isUninitialized())
    return failure();
  const ConstantIntRanges &inferredRange =
      maybeInferredRange->getValue().getValue();
  std::optional<APInt> maybeConstValue = inferredRange.getConstantValue();
  if (!maybeConstValue.has_value())
    return failure();

  Operation *maybeDefiningOp = value.getDefiningOp();
  Dialect *valueDialect =
      maybeDefiningOp ? maybeDefiningOp->getDialect()
                      : value.getParentRegion()->getParentOp()->getDialect();
  Attribute constAttr =
      rewriter.getIntegerAttr(value.getType(), *maybeConstValue);
  Value constant = folder.getOrCreateConstant(
      rewriter.getInsertionBlock(), valueDialect, constAttr, value.getType());
  // Fall back to arith.constant if the dialect materializer doesn't know what
  // to do with an integer constant.
  if (!constant)
    constant = folder.getOrCreateConstant(
        rewriter.getInsertionBlock(),
        rewriter.getContext()->getLoadedDialect<ArithDialect>(), constAttr,
        value.getType());
  if (!constant)
    return failure();

  rewriter.replaceAllUsesWith(value, constant);
  return success();
}

/// Rewrite any results of `op` that were inferred to be constant integers to
/// and replace their uses with that constant. Return success() if all results
/// where thus replaced and the operation is erased.
static LogicalResult foldResultsToConstants(DataFlowSolver &solver,
                                            RewriterBase &rewriter,
                                            OperationFolder &folder,
                                            Operation &op) {
  bool replacedAll = op.getNumResults() != 0;
  for (Value res : op.getResults())
    replacedAll &=
        succeeded(replaceWithConstant(solver, rewriter, folder, res));

  // If all of the results of the operation were replaced, try to erase
  // the operation completely.
  if (replacedAll && wouldOpBeTriviallyDead(&op)) {
    assert(op.use_empty() && "expected all uses to be replaced");
    rewriter.eraseOp(&op);
    return success();
  }
  return failure();
}

/// This function hasn't come from anywhere and is relying on the overall
/// tests of the integer range inference implementation for its correctness.
static LogicalResult deleteTrivialRemainder(DataFlowSolver &solver,
                                            RewriterBase &rewriter,
                                            Operation &op) {
  if (!isa<RemSIOp, RemUIOp>(op))
    return failure();
  Value lhs = op.getOperand(0);
  Value rhs = op.getOperand(1);
  auto rhsConstVal = rhs.getDefiningOp<arith::ConstantIntOp>();
  if (!rhsConstVal)
    return failure();
  int64_t modulus = rhsConstVal.value();
  if (modulus <= 0)
    return failure();
  auto *maybeLhsRange = solver.lookupState<IntegerValueRangeLattice>(lhs);
  if (!maybeLhsRange || maybeLhsRange->getValue().isUninitialized())
    return failure();
  const ConstantIntRanges &lhsRange = maybeLhsRange->getValue().getValue();
  const APInt &min = llvm::isa<RemUIOp>(op) ? lhsRange.umin() : lhsRange.smin();
  const APInt &max = llvm::isa<RemUIOp>(op) ? lhsRange.umax() : lhsRange.smax();
  // The minima and maxima here are given as closed ranges, we must be strictly
  // less than the modulus.
  if (min.isNegative() || min.uge(modulus))
    return failure();
  if (max.isNegative() || max.uge(modulus))
    return failure();
  if (!min.ule(max))
    return failure();

  // With all those conditions out of the way, we know thas this invocation of
  // a remainder is a noop because the input is strictly within the range
  // [0, modulus), so get rid of it.
  rewriter.replaceOp(&op, ValueRange{lhs});
  return success();
}

static void doRewrites(DataFlowSolver &solver, MLIRContext *context,
                       MutableArrayRef<Region> initialRegions) {
  SmallVector<Block *> worklist;
  auto addToWorklist = [&](MutableArrayRef<Region> regions) {
    for (Region &region : regions)
      for (Block &block : llvm::reverse(region))
        worklist.push_back(&block);
  };

  IRRewriter rewriter(context);
  OperationFolder folder(context, rewriter.getListener());

  addToWorklist(initialRegions);
  while (!worklist.empty()) {
    Block *block = worklist.pop_back_val();

    for (Operation &op : llvm::make_early_inc_range(*block)) {
      if (matchPattern(&op, m_Constant())) {
        if (auto arithConstant = dyn_cast<ConstantOp>(op))
          folder.insertKnownConstant(&op, arithConstant.getValue());
        else
          folder.insertKnownConstant(&op);
        continue;
      }
      rewriter.setInsertionPoint(&op);

      // Try rewrites. Success means that the underlying operation was erased.
      if (succeeded(foldResultsToConstants(solver, rewriter, folder, op)))
        continue;
      if (isa<RemSIOp, RemUIOp>(op) &&
          succeeded(deleteTrivialRemainder(solver, rewriter, op)))
        continue;
      // Add any the regions of this operation to the worklist.
      addToWorklist(op.getRegions());
    }

    // Replace any block arguments with constants.
    rewriter.setInsertionPointToStart(block);
    for (BlockArgument arg : block->getArguments())
      (void)replaceWithConstant(solver, rewriter, folder, arg);
  }
}

namespace {
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

    doRewrites(solver, ctx, op->getRegions());
  }
};
} // namespace

std::unique_ptr<Pass> mlir::arith::createIntRangeOptimizationsPass() {
  return std::make_unique<IntRangeOptimizationsPass>();
}
