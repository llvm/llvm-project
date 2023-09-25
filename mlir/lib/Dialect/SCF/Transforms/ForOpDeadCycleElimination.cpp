//==-- ForOpDeadCycleElimination.cpp - dead code elimination for scf.for ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::scf;
namespace {
/// Detect dead arguments in scf.for op by assuming all the values are dead and
/// propagate liveness property.
struct ForOpDeadArgElimination : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const final {
    Block &block = *forOp.getBody();
    auto yieldOp = cast<scf::YieldOp>(block.getTerminator());
    // Assume that nothing is live at the beginning and mark values as live
    // based on uses.
    DenseSet<Value> aliveValues;
    SmallVector<Value> queue;
    // Helper to mark values as live and add them to the queue of value to
    // propagate if it is the first time we detect the value as live.
    auto markLive = [&](Value val) {
      if (!forOp->isAncestor(val.getParentRegion()->getParentOp()))
        return;
      if (aliveValues.insert(val).second)
        queue.push_back(val);
    };
    // Mark all yield operands as live if the associated forOp result has any
    // use.
    for (auto result : llvm::enumerate(forOp.getResults())) {
      if (!result.value().use_empty())
        markLive(yieldOp.getOperand(result.index()));
    }
    if (aliveValues.size() == forOp.getNumResults())
      return failure();
    // Operations with side-effects are always live. Mark all theirs operands as
    // live except for scf.for and scf.if that have special handling.
    block.walk([&](Operation *op) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
        if (isa<scf::ForOp, scf::IfOp>(yieldOp->getParentOp()))
          return;
      }
      if (!isa<scf::ForOp, scf::IfOp>(op) && !wouldOpBeTriviallyDead(op)) {
        for (Value operand : op->getOperands())
          markLive(operand);
      }
    });
    // Propagate live property until reaching a fixed point.
    while (!queue.empty()) {
      Value value = queue.pop_back_val();
      if (auto nestedFor = value.getDefiningOp<scf::ForOp>()) {
        auto result = value.cast<OpResult>();
        OpOperand &forOperand = nestedFor.getOpOperandForResult(result);
        markLive(forOperand.get());
        auto nestedYieldOp =
            cast<scf::YieldOp>(nestedFor.getBody()->getTerminator());
        Value nestedYieldOperand =
            nestedYieldOp.getOperand(result.getResultNumber());
        markLive(nestedYieldOperand);
        continue;
      }
      if (auto nestedIf = value.getDefiningOp<scf::IfOp>()) {
        auto result = value.cast<OpResult>();
        for (scf::YieldOp nestedYieldOp :
             {nestedIf.thenYield(), nestedIf.elseYield()}) {
          Value nestedYieldOperand =
              nestedYieldOp.getOperand(result.getResultNumber());
          markLive(nestedYieldOperand);
        }
        continue;
      }
      if (Operation *def = value.getDefiningOp()) {
        for (Value operand : def->getOperands())
          markLive(operand);
        continue;
      }
      // If an argument block is live then the associated yield operand and
      // forOp operand are live.
      auto arg = value.cast<BlockArgument>();
      if (auto forOwner = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp())) {
        if (arg.getArgNumber() < forOwner.getNumInductionVars())
          continue;
        unsigned iterIdx = arg.getArgNumber() - forOwner.getNumInductionVars();
        Value yieldOperand =
            forOwner.getBody()->getTerminator()->getOperand(iterIdx);
        markLive(yieldOperand);
        markLive(forOwner.getInitArgs()[iterIdx]);
      }
    }
    SmallVector<unsigned> deadArg;
    for (auto yieldOperand : llvm::enumerate(yieldOp->getOperands())) {
      if (aliveValues.contains(yieldOperand.value()))
        continue;
      if (yieldOperand.value() == block.getArgument(yieldOperand.index() + 1))
        continue;
      deadArg.push_back(yieldOperand.index());
    }
    if (deadArg.empty())
      return failure();
    rewriter.updateRootInPlace(forOp, [&]() {
      // For simplicity we just change the dead yield operand to use the
      // associated argument and leave the operations and argument removal to
      // dead code elimination.
      for (unsigned deadArgIdx : deadArg) {
        BlockArgument arg = block.getArgument(deadArgIdx + 1);
        yieldOp.setOperand(deadArgIdx, arg);
      }
    });
    return success();
  }
};

} // namespace

void mlir::scf::populateForOpDeadCycleEliminationPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ForOpDeadArgElimination>(patterns.getContext());
}
