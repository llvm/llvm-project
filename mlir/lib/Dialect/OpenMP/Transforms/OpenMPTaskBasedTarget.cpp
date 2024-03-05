//===- OpenMPTaskBasedTarget.cpp - Implementation of OpenMPTaskBasedTargetPass
//---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that transforms certain omp.target.
// Specifically, an omp.target op that has the depend clause on it is
// transformed into an omp.task clause with the same depend clause on it.
// The original omp.target loses its depend clause and is contained in
// the new task region.
//
// omp.target depend(..) {
//  omp.terminator
//
// }
//
// =>
//
// omp.task depend(..) {
//   omp.target {
//     omp.terminator
//   }
//   omp.terminator
// }
//
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenMP/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_OPENMPTASKBASEDTARGET
#include "mlir/Dialect/OpenMP/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::omp;

#define DEBUG_TYPE "openmp-task-based-target"

namespace {

struct OpenMPTaskBasedTargetPass
    : public impl::OpenMPTaskBasedTargetBase<OpenMPTaskBasedTargetPass> {

  void runOnOperation() override;
};
template <typename OpTy>
class OmpTaskBasedTargetRewritePattern : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {

    // Only match a target op with a  'depend' clause on it.
    if (op.getDependVars().empty()) {
      return rewriter.notifyMatchFailure(op, "depend clause not found on op");
    }

    // Step 1: Create a new task op and tack on the dependency from the 'depend'
    // clause on it.
    omp::TaskOp taskOp = rewriter.create<omp::TaskOp>(
        op.getLoc(), /*if_expr*/ Value(),
        /*final_expr*/ Value(),
        /*untied*/ UnitAttr(),
        /*mergeable*/ UnitAttr(),
        /*in_reduction_vars*/ ValueRange(),
        /*in_reductions*/ nullptr,
        /*priority*/ Value(), op.getDepends().value(), op.getDependVars(),
        /*allocate_vars*/ ValueRange(),
        /*allocate_vars*/ ValueRange());
    Block *block = rewriter.createBlock(&taskOp.getRegion());
    rewriter.setInsertionPointToEnd(block);
    // Step 2: Clone and put the entire target op inside the newly created
    // task's region.
    Operation *clonedTargetOperation = rewriter.clone(*op.getOperation());
    rewriter.create<mlir::omp::TerminatorOp>(op.getLoc());

    // Step 3: Remove the dependency information from the clone target op.
    OpTy clonedTargetOp = llvm::dyn_cast<OpTy>(clonedTargetOperation);
    if (clonedTargetOp) {
      clonedTargetOp.removeDependsAttr();
      clonedTargetOp.getDependVarsMutable().clear();
    }
    // Step 4: Erase the original target op
    rewriter.eraseOp(op.getOperation());
    return success();
  }
};
} // namespace
static void
populateOmpTaskBasedTargetRewritePatterns(RewritePatternSet &patterns) {
  patterns.add<OmpTaskBasedTargetRewritePattern<omp::TargetOp>,
               OmpTaskBasedTargetRewritePattern<omp::EnterDataOp>,
               OmpTaskBasedTargetRewritePattern<omp::UpdateDataOp>,
               OmpTaskBasedTargetRewritePattern<omp::ExitDataOp>>(
      patterns.getContext());
}

void OpenMPTaskBasedTargetPass::runOnOperation() {
  Operation *op = getOperation();

  RewritePatternSet patterns(op->getContext());
  populateOmpTaskBasedTargetRewritePatterns(patterns);

  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
    signalPassFailure();
}
std::unique_ptr<Pass> mlir::createOpenMPTaskBasedTargetPass() {
  return std::make_unique<OpenMPTaskBasedTargetPass>();
}
