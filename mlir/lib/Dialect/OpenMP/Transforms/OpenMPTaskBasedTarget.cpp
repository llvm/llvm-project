//===- OpenMPTaskBasedTarget.cpp - Implementation of OpenMPTaskBasedTargetPass
//---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements scf.parallel to scf.for + async.execute conversion pass.
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
    if (op.getDependVars().empty()) {
      return rewriter.notifyMatchFailure(op, "depend clause not found on op");
    }
    return success();
  }
};
} // namespace
static void
populateOmpTaskBasedTargetRewritePatterns(RewritePatternSet &patterns) {
  patterns.add<OmpTaskBasedTargetRewritePattern<omp::TargetOp>>(
      patterns.getContext());
}

void OpenMPTaskBasedTargetPass::runOnOperation() {
  Operation *op = getOperation();
  LLVM_DEBUG(llvm::dbgs() << "Running on the following operation\n");
  //  LLVM_DEBUG(llvm::dbgs() << op->dump());

  RewritePatternSet patterns(op->getContext());
  populateOmpTaskBasedTargetRewritePatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
    signalPassFailure();
}
std::unique_ptr<Pass> mlir::createOpenMPTaskBasedTargetPass() {
  return std::make_unique<OpenMPTaskBasedTargetPass>();
}
