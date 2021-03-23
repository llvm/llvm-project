//===- SCFToOpenMP.cpp - Structured Control Flow to OpenMP conversion -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert scf.parallel operations into OpenMP
// parallel loops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "../PassDetail.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

/// Converts SCF parallel operation into an OpenMP workshare loop construct.
struct ParallelOpLowering : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ParallelOp parallelOp,
                                PatternRewriter &rewriter) const override {
    // TODO: add support for reductions when OpenMP loops have them.
    if (parallelOp.getNumResults() != 0)
      return rewriter.notifyMatchFailure(
          parallelOp,
          "OpenMP dialect does not yet support loops with reductions");

    // Replace SCF yield with OpenMP yield.
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(parallelOp.getBody());
      assert(llvm::hasSingleElement(parallelOp.region()) &&
             "expected scf.parallel to have one block");
      rewriter.replaceOpWithNewOp<omp::YieldOp>(
          parallelOp.getBody()->getTerminator(), ValueRange());
    }

    // Replace the loop.
    auto loop = rewriter.create<omp::WsLoopOp>(
        parallelOp.getLoc(), parallelOp.lowerBound(), parallelOp.upperBound(),
        parallelOp.step());
    rewriter.inlineRegionBefore(parallelOp.region(), loop.region(),
                                loop.region().begin());
    rewriter.eraseOp(parallelOp);
    return success();
  }
};

/// Inserts OpenMP "parallel" operations around top-level SCF "parallel"
/// operations in the given function. This is implemented as a direct IR
/// modification rather than as a conversion pattern because it does not
/// modify the top-level operation it matches, which is a requirement for
/// rewrite patterns.
//
// TODO: consider creating nested parallel operations when necessary.
static void insertOpenMPParallel(FuncOp func) {
  // Collect top-level SCF "parallel" ops.
  SmallVector<scf::ParallelOp, 4> topLevelParallelOps;
  func.walk([&topLevelParallelOps](scf::ParallelOp parallelOp) {
    // Ignore ops that are already within OpenMP parallel construct.
    if (!parallelOp->getParentOfType<scf::ParallelOp>())
      topLevelParallelOps.push_back(parallelOp);
  });

  // Wrap SCF ops into OpenMP "parallel" ops.
  for (scf::ParallelOp parallelOp : topLevelParallelOps) {
    OpBuilder builder(parallelOp);
    auto omp = builder.create<omp::ParallelOp>(parallelOp.getLoc());
    Block *block = builder.createBlock(&omp.getRegion());
    builder.create<omp::TerminatorOp>(parallelOp.getLoc());
    block->getOperations().splice(block->begin(),
                                  parallelOp->getBlock()->getOperations(),
                                  parallelOp.getOperation());
  }
}

/// Applies the conversion patterns in the given function.
static LogicalResult applyPatterns(FuncOp func) {
  ConversionTarget target(*func.getContext());
  target.addIllegalOp<scf::ParallelOp>();
  target.addDynamicallyLegalOp<scf::YieldOp>(
      [](scf::YieldOp op) { return !isa<scf::ParallelOp>(op->getParentOp()); });
  target.addLegalDialect<omp::OpenMPDialect>();

  RewritePatternSet patterns(func.getContext());
  patterns.add<ParallelOpLowering>(func.getContext());
  FrozenRewritePatternSet frozen(std::move(patterns));
  return applyPartialConversion(func, target, frozen);
}

/// A pass converting SCF operations to OpenMP operations.
struct SCFToOpenMPPass : public ConvertSCFToOpenMPBase<SCFToOpenMPPass> {
  /// Pass entry point.
  void runOnFunction() override {
    insertOpenMPParallel(getFunction());
    if (failed(applyPatterns(getFunction())))
      signalPassFailure();
  }
};

} // end namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::createConvertSCFToOpenMPPass() {
  return std::make_unique<SCFToOpenMPPass>();
}
