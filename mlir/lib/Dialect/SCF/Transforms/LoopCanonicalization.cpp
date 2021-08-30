//===- LoopCanonicalization.cpp - Cross-dialect canonicalization patterns -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains cross-dialect canonicalization patterns that cannot be
// actual canonicalization patterns due to undesired additional dependencies.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::scf;

namespace {
/// Fold dim ops of iter_args to dim ops of their respective init args. E.g.:
///
/// ```
/// %0 = ... : tensor<?x?xf32>
/// scf.for ... iter_args(%arg0 = %0) -> (tensor<?x?xf32>) {
///   %1 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
///   ...
/// }
/// ```
///
/// is folded to:
///
/// ```
/// %0 = ... : tensor<?x?xf32>
/// scf.for ... iter_args(%arg0 = %0) -> (tensor<?x?xf32>) {
///   %1 = tensor.dim %0, %c0 : tensor<?x?xf32>
///   ...
/// }
/// ```
template <typename OpTy>
struct DimOfIterArgFolder : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy dimOp,
                                PatternRewriter &rewriter) const override {
    auto blockArg = dimOp.source().template dyn_cast<BlockArgument>();
    if (!blockArg)
      return failure();
    auto forOp = dyn_cast<ForOp>(blockArg.getParentBlock()->getParentOp());
    if (!forOp)
      return failure();

    Value initArg = forOp.getOpOperandForRegionIterArg(blockArg).get();
    rewriter.updateRootInPlace(
        dimOp, [&]() { dimOp.sourceMutable().assign(initArg); });

    return success();
  };
};

/// Canonicalize AffineMinOp/AffineMaxOp operations in the context of scf.for
/// and scf.parallel loops with a known range.
template <typename OpTy, bool IsMin>
struct AffineOpSCFCanonicalizationPattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    auto loopMatcher = [](Value iv, Value &lb, Value &ub, Value &step) {
      if (scf::ForOp forOp = scf::getForInductionVarOwner(iv)) {
        lb = forOp.lowerBound();
        ub = forOp.upperBound();
        step = forOp.step();
        return success();
      }
      if (scf::ParallelOp parOp = scf::getParallelForInductionVarOwner(iv)) {
        for (unsigned idx = 0; idx < parOp.getNumLoops(); ++idx) {
          if (parOp.getInductionVars()[idx] == iv) {
            lb = parOp.lowerBound()[idx];
            ub = parOp.upperBound()[idx];
            step = parOp.step()[idx];
            return success();
          }
        }
        return failure();
      }
      return failure();
    };

    return scf::canonicalizeMinMaxOpInLoop(rewriter, op, op.getAffineMap(),
                                           op.operands(), IsMin, loopMatcher);
  }
};

struct SCFForLoopCanonicalization
    : public SCFForLoopCanonicalizationBase<SCFForLoopCanonicalization> {
  void runOnFunction() override {
    FuncOp funcOp = getFunction();
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(ctx);
    scf::populateSCFForLoopCanonicalizationPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void mlir::scf::populateSCFForLoopCanonicalizationPatterns(
    RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns
      .insert<AffineOpSCFCanonicalizationPattern<AffineMinOp, /*IsMin=*/true>,
              AffineOpSCFCanonicalizationPattern<AffineMaxOp, /*IsMin=*/false>,
              DimOfIterArgFolder<tensor::DimOp>,
              DimOfIterArgFolder<memref::DimOp>>(ctx);
}

std::unique_ptr<Pass> mlir::createSCFForLoopCanonicalizationPass() {
  return std::make_unique<SCFForLoopCanonicalization>();
}
