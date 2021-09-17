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
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::scf;

/// A simple, conservative analysis to determine if the loop is shape
/// conserving. I.e., the type of the arg-th yielded value is the same as the
/// type of the corresponding basic block argument of the loop.
/// Note: This function handles only simple cases. Expand as needed.
static bool isShapePreserving(ForOp forOp, int64_t arg) {
  auto yieldOp = cast<YieldOp>(forOp.getBody()->getTerminator());
  assert(arg < static_cast<int64_t>(yieldOp.results().size()) &&
         "arg is out of bounds");
  Value value = yieldOp.results()[arg];
  while (value) {
    if (value == forOp.getRegionIterArgs()[arg])
      return true;
    OpResult opResult = value.dyn_cast<OpResult>();
    if (!opResult)
      return false;

    using tensor::InsertSliceOp;
    value =
        llvm::TypeSwitch<Operation *, Value>(opResult.getOwner())
            .template Case<InsertSliceOp>(
                [&](InsertSliceOp op) { return op.dest(); })
            .template Case<ForOp>([&](ForOp forOp) {
              return isShapePreserving(forOp, opResult.getResultNumber())
                         ? forOp.getIterOperands()[opResult.getResultNumber()]
                         : Value();
            })
            .Default([&](auto op) { return Value(); });
  }
  return false;
}

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
///
/// Note: Dim ops are folded only if it can be proven that the runtime type of
/// the iter arg does not change with loop iterations.
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
    if (!isShapePreserving(forOp, blockArg.getArgNumber() - 1))
      return failure();

    Value initArg = forOp.getOpOperandForRegionIterArg(blockArg).get();
    rewriter.updateRootInPlace(
        dimOp, [&]() { dimOp.sourceMutable().assign(initArg); });

    return success();
  };
};

/// Fold dim ops of loop results to dim ops of their respective init args. E.g.:
///
/// ```
/// %0 = ... : tensor<?x?xf32>
/// %r = scf.for ... iter_args(%arg0 = %0) -> (tensor<?x?xf32>) {
///   ...
/// }
/// %1 = tensor.dim %r, %c0 : tensor<?x?xf32>
/// ```
///
/// is folded to:
///
/// ```
/// %0 = ... : tensor<?x?xf32>
/// %r = scf.for ... iter_args(%arg0 = %0) -> (tensor<?x?xf32>) {
///   ...
/// }
/// %1 = tensor.dim %0, %c0 : tensor<?x?xf32>
/// ```
///
/// Note: Dim ops are folded only if it can be proven that the runtime type of
/// the iter arg does not change with loop iterations.
template <typename OpTy>
struct DimOfLoopResultFolder : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy dimOp,
                                PatternRewriter &rewriter) const override {
    auto forOp = dimOp.source().template getDefiningOp<scf::ForOp>();
    if (!forOp)
      return failure();
    auto opResult = dimOp.source().template cast<OpResult>();
    unsigned resultNumber = opResult.getResultNumber();
    if (!isShapePreserving(forOp, resultNumber))
      return failure();
    rewriter.updateRootInPlace(dimOp, [&](){
      dimOp.sourceMutable().assign(forOp.getIterOperands()[resultNumber]);
    });
    return success();
  }
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
              DimOfIterArgFolder<memref::DimOp>,
              DimOfLoopResultFolder<tensor::DimOp>,
              DimOfLoopResultFolder<memref::DimOp>>(ctx);
}

std::unique_ptr<Pass> mlir::createSCFForLoopCanonicalizationPass() {
  return std::make_unique<SCFForLoopCanonicalization>();
}
