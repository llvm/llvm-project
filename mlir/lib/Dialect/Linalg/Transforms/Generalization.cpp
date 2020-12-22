//===- Generalization.cpp - linalg named ops to generic ops  --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Linalg generalization pass. It converts named
// Linalg ops to linalg.generic ops.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-generalization"

using namespace mlir;

// Creates a linalg.generic op from the given `namedOp`. Returns a null op if
// the given `namedOp` does not have a region builder.
static linalg::GenericOp createGenericOpFromNamedOp(linalg::LinalgOp namedOp,
                                                    OpBuilder &builder) {
  auto regionBuilder = namedOp.getRegionBuilder();
  if (!regionBuilder) {
    LLVM_DEBUG(llvm::dbgs() << "no region builder for op: " << namedOp << "\n");
    return nullptr;
  }

  SmallVector<AffineMap, 4> indexingMaps = namedOp.getIndexingMaps();
  auto iterators = llvm::to_vector<4>(
      namedOp.iterator_types().getAsValueRange<StringAttr>());
  auto resultTypes = namedOp.getOutputTensorTypes();
  SmallVector<Type, 4> types(resultTypes.begin(), resultTypes.end());

  return builder.create<linalg::GenericOp>(
      namedOp.getLoc(), types, namedOp.getInputs(), namedOp.getOutputs(),
      indexingMaps, iterators,
      [&regionBuilder](OpBuilder &bodyBuilder, Location loc, ValueRange) {
        edsc::ScopedContext scope(bodyBuilder, loc);
        regionBuilder(*bodyBuilder.getBlock());
      });
}

namespace {

/// Base class for all linalg generalization patterns. A subclass must provide
/// the following method:
///   linalg::GenericOp createGenericOp(RootOp, PatternRewriter &)
/// for creating the generic op.
// TODO: remove this pattern after migrating all manually-written named ops
// into auto-generated ones.
template <typename ConcretePattern, typename RootOp>
struct LinalgGeneralizationPattern : OpRewritePattern<RootOp> {
  LinalgGeneralizationPattern(MLIRContext *context, linalg::LinalgMarker marker,
                              PatternBenefit benefit = 1)
      : OpRewritePattern<RootOp>(context, benefit), marker(std::move(marker)) {}

  LogicalResult matchAndRewrite(RootOp rootOp,
                                PatternRewriter &rewriter) const override {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(rootOp.getOperation());
    if (!linalgOp)
      return failure();
    if (failed(marker.checkAndNotify(rewriter, linalgOp)))
      return failure();

    auto *pattern = static_cast<const ConcretePattern *>(this);
    linalg::GenericOp genericOp = pattern->createGenericOp(rootOp, rewriter);
    if (!genericOp)
      return failure();

    rewriter.replaceOp(rootOp, genericOp.getResults());
    marker.replaceLinalgMarker(rewriter, genericOp.getOperation());
    return success();
  }

private:
  linalg::LinalgMarker marker;
};

struct GeneralizeConvOp
    : public LinalgGeneralizationPattern<GeneralizeConvOp, linalg::ConvOp> {
  using LinalgGeneralizationPattern::LinalgGeneralizationPattern;

  linalg::GenericOp createGenericOp(linalg::ConvOp, OpBuilder &rewriter) const;
};

/// Catch-all pattern for converting all named ops with a region builder into
/// linalg.generic.
struct LinalgNamedOpGeneralizationPattern : RewritePattern {
  LinalgNamedOpGeneralizationPattern(MLIRContext *context,
                                     linalg::LinalgMarker marker,
                                     PatternBenefit benefit = 1)
      : RewritePattern(benefit, MatchAnyOpTypeTag()),
        marker(std::move(marker)) {}

  LogicalResult matchAndRewrite(Operation *rootOp,
                                PatternRewriter &rewriter) const override {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(rootOp);
    if (!linalgOp)
      return failure();
    if (failed(marker.checkAndNotify(rewriter, linalgOp)))
      return failure();

    // No nothing to do for linalg.generic and linalg.indexed_generic.
    if (isa<linalg::GenericOp, linalg::IndexedGenericOp>(rootOp))
      return failure();

    linalg::GenericOp genericOp =
        createGenericOpFromNamedOp(linalgOp, rewriter);
    if (!genericOp)
      return failure();

    rewriter.replaceOp(rootOp, genericOp.getResults());
    marker.replaceLinalgMarker(rewriter, genericOp.getOperation());
    return success();
  }

private:
  linalg::LinalgMarker marker;
};

struct LinalgGeneralizationPass
    : public LinalgGeneralizationBase<LinalgGeneralizationPass> {
  void runOnFunction() override;
};

} // namespace

void LinalgGeneralizationPass::runOnFunction() {
  FuncOp func = getFunction();
  OwningRewritePatternList patterns;
  linalg::populateLinalgConvGeneralizationPatterns(&getContext(), patterns);
  linalg::populateLinalgNamedOpsGeneralizationPatterns(&getContext(), patterns);
  applyPatternsAndFoldGreedily(func.getBody(), std::move(patterns));
}

linalg::GenericOp GeneralizeConvOp::createGenericOp(linalg::ConvOp convOp,
                                                    OpBuilder &builder) const {
  SmallVector<AffineMap, 4> indexingMaps = convOp.getIndexingMaps();
  auto iterators =
      llvm::to_vector<4>(convOp.iterator_types().getAsValueRange<StringAttr>());
  return builder.create<linalg::GenericOp>(
      convOp.getLoc(), /*resultTensorTypes=*/ArrayRef<Type>(),
      convOp.getInputBuffers(), convOp.getOutputBuffers(), indexingMaps,
      iterators,
      [](OpBuilder &bodyBuilder, Location bodyLoc, ValueRange bodyArgs) {
        Value mul =
            bodyBuilder.create<MulFOp>(bodyLoc, bodyArgs[0], bodyArgs[1]);
        Value add = bodyBuilder.create<AddFOp>(bodyLoc, mul, bodyArgs[2]);
        bodyBuilder.create<linalg::YieldOp>(bodyLoc, add);
      });
}

void mlir::linalg::populateLinalgConvGeneralizationPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns,
    linalg::LinalgMarker marker) {
  patterns.insert<GeneralizeConvOp>(context, marker);
}

void mlir::linalg::populateLinalgNamedOpsGeneralizationPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns,
    linalg::LinalgMarker marker) {
  patterns.insert<LinalgNamedOpGeneralizationPattern>(context, marker);
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createLinalgGeneralizationPass() {
  return std::make_unique<LinalgGeneralizationPass>();
}
