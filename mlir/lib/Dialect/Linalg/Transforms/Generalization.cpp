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
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-generalization"

using namespace mlir;
using namespace mlir::linalg;

// Creates a linalg.generic op from the given `namedOp`. Returns a null op if
// the given `namedOp` does not have a region builder.
static GenericOp createGenericOpFromNamedOp(LinalgOp namedOp,
                                            PatternRewriter &rewriter) {
  SmallVector<Value> inputOperands = namedOp.getInputOperands();
  SmallVector<Value> outputOperands = namedOp.getOutputOperands();
  SmallVector<AffineMap> indexingMaps = namedOp.getIndexingMaps();
  SmallVector<StringRef> iterators = llvm::to_vector<4>(
      namedOp.iterator_types().getAsValueRange<StringAttr>());
  SmallVector<RankedTensorType> resultTypes = namedOp.getOutputTensorTypes();
  SmallVector<Type> types(resultTypes.begin(), resultTypes.end());

  // Inline the existing region if the named operation has a region attached.
  if (namedOp->getNumRegions() == 1) {
    GenericOp genericOp =
        rewriter.create<GenericOp>(namedOp.getLoc(), types, inputOperands,
                                   outputOperands, indexingMaps, iterators);
    rewriter.inlineRegionBefore(namedOp->getRegion(0), genericOp.region(),
                                genericOp.region().begin());
    return genericOp;
  }

  // Otherwise use the region builder to generate a new region.
  // TODO: Remove this path once all linag operations have a region attached.
  auto regionBuilder = namedOp.getRegionBuilder();
  if (!regionBuilder) {
    LLVM_DEBUG(llvm::dbgs() << "no region builder for op: " << namedOp << "\n");
    return nullptr;
  }
  return rewriter.create<GenericOp>(
      namedOp.getLoc(), types, inputOperands, outputOperands, indexingMaps,
      iterators,
      [&regionBuilder](OpBuilder &bodyBuilder, Location loc, ValueRange) {
        ImplicitLocOpBuilder b(loc, bodyBuilder);
        regionBuilder(b, *bodyBuilder.getBlock());
      });
}

namespace {

/// Base class for all linalg generalization patterns. A subclass must provide
/// the following method:
///   GenericOp createGenericOp(RootOp, PatternRewriter &)
/// for creating the generic op.
// TODO: remove this pattern after migrating all manually-written named ops
// into auto-generated ones.
template <typename ConcretePattern, typename RootOp>
struct LinalgGeneralizationPattern : OpRewritePattern<RootOp> {
  LinalgGeneralizationPattern(MLIRContext *context,
                              LinalgTransformationFilter marker,
                              PatternBenefit benefit = 1)
      : OpRewritePattern<RootOp>(context, benefit), marker(std::move(marker)) {}

  LogicalResult matchAndRewrite(RootOp rootOp,
                                PatternRewriter &rewriter) const override {
    auto linalgOp = dyn_cast<LinalgOp>(rootOp.getOperation());
    if (!linalgOp)
      return failure();
    if (failed(marker.checkAndNotify(rewriter, linalgOp)))
      return failure();

    auto *pattern = static_cast<const ConcretePattern *>(this);
    GenericOp genericOp = pattern->createGenericOp(rootOp, rewriter);
    if (!genericOp)
      return failure();

    rewriter.replaceOp(rootOp, genericOp.getResults());
    marker.replaceLinalgTransformationFilter(rewriter,
                                             genericOp.getOperation());
    return success();
  }

private:
  LinalgTransformationFilter marker;
};

struct GeneralizeConvOp
    : public LinalgGeneralizationPattern<GeneralizeConvOp, ConvOp> {
  using LinalgGeneralizationPattern::LinalgGeneralizationPattern;

  GenericOp createGenericOp(ConvOp convOp, OpBuilder &builder) const;
};

/// Catch-all pattern for converting all named ops with a region builder into
/// linalg.generic.
struct LinalgNamedOpGeneralizationPattern : RewritePattern {
  LinalgNamedOpGeneralizationPattern(MLIRContext *context,
                                     LinalgTransformationFilter marker,
                                     PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context),
        marker(std::move(marker)) {}

  LogicalResult matchAndRewrite(Operation *rootOp,
                                PatternRewriter &rewriter) const override {
    auto linalgOp = dyn_cast<LinalgOp>(rootOp);
    if (!linalgOp)
      return failure();
    if (failed(marker.checkAndNotify(rewriter, linalgOp)))
      return failure();

    // No nothing to do for linalg.generic.
    if (isa<GenericOp>(rootOp))
      return failure();

    GenericOp genericOp = createGenericOpFromNamedOp(linalgOp, rewriter);
    if (!genericOp)
      return failure();

    rewriter.replaceOp(rootOp, genericOp.getResults());
    marker.replaceLinalgTransformationFilter(rewriter,
                                             genericOp.getOperation());
    return success();
  }

private:
  LinalgTransformationFilter marker;
};

struct LinalgGeneralizationPass
    : public LinalgGeneralizationBase<LinalgGeneralizationPass> {
  void runOnFunction() override;
};

} // namespace

void LinalgGeneralizationPass::runOnFunction() {
  FuncOp func = getFunction();
  RewritePatternSet patterns(&getContext());
  populateLinalgConvGeneralizationPatterns(patterns);
  populateLinalgNamedOpsGeneralizationPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(func.getBody(), std::move(patterns));
}

GenericOp GeneralizeConvOp::createGenericOp(ConvOp convOp,
                                            OpBuilder &builder) const {
  SmallVector<AffineMap> indexingMaps = convOp.getIndexingMaps();
  auto iterators =
      llvm::to_vector<4>(convOp.iterator_types().getAsValueRange<StringAttr>());
  SmallVector<Value> inputBuffers = convOp.getInputBufferOperands();
  SmallVector<Value> outputBuffers = convOp.getOutputBufferOperands();
  return builder.create<GenericOp>(
      convOp.getLoc(), /*resultTensorTypes=*/ArrayRef<Type>(), inputBuffers,
      outputBuffers, indexingMaps, iterators,
      [](OpBuilder &bodyBuilder, Location bodyLoc, ValueRange bodyArgs) {
        Value mul =
            bodyBuilder.create<MulFOp>(bodyLoc, bodyArgs[0], bodyArgs[1]);
        Value add = bodyBuilder.create<AddFOp>(bodyLoc, mul, bodyArgs[2]);
        bodyBuilder.create<YieldOp>(bodyLoc, add);
      });
}

void mlir::linalg::populateLinalgConvGeneralizationPatterns(
    RewritePatternSet &patterns, LinalgTransformationFilter marker) {
  patterns.add<GeneralizeConvOp>(patterns.getContext(), marker);
}

void mlir::linalg::populateLinalgNamedOpsGeneralizationPatterns(
    RewritePatternSet &patterns, LinalgTransformationFilter marker) {
  patterns.add<LinalgNamedOpGeneralizationPattern>(patterns.getContext(),
                                                   marker);
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createLinalgGeneralizationPass() {
  return std::make_unique<LinalgGeneralizationPass>();
}
