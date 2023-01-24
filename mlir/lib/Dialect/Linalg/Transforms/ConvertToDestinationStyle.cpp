//===- ConvertToDestinationStyle.cpp - Convert non-DPS to DPS ops ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains patterns to convert non-DPS ops to DPS ops. New
// tensor.empty ops are inserted as a destination. Such tensor.empty can be
// eliminated with "empty tensor elimination", allowing them to bufferize
// without an allocation (assuming there are no further conflicts).
//
//===----------------------------------------------------------------------===//
//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::tensor;

namespace {

/// Lower tensor.generate to linalg.generic.
struct GenerateOpConverter : public OpRewritePattern<GenerateOp> {
  using OpRewritePattern<GenerateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenerateOp generateOp,
                                PatternRewriter &rewriter) const override {
    // Only ops with exactly one block are supported.
    if (!generateOp.getBody().hasOneBlock())
      return failure();

    Location loc = generateOp.getLoc();
    RankedTensorType tensorType = generateOp.getType().cast<RankedTensorType>();

    // Create tensor.empty.
    auto emptyOp = rewriter.create<EmptyOp>(loc, tensorType,
                                            generateOp.getDynamicExtents());

    // Create linalg.generic.
    SmallVector<utils::IteratorType> iteratorTypes(
        tensorType.getRank(), utils::IteratorType::parallel);
    SmallVector<AffineMap> indexingMaps(
        1, rewriter.getMultiDimIdentityMap(tensorType.getRank()));
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, tensorType, /*inputs=*/ValueRange(),
        /*outputs=*/ValueRange{emptyOp.getResult()}, /*indexingMaps=*/
        indexingMaps, iteratorTypes);
    Block *body = rewriter.createBlock(&genericOp->getRegion(0), {},
                                       tensorType.getElementType(), loc);
    rewriter.setInsertionPointToStart(body);
    SmallVector<Value> bbArgReplacements;
    for (int64_t i = 0; i < tensorType.getRank(); ++i)
      bbArgReplacements.push_back(rewriter.create<linalg::IndexOp>(loc, i));
    rewriter.mergeBlocks(&generateOp.getBody().front(), body,
                         bbArgReplacements);

    // Update terminator.
    auto yieldOp = cast<tensor::YieldOp>(body->getTerminator());
    rewriter.replaceOpWithNewOp<linalg::YieldOp>(yieldOp, yieldOp.getValue());

    // Replace tensor.generate.
    rewriter.replaceOp(generateOp, genericOp->getResult(0));
    return success();
  }
};

/// Lower tensor.pad to linalg.generic + tensor.insert_slice.
struct PadOpConverter : public OpRewritePattern<PadOp> {
  using OpRewritePattern<PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PadOp padOp,
                                PatternRewriter &rewriter) const override {
    // Only ops with exactly one block are supported.
    if (!padOp.getBodyRegion().hasOneBlock())
      return failure();

    // Create tensor.empty.
    Location loc = padOp.getLoc();
    RankedTensorType resultType = padOp.getResultType();
    ReifiedRankedShapedTypeDims reifiedShape;
    if (failed(cast<ReifyRankedShapedTypeOpInterface>(padOp.getOperation())
                   .reifyResultShapes(rewriter, reifiedShape)))
      return rewriter.notifyMatchFailure(
          padOp, "failed to reify tensor.pad op result shape");
    SmallVector<Value> dynamicSizes;
    for (int64_t i = 0; i < resultType.getRank(); ++i)
      if (resultType.isDynamicDim(i))
        dynamicSizes.push_back(reifiedShape[0][i]);
    auto emptyOp = rewriter.create<EmptyOp>(loc, resultType, dynamicSizes);

    // Examine the yielded value to decide if a linalg.generic is neede or a
    // linalg.fill is sufficient.
    Value filled;
    Value yieldedValue =
        cast<tensor::YieldOp>(padOp.getBody()->getTerminator()).getValue();
    Attribute constYieldedValue;
    // Is the yielded value a bbArg defined outside of the PadOp?
    bool outsideBbArg =
        yieldedValue.isa<BlockArgument>() &&
        yieldedValue.cast<BlockArgument>().getOwner()->getParentOp() !=
            padOp.getOperation();
    // Is the yielded value an OpResult defined outside of the PadOp?
    bool outsideOpResult =
        yieldedValue.isa<OpResult>() &&
        yieldedValue.getDefiningOp()->getParentOp() != padOp.getOperation();
    bool invariantYieldedValue = outsideBbArg || outsideOpResult;
    if (matchPattern(yieldedValue, m_Constant(&constYieldedValue))) {
      // Padding with a constant: Create linalg.fill.
      Dialect *arithDialect =
          rewriter.getContext()->getLoadedDialect<arith::ArithDialect>();
      Value fillValue = arithDialect
                            ->materializeConstant(rewriter, constYieldedValue,
                                                  yieldedValue.getType(),
                                                  yieldedValue.getLoc())
                            ->getResult(0);
      auto fillOp = rewriter.create<linalg::FillOp>(
          loc, ValueRange(fillValue), ValueRange(emptyOp.getResult()));
      rewriter.setInsertionPointAfter(fillOp);
      filled = fillOp.getResult(0);
    } else if (invariantYieldedValue) {
      // Padding with an invariant value.
      auto fillOp = rewriter.create<linalg::FillOp>(
          loc, ValueRange(yieldedValue), ValueRange(emptyOp.getResult()));
      rewriter.setInsertionPointAfter(fillOp);
      filled = fillOp.getResult(0);
    } else {
      // Create linalg.generic.
      SmallVector<utils::IteratorType> iteratorTypes(
          resultType.getRank(), utils::IteratorType::parallel);
      SmallVector<AffineMap> indexingMaps(
          1, rewriter.getMultiDimIdentityMap(resultType.getRank()));
      auto genericOp = rewriter.create<linalg::GenericOp>(
          loc, resultType, /*inputs=*/ValueRange(),
          /*outputs=*/ValueRange{emptyOp.getResult()}, /*indexingMaps=*/
          indexingMaps, iteratorTypes);
      Block *body = rewriter.createBlock(&genericOp->getRegion(0), {},
                                         resultType.getElementType(), loc);
      rewriter.setInsertionPointToStart(body);
      SmallVector<Value> bbArgReplacements;
      for (int64_t i = 0; i < resultType.getRank(); ++i)
        bbArgReplacements.push_back(rewriter.create<linalg::IndexOp>(loc, i));
      rewriter.mergeBlocks(padOp.getBody(), body, bbArgReplacements);

      // Update terminator.
      auto yieldOp = cast<tensor::YieldOp>(body->getTerminator());
      rewriter.replaceOpWithNewOp<linalg::YieldOp>(yieldOp, yieldOp.getValue());
      rewriter.setInsertionPointAfter(genericOp);
      filled = genericOp->getResult(0);
    }

    // Create tensor::InsertSliceOp.
    SmallVector<OpFoldResult> sliceSizes =
        getMixedSizes(rewriter, loc, padOp.getSource());
    SmallVector<OpFoldResult> sliceStrides(resultType.getRank(),
                                           rewriter.getIndexAttr(1));
    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
        padOp, padOp.getSource(), filled,
        /*offsets=*/padOp.getMixedLowPad(), sliceSizes, sliceStrides);

    return success();
  }
};

} // namespace

void linalg::populateConvertToDestinationStylePatterns(
    RewritePatternSet &patterns) {
  patterns.insert<GenerateOpConverter, PadOpConverter>(patterns.getContext());
}
