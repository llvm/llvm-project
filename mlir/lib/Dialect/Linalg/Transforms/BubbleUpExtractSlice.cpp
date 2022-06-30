//===- BubbleUpExtractSlice.cpp - bubble up tensor.extract_slice ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns that transforms linalg.<op> +
// tensor.extract_slice into tensor.extract_slice + linalg.<op> to reduce
// the computation for the linalg op.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {
/// Bubble up extract_slice above Linalg operation.
///
/// A sequence of operations
///
/// ```mlir
/// %0 = linalg.<op> ... arg0, arg1, ...
/// %1 = tensor.extract_slice %0 ...
/// ```
///
/// can be replaced with
///
/// ```mlir
/// %0 = tensor.extract_slice %arg0
/// %1 = tensor.extract_slice %arg1
/// %2 = linalg.<op> ... %0, %1, ...
/// ```
///
/// This results in the reduce computation of the linalg operation.
///
struct BubbleUpExtractSliceOpPattern
    : OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const final {
    Value source = sliceOp.getSource();
    auto linalgOp = source.getDefiningOp<LinalgOp>();
    if (!linalgOp) {
      return rewriter.notifyMatchFailure(sliceOp,
                                         "expected source to be linalg op");
    }

    // TODO: we might relax this if we want heuristics to detect that all uses
    // are small portion of the output.
    if (!linalgOp->hasOneUse()) {
      return rewriter.notifyMatchFailure(sliceOp,
                                         "expected single use of linalg op");
    }

    if (linalgOp.getNumOutputs() != 1) {
      return rewriter.notifyMatchFailure(sliceOp,
                                         "expected single output of linalg op");
    }

    if (!linalgOp.hasTensorSemantics()) {
      return rewriter.notifyMatchFailure(sliceOp,
                                         "expected tensor of linalg op");
    }

    if (!sliceOp.hasUnitStride())
      return rewriter.notifyMatchFailure(sliceOp, "expected unit stride");

    if (sliceOp.getType().getRank() != sliceOp.getSourceType().getRank()) {
      return rewriter.notifyMatchFailure(sliceOp, "expected no rank reduction");
    }

    OpOperand *outOperand = linalgOp.getOutputOperand(0);
    AffineMap indexingMap = linalgOp.getTiedIndexingMap(outOperand);
    if (!indexingMap.isProjectedPermutation()) {
      return rewriter.notifyMatchFailure(
          sliceOp, "expected a projected permutation for output");
    }

    auto linalgLoc = linalgOp.getLoc();
    auto allShapeSizes =
        linalgOp.createFlatListOfOperandDims(rewriter, linalgLoc);
    AffineMap shapeSizesToLoopsMap = linalgOp.getShapesToLoopsMap();
    if (!shapeSizesToLoopsMap) {
      return rewriter.notifyMatchFailure(
          linalgOp, "failed to get loops map from shape sizes");
    }
    auto sizeBounds = applyMapToValues(rewriter, linalgLoc,
                                       shapeSizesToLoopsMap, allShapeSizes);

    auto sliceLoc = sliceOp.getLoc();
    auto offsetVals = getValueOrCreateConstantIndexOp(
        rewriter, sliceLoc, sliceOp.getMixedOffsets());
    auto sizeVals = getValueOrCreateConstantIndexOp(rewriter, sliceLoc,
                                                    sliceOp.getMixedSizes());

    // The offsets and sizes from the slice operation only give you the tile
    // size of the output. Use that compute the tile sizes and offsets of the
    // loops. For loops not used to access the output, set the tile sizes to
    // loop bounds and set the offset to 0.
    Value zero = rewriter.create<arith::ConstantIndexOp>(linalgLoc, 0);
    SmallVector<Value, 4> tileOffsets(sizeBounds.size(), zero);
    SmallVector<Value, 4> tileSizes = sizeBounds;
    for (auto const &result : enumerate(indexingMap.getResults())) {
      unsigned position = result.value().cast<AffineDimExpr>().getPosition();
      tileOffsets[position] = offsetVals[result.index()];
      tileSizes[position] = sizeVals[result.index()];
    }

    SmallVector<Value> valuesToTile = linalgOp.getInputAndOutputOperands();

    SmallVector<Value, 4> tiledOperands = makeTiledShapes(
        rewriter, linalgLoc, linalgOp, valuesToTile, tileOffsets, tileSizes,
        sizeBounds, /*omitPartialTileCheck=*/true);

    SmallVector<Type, 4> resultTensorTypes;
    for (OpOperand *opOperand : linalgOp.getOutputTensorOperands())
      resultTensorTypes.push_back(
          tiledOperands[opOperand->getOperandNumber()].getType());

    Operation *newOp =
        linalgOp.clone(rewriter, linalgLoc, resultTensorTypes, tiledOperands);
    rewriter.replaceOp(sliceOp, newOp->getResults());
    return success();
  }
};
} // namespace

void mlir::linalg::populateBubbleUpExtractSliceOpPatterns(
    RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  patterns.add<BubbleUpExtractSliceOpPattern>(context);
}
