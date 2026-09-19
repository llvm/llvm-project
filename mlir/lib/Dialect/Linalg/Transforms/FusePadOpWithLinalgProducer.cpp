//===- FusePadOpWithLinalgProducer.cpp ---- Fuse pad with linalg producer -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns that fuses a linalg.generic -> tensor.pad op
// chain into a tensor.extract_slice -> linalg.generic -> tensor.insert_slice
// op chain.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

using namespace mlir;

/// Fill the padding in disjoint slabs. After filling both sides of a dimension,
/// restrict it to the interior so subsequent slabs do not overlap.
static Value fillPaddedBoundary(RewriterBase &rewriter, Location loc,
                                Value dest, Value padValue,
                                ArrayRef<int64_t> lowPad,
                                ArrayRef<int64_t> sourceShape) {
  ArrayRef<int64_t> paddedShape =
      cast<RankedTensorType>(dest.getType()).getShape();
  SmallVector<int64_t> offsets(paddedShape.size(), 0);
  SmallVector<int64_t> sizes(paddedShape);
  SmallVector<OpFoldResult> strides(paddedShape.size(),
                                    rewriter.getIndexAttr(1));

  auto fillSlab = [&](unsigned dim, int64_t offset, int64_t size) {
    offsets[dim] = offset;
    sizes[dim] = size;
    if (llvm::is_contained(sizes, 0))
      return;
    auto sliceOffsets = getAsIndexOpFoldResult(rewriter.getContext(), offsets);
    auto sliceSizes = getAsIndexOpFoldResult(rewriter.getContext(), sizes);
    auto slice = tensor::ExtractSliceOp::create(
        rewriter, loc, dest, sliceOffsets, sliceSizes, strides);
    auto filled =
        linalg::FillOp::create(rewriter, loc, padValue, slice.getResult());
    dest =
        tensor::InsertSliceOp::create(rewriter, loc, filled.getResult(0), dest,
                                      sliceOffsets, sliceSizes, strides);
  };

  for (auto [dim, sourceSize] : llvm::enumerate(sourceShape)) {
    int64_t highOffset = lowPad[dim] + sourceSize;
    fillSlab(dim, 0, lowPad[dim]);
    fillSlab(dim, highOffset, paddedShape[dim] - highOffset);
    offsets[dim] = lowPad[dim];
    sizes[dim] = sourceSize;
  }
  return dest;
}

static bool canLeaveInteriorUninitialized(linalg::GenericOp linalgOp,
                                          OpOperand *initOperand,
                                          tensor::PadOp padOp) {
  if (!cast<RankedTensorType>(padOp.getSource().getType()).hasStaticShape() ||
      !padOp.getResultType().hasStaticShape() ||
      llvm::any_of(padOp.getStaticLow(), ShapedType::isDynamic))
    return false;

  if (linalgOp.payloadUsesValueFromOperand(initOperand))
    return false;

  // A projected permutation is not enough: it may drop a loop dimension, e.g.
  // `(d0, d1) -> (d0)`, and if that dimension has zero extent the producer runs
  // no iterations at all and leaves the interior uninitialized.
  return linalgOp.getMatchingIndexingMap(initOperand).isPermutation();
}

namespace {

/// A sequence of operations
///
/// ```mlir
/// %0 = linalg. ...
/// %1 = tensor.pad %0 ...
/// ```
///
/// can be replaced with
///
/// ```mlir
/// %0 = linalg.fill
/// %1 = tensor.extract_slice %0 ...
/// %2 = linalg. .... outs(..., %1, ....) ....
/// %3 = tensor.insert_slice %2 into %1 ...
/// ```
///
/// if the `linalg.generic` has all parallel iterator types.
struct FusePadOp : OpRewritePattern<tensor::PadOp> {
  FusePadOp(MLIRContext *context, bool fillBoundaryOnly,
            PatternBenefit benefit = 1)
      : OpRewritePattern<tensor::PadOp>(context, benefit),
        fillBoundaryOnly(fillBoundaryOnly) {}

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    // Only works on padding op that sets the padded value to a constant.
    Value padValue = padOp.getConstantPaddingValue();
    if (!padValue)
      return rewriter.notifyMatchFailure(padOp, "non constant padding");

    // This pattern could work for any Linalg op. For now restrict it to generic
    // ops.
    Value source = padOp.getSource();
    auto linalgOp = source.getDefiningOp<linalg::GenericOp>();
    if (!linalgOp) {
      return rewriter.notifyMatchFailure(
          padOp, "expected source to be linalg.generic op");
    }
    // All iterator types need to be parallel.
    if (linalgOp.getNumLoops() != linalgOp.getNumParallelLoops()) {
      return rewriter.notifyMatchFailure(
          padOp, "only supported for ops with all parallel iterator types");
    }
    ReifiedRankedShapedTypeDims resultShape;
    if (failed(reifyResultShapes(rewriter, padOp, resultShape)) ||
        resultShape.size() != 1) {
      return rewriter.notifyMatchFailure(
          padOp, "failed to get shape of pad op result");
    }

    Location loc = padOp.getLoc();

    // Create the tensor of same size as output of the pad op.
    RankedTensorType padResultType = padOp.getResultType();
    auto resultSizes = resultShape[0];
    auto emptyTensor = tensor::EmptyOp::create(rewriter, loc, resultSizes,
                                               padResultType.getElementType());

    unsigned resultNumber = cast<OpResult>(source).getResultNumber();
    auto sourceType = cast<RankedTensorType>(source.getType());

    Value fillTensor;
    if (fillBoundaryOnly &&
        canLeaveInteriorUninitialized(
            linalgOp, linalgOp.getDpsInitOperand(resultNumber), padOp)) {
      fillTensor =
          fillPaddedBoundary(rewriter, loc, emptyTensor.getResult(), padValue,
                             padOp.getStaticLow(), sourceType.getShape());
    } else {
      fillTensor = linalg::FillOp::create(rewriter, loc, padValue,
                                          emptyTensor.getResult())
                       .getResult(0);
    }

    // Construct a slice of the fill result that is to be replaced with the
    // result of the generic op. The low pad values are the offsets, the size of
    // the source is the size of the slice.
    // TODO: This insert/extract could be potentially made a utility method.
    SmallVector<OpFoldResult> offsets = padOp.getMixedLowPad();
    SmallVector<OpFoldResult> sizes;
    sizes.reserve(offsets.size());
    for (const auto &shape : llvm::enumerate(sourceType.getShape())) {
      if (ShapedType::isDynamic(shape.value())) {
        sizes.push_back(
            tensor::DimOp::create(rewriter, loc, source, shape.index())
                .getResult());
      } else {
        sizes.push_back(rewriter.getIndexAttr(shape.value()));
      }
    }
    SmallVector<OpFoldResult> strides(offsets.size(), rewriter.getIndexAttr(1));
    auto slice = tensor::ExtractSliceOp::create(rewriter, loc, fillTensor,
                                                offsets, sizes, strides);

    // Clone the generic op.
    auto clonedOp =
        cast<linalg::GenericOp>(rewriter.clone(*linalgOp.getOperation()));
    clonedOp.setDpsInitOperand(resultNumber, slice.getResult());

    // Insert it back into the result of the fill.
    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
        padOp, clonedOp.getResult(resultNumber), fillTensor, offsets, sizes,
        strides);
    return success();
  }

private:
  bool fillBoundaryOnly;
};
} // namespace

void mlir::linalg::populateFuseTensorPadWithProducerLinalgOpPatterns(
    RewritePatternSet &patterns, bool fillBoundaryOnly) {
  patterns.add<FusePadOp>(patterns.getContext(), fillBoundaryOnly);
}
