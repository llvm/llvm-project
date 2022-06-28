//===- PadOpInterchange.cpp - Interchange tensor.pad with linalg producer -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns that intechanges a linalg.generic -> tensor.pad
// op chain into a tensor.extract_slice -> linalg.generic -> tensor.insert_slice
// op chain.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

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
  using OpRewritePattern<tensor::PadOp>::OpRewritePattern;

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
    ReifyRankedShapedTypeOpInterface reifyShapedTypeInterface =
        dyn_cast<ReifyRankedShapedTypeOpInterface>(padOp.getOperation());
    if (failed(reifyShapedTypeInterface.reifyResultShapes(rewriter,
                                                          resultShape)) ||
        resultShape.size() != 1) {
      return rewriter.notifyMatchFailure(
          padOp, "failed to get shape of pad op result");
    }

    Location loc = padOp.getLoc();

    // Create the tensor of same size as output of the pad op.
    RankedTensorType padResultType = padOp.getResultType();
    auto resultSizes = getAsOpFoldResult(resultShape[0]);
    auto initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, resultSizes, padResultType.getElementType());

    // Fill the tensor with the pad value.
    // TODO: There is an option to fill only the boundaries. For now just
    // filling the whole tensor.
    auto fillTensor =
        rewriter.create<linalg::FillOp>(loc, padValue, initTensor.getResult());

    // Construct a slice of the fill result that is to be replaced with the
    // result of the generic op. The low pad values are the offsets, the size of
    // the source is the size of the slice.
    // TODO: This insert/extract could be potentially made a utility method.
    unsigned resultNumber = source.cast<OpResult>().getResultNumber();
    SmallVector<OpFoldResult> offsets = padOp.getMixedLowPad();
    SmallVector<OpFoldResult> sizes;
    sizes.reserve(offsets.size());
    for (const auto &shape : llvm::enumerate(
             source.getType().cast<RankedTensorType>().getShape())) {
      if (ShapedType::isDynamic(shape.value())) {
        sizes.push_back(
            rewriter.create<tensor::DimOp>(loc, source, shape.index())
                .getResult());
      } else {
        sizes.push_back(rewriter.getIndexAttr(shape.value()));
      }
    }
    SmallVector<OpFoldResult> strides(offsets.size(), rewriter.getIndexAttr(1));
    auto slice = rewriter.create<tensor::ExtractSliceOp>(
        loc, fillTensor.getResult(0), offsets, sizes, strides);

    // Clone the generic op.
    auto clonedOp =
        cast<linalg::GenericOp>(rewriter.clone(*linalgOp.getOperation()));
    clonedOp.setOutputOperand(resultNumber, slice.getResult());

    // Insert it back into the result of the fill.
    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
        padOp, clonedOp.getResult(resultNumber), fillTensor.getResult(0),
        offsets, sizes, strides);
    return success();
  }
};
} // namespace

void mlir::linalg::populateFuseTensorPadWithProducerLinalgOpPatterns(
    RewritePatternSet &patterns) {
  patterns.add<FusePadOp>(patterns.getContext());
}
