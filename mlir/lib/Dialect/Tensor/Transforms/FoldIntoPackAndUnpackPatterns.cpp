//===- FoldIntoPackAndUnpackPatterns.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace tensor {
namespace {

static bool areAllConstantIntValue(ArrayRef<OpFoldResult> ofrs, int64_t value) {
  return llvm::all_of(
      ofrs, [&](OpFoldResult ofr) { return isConstantIntValue(ofr, value); });
}

/// Fold a `pad` -> `pack` into `pack` if they have the same padding values and
/// the pad op has zero low paddings, or if `pack` has no padding values.
struct FoldPadWithPackOp : public OpRewritePattern<PackOp> {
  using OpRewritePattern<PackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PackOp packOp,
                                PatternRewriter &rewriter) const override {
    auto padOp = packOp.getSource().getDefiningOp<PadOp>();

    if (!padOp || padOp.getNofold() || !padOp.hasZeroLowPad())
      return failure();

    Value constantPaddingValue = padOp.getConstantPaddingValue();
    if (!constantPaddingValue)
      return failure();

    if (auto paddingValue = packOp.getPaddingValue())
      if (!isEqualConstantIntOrValue(paddingValue, constantPaddingValue))
        return failure();

    rewriter.replaceOpWithNewOp<PackOp>(
        packOp, padOp.getSource(), packOp.getDest(), packOp.getInnerDimsPos(),
        packOp.getMixedTiles(), constantPaddingValue,
        packOp.getOuterDimsPerm());
    return success();
  }
};

/// Fold a `unpack` -> `extract_slice` into the `unpack` since it already
/// has extract_slice semantics.
struct FoldUnpackWithExtractSliceOp : public OpRewritePattern<ExtractSliceOp> {
  using OpRewritePattern<ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    auto unpackOp = sliceOp.getSource().getDefiningOp<UnPackOp>();
    if (!unpackOp)
      return failure();

    // Check all offsets are zeros, and all strides are ones.
    if (!areAllConstantIntValue(sliceOp.getMixedOffsets(), 0) ||
        !areAllConstantIntValue(sliceOp.getMixedStrides(), 1)) {
      return rewriter.notifyMatchFailure(
          sliceOp, "expects offsets to be 0s and strides to be 1s");
    }

    // Create a new empty output tensor.
    Type elementType = unpackOp.getDestType().getElementType();
    Value output = rewriter.create<EmptyOp>(
        sliceOp.getLoc(), sliceOp.getMixedSizes(), elementType);
    rewriter.replaceOpWithNewOp<UnPackOp>(
        sliceOp, unpackOp.getSource(), output, unpackOp.getInnerDimsPos(),
        unpackOp.getMixedTiles(), unpackOp.getOuterDimsPerm());
    return success();
  }
};
} // namespace

void populateFoldIntoPackAndUnpackPatterns(RewritePatternSet &patterns) {
  patterns.insert<FoldUnpackWithExtractSliceOp, FoldPadWithPackOp>(
      patterns.getContext());
}

} // namespace tensor
} // namespace mlir
