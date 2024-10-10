//===- EliminateWholeSlicePatterns.cpp - Patterns to remove whole slices --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::tensor;

namespace {

bool checkEliminateOK(PatternRewriter &rewriter,
                      OffsetSizeAndStrideOpInterface sliceOp,
                      mlir::TypedValue<mlir::RankedTensorType> smallerTensor,
                      mlir::TypedValue<mlir::RankedTensorType> largerTensor) {
  auto srcType = largerTensor.getType();
  auto resultType = smallerTensor.getType();
  if (!isSameTypeWithoutEncoding(srcType, resultType)) {
    // fast failure path when in and out types do not match
    return false;
  }
  // both types are ensured to have the same rank
  for (int64_t i = 0; i < resultType.getRank(); ++i) {
    // check the ExtractSliceOp offsets, should be all-zero
    if (sliceOp.isDynamicOffset(i) || sliceOp.getStaticOffset(i) != 0)
      return false;
    // check the ExtractSliceOp Strides, should be all-one
    if (sliceOp.isDynamicStride(i) || sliceOp.getStaticStride(i) != 1)
      return false;
  }
  // check if the dynamic shape matchs
  if (resultType.getNumDynamicDims() != 0) {
    for (int64_t i = 0; i < resultType.getRank(); ++i) {
      if (resultType.isDynamicDim(i)) {
        auto largeDim =
            getMixedSize(rewriter, sliceOp.getLoc(), largerTensor, i);
        auto smallDim = sliceOp.getDynamicSize(i);
        if (largeDim.dyn_cast<Value>() != smallDim) {
          return false;
        }
      }
    }
  }
  // if the tensor is in static-shape, we already checked the shapes match via
  // isSameTypeWithoutEncoding
  return true;
}

struct EliminateWholeSliceExtractSliceOp
    : public OpRewritePattern<ExtractSliceOp> {
  EliminateWholeSliceExtractSliceOp(MLIRContext *ctx)
      : OpRewritePattern<ExtractSliceOp>(ctx) {}

  LogicalResult matchAndRewrite(ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    if (!checkEliminateOK(rewriter, sliceOp, sliceOp.getResult(),
                          sliceOp.getSource())) {
      return failure();
    }
    // all checking are done. Rewrite the IR
    rewriter.replaceAllUsesWith(sliceOp, sliceOp.getSource());
    rewriter.eraseOp(sliceOp);
    return success();
  }
};

struct EliminateWholeSliceInsertSliceOp
    : public OpRewritePattern<InsertSliceOp> {
  EliminateWholeSliceInsertSliceOp(MLIRContext *ctx)
      : OpRewritePattern<InsertSliceOp>(ctx) {}

  LogicalResult matchAndRewrite(InsertSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    if (!checkEliminateOK(rewriter, sliceOp, sliceOp.getSource(),
                          sliceOp.getDest())) {
      return failure();
    }
    // all checking are done. Rewrite the IR
    rewriter.replaceAllUsesWith(sliceOp, sliceOp.getSource());
    rewriter.eraseOp(sliceOp);
    return success();
  }
};

} // namespace

void mlir::tensor::populateEliminateWholeSlicingPatterns(
    RewritePatternSet &patterns) {
  patterns
      .add<EliminateWholeSliceExtractSliceOp, EliminateWholeSliceInsertSliceOp>(
          patterns.getContext());
}
