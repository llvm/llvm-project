//===- LowerVectorStep.cpp - Lower 'vector.step' operation ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements target-independent rewrites and utilities to lower the
// 'vector.step' operation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/IR/PatternMatch.h"

#define DEBUG_TYPE "vector-step-lowering"

using namespace mlir;
using namespace mlir::vector;

namespace {

struct StepToArithConstantOpRewrite final : OpRewritePattern<vector::StepOp> {
  StepToArithConstantOpRewrite(MLIRContext *context, unsigned indexBitwidth,
                               PatternBenefit benefit)
      : OpRewritePattern(context, benefit), indexBitwidth(indexBitwidth) {
    assert(indexBitwidth <= IndexType::kInternalStorageBitWidth &&
           "indexBitwidth cannot exceed the index storage bitwidth");
  }

  LogicalResult matchAndRewrite(vector::StepOp stepOp,
                                PatternRewriter &rewriter) const override {
    auto resultType = cast<VectorType>(stepOp.getType());
    if (resultType.isScalable()) {
      return failure();
    }
    Type elementType = resultType.getElementType();
    // An `indexBitwidth` of 0 means "leave `index`-typed steps alone": the
    // index bitwidth is target-dependent, so callers that don't know it (and
    // defer to a later lowering, e.g. to `llvm.intr.stepvector`) opt out.
    if (elementType.isIndex() && indexBitwidth == 0) {
      return failure();
    }
    // Values wrap around at `computeWidth`. `index` elements are stored in a
    // `DenseElementsAttr` using the internal storage bitwidth, so the wrapped
    // value is widened to it; integer elements use their own bitwidth.
    unsigned computeWidth = elementType.isIndex()
                                ? indexBitwidth
                                : elementType.getIntOrFloatBitWidth();
    unsigned storageWidth = elementType.isIndex()
                                ? IndexType::kInternalStorageBitWidth
                                : computeWidth;
    int64_t elementCount = resultType.getNumElements();
    SmallVector<APInt> indices = llvm::map_to_vector(
        llvm::seq(elementCount), [computeWidth, storageWidth](int64_t i) {
          return APInt(computeWidth, i, /*isSigned=*/false,
                       /*implicitTrunc=*/true)
              .zext(storageWidth);
        });
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        stepOp, DenseElementsAttr::get(resultType, indices));
    return success();
  }

  unsigned indexBitwidth;
};
} // namespace

void mlir::vector::populateVectorStepLoweringPatterns(
    RewritePatternSet &patterns, unsigned indexBitwidth,
    PatternBenefit benefit) {
  patterns.add<StepToArithConstantOpRewrite>(patterns.getContext(),
                                             indexBitwidth, benefit);
}
