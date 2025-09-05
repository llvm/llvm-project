//===- LowerVectorToElements.cpp - Lower 'vector.to_elements' op ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements target-independent rewrites and utilities to lower the
// 'vector.to_elements' operation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"

#define DEBUG_TYPE "lower-vector-to-elements"

using namespace mlir;

namespace {

struct UnrollToElements : OpRewritePattern<vector::ToElementsOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ToElementsOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> vectors;
    LogicalResult match =
        mlir::vector::unrollVectorValue(op.getSource(), rewriter, vectors);
    if (failed(match)) {
      return match;
    }

    // May be large vector.
    std::vector<Value> results;
    for (const auto &vector : vectors) {
      // we need to replace the current result
      auto subElements =
          rewriter.create<vector::ToElementsOp>(op.getLoc(), vector);
      results.insert(results.end(), subElements.getResults().begin(),
                     subElements.getResults().end());
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

/// Flattens 2 or more dimensional `vector.to_elements` ops by
/// `vector.shape_cast` + `vector.to_elements`.
struct FlattenToElements : OpRewritePattern<vector::ToElementsOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ToElementsOp op,
                                PatternRewriter &rewriter) const override {
    VectorType vecType = op.getSource().getType();
    if (vecType.getRank() <= 1)
      return rewriter.notifyMatchFailure(
          op, "the rank is already less than or equal to 1");
    if (vecType.getNumScalableDims() > 0)
      return rewriter.notifyMatchFailure(
          op, "scalable vector is not yet supported");
    auto vec1DType =
        VectorType::get({vecType.getNumElements()}, vecType.getElementType());
    Value shapeCast = vector::ShapeCastOp::create(rewriter, op.getLoc(),
                                                  vec1DType, op.getSource());
    rewriter.replaceOpWithNewOp<vector::ToElementsOp>(op, op.getResultTypes(),
                                                      shapeCast);
    return success();
  }
};

} // namespace

void mlir::vector::populateVectorToElementsLoweringPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<UnrollToElements>(patterns.getContext(), benefit);
}

void mlir::vector::populateVectorToElementsFlatteningPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<FlattenToElements>(patterns.getContext(), benefit);
}
