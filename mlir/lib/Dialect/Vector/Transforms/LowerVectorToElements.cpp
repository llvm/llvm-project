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

struct UnrollToElements final : public OpRewritePattern<vector::ToElementsOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ToElementsOp op,
                                PatternRewriter &rewriter) const override {

    TypedValue<VectorType> source = op.getSource();
    FailureOr<SmallVector<Value>> result =
        vector::unrollVectorValue(source, rewriter);
    if (failed(result)) {
      return failure();
    }
    SmallVector<Value> vectors = *result;

    SmallVector<Value> results;
    for (const Value &vector : vectors) {
      auto subElements =
          vector::ToElementsOp::create(rewriter, op.getLoc(), vector);
      llvm::append_range(results, subElements.getResults());
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

} // namespace

void mlir::vector::populateVectorToElementsLoweringPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<UnrollToElements>(patterns.getContext(), benefit);
}
