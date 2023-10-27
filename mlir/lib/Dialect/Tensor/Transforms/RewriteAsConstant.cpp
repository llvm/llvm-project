//===- RewriteAsConstant.cpp - Patterns to rewrite tensor ops as constants ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::tensor;

namespace {

/// Rewrite tensor.generate with arith.constant if the yielded value is a
/// constant and the tensor type is static.
struct GenerateToConstant : public OpRewritePattern<GenerateOp> {
  using OpRewritePattern<GenerateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenerateOp generateOp,
                                PatternRewriter &rewriter) const override {
    auto tensorType =
        llvm::cast<RankedTensorType>(generateOp.getResult().getType());
    if (!tensorType.hasStaticShape())
      return failure();
    auto terminatorOp =
        cast<tensor::YieldOp>(generateOp.getBody().front().getTerminator());
    Attribute attr;
    if (!matchPattern(terminatorOp.getValue(), m_Constant(&attr)))
      return failure();
    Operation *constantOp =
        rewriter.getContext()
            ->getLoadedDialect<TensorDialect>()
            ->materializeConstant(rewriter,
                                  DenseElementsAttr::get(tensorType, attr),
                                  tensorType, generateOp->getLoc());
    if (!constantOp)
      return failure();
    rewriter.replaceOp(generateOp, constantOp->getResults());
    return success();
  }
};

} // namespace

void mlir::tensor::populateRewriteAsConstantPatterns(
    RewritePatternSet &patterns) {
  patterns.add<GenerateToConstant>(patterns.getContext());
}
