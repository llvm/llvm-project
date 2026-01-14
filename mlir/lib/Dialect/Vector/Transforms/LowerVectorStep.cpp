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
  using Base::Base;

  LogicalResult matchAndRewrite(vector::StepOp stepOp,
                                PatternRewriter &rewriter) const override {
    auto resultType = cast<VectorType>(stepOp.getType());
    if (resultType.isScalable()) {
      return failure();
    }
    int64_t elementCount = resultType.getNumElements();
    SmallVector<APInt> indices =
        llvm::map_to_vector(llvm::seq(elementCount),
                            [](int64_t i) { return APInt(/*width=*/64, i); });
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        stepOp, DenseElementsAttr::get(resultType, indices));
    return success();
  }
};
} // namespace

void mlir::vector::populateVectorStepLoweringPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<StepToArithConstantOpRewrite>(patterns.getContext(), benefit);
}
