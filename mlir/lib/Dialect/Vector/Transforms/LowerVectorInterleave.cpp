//===- LowerVectorInterleave.cpp - Lower 'vector.interleave' operation ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements target-independent rewrites and utilities to lower the
// 'vector.interleave' operation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#define DEBUG_TYPE "vector-interleave-lowering"

using namespace mlir;
using namespace mlir::vector;

namespace {
/// Progressive lowering of InterleaveOp.
class InterleaveOpLowering : public OpRewritePattern<vector::InterleaveOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::InterleaveOp op,
                                PatternRewriter &rewriter) const override {
    VectorType resultType = op.getResultVectorType();
    // 1-D vector.interleave ops can be directly lowered to LLVM (later).
    if (resultType.getRank() == 1)
      return failure();

    // Below we unroll the leading (or front) dimension. If that dimension is
    // scalable we can't unroll it.
    if (resultType.getScalableDims().front())
      return failure();

    // n-D case: Unroll the leading dimension.
    auto loc = op.getLoc();
    Value result = rewriter.create<arith::ConstantOp>(
        loc, resultType, rewriter.getZeroAttr(resultType));
    for (int idx = 0, end = resultType.getDimSize(0); idx < end; ++idx) {
      Value extractLhs = rewriter.create<ExtractOp>(loc, op.getLhs(), idx);
      Value extractRhs = rewriter.create<ExtractOp>(loc, op.getRhs(), idx);
      Value interleave =
          rewriter.create<InterleaveOp>(loc, extractLhs, extractRhs);
      result = rewriter.create<InsertOp>(loc, interleave, result, idx);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void mlir::vector::populateVectorInterleaveLoweringPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<InterleaveOpLowering>(patterns.getContext(), benefit);
}
