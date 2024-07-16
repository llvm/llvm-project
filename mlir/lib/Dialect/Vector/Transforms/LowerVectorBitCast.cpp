//===- LowerVectorBitCast.cpp - Lower 'vector.bitcast' operation ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements target-independent rewrites and utilities to lower the
// 'vector.bitcast' operation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#define DEBUG_TYPE "vector-bitcast-lowering"

using namespace mlir;
using namespace mlir::vector;

namespace {

/// A one-shot unrolling of vector.bitcast to the `targetRank`.
///
/// Example:
///
///   vector.bitcast %a, %b : vector<1x2x3x4xi64> to vector<1x2x3x8xi32>
///
/// Would be unrolled to:
///
/// %result = arith.constant dense<0> : vector<1x2x3x8xi32>
/// %0 = vector.extract %a[0, 0, 0]                 ─┐
///        : vector<4xi64> from vector<1x2x3x4xi64>  |
/// %1 = vector.bitcast %0                           | - Repeated 6x for
///        : vector<4xi64> to vector<8xi32>          |   all leading positions
/// %2 = vector.insert %1, %result [0, 0, 0]         |
///        : vector<8xi64> into vector<1x2x3x8xi32> ─┘
///
/// Note: If any leading dimension before the `targetRank` is scalable the
/// unrolling will stop before the scalable dimension.
class UnrollBitCastOp final : public OpRewritePattern<vector::BitCastOp> {
public:
  UnrollBitCastOp(int64_t targetRank, MLIRContext *context,
                  PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), targetRank(targetRank) {};

  LogicalResult matchAndRewrite(vector::BitCastOp op,
                                PatternRewriter &rewriter) const override {
    VectorType resultType = op.getResultVectorType();
    auto unrollIterator = vector::createUnrollIterator(resultType, targetRank);
    if (!unrollIterator)
      return failure();

    auto unrollRank = unrollIterator->getRank();
    ArrayRef<int64_t> shape = resultType.getShape().drop_front(unrollRank);
    ArrayRef<bool> scalableDims =
        resultType.getScalableDims().drop_front(unrollRank);
    auto bitcastResType =
        VectorType::get(shape, resultType.getElementType(), scalableDims);

    Location loc = op.getLoc();
    Value result = rewriter.create<arith::ConstantOp>(
        loc, resultType, rewriter.getZeroAttr(resultType));
    for (auto position : *unrollIterator) {
      Value extract =
          rewriter.create<vector::ExtractOp>(loc, op.getSource(), position);
      Value bitcast =
          rewriter.create<vector::BitCastOp>(loc, bitcastResType, extract);
      result =
          rewriter.create<vector::InsertOp>(loc, bitcast, result, position);
    }

    rewriter.replaceOp(op, result);
    return success();
  }

private:
  int64_t targetRank = 1;
};

} // namespace

void mlir::vector::populateVectorBitCastLoweringPatterns(
    RewritePatternSet &patterns, int64_t targetRank, PatternBenefit benefit) {
  patterns.add<UnrollBitCastOp>(targetRank, patterns.getContext(), benefit);
}
