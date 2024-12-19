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
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#define DEBUG_TYPE "vector-interleave-lowering"

using namespace mlir;
using namespace mlir::vector;

namespace {

/// A one-shot unrolling of vector.interleave to the `targetRank`.
///
/// Example:
///
/// ```mlir
/// vector.interleave %a, %b : vector<1x2x3x4xi64> -> vector<1x2x3x8xi64>
/// ```
/// Would be unrolled to:
/// ```mlir
/// %result = arith.constant dense<0> : vector<1x2x3x8xi64>
/// %0 = vector.extract %a[0, 0, 0]                 ─┐
///        : vector<4xi64> from vector<1x2x3x4xi64>  |
/// %1 = vector.extract %b[0, 0, 0]                  |
///        : vector<4xi64> from vector<1x2x3x4xi64>  | - Repeated 6x for
/// %2 = vector.interleave %0, %1 :                  |   all leading positions
///        : vector<4xi64> -> vector<8xi64>          |
/// %3 = vector.insert %2, %result [0, 0, 0]         |
///        : vector<8xi64> into vector<1x2x3x8xi64>  ┘
/// ```
///
/// Note: If any leading dimension before the `targetRank` is scalable the
/// unrolling will stop before the scalable dimension.
class UnrollInterleaveOp final : public OpRewritePattern<vector::InterleaveOp> {
public:
  UnrollInterleaveOp(int64_t targetRank, MLIRContext *context,
                     PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), targetRank(targetRank){};

  LogicalResult matchAndRewrite(vector::InterleaveOp op,
                                PatternRewriter &rewriter) const override {
    VectorType resultType = op.getResultVectorType();
    auto unrollIterator = vector::createUnrollIterator(resultType, targetRank);
    if (!unrollIterator)
      return failure();

    auto loc = op.getLoc();
    Value result = rewriter.create<arith::ConstantOp>(
        loc, resultType, rewriter.getZeroAttr(resultType));
    for (auto position : *unrollIterator) {
      Value extractLhs = rewriter.create<ExtractOp>(loc, op.getLhs(), position);
      Value extractRhs = rewriter.create<ExtractOp>(loc, op.getRhs(), position);
      Value interleave =
          rewriter.create<InterleaveOp>(loc, extractLhs, extractRhs);
      result = rewriter.create<InsertOp>(loc, interleave, result, position);
    }

    rewriter.replaceOp(op, result);
    return success();
  }

private:
  int64_t targetRank = 1;
};

/// A one-shot unrolling of vector.deinterleave to the `targetRank`.
///
/// Example:
///
/// ```mlir
/// %0, %1 = vector.deinterleave %a : vector<1x2x3x8xi64> -> vector<1x2x3x4xi64>
/// ```
/// Would be unrolled to:
/// ```mlir
/// %result = arith.constant dense<0> : vector<1x2x3x4xi64>
/// %0 = vector.extract %a[0, 0, 0]                  ─┐
///        : vector<8xi64> from vector<1x2x3x8xi64>   |
/// %1, %2 = vector.deinterleave %0                   |
///        : vector<8xi64> -> vector<4xi64>           | -- Initial deinterleave
/// %3 = vector.insert %1, %result [0, 0, 0]          |    operation unrolled.
///        : vector<4xi64> into vector<1x2x3x4xi64>   |
/// %4 = vector.insert %2, %result [0, 0, 0]          |
///        : vector<4xi64> into vector<1x2x3x4xi64>   ┘
/// %5 = vector.extract %a[0, 0, 1]                  ─┐
///        : vector<8xi64> from vector<1x2x3x8xi64>   |
/// %6, %7 = vector.deinterleave %5                   |
///        : vector<8xi64> -> vector<4xi64>           | -- Recursive pattern for
/// %8 = vector.insert %6, %3 [0, 0, 1]               |    subsequent unrolled
///        : vector<4xi64> into vector<1x2x3x4xi64>   |    deinterleave
/// %9 = vector.insert %7, %4 [0, 0, 1]               |    operations. Repeated
///        : vector<4xi64> into vector<1x2x3x4xi64>   ┘    5x in this case.
/// ```
///
/// Note: If any leading dimension before the `targetRank` is scalable the
/// unrolling will stop before the scalable dimension.
class UnrollDeinterleaveOp final
    : public OpRewritePattern<vector::DeinterleaveOp> {
public:
  UnrollDeinterleaveOp(int64_t targetRank, MLIRContext *context,
                       PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), targetRank(targetRank) {};

  LogicalResult matchAndRewrite(vector::DeinterleaveOp op,
                                PatternRewriter &rewriter) const override {
    VectorType resultType = op.getResultVectorType();
    auto unrollIterator = vector::createUnrollIterator(resultType, targetRank);
    if (!unrollIterator)
      return failure();

    auto loc = op.getLoc();
    Value emptyResult = rewriter.create<arith::ConstantOp>(
        loc, resultType, rewriter.getZeroAttr(resultType));
    Value evenResult = emptyResult;
    Value oddResult = emptyResult;

    for (auto position : *unrollIterator) {
      auto extractSrc =
          rewriter.create<vector::ExtractOp>(loc, op.getSource(), position);
      auto deinterleave =
          rewriter.create<vector::DeinterleaveOp>(loc, extractSrc);
      evenResult = rewriter.create<vector::InsertOp>(
          loc, deinterleave.getRes1(), evenResult, position);
      oddResult = rewriter.create<vector::InsertOp>(loc, deinterleave.getRes2(),
                                                    oddResult, position);
    }
    rewriter.replaceOp(op, ValueRange{evenResult, oddResult});
    return success();
  }

private:
  int64_t targetRank = 1;
};
/// Rewrite vector.interleave op into an equivalent vector.shuffle op, when
/// applicable: `sourceType` must be 1D and non-scalable.
///
/// Example:
///
/// ```mlir
/// vector.interleave %a, %b : vector<7xi16> -> vector<14xi16>
/// ```
///
/// Is rewritten into:
///
/// ```mlir
/// vector.shuffle %arg0, %arg1 [0, 7, 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13]
///   : vector<7xi16>, vector<7xi16>
/// ```
struct InterleaveToShuffle final : OpRewritePattern<vector::InterleaveOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::InterleaveOp op,
                                PatternRewriter &rewriter) const override {
    VectorType sourceType = op.getSourceVectorType();
    if (sourceType.getRank() != 1 || sourceType.isScalable()) {
      return failure();
    }
    int64_t n = sourceType.getNumElements();
    auto seq = llvm::seq<int64_t>(2 * n);
    auto zip = llvm::to_vector(llvm::map_range(
        seq, [n](int64_t i) { return (i % 2 ? n : 0) + i / 2; }));
    rewriter.replaceOpWithNewOp<ShuffleOp>(op, op.getLhs(), op.getRhs(), zip);
    return success();
  }
};

} // namespace

void mlir::vector::populateVectorInterleaveLoweringPatterns(
    RewritePatternSet &patterns, int64_t targetRank, PatternBenefit benefit) {
  patterns.add<UnrollInterleaveOp, UnrollDeinterleaveOp>(
      targetRank, patterns.getContext(), benefit);
}

void mlir::vector::populateVectorInterleaveToShufflePatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<InterleaveToShuffle>(patterns.getContext(), benefit);
}
