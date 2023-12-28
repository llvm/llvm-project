//===- SubgroupReduceLowering.cpp - subgroup_reduce lowering patterns -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements gradual lowering of `gpu.subgroup_reduce` ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>

using namespace mlir;

namespace {

/// Example, assumes `maxShuffleBitwidth` equal to 32:
/// ```
/// %a = gpu.subgroup_reduce add %x : (vector<3xf16>) -> vector<3xf16>
///  ==>
/// %v0 = arith.constant dense<0.0> : vector<3xf16>
/// %e0 = vector.extract_strided_slice %x
///   {offsets = [0], sizes = [2], strides = [1}: vector<3xf32> to vector<2xf32>
/// %r0 = gpu.subgroup_reduce add %e0 : (vector<2xf16>) -> vector<2xf16>
/// %v1 = vector.insert_strided_slice %r0, %v0
///   {offsets = [0], strides = [1}: vector<2xf32> into vector<3xf32>
/// %e1 = vector.extract %x[2] : f16 from vector<2xf16>
/// %r1 = gpu.subgroup_reduce add %e1 : (f16) -> f16
/// %a  = vector.insert %r1, %v1[2] : f16 into vector<3xf16>
/// ```
struct BreakDownSubgroupReduce final : OpRewritePattern<gpu::SubgroupReduceOp> {
  BreakDownSubgroupReduce(MLIRContext *ctx, unsigned maxShuffleBitwidth,
                          PatternBenefit benefit)
      : OpRewritePattern(ctx, benefit), maxShuffleBitwidth(maxShuffleBitwidth) {
  }

  LogicalResult matchAndRewrite(gpu::SubgroupReduceOp op,
                                PatternRewriter &rewriter) const override {
    auto vecTy = dyn_cast<VectorType>(op.getType());
    if (!vecTy || vecTy.getNumElements() < 2)
      return rewriter.notifyMatchFailure(op, "not a multi-element reduction");

    assert(vecTy.getRank() == 1 && "Unexpected vector type");
    assert(!vecTy.isScalable() && "Unexpected vector type");

    Type elemTy = vecTy.getElementType();
    unsigned elemBitwidth = elemTy.getIntOrFloatBitWidth();
    if (elemBitwidth >= maxShuffleBitwidth)
      return rewriter.notifyMatchFailure(
          op, llvm::formatv("element type too large {0}, cannot break down "
                            "into vectors of bitwidth {1} or less",
                            elemBitwidth, maxShuffleBitwidth));

    unsigned elementsPerShuffle = maxShuffleBitwidth / elemBitwidth;
    assert(elementsPerShuffle >= 1);

    unsigned numNewReductions =
        llvm::divideCeil(vecTy.getNumElements(), elementsPerShuffle);
    assert(numNewReductions >= 1);
    if (numNewReductions == 1)
      return rewriter.notifyMatchFailure(op, "nothing to break down");

    Location loc = op.getLoc();
    Value res =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(vecTy));

    for (unsigned i = 0; i != numNewReductions; ++i) {
      int64_t startIdx = i * elementsPerShuffle;
      int64_t endIdx =
          std::min(startIdx + elementsPerShuffle, vecTy.getNumElements());
      int64_t numElems = endIdx - startIdx;

      Value extracted;
      if (numElems == 1) {
        extracted =
            rewriter.create<vector::ExtractOp>(loc, op.getValue(), startIdx);
      } else {
        extracted = rewriter.create<vector::ExtractStridedSliceOp>(
            loc, op.getValue(), /*offsets=*/startIdx, /*sizes=*/numElems,
            /*strides=*/1);
      }

      Value reduce = rewriter.create<gpu::SubgroupReduceOp>(
          loc, extracted, op.getOp(), op.getUniform());
      if (numElems == 1) {
        res = rewriter.create<vector::InsertOp>(loc, reduce, res, startIdx);
        continue;
      }

      res = rewriter.create<vector::InsertStridedSliceOp>(
          loc, reduce, res, /*offsets=*/startIdx, /*strides=*/1);
    }

    rewriter.replaceOp(op, res);
    return success();
  }

private:
  unsigned maxShuffleBitwidth = 0;
};

/// Example:
/// ```
/// %a = gpu.subgroup_reduce add %x : (vector<1xf32>) -> vector<1xf32>
///  ==>
/// %e0 = vector.extract %x[0] : f32 from vector<1xf32>
/// %r0 = gpu.subgroup_reduce add %e0 : (f32) -> f32
/// %a = vector.broadcast %r0 : f32 to vector<1xf32>
/// ```
struct ScalarizeSingleElementReduce final
    : OpRewritePattern<gpu::SubgroupReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::SubgroupReduceOp op,
                                PatternRewriter &rewriter) const override {
    auto vecTy = dyn_cast<VectorType>(op.getType());
    if (!vecTy || vecTy.getNumElements() != 1)
      return rewriter.notifyMatchFailure(op, "not a single-element reduction");

    assert(vecTy.getRank() == 1 && "Unexpected vector type");
    assert(!vecTy.isScalable() && "Unexpected vector type");
    Location loc = op.getLoc();
    Value extracted = rewriter.create<vector::ExtractOp>(loc, op.getValue(), 0);
    Value reduce = rewriter.create<gpu::SubgroupReduceOp>(
        loc, extracted, op.getOp(), op.getUniform());
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(op, vecTy, reduce);
    return success();
  }
};

} // namespace

void mlir::populateGpuBreakDownSubgrupReducePatterns(
    RewritePatternSet &patterns, unsigned maxShuffleBitwidth,
    PatternBenefit benefit) {
  patterns.add<BreakDownSubgroupReduce>(patterns.getContext(),
                                        maxShuffleBitwidth, benefit);
  patterns.add<ScalarizeSingleElementReduce>(patterns.getContext(), benefit);
}
