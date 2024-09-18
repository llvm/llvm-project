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
#include "mlir/Dialect/GPU/Transforms/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>
#include <cstdint>

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
          op, llvm::formatv("element type too large ({0}), cannot break down "
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
          loc, extracted, op.getOp(), op.getUniform(), op.getClusterSize(),
          op.getClusterStride());
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
        loc, extracted, op.getOp(), op.getUniform(), op.getClusterSize(),
        op.getClusterStride());
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(op, vecTy, reduce);
    return success();
  }
};

struct ClusterInfo {
  unsigned clusterStride;
  unsigned clusterSize;
  unsigned subgroupSize;
};

static FailureOr<ClusterInfo>
getAndValidateClusterInfo(gpu::SubgroupReduceOp op, unsigned subgroupSize) {
  assert(llvm::isPowerOf2_32(subgroupSize));

  std::optional<uint32_t> clusterSize = op.getClusterSize();
  assert(!clusterSize ||
         llvm::isPowerOf2_32(*clusterSize)); // Verifier should've caught this.
  if (clusterSize && *clusterSize > subgroupSize)
    return op.emitOpError()
           << "cluster size " << *clusterSize
           << " is greater than subgroup size " << subgroupSize;
  unsigned effectiveClusterSize = clusterSize.value_or(subgroupSize);

  auto clusterStride = op.getClusterStride();
  assert(llvm::isPowerOf2_32(clusterStride)); // Verifier should've caught this.
  if (clusterStride >= subgroupSize)
    return op.emitOpError()
           << "cluster stride " << clusterStride
           << " is not less than subgroup size " << subgroupSize;

  return ClusterInfo{clusterStride, effectiveClusterSize, subgroupSize};
}

/// Emits a subgroup reduction using a sequence of shuffles. Uses the `packFn`
/// and `unpackFn` to convert to the native shuffle type and to the reduction
/// type, respectively. For example, with `input` of type `f16`, `packFn` could
/// build ops to cast the value to `i32` to perform shuffles, while `unpackFn`
/// would cast it back to `f16` to perform arithmetic reduction on. Assumes that
/// the subgroup is `subgroupSize` lanes wide and divides it into clusters of
/// `clusterSize` lanes starting at lane 0 with a stride of `clusterStride` for
/// lanes within a cluster, reducing all lanes in each cluster in parallel.
Value createSubgroupShuffleReduction(OpBuilder &builder, Location loc,
                                     Value input, gpu::AllReduceOperation mode,
                                     const ClusterInfo &ci,
                                     function_ref<Value(Value)> packFn,
                                     function_ref<Value(Value)> unpackFn) {
  // Lane value always stays in the original type. We use it to perform arith
  // reductions.
  Value laneVal = input;
  // Parallel reduction using butterfly shuffles.
  for (unsigned i = ci.clusterStride; i < ci.clusterStride * ci.clusterSize;
       i <<= 1) {
    Value shuffled = builder
                         .create<gpu::ShuffleOp>(loc, packFn(laneVal), i,
                                                 /*width=*/ci.subgroupSize,
                                                 /*mode=*/gpu::ShuffleMode::XOR)
                         .getShuffleResult();
    laneVal = vector::makeArithReduction(builder, loc,
                                         gpu::convertReductionKind(mode),
                                         laneVal, unpackFn(shuffled));
    assert(laneVal.getType() == input.getType());
  }

  return laneVal;
}

/// Lowers scalar gpu subgroup reductions to a series of shuffles.
struct ScalarSubgroupReduceToShuffles final
    : OpRewritePattern<gpu::SubgroupReduceOp> {
  ScalarSubgroupReduceToShuffles(MLIRContext *ctx, unsigned subgroupSize,
                                 unsigned shuffleBitwidth,
                                 PatternBenefit benefit)
      : OpRewritePattern(ctx, benefit), subgroupSize(subgroupSize),
        shuffleBitwidth(shuffleBitwidth) {}

  LogicalResult matchAndRewrite(gpu::SubgroupReduceOp op,
                                PatternRewriter &rewriter) const override {
    auto ci = getAndValidateClusterInfo(op, subgroupSize);
    if (failed(ci))
      return failure();

    Type valueTy = op.getType();
    unsigned elemBitwidth =
        getElementTypeOrSelf(valueTy).getIntOrFloatBitWidth();
    if (!valueTy.isIntOrFloat() || elemBitwidth > shuffleBitwidth)
      return rewriter.notifyMatchFailure(
          op, "value type is not a compatible scalar");

    Location loc = op.getLoc();
    // Since this is already a native shuffle scalar, no packing is necessary.
    if (elemBitwidth == shuffleBitwidth) {
      auto identityFn = [](Value v) { return v; };
      rewriter.replaceOp(op, createSubgroupShuffleReduction(
                                 rewriter, loc, op.getValue(), op.getOp(), *ci,
                                 identityFn, identityFn));
      return success();
    }

    auto shuffleIntType = rewriter.getIntegerType(shuffleBitwidth);
    auto equivIntType = rewriter.getIntegerType(elemBitwidth);
    auto packFn = [loc, &rewriter, equivIntType,
                   shuffleIntType](Value unpackedVal) -> Value {
      auto asInt =
          rewriter.create<arith::BitcastOp>(loc, equivIntType, unpackedVal);
      return rewriter.create<arith::ExtUIOp>(loc, shuffleIntType, asInt);
    };
    auto unpackFn = [loc, &rewriter, equivIntType,
                     valueTy](Value packedVal) -> Value {
      auto asInt =
          rewriter.create<arith::TruncIOp>(loc, equivIntType, packedVal);
      return rewriter.create<arith::BitcastOp>(loc, valueTy, asInt);
    };

    rewriter.replaceOp(
        op, createSubgroupShuffleReduction(rewriter, loc, op.getValue(),
                                           op.getOp(), *ci, packFn, unpackFn));
    return success();
  }

private:
  unsigned subgroupSize = 0;
  unsigned shuffleBitwidth = 0;
};

/// Lowers vector gpu subgroup reductions to a series of shuffles.
struct VectorSubgroupReduceToShuffles final
    : OpRewritePattern<gpu::SubgroupReduceOp> {
  VectorSubgroupReduceToShuffles(MLIRContext *ctx, unsigned subgroupSize,
                                 unsigned shuffleBitwidth,
                                 PatternBenefit benefit)
      : OpRewritePattern(ctx, benefit), subgroupSize(subgroupSize),
        shuffleBitwidth(shuffleBitwidth) {}

  LogicalResult matchAndRewrite(gpu::SubgroupReduceOp op,
                                PatternRewriter &rewriter) const override {
    auto ci = getAndValidateClusterInfo(op, subgroupSize);
    if (failed(ci))
      return failure();

    auto vecTy = dyn_cast<VectorType>(op.getType());
    if (!vecTy)
      return rewriter.notifyMatchFailure(op, "value type is not a vector");

    unsigned vecBitwidth =
        vecTy.getNumElements() * vecTy.getElementTypeBitWidth();
    if (vecBitwidth > shuffleBitwidth)
      return rewriter.notifyMatchFailure(
          op,
          llvm::formatv("vector type bitwidth too large ({0}), cannot lower "
                        "to shuffles of size {1}",
                        vecBitwidth, shuffleBitwidth));

    unsigned elementsPerShuffle =
        shuffleBitwidth / vecTy.getElementTypeBitWidth();
    if (elementsPerShuffle * vecTy.getElementTypeBitWidth() != shuffleBitwidth)
      return rewriter.notifyMatchFailure(
          op, "shuffle bitwidth is not a multiple of the element bitwidth");

    Location loc = op.getLoc();

    // If the reduced type is smaller than the native shuffle size, extend it,
    // perform the shuffles, and extract at the end.
    auto extendedVecTy = VectorType::get(
        static_cast<int64_t>(elementsPerShuffle), vecTy.getElementType());
    Value extendedInput = op.getValue();
    if (vecBitwidth < shuffleBitwidth) {
      auto zero = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getZeroAttr(extendedVecTy));
      extendedInput = rewriter.create<vector::InsertStridedSliceOp>(
          loc, extendedInput, zero, /*offsets=*/0, /*strides=*/1);
    }

    auto shuffleIntType = rewriter.getIntegerType(shuffleBitwidth);
    auto shuffleVecType = VectorType::get(1, shuffleIntType);

    auto packFn = [loc, &rewriter, shuffleVecType](Value unpackedVal) -> Value {
      auto asIntVec =
          rewriter.create<vector::BitCastOp>(loc, shuffleVecType, unpackedVal);
      return rewriter.create<vector::ExtractOp>(loc, asIntVec, 0);
    };
    auto unpackFn = [loc, &rewriter, shuffleVecType,
                     extendedVecTy](Value packedVal) -> Value {
      auto asIntVec =
          rewriter.create<vector::BroadcastOp>(loc, shuffleVecType, packedVal);
      return rewriter.create<vector::BitCastOp>(loc, extendedVecTy, asIntVec);
    };

    Value res = createSubgroupShuffleReduction(
        rewriter, loc, extendedInput, op.getOp(), *ci, packFn, unpackFn);

    if (vecBitwidth < shuffleBitwidth) {
      res = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, res, /*offsets=*/0, /*sizes=*/vecTy.getNumElements(),
          /*strides=*/1);
    }

    rewriter.replaceOp(op, res);
    return success();
  }

private:
  unsigned subgroupSize = 0;
  unsigned shuffleBitwidth = 0;
};
} // namespace

void mlir::populateGpuBreakDownSubgroupReducePatterns(
    RewritePatternSet &patterns, unsigned maxShuffleBitwidth,
    PatternBenefit benefit) {
  patterns.add<BreakDownSubgroupReduce>(patterns.getContext(),
                                        maxShuffleBitwidth, benefit);
  patterns.add<ScalarizeSingleElementReduce>(patterns.getContext(), benefit);
}

void mlir::populateGpuLowerSubgroupReduceToShufflePatterns(
    RewritePatternSet &patterns, unsigned subgroupSize,
    unsigned shuffleBitwidth, PatternBenefit benefit) {
  patterns.add<ScalarSubgroupReduceToShuffles, VectorSubgroupReduceToShuffles>(
      patterns.getContext(), subgroupSize, shuffleBitwidth, benefit);
}
