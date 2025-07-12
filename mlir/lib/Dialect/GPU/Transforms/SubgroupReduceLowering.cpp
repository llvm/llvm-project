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

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/GPU/Utils/GPUUtils.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
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
        arith::ConstantOp::create(rewriter, loc, rewriter.getZeroAttr(vecTy));

    for (unsigned i = 0; i != numNewReductions; ++i) {
      int64_t startIdx = i * elementsPerShuffle;
      int64_t endIdx =
          std::min(startIdx + elementsPerShuffle, vecTy.getNumElements());
      int64_t numElems = endIdx - startIdx;

      Value extracted;
      if (numElems == 1) {
        extracted =
            vector::ExtractOp::create(rewriter, loc, op.getValue(), startIdx);
      } else {
        extracted = vector::ExtractStridedSliceOp::create(
            rewriter, loc, op.getValue(), /*offsets=*/startIdx,
            /*sizes=*/numElems,
            /*strides=*/1);
      }

      Value reduce = gpu::SubgroupReduceOp::create(
          rewriter, loc, extracted, op.getOp(), op.getUniform(),
          op.getClusterSize(), op.getClusterStride());
      if (numElems == 1) {
        res = vector::InsertOp::create(rewriter, loc, reduce, res, startIdx);
        continue;
      }

      res = vector::InsertStridedSliceOp::create(
          rewriter, loc, reduce, res, /*offsets=*/startIdx, /*strides=*/1);
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
    Value extracted =
        vector::ExtractOp::create(rewriter, loc, op.getValue(), 0);
    Value reduce = gpu::SubgroupReduceOp::create(
        rewriter, loc, extracted, op.getOp(), op.getUniform(),
        op.getClusterSize(), op.getClusterStride());
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
                                 unsigned shuffleBitwidth, bool matchClustered,
                                 PatternBenefit benefit)
      : OpRewritePattern(ctx, benefit), subgroupSize(subgroupSize),
        shuffleBitwidth(shuffleBitwidth), matchClustered(matchClustered) {}

  LogicalResult matchAndRewrite(gpu::SubgroupReduceOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getClusterSize().has_value() != matchClustered) {
      return rewriter.notifyMatchFailure(
          op, llvm::formatv("op is {0}clustered but pattern is configured to "
                            "only match {1}clustered ops",
                            matchClustered ? "non-" : "",
                            matchClustered ? "" : "non-"));
    }

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
          arith::BitcastOp::create(rewriter, loc, equivIntType, unpackedVal);
      return arith::ExtUIOp::create(rewriter, loc, shuffleIntType, asInt);
    };
    auto unpackFn = [loc, &rewriter, equivIntType,
                     valueTy](Value packedVal) -> Value {
      auto asInt =
          arith::TruncIOp::create(rewriter, loc, equivIntType, packedVal);
      return arith::BitcastOp::create(rewriter, loc, valueTy, asInt);
    };

    rewriter.replaceOp(
        op, createSubgroupShuffleReduction(rewriter, loc, op.getValue(),
                                           op.getOp(), *ci, packFn, unpackFn));
    return success();
  }

private:
  unsigned subgroupSize = 0;
  unsigned shuffleBitwidth = 0;
  bool matchClustered = false;
};

/// Lowers vector gpu subgroup reductions to a series of shuffles.
struct VectorSubgroupReduceToShuffles final
    : OpRewritePattern<gpu::SubgroupReduceOp> {
  VectorSubgroupReduceToShuffles(MLIRContext *ctx, unsigned subgroupSize,
                                 unsigned shuffleBitwidth, bool matchClustered,
                                 PatternBenefit benefit)
      : OpRewritePattern(ctx, benefit), subgroupSize(subgroupSize),
        shuffleBitwidth(shuffleBitwidth), matchClustered(matchClustered) {}

  LogicalResult matchAndRewrite(gpu::SubgroupReduceOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getClusterSize().has_value() != matchClustered) {
      return rewriter.notifyMatchFailure(
          op, llvm::formatv("op is {0}clustered but pattern is configured to "
                            "only match {1}clustered ops",
                            matchClustered ? "non-" : "",
                            matchClustered ? "" : "non-"));
    }

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
      auto zero = arith::ConstantOp::create(
          rewriter, loc, rewriter.getZeroAttr(extendedVecTy));
      extendedInput = vector::InsertStridedSliceOp::create(
          rewriter, loc, extendedInput, zero, /*offsets=*/0, /*strides=*/1);
    }

    auto shuffleIntType = rewriter.getIntegerType(shuffleBitwidth);
    auto shuffleVecType = VectorType::get(1, shuffleIntType);

    auto packFn = [loc, &rewriter, shuffleVecType](Value unpackedVal) -> Value {
      auto asIntVec =
          vector::BitCastOp::create(rewriter, loc, shuffleVecType, unpackedVal);
      return vector::ExtractOp::create(rewriter, loc, asIntVec, 0);
    };
    auto unpackFn = [loc, &rewriter, shuffleVecType,
                     extendedVecTy](Value packedVal) -> Value {
      auto asIntVec =
          vector::BroadcastOp::create(rewriter, loc, shuffleVecType, packedVal);
      return vector::BitCastOp::create(rewriter, loc, extendedVecTy, asIntVec);
    };

    Value res = createSubgroupShuffleReduction(
        rewriter, loc, extendedInput, op.getOp(), *ci, packFn, unpackFn);

    if (vecBitwidth < shuffleBitwidth) {
      res = vector::ExtractStridedSliceOp::create(
          rewriter, loc, res, /*offsets=*/0, /*sizes=*/vecTy.getNumElements(),
          /*strides=*/1);
    }

    rewriter.replaceOp(op, res);
    return success();
  }

private:
  unsigned subgroupSize = 0;
  unsigned shuffleBitwidth = 0;
  bool matchClustered = false;
};

static FailureOr<Value>
createSubgroupDPPReduction(PatternRewriter &rewriter, gpu::SubgroupReduceOp op,
                           Value input, gpu::AllReduceOperation mode,
                           const ClusterInfo &ci, amdgpu::Chipset chipset) {
  Location loc = op.getLoc();
  Value dpp;
  Value res = input;
  constexpr int allRows = 0xf;
  constexpr int allBanks = 0xf;
  const bool boundCtrl = true;
  if (ci.clusterSize >= 2) {
    // Perform reduction between all lanes N <-> N+1.
    dpp = amdgpu::DPPOp::create(
        rewriter, loc, res.getType(), res, res, amdgpu::DPPPerm::quad_perm,
        rewriter.getI32ArrayAttr({1, 0, 3, 2}), allRows, allBanks, boundCtrl);
    res = vector::makeArithReduction(rewriter, loc,
                                     gpu::convertReductionKind(mode), res, dpp);
  }

  if (ci.clusterSize >= 4) {
    // Perform reduction between all lanes N <-> N+2.
    dpp = amdgpu::DPPOp::create(
        rewriter, loc, res.getType(), res, res, amdgpu::DPPPerm::quad_perm,
        rewriter.getI32ArrayAttr({2, 3, 0, 1}), allRows, allBanks, boundCtrl);
    res = vector::makeArithReduction(rewriter, loc,
                                     gpu::convertReductionKind(mode), res, dpp);
  }
  if (ci.clusterSize >= 8) {
    // Perform reduction between all lanes N <-> 7-N,
    // e.g lane[0] <-> lane[7], lane[1] <-> lane[6]..., lane[3] <-> lane[4].
    dpp = amdgpu::DPPOp::create(rewriter, loc, res.getType(), res, res,
                                amdgpu::DPPPerm::row_half_mirror,
                                rewriter.getUnitAttr(), allRows, allBanks,
                                boundCtrl);
    res = vector::makeArithReduction(rewriter, loc,
                                     gpu::convertReductionKind(mode), res, dpp);
  }
  if (ci.clusterSize >= 16) {
    // Perform reduction between all lanes N <-> 15-N,
    // e.g lane[0] <-> lane[15], lane[1] <-> lane[14]..., lane[7] <-> lane[8].
    dpp = amdgpu::DPPOp::create(
        rewriter, loc, res.getType(), res, res, amdgpu::DPPPerm::row_mirror,
        rewriter.getUnitAttr(), allRows, allBanks, boundCtrl);
    res = vector::makeArithReduction(rewriter, loc,
                                     gpu::convertReductionKind(mode), res, dpp);
  }
  if (ci.clusterSize >= 32) {
    if (chipset.majorVersion <= 9) {
      // Broadcast last value from each row to next row.
      // Use row mask to avoid polluting rows 1 and 3.
      dpp = amdgpu::DPPOp::create(rewriter, loc, res.getType(), res, res,
                                  amdgpu::DPPPerm::row_bcast_15,
                                  rewriter.getUnitAttr(), 0xa, allBanks,
                                  /*bound_ctrl*/ false);
      res = vector::makeArithReduction(
          rewriter, loc, gpu::convertReductionKind(mode), res, dpp);
    } else if (chipset.majorVersion <= 12) {
      // Use a permute lane to cross rows (row 1 <-> row 0, row 3 <-> row 2).
      Value uint32Max = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(-1));
      dpp = ROCDL::PermlaneX16Op::create(rewriter, loc, res.getType(), res, res,
                                         uint32Max, uint32Max,
                                         /*fi=*/true,
                                         /*bound_ctrl=*/false);
      res = vector::makeArithReduction(
          rewriter, loc, gpu::convertReductionKind(mode), res, dpp);
    } else {
      return rewriter.notifyMatchFailure(
          op, "Subgroup reduce lowering to DPP not currently supported for "
              "this device.");
    }
    if (ci.subgroupSize == 32) {
      Value lane31 = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(31));
      res =
          ROCDL::ReadlaneOp::create(rewriter, loc, res.getType(), res, lane31);
    }
  }
  if (ci.clusterSize >= 64) {
    if (chipset.majorVersion <= 9) {
      // Broadcast 31st lane value to rows 2 and 3.
      dpp = amdgpu::DPPOp::create(rewriter, loc, res.getType(), res, res,
                                  amdgpu::DPPPerm::row_bcast_31,
                                  rewriter.getUnitAttr(), 0xf, allBanks,
                                  /*bound_ctrl*/ true);
      res = vector::makeArithReduction(
          rewriter, loc, gpu::convertReductionKind(mode), dpp, res);
      // Obtain reduction from last rows, the previous rows are polluted.
      Value lane63 = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(63));
      res =
          ROCDL::ReadlaneOp::create(rewriter, loc, res.getType(), res, lane63);

    } else if (chipset.majorVersion <= 12) {
      // Assume reduction across 32 lanes has been done.
      // Perform final reduction manually by summing values in lane 0 and
      // lane 32.
      Value lane31 = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(31));
      Value lane63 = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(63));
      lane31 =
          ROCDL::ReadlaneOp::create(rewriter, loc, res.getType(), res, lane31);
      lane63 =
          ROCDL::ReadlaneOp::create(rewriter, loc, res.getType(), res, lane63);
      res = vector::makeArithReduction(
          rewriter, loc, gpu::convertReductionKind(mode), lane31, lane63);
    } else {
      return rewriter.notifyMatchFailure(
          op, "Subgroup reduce lowering to DPP not currently supported for "
              "this device.");
    }
  }
  assert(res.getType() == input.getType());
  return res;
}

/// Collect a set of patterns to lower `gpu.subgroup_reduce` into `amdgpu.dpp`
/// ops over scalar types. Assumes that the subgroup has
/// `subgroupSize` lanes. Applicable only to AMD GPUs.
struct ScalarSubgroupReduceToDPP final
    : OpRewritePattern<gpu::SubgroupReduceOp> {
  ScalarSubgroupReduceToDPP(MLIRContext *ctx, unsigned subgroupSize,
                            bool matchClustered, amdgpu::Chipset chipset,
                            PatternBenefit benefit)
      : OpRewritePattern(ctx, benefit), subgroupSize(subgroupSize),
        matchClustered(matchClustered), chipset(chipset) {}

  LogicalResult matchAndRewrite(gpu::SubgroupReduceOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getClusterSize().has_value() != matchClustered) {
      return rewriter.notifyMatchFailure(
          op, llvm::formatv("op is {0}clustered but pattern is configured to "
                            "only match {1}clustered ops",
                            matchClustered ? "non-" : "",
                            matchClustered ? "" : "non-"));
    }
    auto ci = getAndValidateClusterInfo(op, subgroupSize);
    if (failed(ci))
      return failure();

    if (ci->clusterStride != 1)
      return rewriter.notifyMatchFailure(
          op, "Subgroup reductions using DPP are currently only available for "
              "clusters of contiguous lanes.");

    Type valueTy = op.getType();
    if (!valueTy.isIntOrFloat())
      return rewriter.notifyMatchFailure(
          op, "Value type is not a compatible scalar.");

    FailureOr<Value> dpp = createSubgroupDPPReduction(
        rewriter, op, op.getValue(), op.getOp(), *ci, chipset);
    if (failed(dpp))
      return failure();

    rewriter.replaceOp(op, dpp.value());
    return success();
  }

private:
  unsigned subgroupSize = 0;
  bool matchClustered = false;
  amdgpu::Chipset chipset;
};
} // namespace

void mlir::populateGpuBreakDownSubgroupReducePatterns(
    RewritePatternSet &patterns, unsigned maxShuffleBitwidth,
    PatternBenefit benefit) {
  patterns.add<BreakDownSubgroupReduce>(patterns.getContext(),
                                        maxShuffleBitwidth, benefit);
  patterns.add<ScalarizeSingleElementReduce>(patterns.getContext(), benefit);
}

void mlir::populateGpuLowerSubgroupReduceToDPPPatterns(
    RewritePatternSet &patterns, unsigned subgroupSize, amdgpu::Chipset chipset,
    PatternBenefit benefit) {
  patterns.add<ScalarSubgroupReduceToDPP>(patterns.getContext(), subgroupSize,
                                          /*matchClustered=*/false, chipset,
                                          benefit);
}

void mlir::populateGpuLowerClusteredSubgroupReduceToDPPPatterns(
    RewritePatternSet &patterns, unsigned subgroupSize, amdgpu::Chipset chipset,
    PatternBenefit benefit) {
  patterns.add<ScalarSubgroupReduceToDPP>(patterns.getContext(), subgroupSize,
                                          /*matchClustered=*/true, chipset,
                                          benefit);
}

void mlir::populateGpuLowerSubgroupReduceToShufflePatterns(
    RewritePatternSet &patterns, unsigned subgroupSize,
    unsigned shuffleBitwidth, PatternBenefit benefit) {
  patterns.add<ScalarSubgroupReduceToShuffles, VectorSubgroupReduceToShuffles>(
      patterns.getContext(), subgroupSize, shuffleBitwidth,
      /*matchClustered=*/false, benefit);
}

void mlir::populateGpuLowerClusteredSubgroupReduceToShufflePatterns(
    RewritePatternSet &patterns, unsigned subgroupSize,
    unsigned shuffleBitwidth, PatternBenefit benefit) {
  patterns.add<ScalarSubgroupReduceToShuffles, VectorSubgroupReduceToShuffles>(
      patterns.getContext(), subgroupSize, shuffleBitwidth,
      /*matchClustered=*/true, benefit);
}
