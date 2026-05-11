//===- LowerVectorGather.cpp - Lower 'vector.gather' operation ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements target-independent rewrites and utilities to lower the
// 'vector.gather' operation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#define DEBUG_TYPE "vector-broadcast-lowering"

using namespace mlir;
using namespace mlir::vector;

namespace {
/// Unrolls 2 or more dimensional `vector.gather` ops by unrolling the
/// outermost dimension. For example:
/// ```
/// %g = vector.gather %base[%c0][%v], %mask, %pass_thru :
///        ... into vector<2x3xf32>
///
/// ==>
///
/// %0   = arith.constant dense<0.0> : vector<2x3xf32>
/// %g0  = vector.gather %base[%c0][%v0], %mask0, %pass_thru0 : ...
/// %1   = vector.insert %g0, %0 [0] : vector<3xf32> into vector<2x3xf32>
/// %g1  = vector.gather %base[%c0][%v1], %mask1, %pass_thru1 : ...
/// %g   = vector.insert %g1, %1 [1] : vector<3xf32> into vector<2x3xf32>
/// ```
///
/// When applied exhaustively, this will produce a sequence of 1-d gather ops.
///
/// Supports vector types with a fixed leading dimension.
struct UnrollGather : OpRewritePattern<vector::GatherOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(vector::GatherOp op,
                                PatternRewriter &rewriter) const override {
    OperandRange indexVecs = op.getIndices();
    Value maskVec = op.getMask();
    Value passThruVec = op.getPassThru();

    auto unrollGatherFn = [&](PatternRewriter &rewriter, Location loc,
                              VectorType subTy, int64_t index) {
      int64_t thisIdx[1] = {index};

      SmallVector<Value> indexSubVecs;
      for (Value iv : indexVecs)
        indexSubVecs.push_back(
            vector::ExtractOp::create(rewriter, loc, iv, thisIdx));
      Value maskSubVec =
          vector::ExtractOp::create(rewriter, loc, maskVec, thisIdx);
      Value passThruSubVec =
          vector::ExtractOp::create(rewriter, loc, passThruVec, thisIdx);
      return vector::GatherOp::create(rewriter, loc, subTy, op.getBase(),
                                      op.getOffsets(), indexSubVecs, maskSubVec,
                                      passThruSubVec, op.getAlignmentAttr());
    };

    return unrollVectorOp(op, rewriter, unrollGatherFn);
  }
};

/// Rewrites a vector.gather of a MemRef subview as a gather of the subview's
/// source MemRef with composed offsets and multi-dimensional indices.
///
/// Index vectors are mapped back to their corresponding source dimensions and
/// multiplied by the subview strides. Any missing source dimensions in the
/// indexed suffix, such as rank-reduced dimensions, get zero index vectors.
///
/// ```mlir
///   %subview = memref.subview %M (...)
///     : memref<100x3x5xf32> to memref<100xf32, strided<[15]>>
///   %gather = vector.gather %subview[%c0] [%idxs] (...)
///     : memref<100xf32, strided<[15]>>
/// ```
/// ==>
/// ```mlir
///   %zeros = arith.constant dense<0> : vector<4xindex>
///   %gather = vector.gather %M[%c0, %c0, %c0] [%idxs, %zeros, %zeros] (...)
///     : memref<100x3x5xf32> (...)
/// ```
///
struct RemoveStrideFromGatherSource : OpRewritePattern<vector::GatherOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(vector::GatherOp op,
                                PatternRewriter &rewriter) const override {
    auto subview = op.getBase().getDefiningOp<memref::SubViewOp>();
    if (!subview)
      return rewriter.notifyMatchFailure(op, "base is not a memref.subview");

    Location loc = op.getLoc();
    VectorType vType = op.getIndexVectorType();
    Type indexElementType = vType.getElementType();
    Value zeroVec = arith::ConstantOp::create(rewriter, loc, vType,
                                              rewriter.getZeroAttr(vType));

    auto scaleIndexVec = [&](Value indexVec, OpFoldResult stride) -> Value {
      if (isConstantIntValue(stride, 1))
        return indexVec;

      Value strideForVec =
          getValueOrCreateConstantIndexOp(rewriter, loc, stride);
      if (indexElementType != rewriter.getIndexType())
        strideForVec = arith::IndexCastOp::create(
            rewriter, loc, indexElementType, strideForVec);
      Value strideVec =
          vector::BroadcastOp::create(rewriter, loc, vType, strideForVec);
      return rewriter.createOrFold<arith::MulIOp>(loc, indexVec, strideVec);
    };

    auto scaleOffset = [&](Value offset, OpFoldResult stride) -> Value {
      if (isConstantIntValue(stride, 1))
        return offset;

      Value strideValue =
          getValueOrCreateConstantIndexOp(rewriter, loc, stride);
      return rewriter.createOrFold<arith::MulIOp>(loc, offset, strideValue);
    };

    MemRefType sourceType = subview.getSourceType();
    MemRefType resultType = subview.getResult().getType();
    unsigned sourceRank = sourceType.getRank();
    unsigned resultRank = resultType.getRank();
    unsigned gatherRank = op.getIndices().size();
    unsigned firstGatherResultDim = resultRank - gatherRank;

    llvm::SmallBitVector droppedDims = subview.getDroppedDims();
    SmallVector<OpFoldResult> subviewOffsets = subview.getMixedOffsets();
    SmallVector<OpFoldResult> subviewStrides = subview.getMixedStrides();
    SmallVector<Value> newOffsets;
    SmallVector<Value> newIndexVecs;

    std::optional<unsigned> firstIndexedSourceDim;
    unsigned resultDim = 0;
    for (unsigned sourceDim = 0; sourceDim < sourceRank; ++sourceDim) {
      Value sourceOffset = getValueOrCreateConstantIndexOp(
          rewriter, loc, subviewOffsets[sourceDim]);

      if (!droppedDims.test(sourceDim)) {
        Value scaledOffset =
            scaleOffset(op.getOffsets()[resultDim], subviewStrides[sourceDim]);
        sourceOffset = rewriter.createOrFold<arith::AddIOp>(loc, sourceOffset,
                                                            scaledOffset);

        if (resultDim >= firstGatherResultDim) {
          if (!firstIndexedSourceDim) {
            firstIndexedSourceDim = sourceDim;
            // Handle dropped dimensions by giving them explicit zero indices
            // here.
            newIndexVecs = SmallVector<Value>(sourceRank - sourceDim, zeroVec);
          }
          newIndexVecs[sourceDim - *firstIndexedSourceDim] =
              scaleIndexVec(op.getIndices()[resultDim - firstGatherResultDim],
                            subviewStrides[sourceDim]);
        }

        ++resultDim;
      }

      newOffsets.push_back(sourceOffset);
    }

    assert(resultDim == resultRank && "did not map all subview result dims");
    assert(firstIndexedSourceDim &&
           "expected at least one source dim for the gather indices");

    Value newGather = vector::GatherOp::create(
        rewriter, loc, op.getResult().getType(), subview.getSource(),
        newOffsets, newIndexVecs, op.getMask(), op.getPassThru(),
        op.getAlignmentAttr());
    rewriter.replaceOp(op, newGather);

    return success();
  }
};

/// Turns 1-d `vector.gather` into a scalarized sequence of `vector.loads` or
/// `tensor.extract`s. To avoid out-of-bounds memory accesses, these
/// loads/extracts are made conditional using `scf.if` ops.
///
/// With r index vectors, each one directly offsets one of the r innermost
/// base dimensions. The load indices are:
///   loadOffsets[k-r+j] = offsets[k-r+j] + indexCast(indices[j][i])
struct Gather1DToConditionalLoads : OpRewritePattern<vector::GatherOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(vector::GatherOp op,
                                PatternRewriter &rewriter) const override {
    VectorType resultTy = op.getType();
    if (resultTy.getRank() != 1)
      return rewriter.notifyMatchFailure(op, "unsupported rank");

    if (resultTy.isScalable())
      return rewriter.notifyMatchFailure(op, "not a fixed-width vector");

    Location loc = op.getLoc();
    Type elemTy = resultTy.getElementType();
    // Vector type with a single element. Used to generate `vector.loads`.
    VectorType elemVecTy = VectorType::get({1}, elemTy);

    Value condMask = op.getMask();
    Value base = op.getBase();

    if (auto memType = dyn_cast<MemRefType>(base.getType())) {
      // vector.load requires the most minor memref dim to have unit stride
      // (unless reading exactly 1 element).
      if (auto stridesAttr =
              dyn_cast_if_present<StridedLayoutAttr>(memType.getLayout())) {
        if (stridesAttr.getStrides().back() != 1 &&
            resultTy.getNumElements() != 1)
          return rewriter.notifyMatchFailure(
              op, "most minor memref dim must have unit stride");
      }
    }

    unsigned r = op.getIndices().size();
    VectorType indexVecType =
        op.getIndexVectorType().clone(rewriter.getIndexType());
    SmallVector<Value> indexVecs;
    for (Value iv : op.getIndices()) {
      if (iv.getType() == indexVecType)
        indexVecs.push_back(iv);
      else
        indexVecs.push_back(
            rewriter.createOrFold<arith::IndexCastOp>(loc, indexVecType, iv));
    }

    // Snapshot the offsets so per-element rewrites of the trailing r entries
    // (loadOffsets[rank-r+j] += indices[j][i]) start from the originals each
    // iteration, not from the previous iteration's sum.
    auto loadOffsets = llvm::to_vector(op.getOffsets());
    unsigned rank = loadOffsets.size();
    SmallVector<Value> savedOffsets =
        llvm::to_vector(ArrayRef<Value>(loadOffsets).take_back(r));

    Value result = op.getPassThru();
    BoolAttr nontemporalAttr = nullptr;
    IntegerAttr alignmentAttr = op.getAlignmentAttr();

    // Emit a conditional access for each vector element.
    for (int64_t i = 0, e = resultTy.getNumElements(); i < e; ++i) {
      int64_t thisIdx[1] = {i};
      Value condition =
          vector::ExtractOp::create(rewriter, loc, condMask, thisIdx);

      for (unsigned j = 0; j < r; ++j) {
        Value index =
            vector::ExtractOp::create(rewriter, loc, indexVecs[j], thisIdx);
        loadOffsets[rank - r + j] =
            rewriter.createOrFold<arith::AddIOp>(loc, savedOffsets[j], index);
      }

      auto loadBuilder = [&](OpBuilder &b, Location loc) {
        Value extracted;
        if (isa<MemRefType>(base.getType())) {
          // `vector.load` does not support scalar result; emit a vector load
          // and extract the single result instead.
          Value load =
              vector::LoadOp::create(b, loc, elemVecTy, base, loadOffsets,
                                     nontemporalAttr, alignmentAttr);
          int64_t zeroIdx[1] = {0};
          extracted = vector::ExtractOp::create(b, loc, load, zeroIdx);
        } else {
          extracted = tensor::ExtractOp::create(b, loc, base, loadOffsets);
        }

        Value newResult =
            vector::InsertOp::create(b, loc, extracted, result, thisIdx);
        scf::YieldOp::create(b, loc, newResult);
      };
      auto passThruBuilder = [result](OpBuilder &b, Location loc) {
        scf::YieldOp::create(b, loc, result);
      };

      result = scf::IfOp::create(rewriter, loc, condition,
                                 /*thenBuilder=*/loadBuilder,
                                 /*elseBuilder=*/passThruBuilder)
                   .getResult(0);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

void mlir::vector::populateVectorGatherLoweringPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<UnrollGather>(patterns.getContext(), benefit);
}

void mlir::vector::populateVectorGatherToConditionalLoadPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<RemoveStrideFromGatherSource, Gather1DToConditionalLoads>(
      patterns.getContext(), benefit);
}
