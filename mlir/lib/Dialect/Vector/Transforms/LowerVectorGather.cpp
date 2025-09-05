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
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::GatherOp op,
                                PatternRewriter &rewriter) const override {
    Value indexVec = op.getIndices();
    Value maskVec = op.getMask();
    Value passThruVec = op.getPassThru();

    auto unrollGatherFn = [&](PatternRewriter &rewriter, Location loc,
                              VectorType subTy, int64_t index) {
      int64_t thisIdx[1] = {index};

      Value indexSubVec =
          vector::ExtractOp::create(rewriter, loc, indexVec, thisIdx);
      Value maskSubVec =
          vector::ExtractOp::create(rewriter, loc, maskVec, thisIdx);
      Value passThruSubVec =
          vector::ExtractOp::create(rewriter, loc, passThruVec, thisIdx);
      return vector::GatherOp::create(rewriter, loc, subTy, op.getBase(),
                                      op.getOffsets(), indexSubVec, maskSubVec,
                                      passThruSubVec);
    };

    return unrollVectorOp(op, rewriter, unrollGatherFn);
  }
};

/// Rewrites a vector.gather of a strided MemRef as a gather of a non-strided
/// MemRef with updated indices that model the strided access.
///
/// ```mlir
///   %subview = memref.subview %M (...)
///     : memref<100x3xf32> to memref<100xf32, strided<[3]>>
///   %gather = vector.gather %subview[%idxs] (...)
///     : memref<100xf32, strided<[3]>>
/// ```
/// ==>
/// ```mlir
///   %collapse_shape = memref.collapse_shape %M (...)
///     : memref<100x3xf32> into memref<300xf32>
///   %new_idxs = arith.muli %idxs, %c3 : vector<4xindex>
///   %gather = vector.gather %collapse_shape[%new_idxs] (...)
///     : memref<300xf32> (...)
/// ```
///
/// ATM this is effectively limited to reading a 1D Vector from a 2D MemRef,
/// but should be fairly straightforward to extend beyond that.
struct RemoveStrideFromGatherSource : OpRewritePattern<vector::GatherOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::GatherOp op,
                                PatternRewriter &rewriter) const override {
    Value base = op.getBase();

    // TODO: Strided accesses might be coming from other ops as well
    auto subview = base.getDefiningOp<memref::SubViewOp>();
    if (!subview)
      return failure();

    auto sourceType = subview.getSource().getType();

    // TODO: Allow ranks > 2.
    if (sourceType.getRank() != 2)
      return failure();

    // Get strides
    auto layout = subview.getResult().getType().getLayout();
    auto stridedLayoutAttr = llvm::dyn_cast<StridedLayoutAttr>(layout);
    if (!stridedLayoutAttr)
      return failure();

    // TODO: Allow the access to be strided in multiple dimensions.
    if (stridedLayoutAttr.getStrides().size() != 1)
      return failure();

    int64_t srcTrailingDim = sourceType.getShape().back();

    // Assume that the stride matches the trailing dimension of the source
    // memref.
    // TODO: Relax this assumption.
    if (stridedLayoutAttr.getStrides()[0] != srcTrailingDim)
      return failure();

    // 1. Collapse the input memref so that it's "flat".
    SmallVector<ReassociationIndices> reassoc = {{0, 1}};
    Value collapsed = memref::CollapseShapeOp::create(
        rewriter, op.getLoc(), subview.getSource(), reassoc);

    // 2. Generate new gather indices that will model the
    // strided access.
    IntegerAttr stride = rewriter.getIndexAttr(srcTrailingDim);
    VectorType vType = op.getIndices().getType();
    Value mulCst = arith::ConstantOp::create(
        rewriter, op.getLoc(), vType, DenseElementsAttr::get(vType, stride));

    Value newIdxs =
        arith::MulIOp::create(rewriter, op.getLoc(), op.getIndices(), mulCst);

    // 3. Create an updated gather op with the collapsed input memref and the
    // updated indices.
    Value newGather = vector::GatherOp::create(
        rewriter, op.getLoc(), op.getResult().getType(), collapsed,
        op.getOffsets(), newIdxs, op.getMask(), op.getPassThru());
    rewriter.replaceOp(op, newGather);

    return success();
  }
};

/// Turns 1-d `vector.gather` into a scalarized sequence of `vector.loads` or
/// `tensor.extract`s. To avoid out-of-bounds memory accesses, these
/// loads/extracts are made conditional using `scf.if` ops.
struct Gather1DToConditionalLoads : OpRewritePattern<vector::GatherOp> {
  using OpRewritePattern::OpRewritePattern;

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

    // vector.load requires the most minor memref dim to have unit stride
    // (unless reading exactly 1 element)
    if (auto memType = dyn_cast<MemRefType>(base.getType())) {
      if (auto stridesAttr =
              dyn_cast_if_present<StridedLayoutAttr>(memType.getLayout())) {
        if (stridesAttr.getStrides().back() != 1 &&
            resultTy.getNumElements() != 1)
          return failure();
      }
    }

    Value indexVec = rewriter.createOrFold<arith::IndexCastOp>(
        loc, op.getIndexVectorType().clone(rewriter.getIndexType()),
        op.getIndices());
    auto baseOffsets = llvm::to_vector(op.getOffsets());
    Value lastBaseOffset = baseOffsets.back();

    Value result = op.getPassThru();

    // Emit a conditional access for each vector element.
    for (int64_t i = 0, e = resultTy.getNumElements(); i < e; ++i) {
      int64_t thisIdx[1] = {i};
      Value condition =
          vector::ExtractOp::create(rewriter, loc, condMask, thisIdx);
      Value index = vector::ExtractOp::create(rewriter, loc, indexVec, thisIdx);
      baseOffsets.back() =
          rewriter.createOrFold<arith::AddIOp>(loc, lastBaseOffset, index);

      auto loadBuilder = [&](OpBuilder &b, Location loc) {
        Value extracted;
        if (isa<MemRefType>(base.getType())) {
          // `vector.load` does not support scalar result; emit a vector load
          // and extract the single result instead.
          Value load =
              vector::LoadOp::create(b, loc, elemVecTy, base, baseOffsets);
          int64_t zeroIdx[1] = {0};
          extracted = vector::ExtractOp::create(b, loc, load, zeroIdx);
        } else {
          extracted = tensor::ExtractOp::create(b, loc, base, baseOffsets);
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
