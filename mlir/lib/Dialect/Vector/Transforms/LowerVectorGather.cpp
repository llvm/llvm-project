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

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "vector-broadcast-lowering"

using namespace mlir;
using namespace mlir::vector;

namespace {
/// Flattens 2 or more dimensional `vector.gather` ops by unrolling the
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
struct FlattenGather : OpRewritePattern<vector::GatherOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::GatherOp op,
                                PatternRewriter &rewriter) const override {
    VectorType resultTy = op.getType();
    if (resultTy.getRank() < 2)
      return rewriter.notifyMatchFailure(op, "already flat");

    // Unrolling doesn't take vscale into account. Pattern is disabled for
    // vectors with leading scalable dim(s).
    if (resultTy.getScalableDims().front())
      return rewriter.notifyMatchFailure(op, "cannot unroll scalable dim");

    Location loc = op.getLoc();
    Value indexVec = op.getIndexVec();
    Value maskVec = op.getMask();
    Value passThruVec = op.getPassThru();

    Value result = rewriter.create<arith::ConstantOp>(
        loc, resultTy, rewriter.getZeroAttr(resultTy));

    VectorType subTy = VectorType::Builder(resultTy).dropDim(0);

    for (int64_t i = 0, e = resultTy.getShape().front(); i < e; ++i) {
      int64_t thisIdx[1] = {i};

      Value indexSubVec =
          rewriter.create<vector::ExtractOp>(loc, indexVec, thisIdx);
      Value maskSubVec =
          rewriter.create<vector::ExtractOp>(loc, maskVec, thisIdx);
      Value passThruSubVec =
          rewriter.create<vector::ExtractOp>(loc, passThruVec, thisIdx);
      Value subGather = rewriter.create<vector::GatherOp>(
          loc, subTy, op.getBase(), op.getIndices(), indexSubVec, maskSubVec,
          passThruSubVec);
      result =
          rewriter.create<vector::InsertOp>(loc, subGather, result, thisIdx);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Rewrites a vector.gather of a strided MemRef as a gather of a non-strided
/// MemRef with updated indices that model the strided access.
///
/// ```mlir
///   %subview = memref.subview %M (...)
///     : memref<100x3xf32> to memref<100xf32, strided<[3]>>
///   %gather = vector.gather %subview[%idxs] (...) : memref<100xf32, strided<[3]>>
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
    Value collapsed = rewriter.create<memref::CollapseShapeOp>(
        op.getLoc(), subview.getSource(), reassoc);

    // 2. Generate new gather indices that will model the
    // strided access.
    IntegerAttr stride = rewriter.getIndexAttr(srcTrailingDim);
    VectorType vType = op.getIndexVec().getType();
    Value mulCst = rewriter.create<arith::ConstantOp>(
        op.getLoc(), vType, DenseElementsAttr::get(vType, stride));

    Value newIdxs =
        rewriter.create<arith::MulIOp>(op.getLoc(), op.getIndexVec(), mulCst);

    // 3. Create an updated gather op with the collapsed input memref and the
    // updated indices.
    Value newGather = rewriter.create<vector::GatherOp>(
        op.getLoc(), op.getResult().getType(), collapsed, op.getIndices(),
        newIdxs, op.getMask(), op.getPassThru());
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
    if (auto memType = dyn_cast<MemRefType>(base.getType())) {
      if (auto stridesAttr =
              dyn_cast_if_present<StridedLayoutAttr>(memType.getLayout())) {
        if (stridesAttr.getStrides().back() != 1)
          return failure();
      }
    }

    Value indexVec = rewriter.createOrFold<arith::IndexCastOp>(
        loc, op.getIndexVectorType().clone(rewriter.getIndexType()),
        op.getIndexVec());
    auto baseOffsets = llvm::to_vector(op.getIndices());
    Value lastBaseOffset = baseOffsets.back();

    Value result = op.getPassThru();

    // Emit a conditional access for each vector element.
    for (int64_t i = 0, e = resultTy.getNumElements(); i < e; ++i) {
      int64_t thisIdx[1] = {i};
      Value condition =
          rewriter.create<vector::ExtractOp>(loc, condMask, thisIdx);
      Value index = rewriter.create<vector::ExtractOp>(loc, indexVec, thisIdx);
      baseOffsets.back() =
          rewriter.createOrFold<arith::AddIOp>(loc, lastBaseOffset, index);

      auto loadBuilder = [&](OpBuilder &b, Location loc) {
        Value extracted;
        if (isa<MemRefType>(base.getType())) {
          // `vector.load` does not support scalar result; emit a vector load
          // and extract the single result instead.
          Value load =
              b.create<vector::LoadOp>(loc, elemVecTy, base, baseOffsets);
          int64_t zeroIdx[1] = {0};
          extracted = b.create<vector::ExtractOp>(loc, load, zeroIdx);
        } else {
          extracted = b.create<tensor::ExtractOp>(loc, base, baseOffsets);
        }

        Value newResult =
            b.create<vector::InsertOp>(loc, extracted, result, thisIdx);
        b.create<scf::YieldOp>(loc, newResult);
      };
      auto passThruBuilder = [result](OpBuilder &b, Location loc) {
        b.create<scf::YieldOp>(loc, result);
      };

      result =
          rewriter
              .create<scf::IfOp>(loc, condition, /*thenBuilder=*/loadBuilder,
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
  patterns.add<FlattenGather, RemoveStrideFromGatherSource,
               Gather1DToConditionalLoads>(patterns.getContext(), benefit);
}
