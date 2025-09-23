//===- LowerVectorScatter.cpp - Lower 'vector.scatter' operation ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements target-independent rewrites and utilities to lower the
// 'vector.scatter' operation.
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

#define DEBUG_TYPE "vector-scatter-lowering"

using namespace mlir;
using namespace mlir::vector;

namespace {

/// Unrolls 2 or more dimensional `vector.scatter` ops by unrolling the
/// outermost dimension. For example:
/// ```
/// vector.scatter %base[%c0][%idx], %mask, %value :
///        memref<?xf32>, vector<2x3xi32>, vector<2x3xi1>, vector<2x3xf32>
///
/// ==>
///
/// %v0 = vector.extract %value[0] : vector<3xf32> from vector<2x3xf32>
/// %m0 = vector.extract %mask[0] : vector<3xi1> from vector<2x3xi1>
/// %i0 = vector.extract %idx[0] : vector<3xi32> from vector<2x3xi32>
/// vector.scatter %base[%c0][%i0], %m0, %v0 :
///     memref<?xf32>, vector<3xi32>, vector<3xi1>, vector<3xf32>
///
/// %v1 = vector.extract %value[1] : vector<3xf32> from vector<2x3xf32>
/// %m1 = vector.extract %mask[1] : vector<3xi1> from vector<2x3xi1>
/// %i1 = vector.extract %idx[1] : vector<3xi32> from vector<2x3xi32>
/// vector.scatter %base[%c0][%i1], %m1, %v1 :
///     memref<?xf32>, vector<3xi32>, vector<3xi1>, vector<3xf32>
/// ```
///
/// When applied exhaustively, this will produce a sequence of 1-d scatter ops.
///
/// Supports vector types with a fixed leading dimension.
struct UnrollScatter : OpRewritePattern<vector::ScatterOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ScatterOp op,
                                PatternRewriter &rewriter) const override {
    Value indexVec = op.getIndices();
    Value maskVec = op.getMask();
    Value valueVec = op.getValueToStore();

    // Get the vector type from one of the vector operands
    VectorType vectorTy = dyn_cast<VectorType>(indexVec.getType());
    if (!vectorTy)
      return failure();

    auto unrollScatterFn = [&](PatternRewriter &rewriter, Location loc,
                               VectorType subTy, int64_t index) {
      int64_t thisIdx[1] = {index};

      Value indexSubVec =
          vector::ExtractOp::create(rewriter, loc, indexVec, thisIdx);
      Value maskSubVec =
          vector::ExtractOp::create(rewriter, loc, maskVec, thisIdx);
      Value valueSubVec =
          vector::ExtractOp::create(rewriter, loc, valueVec, thisIdx);

      rewriter.create<vector::ScatterOp>(loc, op.getBase(), op.getOffsets(),
                                         indexSubVec, maskSubVec, valueSubVec,
                                         op.getAlignmentAttr());

      // Return a dummy value since unrollVectorOp expects a Value
      return rewriter.create<ub::PoisonOp>(loc, subTy);
    };

    return unrollVectorOp(op, rewriter, unrollScatterFn, vectorTy);
  }
};

} // namespace

void mlir::vector::populateVectorScatterLoweringPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<UnrollScatter>(patterns.getContext(), benefit);
}
