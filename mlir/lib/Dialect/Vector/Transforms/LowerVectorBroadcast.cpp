//===- LowerVectorBroadcast.cpp - Lower 'vector.broadcast' operation ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements target-independent rewrites and utilities to lower the
// 'vector.broadcast' operation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#define DEBUG_TYPE "vector-broadcast-lowering"

using namespace mlir;
using namespace mlir::vector;

namespace {

/// Convert a vector.broadcast with a vector operand to a lower rank
/// vector.broadcast. vector.broadcast with a scalar operand is expected to be
/// convertible to the lower level target dialect (LLVM, SPIR-V, etc.) directly.
class BroadcastOpLowering : public OpRewritePattern<vector::BroadcastOp> {
public:
  using Base::Base;

  LogicalResult matchAndRewrite(vector::BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    VectorType dstType = op.getResultVectorType();
    VectorType srcType = dyn_cast<VectorType>(op.getSourceType());
    Type eltType = dstType.getElementType();

    // A broadcast from a scalar is considered to be in the lowered form.
    if (!srcType)
      return rewriter.notifyMatchFailure(
          op, "broadcast from scalar already in lowered form");

    // Determine rank of source and destination.
    int64_t srcRank = srcType.getRank();
    int64_t dstRank = dstType.getRank();

    // Here we are broadcasting to a rank-1 vector. Ensure that the source is a
    // scalar.
    if (srcRank <= 1 && dstRank == 1) {
      SmallVector<int64_t> fullRankPosition(srcRank, 0);
      Value ext = vector::ExtractOp::create(rewriter, loc, op.getSource(),
                                            fullRankPosition);
      assert(!isa<VectorType>(ext.getType()) && "expected scalar");
      rewriter.replaceOpWithNewOp<vector::BroadcastOp>(op, dstType, ext);
      return success();
    }

    // Duplicate this rank.
    // For example:
    //   %x = broadcast %y  : k-D to n-D, k < n
    // becomes:
    //   %b = broadcast %y  : k-D to (n-1)-D
    //   %x = [%b,%b,%b,%b] : n-D
    // becomes:
    //   %b = [%y,%y]       : (n-1)-D
    //   %x = [%b,%b,%b,%b] : n-D
    if (srcRank < dstRank) {
      // Duplication.
      VectorType resType = VectorType::Builder(dstType).dropDim(0);
      Value bcst =
          vector::BroadcastOp::create(rewriter, loc, resType, op.getSource());
      Value result = ub::PoisonOp::create(rewriter, loc, dstType);
      for (int64_t d = 0, dim = dstType.getDimSize(0); d < dim; ++d)
        result = vector::InsertOp::create(rewriter, loc, bcst, result, d);
      rewriter.replaceOp(op, result);
      return success();
    }

    // Find non-matching dimension, if any.
    assert(srcRank == dstRank);
    int64_t m = -1;
    for (int64_t r = 0; r < dstRank; r++)
      if (srcType.getDimSize(r) != dstType.getDimSize(r)) {
        m = r;
        break;
      }

    // All trailing dimensions are the same. Simply pass through.
    if (m == -1) {
      rewriter.replaceOp(op, op.getSource());
      return success();
    }

    // Any non-matching dimension forces a stretch along this rank.
    // For example:
    //   %x = broadcast %y : vector<4x1x2xf32> to vector<4x2x2xf32>
    // becomes:
    //   %a = broadcast %y[0] : vector<1x2xf32> to vector<2x2xf32>
    //   %b = broadcast %y[1] : vector<1x2xf32> to vector<2x2xf32>
    //   %c = broadcast %y[2] : vector<1x2xf32> to vector<2x2xf32>
    //   %d = broadcast %y[3] : vector<1x2xf32> to vector<2x2xf32>
    //   %x = [%a,%b,%c,%d]
    // becomes:
    //   %u = broadcast %y[0][0] : vector<2xf32> to vector <2x2xf32>
    //   %v = broadcast %y[1][0] : vector<2xf32> to vector <2x2xf32>
    //   %a = [%u, %v]
    //   ..
    //   %x = [%a,%b,%c,%d]
    VectorType resType =
        VectorType::get(dstType.getShape().drop_front(), eltType,
                        dstType.getScalableDims().drop_front());
    Value result = ub::PoisonOp::create(rewriter, loc, dstType);
    if (m == 0) {
      // Stetch at start.
      Value ext = vector::ExtractOp::create(rewriter, loc, op.getSource(), 0);
      Value bcst = vector::BroadcastOp::create(rewriter, loc, resType, ext);
      for (int64_t d = 0, dim = dstType.getDimSize(0); d < dim; ++d)
        result = vector::InsertOp::create(rewriter, loc, bcst, result, d);
    } else {
      // Stetch not at start.
      if (dstType.getScalableDims()[0]) {
        // TODO: For scalable vectors we should emit an scf.for loop.
        return failure();
      }
      for (int64_t d = 0, dim = dstType.getDimSize(0); d < dim; ++d) {
        Value ext = vector::ExtractOp::create(rewriter, loc, op.getSource(), d);
        Value bcst = vector::BroadcastOp::create(rewriter, loc, resType, ext);
        result = vector::InsertOp::create(rewriter, loc, bcst, result, d);
      }
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

void mlir::vector::populateVectorBroadcastLoweringPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<BroadcastOpLowering>(patterns.getContext(), benefit);
}
