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
/// Progressive lowering of BroadcastOp.
class BroadcastOpLowering : public OpRewritePattern<vector::BroadcastOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    VectorType dstType = op.getResultVectorType();
    VectorType srcType = dyn_cast<VectorType>(op.getSourceType());
    Type eltType = dstType.getElementType();

    // Scalar to any vector can use splat.
    if (!srcType) {
      rewriter.replaceOpWithNewOp<vector::SplatOp>(op, dstType, op.getSource());
      return success();
    }

    // Determine rank of source and destination.
    int64_t srcRank = srcType.getRank();
    int64_t dstRank = dstType.getRank();

    // Stretching scalar inside vector (e.g. vector<1xf32>) can use splat.
    if (srcRank <= 1 && dstRank == 1) {
      Value ext;
      if (srcRank == 0)
        ext = rewriter.create<vector::ExtractElementOp>(loc, op.getSource());
      else
        ext = rewriter.create<vector::ExtractOp>(loc, op.getSource(), 0);
      rewriter.replaceOpWithNewOp<vector::SplatOp>(op, dstType, ext);
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
          rewriter.create<vector::BroadcastOp>(loc, resType, op.getSource());
      Value result = rewriter.create<ub::PoisonOp>(loc, dstType);
      for (int64_t d = 0, dim = dstType.getDimSize(0); d < dim; ++d)
        result = rewriter.create<vector::InsertOp>(loc, bcst, result, d);
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
    Value result = rewriter.create<ub::PoisonOp>(loc, dstType);
    if (m == 0) {
      // Stetch at start.
      Value ext = rewriter.create<vector::ExtractOp>(loc, op.getSource(), 0);
      Value bcst = rewriter.create<vector::BroadcastOp>(loc, resType, ext);
      for (int64_t d = 0, dim = dstType.getDimSize(0); d < dim; ++d)
        result = rewriter.create<vector::InsertOp>(loc, bcst, result, d);
    } else {
      // Stetch not at start.
      if (dstType.getScalableDims()[0]) {
        // TODO: For scalable vectors we should emit an scf.for loop.
        return failure();
      }
      for (int64_t d = 0, dim = dstType.getDimSize(0); d < dim; ++d) {
        Value ext = rewriter.create<vector::ExtractOp>(loc, op.getSource(), d);
        Value bcst = rewriter.create<vector::BroadcastOp>(loc, resType, ext);
        result = rewriter.create<vector::InsertOp>(loc, bcst, result, d);
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
