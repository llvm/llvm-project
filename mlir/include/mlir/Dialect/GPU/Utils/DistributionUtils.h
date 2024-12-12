//===- VectorDistributionUtils.h - Distribution Utilities -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GPU_TRANSFORMS_DISTRIBUTIONUTILS_H_
#define MLIR_DIALECT_GPU_TRANSFORMS_DISTRIBITIONUTILS_H_

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

#include <numeric>
#include <utility>

namespace mlir {
namespace gpu {
/// Move scalar operations with no dependency on the warp op outside of the
/// region.
void moveScalarUniformCode(gpu::WarpExecuteOnLane0Op op);

struct WarpDistributionPattern : OpRewritePattern<WarpExecuteOnLane0Op> {
  using OpRewritePattern<WarpExecuteOnLane0Op>::OpRewritePattern;
  virtual LogicalResult
  matchAndRewrite(WarpExecuteOnLane0Op op,
                  PatternRewriter &rewriter) const override = 0;

protected:
  /// Return a value yielded by `warpOp` which statifies the filter lamdba
  /// condition and is not dead.
  static OpOperand *getWarpResult(WarpExecuteOnLane0Op warpOp,
                                  const std::function<bool(Operation *)> &fn);

  /// Helper to create a new WarpExecuteOnLane0Op with different signature.
  static WarpExecuteOnLane0Op moveRegionToNewWarpOpAndReplaceReturns(
      RewriterBase &rewriter, WarpExecuteOnLane0Op warpOp,
      ValueRange newYieldedValues, TypeRange newReturnTypes);

  /// Helper to create a new WarpExecuteOnLane0Op region with extra outputs.
  /// `indices` return the index of each new output.
  static WarpExecuteOnLane0Op moveRegionToNewWarpOpAndAppendReturns(
      RewriterBase &rewriter, WarpExecuteOnLane0Op warpOp,
      ValueRange newYieldedValues, TypeRange newReturnTypes,
      llvm::SmallVector<size_t> &indices);

  /// Delinearize the given `laneId` into multiple dimensions, where each
  /// dimension's size is determined by `originalShape` and `distributedShape`
  /// together. This function expects the total numbers of threads needed for
  /// distribution is equal to `warpSize`. Returns true and updates
  /// `delinearizedIds` if so.
  static bool delinearizeLaneId(OpBuilder &builder, Location loc,
                                ArrayRef<int64_t> originalShape,
                                ArrayRef<int64_t> distributedShape,
                                int64_t warpSize, Value laneId,
                                SmallVectorImpl<Value> &delinearizedIds);
};

} // namespace gpu
} // namespace mlir

#endif // MLIR_DIALECT_GPU_TRANSFORMS_DISTRIBUTIONUTILS_H_
