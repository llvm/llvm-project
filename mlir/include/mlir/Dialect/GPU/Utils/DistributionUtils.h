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

template <typename T>
struct WarpDistributionPattern : OpRewritePattern<WarpExecuteOnLane0Op> {
  using OpRewritePattern<WarpExecuteOnLane0Op>::OpRewritePattern;
  virtual LogicalResult
  matchAndRewrite(T op, PatternRewriter &rewriter) const override = 0;

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

template <typename T>
WarpExecuteOnLane0Op
WarpDistributionPattern<T>::moveRegionToNewWarpOpAndReplaceReturns(
    RewriterBase &rewriter, WarpExecuteOnLane0Op warpOp,
    ValueRange newYieldedValues, TypeRange newReturnTypes) {
  // Create a new op before the existing one, with the extra operands.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(warpOp);
  auto newWarpOp = rewriter.create<WarpExecuteOnLane0Op>(
      warpOp.getLoc(), newReturnTypes, warpOp.getLaneid(), warpOp.getWarpSize(),
      warpOp.getArgs(), warpOp.getBody()->getArgumentTypes());

  Region &opBody = warpOp.getBodyRegion();
  Region &newOpBody = newWarpOp.getBodyRegion();
  Block &newOpFirstBlock = newOpBody.front();
  rewriter.inlineRegionBefore(opBody, newOpBody, newOpBody.begin());
  rewriter.eraseBlock(&newOpFirstBlock);
  assert(newWarpOp.getWarpRegion().hasOneBlock() &&
         "expected WarpOp with single block");

  auto yield =
      cast<gpu::YieldOp>(newOpBody.getBlocks().begin()->getTerminator());

  rewriter.modifyOpInPlace(
      yield, [&]() { yield.getValuesMutable().assign(newYieldedValues); });
  return newWarpOp;
}

template <typename T>
WarpExecuteOnLane0Op
WarpDistributionPattern<T>::moveRegionToNewWarpOpAndAppendReturns(
    RewriterBase &rewriter, WarpExecuteOnLane0Op warpOp,
    ValueRange newYieldedValues, TypeRange newReturnTypes,
    llvm::SmallVector<size_t> &indices) {
  SmallVector<Type> types(warpOp.getResultTypes().begin(),
                          warpOp.getResultTypes().end());
  auto yield = cast<gpu::YieldOp>(
      warpOp.getBodyRegion().getBlocks().begin()->getTerminator());
  llvm::SmallSetVector<Value, 32> yieldValues(yield.getOperands().begin(),
                                              yield.getOperands().end());
  for (auto newRet : llvm::zip(newYieldedValues, newReturnTypes)) {
    if (yieldValues.insert(std::get<0>(newRet))) {
      types.push_back(std::get<1>(newRet));
      indices.push_back(yieldValues.size() - 1);
    } else {
      // If the value already exit the region don't create a new output.
      for (auto [idx, yieldOperand] :
           llvm::enumerate(yieldValues.getArrayRef())) {
        if (yieldOperand == std::get<0>(newRet)) {
          indices.push_back(idx);
          break;
        }
      }
    }
  }
  yieldValues.insert(newYieldedValues.begin(), newYieldedValues.end());
  WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndReplaceReturns(
      rewriter, warpOp, yieldValues.getArrayRef(), types);
  rewriter.replaceOp(warpOp,
                     newWarpOp.getResults().take_front(warpOp.getNumResults()));
  return newWarpOp;
}

template <typename T>
OpOperand *WarpDistributionPattern<T>::getWarpResult(
    WarpExecuteOnLane0Op warpOp, const std::function<bool(Operation *)> &fn) {
  auto yield = cast<gpu::YieldOp>(
      warpOp.getBodyRegion().getBlocks().begin()->getTerminator());
  for (OpOperand &yieldOperand : yield->getOpOperands()) {
    Value yieldValues = yieldOperand.get();
    Operation *definedOp = yieldValues.getDefiningOp();
    if (definedOp && fn(definedOp)) {
      if (!warpOp.getResult(yieldOperand.getOperandNumber()).use_empty())
        return &yieldOperand;
    }
  }
  return {};
}

template <typename T>
bool WarpDistributionPattern<T>::delinearizeLaneId(
    OpBuilder &builder, Location loc, ArrayRef<int64_t> originalShape,
    ArrayRef<int64_t> distributedShape, int64_t warpSize, Value laneId,
    SmallVectorImpl<Value> &delinearizedIds) {
  // If the original shape and the distributed shape is the same, we don't
  // distribute at all--every thread is handling the whole. For such case, we
  // should not rely on lane IDs later. So just return an empty lane ID vector.
  if (originalShape == distributedShape) {
    delinearizedIds.clear();
    return true;
  }

  SmallVector<int64_t> sizes;
  for (auto [large, small] : llvm::zip_equal(originalShape, distributedShape)) {
    if (large % small != 0)
      return false;
    sizes.push_back(large / small);
  }
  if (std::accumulate(sizes.begin(), sizes.end(), 1,
                      std::multiplies<int64_t>()) != warpSize)
    return false;

  AffineExpr s0, s1;
  bindSymbols(builder.getContext(), s0, s1);

  int64_t usedThreads = 1;

  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  delinearizedIds.assign(sizes.size(), zero);

  for (int i = sizes.size() - 1; i >= 0; --i) {
    usedThreads *= sizes[i];
    if (usedThreads == warpSize) {
      // We've used up all available threads. Don't need to perform modulo
      // anymore. And we can stop the calculation for further dimensions.
      delinearizedIds[i] = laneId;
      break;
    }
    delinearizedIds[i] =
        affine::makeComposedAffineApply(builder, loc, s0 % sizes[i], {laneId});
    laneId = affine::makeComposedAffineApply(
        builder, loc, s0.floorDiv(usedThreads), {laneId});
  }
  return true;
}

} // namespace gpu
} // namespace mlir

#endif // MLIR_DIALECT_GPU_TRANSFORMS_DISTRIBUTIONUTILS_H_
