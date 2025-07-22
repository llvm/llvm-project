//===- DistributionUtils.cpp - Distribution tools for GPUOps --------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements distribution utility methods.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/Utils/DistributionUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Value.h"

#include <numeric>

using namespace mlir;
using namespace mlir::gpu;

WarpExecuteOnLane0Op
WarpDistributionPattern::moveRegionToNewWarpOpAndReplaceReturns(
    RewriterBase &rewriter, WarpExecuteOnLane0Op warpOp,
    ValueRange newYieldedValues, TypeRange newReturnTypes) const {
  // Create a new op before the existing one, with the extra operands.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(warpOp);
  auto newWarpOp = WarpExecuteOnLane0Op::create(
      rewriter, warpOp.getLoc(), newReturnTypes, warpOp.getLaneid(),
      warpOp.getWarpSize(), warpOp.getArgs(),
      warpOp.getBody()->getArgumentTypes());

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

WarpExecuteOnLane0Op
WarpDistributionPattern::moveRegionToNewWarpOpAndAppendReturns(
    RewriterBase &rewriter, WarpExecuteOnLane0Op warpOp,
    ValueRange newYieldedValues, TypeRange newReturnTypes,
    SmallVector<size_t> &indices) const {
  SmallVector<Type> types(warpOp.getResultTypes().begin(),
                          warpOp.getResultTypes().end());
  auto yield = cast<gpu::YieldOp>(
      warpOp.getBodyRegion().getBlocks().begin()->getTerminator());
  llvm::SmallSetVector<Value, 32> yieldValues(yield.getOperands().begin(),
                                              yield.getOperands().end());
  for (auto [value, type] : llvm::zip_equal(newYieldedValues, newReturnTypes)) {
    if (yieldValues.insert(value)) {
      types.push_back(type);
      indices.push_back(yieldValues.size() - 1);
    } else {
      // If the value already exit the region don't create a new output.
      for (auto [idx, yieldOperand] :
           llvm::enumerate(yieldValues.getArrayRef())) {
        if (yieldOperand == value) {
          indices.push_back(idx);
          break;
        }
      }
    }
  }
  yieldValues.insert_range(newYieldedValues);
  WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndReplaceReturns(
      rewriter, warpOp, yieldValues.getArrayRef(), types);
  rewriter.replaceOp(warpOp,
                     newWarpOp.getResults().take_front(warpOp.getNumResults()));
  return newWarpOp;
}

OpOperand *WarpDistributionPattern::getWarpResult(
    WarpExecuteOnLane0Op warpOp,
    llvm::function_ref<bool(Operation *)> fn) const {
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
  return nullptr;
}

bool WarpDistributionPattern::delinearizeLaneId(
    OpBuilder &builder, Location loc, ArrayRef<int64_t> originalShape,
    ArrayRef<int64_t> distributedShape, int64_t warpSize, Value laneId,
    SmallVectorImpl<Value> &delinearizedIds) const {
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

  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
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
