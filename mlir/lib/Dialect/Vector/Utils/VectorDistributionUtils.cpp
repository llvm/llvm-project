//===- VectorDistributionUtils.cpp - Distribution tools for VectorOps -----===//
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

#include "mlir/Dialect/Vector/Utils/VectorDistributionUtils.h"

#include "mlir/IR/Value.h"

using namespace mlir;
using namespace mlir::vector;

mlir::OpOperand *
mlir::vector::getWarpResult(WarpExecuteOnLane0Op warpOp,
                            const std::function<bool(Operation *)> &fn) {
  auto yield = cast<vector::YieldOp>(
      warpOp.getBodyRegion().getBlocks().begin()->getTerminator());
  for (mlir::OpOperand &yieldOperand : yield->getOpOperands()) {
    Value yieldValues = yieldOperand.get();
    Operation *definedOp = yieldValues.getDefiningOp();
    if (definedOp && fn(definedOp)) {
      if (!warpOp.getResult(yieldOperand.getOperandNumber()).use_empty())
        return &yieldOperand;
    }
  }
  return {};
}

WarpExecuteOnLane0Op mlir::vector::moveRegionToNewWarpOpAndReplaceReturns(
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
      cast<vector::YieldOp>(newOpBody.getBlocks().begin()->getTerminator());

  rewriter.modifyOpInPlace(
      yield, [&]() { yield.getOperandsMutable().assign(newYieldedValues); });
  return newWarpOp;
}

WarpExecuteOnLane0Op mlir::vector::moveRegionToNewWarpOpAndAppendReturns(
    RewriterBase &rewriter, WarpExecuteOnLane0Op warpOp,
    ValueRange newYieldedValues, TypeRange newReturnTypes,
    llvm::SmallVector<size_t> &indices) {
  SmallVector<Type> types(warpOp.getResultTypes().begin(),
                          warpOp.getResultTypes().end());
  auto yield = cast<vector::YieldOp>(
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
