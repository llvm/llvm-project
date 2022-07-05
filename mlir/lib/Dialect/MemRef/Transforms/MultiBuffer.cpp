//===----------- MultiBuffering.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements multi buffering transformation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

using namespace mlir;

/// Return true if the op fully overwrite the given `buffer` value.
static bool overrideBuffer(Operation *op, Value buffer) {
  auto copyOp = dyn_cast<memref::CopyOp>(op);
  if (!copyOp)
    return false;
  return copyOp.target() == buffer;
}

/// Replace the uses of `oldOp` with the given `val` and for subview uses
/// propagate the type change. Changing the memref type may require propagating
/// it through subview ops so we cannot just do a replaceAllUse but need to
/// propagate the type change and erase old subview ops.
static void replaceUsesAndPropagateType(Operation *oldOp, Value val,
                                        OpBuilder &builder) {
  SmallVector<Operation *> opToDelete;
  SmallVector<OpOperand *> operandsToReplace;
  for (OpOperand &use : oldOp->getUses()) {
    auto subviewUse = dyn_cast<memref::SubViewOp>(use.getOwner());
    if (!subviewUse) {
      // Save the operand to and replace outside the loop to not invalidate the
      // iterator.
      operandsToReplace.push_back(&use);
      continue;
    }
    builder.setInsertionPoint(subviewUse);
    Type newType = memref::SubViewOp::inferRankReducedResultType(
        subviewUse.getType().getShape(), val.getType().cast<MemRefType>(),
        extractFromI64ArrayAttr(subviewUse.static_offsets()),
        extractFromI64ArrayAttr(subviewUse.static_sizes()),
        extractFromI64ArrayAttr(subviewUse.static_strides()));
    Value newSubview = builder.create<memref::SubViewOp>(
        subviewUse->getLoc(), newType.cast<MemRefType>(), val,
        subviewUse.getMixedOffsets(), subviewUse.getMixedSizes(),
        subviewUse.getMixedStrides());
    replaceUsesAndPropagateType(subviewUse, newSubview, builder);
    opToDelete.push_back(use.getOwner());
  }
  for (OpOperand *operand : operandsToReplace)
    operand->set(val);
  // Clean up old subview ops.
  for (Operation *op : opToDelete)
    op->erase();
}

/// Helper to convert get a value from an OpFoldResult or create it at the
/// builder insert point.
static Value getOrCreateValue(OpFoldResult res, OpBuilder &builder,
                              Location loc) {
  Value value = res.dyn_cast<Value>();
  if (value)
    return value;
  return builder.create<arith::ConstantIndexOp>(
      loc, res.dyn_cast<Attribute>().cast<IntegerAttr>().getInt());
}

// Transformation to do multi-buffering/array expansion to remove dependencies
// on the temporary allocation between consecutive loop iterations.
// Returns success if the transformation happened and failure otherwise.
// This is not a pattern as it requires propagating the new memref type to its
// uses and requires updating subview ops.
LogicalResult mlir::memref::multiBuffer(memref::AllocOp allocOp,
                                        unsigned multiplier) {
  DominanceInfo dom(allocOp->getParentOp());
  LoopLikeOpInterface candidateLoop;
  for (Operation *user : allocOp->getUsers()) {
    auto parentLoop = user->getParentOfType<LoopLikeOpInterface>();
    if (!parentLoop)
      return failure();
    /// Make sure there is no loop carried dependency on the allocation.
    if (!overrideBuffer(user, allocOp.getResult()))
      continue;
    // If this user doesn't dominate all the other users keep looking.
    if (llvm::any_of(allocOp->getUsers(), [&](Operation *otherUser) {
          return !dom.dominates(user, otherUser);
        }))
      continue;
    candidateLoop = parentLoop;
    break;
  }
  if (!candidateLoop)
    return failure();
  llvm::Optional<Value> inductionVar = candidateLoop.getSingleInductionVar();
  llvm::Optional<OpFoldResult> lowerBound = candidateLoop.getSingleLowerBound();
  llvm::Optional<OpFoldResult> singleStep = candidateLoop.getSingleStep();
  if (!inductionVar || !lowerBound || !singleStep)
    return failure();
  OpBuilder builder(candidateLoop);
  Value stepValue =
      getOrCreateValue(*singleStep, builder, candidateLoop->getLoc());
  Value lowerBoundValue =
      getOrCreateValue(*lowerBound, builder, candidateLoop->getLoc());
  SmallVector<int64_t, 4> newShape(1, multiplier);
  ArrayRef<int64_t> oldShape = allocOp.getType().getShape();
  newShape.append(oldShape.begin(), oldShape.end());
  auto newMemref = MemRefType::get(newShape, allocOp.getType().getElementType(),
                                   MemRefLayoutAttrInterface(),
                                   allocOp.getType().getMemorySpace());
  builder.setInsertionPoint(allocOp);
  Location loc = allocOp->getLoc();
  auto newAlloc = builder.create<memref::AllocOp>(loc, newMemref);
  builder.setInsertionPoint(&candidateLoop.getLoopBody().front(),
                            candidateLoop.getLoopBody().front().begin());
  AffineExpr induc = getAffineDimExpr(0, allocOp.getContext());
  AffineExpr init = getAffineDimExpr(1, allocOp.getContext());
  AffineExpr step = getAffineDimExpr(2, allocOp.getContext());
  AffineExpr expr = ((induc - init).floorDiv(step)) % multiplier;
  auto map = AffineMap::get(3, 0, expr);
  std::array<Value, 3> operands = {*inductionVar, lowerBoundValue, stepValue};
  Value bufferIndex = builder.create<AffineApplyOp>(loc, map, operands);
  SmallVector<OpFoldResult> offsets, sizes, strides;
  offsets.push_back(bufferIndex);
  offsets.append(oldShape.size(), builder.getIndexAttr(0));
  strides.assign(oldShape.size() + 1, builder.getIndexAttr(1));
  sizes.push_back(builder.getIndexAttr(1));
  for (int64_t size : oldShape)
    sizes.push_back(builder.getIndexAttr(size));
  auto dstMemref =
      memref::SubViewOp::inferRankReducedResultType(
          allocOp.getType().getShape(), newMemref, offsets, sizes, strides)
          .cast<MemRefType>();
  Value subview = builder.create<memref::SubViewOp>(loc, dstMemref, newAlloc,
                                                    offsets, sizes, strides);
  replaceUsesAndPropagateType(allocOp, subview, builder);
  allocOp.erase();
  return success();
}
