//===- MemorySlotOpInterfaceImpl.cpp - Mem2Reg for Vector ops -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements Mem2Reg `PromotableMemOpInterface` models for
// `vector.transfer_read` and `vector.transfer_write`. These models promote a
// memref slot to a single vector SSA value. The models for `memref.copy` and
// `memref.subview` live in the MemRef dialect.
//
// Transfers must meet the criteria in `isPromotableTransfer`: their vector type
// must match the slot's value type, their indices must be zero, and their
// permutation map must be the identity. A read uses the slot's current value;
// a write defines its next value. For transfers with a mask operand or through
// a dynamic subview, `arith.select` supplies padding for inactive read lanes
// and preserves the previous value for inactive write lanes.
//
// Transfers wrapped in `vector.mask` prevent promotion because that op does
// not implement `PromotableRegionOpInterface`, which Mem2Reg requires to
// promote accesses inside a region.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/Transforms/MemorySlotOpInterfaceImpl.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/MemorySlotOpInterfaceImpl.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"

using namespace mlir;
using namespace mlir::vector;

//===----------------------------------------------------------------------===//
//  Utilities
//===----------------------------------------------------------------------===//

/// Returns whether `xferOp` can be promoted to a load/store of `slot`'s vector
/// value. This requires that the transfer's sole use of the slot is as its
/// base, the transferred vector type equals `slot.valueType`, the indices are
/// all zero (origin), and the permutation map is the identity.
///
/// Partial accesses through dynamic subviews or with mask operands are
/// reconstructed with `arith.select` during promotion.
///
/// For a dynamic subview, the aliaser masks writes to preserve the parent value
/// outside the subview's extent. Reads are masked here because they need the
/// transfer's padding operand, which the aliaser does not have.
///
/// A transfer's mask operand is combined with the subview mask using
/// `arith.andi` for reads. For writes, `getStored` applies it with a separate
/// `arith.select` to preserve inactive lanes.
static bool
isPromotableTransfer(VectorTransferOpInterface xferOp, const MemorySlot &slot,
                     const SmallPtrSetImpl<OpOperand *> &blockingUses) {
  // The sole blocking use must be the slot pointer as the transfer's base.
  if (blockingUses.size() != 1)
    return false;
  Value blockingUse = (*blockingUses.begin())->get();
  if (blockingUse != slot.ptr || xferOp.getBase() != slot.ptr)
    return false;

  // Reject the tensor form (already implied, since slot pointers are memrefs).
  if (!isa<MemRefType>(xferOp.getBase().getType()))
    return false;

  // Exact type match pins rank/extents/element type/scalable dims.
  if (xferOp.getVectorType() != slot.valueType)
    return false;

  // Access must start at the buffer origin in every dimension.
  for (Value index : xferOp.getIndices()) {
    std::optional<int64_t> constIndex = getConstantIntValue(index);
    if (!constIndex || *constIndex != 0)
      return false;
  }

  // Identity map: no broadcast or transpose.
  if (!xferOp.getPermutationMap().isIdentity())
    return false;

  // Out-of-bounds is allowed only for a dynamic view.
  if (xferOp.hasOutOfBoundsDim() && !memref::isDynamicSubViewSlot(slot.ptr))
    return false;

  return true;
}

//===----------------------------------------------------------------------===//
//  Interface models
//===----------------------------------------------------------------------===//

namespace {
/// Mem2Reg model for a `vector.transfer_read` of the slot.
struct TransferReadOpMemOpModel
    : public PromotableMemOpInterface::ExternalModel<TransferReadOpMemOpModel,
                                                     vector::TransferReadOp> {
  bool loadsFrom(Operation *op, const MemorySlot &slot) const {
    return cast<vector::TransferReadOp>(op).getBase() == slot.ptr;
  }

  bool storesTo(Operation *op, const MemorySlot &slot) const { return false; }

  Value getStored(Operation *op, const MemorySlot &slot, OpBuilder &builder,
                  Value reachingDef, const DataLayout &dataLayout) const {
    llvm_unreachable("getStored should not be called on TransferReadOp");
  }

  bool canUsesBeRemoved(Operation *op, const MemorySlot &slot,
                        const SmallPtrSetImpl<OpOperand *> &blockingUses,
                        SmallVectorImpl<OpOperand *> &newBlockingUses,
                        const DataLayout &dataLayout) const {
    return isPromotableTransfer(cast<VectorTransferOpInterface>(op), slot,
                                blockingUses);
  }

  // Replace the read with the reaching definition, selecting padding for lanes
  // outside the subview's extent or disabled by the transfer's mask.
  DeletionKind
  removeBlockingUses(Operation *op, const MemorySlot &slot,
                     const SmallPtrSetImpl<OpOperand *> &blockingUses,
                     OpBuilder &builder, Value reachingDefinition,
                     const DataLayout &dataLayout) const {
    auto readOp = cast<vector::TransferReadOp>(op);
    Location loc = op->getLoc();
    Value mask = memref::buildDynamicSubViewMask(builder, loc, slot.ptr);
    if (Value opMask = readOp.getMask())
      mask = mask
                 ? arith::AndIOp::create(builder, loc, mask, opMask).getResult()
                 : opMask;

    Value result = reachingDefinition;
    if (mask) {
      Value padSplat = vector::BroadcastOp::create(
          builder, loc, readOp.getVectorType(), readOp.getPadding());
      result = arith::SelectOp::create(builder, loc, mask, reachingDefinition,
                                       padSplat);
    }
    readOp.getVector().replaceAllUsesWith(result);
    return DeletionKind::Delete;
  }
};

/// Mem2Reg model for a `vector.transfer_write` to the slot, which qualifies
/// under the same `isPromotableTransfer` criteria.
struct TransferWriteOpMemOpModel
    : public PromotableMemOpInterface::ExternalModel<TransferWriteOpMemOpModel,
                                                     vector::TransferWriteOp> {
  bool loadsFrom(Operation *op, const MemorySlot &slot) const { return false; }

  bool storesTo(Operation *op, const MemorySlot &slot) const {
    return cast<vector::TransferWriteOp>(op).getBase() == slot.ptr;
  }

  // The stored value is the transfer's vector, with inactive lanes preserved
  // from the reaching definition when the transfer has a mask.
  Value getStored(Operation *op, const MemorySlot &slot, OpBuilder &builder,
                  Value reachingDef, const DataLayout &dataLayout) const {
    auto writeOp = cast<vector::TransferWriteOp>(op);
    Value stored = writeOp.getValueToStore();
    // A dynamic-view extent is composed separately, by the aliaser's
    // `projectAliasValueToSlotValue`.
    if (Value mask = writeOp.getMask())
      stored = arith::SelectOp::create(builder, op->getLoc(), mask, stored,
                                       reachingDef);
    return stored;
  }

  bool canUsesBeRemoved(Operation *op, const MemorySlot &slot,
                        const SmallPtrSetImpl<OpOperand *> &blockingUses,
                        SmallVectorImpl<OpOperand *> &newBlockingUses,
                        const DataLayout &dataLayout) const {
    return isPromotableTransfer(cast<VectorTransferOpInterface>(op), slot,
                                blockingUses);
  }

  // `getStored` already provided the value for later uses, so the write can
  // simply be erased.
  DeletionKind
  removeBlockingUses(Operation *op, const MemorySlot &slot,
                     const SmallPtrSetImpl<OpOperand *> &blockingUses,
                     OpBuilder &builder, Value reachingDefinition,
                     const DataLayout &dataLayout) const {
    return DeletionKind::Delete;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
//  Register external models
//===----------------------------------------------------------------------===//

void mlir::vector::registerMemorySlotExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, vector::VectorDialect *dialect) {
    TransferReadOp::attachInterface<TransferReadOpMemOpModel>(*ctx);
    TransferWriteOp::attachInterface<TransferWriteOpMemOpModel>(*ctx);
  });
}
