//===- MemorySlotOpInterfaceImpl.cpp - Mem2Reg for Vector ops -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Mem2Reg `PromotableMemOpInterface` models for the
// vector transfer ops, which let a memref be promoted into a single vector SSA
// value. The models for the memref ops that view or copy such a buffer live in
// the MemRef dialect. Mem2Reg calls the memory it promotes a *slot*: a pointer
// paired with the type of the value it becomes, here a memref and its vector
// type. Promoting a slot replaces it with that value, used as the slot's
// reaching definition.
//
// A slot is promoted when each of its uses is an access these models can
// rewrite: a `vector.transfer_read` or `vector.transfer_write` meeting the
// criteria in `isPromotableTransfer`. A read becomes a use of the slot's
// current value; a write becomes a new definition of it. A transfer with a mask
// operand covers only its active lanes, so it is composed with an
// `arith.select`.
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
/// Two forms of partial access are accepted (rather than rejected) and
/// reconstructed with a `select` during promotion: an out-of-bounds transfer
/// through a dynamic-subview alias, and a transfer with a mask operand.
///
/// Promoting the former takes two steps: the subview aliaser projects between
/// the parent's value and the alias's -- down to the alias for a read, up to
/// the parent for a write -- and the models here rewrite the transfer itself.
/// The view's mask is applied in the projection for a write and in the rewrite
/// for a read:
///   - a write is masked by the up projection: it selects the stored value over
///     the reaching value the aliaser is handed, so lanes outside the extent
///     keep what the parent held;
///   - a read is masked by the rewrite here, because the down projection is the
///     identity: it selects the parent's value over the read's own padding
///     operand, which the aliaser never sees.
///
/// A mask operand is composed on top of that: `arith.andi` with the view's mask
/// on a read, a second `select` over the reaching value in `getStored` on a
/// write.
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
  if (xferOp.hasOutOfBoundsDim() && !memref::isDynamicViewSlot(slot.ptr))
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

  // Promotion replaces the slot with a vector value, so the read becomes a use
  // of that value: the value itself, or a select bringing in the transfer's
  // padding on the lanes it does not read -- those outside the dynamic-view
  // extent or its own mask.
  DeletionKind
  removeBlockingUses(Operation *op, const MemorySlot &slot,
                     const SmallPtrSetImpl<OpOperand *> &blockingUses,
                     OpBuilder &builder, Value reachingDefinition,
                     const DataLayout &dataLayout) const {
    auto readOp = cast<vector::TransferReadOp>(op);
    Location loc = op->getLoc();
    Value mask = memref::buildDynamicViewMask(builder, loc, slot.ptr);
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

  // Promotion replaces the slot with a vector value, so the write becomes the
  // code producing that value: the transfer's vector, or a select of it over
  // the previous value when the write is masked.
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

  // `getStored` already produced the value and the driver threaded it to the
  // slot's later uses, so nothing is emitted here: the write disappears along
  // with the buffer.
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
