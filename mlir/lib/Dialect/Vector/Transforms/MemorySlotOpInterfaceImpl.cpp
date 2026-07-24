//===- MemorySlotOpInterfaceImpl.cpp - Mem2Reg for vector ops -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements Mem2Reg-related interfaces for Vector dialect
// operations. It allows a memref that is only ever accessed as a whole buffer
// through `vector.transfer_read`/`vector.transfer_write` to be promoted into a
// single vector SSA value.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/Transforms/MemorySlotOpInterfaceImpl.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"

using namespace mlir;
using namespace mlir::vector;

//===----------------------------------------------------------------------===//
//  Utilities
//===----------------------------------------------------------------------===//

/// Returns whether the transfer operation `xferOp` accesses exactly the whole
/// contents of the memory slot `slot`, so that it can be treated as a plain
/// whole-buffer load or store during Mem2Reg. `blockingUses` are the uses of
/// the slot pointer that this operation must stop using for promotion.
template <typename TransferOpTy>
static bool
isWholeBufferTransfer(TransferOpTy xferOp, const MemorySlot &slot,
                      const SmallPtrSetImpl<OpOperand *> &blockingUses) {
  // The only blocking use must be the slot pointer itself.
  if (blockingUses.size() != 1)
    return false;
  Value blockingUse = (*blockingUses.begin())->get();
  if (blockingUse != slot.ptr || xferOp.getBase() != slot.ptr)
    return false;

  // Only the memref form can access a memref slot. This is already implied by
  // `getBase() == slot.ptr` above (slot pointers are always memrefs), but guard
  // defensively against the tensor form.
  if (!isa<MemRefType>(xferOp.getBase().getType()))
    return false;

  // The transferred vector must match the slot type exactly. This pins the
  // rank, per-dimension extent and element type, and rejects scalable vectors
  // (which never equal the fixed slot type).
  if (xferOp.getVectorType() != slot.elemType)
    return false;

  // All indices must be constant zero so the access starts at the buffer
  // origin in every dimension.
  for (Value index : xferOp.getIndices()) {
    std::optional<int64_t> constIndex = getConstantIntValue(index);
    if (!constIndex || *constIndex != 0)
      return false;
  }

  // The permutation map must be the identity: no broadcast, no transpose.
  if (!xferOp.getPermutationMap().isIdentity())
    return false;

  // Every dimension must be in bounds so no element lies outside the buffer and
  // no padding takes effect.
  if (xferOp.hasOutOfBoundsDim())
    return false;

  // A mask could disable some elements, making the access partial.
  if (xferOp.getMask())
    return false;

  return true;
}

//===----------------------------------------------------------------------===//
//  Interface models
//===----------------------------------------------------------------------===//

namespace {

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
    return isWholeBufferTransfer(cast<vector::TransferReadOp>(op), slot,
                                 blockingUses);
  }

  DeletionKind removeBlockingUses(
      Operation *op, const MemorySlot &slot,
      const SmallPtrSetImpl<OpOperand *> &blockingUses, OpBuilder &builder,
      Value reachingDefinition, const DataLayout &dataLayout) const {
    // `canUsesBeRemoved` guaranteed a whole-buffer read of the slot.
    cast<vector::TransferReadOp>(op).getVector().replaceAllUsesWith(
        reachingDefinition);
    return DeletionKind::Delete;
  }
};

struct TransferWriteOpMemOpModel
    : public PromotableMemOpInterface::ExternalModel<TransferWriteOpMemOpModel,
                                                     vector::TransferWriteOp> {
  bool loadsFrom(Operation *op, const MemorySlot &slot) const { return false; }

  bool storesTo(Operation *op, const MemorySlot &slot) const {
    return cast<vector::TransferWriteOp>(op).getBase() == slot.ptr;
  }

  Value getStored(Operation *op, const MemorySlot &slot, OpBuilder &builder,
                  Value reachingDef, const DataLayout &dataLayout) const {
    return cast<vector::TransferWriteOp>(op).getValueToStore();
  }

  bool canUsesBeRemoved(Operation *op, const MemorySlot &slot,
                        const SmallPtrSetImpl<OpOperand *> &blockingUses,
                        SmallVectorImpl<OpOperand *> &newBlockingUses,
                        const DataLayout &dataLayout) const {
    auto xferOp = cast<vector::TransferWriteOp>(op);
    // The stored value must not be the slot pointer itself.
    if (xferOp.getValueToStore() == slot.ptr)
      return false;
    return isWholeBufferTransfer(xferOp, slot, blockingUses);
  }

  DeletionKind removeBlockingUses(
      Operation *op, const MemorySlot &slot,
      const SmallPtrSetImpl<OpOperand *> &blockingUses, OpBuilder &builder,
      Value reachingDefinition, const DataLayout &dataLayout) const {
    return DeletionKind::Delete;
  }
};

} // namespace

//===----------------------------------------------------------------------===//
//  Register external models
//===----------------------------------------------------------------------===//

void mlir::vector::registerMemorySlotOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, vector::VectorDialect *dialect) {
    TransferReadOp::attachInterface<TransferReadOpMemOpModel>(*ctx);
    TransferWriteOp::attachInterface<TransferWriteOpMemOpModel>(*ctx);
  });
}
