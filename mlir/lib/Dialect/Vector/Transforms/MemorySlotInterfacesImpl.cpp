//===- MemorySlotInterfacesImpl.cpp - Mem2Reg for Vector ops --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements memory slot interface models for
// `vector.transfer_read` and `vector.transfer_write`. These models implement
// `PromotableMemOpInterface` so Mem2Reg can replace vector transfers with uses
// and definitions of a vector SSA value. The models for `memref.copy` and
// `memref.subview` live in the MemRef dialect.
//
// A read uses the slot's current value; a write defines its next value.
// Static subviews use a vector matching the subview's shape: reads extract that
// sub-vector from the parent, and writes insert it back into the parent.
// Supported transfers through dynamic subviews use a vector type matching the
// whole static parent, and the subview's slot holds that whole vector. Reads
// pad lanes outside the runtime extent, and writes preserve the previous value
// of those lanes. `arith.select` implements this behavior and also handles
// lanes disabled by a transfer's mask operand.
//
// Transfers wrapped in `vector.mask` prevent promotion because that op does
// not implement `PromotableRegionOpInterface`, which Mem2Reg requires to
// promote accesses inside a region.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/Transforms/MemorySlotInterfacesImpl.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/MemorySlotInterfacesImpl.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"

using namespace mlir;
using namespace mlir::vector;

//===----------------------------------------------------------------------===//
//  Utilities
//===----------------------------------------------------------------------===//

/// Returns whether `xferOp` can be promoted to use or define `slot`'s vector
/// value. The transfer's memref base must be the slot pointer, its vector type
/// must equal `slot.valueType`, its indices must be zero, and its permutation
/// map must be the identity. Out-of-bounds dimensions are allowed only for a
/// supported dynamic subview.
///
/// More general transfers may be supportable, but their value depends on
/// motivating use cases, implementation complexity, and generated code quality.
static bool
isPromotableTransfer(VectorTransferOpInterface xferOp, const MemorySlot &slot,
                     const SmallPtrSetImpl<OpOperand *> &blockingUses) {
  // The transfer's memref base must be the slot pointer.
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
//  PromotableMemOpInterface models
//===----------------------------------------------------------------------===//

namespace {
/// `PromotableMemOpInterface` model for `vector.transfer_read`.
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
  // PromotableAliaserInterface::projectSlotValueToAliasValue supplies the whole
  // parent vector for a dynamic subview. Apply the extent mask here because
  // this hook has access to the transfer's padding operand. If the transfer
  // also has a mask operand, combine the two masks with arith.andi.
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

/// `PromotableMemOpInterface` model for `vector.transfer_write`.
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
    // PromotableAliaserInterface::projectAliasValueToSlotValue separately
    // preserves parent lanes outside a dynamic subview's extent.
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
