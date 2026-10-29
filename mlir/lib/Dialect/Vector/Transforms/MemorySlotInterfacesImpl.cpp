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
// and definitions of a vector SSA value. The models for `memref.subview`
// live in the MemRef dialect.
//
// A read uses the slot's current value or extracts a constant-offset sub-vector
// from the slot vector; a write defines its next value.
// Static subviews use a vector matching the subview's shape: reads extract that
// sub-vector from the parent, and writes insert it back into the parent.
// Dynamic subview slots hold the whole static parent vector. Reads
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
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::vector;

//===----------------------------------------------------------------------===//
//  Utilities
//===----------------------------------------------------------------------===//

/// Return whether the transfer's only blocking use is its memref slot base.
static bool
isTransferSlotAccess(VectorTransferOpInterface xferOp, const MemorySlot &slot,
                     const SmallPtrSetImpl<OpOperand *> &blockingUses) {
  return blockingUses.size() == 1 &&
         (*blockingUses.begin())->get() == slot.ptr &&
         xferOp.getBase() == slot.ptr;
}

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
  if (!isTransferSlotAccess(xferOp, slot, blockingUses))
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

/// Return constant offsets for a slice contained in the slot vector. Requires:
/// - Fixed-size vectors of equal rank and element type.
/// - Identity map and constant nonnegative indices.
/// - Matching memref/slot shapes, or a zero-offset, unit-stride, same-rank
///   dynamic subview of a static parent.
///
/// Dynamic subview reads use an extent mask to pad outside the runtime extent.
static FailureOr<SmallVector<int64_t>> getPromotableReadSliceOffsets(
    TransferReadOp readOp, const MemorySlot &slot,
    const SmallPtrSetImpl<OpOperand *> &blockingUses) {
  if (!isTransferSlotAccess(
          cast<VectorTransferOpInterface>(readOp.getOperation()), slot,
          blockingUses) ||
      !readOp.getPermutationMap().isIdentity())
    return failure();

  auto slotType = dyn_cast<VectorType>(slot.valueType);
  VectorType readType = readOp.getVectorType();
  if (!slotType || slotType.isScalable() || readType.isScalable() ||
      readType.getRank() != slotType.getRank() ||
      readType.getElementType() != slotType.getElementType())
    return failure();

  if (!memref::isDynamicSubViewSlot(slot.ptr) &&
      readOp.getShapedType().getShape() != slotType.getShape())
    return failure();

  SmallVector<int64_t> offsets;
  for (auto [index, size, extent] : llvm::zip_equal(
           readOp.getIndices(), readType.getShape(), slotType.getShape())) {
    auto offset = getConstantIntValue(index);
    // Subtract first to avoid overflowing offset + size in the bounds check.
    if (!offset || *offset < 0 || size > extent || *offset > extent - size)
      return failure();
    offsets.push_back(*offset);
  }
  return offsets;
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
                                blockingUses) ||
           succeeded(getPromotableReadSliceOffsets(cast<TransferReadOp>(op),
                                                   slot, blockingUses));
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
    Value result = reachingDefinition;
    Value mask = memref::buildDynamicSubViewMask(builder, loc, slot.ptr);
    if (readOp.getVectorType() != slot.valueType) {
      auto offsets = getPromotableReadSliceOffsets(readOp, slot, blockingUses);
      assert(succeeded(offsets) && "expected a promotable read slice");
      SmallVector<int64_t> strides(offsets->size(), 1);
      auto extractSlice = [&](Value value) -> Value {
        return vector::ExtractStridedSliceOp::create(
            builder, loc, value, *offsets, readOp.getVectorType().getShape(),
            strides);
      };
      result = extractSlice(result);
      if (mask)
        mask = extractSlice(mask);
    }
    if (Value opMask = readOp.getMask())
      mask = mask
                 ? arith::AndIOp::create(builder, loc, mask, opMask).getResult()
                 : opMask;

    if (mask) {
      Value padSplat = vector::BroadcastOp::create(
          builder, loc, readOp.getVectorType(), readOp.getPadding());
      result = arith::SelectOp::create(builder, loc, mask, result, padSplat);
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
