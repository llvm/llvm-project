//===- MemorySlotInterfacesImpl.cpp - Mem2Reg for Vector ops --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements memory slot interface models for
// `vector.transfer_read`, `vector.transfer_write`, and `vector.mask`. They
// implement `PromotableMemOpInterface` to replace vector transfers with uses
// and definitions of a vector SSA value. The models for `memref.copy` and
// `memref.subview` live in the MemRef dialect.
//
// A read uses the slot's current value or extracts a constant-offset sub-vector
// from the slot vector; a write defines its next value.
// Static subviews use a vector matching the subview's shape: reads extract that
// sub-vector from the parent, and writes insert it back into the parent.
// Supported transfers through dynamic subviews use a vector type matching the
// whole static parent, and the subview's slot holds that whole vector. Writes
// preserve the previous value of lanes outside the runtime extent. Reads use
// padding for out-of-bounds lanes unless the transfer guarantees those lanes
// are in-bounds. Writes must be in-bounds or mask off lanes past the extent.
// `arith.select` implements masking by a transfer's mask operand.
//
// `vector.mask` owns its wrapped transfer's blocking use and is promoted as a
// whole. Its region cannot accept extra operations, so `arith.select` is built
// outside it.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/Transforms/MemorySlotInterfacesImpl.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
/// map must be the identity, and it must be in-bounds in every dimension.
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

  // A fixed-size whole-buffer transfer at zero is statically in bounds.
  auto memrefType = cast<MemRefType>(slot.ptr.getType());
  VectorType vectorType = xferOp.getVectorType();
  if (!vectorType.isScalable() && memrefType.hasStaticShape() &&
      memrefType.getShape() == vectorType.getShape())
    return true;

  return !xferOp.hasOutOfBoundsDim();
}

/// Return constant offsets for a slice contained in the slot vector. Requires:
/// - Fixed-size vectors of equal rank and element type.
/// - Identity map and constant nonnegative indices.
/// - Matching memref/slot shapes, or a zero-offset, unit-stride, same-rank
///   dynamic subview of a static parent.
///
/// Static memref shapes prove bounds even without `in_bounds`; dynamic subview
/// reads need an extent mask for dimensions not guaranteed to be in-bounds.
static FailureOr<SmallVector<int64_t>> getPromotableReadSliceOffsets(
    TransferReadOp readOp, const MemorySlot &slot,
    const SmallPtrSetImpl<OpOperand *> &blockingUses) {
  if (!isTransferSlotAccess(readOp, slot, blockingUses) ||
      !readOp.getPermutationMap().isIdentity())
    return failure();

  auto slotType = dyn_cast<VectorType>(slot.valueType);
  VectorType readType = readOp.getVectorType();
  if (!slotType || slotType.isScalable() || readType.isScalable() ||
      readType.getRank() != slotType.getRank() ||
      readType.getElementType() != slotType.getElementType())
    return failure();

  // Dynamic bounds must be available without introducing new uses of the slot.
  if (readOp.hasOutOfBoundsDim() &&
      !cast<MemRefType>(slot.ptr.getType()).hasStaticShape() &&
      !slot.ptr.getDefiningOp<memref::SubViewOp>())
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

/// Build a mask in the read's shape for lanes within a dynamic subview.
/// Dimensions guaranteed in-bounds need no runtime check.
static Value buildReadExtentMask(TransferReadOp readOp, const MemorySlot &slot,
                                 OpBuilder &builder) {
  if (!readOp.hasOutOfBoundsDim() ||
      cast<MemRefType>(slot.ptr.getType()).hasStaticShape())
    return {};

  auto subView = slot.ptr.getDefiningOp<memref::SubViewOp>();
  assert(subView && "expected a promotable dynamic subview read");
  VectorType readType = readOp.getVectorType();
  Location loc = readOp.getLoc();
  SmallVector<Value> bounds;
  for (auto [dim, size] : llvm::enumerate(subView.getMixedSizes())) {
    if (readOp.isDimInBounds(dim)) {
      bounds.push_back(arith::ConstantIndexOp::create(
          builder, loc, readType.getDimSize(dim)));
      continue;
    }
    Value extent = getValueOrCreateConstantIndexOp(builder, loc, size);
    bounds.push_back(builder.createOrFold<arith::SubIOp>(
        loc, extent, readOp.getIndices()[dim]));
  }
  return vector::CreateMaskOp::create(
      builder, loc, VectorType::get(readType.getShape(), builder.getI1Type()),
      bounds);
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
  // outside the runtime extent or disabled by the transfer's mask.
  DeletionKind
  removeBlockingUses(Operation *op, const MemorySlot &slot,
                     const SmallPtrSetImpl<OpOperand *> &blockingUses,
                     OpBuilder &builder, Value reachingDefinition,
                     const DataLayout &dataLayout) const {
    auto readOp = cast<vector::TransferReadOp>(op);
    Location loc = op->getLoc();
    Value result = reachingDefinition;
    Value mask = readOp.getMask();
    if (readOp.getVectorType() != slot.valueType) {
      auto offsets = getPromotableReadSliceOffsets(readOp, slot, blockingUses);
      assert(succeeded(offsets) && "expected a promotable read slice");
      SmallVector<int64_t> strides(offsets->size(), 1);
      result = vector::ExtractStridedSliceOp::create(
          builder, loc, result, *offsets, readOp.getVectorType().getShape(),
          strides);
    }
    if (Value extentMask = buildReadExtentMask(readOp, slot, builder)) {
      if (mask)
        mask = arith::AndIOp::create(builder, loc, mask, extentMask);
      else
        mask = extentMask;
    }
    if (mask) {
      Value padSplat = vector::BroadcastOp::create(
          builder, loc, readOp.getVectorType(), readOp.getPadding());
      result = arith::SelectOp::create(builder, loc, mask, result, padSplat);
    }
    readOp.getResult().replaceAllUsesWith(result);
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

/// Returns the operation wrapped by the `vector.mask` `op`.
PromotableMemOpInterface getWrappedMemOp(Operation *op) {
  Operation *maskableOp = cast<vector::MaskOp>(op).getMaskableOp();
  if (!maskableOp)
    return {};
  if (!isa<vector::TransferReadOp, vector::TransferWriteOp>(maskableOp))
    return {};
  return cast<PromotableMemOpInterface>(maskableOp);
}

/// `PromotableMemOpInterface` model for `vector.mask` wrapping a promotable
/// memory operation. `vector.mask` owns the wrapped operation's slot use,
/// delegates promotion to it, and applies its mask to the result.
struct MaskOpMemOpModel
    : public PromotableMemOpInterface::ExternalModel<MaskOpMemOpModel,
                                                     vector::MaskOp> {
  bool ownsWrappedBlockingUses(Operation *op) const {
    return static_cast<bool>(getWrappedMemOp(op));
  }

  bool loadsFrom(Operation *op, const MemorySlot &slot) const {
    PromotableMemOpInterface memOp = getWrappedMemOp(op);
    return memOp && memOp.loadsFrom(slot);
  }

  bool storesTo(Operation *op, const MemorySlot &slot) const {
    PromotableMemOpInterface memOp = getWrappedMemOp(op);
    return memOp && memOp.storesTo(slot);
  }

  Value getStored(Operation *op, const MemorySlot &slot, OpBuilder &builder,
                  Value reachingDef, const DataLayout &dataLayout) const {
    Value stored =
        getWrappedMemOp(op).getStored(slot, builder, reachingDef, dataLayout);
    return arith::SelectOp::create(builder, op->getLoc(),
                                   cast<vector::MaskOp>(op).getMask(), stored,
                                   reachingDef);
  }

  bool canUsesBeRemoved(Operation *op, const MemorySlot &slot,
                        const SmallPtrSetImpl<OpOperand *> &blockingUses,
                        SmallVectorImpl<OpOperand *> &newBlockingUses,
                        const DataLayout &dataLayout) const {
    PromotableMemOpInterface memOp = getWrappedMemOp(op);
    // Transfers with their own mask operand are unsupported.
    if (!memOp || cast<VectorTransferOpInterface>(*memOp).getMask())
      return false;
    return memOp.canUsesBeRemoved(slot, blockingUses, newBlockingUses,
                                  dataLayout);
  }

  // Let the wrapped memory op promote itself, then apply the wrapper's mask to
  // the value it now yields.
  DeletionKind
  removeBlockingUses(Operation *op, const MemorySlot &slot,
                     const SmallPtrSetImpl<OpOperand *> &blockingUses,
                     OpBuilder &builder, Value reachingDefinition,
                     const DataLayout &dataLayout) const {
    auto maskOp = cast<vector::MaskOp>(op);
    PromotableMemOpInterface memOp = getWrappedMemOp(op);
    memOp.removeBlockingUses(slot, blockingUses, builder, reachingDefinition,
                             dataLayout);
    auto readOp = dyn_cast<vector::TransferReadOp>(*memOp);
    if (!readOp)
      return DeletionKind::Delete;

    Location loc = op->getLoc();
    Value enabled = maskOp.getMaskBlock()->getTerminator()->getOperand(0);
    Value disabled =
        maskOp.hasPassthru()
            ? maskOp.getPassthru()
            : vector::BroadcastOp::create(builder, loc, readOp.getVectorType(),
                                          readOp.getPadding());
    Value result = arith::SelectOp::create(builder, loc, maskOp.getMask(),
                                           enabled, disabled);
    maskOp->getResult(0).replaceAllUsesWith(result);
    return DeletionKind::Delete;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
//  Register external models
//===----------------------------------------------------------------------===//

void mlir::vector::registerMemorySlotInterfacesExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, vector::VectorDialect *dialect) {
    TransferReadOp::attachInterface<TransferReadOpMemOpModel>(*ctx);
    TransferWriteOp::attachInterface<TransferWriteOpMemOpModel>(*ctx);
    MaskOp::attachInterface<MaskOpMemOpModel>(*ctx);
  });
}
