//===- MemorySlotOpInterfaceImpl.cpp - Mem2Reg for vector ops -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements Mem2Reg-related interfaces that let a statically-shaped
// memref be promoted into a single vector SSA value. A memref is promoted when
// every access to it is a `vector` transfer meeting the criteria in
// `isPromotableTransfer` (or a promotable `memref.subview` / `memref.copy`
// built on such transfers). An access need not cover the whole buffer: a masked
// or dynamic-subview transfer that touches only part of it is reconstructed
// with an `arith.select` during promotion. Mem2Reg replaces the memory slot
// with a vector value, used as its reaching definition:
//
//   * `vector.transfer_read` becomes a use of the current vector value;
//     `vector.transfer_write` becomes a new definition of it (see the
//     `PromotableMemOpInterface` models below). A masked or out-of-bounds
//     transfer reads/writes only part of the slot and is composed with an
//     `arith.select`: on the inactive lanes a read yields the transfer's
//     padding and a write keeps the reaching value.
//
//   * a static, same-rank `memref.subview` is exposed as a promotable sub-slice
//     alias of the buffer's slot (via `PromotableAliaserInterface`): a read of
//     the subview projects out of the vector value with
//     `vector.extract_strided_slice`, and a write into it composes back into
//     the value with `vector.insert_strided_slice`. This lets a buffer that is
//     only ever accessed through static subviews promote as well, with partial
//     and overlapping sub-writes composing in program order.
//
//   * a DYNAMIC, same-rank, zero-offset, unit-stride `memref.subview` of a
//     statically-shaped parent is exposed as an alias whose value type is the
//     WHOLE parent vector (the sub-slice is not statically typeable). An
//     out-of-bounds `vector.transfer_read`/`transfer_write` of the parent
//     extent through it is how the dynamic valid region is expressed: the read
//     masks the tail beyond the dynamic size with its padding value via
//     `vector.create_mask` + `arith.select`, and a write composes the stored
//     value onto the parent within the dynamic extent the same way. This is
//     what typically appears after bufferizing a padded, dynamically-shaped
//     op (the padded buffer is static, the real region is a dynamic subview).
//
// Supported subview forms (the parent buffer must be statically shaped so its
// slot has a fixed-shape vector type):
//   - fully static sub-slice           -> extract/insert_strided_slice;
//   - dynamic sub-slice (>=1 dynamic result dim), same-rank, zero-offset,
//     unit-stride                       -> create_mask + arith.select.
// A dynamically-shaped parent is never promoted.
//
// Accesses that do not meet these criteria -- dynamic offsets, rank-reducing or
// non-unit-stride subviews, non-zero transfer indices -- are left untouched, so
// the memref is not promoted.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/Transforms/MemorySlotOpInterfaceImpl.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"

using namespace mlir;
using namespace mlir::vector;

//===----------------------------------------------------------------------===//
//  Utilities
//===----------------------------------------------------------------------===//

/// If `subView` is a same-rank, unit-stride, zero-offset slice of a
/// statically-shaped parent that has at least one dynamic dimension in its
/// result, returns the parent's vector type; otherwise returns nullopt.
///
/// Such a subview is exposed to Mem2Reg as an alias whose element type is the
/// WHOLE parent vector (not the sub-slice, which is not statically typeable).
/// An out-of-bounds `vector.transfer_read`/`transfer_write` of the parent
/// extent through it reads/writes the whole parent value and masks the tail
/// beyond the (dynamic) subview size with `vector.create_mask` + `arith.select`
/// (see the transfer models and the aliaser projections below). This is the
/// dynamic-shape counterpart of `getPromotableSubViewOffsets`, which only
/// handles fully static sub-slices via extract/insert_strided_slice.
static std::optional<VectorType>
getDynamicWholeParentSubView(memref::SubViewOp subView) {
  // The parent (slot) must be statically shaped so the slot has a fixed-shape
  // vector type; the subview may be dynamic (the out-of-bounds transfer through
  // it expresses the valid region).
  auto srcType = dyn_cast<MemRefType>(subView.getSource().getType());
  auto resType = dyn_cast<MemRefType>(subView.getResult().getType());
  if (!srcType || !resType || !srcType.hasStaticShape())
    return std::nullopt;

  // Only dynamic sub-slices; static ones go to `getPromotableSubViewOffsets`.
  if (resType.hasStaticShape())
    return std::nullopt;

  // Same rank: the read/write vector matches the parent rank.
  if (srcType.getRank() != resType.getRank())
    return std::nullopt;

  // Unit strides and zero offsets: a leading, contiguous sub-region starting at
  // the parent's origin, so the valid region on each dim is exactly [0, size).
  for (OpFoldResult stride : subView.getMixedStrides())
    if (getConstantIntValue(stride) != std::optional<int64_t>(1))
      return std::nullopt;
  for (OpFoldResult offset : subView.getMixedOffsets())
    if (getConstantIntValue(offset) != std::optional<int64_t>(0))
      return std::nullopt;

  if (!VectorType::isValidElementType(srcType.getElementType()))
    return std::nullopt;
  return VectorType::get(srcType.getShape(), srcType.getElementType());
}

/// Builds a `vector.create_mask` of `vecType`'s shape marking `subView`'s sizes
/// per dimension -- the in-bounds region of the zero-offset slice.
static Value buildSubViewMask(OpBuilder &builder, Location loc,
                              memref::SubViewOp subView, VectorType vecType) {
  SmallVector<Value> bounds =
      getValueOrCreateConstantIndexOp(builder, loc, subView.getMixedSizes());
  auto maskType = VectorType::get(vecType.getShape(), builder.getI1Type());
  return vector::CreateMaskOp::create(builder, loc, maskType, bounds);
}

/// Reads `mem` at the origin into `vecType` (identity map), marking a dimension
/// in-bounds only when the memref extent is statically at least the vector
/// extent.
static Value readMemRefAsVector(OpBuilder &builder, Location loc, Value mem,
                                VectorType vecType) {
  auto memType = cast<MemRefType>(mem.getType());
  int64_t rank = vecType.getRank();
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  SmallVector<Value> indices(rank, zero);
  Value padding = arith::ConstantOp::create(
      builder, loc, builder.getZeroAttr(vecType.getElementType()));
  SmallVector<bool> inBounds(rank);
  for (int64_t d = 0; d < rank; ++d)
    inBounds[d] = !memType.isDynamicDim(d) &&
                  memType.getDimSize(d) >= vecType.getDimSize(d);
  return vector::TransferReadOp::create(
      builder, loc, vecType, mem, indices,
      AffineMapAttr::get(builder.getMultiDimIdentityMap(rank)), padding,
      /*mask=*/Value(), builder.getBoolArrayAttr(inBounds));
}

/// Writes `vec` into `mem` at the origin (identity map), marking a dimension
/// in-bounds only when the memref extent is statically at least the vector
/// extent.
static void writeVectorToMemRef(OpBuilder &builder, Location loc, Value vec,
                                Value mem) {
  auto memType = cast<MemRefType>(mem.getType());
  auto vecType = cast<VectorType>(vec.getType());
  int64_t rank = vecType.getRank();
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  SmallVector<Value> indices(rank, zero);
  SmallVector<bool> inBounds(rank);
  for (int64_t d = 0; d < rank; ++d)
    inBounds[d] = !memType.isDynamicDim(d) &&
                  memType.getDimSize(d) >= vecType.getDimSize(d);
  vector::TransferWriteOp::create(
      builder, loc, vec, mem, indices,
      AffineMapAttr::get(builder.getMultiDimIdentityMap(rank)),
      /*mask=*/Value(), builder.getBoolArrayAttr(inBounds));
}

/// Returns whether `xferOp` can be promoted to a load/store of `slot`'s vector
/// value. This requires that the transfer's sole use of the slot is as its
/// base, the transferred vector type equals `slot.elemType`, the indices are
/// all zero (origin), and the permutation map is the identity.
///
/// Two forms of partial access are accepted (rather than rejected) and
/// reconstructed with a `select` during promotion (see the transfer models):
///   - a masked transfer, and
///   - an out-of-bounds transfer through a dynamic-subview alias.
/// Their active lanes take the reaching value; the inactive lanes take the
/// transfer's padding (read) or keep the reaching value (write).
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
  if (xferOp.getVectorType() != slot.elemType)
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

  // Out-of-bounds is allowed only for a dynamic-subview alias, whose tail is
  // masked in during promotion; otherwise it would cover only part of the slot.
  if (xferOp.hasOutOfBoundsDim()) {
    auto subView = slot.ptr.getDefiningOp<memref::SubViewOp>();
    if (!subView || !getDynamicWholeParentSubView(subView))
      return false;
  }

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
    return isPromotableTransfer(cast<VectorTransferOpInterface>(op), slot,
                                blockingUses);
  }

  // Replaces the read with the reaching value, masking in the transfer's
  // padding on lanes it does not read -- bounded by the dynamic-subview extent
  // and/or the transfer's own mask (neither: the read covers the whole slot).
  DeletionKind
  removeBlockingUses(Operation *op, const MemorySlot &slot,
                     const SmallPtrSetImpl<OpOperand *> &blockingUses,
                     OpBuilder &builder, Value reachingDefinition,
                     const DataLayout &dataLayout) const {
    auto readOp = cast<vector::TransferReadOp>(op);
    Location loc = op->getLoc();
    Value mask;
    if (auto subView = slot.ptr.getDefiningOp<memref::SubViewOp>())
      if (std::optional<VectorType> vecType =
              getDynamicWholeParentSubView(subView))
        mask = buildSubViewMask(builder, loc, subView, *vecType);
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

struct TransferWriteOpMemOpModel
    : public PromotableMemOpInterface::ExternalModel<TransferWriteOpMemOpModel,
                                                     vector::TransferWriteOp> {
  bool loadsFrom(Operation *op, const MemorySlot &slot) const { return false; }

  bool storesTo(Operation *op, const MemorySlot &slot) const {
    return cast<vector::TransferWriteOp>(op).getBase() == slot.ptr;
  }

  Value getStored(Operation *op, const MemorySlot &slot, OpBuilder &builder,
                  Value reachingDef, const DataLayout &dataLayout) const {
    auto writeOp = cast<vector::TransferWriteOp>(op);
    Value stored = writeOp.getValueToStore();
    // Compose the transfer's own mask here (a masked write updates only its
    // active lanes). The dynamic-subview extent is composed separately, by the
    // aliaser's projectAliasValueToSlotValue.
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

  DeletionKind
  removeBlockingUses(Operation *op, const MemorySlot &slot,
                     const SmallPtrSetImpl<OpOperand *> &blockingUses,
                     OpBuilder &builder, Value reachingDefinition,
                     const DataLayout &dataLayout) const {
    return DeletionKind::Delete;
  }
};

/// Mem2Reg model for `memref.copy`.
///
/// Mem2Reg turns a memref slot into one vector SSA value and tracks which value
/// the buffer holds at each point. Three hooks drive this: (1)
/// `canUsesBeRemoved` checks every access is one we can handle (otherwise the
/// buffer stays in memory); (2) `getStored`, called at each op that writes the
/// buffer, returns the vector value it stores -- this becomes the buffer's
/// value from then on; (3) `removeBlockingUses` rewrites each access to use
/// that vector value instead of the memref. A `memref.copy` fits this as a
/// vector transfer of the value:
///   * copy INTO the slot (target == slot): `getStored` reads the source into a
///     vector (`vector.transfer_read`); that becomes the slot's value, and the
///     copy is deleted.
///   * copy OUT of the slot (source == slot): the slot's value is written to
///   the
///     target with a `vector.transfer_write` -- the copy becomes that write.
/// A copy between two slots promotes one slot at a time, in either order (each
/// promotion rewrites its own side); a dynamic-subview target is masked by the
/// aliaser's projections.
struct CopyOpMemOpModel
    : public PromotableMemOpInterface::ExternalModel<CopyOpMemOpModel,
                                                     memref::CopyOp> {
  bool loadsFrom(Operation *op, const MemorySlot &slot) const {
    return cast<memref::CopyOp>(op).getSource() == slot.ptr;
  }

  bool storesTo(Operation *op, const MemorySlot &slot) const {
    return cast<memref::CopyOp>(op).getTarget() == slot.ptr;
  }

  Value getStored(Operation *op, const MemorySlot &slot, OpBuilder &builder,
                  Value reachingDef, const DataLayout &dataLayout) const {
    // Only reached when storing into the slot (target == slot.ptr): the value
    // is the whole source read as a vector matching the slot's element type.
    auto copyOp = cast<memref::CopyOp>(op);
    return readMemRefAsVector(builder, op->getLoc(), copyOp.getSource(),
                              cast<VectorType>(slot.elemType));
  }

  bool canUsesBeRemoved(Operation *op, const MemorySlot &slot,
                        const SmallPtrSetImpl<OpOperand *> &blockingUses,
                        SmallVectorImpl<OpOperand *> &newBlockingUses,
                        const DataLayout &dataLayout) const {
    auto copyOp = cast<memref::CopyOp>(op);
    auto vecType = dyn_cast<VectorType>(slot.elemType);
    if (!vecType || vecType.isScalable())
      return false;
    bool srcIsSlot = copyOp.getSource() == slot.ptr;
    bool dstIsSlot = copyOp.getTarget() == slot.ptr;
    // Exactly one side must be this slot. A self-copy (both sides the slot) is
    // not modeled here.
    if (srcIsSlot == dstIsSlot)
      return false;
    // memref.copy requires both operands to have the same shape and element
    // type, so the other side already matches the slot's extent (a plain slot
    // is fully covered; a dynamic-subview alias covers its sub-region and is
    // masked by the aliaser). We only need it to be a rank-matching memref; if
    // it is itself a promotable slot, the transfer emitted here becomes a use
    // of that slot and is resolved when it is promoted.
    Value other = srcIsSlot ? copyOp.getTarget() : copyOp.getSource();
    auto otherType = dyn_cast<MemRefType>(other.getType());
    return otherType && otherType.getRank() == vecType.getRank();
  }

  DeletionKind
  removeBlockingUses(Operation *op, const MemorySlot &slot,
                     const SmallPtrSetImpl<OpOperand *> &blockingUses,
                     OpBuilder &builder, Value reachingDefinition,
                     const DataLayout &dataLayout) const {
    auto copyOp = cast<memref::CopyOp>(op);
    // Copy out of the slot: materialize the slot value into the target memref.
    // (Copy into the slot is captured through getStored; nothing to emit.)
    if (copyOp.getSource() == slot.ptr)
      writeVectorToMemRef(builder, op->getLoc(), reachingDefinition,
                          copyOp.getTarget());
    return DeletionKind::Delete;
  }
};

} // namespace

//===----------------------------------------------------------------------===//
//  memref.subview aliaser
//===----------------------------------------------------------------------===//

/// Returns the offsets of `subView` as a static, contiguous, same-rank slice of
/// its source, or nullopt if the subview is not promotable as a static
/// sub-slice. Promotion projects the parent buffer's vector value through
/// `vector.extract_strided_slice` / `insert_strided_slice`, which require:
///   * fully static offsets and sizes,
///   * unit strides,
///   * no rank reduction (result rank == source rank),
/// so a dropped or dynamic dimension disqualifies the subview.
static std::optional<SmallVector<int64_t>>
getPromotableSubViewOffsets(memref::SubViewOp subView) {
  auto srcType = dyn_cast<MemRefType>(subView.getSource().getType());
  auto resType = dyn_cast<MemRefType>(subView.getResult().getType());
  if (!srcType || !resType || !srcType.hasStaticShape() ||
      !resType.hasStaticShape())
    return std::nullopt;

  // No rank reduction: extract/insert_strided_slice operate at a single rank.
  if (srcType.getRank() != resType.getRank())
    return std::nullopt;

  // Unit strides only.
  for (OpFoldResult stride : subView.getMixedStrides()) {
    std::optional<int64_t> s = getConstantIntValue(stride);
    if (!s || *s != 1)
      return std::nullopt;
  }

  // Static offsets.
  SmallVector<int64_t> offsets;
  for (OpFoldResult offset : subView.getMixedOffsets()) {
    std::optional<int64_t> o = getConstantIntValue(offset);
    if (!o)
      return std::nullopt;
    offsets.push_back(*o);
  }

  // Static sizes (already implied by the result's static shape, but the sizes
  // must match the result shape so the slice covers exactly the subview).
  for (auto [size, dim] :
       llvm::zip_equal(subView.getMixedSizes(), resType.getShape())) {
    std::optional<int64_t> s = getConstantIntValue(size);
    if (!s || *s != dim)
      return std::nullopt;
  }
  return offsets;
}

namespace {

/// Exposes a same-rank `memref.subview` as a sub-slice alias of a vector slot,
/// so a buffer accessed through subviews still promotes. When an access (a
/// transfer or a copy) goes through the subview, Mem2Reg converts between the
/// parent value and the alias value with two hooks, run around that access's
/// own mem-op hooks:
///   * a load reads the parent value projected DOWN to the alias
///     (`projectSlotValueToAliasValue`);
///   * a store runs down-project -> `getStored` -> up-project: the parent value
///     is projected down to feed `getStored`'s `reachingDef`, then
///     `getStored`'s result is projected UP to the parent
///     (`projectAliasValueToSlotValue`).
/// The projections depend on the subview's shape:
///   * static sub-slice:  `extract_strided_slice` (down) /
///   `insert_strided_slice`
///     (up);
///   * dynamic sub-slice: identity (down) / `select(create_mask(sizes), value,
///     reachingDef)` (up) -- see `getDynamicWholeParentSubView`.
struct SubViewOpAliasModel
    : public PromotableAliaserInterface::ExternalModel<SubViewOpAliasModel,
                                                       memref::SubViewOp> {
  void getPromotableSlotAliases(Operation *op,
                                OpOperand &aliasedSlotPointerOperand,
                                const MemorySlot &parentSlot,
                                SmallVectorImpl<MemorySlot> &newSlots) const {
    auto subView = cast<memref::SubViewOp>(op);
    if (aliasedSlotPointerOperand.get() != subView.getSource())
      return;

    // The parent slot must promote to a vector. A scalar (single-element)
    // parent slot cannot be sliced.
    auto parentVecType = dyn_cast<VectorType>(parentSlot.elemType);
    if (!parentVecType)
      return;

    // Dynamic sub-region: the alias exposes the WHOLE parent vector; readers /
    // writers mask the tail beyond the dynamic size with arith.select.
    if (getDynamicWholeParentSubView(subView)) {
      newSlots.push_back(MemorySlot{subView.getResult(), parentVecType});
      return;
    }

    if (!getPromotableSubViewOffsets(subView))
      return;

    // The alias's value type is the sub-vector matching the subview's shape.
    auto resType = cast<MemRefType>(subView.getResult().getType());
    if (!VectorType::isValidElementType(resType.getElementType()))
      return;
    VectorType aliasVecType =
        VectorType::get(resType.getShape(), resType.getElementType());
    newSlots.push_back(MemorySlot{subView.getResult(), aliasVecType});
  }

  Value projectSlotValueToAliasValue(Operation *op,
                                     OpOperand & /*aliasedSlotPointerOperand*/,
                                     const MemorySlot & /*parentSlot*/,
                                     const MemorySlot &aliasSlot,
                                     Value slotValue,
                                     OpBuilder &builder) const {
    auto subView = cast<memref::SubViewOp>(op);
    // Dynamic sub-region alias holds the whole parent value, so the down
    // projection is the identity. Masking is not applied here: it needs the
    // consuming read's padding value, which this hook cannot see (one alias
    // feeds reads with different paddings), so it is applied per-read in
    // `TransferReadOpMemOpModel::removeBlockingUses` (create_mask + select).
    if (getDynamicWholeParentSubView(subView))
      return slotValue;

    SmallVector<int64_t> offsets = *getPromotableSubViewOffsets(subView);
    auto aliasVecType = cast<VectorType>(aliasSlot.elemType);
    SmallVector<int64_t> strides(offsets.size(), 1);
    return vector::ExtractStridedSliceOp::create(
               builder, op->getLoc(), slotValue, offsets,
               aliasVecType.getShape(), strides)
        .getResult();
  }

  Value projectAliasValueToSlotValue(Operation *op,
                                     OpOperand & /*aliasedSlotPointerOperand*/,
                                     const MemorySlot & /*parentSlot*/,
                                     const MemorySlot & /*aliasSlot*/,
                                     Value aliasValue, Value reachingDef,
                                     OpBuilder &builder) const {
    auto subView = cast<memref::SubViewOp>(op);
    // Dynamic sub-region write: compose the stored value within the (dynamic)
    // subview extent back onto the parent, keeping the reaching value
    // elsewhere: select(create_mask(subview sizes), stored, reachingDef).
    if (std::optional<VectorType> vecType =
            getDynamicWholeParentSubView(subView)) {
      Location loc = op->getLoc();
      Value mask = buildSubViewMask(builder, loc, subView, *vecType);
      return arith::SelectOp::create(builder, loc, mask, aliasValue,
                                     reachingDef);
    }

    SmallVector<int64_t> offsets = *getPromotableSubViewOffsets(subView);
    SmallVector<int64_t> strides(offsets.size(), 1);
    return vector::InsertStridedSliceOp::create(
               builder, op->getLoc(), aliasValue, reachingDef, offsets, strides)
        .getResult();
  }
};

/// Companion `PromotableOpInterface` model: once the slot is promoted, the
/// subview has no remaining memory uses and is erased.
struct SubViewOpPromotableModel
    : public PromotableOpInterface::ExternalModel<SubViewOpPromotableModel,
                                                  memref::SubViewOp> {
  bool canUsesBeRemoved(Operation *op,
                        const SmallPtrSetImpl<OpOperand *> &blockingUses,
                        SmallVectorImpl<OpOperand *> &newBlockingUses,
                        const DataLayout &dataLayout) const {
    // The subview result is itself a blocking use of the parent slot; its own
    // users (the transfers) are resolved through the alias projections.
    for (OpOperand &use : op->getResult(0).getUses())
      newBlockingUses.push_back(&use);
    return true;
  }

  DeletionKind
  removeBlockingUses(Operation *op,
                     const SmallPtrSetImpl<OpOperand *> &blockingUses,
                     OpBuilder &builder) const {
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
  // The subview aliaser attaches to a MemRef op but lives here because the
  // projections build Vector ops; Vector already depends on MemRef.
  registry.addExtension(+[](MLIRContext *ctx, memref::MemRefDialect *dialect) {
    memref::SubViewOp::attachInterface<SubViewOpAliasModel>(*ctx);
    memref::SubViewOp::attachInterface<SubViewOpPromotableModel>(*ctx);
    memref::CopyOp::attachInterface<CopyOpMemOpModel>(*ctx);
  });
}
