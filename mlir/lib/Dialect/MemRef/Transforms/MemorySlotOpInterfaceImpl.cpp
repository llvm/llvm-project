//===- MemorySlotOpInterfaceImpl.cpp - Mem2Reg for MemRef ops -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements Mem2Reg-related interfaces that let a memref be promoted
// into a single vector SSA value: `PromotableMemOpInterface` models for the ops
// that access such a memref and `PromotableAliaserInterface` models for the ops
// that view it. Mem2Reg calls the memory it promotes a *slot*: a pointer paired
// with the type of the value it can be promoted to, here a memref and its
// vector type. Promoting a slot replaces it with that value, used as the slot's
// reaching definition.
//
// A slot is promoted when each of its uses is an access these models can
// rewrite: a `memref.copy` here, or a vector transfer op in the Vector dialect.
//
// A `memref.subview` of the slot is allowed: the view gets its own slot,
// pairing its result with the vector type corresponding to the view, and is
// promoted together with the parent, as long as every use of the view is in
// turn such an access, or another such view.
//
// The accesses are rewritten as follows:
//
//   * `memref.copy` is modeled as a transfer of the slot value: if the slot is
//     the copy's target, the value stored into it is a `vector.transfer_read`
//     of the copy's source; if the slot is the copy's source, its value is
//     written to the target with a `vector.transfer_write`. Either way the copy
//     itself is removed. A copy between two slots is resolved one slot at a
//     time, each promotion rewriting its own side (see `CopyOpMemOpModel`).
//
//   * a same-rank, unit-stride `memref.subview` is exposed as an alias slot
//     (via `PromotableAliaserInterface`), so a slot accessed through subviews
//     promotes as well:
//
//     - a static sub-slice becomes an alias of that sub-vector: a read projects
//       it out of the slot value with `vector.extract_strided_slice` and a
//       write composes back into the value with `vector.insert_strided_slice`;
//
//     - a dynamic sub-slice has no vector type, since vector shapes must be
//       static. Its alias therefore holds the WHOLE parent vector, and accesses
//       through it are out-of-bounds transfers of that shape. Promotion masks
//       the value down to the extent carried by the subview's size operands,
//       using `vector.create_mask` and `arith.select`. Such a subview must
//       start at the buffer origin.
//
// A memref accessed through the subviews or copies above must be statically
// shaped, so that its slot has a fixed-shape vector type. A scalable slot may
// instead be dynamically shaped, but it is 1-D and promotes through
// whole-buffer transfers alone: it cannot be subviewed or copied.
//
// Accesses that do not meet the criteria above -- dynamic offsets,
// rank-reducing or non-unit-stride subviews, non-zero transfer indices -- are
// left untouched, so the memref is not promoted.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/Transforms/MemorySlotOpInterfaceImpl.h"

#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/IR/MemRefDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"

using namespace mlir;
using namespace mlir::memref;
using namespace mlir::vector;
//===----------------------------------------------------------------------===//
//  Utilities
//===----------------------------------------------------------------------===//

/// Whether `subView` is a statically-shaped slice that can be aliased as the
/// sub-vector it covers, the alias its promotion uses. Promotion projects the
/// parent buffer's vector value through `vector.extract_strided_slice` /
/// `insert_strided_slice`, which require:
///   * fully static offsets and sizes,
///   * unit strides,
///   * no rank reduction (result rank == source rank),
/// so a dropped or dynamic dimension disqualifies the subview. A dynamically-
/// shaped slice aliases the whole parent buffer instead
/// (`isAliasableDynamicShapeSubView`).
static bool isAliasableStaticShapeSubView(memref::SubViewOp subView) {
  auto srcType = dyn_cast<MemRefType>(subView.getSource().getType());
  auto resType = dyn_cast<MemRefType>(subView.getResult().getType());
  if (!srcType || !resType || !srcType.hasStaticShape() ||
      !resType.hasStaticShape())
    return false;

  // No rank reduction: extract/insert_strided_slice operate at a single rank.
  if (srcType.getRank() != resType.getRank())
    return false;

  // Unit strides only.
  for (OpFoldResult stride : subView.getMixedStrides()) {
    std::optional<int64_t> s = getConstantIntValue(stride);
    if (!s || *s != 1)
      return false;
  }

  // Static offsets.
  for (OpFoldResult offset : subView.getMixedOffsets()) {
    if (!getConstantIntValue(offset))
      return false;
  }

  // Static sizes (already implied by the result's static shape, but the sizes
  // must match the result shape so the slice covers exactly the subview).
  for (auto [size, dim] :
       llvm::zip_equal(subView.getMixedSizes(), resType.getShape())) {
    std::optional<int64_t> s = getConstantIntValue(size);
    if (!s || *s != dim)
      return false;
  }
  return true;
}

/// The offsets at which the alias sub-vector of a statically-shaped `subView`
/// sits within the parent slot's value, for `vector.extract_strided_slice` /
/// `insert_strided_slice`.
static SmallVector<int64_t> getStaticSubViewOffsets(memref::SubViewOp subView) {
  SmallVector<int64_t> offsets;
  for (OpFoldResult offset : subView.getMixedOffsets()) {
    std::optional<int64_t> value = getConstantIntValue(offset);
    assert(value && "expected a static offset");
    offsets.push_back(*value);
  }
  return offsets;
}

/// Whether `subView` is a dynamically-shaped slice that can be aliased as the
/// whole parent buffer, the alias its promotion uses. A statically-shaped slice
/// aliases just its sub-vector instead (`isAliasableStaticShapeSubView`).
static bool isAliasableDynamicShapeSubView(memref::SubViewOp subView) {
  // The parent must be statically shaped so the slot has a fixed-shape vector
  // type; only the subview's sizes may be dynamic.
  auto srcType = dyn_cast<MemRefType>(subView.getSource().getType());
  auto resType = dyn_cast<MemRefType>(subView.getResult().getType());
  if (!srcType || !resType || !srcType.hasStaticShape())
    return false;

  // At least one dynamic result dimension; a fully static sub-slice is handled
  // by `isAliasableStaticShapeSubView` instead.
  if (resType.hasStaticShape())
    return false;

  // Same rank: the read/write vector matches the parent rank.
  if (srcType.getRank() != resType.getRank())
    return false;

  // Unit strides and zero offsets: a leading, contiguous sub-region starting at
  // the parent's origin, so the valid region on each dim is exactly [0, size).
  if (!subView.hasUnitStride() || !subView.hasZeroOffset())
    return false;

  return VectorType::isValidElementType(srcType.getElementType());
}

/// The vector type covering `subView`'s whole parent buffer, which is the type
/// the alias slot of a dynamically-shaped subview takes.
static VectorType getWholeParentVectorType(memref::SubViewOp subView) {
  auto parentType = cast<MemRefType>(subView.getSource().getType());
  assert(parentType.hasStaticShape() && "expected a statically-shaped parent");
  return VectorType::get(parentType.getShape(), parentType.getElementType());
}

/// If `slotPtr` is a dynamic subview, returns it.
static memref::SubViewOp getDynamicSubView(Value slotPtr) {
  auto subView = slotPtr.getDefiningOp<memref::SubViewOp>();
  if (subView && isAliasableDynamicShapeSubView(subView))
    return subView;
  return {};
}

/// Builds the mask of `subView`'s valid region: a `vector.create_mask` of the
/// subview's sizes, in the shape of the whole parent that its alias holds.
static Value buildSubViewMask(OpBuilder &builder, Location loc,
                              memref::SubViewOp subView) {
  VectorType parentVecType = getWholeParentVectorType(subView);
  SmallVector<Value> bounds =
      getValueOrCreateConstantIndexOp(builder, loc, subView.getMixedSizes());
  return vector::CreateMaskOp::create(
      builder, loc,
      VectorType::get(parentVecType.getShape(), builder.getI1Type()), bounds);
}

/// Produces the value that a `memref.copy` stores into a slot: reads `mem` at
/// the origin into `vecType` with an identity map, marking a dimension
/// in-bounds only when the memref extent is statically at least the vector
/// extent.
///
/// The padding is a don't-care: `memref.copy` requires both sides to have the
/// same shape, so the source covers the whole vector at runtime and a dynamic
/// extent only forces a conservative out-of-bounds marking. For a
/// dynamic-subview slot, lanes past its extent are in any case discarded by the
/// aliaser's masked select.
static Value readMemRefAsVector(OpBuilder &builder, Location loc, Value mem,
                                VectorType vecType) {
  assert(!vecType.isScalable() && "expected a fixed-size vector");
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

/// Stores a slot's value to the target of a `memref.copy`: writes `vec` into
/// `mem` at the origin, with the identity map and in-bounds rule of
/// `readMemRefAsVector`. `mask`, when set, restricts the write to the lanes it
/// selects.
static void writeVectorToMemRef(OpBuilder &builder, Location loc, Value vec,
                                Value mem, Value mask = {}) {
  auto memType = cast<MemRefType>(mem.getType());
  auto vecType = cast<VectorType>(vec.getType());
  assert(!vecType.isScalable() && "expected a fixed-size vector");
  int64_t rank = vecType.getRank();
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  SmallVector<Value> indices(rank, zero);
  SmallVector<bool> inBounds(rank);
  for (int64_t d = 0; d < rank; ++d)
    inBounds[d] = !memType.isDynamicDim(d) &&
                  memType.getDimSize(d) >= vecType.getDimSize(d);
  vector::TransferWriteOp::create(
      builder, loc, vec, mem, indices,
      AffineMapAttr::get(builder.getMultiDimIdentityMap(rank)), mask,
      builder.getBoolArrayAttr(inBounds));
}

//===----------------------------------------------------------------------===//
//  memref.subview aliaser
//===----------------------------------------------------------------------===//

/// Returns the offsets of `subView` as a static, contiguous, same-rank slice of
/// its source, or nullopt if the subview is not promotable as a whole-buffer
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
///   * copy OUT of the slot (source == slot): on removal the slot's value is
///     written to the target with a `vector.transfer_write` -- the copy becomes
///     that write, masked to the valid region if the slot is a dynamic view.
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

  // Promotion replaces the slot with a vector value, so a copy into the slot
  // becomes the code producing that value: a read of the copy's source.
  Value getStored(Operation *op, const MemorySlot &slot, OpBuilder &builder,
                  Value reachingDef, const DataLayout &dataLayout) const {
    auto copyOp = cast<memref::CopyOp>(op);
    return readMemRefAsVector(builder, op->getLoc(), copyOp.getSource(),
                              cast<VectorType>(slot.valueType));
  }

  bool canUsesBeRemoved(Operation *op, const MemorySlot &slot,
                        const SmallPtrSetImpl<OpOperand *> &blockingUses,
                        SmallVectorImpl<OpOperand *> &newBlockingUses,
                        const DataLayout &dataLayout) const {
    auto copyOp = cast<memref::CopyOp>(op);
    auto vecType = dyn_cast<VectorType>(slot.valueType);
    if (!vecType || vecType.isScalable())
      return false;
    bool srcIsSlot = copyOp.getSource() == slot.ptr;
    bool dstIsSlot = copyOp.getTarget() == slot.ptr;
    // Exactly one side must be this slot. A self-copy (both sides the slot) is
    // not modeled here.
    if (srcIsSlot == dstIsSlot)
      return false;
    // No further check: `memref.copy` verifies that both operands have the same
    // shape and element type.
    return true;
  }

  // A copy out of the slot is a use of the slot's vector value: it becomes a
  // `vector.transfer_write` of that value into the copy's target.
  DeletionKind
  removeBlockingUses(Operation *op, const MemorySlot &slot,
                     const SmallPtrSetImpl<OpOperand *> &blockingUses,
                     OpBuilder &builder, Value reachingDefinition,
                     const DataLayout &dataLayout) const {
    auto copyOp = cast<memref::CopyOp>(op);
    if (copyOp.getSource() == slot.ptr) {
      Location loc = op->getLoc();
      Value mask;
      if (memref::SubViewOp subView = getDynamicSubView(slot.ptr))
        mask = buildSubViewMask(builder, loc, subView);
      writeVectorToMemRef(builder, loc, reachingDefinition, copyOp.getTarget(),
                          mask);
    }
    return DeletionKind::Delete;
  }
};

//===----------------------------------------------------------------------===//
//  memref.subview aliaser
//===----------------------------------------------------------------------===//

/// Companion `PromotableOpInterface` model for the `memref.subview` aliased
/// below: once the slot is promoted, the view has no remaining memory uses and
/// is erased.
struct SubViewOpPromotableModel
    : public PromotableOpInterface::ExternalModel<SubViewOpPromotableModel,
                                                  memref::SubViewOp> {
  bool canUsesBeRemoved(Operation *op,
                        const SmallPtrSetImpl<OpOperand *> &blockingUses,
                        SmallVectorImpl<OpOperand *> &newBlockingUses,
                        const DataLayout &dataLayout) const {
    // The view result is itself a blocking use of the parent slot; its own
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
///   * static sub-slice: `extract_strided_slice` down, `insert_strided_slice`
///     up;
///   * dynamic sub-slice: identity down, `select(create_mask(sizes), value,
///     reachingDef)` up -- see `isAliasableDynamicShapeSubView`.
struct SubViewOpAliasModel
    : public PromotableAliaserInterface::ExternalModel<SubViewOpAliasModel,
                                                       memref::SubViewOp> {
  void getPromotableSlotAliases(Operation *op,
                                OpOperand &aliasedSlotPointerOperand,
                                const MemorySlot &parentSlot,
                                SmallVectorImpl<MemorySlot> &newSlots) const {
    auto subView = cast<memref::SubViewOp>(op);
    // Called once per operand holding the slot pointer; only the viewed source
    // exposes an alias.
    if (aliasedSlotPointerOperand.get() != subView.getSource())
      return;

    // The parent slot must promote to a vector. A scalar (single-element)
    // parent slot cannot be sliced.
    auto parentVecType = dyn_cast<VectorType>(parentSlot.valueType);
    if (!parentVecType)
      return;

    // A static slice aliases the sub-vector matching the subview's shape.
    if (isAliasableStaticShapeSubView(subView)) {
      auto resType = cast<MemRefType>(subView.getResult().getType());
      if (!VectorType::isValidElementType(resType.getElementType()))
        return;
      VectorType aliasVecType =
          VectorType::get(resType.getShape(), resType.getElementType());
      newSlots.push_back(MemorySlot{subView.getResult(), aliasVecType});
    }

    // A dynamic slice aliases the whole parent value.
    if (isAliasableDynamicShapeSubView(subView)) {
      newSlots.push_back(MemorySlot{subView.getResult(), parentVecType});
    }
  }

  Value projectSlotValueToAliasValue(Operation *op,
                                     OpOperand & /*aliasedSlotPointerOperand*/,
                                     const MemorySlot & /*parentSlot*/,
                                     const MemorySlot &aliasSlot,
                                     Value slotValue,
                                     OpBuilder &builder) const {
    auto subView = cast<memref::SubViewOp>(op);
    // Identity: the alias holds the whole parent value. Masking is left to
    // `TransferReadOpMemOpModel::removeBlockingUses`, which unlike this hook
    // can see the consuming read's padding value.
    if (isAliasableDynamicShapeSubView(subView))
      return slotValue;

    SmallVector<int64_t> offsets = getStaticSubViewOffsets(subView);
    auto aliasVecType = cast<VectorType>(aliasSlot.valueType);
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
    // Compose the stored value over the subview's extent only, leaving the rest
    // of the parent value untouched.
    if (isAliasableDynamicShapeSubView(subView)) {
      Location loc = op->getLoc();
      Value mask = buildSubViewMask(builder, loc, subView);
      return arith::SelectOp::create(builder, loc, mask, aliasValue,
                                     reachingDef);
    }

    SmallVector<int64_t> offsets = getStaticSubViewOffsets(subView);
    SmallVector<int64_t> strides(offsets.size(), 1);
    return vector::InsertStridedSliceOp::create(
               builder, op->getLoc(), aliasValue, reachingDef, offsets, strides)
        .getResult();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
//  Queries for the models of ops that access a slot
//===----------------------------------------------------------------------===//

bool mlir::memref::isDynamicViewSlot(Value slotPtr) {
  return static_cast<bool>(getDynamicSubView(slotPtr));
}

Value mlir::memref::buildDynamicViewMask(OpBuilder &builder, Location loc,
                                         Value slotPtr) {
  memref::SubViewOp subView = getDynamicSubView(slotPtr);
  if (!subView)
    return {};
  return buildSubViewMask(builder, loc, subView);
}

//===----------------------------------------------------------------------===//
//  Register external models
//===----------------------------------------------------------------------===//

void mlir::memref::registerMemorySlotOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, memref::MemRefDialect *dialect) {
    memref::SubViewOp::attachInterface<SubViewOpAliasModel>(*ctx);
    memref::SubViewOp::attachInterface<SubViewOpPromotableModel>(*ctx);
    memref::CopyOp::attachInterface<CopyOpMemOpModel>(*ctx);
  });
}
