//===- MemorySlotOpInterfaceImpl.cpp - Mem2Reg for vector ops -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements Mem2Reg-related interfaces that let a statically-shaped
// memref be promoted into a single vector SSA value: `PromotableMemOpInterface`
// models for the ops that access such a memref and `PromotableAliaserInterface`
// models for the ops that view it. Mem2Reg calls the memory it promotes a
// *slot*: a pointer paired with the type of the value it becomes, here a memref
// and its vector type. Promoting a slot replaces it with that value, used as
// the slot's reaching definition: the value the slot holds at a given program
// point, merged with a block argument where control flow joins.
//
// A slot is promoted when each of its uses is an access these models can
// rewrite: a `vector.transfer_read` or `vector.transfer_write` meeting the
// criteria in `isPromotableTransfer`, or a `memref.copy` that has the slot as
// its source or its target. A view of the slot (`memref.subview`,
// `memref.expand_shape`, `memref.collapse_shape`) is allowed as well: the view
// becomes an alias slot of its own, promoted with its parent, as long as every
// use of the view is in turn such an access, or another such view.
//
// The accesses are rewritten as follows:
//
//   * `vector.transfer_read` becomes a use of the current vector value;
//     `vector.transfer_write` becomes a new definition of it. A masked transfer
//     covers only its active lanes, so it is composed with an `arith.select`.
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
//   * a `memref.expand_shape` / `memref.collapse_shape` is exposed as an alias
//     of the reshaped value: a contiguous reshape keeps the element order, so
//     both projections are a `vector.shape_cast`. Reshapes compose with the
//     subview aliases above, so a reshaped dynamic view (e.g. `collapse_shape`
//     of a dynamic subview) promotes as well: its alias shape comes from the
//     parent's extents and its mask is reshaped along with the value.
//
// The promoted memref must be statically shaped, so that its slot has a
// fixed-shape vector type. Accesses that do not meet the criteria above --
// dynamic offsets, rank-reducing or non-unit-stride subviews, non-contiguous
// reshapes, non-zero transfer indices -- are left untouched, so the memref is
// not promoted.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/Transforms/MemorySlotOpInterfaceImpl.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"

using namespace mlir;
using namespace mlir::vector;

//===----------------------------------------------------------------------===//
//  Utilities
//===----------------------------------------------------------------------===//

// Defined below, after the subview helpers it builds on.
static memref::SubViewOp getUnderlyingDynamicSubView(Value slotPtr);

/// Returns whether `xferOp` can be promoted to a load/store of `slot`'s vector
/// value. This requires that the transfer's sole use of the slot is as its
/// base, the transferred vector type equals `slot.elemType`, the indices are
/// all zero (origin), and the permutation map is the identity.
///
/// Two forms of partial access are accepted (rather than rejected) and
/// reconstructed with a `select` during promotion (see the transfer models):
///   - a masked transfer, and
///   - an out-of-bounds transfer through a dynamic-subview alias, possibly
///     reached through a reassociative reshape.
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

  // Out-of-bounds is allowed only for a dynamic view.
  if (xferOp.hasOutOfBoundsDim() && !getUnderlyingDynamicSubView(slot.ptr))
    return false;

  return true;
}

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
static VectorType getWholeParentBufferVectorType(memref::SubViewOp subView) {
  auto parentType = cast<MemRefType>(subView.getSource().getType());
  assert(parentType.hasStaticShape() && "expected a statically-shaped parent");
  return VectorType::get(parentType.getShape(), parentType.getElementType());
}

/// Returns the dynamic subview `slotPtr` is a view of, looking through
/// reassociative reshapes (`expand_shape` / `collapse_shape`), or null if there
/// is none. A slot carries only a pointer and a vector type, so the dynamic
/// extent is not part of it: it is recovered from this subview's size operands
/// whenever a mask is needed. A reshaped dynamic subview is still a dynamic
/// view of the parent: the reshape preserves element order, so the valid region
/// is the same set of elements, just indexed differently (see
/// `buildDynamicViewMask`).
static memref::SubViewOp getUnderlyingDynamicSubView(Value slotPtr) {
  Operation *def = slotPtr.getDefiningOp();
  while (def) {
    if (auto subView = dyn_cast<memref::SubViewOp>(def)) {
      if (isAliasableDynamicShapeSubView(subView))
        return subView;
      return {};
    }
    if (auto expand = dyn_cast<memref::ExpandShapeOp>(def)) {
      def = expand.getSrc().getDefiningOp();
      continue;
    }
    if (auto collapse = dyn_cast<memref::CollapseShapeOp>(def)) {
      def = collapse.getSrc().getDefiningOp();
      continue;
    }
    return {};
  }
  return {};
}

/// Returns whether the view lays its elements out in the same (row-major) order
/// as `parentShape`, the parent slot's vector shape, which is what makes a
/// `vector.shape_cast` model it. These are the conditions under which
/// `getReassociatedShape` below is meaningful, and it may be called only when
/// they hold.
///
/// Collapsing is order preserving exactly when each reassociation group is
/// contiguous in the source; that also admits a dynamic but contiguous source,
/// such as a dynamic subview whose trailing dimensions are full.
static bool isOrderPreservingReshape(memref::CollapseShapeOp op,
                                     ArrayRef<int64_t> parentShape) {
  SmallVector<ReassociationIndices, 4> groups = op.getReassociationIndices();
  // The groups must partition the parent's dimensions, which they index into.
  int64_t numGroupedDims = 0;
  for (ReassociationIndices group : groups) {
    for (int64_t dim : group) {
      if (dim < 0 || dim >= static_cast<int64_t>(parentShape.size()))
        return false;
    }
    numGroupedDims += group.size();
  }
  if (numGroupedDims != static_cast<int64_t>(parentShape.size()))
    return false;

  return memref::CollapseShapeOp::isGuaranteedCollapsible(
      op.getSrcType(), op.getReassociationIndices());
}

/// An expansion only subdivides dimensions, so element order carries over once
/// the source's own elements are row-major.
///
/// A STATIC source must have row-major strides -- read off the strides, since a
/// subview always carries a strided layout: `[4, 16]` of `memref<8x16xf32>`
/// qualifies (strides `[16, 1]`), `[4, 8]` does not (strides stay `[16, 1]`
/// where `[8, 1]` is needed). Nothing more: a static view's alias takes the
/// view's own shape.
///
/// A DYNAMIC source needs no strides. It qualifies as a supported dynamic view
/// (`getUnderlyingDynamicSubView`), whose index space coincides with its static
/// parent's.
///
/// Its alias, though, is shaped from the parent's extents, as if the view were
/// as large as the parent -- and at runtime the view can be smaller. Take a
/// `vector<8xf32>` parent and `[2, %m]`: the alias is always a `shape_cast` to
/// `2 x 4`, while a subview extent of 4 makes the view `2 x 2`, so `%m` is 2,
/// not 4. Row 1 then begins at element 4 in the alias and at element 2 in the
/// view -- the same lane reads different data, which no mask can repair.
///
/// So the group rules below require every dimension ahead of the dynamic one to
/// be unit: its index is then always 0, so a wrong extent behind it shifts
/// nothing. `[%m, 2]` passes for that reason -- nothing precedes the dynamic
/// dimension, and the `2` after it is the same in both shapes.
static bool isOrderPreservingReshape(memref::ExpandShapeOp op,
                                     ArrayRef<int64_t> parentShape) {
  if (!memref::isStaticShapeAndContiguousRowMajor(op.getSrcType()) &&
      !getUnderlyingDynamicSubView(op.getSrc()))
    return false;

  ArrayRef<int64_t> resShape = cast<MemRefType>(op.getType()).getShape();
  SmallVector<ReassociationIndices, 4> groups = op.getReassociationIndices();
  if (groups.size() != parentShape.size())
    return false;

  for (auto [srcDim, group] : llvm::enumerate(groups)) {
    int64_t staticProduct = 1;
    int64_t dynamicDim = -1;
    for (int64_t dim : group) {
      if (dim < 0 || dim >= static_cast<int64_t>(resShape.size()))
        return false;
      if (!ShapedType::isDynamic(resShape[dim])) {
        staticProduct *= resShape[dim];
        continue;
      }
      if (dynamicDim >= 0)
        return false; // More than one dynamic dimension: not determined.
      dynamicDim = dim;
    }

    int64_t parentDim = parentShape[srcDim];
    // Fully static group: its extents must account for the parent's exactly.
    if (dynamicDim < 0) {
      if (staticProduct != parentDim)
        return false;
      continue;
    }
    // Only unit dimensions may precede the dynamic one within the group.
    for (int64_t dim : group) {
      if (dim == dynamicDim)
        break;
      if (resShape[dim] != 1)
        return false;
    }
    // The dynamic extent must come out of the parent's as a whole number.
    if (staticProduct == 0 || parentDim % staticProduct != 0)
      return false;
  }
  return true;
}

/// Returns the shape `parentShape` takes under the view's reassociation, i.e.
/// the alias value's shape. It is derived from the parent slot's shape rather
/// than the result memref so that a reshape of a dynamically-shaped view is
/// still typeable. Requires `isOrderPreservingReshape`.
static SmallVector<int64_t>
getReassociatedShape(memref::CollapseShapeOp op,
                     ArrayRef<int64_t> parentShape) {
  SmallVector<int64_t> shape;
  for (ReassociationIndices group : op.getReassociationIndices()) {
    int64_t size = 1;
    for (int64_t dim : group)
      size *= parentShape[dim];
    shape.push_back(size);
  }
  return shape;
}

/// An expansion keeps its result's static extents; each dynamic one is the
/// parent extent its group splits, divided by the group's static extents.
static SmallVector<int64_t>
getReassociatedShape(memref::ExpandShapeOp op, ArrayRef<int64_t> parentShape) {
  ArrayRef<int64_t> resShape = cast<MemRefType>(op.getType()).getShape();
  SmallVector<int64_t> shape(resShape);
  for (auto [srcDim, group] : llvm::enumerate(op.getReassociationIndices())) {
    int64_t staticProduct = 1;
    int64_t dynamicDim = -1;
    for (int64_t dim : group) {
      if (ShapedType::isDynamic(resShape[dim]))
        dynamicDim = dim;
      else
        staticProduct *= resShape[dim];
    }
    if (dynamicDim >= 0)
      shape[dynamicDim] = parentShape[srcDim] / staticProduct;
  }
  return shape;
}

/// Builds the mask of `subView`'s valid region, expressed in `vecType`'s shape:
/// a `vector.create_mask` of the subview's sizes in the parent's shape,
/// reshaped with `vector.shape_cast` when the slot is a reassociated view of
/// the parent (both describe the same elements in row-major order).
static Value buildDynamicViewMask(OpBuilder &builder, Location loc,
                                  memref::SubViewOp subView,
                                  VectorType vecType) {
  VectorType parentVecType = getWholeParentBufferVectorType(subView);
  SmallVector<Value> bounds =
      getValueOrCreateConstantIndexOp(builder, loc, subView.getMixedSizes());
  Value mask = vector::CreateMaskOp::create(
      builder, loc,
      VectorType::get(parentVecType.getShape(), builder.getI1Type()), bounds);
  if (parentVecType.getShape() != vecType.getShape())
    mask = vector::ShapeCastOp::create(
        builder, loc, VectorType::get(vecType.getShape(), builder.getI1Type()),
        mask);
  return mask;
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
    Value mask;
    if (memref::SubViewOp subView = getUnderlyingDynamicSubView(slot.ptr))
      mask =
          buildDynamicViewMask(builder, loc, subView, readOp.getVectorType());
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
      if (memref::SubViewOp subView = getUnderlyingDynamicSubView(slot.ptr))
        mask = buildDynamicViewMask(builder, loc, subView,
                                    cast<VectorType>(slot.elemType));
      writeVectorToMemRef(builder, loc, reachingDefinition, copyOp.getTarget(),
                          mask);
    }
    return DeletionKind::Delete;
  }
};

} // namespace

//===----------------------------------------------------------------------===//
//  memref view aliasers
//===----------------------------------------------------------------------===//

namespace {

/// Companion `PromotableOpInterface` model for the view ops aliased below
/// (`memref.subview`, `memref.expand_shape`, `memref.collapse_shape`): once the
/// slot is promoted, the view has no remaining memory uses and is erased.
template <typename OpTy>
struct ViewOpPromotableModel
    : public PromotableOpInterface::ExternalModel<ViewOpPromotableModel<OpTy>,
                                                  OpTy> {
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
    auto parentVecType = dyn_cast<VectorType>(parentSlot.elemType);
    if (!parentVecType)
      return;

    // A dynamic slice aliases the whole parent value.
    if (isAliasableDynamicShapeSubView(subView)) {
      newSlots.push_back(MemorySlot{subView.getResult(), parentVecType});
      return;
    }

    // A static slice aliases the sub-vector matching the subview's shape.
    if (isAliasableStaticShapeSubView(subView)) {
      auto resType = cast<MemRefType>(subView.getResult().getType());
      if (!VectorType::isValidElementType(resType.getElementType()))
        return;
      VectorType aliasVecType =
          VectorType::get(resType.getShape(), resType.getElementType());
      newSlots.push_back(MemorySlot{subView.getResult(), aliasVecType});
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
    // Compose the stored value over the subview's extent only, leaving the rest
    // of the parent value untouched.
    if (isAliasableDynamicShapeSubView(subView)) {
      Location loc = op->getLoc();
      VectorType parentVecType = getWholeParentBufferVectorType(subView);
      Value mask = buildDynamicViewMask(builder, loc, subView, parentVecType);
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

/// Exposes a `memref.expand_shape` / `memref.collapse_shape` as an alias of a
/// vector slot: the view holds the same elements in the same order, only with a
/// different rank, so both projections are a `vector.shape_cast` (see
/// `isOrderPreservingReshape` for when that holds).
///
/// The alias shape comes from the parent slot's vector shape rather than the
/// result memref (see `getReassociatedShape`), so a reshape of a dynamic view
/// is typeable as well; the dynamic extent is masked where the underlying
/// subview is composed, not here. The view covers the alias value in full, so
/// the up projection does not need `reachingDef`.
template <typename OpTy>
struct ReassociativeViewOpAliasModel
    : public PromotableAliaserInterface::ExternalModel<
          ReassociativeViewOpAliasModel<OpTy>, OpTy> {
  void getPromotableSlotAliases(Operation *op,
                                OpOperand &aliasedSlotPointerOperand,
                                const MemorySlot &parentSlot,
                                SmallVectorImpl<MemorySlot> &newSlots) const {
    auto viewOp = cast<OpTy>(op);
    // Called once per operand holding the slot pointer; only the reshaped
    // source exposes an alias.
    if (aliasedSlotPointerOperand.get() != viewOp.getSrc())
      return;

    // The parent slot must promote to a vector; a scalar slot has no shape to
    // reassociate.
    auto parentVecType = dyn_cast<VectorType>(parentSlot.elemType);
    if (!parentVecType || parentVecType.isScalable())
      return;

    auto resType = cast<MemRefType>(viewOp.getResult().getType());
    if (!VectorType::isValidElementType(resType.getElementType()) ||
        resType.getElementType() != parentVecType.getElementType())
      return;
    if (!isOrderPreservingReshape(viewOp, parentVecType.getShape()))
      return;
    SmallVector<int64_t> aliasShape =
        getReassociatedShape(viewOp, parentVecType.getShape());
    auto aliasVecType = VectorType::get(aliasShape, resType.getElementType());
    // Guard the `vector.shape_cast` contract; implied by the view's semantics.
    if (aliasVecType.getNumElements() != parentVecType.getNumElements())
      return;
    newSlots.push_back(MemorySlot{viewOp.getResult(), aliasVecType});
  }

  Value projectSlotValueToAliasValue(Operation *op,
                                     OpOperand & /*aliasedSlotPointerOperand*/,
                                     const MemorySlot & /*parentSlot*/,
                                     const MemorySlot &aliasSlot,
                                     Value slotValue,
                                     OpBuilder &builder) const {
    return vector::ShapeCastOp::create(builder, op->getLoc(),
                                       cast<VectorType>(aliasSlot.elemType),
                                       slotValue)
        .getResult();
  }

  Value projectAliasValueToSlotValue(Operation *op,
                                     OpOperand & /*aliasedSlotPointerOperand*/,
                                     const MemorySlot &parentSlot,
                                     const MemorySlot & /*aliasSlot*/,
                                     Value aliasValue, Value /*reachingDef*/,
                                     OpBuilder &builder) const {
    return vector::ShapeCastOp::create(builder, op->getLoc(),
                                       cast<VectorType>(parentSlot.elemType),
                                       aliasValue)
        .getResult();
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
    memref::SubViewOp::attachInterface<
        ViewOpPromotableModel<memref::SubViewOp>>(*ctx);
    memref::ExpandShapeOp::attachInterface<
        ReassociativeViewOpAliasModel<memref::ExpandShapeOp>>(*ctx);
    memref::ExpandShapeOp::attachInterface<
        ViewOpPromotableModel<memref::ExpandShapeOp>>(*ctx);
    memref::CollapseShapeOp::attachInterface<
        ReassociativeViewOpAliasModel<memref::CollapseShapeOp>>(*ctx);
    memref::CollapseShapeOp::attachInterface<
        ViewOpPromotableModel<memref::CollapseShapeOp>>(*ctx);
    memref::CopyOp::attachInterface<CopyOpMemOpModel>(*ctx);
  });
}
