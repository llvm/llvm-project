//===- MemorySlotInterfacesImpl.cpp - Mem2Reg for MemRef ops -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// External-model-based implementation of Memory Slot Interfaces for MemRef Ops.
//
// This file implements memory slot interface models used by Mem2Reg to promote
// a memref to a single vector SSA value. `PromotableMemOpInterface` handles
// `memref.copy`. `PromotableAliaserInterface` exposes `memref.subview`,
// `memref.expand_shape` and `memref.collapse_shape` aliases, and
// `PromotableOpInterface` removes those views after promotion. Vector transfer
// ops are handled by the models in the Vector dialect.
//
// A memory slot pairs a memref with the vector type used to represent its
// contents. Promotion replaces accesses to the slot with uses and definitions
// of that vector value. All uses, including those through views, must be
// supported by the models.
//
// The accesses are rewritten as follows:
//
//   * A `memref.copy` into the slot becomes a `vector.transfer_read` of the
//     source. A copy out of the slot becomes a `vector.transfer_write` to the
//     target. A copy between two slots is rewritten one slot at a time.
//
//   * A same-rank, unit-stride `memref.subview` gets an alias slot that is
//     promoted together with the parent:
//
//     - A static subview holds the corresponding sub-vector. Reads extract it
//       with `vector.extract_strided_slice`; writes update the parent with
//       `vector.insert_strided_slice`.
//
//     - A dynamic subview holds the whole parent vector because vector shapes
//       must be static. `vector.create_mask` and `arith.select` restrict
//       accesses to the subview's extent. The subview must start at the
//       parent's origin.
//
//   * A supported `memref.expand_shape` or `memref.collapse_shape` reshapes
//     the parent slot's vector with `vector.shape_cast` in both directions.
//     For a dynamic subview, the alias retains all elements of the parent
//     vector; the subview mask is cast to the alias shape when reading.
//
// Copies and subviews require a statically shaped parent buffer. A dynamically
// shaped buffer can instead promote to a 1-D scalable vector through
// whole-buffer transfers, but copies and subviews are not supported.
//
// Unsupported accesses, such as dynamic subview offsets, rank reduction,
// non-unit subview strides, reshapes that cannot be represented by
// `vector.shape_cast`, or non-zero transfer indices, prevent promotion.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/Transforms/MemorySlotInterfacesImpl.h"

#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/IR/MemRefDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"

using namespace mlir;
using namespace mlir::memref;
using namespace mlir::vector;

//===----------------------------------------------------------------------===//
//  Utilities
//===----------------------------------------------------------------------===//

/// Checks the requirements shared by static and dynamic subview aliases: a
/// parent with a static shape and a valid vector element type, no rank
/// reduction, and unit slice strides.
static bool isUnitStrideSliceOfStaticParent(memref::SubViewOp subView) {
  MemRefType srcType = subView.getSourceType();
  // vector requires a static parent shape.
  if (!srcType.hasStaticShape())
    return false;

  // implementation limitation: vector.extract_strided_slice must have the same
  // rank (static). Alternative vector.extract could be used but only drop
  // leading dimensions.
  if (srcType.getRank() != subView.getType().getRank())
    return false;

  // No straightforward mapping to vector code for non-unit stride slices.
  if (!subView.hasUnitStride())
    return false;

  return VectorType::isValidElementType(srcType.getElementType());
}

/// Returns whether static `subView` can alias a slice of the parent vector
/// using `vector.extract_strided_slice` and `vector.insert_strided_slice`.
/// Requires static source and result shapes, constant offsets, unit strides,
/// and no rank reduction.
static bool isAliasableStaticShapeSubView(memref::SubViewOp subView) {
  if (!subView.getType().hasStaticShape())
    return false;

  if (!isUnitStrideSliceOfStaticParent(subView))
    return false;

  // extract/insert_strided_slice requires all offsets to be constant.
  return llvm::all_of(subView.getMixedOffsets(), [](OpFoldResult offset) {
    return getConstantIntValue(offset).has_value();
  });
}

/// Returns whether dynamic `subView` can alias the whole parent vector. The
/// alias uses the parent vector's shape because vector dimensions cannot be
/// dynamic. Requires a static source shape, a dynamic result shape, zero
/// offsets, unit strides, and no rank reduction.
static bool isAliasableDynamicShapeSubView(memref::SubViewOp subView) {
  if (subView.getType().hasStaticShape())
    return false;

  if (!isUnitStrideSliceOfStaticParent(subView))
    return false;

  /// TODO: Allow non-zero offsets by improving buildSubViewMask().
  return subView.hasZeroOffset();
}

/// Returns the static offsets of `subView` within the parent vector for
/// `vector.extract_strided_slice` and `vector.insert_strided_slice`.
static SmallVector<int64_t> getStaticSubViewOffsets(memref::SubViewOp subView) {
  SmallVector<int64_t> offsets;
  for (OpFoldResult offset : subView.getMixedOffsets()) {
    std::optional<int64_t> value = getConstantIntValue(offset);
    assert(value && "expected a static offset");
    offsets.push_back(*value);
  }
  return offsets;
}

/// The vector type covering `subView`'s whole parent buffer, which is the type
/// the alias slot of a dynamically-shaped subview takes.
static VectorType getWholeParentVectorType(memref::SubViewOp subView) {
  auto parentType = cast<MemRefType>(subView.getSource().getType());
  assert(parentType.hasStaticShape() && "expected a statically-shaped parent");
  return VectorType::get(parentType.getShape(), parentType.getElementType());
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

/// Checks whether a reassociation group preserves the mapping of valid view
/// elements to vector lanes. `viewExtents` are the memref dimensions;
/// `aliasExtents` are the corresponding static vector dimensions.
///
/// Leading unit dimensions contribute no offset. The first non-unit dimension
/// may differ because its extent does not affect its own row-major stride.
/// All subsequent extents must match to keep that stride unchanged. Thus a
/// supported group with a dynamic extent has the form
/// `[1, ..., 1, dynamic, static, ..., static]`.
static bool shapeCastModelsGroup(ArrayRef<int64_t> viewExtents,
                                 ArrayRef<int64_t> aliasExtents) {
  assert(viewExtents.size() == aliasExtents.size() && "group size mismatch");
  // Leading unit dimensions always have index zero.
  size_t pos = 0;
  while (pos < viewExtents.size() && viewExtents[pos] == 1)
    ++pos;
  // Allow the first non-unit extent to differ, but require its stride factors
  // (all following extents) to match.
  if (pos < viewExtents.size())
    ++pos;
  for (; pos < viewExtents.size(); ++pos)
    if (viewExtents[pos] != aliasExtents[pos])
      return false;
  return true;
}

/// Computes the collapsed alias shape by multiplying the extents from
/// `parentSlotShape` in each reassociation group. These are the parent slot's
/// static vector extents, even when the source memref has dynamic dimensions.
/// Fails if a group cannot be represented by `vector.shape_cast`.
static FailureOr<SmallVector<int64_t>>
getShapeCastableAliasShape(memref::CollapseShapeOp op,
                           ArrayRef<int64_t> parentSlotShape) {
  ArrayRef<int64_t> viewShape = op.getSrcType().getShape();
  // Reassociation indices must address both the memref and vector shapes.
  if (viewShape.size() != parentSlotShape.size())
    return failure();

  SmallVector<int64_t> aliasShape;
  for (ReassociationIndices group : op.getReassociationIndices()) {
    SmallVector<int64_t> viewExtents, aliasExtents;
    for (int64_t dim : group) {
      viewExtents.push_back(viewShape[dim]);
      aliasExtents.push_back(parentSlotShape[dim]);
    }
    if (!shapeCastModelsGroup(viewExtents, aliasExtents))
      return failure();
    aliasShape.push_back(ShapedType::getNumElements(aliasExtents));
  }
  if (ShapedType::getNumElements(aliasShape) !=
      ShapedType::getNumElements(parentSlotShape))
    return failure();
  return aliasShape;
}

/// Computes a static expanded alias shape from `parentSlotShape`, the parent
/// slot's static vector extents. For each reassociation group, divides the
/// corresponding extent from `parentSlotShape` by the static result extents,
/// starting at the innermost dimension. The first dynamic dimension receives
/// the remaining extent; dimensions before it remain 1.
///
/// Fails if the parent extent cannot be evenly divided among the static result
/// dimensions, or if a group fails `shapeCastModelsGroup`. A group with
/// dynamic dimensions must have the form
/// `[1, ..., 1, dynamic, static, ..., static]`.
static FailureOr<SmallVector<int64_t>>
getShapeCastableAliasShape(memref::ExpandShapeOp op,
                           ArrayRef<int64_t> parentSlotShape) {
  ArrayRef<int64_t> viewShape = op.getType().getShape();
  SmallVector<ReassociationIndices, 4> groups = op.getReassociationIndices();
  if (groups.size() != parentSlotShape.size())
    return failure();

  SmallVector<int64_t> aliasShape(viewShape.size(), 1);
  for (auto [srcDim, group] : llvm::enumerate(groups)) {
    int64_t remaining = parentSlotShape[srcDim];
    for (int64_t dim : llvm::reverse(group)) {
      if (ShapedType::isDynamic(viewShape[dim])) {
        aliasShape[dim] = remaining;
        break;
      }
      if (viewShape[dim] == 0 || remaining % viewShape[dim] != 0)
        return failure();
      aliasShape[dim] = viewShape[dim];
      remaining /= viewShape[dim];
    }

    SmallVector<int64_t> viewExtents, aliasExtents;
    for (int64_t dim : group) {
      viewExtents.push_back(viewShape[dim]);
      aliasExtents.push_back(aliasShape[dim]);
    }
    if (!shapeCastModelsGroup(viewExtents, aliasExtents))
      return failure();
  }
  if (ShapedType::getNumElements(aliasShape) !=
      ShapedType::getNumElements(parentSlotShape))
    return failure();
  return aliasShape;
}

//===----------------------------------------------------------------------===//
//  PromotableMemOpInterface model for memref.copy
//===----------------------------------------------------------------------===//
namespace {
/// `PromotableMemOpInterface` model for `memref.copy`.
///
/// A copy into the slot becomes a `vector.transfer_read` of the source in
/// `getStored`. A copy out of the slot becomes a `vector.transfer_write` of the
/// reaching definition in `removeBlockingUses`.
///
/// A copy between two slots is rewritten one slot at a time, in either order.
/// For dynamic subviews,
/// `PromotableAliaserInterface::projectAliasValueToSlotValue` masks writes into
/// the slot.
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
                  Value /*reachingDef*/, const DataLayout &dataLayout) const {
    auto copyOp = cast<memref::CopyOp>(op);
    return vector::createReadOrMaskedRead(builder, op->getLoc(),
                                          copyOp.getSource(),
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
    // No further check: `memref.copy` requires both operands to have the same
    // element type and runtime shape; a shape mismatch is undefined behavior.
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
    if (copyOp.getSource() == slot.ptr)
      vector::createWriteOrMaskedWrite(builder, op->getLoc(),
                                       reachingDefinition, copyOp.getTarget());
    return DeletionKind::Delete;
  }
};

//===----------------------------------------------------------------------===//
//  PromotableOpInterface model for the memref view ops
//===----------------------------------------------------------------------===//

/// Removes `memref.subview`, `memref.expand_shape`, and `memref.collapse_shape`
/// once all uses of their results have been promoted.
template <typename OpTy>
struct ViewOpPromotableModel
    : public PromotableOpInterface::ExternalModel<ViewOpPromotableModel<OpTy>,
                                                  OpTy> {
  bool canUsesBeRemoved(Operation *op,
                        const SmallPtrSetImpl<OpOperand *> &blockingUses,
                        SmallVectorImpl<OpOperand *> &newBlockingUses,
                        const DataLayout &dataLayout) const {
    // Removing the view requires removing all uses of its memref result.
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

//===----------------------------------------------------------------------===//
//  PromotableAliaserInterface model for memref.subview
//===----------------------------------------------------------------------===//

/// `PromotableAliaserInterface` model exposing a same-rank `memref.subview` as
/// an alias of a vector slot.
///
/// `projectSlotValueToAliasValue` provides the alias value for loads and for a
/// store's `getStored` hook. `projectAliasValueToSlotValue` merges the stored
/// alias value back into the parent.
///
/// Static subviews use `vector.extract_strided_slice` and
/// `vector.insert_strided_slice`. Dynamic subviews keep the whole parent
/// vector: loads use it directly, and stores use `arith.select` with a mask of
/// the subview's sizes to preserve the parent value outside the subview.
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
    // Identity: the alias holds the whole parent value. Lanes past the
    // subview's extent are expected to be masked off by the consuming read.
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

//===----------------------------------------------------------------------===//
//  PromotableAliaserInterface model for the memref reshape ops
//===----------------------------------------------------------------------===//

/// Exposes supported `memref.expand_shape` and `memref.collapse_shape` ops as
/// aliases of vector slots. Both projections use `vector.shape_cast`, which
/// preserves element order. `getShapeCastableAliasShape` checks eligibility
/// and derives the alias shape from the parent vector's static extents.
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

    // Only fixed-size vector slots support these shape casts.
    auto parentVecType = dyn_cast<VectorType>(parentSlot.valueType);
    if (!parentVecType || parentVecType.isScalable())
      return;

    auto resType = cast<MemRefType>(viewOp.getResult().getType());
    if (!VectorType::isValidElementType(resType.getElementType()) ||
        resType.getElementType() != parentVecType.getElementType())
      return;
    FailureOr<SmallVector<int64_t>> aliasShape =
        getShapeCastableAliasShape(viewOp, parentVecType.getShape());
    if (failed(aliasShape))
      return;
    auto aliasVecType = VectorType::get(*aliasShape, resType.getElementType());
    newSlots.push_back(MemorySlot{viewOp.getResult(), aliasVecType});
  }

  Value projectSlotValueToAliasValue(Operation *op,
                                     OpOperand & /*aliasedSlotPointerOperand*/,
                                     const MemorySlot & /*parentSlot*/,
                                     const MemorySlot &aliasSlot,
                                     Value slotValue,
                                     OpBuilder &builder) const {
    return vector::ShapeCastOp::create(builder, op->getLoc(),
                                       cast<VectorType>(aliasSlot.valueType),
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
                                       cast<VectorType>(parentSlot.valueType),
                                       aliasValue)
        .getResult();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
//  Register external models
//===----------------------------------------------------------------------===//

void mlir::memref::registerMemorySlotInterfacesExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, memref::MemRefDialect *dialect) {
    memref::SubViewOp::attachInterface<SubViewOpAliasModel>(*ctx);
    memref::SubViewOp::attachInterface<
        ViewOpPromotableModel<memref::SubViewOp>>(*ctx);
    memref::CopyOp::attachInterface<CopyOpMemOpModel>(*ctx);
    memref::ExpandShapeOp::attachInterface<
        ReassociativeViewOpAliasModel<memref::ExpandShapeOp>>(*ctx);
    memref::ExpandShapeOp::attachInterface<
        ViewOpPromotableModel<memref::ExpandShapeOp>>(*ctx);
    memref::CollapseShapeOp::attachInterface<
        ReassociativeViewOpAliasModel<memref::CollapseShapeOp>>(*ctx);
    memref::CollapseShapeOp::attachInterface<
        ViewOpPromotableModel<memref::CollapseShapeOp>>(*ctx);
  });
}
