//===- MemorySlotOpInterfaceImpl.cpp - Mem2Reg for MemRef ops -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements memory slot interface models used by Mem2Reg to promote
// a memref to a single vector SSA value. `PromotableAliaserInterface` exposes
// `memref.subview` aliases, and `PromotableOpInterface` removes the subviews
// after promotion. Vector transfer ops are handled by the models in the
// Vector dialect.
//
// A memory slot pairs a memref with the vector type used to represent its
// contents. Promotion replaces accesses to the slot with uses and definitions
// of that vector value. All uses, including those through subviews, must be
// supported by the models.
//
// The accesses are rewritten as follows:
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
// Subviews require a statically shaped parent buffer. A dynamically
// shaped buffer can instead promote to a 1-D scalable vector through
// whole-buffer transfers, but subviews are not supported.
//
// Unsupported accesses, such as dynamic subview offsets, rank reduction,
// non-unit subview strides, or non-zero transfer indices, prevent promotion.
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

/// Returns whether `subView` can be promoted as an alias of a sub-vector.
/// Promotion uses `vector.extract_strided_slice` and
/// `vector.insert_strided_slice`, which require:
///   * fully static offsets and sizes,
///   * unit strides,
///   * no rank reduction (result rank == source rank),
/// Dynamic shapes are handled by `isAliasableDynamicShapeSubView` instead.
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

/// Returns whether `subView` can be promoted as an alias of the whole parent
/// vector. Static shapes are handled by `isAliasableStaticShapeSubView`
/// instead.
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

namespace {
//===----------------------------------------------------------------------===//
//  PromotableOpInterface model for memref.subview
//===----------------------------------------------------------------------===//

/// `PromotableOpInterface` model for `memref.subview`. Works with the
/// `PromotableAliaserInterface` model below to remove the subview once its
/// result's uses have been promoted.
struct SubViewOpPromotableModel
    : public PromotableOpInterface::ExternalModel<SubViewOpPromotableModel,
                                                  memref::SubViewOp> {
  bool canUsesBeRemoved(Operation *op,
                        const SmallPtrSetImpl<OpOperand *> &blockingUses,
                        SmallVectorImpl<OpOperand *> &newBlockingUses,
                        const DataLayout &dataLayout) const {
    // Removing the subview requires removing all uses of its memref result.
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

bool mlir::memref::isDynamicSubViewSlot(Value slotPtr) {
  return static_cast<bool>(getDynamicSubView(slotPtr));
}

Value mlir::memref::buildDynamicSubViewMask(OpBuilder &builder, Location loc,
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
  });
}
