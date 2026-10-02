//===- MemorySlotOpInterfaceImpl.cpp - Mem2Reg for MemRef ops -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/Transforms/MemorySlotOpInterfaceImpl.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/IR/MemRefDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"

using namespace mlir;
using namespace mlir::memref;
using namespace mlir::vector;

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
/// Exposes a static, same-rank `memref.subview` as a sub-slice alias of a
/// whole-buffer vector slot. Reads of the subview become
/// `vector.extract_strided_slice` of the parent value; writes become
/// `vector.insert_strided_slice` into the current reaching definition.
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

    // The parent slot must promote to a vector (whole-buffer promotion). A
    // scalar (single-element) parent slot cannot be sliced.
    auto parentVecType = dyn_cast<VectorType>(parentSlot.valueType);
    if (!parentVecType)
      return;

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
    SmallVector<int64_t> offsets = *getPromotableSubViewOffsets(subView);
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

void mlir::memref::registerMemorySlotOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, memref::MemRefDialect *dialect) {
    memref::SubViewOp::attachInterface<SubViewOpAliasModel>(*ctx);
    memref::SubViewOp::attachInterface<SubViewOpPromotableModel>(*ctx);
  });
}
