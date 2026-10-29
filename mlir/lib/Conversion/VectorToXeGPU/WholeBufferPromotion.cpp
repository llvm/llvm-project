//===- WholeBufferPromotion.cpp - Mem2Reg for XeGPU preprocessing ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements a VectorToXeGPU-local, Mem2Reg-style promotion of whole-buffer
// `memref.alloc`s into vector SSA values. See WholeBufferPromotion.h for the
// rationale. The heavy lifting (dominator walk, reaching-definition
// construction, and `scf.for` iter_arg/result threading) is delegated to the
// upstream `tryToPromoteMemorySlots` driver; this file only supplies the
// interface models that teach that driver how a whole-buffer transfer behaves
// as a load/store.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/VectorToXeGPU/VectorToXeGPU.h"

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "mlir/Transforms/Mem2Reg.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
//  Utilities
//===----------------------------------------------------------------------===//

/// Returns whether `xferOp` accesses exactly the whole contents of `slot`, so
/// it can act as a plain whole-buffer load/store during promotion.
static bool
isWholeBufferTransfer(VectorTransferOpInterface xferOp, const MemorySlot &slot,
                      const SmallPtrSetImpl<OpOperand *> &blockingUses) {
  // The sole blocking use must be the slot pointer as the transfer's base.
  if (blockingUses.size() != 1)
    return false;
  Value blockingUse = (*blockingUses.begin())->get();
  if (blockingUse != slot.ptr || xferOp.getBase() != slot.ptr)
    return false;

  // Reject the tensor form (slot pointers are memrefs, so this is defensive).
  if (!isa<MemRefType>(xferOp.getBase().getType()))
    return false;

  // Exact type match pins rank/extents/element type and rejects scalable
  // vectors.
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

  // All dimensions in bounds: no out-of-buffer element, no padding applied.
  if (xferOp.hasOutOfBoundsDim())
    return false;

  // A mask could make the access partial.
  if (xferOp.getMask())
    return false;

  return true;
}

//===----------------------------------------------------------------------===//
//  Interface models
//===----------------------------------------------------------------------===//

namespace {

/// Promotable-allocation model for `memref.alloc`. `memref.alloca` already
/// carries this interface from ODS (and is only ever promoted as a scalar
/// slot); attaching to `alloc` is conflict-free and lets whole-buffer heap
/// scratch produced by bufferization be promoted without a prior
/// promote-buffers-to-stack step.
struct AllocOpPromotableModel
    : public PromotableAllocationOpInterface::ExternalModel<
          AllocOpPromotableModel, memref::AllocOp> {
  SmallVector<MemorySlot> getPromotableSlots(Operation *op) const {
    auto allocOp = cast<memref::AllocOp>(op);
    MemRefType type = allocOp.getType();
    if (!type.hasStaticShape())
      return {};

    // Only a whole-buffer vector slot is offered here; the single-element
    // scalar case is already covered by memref.alloca upstream and is not the
    // pattern this preprocessing targets. The buffer-size cap is applied by the
    // caller (promoteWholeBufferAllocs), which filters the allocator list.
    if (type.getNumElements() <= 1)
      return {};
    if (!VectorType::isValidElementType(type.getElementType()))
      return {};

    return {
        MemorySlot{allocOp.getResult(),
                   VectorType::get(type.getShape(), type.getElementType())}};
  }

  Value getDefaultValue(Operation *op, const MemorySlot &slot,
                        OpBuilder &builder) const {
    return ub::PoisonOp::create(builder, op->getLoc(), slot.elemType);
  }

  void handleBlockArgument(Operation *op, const MemorySlot &slot,
                           BlockArgument argument, OpBuilder &builder) const {}

  std::optional<PromotableAllocationOpInterface>
  handlePromotionComplete(Operation *op, const MemorySlot &slot,
                          Value defaultValue, OpBuilder &builder) const {
    if (defaultValue && defaultValue.use_empty())
      defaultValue.getDefiningOp()->erase();
    op->erase();
    return std::nullopt;
  }
};

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

  DeletionKind
  removeBlockingUses(Operation *op, const MemorySlot &slot,
                     const SmallPtrSetImpl<OpOperand *> &blockingUses,
                     OpBuilder &builder, Value reachingDefinition,
                     const DataLayout &dataLayout) const {
    // Whole-buffer read: replace the loaded vector with the reaching
    // definition.
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
    // No self-store guard needed: a vector value can never equal a memref slot.
    return isWholeBufferTransfer(cast<vector::TransferWriteOp>(op), slot,
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

} // namespace

//===----------------------------------------------------------------------===//
//  Public entry points
//===----------------------------------------------------------------------===//

void mlir::xegpu::registerWholeBufferPromotionExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, memref::MemRefDialect *dialect) {
    memref::AllocOp::attachInterface<AllocOpPromotableModel>(*ctx);
  });
  registry.addExtension(+[](MLIRContext *ctx, vector::VectorDialect *dialect) {
    vector::TransferReadOp::attachInterface<TransferReadOpMemOpModel>(*ctx);
    vector::TransferWriteOp::attachInterface<TransferWriteOpMemOpModel>(*ctx);
  });
}

void mlir::xegpu::promoteWholeBufferAllocs(Operation *scopeOp,
                                           uint64_t maxPromotedBytes) {
  DataLayoutAnalysis dataLayoutAnalysis(scopeOp);
  const DataLayout &dataLayout = dataLayoutAnalysis.getAtOrAbove(scopeOp);
  DominanceInfo dominance(scopeOp);

  // Returns the in-memory size in bytes of `allocOp`'s static, whole-buffer
  // result, or nullopt if it is not a candidate for size-capped promotion.
  auto allocByteSize = [&](PromotableAllocationOpInterface allocator)
      -> std::optional<uint64_t> {
    auto allocOp = dyn_cast<memref::AllocOp>(allocator.getOperation());
    if (!allocOp)
      return std::nullopt;
    MemRefType type = allocOp.getType();
    if (!type.hasStaticShape())
      return std::nullopt;
    uint64_t elementBytes = dataLayout.getTypeSize(type.getElementType());
    return elementBytes * static_cast<uint64_t>(type.getNumElements());
  };

  for (Region &region : scopeOp->getRegions()) {
    if (region.getBlocks().empty())
      continue;

    OpBuilder builder(&region.front(), region.front().begin());

    SmallVector<PromotableAllocationOpInterface> allocators;
    region.walk([&](PromotableAllocationOpInterface allocator) {
      // Enforce the buffer-size cap here (the interface model cannot carry the
      // pass option). Allocators too large to promote are simply not offered.
      if (std::optional<uint64_t> bytes = allocByteSize(allocator))
        if (*bytes > maxPromotedBytes)
          return;
      allocators.emplace_back(allocator);
    });

    // Iteratively promote as many slots as possible; the driver leaves any
    // non-whole-buffer allocation untouched with zero IR mutation.
    (void)tryToPromoteMemorySlots(allocators, builder, dataLayout, dominance);
  }
}
