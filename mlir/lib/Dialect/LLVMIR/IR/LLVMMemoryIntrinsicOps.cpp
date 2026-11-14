//===- LLVMMemoryIntrinsicOps.cpp - LLVM dialect memory intrinsics --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ops of op partition `MemoryIntrinsic` of the LLVM
// dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMMemoryIntrinsicOps.h"

#include "IR/LLVMOps.h"
#include "mlir/Dialect/LLVMIR/LLVMArithOps.h"
#include "mlir/Dialect/LLVMIR/LLVMCastOps.h"
#include "mlir/Dialect/LLVMIR/LLVMConstantOps.h"
#include "mlir/Dialect/LLVMIR/LLVMMemoryOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include <optional>

using namespace mlir;
using namespace mlir::LLVM;

bool LLVM::LifetimeStartOp::canUsesBeRemoved(
    const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  return true;
}

DeletionKind LLVM::LifetimeStartOp::removeBlockingUses(
    const SmallPtrSetImpl<OpOperand *> &blockingUses, OpBuilder &builder) {
  return DeletionKind::Delete;
}

bool LLVM::LifetimeEndOp::canUsesBeRemoved(
    const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  return true;
}

DeletionKind LLVM::LifetimeEndOp::removeBlockingUses(
    const SmallPtrSetImpl<OpOperand *> &blockingUses, OpBuilder &builder) {
  return DeletionKind::Delete;
}

bool LLVM::InvariantStartOp::canUsesBeRemoved(
    const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  return true;
}

DeletionKind LLVM::InvariantStartOp::removeBlockingUses(
    const SmallPtrSetImpl<OpOperand *> &blockingUses, OpBuilder &builder) {
  return DeletionKind::Delete;
}

bool LLVM::InvariantEndOp::canUsesBeRemoved(
    const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  return true;
}

DeletionKind LLVM::InvariantEndOp::removeBlockingUses(
    const SmallPtrSetImpl<OpOperand *> &blockingUses, OpBuilder &builder) {
  return DeletionKind::Delete;
}

bool LLVM::LaunderInvariantGroupOp::canUsesBeRemoved(
    const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  return forwardToUsers(*this, newBlockingUses);
}

DeletionKind LLVM::LaunderInvariantGroupOp::removeBlockingUses(
    const SmallPtrSetImpl<OpOperand *> &blockingUses, OpBuilder &builder) {
  return DeletionKind::Delete;
}

//===----------------------------------------------------------------------===//
// Utilities for memory intrinsics
//===----------------------------------------------------------------------===//

namespace {
/// Returns the length of the given memory intrinsic in bytes if it can be known
/// at compile-time on a best-effort basis, nothing otherwise.
template <class MemIntr>
std::optional<uint64_t> getStaticMemIntrLen(MemIntr op) {
  APInt memIntrLen;
  if (!matchPattern(op.getLen(), m_ConstantInt(&memIntrLen)))
    return {};
  if (memIntrLen.getBitWidth() > 64)
    return {};
  return memIntrLen.getZExtValue();
}

/// Returns the length of the given memory intrinsic in bytes if it can be known
/// at compile-time on a best-effort basis, nothing otherwise.
/// Because MemcpyInlineOp has its length encoded as an attribute, this requires
/// specialized handling.
template <>
std::optional<uint64_t> getStaticMemIntrLen(LLVM::MemcpyInlineOp op) {
  APInt memIntrLen = op.getLen();
  if (memIntrLen.getBitWidth() > 64)
    return {};
  return memIntrLen.getZExtValue();
}

/// Returns the length of the given memory intrinsic in bytes if it can be known
/// at compile-time on a best-effort basis, nothing otherwise.
/// Because MemsetInlineOp has its length encoded as an attribute, this requires
/// specialized handling.
template <>
std::optional<uint64_t> getStaticMemIntrLen(LLVM::MemsetInlineOp op) {
  APInt memIntrLen = op.getLen();
  if (memIntrLen.getBitWidth() > 64)
    return {};
  return memIntrLen.getZExtValue();
}

/// Returns an integer attribute representing the length of a memset intrinsic
template <class MemsetIntr>
IntegerAttr createMemsetLenAttr(MemsetIntr op) {
  IntegerAttr memsetLenAttr;
  bool successfulMatch =
      matchPattern(op.getLen(), m_Constant<IntegerAttr>(&memsetLenAttr));
  (void)successfulMatch;
  assert(successfulMatch);
  return memsetLenAttr;
}

/// Returns an integer attribute representing the length of a memset intrinsic
/// Because MemsetInlineOp has its length encoded as an attribute, this requires
/// specialized handling.
template <>
IntegerAttr createMemsetLenAttr(LLVM::MemsetInlineOp op) {
  return op.getLenAttr();
}

/// Creates a memset intrinsic of that matches the `toReplace` intrinsic
/// using the provided parameters. There are template specializations for
/// MemsetOp and MemsetInlineOp.
template <class MemsetIntr>
void createMemsetIntr(OpBuilder &builder, MemsetIntr toReplace,
                      IntegerAttr memsetLenAttr, uint64_t newMemsetSize,
                      DenseMap<Attribute, MemorySlot> &subslots,
                      Attribute index);

template <>
void createMemsetIntr(OpBuilder &builder, LLVM::MemsetOp toReplace,
                      IntegerAttr memsetLenAttr, uint64_t newMemsetSize,
                      DenseMap<Attribute, MemorySlot> &subslots,
                      Attribute index) {
  Value newMemsetSizeValue =
      LLVM::ConstantOp::create(
          builder, toReplace.getLen().getLoc(),
          IntegerAttr::get(memsetLenAttr.getType(), newMemsetSize))
          .getResult();

  LLVM::MemsetOp::create(builder, toReplace.getLoc(), subslots.at(index).ptr,
                         toReplace.getVal(), newMemsetSizeValue,
                         toReplace.getIsVolatile());
}

template <>
void createMemsetIntr(OpBuilder &builder, LLVM::MemsetInlineOp toReplace,
                      IntegerAttr memsetLenAttr, uint64_t newMemsetSize,
                      DenseMap<Attribute, MemorySlot> &subslots,
                      Attribute index) {
  auto newMemsetSizeValue =
      IntegerAttr::get(memsetLenAttr.getType(), newMemsetSize);

  LLVM::MemsetInlineOp::create(builder, toReplace.getLoc(),
                               subslots.at(index).ptr, toReplace.getVal(),
                               newMemsetSizeValue, toReplace.getIsVolatile());
}
} // namespace

/// Returns whether one can be sure the memory intrinsic does not write outside
/// of the bounds of the given slot, on a best-effort basis.
template <class MemIntr>
static bool definitelyWritesOnlyWithinSlot(MemIntr op, const MemorySlot &slot,
                                           const DataLayout &dataLayout) {
  if (!isa<LLVM::LLVMPointerType>(slot.ptr.getType()) ||
      op.getDst() != slot.ptr)
    return false;

  std::optional<uint64_t> memIntrLen = getStaticMemIntrLen(op);
  return memIntrLen && *memIntrLen <= dataLayout.getTypeSize(slot.valueType);
}

/// Checks whether all indices are i32. This is used to check GEPs can index
/// into them.
static bool areAllIndicesI32(const DestructurableMemorySlot &slot) {
  Type i32 = IntegerType::get(slot.ptr.getContext(), 32);
  return llvm::all_of(llvm::make_first_range(slot.subelementTypes),
                      [&](Attribute index) {
                        auto intIndex = dyn_cast<IntegerAttr>(index);
                        return intIndex && intIndex.getType() == i32;
                      });
}

//===----------------------------------------------------------------------===//
// Interfaces for memset and memset.inline
//===----------------------------------------------------------------------===//

template <class MemsetIntr>
static bool memsetCanRewire(MemsetIntr op, const DestructurableMemorySlot &slot,
                            SmallPtrSetImpl<Attribute> &usedIndices,
                            SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
                            const DataLayout &dataLayout) {
  if (&slot.valueType.getDialect() != op.getOperation()->getDialect())
    return false;

  if (op.getIsVolatile())
    return false;

  if (!cast<DestructurableTypeInterface>(slot.valueType)
           .getSubelementIndexMap())
    return false;

  if (!areAllIndicesI32(slot))
    return false;

  return definitelyWritesOnlyWithinSlot(op, slot, dataLayout);
}

template <class MemsetIntr>
static Value memsetGetStored(MemsetIntr op, const MemorySlot &slot,
                             OpBuilder &builder) {
  /// Returns an integer value that is `width` bits wide representing the value
  /// assigned to the slot by memset.
  auto buildMemsetValue = [&](unsigned width) -> Value {
    assert(width % 8 == 0);
    auto intType = IntegerType::get(op.getContext(), width);

    // If we know the pattern at compile time, we can compute and assign a
    // constant directly.
    IntegerAttr constantPattern;
    if (matchPattern(op.getVal(), m_Constant(&constantPattern))) {
      assert(constantPattern.getValue().getBitWidth() == 8);
      APInt memsetVal(/*numBits=*/width, /*val=*/0);
      for (unsigned loBit = 0; loBit < width; loBit += 8)
        memsetVal.insertBits(constantPattern.getValue(), loBit);
      return LLVM::ConstantOp::create(builder, op.getLoc(),
                                      IntegerAttr::get(intType, memsetVal));
    }

    // If the output is a single byte, we can return the pattern directly.
    if (width == 8)
      return op.getVal();

    // Otherwise build the memset integer at runtime by repeatedly shifting the
    // value and or-ing it with the previous value.
    uint64_t coveredBits = 8;
    Value currentValue =
        LLVM::ZExtOp::create(builder, op.getLoc(), intType, op.getVal());
    while (coveredBits < width) {
      Value shiftBy =
          LLVM::ConstantOp::create(builder, op.getLoc(), intType, coveredBits);
      Value shifted =
          LLVM::ShlOp::create(builder, op.getLoc(), currentValue, shiftBy);
      currentValue =
          LLVM::OrOp::create(builder, op.getLoc(), currentValue, shifted);
      coveredBits *= 2;
    }

    return currentValue;
  };
  return TypeSwitch<Type, Value>(slot.valueType)
      .Case([&](IntegerType type) -> Value {
        return buildMemsetValue(type.getWidth());
      })
      .Case([&](FloatType type) -> Value {
        Value intVal = buildMemsetValue(type.getWidth());
        return LLVM::BitcastOp::create(builder, op.getLoc(), type, intVal);
      })
      .DefaultUnreachable(
          "getStored should not be called on memset to unsupported type");
}

template <class MemsetIntr>
static bool
memsetCanUsesBeRemoved(MemsetIntr op, const MemorySlot &slot,
                       const SmallPtrSetImpl<OpOperand *> &blockingUses,
                       SmallVectorImpl<OpOperand *> &newBlockingUses,
                       const DataLayout &dataLayout) {
  bool canConvertType =
      TypeSwitch<Type, bool>(slot.valueType)
          .Case<IntegerType, FloatType>([](auto type) {
            return type.getWidth() % 8 == 0 && type.getWidth() > 0;
          })
          .Default(false);
  if (!canConvertType)
    return false;

  if (op.getIsVolatile())
    return false;

  return getStaticMemIntrLen(op) == dataLayout.getTypeSize(slot.valueType);
}

template <class MemsetIntr>
static DeletionKind
memsetRewire(MemsetIntr op, const DestructurableMemorySlot &slot,
             DenseMap<Attribute, MemorySlot> &subslots, OpBuilder &builder,
             const DataLayout &dataLayout) {

  std::optional<DenseMap<Attribute, Type>> types =
      cast<DestructurableTypeInterface>(slot.valueType).getSubelementIndexMap();

  IntegerAttr memsetLenAttr = createMemsetLenAttr(op);

  auto structType = dyn_cast<LLVM::LLVMStructType>(slot.valueType);

  Type i32 = IntegerType::get(op.getContext(), 32);
  uint64_t memsetLen = memsetLenAttr.getValue().getZExtValue();
  for (size_t i = 0; i < types->size(); i++) {
    // Create indices on the fly to get elements in the right order.
    Attribute index = IntegerAttr::get(i32, i);
    Type elemType = types->at(index);
    uint64_t typeSize = dataLayout.getTypeSize(elemType);
    uint64_t offset = structType ? structType.getElementOffset(dataLayout, i)
                                 : i * dataLayout.getTypeAllocSize(elemType);
    if (offset >= memsetLen)
      break;

    // If this subslot is used, apply a new memset to it.
    if (subslots.contains(index)) {
      uint64_t newMemsetSize = std::min(memsetLen - offset, typeSize);
      createMemsetIntr(builder, op, memsetLenAttr, newMemsetSize, subslots,
                       index);
    }
  }

  return DeletionKind::Delete;
}

bool LLVM::MemsetOp::loadsFrom(const MemorySlot &slot) { return false; }

bool LLVM::MemsetOp::storesTo(const MemorySlot &slot) {
  return getDst() == slot.ptr;
}

Value LLVM::MemsetOp::getStored(const MemorySlot &slot, OpBuilder &builder,
                                Value reachingDef,
                                const DataLayout &dataLayout) {
  return memsetGetStored(*this, slot, builder);
}

bool LLVM::MemsetOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  return memsetCanUsesBeRemoved(*this, slot, blockingUses, newBlockingUses,
                                dataLayout);
}

DeletionKind LLVM::MemsetOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    OpBuilder &builder, Value reachingDefinition,
    const DataLayout &dataLayout) {
  return DeletionKind::Delete;
}

LogicalResult LLVM::MemsetOp::ensureOnlySafeAccesses(
    const MemorySlot &slot, SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
    const DataLayout &dataLayout) {
  return success(definitelyWritesOnlyWithinSlot(*this, slot, dataLayout));
}

bool LLVM::MemsetOp::canRewire(const DestructurableMemorySlot &slot,
                               SmallPtrSetImpl<Attribute> &usedIndices,
                               SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
                               const DataLayout &dataLayout) {
  return memsetCanRewire(*this, slot, usedIndices, mustBeSafelyUsed,
                         dataLayout);
}

DeletionKind LLVM::MemsetOp::rewire(const DestructurableMemorySlot &slot,
                                    DenseMap<Attribute, MemorySlot> &subslots,
                                    OpBuilder &builder,
                                    const DataLayout &dataLayout) {
  return memsetRewire(*this, slot, subslots, builder, dataLayout);
}

bool LLVM::MemsetInlineOp::loadsFrom(const MemorySlot &slot) { return false; }

bool LLVM::MemsetInlineOp::storesTo(const MemorySlot &slot) {
  return getDst() == slot.ptr;
}

Value LLVM::MemsetInlineOp::getStored(const MemorySlot &slot,
                                      OpBuilder &builder, Value reachingDef,
                                      const DataLayout &dataLayout) {
  return memsetGetStored(*this, slot, builder);
}

bool LLVM::MemsetInlineOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  return memsetCanUsesBeRemoved(*this, slot, blockingUses, newBlockingUses,
                                dataLayout);
}

DeletionKind LLVM::MemsetInlineOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    OpBuilder &builder, Value reachingDefinition,
    const DataLayout &dataLayout) {
  return DeletionKind::Delete;
}

LogicalResult LLVM::MemsetInlineOp::ensureOnlySafeAccesses(
    const MemorySlot &slot, SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
    const DataLayout &dataLayout) {
  return success(definitelyWritesOnlyWithinSlot(*this, slot, dataLayout));
}

bool LLVM::MemsetInlineOp::canRewire(
    const DestructurableMemorySlot &slot,
    SmallPtrSetImpl<Attribute> &usedIndices,
    SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
    const DataLayout &dataLayout) {
  return memsetCanRewire(*this, slot, usedIndices, mustBeSafelyUsed,
                         dataLayout);
}

DeletionKind
LLVM::MemsetInlineOp::rewire(const DestructurableMemorySlot &slot,
                             DenseMap<Attribute, MemorySlot> &subslots,
                             OpBuilder &builder, const DataLayout &dataLayout) {
  return memsetRewire(*this, slot, subslots, builder, dataLayout);
}

//===----------------------------------------------------------------------===//
// Interfaces for memcpy/memmove
//===----------------------------------------------------------------------===//

template <class MemcpyLike>
static bool memcpyLoadsFrom(MemcpyLike op, const MemorySlot &slot) {
  return op.getSrc() == slot.ptr;
}

template <class MemcpyLike>
static bool memcpyStoresTo(MemcpyLike op, const MemorySlot &slot) {
  return op.getDst() == slot.ptr;
}

template <class MemcpyLike>
static Value memcpyGetStored(MemcpyLike op, const MemorySlot &slot,
                             OpBuilder &builder) {
  return LLVM::LoadOp::create(builder, op.getLoc(), slot.valueType,
                              op.getSrc());
}

template <class MemcpyLike>
static bool
memcpyCanUsesBeRemoved(MemcpyLike op, const MemorySlot &slot,
                       const SmallPtrSetImpl<OpOperand *> &blockingUses,
                       SmallVectorImpl<OpOperand *> &newBlockingUses,
                       const DataLayout &dataLayout) {
  // If source and destination are the same, memcpy behavior is undefined and
  // memmove is a no-op. Because there is no memory change happening here,
  // simplifying such operations is left to canonicalization.
  if (op.getDst() == op.getSrc())
    return false;

  if (op.getIsVolatile())
    return false;

  return getStaticMemIntrLen(op) == dataLayout.getTypeSize(slot.valueType);
}

template <class MemcpyLike>
static DeletionKind
memcpyRemoveBlockingUses(MemcpyLike op, const MemorySlot &slot,
                         const SmallPtrSetImpl<OpOperand *> &blockingUses,
                         OpBuilder &builder, Value reachingDefinition) {
  if (op.loadsFrom(slot))
    LLVM::StoreOp::create(builder, op.getLoc(), reachingDefinition,
                          op.getDst());
  return DeletionKind::Delete;
}

template <class MemcpyLike>
static LogicalResult
memcpyEnsureOnlySafeAccesses(MemcpyLike op, const MemorySlot &slot,
                             SmallVectorImpl<MemorySlot> &mustBeSafelyUsed) {
  DataLayout dataLayout = DataLayout::closest(op);
  // While rewiring memcpy-like intrinsics only supports full copies, partial
  // copies are still safe accesses so it is enough to only check for writes
  // within bounds.
  return success(definitelyWritesOnlyWithinSlot(op, slot, dataLayout));
}

template <class MemcpyLike>
static bool memcpyCanRewire(MemcpyLike op, const DestructurableMemorySlot &slot,
                            SmallPtrSetImpl<Attribute> &usedIndices,
                            SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
                            const DataLayout &dataLayout) {
  if (op.getIsVolatile())
    return false;

  if (!cast<DestructurableTypeInterface>(slot.valueType)
           .getSubelementIndexMap())
    return false;

  if (!areAllIndicesI32(slot))
    return false;

  // Only full copies are supported.
  if (getStaticMemIntrLen(op) != dataLayout.getTypeSize(slot.valueType))
    return false;

  if (op.getSrc() == slot.ptr)
    usedIndices.insert_range(llvm::make_first_range(slot.subelementTypes));

  return true;
}

namespace {
template <class MemcpyLike>
void createMemcpyLikeToReplace(OpBuilder &builder, const DataLayout &layout,
                               MemcpyLike toReplace, Value dst, Value src,
                               Type toCpy, bool isVolatile) {
  Value memcpySize =
      LLVM::ConstantOp::create(builder, toReplace.getLoc(),
                               IntegerAttr::get(toReplace.getLen().getType(),
                                                layout.getTypeSize(toCpy)));
  MemcpyLike::create(builder, toReplace.getLoc(), dst, src, memcpySize,
                     isVolatile);
}

template <>
void createMemcpyLikeToReplace(OpBuilder &builder, const DataLayout &layout,
                               LLVM::MemcpyInlineOp toReplace, Value dst,
                               Value src, Type toCpy, bool isVolatile) {
  Type lenType = IntegerType::get(toReplace->getContext(),
                                  toReplace.getLen().getBitWidth());
  LLVM::MemcpyInlineOp::create(
      builder, toReplace.getLoc(), dst, src,
      IntegerAttr::get(lenType, layout.getTypeSize(toCpy)), isVolatile);
}
} // namespace

/// Rewires a memcpy-like operation. Only copies to or from the full slot are
/// supported.
template <class MemcpyLike>
static DeletionKind
memcpyRewire(MemcpyLike op, const DestructurableMemorySlot &slot,
             DenseMap<Attribute, MemorySlot> &subslots, OpBuilder &builder,
             const DataLayout &dataLayout) {
  if (subslots.empty())
    return DeletionKind::Delete;

  assert((slot.ptr == op.getDst()) != (slot.ptr == op.getSrc()));
  bool isDst = slot.ptr == op.getDst();

#ifndef NDEBUG
  size_t slotsTreated = 0;
#endif

  // It was previously checked that index types are consistent, so this type can
  // be fetched now.
  Type indexType = cast<IntegerAttr>(subslots.begin()->first).getType();
  for (size_t i = 0, e = slot.subelementTypes.size(); i != e; i++) {
    Attribute index = IntegerAttr::get(indexType, i);
    if (!subslots.contains(index))
      continue;
    const MemorySlot &subslot = subslots.at(index);

#ifndef NDEBUG
    slotsTreated++;
#endif

    // First get a pointer to the equivalent of this subslot from the source
    // pointer.
    SmallVector<LLVM::GEPArg> gepIndices{
        0, static_cast<int32_t>(
               cast<IntegerAttr>(index).getValue().getZExtValue())};
    Value subslotPtrInOther = LLVM::GEPOp::create(
        builder, op.getLoc(), LLVM::LLVMPointerType::get(op.getContext()),
        slot.valueType, isDst ? op.getSrc() : op.getDst(), gepIndices);

    // Then create a new memcpy out of this source pointer.
    createMemcpyLikeToReplace(builder, dataLayout, op,
                              isDst ? subslot.ptr : subslotPtrInOther,
                              isDst ? subslotPtrInOther : subslot.ptr,
                              subslot.valueType, op.getIsVolatile());
  }

  assert(subslots.size() == slotsTreated);

  return DeletionKind::Delete;
}

bool LLVM::MemcpyOp::loadsFrom(const MemorySlot &slot) {
  return memcpyLoadsFrom(*this, slot);
}

bool LLVM::MemcpyOp::storesTo(const MemorySlot &slot) {
  return memcpyStoresTo(*this, slot);
}

Value LLVM::MemcpyOp::getStored(const MemorySlot &slot, OpBuilder &builder,
                                Value reachingDef,
                                const DataLayout &dataLayout) {
  return memcpyGetStored(*this, slot, builder);
}

bool LLVM::MemcpyOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  return memcpyCanUsesBeRemoved(*this, slot, blockingUses, newBlockingUses,
                                dataLayout);
}

DeletionKind LLVM::MemcpyOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    OpBuilder &builder, Value reachingDefinition,
    const DataLayout &dataLayout) {
  return memcpyRemoveBlockingUses(*this, slot, blockingUses, builder,
                                  reachingDefinition);
}

LogicalResult LLVM::MemcpyOp::ensureOnlySafeAccesses(
    const MemorySlot &slot, SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
    const DataLayout &dataLayout) {
  return memcpyEnsureOnlySafeAccesses(*this, slot, mustBeSafelyUsed);
}

bool LLVM::MemcpyOp::canRewire(const DestructurableMemorySlot &slot,
                               SmallPtrSetImpl<Attribute> &usedIndices,
                               SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
                               const DataLayout &dataLayout) {
  return memcpyCanRewire(*this, slot, usedIndices, mustBeSafelyUsed,
                         dataLayout);
}

DeletionKind LLVM::MemcpyOp::rewire(const DestructurableMemorySlot &slot,
                                    DenseMap<Attribute, MemorySlot> &subslots,
                                    OpBuilder &builder,
                                    const DataLayout &dataLayout) {
  return memcpyRewire(*this, slot, subslots, builder, dataLayout);
}

bool LLVM::MemcpyInlineOp::loadsFrom(const MemorySlot &slot) {
  return memcpyLoadsFrom(*this, slot);
}

bool LLVM::MemcpyInlineOp::storesTo(const MemorySlot &slot) {
  return memcpyStoresTo(*this, slot);
}

Value LLVM::MemcpyInlineOp::getStored(const MemorySlot &slot,
                                      OpBuilder &builder, Value reachingDef,
                                      const DataLayout &dataLayout) {
  return memcpyGetStored(*this, slot, builder);
}

bool LLVM::MemcpyInlineOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  return memcpyCanUsesBeRemoved(*this, slot, blockingUses, newBlockingUses,
                                dataLayout);
}

DeletionKind LLVM::MemcpyInlineOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    OpBuilder &builder, Value reachingDefinition,
    const DataLayout &dataLayout) {
  return memcpyRemoveBlockingUses(*this, slot, blockingUses, builder,
                                  reachingDefinition);
}

LogicalResult LLVM::MemcpyInlineOp::ensureOnlySafeAccesses(
    const MemorySlot &slot, SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
    const DataLayout &dataLayout) {
  return memcpyEnsureOnlySafeAccesses(*this, slot, mustBeSafelyUsed);
}

bool LLVM::MemcpyInlineOp::canRewire(
    const DestructurableMemorySlot &slot,
    SmallPtrSetImpl<Attribute> &usedIndices,
    SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
    const DataLayout &dataLayout) {
  return memcpyCanRewire(*this, slot, usedIndices, mustBeSafelyUsed,
                         dataLayout);
}

DeletionKind
LLVM::MemcpyInlineOp::rewire(const DestructurableMemorySlot &slot,
                             DenseMap<Attribute, MemorySlot> &subslots,
                             OpBuilder &builder, const DataLayout &dataLayout) {
  return memcpyRewire(*this, slot, subslots, builder, dataLayout);
}

bool LLVM::MemmoveOp::loadsFrom(const MemorySlot &slot) {
  return memcpyLoadsFrom(*this, slot);
}

bool LLVM::MemmoveOp::storesTo(const MemorySlot &slot) {
  return memcpyStoresTo(*this, slot);
}

Value LLVM::MemmoveOp::getStored(const MemorySlot &slot, OpBuilder &builder,
                                 Value reachingDef,
                                 const DataLayout &dataLayout) {
  return memcpyGetStored(*this, slot, builder);
}

bool LLVM::MemmoveOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  return memcpyCanUsesBeRemoved(*this, slot, blockingUses, newBlockingUses,
                                dataLayout);
}

DeletionKind LLVM::MemmoveOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    OpBuilder &builder, Value reachingDefinition,
    const DataLayout &dataLayout) {
  return memcpyRemoveBlockingUses(*this, slot, blockingUses, builder,
                                  reachingDefinition);
}

LogicalResult LLVM::MemmoveOp::ensureOnlySafeAccesses(
    const MemorySlot &slot, SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
    const DataLayout &dataLayout) {
  return memcpyEnsureOnlySafeAccesses(*this, slot, mustBeSafelyUsed);
}

bool LLVM::MemmoveOp::canRewire(const DestructurableMemorySlot &slot,
                                SmallPtrSetImpl<Attribute> &usedIndices,
                                SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
                                const DataLayout &dataLayout) {
  return memcpyCanRewire(*this, slot, usedIndices, mustBeSafelyUsed,
                         dataLayout);
}

DeletionKind LLVM::MemmoveOp::rewire(const DestructurableMemorySlot &slot,
                                     DenseMap<Attribute, MemorySlot> &subslots,
                                     OpBuilder &builder,
                                     const DataLayout &dataLayout) {
  return memcpyRewire(*this, slot, subslots, builder, dataLayout);
}

SmallVector<Value> mlir::LLVM::MemcpyOp::getAccessedOperands() {
  return {getDst(), getSrc()};
}

SmallVector<Value> mlir::LLVM::MemcpyInlineOp::getAccessedOperands() {
  return {getDst(), getSrc()};
}

SmallVector<Value> mlir::LLVM::MemmoveOp::getAccessedOperands() {
  return {getDst(), getSrc()};
}

SmallVector<Value> mlir::LLVM::MemsetOp::getAccessedOperands() {
  return {getDst()};
}

SmallVector<Value> mlir::LLVM::MemsetInlineOp::getAccessedOperands() {
  return {getDst()};
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMMemoryIntrinsicOps.cpp.inc"
