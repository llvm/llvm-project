//===- LLVMMemorySlot.cpp - MemorySlot interfaces ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements MemorySlot-related interfaces for LLVM dialect
// operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Interfaces for AllocaOp
//===----------------------------------------------------------------------===//

llvm::SmallVector<MemorySlot> LLVM::AllocaOp::getPromotableSlots() {
  if (!getOperation()->getBlock()->isEntryBlock())
    return {};

  return {MemorySlot{getResult(), getElemType()}};
}

Value LLVM::AllocaOp::getDefaultValue(const MemorySlot &slot,
                                      RewriterBase &rewriter) {
  return rewriter.create<LLVM::UndefOp>(getLoc(), slot.elemType);
}

void LLVM::AllocaOp::handleBlockArgument(const MemorySlot &slot,
                                         BlockArgument argument,
                                         RewriterBase &rewriter) {
  for (Operation *user : getOperation()->getUsers())
    if (auto declareOp = llvm::dyn_cast<LLVM::DbgDeclareOp>(user))
      rewriter.create<LLVM::DbgValueOp>(declareOp.getLoc(), argument,
                                        declareOp.getVarInfo(),
                                        declareOp.getLocationExpr());
}

void LLVM::AllocaOp::handlePromotionComplete(const MemorySlot &slot,
                                             Value defaultValue,
                                             RewriterBase &rewriter) {
  if (defaultValue && defaultValue.use_empty())
    rewriter.eraseOp(defaultValue.getDefiningOp());
  rewriter.eraseOp(*this);
}

SmallVector<DestructurableMemorySlot> LLVM::AllocaOp::getDestructurableSlots() {
  if (!mlir::matchPattern(getArraySize(), m_One()))
    return {};

  auto destructurable = dyn_cast<DestructurableTypeInterface>(getElemType());
  if (!destructurable)
    return {};

  std::optional<DenseMap<Attribute, Type>> destructuredType =
      destructurable.getSubelementIndexMap();
  if (!destructuredType)
    return {};

  return {DestructurableMemorySlot{{getResult(), getElemType()},
                                   *destructuredType}};
}

DenseMap<Attribute, MemorySlot>
LLVM::AllocaOp::destructure(const DestructurableMemorySlot &slot,
                            const SmallPtrSetImpl<Attribute> &usedIndices,
                            RewriterBase &rewriter) {
  assert(slot.ptr == getResult());
  rewriter.setInsertionPointAfter(*this);

  auto destructurableType = cast<DestructurableTypeInterface>(getElemType());
  DenseMap<Attribute, MemorySlot> slotMap;
  for (Attribute index : usedIndices) {
    Type elemType = destructurableType.getTypeAtIndex(index);
    assert(elemType && "used index must exist");
    auto subAlloca = rewriter.create<LLVM::AllocaOp>(
        getLoc(), LLVM::LLVMPointerType::get(getContext()), elemType,
        getArraySize());
    slotMap.try_emplace<MemorySlot>(index, {subAlloca.getResult(), elemType});
  }

  return slotMap;
}

void LLVM::AllocaOp::handleDestructuringComplete(
    const DestructurableMemorySlot &slot, RewriterBase &rewriter) {
  assert(slot.ptr == getResult());
  rewriter.eraseOp(*this);
}

//===----------------------------------------------------------------------===//
// Interfaces for discardable OPs
//===----------------------------------------------------------------------===//

/// Conditions the deletion of the operation to the removal of all its uses.
static bool forwardToUsers(Operation *op,
                           SmallVectorImpl<OpOperand *> &newBlockingUses) {
  for (Value result : op->getResults())
    for (OpOperand &use : result.getUses())
      newBlockingUses.push_back(&use);
  return true;
}

bool LLVM::BitcastOp::canUsesBeRemoved(
    const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  return forwardToUsers(*this, newBlockingUses);
}

DeletionKind LLVM::BitcastOp::removeBlockingUses(
    const SmallPtrSetImpl<OpOperand *> &blockingUses, RewriterBase &rewriter) {
  return DeletionKind::Delete;
}

bool LLVM::LifetimeStartOp::canUsesBeRemoved(
    const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  return true;
}

DeletionKind LLVM::LifetimeStartOp::removeBlockingUses(
    const SmallPtrSetImpl<OpOperand *> &blockingUses, RewriterBase &rewriter) {
  return DeletionKind::Delete;
}

bool LLVM::LifetimeEndOp::canUsesBeRemoved(
    const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  return true;
}

DeletionKind LLVM::LifetimeEndOp::removeBlockingUses(
    const SmallPtrSetImpl<OpOperand *> &blockingUses, RewriterBase &rewriter) {
  return DeletionKind::Delete;
}

bool LLVM::InvariantStartOp::canUsesBeRemoved(
    const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  return true;
}

DeletionKind LLVM::InvariantStartOp::removeBlockingUses(
    const SmallPtrSetImpl<OpOperand *> &blockingUses, RewriterBase &rewriter) {
  return DeletionKind::Delete;
}

bool LLVM::InvariantEndOp::canUsesBeRemoved(
    const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  return true;
}

DeletionKind LLVM::InvariantEndOp::removeBlockingUses(
    const SmallPtrSetImpl<OpOperand *> &blockingUses, RewriterBase &rewriter) {
  return DeletionKind::Delete;
}

bool LLVM::DbgDeclareOp::canUsesBeRemoved(
    const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  return true;
}

DeletionKind LLVM::DbgDeclareOp::removeBlockingUses(
    const SmallPtrSetImpl<OpOperand *> &blockingUses, RewriterBase &rewriter) {
  return DeletionKind::Delete;
}

bool LLVM::DbgDeclareOp::requiresVisitingMutatedDefs() { return true; }

void LLVM::DbgDeclareOp::visitMutatedDefs(
    ArrayRef<std::pair<Operation *, Value>> definitions,
    RewriterBase &rewriter) {
  for (auto [op, value] : definitions) {
    rewriter.setInsertionPointAfter(op);
    rewriter.create<LLVM::DbgValueOp>(getLoc(), value, getVarInfo(),
                                      getLocationExpr());
  }
}

bool LLVM::DbgValueOp::canUsesBeRemoved(
    const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  // There is only one operand that we can remove the use of.
  if (blockingUses.size() != 1)
    return false;

  return (*blockingUses.begin())->get() == getValue();
}

DeletionKind LLVM::DbgValueOp::removeBlockingUses(
    const SmallPtrSetImpl<OpOperand *> &blockingUses, RewriterBase &rewriter) {
  // Rewriter by default is after '*this', but we need it before '*this'.
  rewriter.setInsertionPoint(*this);

  // Rather than dropping the debug value, replace it with undef to preserve the
  // debug local variable info. This allows the debugger to inform the user that
  // the variable has been optimized out.
  auto undef =
      rewriter.create<UndefOp>(getValue().getLoc(), getValue().getType());
  rewriter.modifyOpInPlace(*this, [&] { getValueMutable().assign(undef); });
  return DeletionKind::Keep;
}

//===----------------------------------------------------------------------===//
// Interfaces for GEPOp
//===----------------------------------------------------------------------===//

static bool hasAllZeroIndices(LLVM::GEPOp gepOp) {
  return llvm::all_of(gepOp.getIndices(), [](auto index) {
    auto indexAttr = llvm::dyn_cast_if_present<IntegerAttr>(index);
    return indexAttr && indexAttr.getValue() == 0;
  });
}

bool LLVM::GEPOp::canUsesBeRemoved(
    const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  // GEP can be removed as long as it is a no-op and its users can be removed.
  if (!hasAllZeroIndices(*this))
    return false;
  return forwardToUsers(*this, newBlockingUses);
}

DeletionKind LLVM::GEPOp::removeBlockingUses(
    const SmallPtrSetImpl<OpOperand *> &blockingUses, RewriterBase &rewriter) {
  return DeletionKind::Delete;
}

static bool isFirstIndexZero(LLVM::GEPOp gep) {
  IntegerAttr index =
      llvm::dyn_cast_if_present<IntegerAttr>(gep.getIndices()[0]);
  return index && index.getInt() == 0;
}

LogicalResult LLVM::GEPOp::ensureOnlySafeAccesses(
    const MemorySlot &slot, SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
    const DataLayout &dataLayout) {
  if (getBase() != slot.ptr)
    return success();
  if (slot.elemType != getElemType())
    return failure();
  if (!isFirstIndexZero(*this))
    return failure();
  // Dynamic indices can be out-of-bounds (even negative), so an access with
  // dynamic indices can never be considered safe.
  if (!getDynamicIndices().empty())
    return failure();
  Type reachedType = getResultPtrElementType();
  if (!reachedType)
    return failure();
  mustBeSafelyUsed.emplace_back<MemorySlot>({getResult(), reachedType});
  return success();
}

bool LLVM::GEPOp::canRewire(const DestructurableMemorySlot &slot,
                            SmallPtrSetImpl<Attribute> &usedIndices,
                            SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
                            const DataLayout &dataLayout) {
  if (!isa<LLVM::LLVMPointerType>(getBase().getType()))
    return false;

  if (getBase() != slot.ptr || slot.elemType != getElemType())
    return false;
  if (!isFirstIndexZero(*this))
    return false;
  // Dynamic indices can be out-of-bounds (even negative), so an access with
  // dynamic indices can never be properly rewired.
  if (!getDynamicIndices().empty())
    return false;
  Type reachedType = getResultPtrElementType();
  if (!reachedType || getIndices().size() < 2)
    return false;
  auto firstLevelIndex = dyn_cast<IntegerAttr>(getIndices()[1]);
  if (!firstLevelIndex)
    return false;
  mustBeSafelyUsed.emplace_back<MemorySlot>({getResult(), reachedType});
  assert(slot.elementPtrs.contains(firstLevelIndex));
  usedIndices.insert(firstLevelIndex);
  return true;
}

DeletionKind LLVM::GEPOp::rewire(const DestructurableMemorySlot &slot,
                                 DenseMap<Attribute, MemorySlot> &subslots,
                                 RewriterBase &rewriter,
                                 const DataLayout &dataLayout) {
  IntegerAttr firstLevelIndex =
      llvm::dyn_cast_if_present<IntegerAttr>(getIndices()[1]);
  const MemorySlot &newSlot = subslots.at(firstLevelIndex);

  ArrayRef<int32_t> remainingIndices = getRawConstantIndices().slice(2);

  // If the GEP would become trivial after this transformation, eliminate it.
  // A GEP should only be eliminated if it has no indices (except the first
  // pointer index), as simplifying GEPs with all-zero indices would eliminate
  // structure information useful for further destruction.
  if (remainingIndices.empty()) {
    rewriter.replaceAllUsesWith(getResult(), newSlot.ptr);
    return DeletionKind::Delete;
  }

  rewriter.modifyOpInPlace(*this, [&]() {
    // Rewire the indices by popping off the second index.
    // Start with a single zero, then add the indices beyond the second.
    SmallVector<int32_t> newIndices(1);
    newIndices.append(remainingIndices.begin(), remainingIndices.end());
    setRawConstantIndices(newIndices);

    // Rewire the pointed type.
    setElemType(newSlot.elemType);

    // Rewire the pointer.
    getBaseMutable().assign(newSlot.ptr);
  });

  return DeletionKind::Keep;
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
  return memIntrLen && *memIntrLen <= dataLayout.getTypeSize(slot.elemType);
}

/// Checks whether all indices are i32. This is used to check GEPs can index
/// into them.
static bool areAllIndicesI32(const DestructurableMemorySlot &slot) {
  Type i32 = IntegerType::get(slot.ptr.getContext(), 32);
  return llvm::all_of(llvm::make_first_range(slot.elementPtrs),
                      [&](Attribute index) {
                        auto intIndex = dyn_cast<IntegerAttr>(index);
                        return intIndex && intIndex.getType() == i32;
                      });
}

//===----------------------------------------------------------------------===//
// Interfaces for memset
//===----------------------------------------------------------------------===//

bool LLVM::MemsetOp::loadsFrom(const MemorySlot &slot) { return false; }

bool LLVM::MemsetOp::storesTo(const MemorySlot &slot) {
  return getDst() == slot.ptr;
}

Value LLVM::MemsetOp::getStored(const MemorySlot &slot,
                                RewriterBase &rewriter) {
  // TODO: Support non-integer types.
  return TypeSwitch<Type, Value>(slot.elemType)
      .Case([&](IntegerType intType) -> Value {
        if (intType.getWidth() == 8)
          return getVal();

        assert(intType.getWidth() % 8 == 0);

        // Build the memset integer by repeatedly shifting the value and
        // or-ing it with the previous value.
        uint64_t coveredBits = 8;
        Value currentValue =
            rewriter.create<LLVM::ZExtOp>(getLoc(), intType, getVal());
        while (coveredBits < intType.getWidth()) {
          Value shiftBy =
              rewriter.create<LLVM::ConstantOp>(getLoc(), intType, coveredBits);
          Value shifted =
              rewriter.create<LLVM::ShlOp>(getLoc(), currentValue, shiftBy);
          currentValue =
              rewriter.create<LLVM::OrOp>(getLoc(), currentValue, shifted);
          coveredBits *= 2;
        }

        return currentValue;
      })
      .Default([](Type) -> Value {
        llvm_unreachable(
            "getStored should not be called on memset to unsupported type");
      });
}

bool LLVM::MemsetOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  // TODO: Support non-integer types.
  bool canConvertType =
      TypeSwitch<Type, bool>(slot.elemType)
          .Case([](IntegerType intType) {
            return intType.getWidth() % 8 == 0 && intType.getWidth() > 0;
          })
          .Default([](Type) { return false; });
  if (!canConvertType)
    return false;

  if (getIsVolatile())
    return false;

  return getStaticMemIntrLen(*this) == dataLayout.getTypeSize(slot.elemType);
}

DeletionKind LLVM::MemsetOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    RewriterBase &rewriter, Value reachingDefinition) {
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
  if (&slot.elemType.getDialect() != getOperation()->getDialect())
    return false;

  if (getIsVolatile())
    return false;

  if (!slot.elemType.cast<DestructurableTypeInterface>()
           .getSubelementIndexMap())
    return false;

  if (!areAllIndicesI32(slot))
    return false;

  return definitelyWritesOnlyWithinSlot(*this, slot, dataLayout);
}

DeletionKind LLVM::MemsetOp::rewire(const DestructurableMemorySlot &slot,
                                    DenseMap<Attribute, MemorySlot> &subslots,
                                    RewriterBase &rewriter,
                                    const DataLayout &dataLayout) {
  std::optional<DenseMap<Attribute, Type>> types =
      slot.elemType.cast<DestructurableTypeInterface>().getSubelementIndexMap();

  IntegerAttr memsetLenAttr;
  bool successfulMatch =
      matchPattern(getLen(), m_Constant<IntegerAttr>(&memsetLenAttr));
  (void)successfulMatch;
  assert(successfulMatch);

  bool packed = false;
  if (auto structType = dyn_cast<LLVM::LLVMStructType>(slot.elemType))
    packed = structType.isPacked();

  Type i32 = IntegerType::get(getContext(), 32);
  uint64_t memsetLen = memsetLenAttr.getValue().getZExtValue();
  uint64_t covered = 0;
  for (size_t i = 0; i < types->size(); i++) {
    // Create indices on the fly to get elements in the right order.
    Attribute index = IntegerAttr::get(i32, i);
    Type elemType = types->at(index);
    uint64_t typeSize = dataLayout.getTypeSize(elemType);

    if (!packed)
      covered =
          llvm::alignTo(covered, dataLayout.getTypeABIAlignment(elemType));

    if (covered >= memsetLen)
      break;

    // If this subslot is used, apply a new memset to it.
    // Otherwise, only compute its offset within the original memset.
    if (subslots.contains(index)) {
      uint64_t newMemsetSize = std::min(memsetLen - covered, typeSize);

      Value newMemsetSizeValue =
          rewriter
              .create<LLVM::ConstantOp>(
                  getLen().getLoc(),
                  IntegerAttr::get(memsetLenAttr.getType(), newMemsetSize))
              .getResult();

      rewriter.create<LLVM::MemsetOp>(getLoc(), subslots.at(index).ptr,
                                      getVal(), newMemsetSizeValue,
                                      getIsVolatile());
    }

    covered += typeSize;
  }

  return DeletionKind::Delete;
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
                             RewriterBase &rewriter) {
  return rewriter.create<LLVM::LoadOp>(op.getLoc(), slot.elemType, op.getSrc());
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

  return getStaticMemIntrLen(op) == dataLayout.getTypeSize(slot.elemType);
}

template <class MemcpyLike>
static DeletionKind
memcpyRemoveBlockingUses(MemcpyLike op, const MemorySlot &slot,
                         const SmallPtrSetImpl<OpOperand *> &blockingUses,
                         RewriterBase &rewriter, Value reachingDefinition) {
  if (op.loadsFrom(slot))
    rewriter.create<LLVM::StoreOp>(op.getLoc(), reachingDefinition,
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

  if (!slot.elemType.cast<DestructurableTypeInterface>()
           .getSubelementIndexMap())
    return false;

  if (!areAllIndicesI32(slot))
    return false;

  // Only full copies are supported.
  if (getStaticMemIntrLen(op) != dataLayout.getTypeSize(slot.elemType))
    return false;

  if (op.getSrc() == slot.ptr)
    for (Attribute index : llvm::make_first_range(slot.elementPtrs))
      usedIndices.insert(index);

  return true;
}

namespace {

template <class MemcpyLike>
void createMemcpyLikeToReplace(RewriterBase &rewriter, const DataLayout &layout,
                               MemcpyLike toReplace, Value dst, Value src,
                               Type toCpy, bool isVolatile) {
  Value memcpySize = rewriter.create<LLVM::ConstantOp>(
      toReplace.getLoc(), IntegerAttr::get(toReplace.getLen().getType(),
                                           layout.getTypeSize(toCpy)));
  rewriter.create<MemcpyLike>(toReplace.getLoc(), dst, src, memcpySize,
                              isVolatile);
}

template <>
void createMemcpyLikeToReplace(RewriterBase &rewriter, const DataLayout &layout,
                               LLVM::MemcpyInlineOp toReplace, Value dst,
                               Value src, Type toCpy, bool isVolatile) {
  Type lenType = IntegerType::get(toReplace->getContext(),
                                  toReplace.getLen().getBitWidth());
  rewriter.create<LLVM::MemcpyInlineOp>(
      toReplace.getLoc(), dst, src,
      IntegerAttr::get(lenType, layout.getTypeSize(toCpy)), isVolatile);
}

} // namespace

/// Rewires a memcpy-like operation. Only copies to or from the full slot are
/// supported.
template <class MemcpyLike>
static DeletionKind
memcpyRewire(MemcpyLike op, const DestructurableMemorySlot &slot,
             DenseMap<Attribute, MemorySlot> &subslots, RewriterBase &rewriter,
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
  for (size_t i = 0, e = slot.elementPtrs.size(); i != e; i++) {
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
    Value subslotPtrInOther = rewriter.create<LLVM::GEPOp>(
        op.getLoc(), LLVM::LLVMPointerType::get(op.getContext()), slot.elemType,
        isDst ? op.getSrc() : op.getDst(), gepIndices);

    // Then create a new memcpy out of this source pointer.
    createMemcpyLikeToReplace(rewriter, dataLayout, op,
                              isDst ? subslot.ptr : subslotPtrInOther,
                              isDst ? subslotPtrInOther : subslot.ptr,
                              subslot.elemType, op.getIsVolatile());
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

Value LLVM::MemcpyOp::getStored(const MemorySlot &slot,
                                RewriterBase &rewriter) {
  return memcpyGetStored(*this, slot, rewriter);
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
    RewriterBase &rewriter, Value reachingDefinition) {
  return memcpyRemoveBlockingUses(*this, slot, blockingUses, rewriter,
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
                                    RewriterBase &rewriter,
                                    const DataLayout &dataLayout) {
  return memcpyRewire(*this, slot, subslots, rewriter, dataLayout);
}

bool LLVM::MemcpyInlineOp::loadsFrom(const MemorySlot &slot) {
  return memcpyLoadsFrom(*this, slot);
}

bool LLVM::MemcpyInlineOp::storesTo(const MemorySlot &slot) {
  return memcpyStoresTo(*this, slot);
}

Value LLVM::MemcpyInlineOp::getStored(const MemorySlot &slot,
                                      RewriterBase &rewriter) {
  return memcpyGetStored(*this, slot, rewriter);
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
    RewriterBase &rewriter, Value reachingDefinition) {
  return memcpyRemoveBlockingUses(*this, slot, blockingUses, rewriter,
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
                             RewriterBase &rewriter,
                             const DataLayout &dataLayout) {
  return memcpyRewire(*this, slot, subslots, rewriter, dataLayout);
}

bool LLVM::MemmoveOp::loadsFrom(const MemorySlot &slot) {
  return memcpyLoadsFrom(*this, slot);
}

bool LLVM::MemmoveOp::storesTo(const MemorySlot &slot) {
  return memcpyStoresTo(*this, slot);
}

Value LLVM::MemmoveOp::getStored(const MemorySlot &slot,
                                 RewriterBase &rewriter) {
  return memcpyGetStored(*this, slot, rewriter);
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
    RewriterBase &rewriter, Value reachingDefinition) {
  return memcpyRemoveBlockingUses(*this, slot, blockingUses, rewriter,
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
                                     RewriterBase &rewriter,
                                     const DataLayout &dataLayout) {
  return memcpyRewire(*this, slot, subslots, rewriter, dataLayout);
}

//===----------------------------------------------------------------------===//
// Interfaces for destructurable types
//===----------------------------------------------------------------------===//

std::optional<DenseMap<Attribute, Type>>
LLVM::LLVMStructType::getSubelementIndexMap() {
  Type i32 = IntegerType::get(getContext(), 32);
  DenseMap<Attribute, Type> destructured;
  for (const auto &[index, elemType] : llvm::enumerate(getBody()))
    destructured.insert({IntegerAttr::get(i32, index), elemType});
  return destructured;
}

Type LLVM::LLVMStructType::getTypeAtIndex(Attribute index) {
  auto indexAttr = llvm::dyn_cast<IntegerAttr>(index);
  if (!indexAttr || !indexAttr.getType().isInteger(32))
    return {};
  int32_t indexInt = indexAttr.getInt();
  ArrayRef<Type> body = getBody();
  if (indexInt < 0 || body.size() <= static_cast<uint32_t>(indexInt))
    return {};
  return body[indexInt];
}

std::optional<DenseMap<Attribute, Type>>
LLVM::LLVMArrayType::getSubelementIndexMap() const {
  constexpr size_t maxArraySizeForDestructuring = 16;
  if (getNumElements() > maxArraySizeForDestructuring)
    return {};
  int32_t numElements = getNumElements();

  Type i32 = IntegerType::get(getContext(), 32);
  DenseMap<Attribute, Type> destructured;
  for (int32_t index = 0; index < numElements; ++index)
    destructured.insert({IntegerAttr::get(i32, index), getElementType()});
  return destructured;
}

Type LLVM::LLVMArrayType::getTypeAtIndex(Attribute index) const {
  auto indexAttr = llvm::dyn_cast<IntegerAttr>(index);
  if (!indexAttr || !indexAttr.getType().isInteger(32))
    return {};
  int32_t indexInt = indexAttr.getInt();
  if (indexInt < 0 || getNumElements() <= static_cast<uint32_t>(indexInt))
    return {};
  return getElementType();
}
