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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Interfaces for AllocaOp
//===----------------------------------------------------------------------===//

static Type getAllocaElementType(LLVM::AllocaOp alloca) {
  // This will become trivial once non-opaque pointers are gone.
  return alloca.getElemType().has_value()
             ? *alloca.getElemType()
             : alloca.getResult().getType().getElementType();
}

llvm::SmallVector<MemorySlot> LLVM::AllocaOp::getPromotableSlots() {
  if (!getOperation()->getBlock()->isEntryBlock())
    return {};

  return {MemorySlot{getResult(), getAllocaElementType(*this)}};
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
                                        declareOp.getVarInfo());
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

  Type elemType = getAllocaElementType(*this);
  auto destructurable = dyn_cast<DestructurableTypeInterface>(elemType);
  if (!destructurable)
    return {};

  std::optional<DenseMap<Attribute, Type>> destructuredType =
      destructurable.getSubelementIndexMap();
  if (!destructuredType)
    return {};

  DenseMap<Attribute, Type> allocaTypeMap;
  for (Attribute index : llvm::make_first_range(destructuredType.value()))
    allocaTypeMap.insert({index, LLVM::LLVMPointerType::get(getContext())});

  return {DestructurableMemorySlot{{getResult(), elemType}, {allocaTypeMap}}};
}

DenseMap<Attribute, MemorySlot>
LLVM::AllocaOp::destructure(const DestructurableMemorySlot &slot,
                            const SmallPtrSetImpl<Attribute> &usedIndices,
                            RewriterBase &rewriter) {
  assert(slot.ptr == getResult());
  Type elemType =
      getElemType() ? *getElemType() : getResult().getType().getElementType();

  rewriter.setInsertionPointAfter(*this);

  auto destructurableType = cast<DestructurableTypeInterface>(elemType);
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
// Interfaces for LoadOp/StoreOp
//===----------------------------------------------------------------------===//

bool LLVM::LoadOp::loadsFrom(const MemorySlot &slot) {
  return getAddr() == slot.ptr;
}

Value LLVM::LoadOp::getStored(const MemorySlot &slot) { return {}; }

bool LLVM::StoreOp::loadsFrom(const MemorySlot &slot) { return false; }

Value LLVM::StoreOp::getStored(const MemorySlot &slot) {
  return getAddr() == slot.ptr ? getValue() : Value();
}

bool LLVM::LoadOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses) {
  if (blockingUses.size() != 1)
    return false;
  Value blockingUse = (*blockingUses.begin())->get();
  // If the blocking use is the slot ptr itself, there will be enough
  // context to reconstruct the result of the load at removal time, so it can
  // be removed (provided it loads the exact stored value and is not
  // volatile).
  return blockingUse == slot.ptr && getAddr() == slot.ptr &&
         getResult().getType() == slot.elemType && !getVolatile_();
}

DeletionKind LLVM::LoadOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    RewriterBase &rewriter, Value reachingDefinition) {
  // `canUsesBeRemoved` checked this blocking use must be the loaded slot
  // pointer.
  rewriter.replaceAllUsesWith(getResult(), reachingDefinition);
  return DeletionKind::Delete;
}

bool LLVM::StoreOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses) {
  if (blockingUses.size() != 1)
    return false;
  Value blockingUse = (*blockingUses.begin())->get();
  // If the blocking use is the slot ptr itself, dropping the store is
  // fine, provided we are currently promoting its target value. Don't allow a
  // store OF the slot pointer, only INTO the slot pointer.
  return blockingUse == slot.ptr && getAddr() == slot.ptr &&
         getValue() != slot.ptr && getValue().getType() == slot.elemType &&
         !getVolatile_();
}

DeletionKind LLVM::StoreOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    RewriterBase &rewriter, Value reachingDefinition) {
  // `canUsesBeRemoved` checked this blocking use must be the stored slot
  // pointer.
  for (Operation *user : slot.ptr.getUsers())
    if (auto declareOp = dyn_cast<LLVM::DbgDeclareOp>(user))
      rewriter.create<LLVM::DbgValueOp>(declareOp->getLoc(), getValue(),
                                        declareOp.getVarInfo());
  return DeletionKind::Delete;
}

LogicalResult LLVM::LoadOp::ensureOnlySafeAccesses(
    const MemorySlot &slot, SmallVectorImpl<MemorySlot> &mustBeSafelyUsed) {
  return success(getAddr() != slot.ptr || getType() == slot.elemType);
}

LogicalResult LLVM::StoreOp::ensureOnlySafeAccesses(
    const MemorySlot &slot, SmallVectorImpl<MemorySlot> &mustBeSafelyUsed) {
  return success(getAddr() != slot.ptr ||
                 getValue().getType() == slot.elemType);
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
    SmallVectorImpl<OpOperand *> &newBlockingUses) {
  return forwardToUsers(*this, newBlockingUses);
}

DeletionKind LLVM::BitcastOp::removeBlockingUses(
    const SmallPtrSetImpl<OpOperand *> &blockingUses, RewriterBase &rewriter) {
  return DeletionKind::Delete;
}

bool LLVM::AddrSpaceCastOp::canUsesBeRemoved(
    const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses) {
  return forwardToUsers(*this, newBlockingUses);
}

DeletionKind LLVM::AddrSpaceCastOp::removeBlockingUses(
    const SmallPtrSetImpl<OpOperand *> &blockingUses, RewriterBase &rewriter) {
  return DeletionKind::Delete;
}

bool LLVM::LifetimeStartOp::canUsesBeRemoved(
    const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses) {
  return true;
}

DeletionKind LLVM::LifetimeStartOp::removeBlockingUses(
    const SmallPtrSetImpl<OpOperand *> &blockingUses, RewriterBase &rewriter) {
  return DeletionKind::Delete;
}

bool LLVM::LifetimeEndOp::canUsesBeRemoved(
    const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses) {
  return true;
}

DeletionKind LLVM::LifetimeEndOp::removeBlockingUses(
    const SmallPtrSetImpl<OpOperand *> &blockingUses, RewriterBase &rewriter) {
  return DeletionKind::Delete;
}

bool LLVM::DbgDeclareOp::canUsesBeRemoved(
    const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses) {
  return true;
}

DeletionKind LLVM::DbgDeclareOp::removeBlockingUses(
    const SmallPtrSetImpl<OpOperand *> &blockingUses, RewriterBase &rewriter) {
  return DeletionKind::Delete;
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
    SmallVectorImpl<OpOperand *> &newBlockingUses) {
  // GEP can be removed as long as it is a no-op and its users can be removed.
  if (!hasAllZeroIndices(*this))
    return false;
  return forwardToUsers(*this, newBlockingUses);
}

DeletionKind LLVM::GEPOp::removeBlockingUses(
    const SmallPtrSetImpl<OpOperand *> &blockingUses, RewriterBase &rewriter) {
  return DeletionKind::Delete;
}

/// Returns the type the resulting pointer of the GEP points to. If such a type
/// is not clear, returns null type.
static Type computeReachedGEPType(LLVM::GEPOp gep) {
  if (gep.getIndices().empty())
    return {};

  // Ensures all indices are static and fetches them.
  SmallVector<IntegerAttr> indices;
  for (auto index : gep.getIndices()) {
    IntegerAttr indexInt = llvm::dyn_cast_if_present<IntegerAttr>(index);
    if (!indexInt)
      return {};
    indices.push_back(indexInt);
  }

  // Check the pointer indexing only targets the first element.
  if (indices[0].getInt() != 0)
    return {};

  // Set the initial type currently being used for indexing. This will be
  // updated as the indices get walked over.
  std::optional<Type> maybeSelectedType = gep.getElemType();
  if (!maybeSelectedType)
    return {};
  Type selectedType = *maybeSelectedType;

  // Follow the indexed elements in the gep.
  for (IntegerAttr index : llvm::drop_begin(indices)) {
    // Ensure the structure of the type being indexed can be reasoned about.
    // This includes rejecting any potential typed pointer.
    auto destructurable = llvm::dyn_cast<DestructurableTypeInterface>(selectedType);
    if (!destructurable)
      return {};

    // Follow the type at the index the gep is accessing, making it the new type
    // used for indexing.
    Type field = destructurable.getTypeAtIndex(index);
    if (!field)
      return {};
    selectedType = field;
  }

  // When there are no more indices, the type currently being used for indexing
  // is the type of the value pointed at by the returned indexed pointer.
  return selectedType;
}

LogicalResult LLVM::GEPOp::ensureOnlySafeAccesses(
    const MemorySlot &slot, SmallVectorImpl<MemorySlot> &mustBeSafelyUsed) {
  if (getBase() != slot.ptr)
    return success();
  if (slot.elemType != getElemType())
    return failure();
  Type reachedType = computeReachedGEPType(*this);
  if (!reachedType)
    return failure();
  mustBeSafelyUsed.emplace_back<MemorySlot>({getResult(), reachedType});
  return success();
}

bool LLVM::GEPOp::canRewire(const DestructurableMemorySlot &slot,
                            SmallPtrSetImpl<Attribute> &usedIndices,
                            SmallVectorImpl<MemorySlot> &mustBeSafelyUsed) {
  auto basePtrType = llvm::dyn_cast<LLVM::LLVMPointerType>(getBase().getType());
  if (!basePtrType)
    return false;

  // Typed pointers are not supported. This should be removed once typed
  // pointers are removed from the LLVM dialect.
  if (!basePtrType.isOpaque())
    return false;

  if (getBase() != slot.ptr || slot.elemType != getElemType())
    return false;
  Type reachedType = computeReachedGEPType(*this);
  if (!reachedType || getIndices().size() < 2)
    return false;
  auto firstLevelIndex = cast<IntegerAttr>(getIndices()[1]);
  assert(slot.elementPtrs.contains(firstLevelIndex));
  if (!llvm::isa<LLVM::LLVMPointerType>(slot.elementPtrs.at(firstLevelIndex)))
    return false;
  mustBeSafelyUsed.emplace_back<MemorySlot>({getResult(), reachedType});
  usedIndices.insert(firstLevelIndex);
  return true;
}

DeletionKind LLVM::GEPOp::rewire(const DestructurableMemorySlot &slot,
                                 DenseMap<Attribute, MemorySlot> &subslots,
                                 RewriterBase &rewriter) {
  IntegerAttr firstLevelIndex = llvm::dyn_cast_if_present<IntegerAttr>(getIndices()[1]);
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

  rewriter.updateRootInPlace(*this, [&]() {
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
