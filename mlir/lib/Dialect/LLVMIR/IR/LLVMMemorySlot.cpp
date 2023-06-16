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
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"

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

bool LLVM::LoadOp::storesTo(const MemorySlot &slot) { return false; }

Value LLVM::LoadOp::getStored(const MemorySlot &slot, RewriterBase &rewriter) {
  llvm_unreachable("getStored should not be called on LoadOp");
}

bool LLVM::StoreOp::loadsFrom(const MemorySlot &slot) { return false; }

bool LLVM::StoreOp::storesTo(const MemorySlot &slot) {
  return getAddr() == slot.ptr;
}

Value LLVM::StoreOp::getStored(const MemorySlot &slot, RewriterBase &rewriter) {
  return getValue();
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
    auto destructurable =
        llvm::dyn_cast<DestructurableTypeInterface>(selectedType);
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
// Interfaces for memset
//===----------------------------------------------------------------------===//

/// Returns the length of the given memset in bytes if it can be known at
/// compile-time on a best-effort basis, nothing otherwise.
static std::optional<uint64_t> getStaticMemsetLen(LLVM::MemsetOp op) {
  IntegerAttr::ValueType memsetLen;
  if (!matchPattern(op.getLen(), m_ConstantInt(&memsetLen)))
    return {};

  if (memsetLen.getBitWidth() > 64)
    return {};

  return memsetLen.getZExtValue();
}

/// Returns whether one can be sure the memset does not write outside of the
/// bounds of the given slot, on a best-effort basis.
static bool definitelyWritesOnlyWithinSlot(LLVM::MemsetOp op,
                                           const MemorySlot &slot,
                                           DataLayout &dataLayout) {
  if (!isa<LLVM::LLVMPointerType>(slot.ptr.getType()) ||
      op.getDst() != slot.ptr)
    return false;

  std::optional<uint64_t> memsetLen = getStaticMemsetLen(op);
  return memsetLen && *memsetLen <= dataLayout.getTypeSize(slot.elemType);
}

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

        // Build the memset integer by repeatedly shifting the value and or-ing
        // it with the previous value.
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
    SmallVectorImpl<OpOperand *> &newBlockingUses) {
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

  DataLayout layout = DataLayout::closest(*this);
  return getStaticMemsetLen(*this) == layout.getTypeSize(slot.elemType);
}

DeletionKind LLVM::MemsetOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    RewriterBase &rewriter, Value reachingDefinition) {
  return DeletionKind::Delete;
}

LogicalResult LLVM::MemsetOp::ensureOnlySafeAccesses(
    const MemorySlot &slot, SmallVectorImpl<MemorySlot> &mustBeSafelyUsed) {
  DataLayout dataLayout = DataLayout::closest(*this);
  return success(definitelyWritesOnlyWithinSlot(*this, slot, dataLayout));
}

bool LLVM::MemsetOp::canRewire(const DestructurableMemorySlot &slot,
                               SmallPtrSetImpl<Attribute> &usedIndices,
                               SmallVectorImpl<MemorySlot> &mustBeSafelyUsed) {
  if (&slot.elemType.getDialect() != getOperation()->getDialect())
    return false;

  if (getIsVolatile())
    return false;

  if (!slot.elemType.cast<DestructurableTypeInterface>()
           .getSubelementIndexMap())
    return false;

  DataLayout dataLayout = DataLayout::closest(*this);
  return definitelyWritesOnlyWithinSlot(*this, slot, dataLayout);
}

DeletionKind LLVM::MemsetOp::rewire(const DestructurableMemorySlot &slot,
                                    DenseMap<Attribute, MemorySlot> &subslots,
                                    RewriterBase &rewriter) {
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
  DataLayout dataLayout = DataLayout::closest(*this);
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
