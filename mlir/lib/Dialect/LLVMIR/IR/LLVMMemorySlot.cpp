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

#define DEBUG_TYPE "sroa"

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
                                      OpBuilder &builder) {
  return builder.create<LLVM::UndefOp>(getLoc(), slot.elemType);
}

void LLVM::AllocaOp::handleBlockArgument(const MemorySlot &slot,
                                         BlockArgument argument,
                                         OpBuilder &builder) {
  for (Operation *user : getOperation()->getUsers())
    if (auto declareOp = llvm::dyn_cast<LLVM::DbgDeclareOp>(user))
      builder.create<LLVM::DbgValueOp>(declareOp.getLoc(), argument,
                                       declareOp.getVarInfo(),
                                       declareOp.getLocationExpr());
}

std::optional<PromotableAllocationOpInterface>
LLVM::AllocaOp::handlePromotionComplete(const MemorySlot &slot,
                                        Value defaultValue,
                                        OpBuilder &builder) {
  if (defaultValue && defaultValue.use_empty())
    defaultValue.getDefiningOp()->erase();
  this->erase();
  return std::nullopt;
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

DenseMap<Attribute, MemorySlot> LLVM::AllocaOp::destructure(
    const DestructurableMemorySlot &slot,
    const SmallPtrSetImpl<Attribute> &usedIndices, OpBuilder &builder,
    SmallVectorImpl<DestructurableAllocationOpInterface> &newAllocators) {
  assert(slot.ptr == getResult());
  builder.setInsertionPointAfter(*this);

  auto destructurableType = cast<DestructurableTypeInterface>(getElemType());
  DenseMap<Attribute, MemorySlot> slotMap;
  for (Attribute index : usedIndices) {
    Type elemType = destructurableType.getTypeAtIndex(index);
    assert(elemType && "used index must exist");
    auto subAlloca = builder.create<LLVM::AllocaOp>(
        getLoc(), LLVM::LLVMPointerType::get(getContext()), elemType,
        getArraySize());
    newAllocators.push_back(subAlloca);
    slotMap.try_emplace<MemorySlot>(index, {subAlloca.getResult(), elemType});
  }

  return slotMap;
}

std::optional<DestructurableAllocationOpInterface>
LLVM::AllocaOp::handleDestructuringComplete(
    const DestructurableMemorySlot &slot, OpBuilder &builder) {
  assert(slot.ptr == getResult());
  this->erase();
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Interfaces for LoadOp/StoreOp
//===----------------------------------------------------------------------===//

bool LLVM::LoadOp::loadsFrom(const MemorySlot &slot) {
  return getAddr() == slot.ptr;
}

bool LLVM::LoadOp::storesTo(const MemorySlot &slot) { return false; }

Value LLVM::LoadOp::getStored(const MemorySlot &slot, OpBuilder &builder,
                              Value reachingDef, const DataLayout &dataLayout) {
  llvm_unreachable("getStored should not be called on LoadOp");
}

bool LLVM::StoreOp::loadsFrom(const MemorySlot &slot) { return false; }

bool LLVM::StoreOp::storesTo(const MemorySlot &slot) {
  return getAddr() == slot.ptr;
}

/// Checks if `type` can be used in any kind of conversion sequences.
static bool isSupportedTypeForConversion(Type type) {
  // Aggregate types are not bitcastable.
  if (isa<LLVM::LLVMStructType, LLVM::LLVMArrayType>(type))
    return false;

  // LLVM vector types are only used for either pointers or target specific
  // types. These types cannot be casted in the general case, thus the memory
  // optimizations do not support them.
  if (isa<LLVM::LLVMFixedVectorType, LLVM::LLVMScalableVectorType>(type))
    return false;

  // Scalable types are not supported.
  if (auto vectorType = dyn_cast<VectorType>(type))
    return !vectorType.isScalable();
  return true;
}

/// Checks that `rhs` can be converted to `lhs` by a sequence of casts and
/// truncations. Checks for narrowing or widening conversion compatibility
/// depending on `narrowingConversion`.
static bool areConversionCompatible(const DataLayout &layout, Type targetType,
                                    Type srcType, bool narrowingConversion) {
  if (targetType == srcType)
    return true;

  if (!isSupportedTypeForConversion(targetType) ||
      !isSupportedTypeForConversion(srcType))
    return false;

  uint64_t targetSize = layout.getTypeSize(targetType);
  uint64_t srcSize = layout.getTypeSize(srcType);

  // Pointer casts will only be sane when the bitsize of both pointer types is
  // the same.
  if (isa<LLVM::LLVMPointerType>(targetType) &&
      isa<LLVM::LLVMPointerType>(srcType))
    return targetSize == srcSize;

  if (narrowingConversion)
    return targetSize <= srcSize;
  return targetSize >= srcSize;
}

/// Checks if `dataLayout` describes a little endian layout.
static bool isBigEndian(const DataLayout &dataLayout) {
  auto endiannessStr = dyn_cast_or_null<StringAttr>(dataLayout.getEndianness());
  return endiannessStr && endiannessStr == "big";
}

/// Converts a value to an integer type of the same size.
/// Assumes that the type can be converted.
static Value castToSameSizedInt(OpBuilder &builder, Location loc, Value val,
                                const DataLayout &dataLayout) {
  Type type = val.getType();
  assert(isSupportedTypeForConversion(type) &&
         "expected value to have a convertible type");

  if (isa<IntegerType>(type))
    return val;

  uint64_t typeBitSize = dataLayout.getTypeSizeInBits(type);
  IntegerType valueSizeInteger = builder.getIntegerType(typeBitSize);

  if (isa<LLVM::LLVMPointerType>(type))
    return builder.createOrFold<LLVM::PtrToIntOp>(loc, valueSizeInteger, val);
  return builder.createOrFold<LLVM::BitcastOp>(loc, valueSizeInteger, val);
}

/// Converts a value with an integer type to `targetType`.
static Value castIntValueToSameSizedType(OpBuilder &builder, Location loc,
                                         Value val, Type targetType) {
  assert(isa<IntegerType>(val.getType()) &&
         "expected value to have an integer type");
  assert(isSupportedTypeForConversion(targetType) &&
         "expected the target type to be supported for conversions");
  if (val.getType() == targetType)
    return val;
  if (isa<LLVM::LLVMPointerType>(targetType))
    return builder.createOrFold<LLVM::IntToPtrOp>(loc, targetType, val);
  return builder.createOrFold<LLVM::BitcastOp>(loc, targetType, val);
}

/// Constructs operations that convert `srcValue` into a new value of type
/// `targetType`. Assumes the types have the same bitsize.
static Value castSameSizedTypes(OpBuilder &builder, Location loc,
                                Value srcValue, Type targetType,
                                const DataLayout &dataLayout) {
  Type srcType = srcValue.getType();
  assert(areConversionCompatible(dataLayout, targetType, srcType,
                                 /*narrowingConversion=*/true) &&
         "expected that the compatibility was checked before");

  // Nothing has to be done if the types are already the same.
  if (srcType == targetType)
    return srcValue;

  // In the special case of casting one pointer to another, we want to generate
  // an address space cast. Bitcasts of pointers are not allowed and using
  // pointer to integer conversions are not equivalent due to the loss of
  // provenance.
  if (isa<LLVM::LLVMPointerType>(targetType) &&
      isa<LLVM::LLVMPointerType>(srcType))
    return builder.createOrFold<LLVM::AddrSpaceCastOp>(loc, targetType,
                                                       srcValue);

  // For all other castable types, casting through integers is necessary.
  Value replacement = castToSameSizedInt(builder, loc, srcValue, dataLayout);
  return castIntValueToSameSizedType(builder, loc, replacement, targetType);
}

/// Constructs operations that convert `srcValue` into a new value of type
/// `targetType`. Performs bit-level extraction if the source type is larger
/// than the target type. Assumes that this conversion is possible.
static Value createExtractAndCast(OpBuilder &builder, Location loc,
                                  Value srcValue, Type targetType,
                                  const DataLayout &dataLayout) {
  // Get the types of the source and target values.
  Type srcType = srcValue.getType();
  assert(areConversionCompatible(dataLayout, targetType, srcType,
                                 /*narrowingConversion=*/true) &&
         "expected that the compatibility was checked before");

  uint64_t srcTypeSize = dataLayout.getTypeSizeInBits(srcType);
  uint64_t targetTypeSize = dataLayout.getTypeSizeInBits(targetType);
  if (srcTypeSize == targetTypeSize)
    return castSameSizedTypes(builder, loc, srcValue, targetType, dataLayout);

  // First, cast the value to a same-sized integer type.
  Value replacement = castToSameSizedInt(builder, loc, srcValue, dataLayout);

  // Truncate the integer if the size of the target is less than the value.
  if (isBigEndian(dataLayout)) {
    uint64_t shiftAmount = srcTypeSize - targetTypeSize;
    auto shiftConstant = builder.create<LLVM::ConstantOp>(
        loc, builder.getIntegerAttr(srcType, shiftAmount));
    replacement =
        builder.createOrFold<LLVM::LShrOp>(loc, srcValue, shiftConstant);
  }

  replacement = builder.create<LLVM::TruncOp>(
      loc, builder.getIntegerType(targetTypeSize), replacement);

  // Now cast the integer to the actual target type if required.
  return castIntValueToSameSizedType(builder, loc, replacement, targetType);
}

/// Constructs operations that insert the bits of `srcValue` into the
/// "beginning" of `reachingDef` (beginning is endianness dependent).
/// Assumes that this conversion is possible.
static Value createInsertAndCast(OpBuilder &builder, Location loc,
                                 Value srcValue, Value reachingDef,
                                 const DataLayout &dataLayout) {

  assert(areConversionCompatible(dataLayout, reachingDef.getType(),
                                 srcValue.getType(),
                                 /*narrowingConversion=*/false) &&
         "expected that the compatibility was checked before");
  uint64_t valueTypeSize = dataLayout.getTypeSizeInBits(srcValue.getType());
  uint64_t slotTypeSize = dataLayout.getTypeSizeInBits(reachingDef.getType());
  if (slotTypeSize == valueTypeSize)
    return castSameSizedTypes(builder, loc, srcValue, reachingDef.getType(),
                              dataLayout);

  // In the case where the store only overwrites parts of the memory,
  // bit fiddling is required to construct the new value.

  // First convert both values to integers of the same size.
  Value defAsInt = castToSameSizedInt(builder, loc, reachingDef, dataLayout);
  Value valueAsInt = castToSameSizedInt(builder, loc, srcValue, dataLayout);
  // Extend the value to the size of the reaching definition.
  valueAsInt =
      builder.createOrFold<LLVM::ZExtOp>(loc, defAsInt.getType(), valueAsInt);
  uint64_t sizeDifference = slotTypeSize - valueTypeSize;
  if (isBigEndian(dataLayout)) {
    // On big endian systems, a store to the base pointer overwrites the most
    // significant bits. To accomodate for this, the stored value needs to be
    // shifted into the according position.
    Value bigEndianShift = builder.create<LLVM::ConstantOp>(
        loc, builder.getIntegerAttr(defAsInt.getType(), sizeDifference));
    valueAsInt =
        builder.createOrFold<LLVM::ShlOp>(loc, valueAsInt, bigEndianShift);
  }

  // Construct the mask that is used to erase the bits that are overwritten by
  // the store.
  APInt maskValue;
  if (isBigEndian(dataLayout)) {
    // Build a mask that has the most significant bits set to zero.
    // Note: This is the same as 2^sizeDifference - 1
    maskValue = APInt::getAllOnes(sizeDifference).zext(slotTypeSize);
  } else {
    // Build a mask that has the least significant bits set to zero.
    // Note: This is the same as -(2^valueTypeSize)
    maskValue = APInt::getAllOnes(valueTypeSize).zext(slotTypeSize);
    maskValue.flipAllBits();
  }

  // Mask out the affected bits ...
  Value mask = builder.create<LLVM::ConstantOp>(
      loc, builder.getIntegerAttr(defAsInt.getType(), maskValue));
  Value masked = builder.createOrFold<LLVM::AndOp>(loc, defAsInt, mask);

  // ... and combine the result with the new value.
  Value combined = builder.createOrFold<LLVM::OrOp>(loc, masked, valueAsInt);

  return castIntValueToSameSizedType(builder, loc, combined,
                                     reachingDef.getType());
}

Value LLVM::StoreOp::getStored(const MemorySlot &slot, OpBuilder &builder,
                               Value reachingDef,
                               const DataLayout &dataLayout) {
  assert(reachingDef && reachingDef.getType() == slot.elemType &&
         "expected the reaching definition's type to match the slot's type");
  return createInsertAndCast(builder, getLoc(), getValue(), reachingDef,
                             dataLayout);
}

bool LLVM::LoadOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  if (blockingUses.size() != 1)
    return false;
  Value blockingUse = (*blockingUses.begin())->get();
  // If the blocking use is the slot ptr itself, there will be enough
  // context to reconstruct the result of the load at removal time, so it can
  // be removed (provided it is not volatile).
  return blockingUse == slot.ptr && getAddr() == slot.ptr &&
         areConversionCompatible(dataLayout, getResult().getType(),
                                 slot.elemType, /*narrowingConversion=*/true) &&
         !getVolatile_();
}

DeletionKind LLVM::LoadOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    OpBuilder &builder, Value reachingDefinition,
    const DataLayout &dataLayout) {
  // `canUsesBeRemoved` checked this blocking use must be the loaded slot
  // pointer.
  Value newResult = createExtractAndCast(builder, getLoc(), reachingDefinition,
                                         getResult().getType(), dataLayout);
  getResult().replaceAllUsesWith(newResult);
  return DeletionKind::Delete;
}

bool LLVM::StoreOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  if (blockingUses.size() != 1)
    return false;
  Value blockingUse = (*blockingUses.begin())->get();
  // If the blocking use is the slot ptr itself, dropping the store is
  // fine, provided we are currently promoting its target value. Don't allow a
  // store OF the slot pointer, only INTO the slot pointer.
  return blockingUse == slot.ptr && getAddr() == slot.ptr &&
         getValue() != slot.ptr &&
         areConversionCompatible(dataLayout, slot.elemType,
                                 getValue().getType(),
                                 /*narrowingConversion=*/false) &&
         !getVolatile_();
}

DeletionKind LLVM::StoreOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    OpBuilder &builder, Value reachingDefinition,
    const DataLayout &dataLayout) {
  return DeletionKind::Delete;
}

/// Checks if `slot` can be accessed through the provided access type.
static bool isValidAccessType(const MemorySlot &slot, Type accessType,
                              const DataLayout &dataLayout) {
  return dataLayout.getTypeSize(accessType) <=
         dataLayout.getTypeSize(slot.elemType);
}

LogicalResult LLVM::LoadOp::ensureOnlySafeAccesses(
    const MemorySlot &slot, SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
    const DataLayout &dataLayout) {
  return success(getAddr() != slot.ptr ||
                 isValidAccessType(slot, getType(), dataLayout));
}

LogicalResult LLVM::StoreOp::ensureOnlySafeAccesses(
    const MemorySlot &slot, SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
    const DataLayout &dataLayout) {
  return success(getAddr() != slot.ptr ||
                 isValidAccessType(slot, getValue().getType(), dataLayout));
}

/// Returns the subslot's type at the requested index.
static Type getTypeAtIndex(const DestructurableMemorySlot &slot,
                           Attribute index) {
  auto subelementIndexMap =
      cast<DestructurableTypeInterface>(slot.elemType).getSubelementIndexMap();
  if (!subelementIndexMap)
    return {};
  assert(!subelementIndexMap->empty());

  // Note: Returns a null-type when no entry was found.
  return subelementIndexMap->lookup(index);
}

bool LLVM::LoadOp::canRewire(const DestructurableMemorySlot &slot,
                             SmallPtrSetImpl<Attribute> &usedIndices,
                             SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
                             const DataLayout &dataLayout) {
  if (getVolatile_())
    return false;

  // A load always accesses the first element of the destructured slot.
  auto index = IntegerAttr::get(IntegerType::get(getContext(), 32), 0);
  Type subslotType = getTypeAtIndex(slot, index);
  if (!subslotType)
    return false;

  // The access can only be replaced when the subslot is read within its bounds.
  if (dataLayout.getTypeSize(getType()) > dataLayout.getTypeSize(subslotType))
    return false;

  usedIndices.insert(index);
  return true;
}

DeletionKind LLVM::LoadOp::rewire(const DestructurableMemorySlot &slot,
                                  DenseMap<Attribute, MemorySlot> &subslots,
                                  OpBuilder &builder,
                                  const DataLayout &dataLayout) {
  auto index = IntegerAttr::get(IntegerType::get(getContext(), 32), 0);
  auto it = subslots.find(index);
  assert(it != subslots.end());

  getAddrMutable().set(it->getSecond().ptr);
  return DeletionKind::Keep;
}

bool LLVM::StoreOp::canRewire(const DestructurableMemorySlot &slot,
                              SmallPtrSetImpl<Attribute> &usedIndices,
                              SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
                              const DataLayout &dataLayout) {
  if (getVolatile_())
    return false;

  // Storing the pointer to memory cannot be dealt with.
  if (getValue() == slot.ptr)
    return false;

  // A store always accesses the first element of the destructured slot.
  auto index = IntegerAttr::get(IntegerType::get(getContext(), 32), 0);
  Type subslotType = getTypeAtIndex(slot, index);
  if (!subslotType)
    return false;

  // The access can only be replaced when the subslot is read within its bounds.
  if (dataLayout.getTypeSize(getValue().getType()) >
      dataLayout.getTypeSize(subslotType))
    return false;

  usedIndices.insert(index);
  return true;
}

DeletionKind LLVM::StoreOp::rewire(const DestructurableMemorySlot &slot,
                                   DenseMap<Attribute, MemorySlot> &subslots,
                                   OpBuilder &builder,
                                   const DataLayout &dataLayout) {
  auto index = IntegerAttr::get(IntegerType::get(getContext(), 32), 0);
  auto it = subslots.find(index);
  assert(it != subslots.end());

  getAddrMutable().set(it->getSecond().ptr);
  return DeletionKind::Keep;
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
    const SmallPtrSetImpl<OpOperand *> &blockingUses, OpBuilder &builder) {
  return DeletionKind::Delete;
}

bool LLVM::AddrSpaceCastOp::canUsesBeRemoved(
    const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  return forwardToUsers(*this, newBlockingUses);
}

DeletionKind LLVM::AddrSpaceCastOp::removeBlockingUses(
    const SmallPtrSetImpl<OpOperand *> &blockingUses, OpBuilder &builder) {
  return DeletionKind::Delete;
}

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

bool LLVM::DbgDeclareOp::canUsesBeRemoved(
    const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  return true;
}

DeletionKind LLVM::DbgDeclareOp::removeBlockingUses(
    const SmallPtrSetImpl<OpOperand *> &blockingUses, OpBuilder &builder) {
  return DeletionKind::Delete;
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
    const SmallPtrSetImpl<OpOperand *> &blockingUses, OpBuilder &builder) {
  // builder by default is after '*this', but we need it before '*this'.
  builder.setInsertionPoint(*this);

  // Rather than dropping the debug value, replace it with undef to preserve the
  // debug local variable info. This allows the debugger to inform the user that
  // the variable has been optimized out.
  auto undef =
      builder.create<UndefOp>(getValue().getLoc(), getValue().getType());
  getValueMutable().assign(undef);
  return DeletionKind::Keep;
}

bool LLVM::DbgDeclareOp::requiresReplacedValues() { return true; }

void LLVM::DbgDeclareOp::visitReplacedValues(
    ArrayRef<std::pair<Operation *, Value>> definitions, OpBuilder &builder) {
  for (auto [op, value] : definitions) {
    builder.setInsertionPointAfter(op);
    builder.create<LLVM::DbgValueOp>(getLoc(), value, getVarInfo(),
                                     getLocationExpr());
  }
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
    const SmallPtrSetImpl<OpOperand *> &blockingUses, OpBuilder &builder) {
  return DeletionKind::Delete;
}

/// Returns the amount of bytes the provided GEP elements will offset the
/// pointer by. Returns nullopt if no constant offset could be computed.
static std::optional<uint64_t> gepToByteOffset(const DataLayout &dataLayout,
                                               LLVM::GEPOp gep) {
  // Collects all indices.
  SmallVector<uint64_t> indices;
  for (auto index : gep.getIndices()) {
    auto constIndex = dyn_cast<IntegerAttr>(index);
    if (!constIndex)
      return {};
    int64_t gepIndex = constIndex.getInt();
    // Negative indices are not supported.
    if (gepIndex < 0)
      return {};
    indices.push_back(gepIndex);
  }

  Type currentType = gep.getElemType();
  uint64_t offset = indices[0] * dataLayout.getTypeSize(currentType);

  for (uint64_t index : llvm::drop_begin(indices)) {
    bool shouldCancel =
        TypeSwitch<Type, bool>(currentType)
            .Case([&](LLVM::LLVMArrayType arrayType) {
              offset +=
                  index * dataLayout.getTypeSize(arrayType.getElementType());
              currentType = arrayType.getElementType();
              return false;
            })
            .Case([&](LLVM::LLVMStructType structType) {
              ArrayRef<Type> body = structType.getBody();
              assert(index < body.size() && "expected valid struct indexing");
              for (uint32_t i : llvm::seq(index)) {
                if (!structType.isPacked())
                  offset = llvm::alignTo(
                      offset, dataLayout.getTypeABIAlignment(body[i]));
                offset += dataLayout.getTypeSize(body[i]);
              }

              // Align for the current type as well.
              if (!structType.isPacked())
                offset = llvm::alignTo(
                    offset, dataLayout.getTypeABIAlignment(body[index]));
              currentType = body[index];
              return false;
            })
            .Default([&](Type type) {
              LLVM_DEBUG(llvm::dbgs()
                         << "[sroa] Unsupported type for offset computations"
                         << type << "\n");
              return true;
            });

    if (shouldCancel)
      return std::nullopt;
  }

  return offset;
}

namespace {
/// A struct that stores both the index into the aggregate type of the slot as
/// well as the corresponding byte offset in memory.
struct SubslotAccessInfo {
  /// The parent slot's index that the access falls into.
  uint32_t index;
  /// The offset into the subslot of the access.
  uint64_t subslotOffset;
};
} // namespace

/// Computes subslot access information for an access into `slot` with the given
/// offset.
/// Returns nullopt when the offset is out-of-bounds or when the access is into
/// the padding of `slot`.
static std::optional<SubslotAccessInfo>
getSubslotAccessInfo(const DestructurableMemorySlot &slot,
                     const DataLayout &dataLayout, LLVM::GEPOp gep) {
  std::optional<uint64_t> offset = gepToByteOffset(dataLayout, gep);
  if (!offset)
    return {};

  // Helper to check that a constant index is in the bounds of the GEP index
  // representation. LLVM dialects's GEP arguments have a limited bitwidth, thus
  // this additional check is necessary.
  auto isOutOfBoundsGEPIndex = [](uint64_t index) {
    return index >= (1 << LLVM::kGEPConstantBitWidth);
  };

  Type type = slot.elemType;
  if (*offset >= dataLayout.getTypeSize(type))
    return {};
  return TypeSwitch<Type, std::optional<SubslotAccessInfo>>(type)
      .Case([&](LLVM::LLVMArrayType arrayType)
                -> std::optional<SubslotAccessInfo> {
        // Find which element of the array contains the offset.
        uint64_t elemSize = dataLayout.getTypeSize(arrayType.getElementType());
        uint64_t index = *offset / elemSize;
        if (isOutOfBoundsGEPIndex(index))
          return {};
        return SubslotAccessInfo{static_cast<uint32_t>(index),
                                 *offset - (index * elemSize)};
      })
      .Case([&](LLVM::LLVMStructType structType)
                -> std::optional<SubslotAccessInfo> {
        uint64_t distanceToStart = 0;
        // Walk over the elements of the struct to find in which of
        // them the offset is.
        for (auto [index, elem] : llvm::enumerate(structType.getBody())) {
          uint64_t elemSize = dataLayout.getTypeSize(elem);
          if (!structType.isPacked()) {
            distanceToStart = llvm::alignTo(
                distanceToStart, dataLayout.getTypeABIAlignment(elem));
            // If the offset is in padding, cancel the rewrite.
            if (offset < distanceToStart)
              return {};
          }

          if (offset < distanceToStart + elemSize) {
            if (isOutOfBoundsGEPIndex(index))
              return {};
            // The offset is within this element, stop iterating the
            // struct and return the index.
            return SubslotAccessInfo{static_cast<uint32_t>(index),
                                     *offset - distanceToStart};
          }

          // The offset is not within this element, continue walking
          // over the struct.
          distanceToStart += elemSize;
        }

        return {};
      });
}

/// Constructs a byte array type of the given size.
static LLVM::LLVMArrayType getByteArrayType(MLIRContext *context,
                                            unsigned size) {
  auto byteType = IntegerType::get(context, 8);
  return LLVM::LLVMArrayType::get(context, byteType, size);
}

LogicalResult LLVM::GEPOp::ensureOnlySafeAccesses(
    const MemorySlot &slot, SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
    const DataLayout &dataLayout) {
  if (getBase() != slot.ptr)
    return success();
  std::optional<uint64_t> gepOffset = gepToByteOffset(dataLayout, *this);
  if (!gepOffset)
    return failure();
  uint64_t slotSize = dataLayout.getTypeSize(slot.elemType);
  // Check that the access is strictly inside the slot.
  if (*gepOffset >= slotSize)
    return failure();
  // Every access that remains in bounds of the remaining slot is considered
  // legal.
  mustBeSafelyUsed.emplace_back<MemorySlot>(
      {getRes(), getByteArrayType(getContext(), slotSize - *gepOffset)});
  return success();
}

bool LLVM::GEPOp::canRewire(const DestructurableMemorySlot &slot,
                            SmallPtrSetImpl<Attribute> &usedIndices,
                            SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
                            const DataLayout &dataLayout) {
  if (!isa<LLVM::LLVMPointerType>(getBase().getType()))
    return false;

  if (getBase() != slot.ptr)
    return false;
  std::optional<SubslotAccessInfo> accessInfo =
      getSubslotAccessInfo(slot, dataLayout, *this);
  if (!accessInfo)
    return false;
  auto indexAttr =
      IntegerAttr::get(IntegerType::get(getContext(), 32), accessInfo->index);
  assert(slot.elementPtrs.contains(indexAttr));
  usedIndices.insert(indexAttr);

  // The remainder of the subslot should be accesses in-bounds. Thus, we create
  // a dummy slot with the size of the remainder.
  Type subslotType = slot.elementPtrs.lookup(indexAttr);
  uint64_t slotSize = dataLayout.getTypeSize(subslotType);
  LLVM::LLVMArrayType remainingSlotType =
      getByteArrayType(getContext(), slotSize - accessInfo->subslotOffset);
  mustBeSafelyUsed.emplace_back<MemorySlot>({getRes(), remainingSlotType});

  return true;
}

DeletionKind LLVM::GEPOp::rewire(const DestructurableMemorySlot &slot,
                                 DenseMap<Attribute, MemorySlot> &subslots,
                                 OpBuilder &builder,
                                 const DataLayout &dataLayout) {
  std::optional<SubslotAccessInfo> accessInfo =
      getSubslotAccessInfo(slot, dataLayout, *this);
  assert(accessInfo && "expected access info to be checked before");
  auto indexAttr =
      IntegerAttr::get(IntegerType::get(getContext(), 32), accessInfo->index);
  const MemorySlot &newSlot = subslots.at(indexAttr);

  auto byteType = IntegerType::get(builder.getContext(), 8);
  auto newPtr = builder.createOrFold<LLVM::GEPOp>(
      getLoc(), getResult().getType(), byteType, newSlot.ptr,
      ArrayRef<GEPArg>(accessInfo->subslotOffset), getInbounds());
  getResult().replaceAllUsesWith(newPtr);
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

Value LLVM::MemsetOp::getStored(const MemorySlot &slot, OpBuilder &builder,
                                Value reachingDef,
                                const DataLayout &dataLayout) {
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
            builder.create<LLVM::ZExtOp>(getLoc(), intType, getVal());
        while (coveredBits < intType.getWidth()) {
          Value shiftBy =
              builder.create<LLVM::ConstantOp>(getLoc(), intType, coveredBits);
          Value shifted =
              builder.create<LLVM::ShlOp>(getLoc(), currentValue, shiftBy);
          currentValue =
              builder.create<LLVM::OrOp>(getLoc(), currentValue, shifted);
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
  if (&slot.elemType.getDialect() != getOperation()->getDialect())
    return false;

  if (getIsVolatile())
    return false;

  if (!cast<DestructurableTypeInterface>(slot.elemType).getSubelementIndexMap())
    return false;

  if (!areAllIndicesI32(slot))
    return false;

  return definitelyWritesOnlyWithinSlot(*this, slot, dataLayout);
}

DeletionKind LLVM::MemsetOp::rewire(const DestructurableMemorySlot &slot,
                                    DenseMap<Attribute, MemorySlot> &subslots,
                                    OpBuilder &builder,
                                    const DataLayout &dataLayout) {
  std::optional<DenseMap<Attribute, Type>> types =
      cast<DestructurableTypeInterface>(slot.elemType).getSubelementIndexMap();

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
          builder
              .create<LLVM::ConstantOp>(
                  getLen().getLoc(),
                  IntegerAttr::get(memsetLenAttr.getType(), newMemsetSize))
              .getResult();

      builder.create<LLVM::MemsetOp>(getLoc(), subslots.at(index).ptr, getVal(),
                                     newMemsetSizeValue, getIsVolatile());
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
                             OpBuilder &builder) {
  return builder.create<LLVM::LoadOp>(op.getLoc(), slot.elemType, op.getSrc());
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
                         OpBuilder &builder, Value reachingDefinition) {
  if (op.loadsFrom(slot))
    builder.create<LLVM::StoreOp>(op.getLoc(), reachingDefinition, op.getDst());
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

  if (!cast<DestructurableTypeInterface>(slot.elemType).getSubelementIndexMap())
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
void createMemcpyLikeToReplace(OpBuilder &builder, const DataLayout &layout,
                               MemcpyLike toReplace, Value dst, Value src,
                               Type toCpy, bool isVolatile) {
  Value memcpySize = builder.create<LLVM::ConstantOp>(
      toReplace.getLoc(), IntegerAttr::get(toReplace.getLen().getType(),
                                           layout.getTypeSize(toCpy)));
  builder.create<MemcpyLike>(toReplace.getLoc(), dst, src, memcpySize,
                             isVolatile);
}

template <>
void createMemcpyLikeToReplace(OpBuilder &builder, const DataLayout &layout,
                               LLVM::MemcpyInlineOp toReplace, Value dst,
                               Value src, Type toCpy, bool isVolatile) {
  Type lenType = IntegerType::get(toReplace->getContext(),
                                  toReplace.getLen().getBitWidth());
  builder.create<LLVM::MemcpyInlineOp>(
      toReplace.getLoc(), dst, src,
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
    Value subslotPtrInOther = builder.create<LLVM::GEPOp>(
        op.getLoc(), LLVM::LLVMPointerType::get(op.getContext()), slot.elemType,
        isDst ? op.getSrc() : op.getDst(), gepIndices);

    // Then create a new memcpy out of this source pointer.
    createMemcpyLikeToReplace(builder, dataLayout, op,
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
