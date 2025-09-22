//===- PtrDialect.cpp - Pointer dialect ---------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Pointer dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::ptr;

//===----------------------------------------------------------------------===//
// Pointer dialect
//===----------------------------------------------------------------------===//

void PtrDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Ptr/IR/PtrOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/Ptr/IR/PtrOpsAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Ptr/IR/PtrOpsTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Common helper functions.
//===----------------------------------------------------------------------===//

/// Verifies that the alignment attribute is a power of 2 if present.
static LogicalResult
verifyAlignment(std::optional<int64_t> alignment,
                function_ref<InFlightDiagnostic()> emitError) {
  if (!alignment)
    return success();
  if (alignment.value() <= 0)
    return emitError() << "alignment must be positive";
  if (!llvm::isPowerOf2_64(alignment.value()))
    return emitError() << "alignment must be a power of 2";
  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) { return getValue(); }

//===----------------------------------------------------------------------===//
// FromPtrOp
//===----------------------------------------------------------------------===//

OpFoldResult FromPtrOp::fold(FoldAdaptor adaptor) {
  // Fold the pattern:
  // %ptr = ptr.to_ptr %v : type -> ptr
  // (%mda = ptr.get_metadata %v : type)?
  // %val = ptr.from_ptr %ptr (metadata %mda)? : ptr -> type
  // To:
  // %val -> %v
  Value ptrLike;
  FromPtrOp fromPtr = *this;
  while (fromPtr != nullptr) {
    auto toPtr = fromPtr.getPtr().getDefiningOp<ToPtrOp>();
    // Cannot fold if it's not a `to_ptr` op or the initial and final types are
    // different.
    if (!toPtr || toPtr.getPtr().getType() != fromPtr.getType())
      return ptrLike;
    Value md = fromPtr.getMetadata();
    // If the type has trivial metadata fold.
    if (!fromPtr.getType().hasPtrMetadata()) {
      ptrLike = toPtr.getPtr();
    } else if (md) {
      // Fold if the metadata can be verified to be equal.
      if (auto mdOp = md.getDefiningOp<GetMetadataOp>();
          mdOp && mdOp.getPtr() == toPtr.getPtr())
        ptrLike = toPtr.getPtr();
    }
    // Check for a sequence of casts.
    fromPtr = ptrLike ? ptrLike.getDefiningOp<FromPtrOp>() : nullptr;
  }
  return ptrLike;
}

LogicalResult FromPtrOp::verify() {
  if (isa<PtrType>(getType()))
    return emitError() << "the result type cannot be `!ptr.ptr`";
  if (getType().getMemorySpace() != getPtr().getType().getMemorySpace()) {
    return emitError()
           << "expected the input and output to have the same memory space";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

void GatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // Gather performs reads from multiple memory locations specified by ptrs
  effects.emplace_back(MemoryEffects::Read::get(), &getPtrsMutable());
}

LogicalResult GatherOp::verify() {
  auto emitDiag = [&]() -> InFlightDiagnostic { return emitError(); };

  // Verify that the pointer type's memory space allows loads.
  MemorySpaceAttrInterface ms =
      cast<PtrType>(getPtrs().getType().getElementType()).getMemorySpace();
  DataLayout dataLayout = DataLayout::closest(*this);
  if (!ms.isValidLoad(getResult().getType(), AtomicOrdering::not_atomic,
                      getAlignment(), &dataLayout, emitDiag))
    return failure();

  // Verify the alignment.
  return verifyAlignment(getAlignment(), emitDiag);
}

void GatherOp::build(OpBuilder &builder, OperationState &state, Type resultType,
                     Value ptrs, Value mask, Value passthrough,
                     unsigned alignment) {
  build(builder, state, resultType, ptrs, mask, passthrough,
        alignment ? std::optional<int64_t>(alignment) : std::nullopt);
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

/// Verifies the attributes and the type of atomic memory access operations.
template <typename OpTy>
static LogicalResult
verifyAtomicMemOp(OpTy memOp, ArrayRef<AtomicOrdering> unsupportedOrderings) {
  if (memOp.getOrdering() != AtomicOrdering::not_atomic) {
    if (llvm::is_contained(unsupportedOrderings, memOp.getOrdering()))
      return memOp.emitOpError("unsupported ordering '")
             << stringifyAtomicOrdering(memOp.getOrdering()) << "'";
    if (!memOp.getAlignment())
      return memOp.emitOpError("expected alignment for atomic access");
    return success();
  }
  if (memOp.getSyncscope()) {
    return memOp.emitOpError(
        "expected syncscope to be null for non-atomic access");
  }
  return success();
}

void LoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getPtrMutable());
  // Volatile operations can have target-specific read-write effects on
  // memory besides the one referred to by the pointer operand.
  // Similarly, atomic operations that are monotonic or stricter cause
  // synchronization that from a language point-of-view, are arbitrary
  // read-writes into memory.
  if (getVolatile_() || (getOrdering() != AtomicOrdering::not_atomic &&
                         getOrdering() != AtomicOrdering::unordered)) {
    effects.emplace_back(MemoryEffects::Write::get());
    effects.emplace_back(MemoryEffects::Read::get());
  }
}

LogicalResult LoadOp::verify() {
  auto emitDiag = [&]() -> InFlightDiagnostic { return emitError(); };
  MemorySpaceAttrInterface ms = getPtr().getType().getMemorySpace();
  DataLayout dataLayout = DataLayout::closest(*this);
  if (!ms.isValidLoad(getResult().getType(), getOrdering(), getAlignment(),
                      &dataLayout, emitDiag))
    return failure();
  if (failed(verifyAlignment(getAlignment(), emitDiag)))
    return failure();
  return verifyAtomicMemOp(*this,
                           {AtomicOrdering::release, AtomicOrdering::acq_rel});
}

void LoadOp::build(OpBuilder &builder, OperationState &state, Type type,
                   Value addr, unsigned alignment, bool isVolatile,
                   bool isNonTemporal, bool isInvariant, bool isInvariantGroup,
                   AtomicOrdering ordering, StringRef syncscope) {
  build(builder, state, type, addr,
        alignment ? std::optional<int64_t>(alignment) : std::nullopt,
        isVolatile, isNonTemporal, isInvariant, isInvariantGroup, ordering,
        syncscope.empty() ? nullptr : builder.getStringAttr(syncscope));
}
//===----------------------------------------------------------------------===//
// MaskedLoadOp
//===----------------------------------------------------------------------===//

void MaskedLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // MaskedLoad performs reads from the memory location specified by ptr.
  effects.emplace_back(MemoryEffects::Read::get(), &getPtrMutable());
}

LogicalResult MaskedLoadOp::verify() {
  auto emitDiag = [&]() -> InFlightDiagnostic { return emitError(); };
  // Verify that the pointer type's memory space allows loads.
  MemorySpaceAttrInterface ms = getPtr().getType().getMemorySpace();
  DataLayout dataLayout = DataLayout::closest(*this);
  if (!ms.isValidLoad(getResult().getType(), AtomicOrdering::not_atomic,
                      getAlignment(), &dataLayout, emitDiag))
    return failure();

  // Verify the alignment.
  return verifyAlignment(getAlignment(), emitDiag);
}

void MaskedLoadOp::build(OpBuilder &builder, OperationState &state,
                         Type resultType, Value ptr, Value mask,
                         Value passthrough, unsigned alignment) {
  build(builder, state, resultType, ptr, mask, passthrough,
        alignment ? std::optional<int64_t>(alignment) : std::nullopt);
}

//===----------------------------------------------------------------------===//
// MaskedStoreOp
//===----------------------------------------------------------------------===//

void MaskedStoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // MaskedStore performs writes to the memory location specified by ptr
  effects.emplace_back(MemoryEffects::Write::get(), &getPtrMutable());
}

LogicalResult MaskedStoreOp::verify() {
  auto emitDiag = [&]() -> InFlightDiagnostic { return emitError(); };
  // Verify that the pointer type's memory space allows stores.
  MemorySpaceAttrInterface ms = getPtr().getType().getMemorySpace();
  DataLayout dataLayout = DataLayout::closest(*this);
  if (!ms.isValidStore(getValue().getType(), AtomicOrdering::not_atomic,
                       getAlignment(), &dataLayout, emitDiag))
    return failure();

  // Verify the alignment.
  return verifyAlignment(getAlignment(), emitDiag);
}

void MaskedStoreOp::build(OpBuilder &builder, OperationState &state,
                          Value value, Value ptr, Value mask,
                          unsigned alignment) {
  build(builder, state, value, ptr, mask,
        alignment ? std::optional<int64_t>(alignment) : std::nullopt);
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

void ScatterOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // Scatter performs writes to multiple memory locations specified by ptrs
  effects.emplace_back(MemoryEffects::Write::get(), &getPtrsMutable());
}

LogicalResult ScatterOp::verify() {
  auto emitDiag = [&]() -> InFlightDiagnostic { return emitError(); };

  // Verify that the pointer type's memory space allows stores.
  MemorySpaceAttrInterface ms =
      cast<PtrType>(getPtrs().getType().getElementType()).getMemorySpace();
  DataLayout dataLayout = DataLayout::closest(*this);
  if (!ms.isValidStore(getValue().getType(), AtomicOrdering::not_atomic,
                       getAlignment(), &dataLayout, emitDiag))
    return failure();

  // Verify the alignment.
  return verifyAlignment(getAlignment(), emitDiag);
}

void ScatterOp::build(OpBuilder &builder, OperationState &state, Value value,
                      Value ptrs, Value mask, unsigned alignment) {
  build(builder, state, value, ptrs, mask,
        alignment ? std::optional<int64_t>(alignment) : std::nullopt);
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

void StoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getPtrMutable());
  // Volatile operations can have target-specific read-write effects on
  // memory besides the one referred to by the pointer operand.
  // Similarly, atomic operations that are monotonic or stricter cause
  // synchronization that from a language point-of-view, are arbitrary
  // read-writes into memory.
  if (getVolatile_() || (getOrdering() != AtomicOrdering::not_atomic &&
                         getOrdering() != AtomicOrdering::unordered)) {
    effects.emplace_back(MemoryEffects::Write::get());
    effects.emplace_back(MemoryEffects::Read::get());
  }
}

LogicalResult StoreOp::verify() {
  auto emitDiag = [&]() -> InFlightDiagnostic { return emitError(); };
  MemorySpaceAttrInterface ms = getPtr().getType().getMemorySpace();
  DataLayout dataLayout = DataLayout::closest(*this);
  if (!ms.isValidStore(getValue().getType(), getOrdering(), getAlignment(),
                       &dataLayout, emitDiag))
    return failure();
  if (failed(verifyAlignment(getAlignment(), emitDiag)))
    return failure();
  return verifyAtomicMemOp(*this,
                           {AtomicOrdering::acquire, AtomicOrdering::acq_rel});
}

void StoreOp::build(OpBuilder &builder, OperationState &state, Value value,
                    Value addr, unsigned alignment, bool isVolatile,
                    bool isNonTemporal, bool isInvariantGroup,
                    AtomicOrdering ordering, StringRef syncscope) {
  build(builder, state, value, addr,
        alignment ? std::optional<int64_t>(alignment) : std::nullopt,
        isVolatile, isNonTemporal, isInvariantGroup, ordering,
        syncscope.empty() ? nullptr : builder.getStringAttr(syncscope));
}

//===----------------------------------------------------------------------===//
// PtrAddOp
//===----------------------------------------------------------------------===//

/// Fold: ptradd ptr + 0 ->  ptr
OpFoldResult PtrAddOp::fold(FoldAdaptor adaptor) {
  Attribute attr = adaptor.getOffset();
  if (!attr)
    return nullptr;
  if (llvm::APInt value; m_ConstantInt(&value).match(attr) && value.isZero())
    return getBase();
  return nullptr;
}

LogicalResult PtrAddOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // Get the base pointer and offset types.
  Type baseType = operands[0].getType();
  Type offsetType = operands[1].getType();

  auto offTy = dyn_cast<ShapedType>(offsetType);
  if (!offTy) {
    // If the offset isn't shaped, the result is always the base type.
    inferredReturnTypes.push_back(baseType);
    return success();
  }
  auto baseTy = dyn_cast<ShapedType>(baseType);
  if (!baseTy) {
    // Base isn't shaped, but offset is, use the ShapedType from offset with the
    // base pointer as element type.
    inferredReturnTypes.push_back(offTy.clone(baseType));
    return success();
  }

  // Both are shaped, their shape must match.
  if (offTy.getShape() != baseTy.getShape()) {
    if (location)
      mlir::emitError(*location) << "shapes of base and offset must match";
    return failure();
  }

  // Make sure they are the same kind of shaped type.
  if (baseType.getTypeID() != offsetType.getTypeID()) {
    if (location)
      mlir::emitError(*location) << "the shaped containers type must match";
    return failure();
  }
  inferredReturnTypes.push_back(baseType);
  return success();
}

//===----------------------------------------------------------------------===//
// ToPtrOp
//===----------------------------------------------------------------------===//

OpFoldResult ToPtrOp::fold(FoldAdaptor adaptor) {
  // Fold the pattern:
  // %val = ptr.from_ptr %p (metadata ...)? : ptr -> type
  // %ptr = ptr.to_ptr %val : type -> ptr
  // To:
  // %ptr -> %p
  Value ptr;
  ToPtrOp toPtr = *this;
  while (toPtr != nullptr) {
    auto fromPtr = toPtr.getPtr().getDefiningOp<FromPtrOp>();
    // Cannot fold if it's not a `from_ptr` op.
    if (!fromPtr)
      return ptr;
    ptr = fromPtr.getPtr();
    // Check for chains of casts.
    toPtr = ptr.getDefiningOp<ToPtrOp>();
  }
  return ptr;
}

LogicalResult ToPtrOp::verify() {
  if (isa<PtrType>(getPtr().getType()))
    return emitError() << "the input value cannot be of type `!ptr.ptr`";
  if (getType().getMemorySpace() != getPtr().getType().getMemorySpace()) {
    return emitError()
           << "expected the input and output to have the same memory space";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TypeOffsetOp
//===----------------------------------------------------------------------===//

llvm::TypeSize TypeOffsetOp::getTypeSize(std::optional<DataLayout> layout) {
  if (layout)
    return layout->getTypeSize(getElementType());
  DataLayout dl = DataLayout::closest(*this);
  return dl.getTypeSize(getElementType());
}

//===----------------------------------------------------------------------===//
// Pointer API.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Ptr/IR/PtrOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Ptr/IR/PtrOpsAttrs.cpp.inc"

#include "mlir/Dialect/Ptr/IR/PtrOpsEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Ptr/IR/PtrOpsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Ptr/IR/PtrOps.cpp.inc"
