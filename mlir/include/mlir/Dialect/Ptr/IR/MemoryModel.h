//===-- MemoryModel.h - ptr dialect memory model  ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ptr's dialect memory model class and related
// interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTR_IR_MEMORYMODEL_H
#define MLIR_DIALECT_PTR_IR_MEMORYMODEL_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
class Operation;
namespace ptr {
/// This method checks if it's valid to perform an `addrspacecast` op in the
/// memory space.
/// Compatible types are:
/// Vectors of rank 1, or scalars of `ptr` type.
LogicalResult isValidAddrSpaceCastImpl(Type tgt, Type src,
                                       Operation *diagnosticOp);

/// This method checks if it's valid to perform a `ptrtoint` or `inttoptr` op in
/// the memory space.
/// Compatible types are:
/// IntLikeTy: Vectors of rank 1, or scalars of integer types or `index` type.
/// PtrLikeTy: Vectors of rank 1, or scalars of `ptr` type.
LogicalResult isValidPtrIntCastImpl(Type intLikeTy, Type ptrLikeTy,
                                    Operation *diagnosticOp);

enum class AtomicBinOp : uint64_t;
enum class AtomicOrdering : uint64_t;
} // namespace ptr
} // namespace mlir

#include "mlir/Dialect/Ptr/IR/MemorySpaceAttrInterfaces.h.inc"

namespace mlir {
namespace ptr {
/// This class wraps the `MemorySpaceAttrInterface` interface, providing a safe
/// mechanism to specify the default behavior assumed by the ptr dialect.
class MemoryModel {
public:
  MemoryModel() = default;
  MemoryModel(std::nullptr_t) {}
  MemoryModel(MemorySpaceAttrInterface memorySpace)
      : memorySpaceAttr(memorySpace), memorySpace(memorySpace) {}
  MemoryModel(Attribute memorySpace)
      : memorySpaceAttr(memorySpace),
        memorySpace(dyn_cast_or_null<MemorySpaceAttrInterface>(memorySpace)) {}

  operator Attribute() const { return memorySpaceAttr; }
  operator MemorySpaceAttrInterface() const { return memorySpace; }
  bool operator==(const MemoryModel &memSpace) const {
    return memSpace.memorySpaceAttr == memorySpaceAttr;
  }

  /// Returns the underlying memory space.
  Attribute getUnderlyingSpace() const { return memorySpaceAttr; }

  /// Returns true if the underlying memory space is null.
  bool isDefaultModel() const { return memorySpace == nullptr; }

  /// Returns the memory space as an integer, or 0 if using the default model.
  unsigned getAddressSpace() const {
    if (memorySpace)
      return memorySpace.getAddressSpace();
    if (auto intAttr = llvm::dyn_cast_or_null<IntegerAttr>(memorySpaceAttr))
      return intAttr.getInt();
    return 0;
  }

  /// Returns the default memory space as an attribute, or nullptr if using the
  /// default model.
  Attribute getDefaultMemorySpace() const {
    return memorySpace ? memorySpace.getDefaultMemorySpace() : nullptr;
  }

  /// This method checks if it's valid to load a value from the memory space
  /// with a specific type, alignment, and atomic ordering. The default model
  /// assumes all values are loadable.
  LogicalResult isValidLoad(Type type, AtomicOrdering ordering,
                            IntegerAttr alignment,
                            Operation *diagnosticOp = nullptr) const {
    return memorySpace ? memorySpace.isValidLoad(type, ordering, alignment,
                                                 diagnosticOp)
                       : success();
  }

  /// This method checks if it's valid to store a value in the memory space with
  /// a specific type, alignment, and atomic ordering. The default model assumes
  /// all values are loadable.
  LogicalResult isValidStore(Type type, AtomicOrdering ordering,
                             IntegerAttr alignment,
                             Operation *diagnosticOp = nullptr) const {
    return memorySpace ? memorySpace.isValidStore(type, ordering, alignment,
                                                  diagnosticOp)
                       : success();
  }

  /// This method checks if it's valid to perform an atomic operation in the
  /// memory space with a specific type, alignment, and atomic ordering.
  LogicalResult isValidAtomicOp(AtomicBinOp op, Type type,
                                AtomicOrdering ordering, IntegerAttr alignment,
                                Operation *diagnosticOp = nullptr) const {
    return memorySpace ? memorySpace.isValidAtomicOp(op, type, ordering,
                                                     alignment, diagnosticOp)
                       : success();
  }

  /// This method checks if it's valid to perform an atomic operation in the
  /// memory space with a specific type, alignment, and atomic ordering.
  LogicalResult isValidAtomicXchg(Type type, AtomicOrdering successOrdering,
                                  AtomicOrdering failureOrdering,
                                  IntegerAttr alignment,
                                  Operation *diagnosticOp = nullptr) const {
    return memorySpace ? memorySpace.isValidAtomicXchg(type, successOrdering,
                                                       failureOrdering,
                                                       alignment, diagnosticOp)
                       : success();
  }

  /// This method checks if it's valid to perform an `addrspacecast` op in the
  /// memory space.
  LogicalResult isValidAddrSpaceCast(Type tgt, Type src,
                                     Operation *diagnosticOp = nullptr) const {
    return memorySpace
               ? memorySpace.isValidAddrSpaceCast(tgt, src, diagnosticOp)
               : isValidAddrSpaceCastImpl(tgt, src, diagnosticOp);
  }

  /// This method checks if it's valid to perform a `ptrtoint` or `inttoptr` op
  /// in the memory space.
  LogicalResult isValidPtrIntCast(Type intLikeTy, Type ptrLikeTy,
                                  Operation *diagnosticOp = nullptr) const {
    return memorySpace
               ? memorySpace.isValidPtrIntCast(intLikeTy, ptrLikeTy,
                                               diagnosticOp)
               : isValidPtrIntCastImpl(intLikeTy, ptrLikeTy, diagnosticOp);
  }

protected:
  /// Underlying memory space.
  Attribute memorySpaceAttr{};
  /// Memory space.
  MemorySpaceAttrInterface memorySpace{};
};
} // namespace ptr
} // namespace mlir

#include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.h.inc"

#endif // MLIR_DIALECT_PTR_IR_MEMORYMODEL_H
