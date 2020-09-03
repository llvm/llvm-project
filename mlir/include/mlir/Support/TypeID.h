//===- TypeID.h - TypeID RTTI class -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a definition of the TypeID class. This provides a non
// RTTI mechanism for producing unique type IDs in LLVM.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_TYPEID_H
#define MLIR_SUPPORT_TYPEID_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/Support/PointerLikeTypeTraits.h"

namespace mlir {

/// This class provides an efficient unique identifier for a specific C++ type.
/// This allows for a C++ type to be compared, hashed, and stored in an opaque
/// context. This class is similar in some ways to std::type_index, but can be
/// used for any type. For example, this class could be used to implement LLVM
/// style isa/dyn_cast functionality for a type hierarchy:
///
///  struct Base {
///    Base(TypeID typeID) : typeID(typeID) {}
///    TypeID typeID;
///  };
///
///  struct DerivedA : public Base {
///    DerivedA() : Base(TypeID::get<DerivedA>()) {}
///
///    static bool classof(const Base *base) {
///      return base->typeID == TypeID::get<DerivedA>();
///    }
///  };
///
///  void foo(Base *base) {
///    if (DerivedA *a = llvm::dyn_cast<DerivedA>(base))
///       ...
///  }
///
class TypeID {
  /// This class represents the storage of a type info object.
  /// Note: We specify an explicit alignment here to allow use with
  /// PointerIntPair and other utilities/data structures that require a known
  /// pointer alignment.
  struct alignas(8) Storage {};

public:
  TypeID() : TypeID(get<void>()) {}
  TypeID(const TypeID &) = default;

  /// Comparison operations.
  bool operator==(const TypeID &other) const {
    return storage == other.storage;
  }
  bool operator!=(const TypeID &other) const { return !(*this == other); }

  /// Construct a type info object for the given type T.
  /// TODO: This currently won't work when using DLLs as it requires properly
  /// attaching dllimport and dllexport. Fix this when that information is
  /// available within LLVM.
  template <typename T>
  LLVM_EXTERNAL_VISIBILITY static TypeID get() {
    static Storage instance;
    return TypeID(&instance);
  }
  template <template <typename> class Trait>
  LLVM_EXTERNAL_VISIBILITY static TypeID get() {
    static Storage instance;
    return TypeID(&instance);
  }

  /// Methods for supporting PointerLikeTypeTraits.
  const void *getAsOpaquePointer() const {
    return static_cast<const void *>(storage);
  }
  static TypeID getFromOpaquePointer(const void *pointer) {
    return TypeID(reinterpret_cast<const Storage *>(pointer));
  }

  /// Enable hashing TypeID.
  friend ::llvm::hash_code hash_value(TypeID id);

private:
  TypeID(const Storage *storage) : storage(storage) {}

  /// The storage of this type info object.
  const Storage *storage;
};

/// Enable hashing TypeID.
inline ::llvm::hash_code hash_value(TypeID id) {
  return llvm::hash_value(id.storage);
}

} // end namespace mlir

namespace llvm {
template <> struct DenseMapInfo<mlir::TypeID> {
  static mlir::TypeID getEmptyKey() {
    void *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::TypeID::getFromOpaquePointer(pointer);
  }
  static mlir::TypeID getTombstoneKey() {
    void *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::TypeID::getFromOpaquePointer(pointer);
  }
  static unsigned getHashValue(mlir::TypeID val) {
    return mlir::hash_value(val);
  }
  static bool isEqual(mlir::TypeID lhs, mlir::TypeID rhs) { return lhs == rhs; }
};

/// We align TypeID::Storage by 8, so allow LLVM to steal the low bits.
template <> struct PointerLikeTypeTraits<mlir::TypeID> {
  static inline void *getAsVoidPointer(mlir::TypeID info) {
    return const_cast<void *>(info.getAsOpaquePointer());
  }
  static inline mlir::TypeID getFromVoidPointer(void *ptr) {
    return mlir::TypeID::getFromOpaquePointer(ptr);
  }
  static constexpr int NumLowBitsAvailable = 3;
};

} // end namespace llvm

#endif // MLIR_SUPPORT_TYPEID_H
