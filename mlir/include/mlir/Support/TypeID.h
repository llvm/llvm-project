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
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/PointerLikeTypeTraits.h"

namespace mlir {

namespace detail {
struct TypeIDExported;
} // namespace detail

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

  /// Comparison operations.
  inline bool operator==(const TypeID &other) const {
    return storage == other.storage;
  }
  inline bool operator!=(const TypeID &other) const {
    return !(*this == other);
  }

  /// Construct a type info object for the given type T.
  template <typename T>
  static TypeID get();
  template <template <typename> class Trait>
  static TypeID get();

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

  // See TypeIDExported below for an explanation of the trampoline behavior.
  friend struct detail::TypeIDExported;
};

/// Enable hashing TypeID.
inline ::llvm::hash_code hash_value(TypeID id) {
  return DenseMapInfo<const TypeID::Storage *>::getHashValue(id.storage);
}

namespace detail {

/// The static local instance of each get method must be emitted with
/// "default" (public) visibility across all shared libraries, regardless of
/// whether they are compiled with hidden visibility or not. The only reliable
/// way to make this happen is to set the visibility attribute at the
/// containing namespace/struct scope. We don't do this on the TypeID (internal
/// API) class in order to reduce the scope of what gets exported with
/// public visibility. Instead, the get() methods on TypeID trampoline
/// through those on this detail class with specific visibility controls
/// applied, making visibility declarations on the internal TypeID class not
/// required (all visibility relevant pieces are here).
/// TODO: This currently won't work when using DLLs as it requires properly
/// attaching dllimport and dllexport. Fix this when that information is
/// available within LLVM.
struct LLVM_EXTERNAL_VISIBILITY TypeIDExported {
  template <typename T>
  static TypeID get() {
    static TypeID::Storage instance;
    return TypeID(&instance);
  }
  template <template <typename> class Trait>
  static TypeID get() {
    static TypeID::Storage instance;
    return TypeID(&instance);
  }
};

} // namespace detail

template <typename T>
TypeID TypeID::get() {
  return detail::TypeIDExported::get<T>();
}
template <template <typename> class Trait>
TypeID TypeID::get() {
  return detail::TypeIDExported::get<Trait>();
}

} // namespace mlir

// Declare/define an explicit specialization for TypeID: this forces the
// compiler to emit a strong definition for a class and controls which
// translation unit and shared object will actually have it.
// This can be useful to turn to a link-time failure what would be in other
// circumstances a hard-to-catch runtime bug when a TypeID is hidden in two
// different shared libraries and instances of the same class only gets the same
// TypeID inside a given DSO.
#define DECLARE_EXPLICIT_TYPE_ID(CLASS_NAME)                                   \
  namespace mlir {                                                             \
  namespace detail {                                                           \
  template <>                                                                  \
  LLVM_EXTERNAL_VISIBILITY TypeID TypeIDExported::get<CLASS_NAME>();           \
  }                                                                            \
  }

#define DEFINE_EXPLICIT_TYPE_ID(CLASS_NAME)                                    \
  namespace mlir {                                                             \
  namespace detail {                                                           \
  template <>                                                                  \
  LLVM_EXTERNAL_VISIBILITY TypeID TypeIDExported::get<CLASS_NAME>() {          \
    static TypeID::Storage instance;                                           \
    return TypeID(&instance);                                                  \
  }                                                                            \
  }                                                                            \
  }

namespace llvm {
template <> struct DenseMapInfo<mlir::TypeID> {
  static inline mlir::TypeID getEmptyKey() {
    void *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::TypeID::getFromOpaquePointer(pointer);
  }
  static inline mlir::TypeID getTombstoneKey() {
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

} // namespace llvm

#endif // MLIR_SUPPORT_TYPEID_H
