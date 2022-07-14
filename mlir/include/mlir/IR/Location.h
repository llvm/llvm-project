//===- Location.h - MLIR Location Classes -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These classes provide the ability to relate MLIR objects back to source
// location position information.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_LOCATION_H
#define MLIR_IR_LOCATION_H

#include "mlir/IR/Attributes.h"
#include "llvm/Support/PointerLikeTypeTraits.h"

namespace mlir {

class Location;
class WalkResult;

//===----------------------------------------------------------------------===//
// LocationAttr
//===----------------------------------------------------------------------===//

/// Location objects represent source locations information in MLIR.
/// LocationAttr acts as the anchor for all Location based attributes.
class LocationAttr : public Attribute {
public:
  using Attribute::Attribute;

  /// Walk all of the locations nested under, and including, the current.
  WalkResult walk(function_ref<WalkResult(Location)> walkFn);

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(Attribute attr);
};

//===----------------------------------------------------------------------===//
// Location
//===----------------------------------------------------------------------===//

/// This class defines the main interface for locations in MLIR and acts as a
/// non-nullable wrapper around a LocationAttr.
class Location {
public:
  Location(LocationAttr loc) : impl(loc) {
    assert(loc && "location should never be null.");
  }
  Location(const LocationAttr::ImplType *impl) : impl(impl) {
    assert(impl && "location should never be null.");
  }

  /// Return the context this location is uniqued in.
  MLIRContext *getContext() const { return impl.getContext(); }

  /// Access the impl location attribute.
  operator LocationAttr() const { return impl; }
  LocationAttr *operator->() const { return const_cast<LocationAttr *>(&impl); }

  /// Type casting utilities on the underlying location.
  template <typename U>
  bool isa() const {
    return impl.isa<U>();
  }
  template <typename U>
  U dyn_cast() const {
    return impl.dyn_cast<U>();
  }
  template <typename U>
  U cast() const {
    return impl.cast<U>();
  }

  /// Comparison operators.
  bool operator==(Location rhs) const { return impl == rhs.impl; }
  bool operator!=(Location rhs) const { return !(*this == rhs); }

  /// Print the location.
  void print(raw_ostream &os) const { impl.print(os); }
  void dump() const { impl.dump(); }

  friend ::llvm::hash_code hash_value(Location arg);

  /// Methods for supporting PointerLikeTypeTraits.
  const void *getAsOpaquePointer() const { return impl.getAsOpaquePointer(); }
  static Location getFromOpaquePointer(const void *pointer) {
    return LocationAttr(reinterpret_cast<const AttributeStorage *>(pointer));
  }

protected:
  /// The internal backing location attribute.
  LocationAttr impl;
};

inline raw_ostream &operator<<(raw_ostream &os, const Location &loc) {
  loc.print(os);
  return os;
}

// Make Location hashable.
inline ::llvm::hash_code hash_value(Location arg) {
  return hash_value(arg.impl);
}

} // namespace mlir

//===----------------------------------------------------------------------===//
// Tablegen Attribute Declarations
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "mlir/IR/BuiltinLocationAttributes.h.inc"

namespace mlir {

//===----------------------------------------------------------------------===//
// OpaqueLoc
//===----------------------------------------------------------------------===//

/// Returns an instance of opaque location which contains a given pointer to
/// an object. The corresponding MLIR location is set to UnknownLoc.
template <typename T>
inline OpaqueLoc OpaqueLoc::get(T underlyingLocation, MLIRContext *context) {
  return get(reinterpret_cast<uintptr_t>(underlyingLocation), TypeID::get<T>(),
             UnknownLoc::get(context));
}
} // namespace mlir

//===----------------------------------------------------------------------===//
// LLVM Utilities
//===----------------------------------------------------------------------===//

namespace llvm {

// Type hash just like pointers.
template <>
struct DenseMapInfo<mlir::Location> {
  static mlir::Location getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::Location::getFromOpaquePointer(pointer);
  }
  static mlir::Location getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::Location::getFromOpaquePointer(pointer);
  }
  static unsigned getHashValue(mlir::Location val) {
    return mlir::hash_value(val);
  }
  static bool isEqual(mlir::Location LHS, mlir::Location RHS) {
    return LHS == RHS;
  }
};

/// We align LocationStorage by 8, so allow LLVM to steal the low bits.
template <>
struct PointerLikeTypeTraits<mlir::Location> {
public:
  static inline void *getAsVoidPointer(mlir::Location I) {
    return const_cast<void *>(I.getAsOpaquePointer());
  }
  static inline mlir::Location getFromVoidPointer(void *P) {
    return mlir::Location::getFromOpaquePointer(P);
  }
  static constexpr int NumLowBitsAvailable =
      PointerLikeTypeTraits<mlir::Attribute>::NumLowBitsAvailable;
};

} // namespace llvm

#endif
