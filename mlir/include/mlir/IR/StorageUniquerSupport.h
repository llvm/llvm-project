//===- StorageUniquerSupport.h - MLIR Storage Uniquer Utilities -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines utility classes for interfacing with StorageUniquer.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_STORAGEUNIQUERSUPPORT_H
#define MLIR_IR_STORAGEUNIQUERSUPPORT_H

#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/StorageUniquer.h"
#include "mlir/Support/TypeID.h"

namespace mlir {
class AttributeStorage;
class MLIRContext;

namespace detail {
/// Utility method to generate a raw default location for use when checking the
/// construction invariants of a storage object. This is defined out-of-line to
/// avoid the need to include Location.h.
const AttributeStorage *generateUnknownStorageLocation(MLIRContext *ctx);

/// Utility class for implementing users of storage classes uniqued by a
/// StorageUniquer. Clients are not expected to interact with this class
/// directly.
template <typename ConcreteT, typename BaseT, typename StorageT,
          typename UniquerT>
class StorageUserBase : public BaseT {
public:
  using BaseT::BaseT;

  /// Utility declarations for the concrete attribute class.
  using Base = StorageUserBase<ConcreteT, BaseT, StorageT, UniquerT>;
  using ImplType = StorageT;

  /// Return a unique identifier for the concrete type.
  static TypeID getTypeID() { return TypeID::get<ConcreteT>(); }

  /// Provide a default implementation of 'classof' that invokes a 'kindof'
  /// method on the concrete type.
  template <typename T> static bool classof(T val) {
    static_assert(std::is_convertible<ConcreteT, T>::value,
                  "casting from a non-convertible type");
    return ConcreteT::kindof(val.getKind());
  }

protected:
  /// Get or create a new ConcreteT instance within the ctx. This
  /// function is guaranteed to return a non null object and will assert if
  /// the arguments provided are invalid.
  template <typename... Args>
  static ConcreteT get(MLIRContext *ctx, unsigned kind, Args... args) {
    // Ensure that the invariants are correct for construction.
    assert(succeeded(ConcreteT::verifyConstructionInvariants(
        generateUnknownStorageLocation(ctx), args...)));
    return UniquerT::template get<ConcreteT>(ctx, kind, args...);
  }

  /// Get or create a new ConcreteT instance within the ctx, defined at
  /// the given, potentially unknown, location. If the arguments provided are
  /// invalid then emit errors and return a null object.
  template <typename LocationT, typename... Args>
  static ConcreteT getChecked(LocationT loc, unsigned kind, Args... args) {
    // If the construction invariants fail then we return a null attribute.
    if (failed(ConcreteT::verifyConstructionInvariants(loc, args...)))
      return ConcreteT();
    return UniquerT::template get<ConcreteT>(loc.getContext(), kind, args...);
  }

  /// Default implementation that just returns success.
  template <typename... Args>
  static LogicalResult verifyConstructionInvariants(Args... args) {
    return success();
  }

  /// Utility for easy access to the storage instance.
  ImplType *getImpl() const { return static_cast<ImplType *>(this->impl); }
};
} // namespace detail
} // namespace mlir

#endif
