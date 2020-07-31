//===- TypeSupport.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines support types for registering dialect extended types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_TYPE_SUPPORT_H
#define MLIR_IR_TYPE_SUPPORT_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StorageUniquerSupport.h"

namespace mlir {
class Dialect;
class MLIRContext;

//===----------------------------------------------------------------------===//
// AbstractType
//===----------------------------------------------------------------------===//

/// This class contains all of the static information common to all instances of
/// a registered Type.
class AbstractType {
public:
  /// Look up the specified abstract type in the MLIRContext and return a
  /// reference to it.
  static const AbstractType &lookup(TypeID typeID, MLIRContext *context);

  /// This method is used by Dialect objects when they register the list of
  /// types they contain.
  template <typename T> static AbstractType get(Dialect &dialect) {
    return AbstractType(dialect, T::getInterfaceMap());
  }

  /// Return the dialect this type was registered to.
  Dialect &getDialect() const { return const_cast<Dialect &>(dialect); }

  /// Returns an instance of the concept object for the given interface if it
  /// was registered to this type, null otherwise. This should not be used
  /// directly.
  template <typename T> typename T::Concept *getInterface() const {
    return interfaceMap.lookup<T>();
  }

private:
  AbstractType(Dialect &dialect, detail::InterfaceMap &&interfaceMap)
      : dialect(dialect), interfaceMap(std::move(interfaceMap)) {}

  /// This is the dialect that this type was registered to.
  Dialect &dialect;

  /// This is a collection of the interfaces registered to this type.
  detail::InterfaceMap interfaceMap;
};

//===----------------------------------------------------------------------===//
// TypeStorage
//===----------------------------------------------------------------------===//

namespace detail {
struct TypeUniquer;
} // end namespace detail

/// Base storage class appearing in a Type.
class TypeStorage : public StorageUniquer::BaseStorage {
  friend detail::TypeUniquer;
  friend StorageUniquer;

public:
  /// Return the abstract type descriptor for this type.
  const AbstractType &getAbstractType() {
    assert(abstractType && "Malformed type storage object.");
    return *abstractType;
  }

  /// Get the subclass data.
  unsigned getSubclassData() const { return subclassData; }

  /// Set the subclass data.
  void setSubclassData(unsigned val) { subclassData = val; }

protected:
  /// This constructor is used by derived classes as part of the TypeUniquer.
  TypeStorage(unsigned subclassData = 0)
      : abstractType(nullptr), subclassData(subclassData) {}

private:
  /// Set the abstract type for this storage instance. This is used by the
  /// TypeUniquer when initializing a newly constructed type storage object.
  void initialize(const AbstractType &abstractTy) {
    abstractType = &abstractTy;
  }

  /// The abstract description for this type.
  const AbstractType *abstractType;

  /// Space for subclasses to store data.
  unsigned subclassData;
};

/// Default storage type for types that require no additional initialization or
/// storage.
using DefaultTypeStorage = TypeStorage;

//===----------------------------------------------------------------------===//
// TypeStorageAllocator
//===----------------------------------------------------------------------===//

/// This is a utility allocator used to allocate memory for instances of derived
/// Types.
using TypeStorageAllocator = StorageUniquer::StorageAllocator;

//===----------------------------------------------------------------------===//
// TypeUniquer
//===----------------------------------------------------------------------===//
namespace detail {
/// A utility class to get, or create, unique instances of types within an
/// MLIRContext. This class manages all creation and uniquing of types.
struct TypeUniquer {
  /// Get an uniqued instance of a type T.
  template <typename T, typename... Args>
  static T get(MLIRContext *ctx, unsigned kind, Args &&... args) {
    return ctx->getTypeUniquer().get<typename T::ImplType>(
        [&](TypeStorage *storage) {
          storage->initialize(AbstractType::lookup(T::getTypeID(), ctx));
        },
        kind, std::forward<Args>(args)...);
  }

  /// Change the mutable component of the given type instance in the provided
  /// context.
  template <typename ImplType, typename... Args>
  static LogicalResult mutate(MLIRContext *ctx, ImplType *impl,
                              Args &&...args) {
    assert(impl && "cannot mutate null type");
    return ctx->getTypeUniquer().mutate(impl, std::forward<Args>(args)...);
  }
};
} // namespace detail

} // end namespace mlir

#endif
