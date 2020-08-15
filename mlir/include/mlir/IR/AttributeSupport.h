//===- AttributeSupport.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines support types for registering dialect extended attributes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_ATTRIBUTESUPPORT_H
#define MLIR_IR_ATTRIBUTESUPPORT_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StorageUniquerSupport.h"
#include "llvm/ADT/PointerIntPair.h"

namespace mlir {
class MLIRContext;
class Type;

//===----------------------------------------------------------------------===//
// AbstractAttribute
//===----------------------------------------------------------------------===//

/// This class contains all of the static information common to all instances of
/// a registered Attribute.
class AbstractAttribute {
public:
  /// Look up the specified abstract attribute in the MLIRContext and return a
  /// reference to it.
  static const AbstractAttribute &lookup(TypeID typeID, MLIRContext *context);

  /// This method is used by Dialect objects when they register the list of
  /// attributes they contain.
  template <typename T> static AbstractAttribute get(Dialect &dialect) {
    return AbstractAttribute(dialect, T::getInterfaceMap(), T::getTypeID());
  }

  /// Return the dialect this attribute was registered to.
  Dialect &getDialect() const { return const_cast<Dialect &>(dialect); }

  /// Returns an instance of the concept object for the given interface if it
  /// was registered to this attribute, null otherwise. This should not be used
  /// directly.
  template <typename T> typename T::Concept *getInterface() const {
    return interfaceMap.lookup<T>();
  }

  /// Return the unique identifier representing the concrete attribute class.
  TypeID getTypeID() const { return typeID; }

private:
  AbstractAttribute(Dialect &dialect, detail::InterfaceMap &&interfaceMap,
                    TypeID typeID)
      : dialect(dialect), interfaceMap(std::move(interfaceMap)),
        typeID(typeID) {}

  /// This is the dialect that this attribute was registered to.
  Dialect &dialect;

  /// This is a collection of the interfaces registered to this attribute.
  detail::InterfaceMap interfaceMap;

  /// The unique identifier of the derived Attribute class.
  TypeID typeID;
};

//===----------------------------------------------------------------------===//
// AttributeStorage
//===----------------------------------------------------------------------===//

namespace detail {
class AttributeUniquer;
} // end namespace detail

/// Base storage class appearing in an attribute. Derived storage classes should
/// only be constructed within the context of the AttributeUniquer.
class AttributeStorage : public StorageUniquer::BaseStorage {
  friend detail::AttributeUniquer;
  friend StorageUniquer;

public:
  /// Get the type of this attribute.
  Type getType() const;

  /// Return the abstract descriptor for this attribute.
  const AbstractAttribute &getAbstractAttribute() const {
    assert(abstractAttribute && "Malformed attribute storage object.");
    return *abstractAttribute;
  }

protected:
  /// Construct a new attribute storage instance with the given type.
  /// Note: All attributes require a valid type. If no type is provided here,
  ///       the type of the attribute will automatically default to NoneType
  ///       upon initialization in the uniquer.
  AttributeStorage(Type type);
  AttributeStorage();

  /// Set the type of this attribute.
  void setType(Type type);

  // Set the abstract attribute for this storage instance. This is used by the
  // AttributeUniquer when initializing a newly constructed storage object.
  void initialize(const AbstractAttribute &abstractAttr) {
    abstractAttribute = &abstractAttr;
  }

private:
  /// The abstract descriptor for this attribute.
  const AbstractAttribute *abstractAttribute;

  /// The opaque type of the attribute value.
  const void *type;
};

/// Default storage type for attributes that require no additional
/// initialization or storage.
using DefaultAttributeStorage = AttributeStorage;

//===----------------------------------------------------------------------===//
// AttributeStorageAllocator
//===----------------------------------------------------------------------===//

// This is a utility allocator used to allocate memory for instances of derived
// Attributes.
using AttributeStorageAllocator = StorageUniquer::StorageAllocator;

//===----------------------------------------------------------------------===//
// AttributeUniquer
//===----------------------------------------------------------------------===//
namespace detail {
// A utility class to get, or create, unique instances of attributes within an
// MLIRContext. This class manages all creation and uniquing of attributes.
class AttributeUniquer {
public:
  /// Get an uniqued instance of attribute T.
  template <typename T, typename... Args>
  static T get(MLIRContext *ctx, unsigned kind, Args &&... args) {
    return ctx->getAttributeUniquer().get<typename T::ImplType>(
        T::getTypeID(),
        [ctx](AttributeStorage *storage) {
          initializeAttributeStorage(storage, ctx, T::getTypeID());
        },
        kind, std::forward<Args>(args)...);
  }

  template <typename T, typename... Args>
  static LogicalResult mutate(MLIRContext *ctx, typename T::ImplType *impl,
                              Args &&...args) {
    assert(impl && "cannot mutate null attribute");
    return ctx->getAttributeUniquer().mutate(T::getTypeID(), impl,
                                             std::forward<Args>(args)...);
  }

private:
  /// Initialize the given attribute storage instance.
  static void initializeAttributeStorage(AttributeStorage *storage,
                                         MLIRContext *ctx, TypeID attrID);
};
} // namespace detail

} // end namespace mlir

#endif
