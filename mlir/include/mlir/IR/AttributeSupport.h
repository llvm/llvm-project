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
#include "llvm/ADT/Twine.h"

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

  /// Returns true if the attribute has the interface with the given ID
  /// registered.
  bool hasInterface(TypeID interfaceID) const {
    return interfaceMap.contains(interfaceID);
  }

  /// Return the unique identifier representing the concrete attribute class.
  TypeID getTypeID() const { return typeID; }

private:
  AbstractAttribute(Dialect &dialect, detail::InterfaceMap &&interfaceMap,
                    TypeID typeID)
      : dialect(dialect), interfaceMap(std::move(interfaceMap)),
        typeID(typeID) {}

  /// Give StorageUserBase access to the mutable lookup.
  template <typename ConcreteT, typename BaseT, typename StorageT,
            typename UniquerT, template <typename T> class... Traits>
  friend class detail::StorageUserBase;

  /// Look up the specified abstract attribute in the MLIRContext and return a
  /// (mutable) pointer to it. Return a null pointer if the attribute could not
  /// be found in the context.
  static AbstractAttribute *lookupMutable(TypeID typeID, MLIRContext *context);

  /// This is the dialect that this attribute was registered to.
  const Dialect &dialect;

  /// This is a collection of the interfaces registered to this attribute.
  detail::InterfaceMap interfaceMap;

  /// The unique identifier of the derived Attribute class.
  const TypeID typeID;
};

//===----------------------------------------------------------------------===//
// AttributeStorage
//===----------------------------------------------------------------------===//

namespace detail {
class AttributeUniquer;
} // end namespace detail

/// Base storage class appearing in an attribute. Derived storage classes should
/// only be constructed within the context of the AttributeUniquer.
class alignas(8) AttributeStorage : public StorageUniquer::BaseStorage {
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
  /// Get an uniqued instance of a parametric attribute T.
  template <typename T, typename... Args>
  static typename std::enable_if_t<
      !std::is_same<typename T::ImplType, AttributeStorage>::value, T>
  get(MLIRContext *ctx, Args &&...args) {
#ifndef NDEBUG
    if (!ctx->getAttributeUniquer().isParametricStorageInitialized(
            T::getTypeID()))
      llvm::report_fatal_error(llvm::Twine("can't create Attribute '") +
                               llvm::getTypeName<T>() +
                               "' because storage uniquer isn't initialized: "
                               "the dialect was likely not loaded.");
#endif
    return ctx->getAttributeUniquer().get<typename T::ImplType>(
        [ctx](AttributeStorage *storage) {
          initializeAttributeStorage(storage, ctx, T::getTypeID());
        },
        T::getTypeID(), std::forward<Args>(args)...);
  }
  /// Get an uniqued instance of a singleton attribute T.
  template <typename T>
  static typename std::enable_if_t<
      std::is_same<typename T::ImplType, AttributeStorage>::value, T>
  get(MLIRContext *ctx) {
#ifndef NDEBUG
    if (!ctx->getAttributeUniquer().isSingletonStorageInitialized(
            T::getTypeID()))
      llvm::report_fatal_error(llvm::Twine("can't create Attribute '") +
                               llvm::getTypeName<T>() +
                               "' because storage uniquer isn't initialized: "
                               "the dialect was likely not loaded.");
#endif
    return ctx->getAttributeUniquer().get<typename T::ImplType>(T::getTypeID());
  }

  template <typename T, typename... Args>
  static LogicalResult mutate(MLIRContext *ctx, typename T::ImplType *impl,
                              Args &&...args) {
    assert(impl && "cannot mutate null attribute");
    return ctx->getAttributeUniquer().mutate(T::getTypeID(), impl,
                                             std::forward<Args>(args)...);
  }

  /// Register a parametric attribute instance T with the uniquer.
  template <typename T>
  static typename std::enable_if_t<
      !std::is_same<typename T::ImplType, AttributeStorage>::value>
  registerAttribute(MLIRContext *ctx) {
    ctx->getAttributeUniquer()
        .registerParametricStorageType<typename T::ImplType>(T::getTypeID());
  }
  /// Register a singleton attribute instance T with the uniquer.
  template <typename T>
  static typename std::enable_if_t<
      std::is_same<typename T::ImplType, AttributeStorage>::value>
  registerAttribute(MLIRContext *ctx) {
    ctx->getAttributeUniquer()
        .registerSingletonStorageType<typename T::ImplType>(
            T::getTypeID(), [ctx](AttributeStorage *storage) {
              initializeAttributeStorage(storage, ctx, T::getTypeID());
            });
  }

private:
  /// Initialize the given attribute storage instance.
  static void initializeAttributeStorage(AttributeStorage *storage,
                                         MLIRContext *ctx, TypeID attrID);
};
} // namespace detail

} // end namespace mlir

#endif
