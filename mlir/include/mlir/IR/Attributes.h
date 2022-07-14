//===- Attributes.h - MLIR Attribute Classes --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_ATTRIBUTES_H
#define MLIR_IR_ATTRIBUTES_H

#include "mlir/IR/AttributeSupport.h"
#include "llvm/Support/PointerLikeTypeTraits.h"

namespace mlir {
class StringAttr;

/// Attributes are known-constant values of operations.
///
/// Instances of the Attribute class are references to immortal key-value pairs
/// with immutable, uniqued keys owned by MLIRContext. As such, an Attribute is
/// a thin wrapper around an underlying storage pointer. Attributes are usually
/// passed by value.
class Attribute {
public:
  /// Utility class for implementing attributes.
  template <typename ConcreteType, typename BaseType, typename StorageType,
            template <typename T> class... Traits>
  using AttrBase = detail::StorageUserBase<ConcreteType, BaseType, StorageType,
                                           detail::AttributeUniquer, Traits...>;

  using ImplType = AttributeStorage;
  using ValueType = void;
  using AbstractTy = AbstractAttribute;

  constexpr Attribute() {}
  /* implicit */ Attribute(const ImplType *impl)
      : impl(const_cast<ImplType *>(impl)) {}

  Attribute(const Attribute &other) = default;
  Attribute &operator=(const Attribute &other) = default;

  bool operator==(Attribute other) const { return impl == other.impl; }
  bool operator!=(Attribute other) const { return !(*this == other); }
  explicit operator bool() const { return impl; }

  bool operator!() const { return impl == nullptr; }

  template <typename U>
  bool isa() const;
  template <typename First, typename Second, typename... Rest>
  bool isa() const;
  template <typename First, typename... Rest>
  bool isa_and_nonnull() const;
  template <typename U>
  U dyn_cast() const;
  template <typename U>
  U dyn_cast_or_null() const;
  template <typename U>
  U cast() const;

  // Support dyn_cast'ing Attribute to itself.
  static bool classof(Attribute) { return true; }

  /// Return a unique identifier for the concrete attribute type. This is used
  /// to support dynamic type casting.
  TypeID getTypeID() { return impl->getAbstractAttribute().getTypeID(); }

  /// Return the type of this attribute.
  Type getType() const { return impl->getType(); }

  /// Return the context this attribute belongs to.
  MLIRContext *getContext() const;

  /// Get the dialect this attribute is registered to.
  Dialect &getDialect() const {
    return impl->getAbstractAttribute().getDialect();
  }

  /// Print the attribute.
  void print(raw_ostream &os) const;
  void dump() const;

  /// Get an opaque pointer to the attribute.
  const void *getAsOpaquePointer() const { return impl; }
  /// Construct an attribute from the opaque pointer representation.
  static Attribute getFromOpaquePointer(const void *ptr) {
    return Attribute(reinterpret_cast<const ImplType *>(ptr));
  }

  friend ::llvm::hash_code hash_value(Attribute arg);

  /// Returns true if the type was registered with a particular trait.
  template <template <typename T> class Trait>
  bool hasTrait() {
    return getAbstractAttribute().hasTrait<Trait>();
  }

  /// Return the abstract descriptor for this attribute.
  const AbstractTy &getAbstractAttribute() const {
    return impl->getAbstractAttribute();
  }

protected:
  ImplType *impl{nullptr};
};

inline raw_ostream &operator<<(raw_ostream &os, Attribute attr) {
  attr.print(os);
  return os;
}

template <typename U>
bool Attribute::isa() const {
  assert(impl && "isa<> used on a null attribute.");
  return U::classof(*this);
}

template <typename First, typename Second, typename... Rest>
bool Attribute::isa() const {
  return isa<First>() || isa<Second, Rest...>();
}

template <typename First, typename... Rest>
bool Attribute::isa_and_nonnull() const {
  return impl && isa<First, Rest...>();
}

template <typename U>
U Attribute::dyn_cast() const {
  return isa<U>() ? U(impl) : U(nullptr);
}
template <typename U>
U Attribute::dyn_cast_or_null() const {
  return (impl && isa<U>()) ? U(impl) : U(nullptr);
}
template <typename U>
U Attribute::cast() const {
  assert(isa<U>());
  return U(impl);
}

inline ::llvm::hash_code hash_value(Attribute arg) {
  return DenseMapInfo<const Attribute::ImplType *>::getHashValue(arg.impl);
}

//===----------------------------------------------------------------------===//
// NamedAttribute
//===----------------------------------------------------------------------===//

/// NamedAttribute represents a combination of a name and an Attribute value.
class NamedAttribute {
public:
  NamedAttribute(StringAttr name, Attribute value);

  /// Return the name of the attribute.
  StringAttr getName() const;

  /// Return the dialect of the name of this attribute, if the name is prefixed
  /// by a dialect namespace. For example, `llvm.fast_math` would return the
  /// LLVM dialect (if it is loaded). Returns nullptr if the dialect isn't
  /// loaded, or if the name is not prefixed by a dialect namespace.
  Dialect *getNameDialect() const;

  /// Return the value of the attribute.
  Attribute getValue() const { return value; }

  /// Set the name of this attribute.
  void setName(StringAttr newName);

  /// Set the value of this attribute.
  void setValue(Attribute newValue) {
    assert(value && "expected valid attribute value");
    value = newValue;
  }

  /// Compare this attribute to the provided attribute, ordering by name.
  bool operator<(const NamedAttribute &rhs) const;
  /// Compare this attribute to the provided string, ordering by name.
  bool operator<(StringRef rhs) const;

  bool operator==(const NamedAttribute &rhs) const {
    return name == rhs.name && value == rhs.value;
  }
  bool operator!=(const NamedAttribute &rhs) const { return !(*this == rhs); }

private:
  NamedAttribute(Attribute name, Attribute value) : name(name), value(value) {}

  /// Allow access to internals to enable hashing.
  friend ::llvm::hash_code hash_value(const NamedAttribute &arg);
  friend DenseMapInfo<NamedAttribute>;

  /// The name of the attribute. This is represented as a StringAttr, but
  /// type-erased to Attribute in the field.
  Attribute name;
  /// The value of the attribute.
  Attribute value;
};

inline ::llvm::hash_code hash_value(const NamedAttribute &arg) {
  using AttrPairT = std::pair<Attribute, Attribute>;
  return DenseMapInfo<AttrPairT>::getHashValue(AttrPairT(arg.name, arg.value));
}

//===----------------------------------------------------------------------===//
// AttributeTraitBase
//===----------------------------------------------------------------------===//

namespace AttributeTrait {
/// This class represents the base of an attribute trait.
template <typename ConcreteType, template <typename> class TraitType>
using TraitBase = detail::StorageUserTraitBase<ConcreteType, TraitType>;
} // namespace AttributeTrait

//===----------------------------------------------------------------------===//
// AttributeInterface
//===----------------------------------------------------------------------===//

/// This class represents the base of an attribute interface. See the definition
/// of `detail::Interface` for requirements on the `Traits` type.
template <typename ConcreteType, typename Traits>
class AttributeInterface
    : public detail::Interface<ConcreteType, Attribute, Traits, Attribute,
                               AttributeTrait::TraitBase> {
public:
  using Base = AttributeInterface<ConcreteType, Traits>;
  using InterfaceBase = detail::Interface<ConcreteType, Attribute, Traits,
                                          Attribute, AttributeTrait::TraitBase>;
  using InterfaceBase::InterfaceBase;

private:
  /// Returns the impl interface instance for the given type.
  static typename InterfaceBase::Concept *getInterfaceFor(Attribute attr) {
    return attr.getAbstractAttribute().getInterface<ConcreteType>();
  }

  /// Allow access to 'getInterfaceFor'.
  friend InterfaceBase;
};

//===----------------------------------------------------------------------===//
// Core AttributeTrait
//===----------------------------------------------------------------------===//

/// This trait is used to determine if an attribute is mutable or not. It is
/// attached on an attribute if the corresponding ImplType defines a `mutate`
/// function with proper signature.
namespace AttributeTrait {
template <typename ConcreteType>
using IsMutable = detail::StorageUserTrait::IsMutable<ConcreteType>;
} // namespace AttributeTrait

} // namespace mlir.

namespace llvm {

// Attribute hash just like pointers.
template <>
struct DenseMapInfo<mlir::Attribute> {
  static mlir::Attribute getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::Attribute(static_cast<mlir::Attribute::ImplType *>(pointer));
  }
  static mlir::Attribute getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::Attribute(static_cast<mlir::Attribute::ImplType *>(pointer));
  }
  static unsigned getHashValue(mlir::Attribute val) {
    return mlir::hash_value(val);
  }
  static bool isEqual(mlir::Attribute LHS, mlir::Attribute RHS) {
    return LHS == RHS;
  }
};
template <typename T>
struct DenseMapInfo<
    T, std::enable_if_t<std::is_base_of<mlir::Attribute, T>::value &&
                        !mlir::detail::IsInterface<T>::value>>
    : public DenseMapInfo<mlir::Attribute> {
  static T getEmptyKey() {
    const void *pointer = llvm::DenseMapInfo<const void *>::getEmptyKey();
    return T::getFromOpaquePointer(pointer);
  }
  static T getTombstoneKey() {
    const void *pointer = llvm::DenseMapInfo<const void *>::getTombstoneKey();
    return T::getFromOpaquePointer(pointer);
  }
};

/// Allow LLVM to steal the low bits of Attributes.
template <>
struct PointerLikeTypeTraits<mlir::Attribute> {
  static inline void *getAsVoidPointer(mlir::Attribute attr) {
    return const_cast<void *>(attr.getAsOpaquePointer());
  }
  static inline mlir::Attribute getFromVoidPointer(void *ptr) {
    return mlir::Attribute::getFromOpaquePointer(ptr);
  }
  static constexpr int NumLowBitsAvailable = llvm::PointerLikeTypeTraits<
      mlir::AttributeStorage *>::NumLowBitsAvailable;
};

template <>
struct DenseMapInfo<mlir::NamedAttribute> {
  static mlir::NamedAttribute getEmptyKey() {
    auto emptyAttr = llvm::DenseMapInfo<mlir::Attribute>::getEmptyKey();
    return mlir::NamedAttribute(emptyAttr, emptyAttr);
  }
  static mlir::NamedAttribute getTombstoneKey() {
    auto tombAttr = llvm::DenseMapInfo<mlir::Attribute>::getTombstoneKey();
    return mlir::NamedAttribute(tombAttr, tombAttr);
  }
  static unsigned getHashValue(mlir::NamedAttribute val) {
    return mlir::hash_value(val);
  }
  static bool isEqual(mlir::NamedAttribute lhs, mlir::NamedAttribute rhs) {
    return lhs == rhs;
  }
};

} // namespace llvm

#endif
