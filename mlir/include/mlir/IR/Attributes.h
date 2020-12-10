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
class Identifier;

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

  constexpr Attribute() : impl(nullptr) {}
  /* implicit */ Attribute(const ImplType *impl)
      : impl(const_cast<ImplType *>(impl)) {}

  Attribute(const Attribute &other) = default;
  Attribute &operator=(const Attribute &other) = default;

  bool operator==(Attribute other) const { return impl == other.impl; }
  bool operator!=(Attribute other) const { return !(*this == other); }
  explicit operator bool() const { return impl; }

  bool operator!() const { return impl == nullptr; }

  template <typename U> bool isa() const;
  template <typename First, typename Second, typename... Rest>
  bool isa() const;
  template <typename U> U dyn_cast() const;
  template <typename U> U dyn_cast_or_null() const;
  template <typename U> U cast() const;

  // Support dyn_cast'ing Attribute to itself.
  static bool classof(Attribute) { return true; }

  /// Return a unique identifier for the concrete attribute type. This is used
  /// to support dynamic type casting.
  TypeID getTypeID() { return impl->getAbstractAttribute().getTypeID(); }

  /// Return the type of this attribute.
  Type getType() const;

  /// Return the context this attribute belongs to.
  MLIRContext *getContext() const;

  /// Get the dialect this attribute is registered to.
  Dialect &getDialect() const;

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

  /// Return the abstract descriptor for this attribute.
  const AbstractAttribute &getAbstractAttribute() const {
    return impl->getAbstractAttribute();
  }

protected:
  ImplType *impl;
};

inline raw_ostream &operator<<(raw_ostream &os, Attribute attr) {
  attr.print(os);
  return os;
}

template <typename U> bool Attribute::isa() const {
  assert(impl && "isa<> used on a null attribute.");
  return U::classof(*this);
}

template <typename First, typename Second, typename... Rest>
bool Attribute::isa() const {
  return isa<First>() || isa<Second, Rest...>();
}

template <typename U> U Attribute::dyn_cast() const {
  return isa<U>() ? U(impl) : U(nullptr);
}
template <typename U> U Attribute::dyn_cast_or_null() const {
  return (impl && isa<U>()) ? U(impl) : U(nullptr);
}
template <typename U> U Attribute::cast() const {
  assert(isa<U>());
  return U(impl);
}

inline ::llvm::hash_code hash_value(Attribute arg) {
  return ::llvm::hash_value(arg.impl);
}

//===----------------------------------------------------------------------===//
// NamedAttribute
//===----------------------------------------------------------------------===//

/// NamedAttribute is combination of a name, represented by an Identifier, and a
/// value, represented by an Attribute. The attribute pointer should always be
/// non-null.
using NamedAttribute = std::pair<Identifier, Attribute>;

bool operator<(const NamedAttribute &lhs, const NamedAttribute &rhs);
bool operator<(const NamedAttribute &lhs, StringRef rhs);

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

} // end namespace mlir.

namespace llvm {

// Attribute hash just like pointers.
template <> struct DenseMapInfo<mlir::Attribute> {
  static mlir::Attribute getEmptyKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::Attribute(static_cast<mlir::Attribute::ImplType *>(pointer));
  }
  static mlir::Attribute getTombstoneKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::Attribute(static_cast<mlir::Attribute::ImplType *>(pointer));
  }
  static unsigned getHashValue(mlir::Attribute val) {
    return mlir::hash_value(val);
  }
  static bool isEqual(mlir::Attribute LHS, mlir::Attribute RHS) {
    return LHS == RHS;
  }
};

/// Allow LLVM to steal the low bits of Attributes.
template <> struct PointerLikeTypeTraits<mlir::Attribute> {
  static inline void *getAsVoidPointer(mlir::Attribute attr) {
    return const_cast<void *>(attr.getAsOpaquePointer());
  }
  static inline mlir::Attribute getFromVoidPointer(void *ptr) {
    return mlir::Attribute::getFromOpaquePointer(ptr);
  }
  static constexpr int NumLowBitsAvailable = llvm::PointerLikeTypeTraits<
      mlir::AttributeStorage *>::NumLowBitsAvailable;
};

} // namespace llvm

#endif
