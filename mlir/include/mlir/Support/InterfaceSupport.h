//===- InterfaceSupport.h - MLIR Interface Support Classes ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines several support classes for defining interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_INTERFACESUPPORT_H
#define MLIR_SUPPORT_INTERFACESUPPORT_H

#include "mlir/Support/TypeID.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/TypeName.h"

namespace mlir {
namespace detail {
//===----------------------------------------------------------------------===//
// Interface
//===----------------------------------------------------------------------===//

/// This class represents an abstract interface. An interface is a simplified
/// mechanism for attaching concept based polymorphism to a class hierarchy. An
/// interface is comprised of two components:
/// * The derived interface class: This is what users interact with, and invoke
///   methods on.
/// * An interface `Trait` class: This is the class that is attached to the
///   object implementing the interface. It is the mechanism with which models
///   are specialized.
///
/// Derived interfaces types must provide the following template types:
/// * ConcreteType: The CRTP derived type.
/// * ValueT: The opaque type the derived interface operates on. For example
///           `Operation*` for operation interfaces, or `Attribute` for
///           attribute interfaces.
/// * Traits: A class that contains definitions for a 'Concept' and a 'Model'
///           class. The 'Concept' class defines an abstract virtual interface,
///           where as the 'Model' class implements this interface for a
///           specific derived T type. Both of these classes *must* not contain
///           non-static data. A simple example is shown below:
///
/// ```c++
///    struct ExampleInterfaceTraits {
///      struct Concept {
///        virtual unsigned getNumInputs(T t) const = 0;
///      };
///      template <typename DerivedT> class Model {
///        unsigned getNumInputs(T t) const final {
///          return cast<DerivedT>(t).getNumInputs();
///        }
///      };
///    };
/// ```
///
/// * BaseType: A desired base type for the interface. This is a class that
///             provides that provides specific functionality for the `ValueT`
///             value. For instance the specific `Op` that will wrap the
///             `Operation*` for an `OpInterface`.
/// * BaseTrait: The base type for the interface trait. This is the base class
///              to use for the interface trait that will be attached to each
///              instance of `ValueT` that implements this interface.
///
template <typename ConcreteType, typename ValueT, typename Traits,
          typename BaseType,
          template <typename, template <typename> class> class BaseTrait>
class Interface : public BaseType {
public:
  using Concept = typename Traits::Concept;
  template <typename T> using Model = typename Traits::template Model<T>;
  template <typename T>
  using FallbackModel = typename Traits::template FallbackModel<T>;
  using InterfaceBase =
      Interface<ConcreteType, ValueT, Traits, BaseType, BaseTrait>;
  template <typename T, typename U>
  using ExternalModel = typename Traits::template ExternalModel<T, U>;

  /// This is a special trait that registers a given interface with an object.
  template <typename ConcreteT>
  struct Trait : public BaseTrait<ConcreteT, Trait> {
    using ModelT = Model<ConcreteT>;

    /// Define an accessor for the ID of this interface.
    static TypeID getInterfaceID() { return TypeID::get<ConcreteType>(); }
  };

  /// Construct an interface from an instance of the value type.
  Interface(ValueT t = ValueT())
      : BaseType(t), impl(t ? ConcreteType::getInterfaceFor(t) : nullptr) {
    assert((!t || impl) && "expected value to provide interface instance");
  }

  /// Construct an interface instance from a type that implements this
  /// interface's trait.
  template <typename T, typename std::enable_if_t<
                            std::is_base_of<Trait<T>, T>::value> * = nullptr>
  Interface(T t)
      : BaseType(t), impl(t ? ConcreteType::getInterfaceFor(t) : nullptr) {
    assert((!t || impl) && "expected value to provide interface instance");
  }

  /// Support 'classof' by checking if the given object defines the concrete
  /// interface.
  static bool classof(ValueT t) { return ConcreteType::getInterfaceFor(t); }

  /// Define an accessor for the ID of this interface.
  static TypeID getInterfaceID() { return TypeID::get<ConcreteType>(); }

protected:
  /// Get the raw concept in the correct derived concept type.
  const Concept *getImpl() const { return impl; }
  Concept *getImpl() { return impl; }

private:
  /// A pointer to the impl concept object.
  Concept *impl;
};

//===----------------------------------------------------------------------===//
// InterfaceMap
//===----------------------------------------------------------------------===//

/// Utility to filter a given sequence of types base upon a predicate.
template <bool>
struct FilterTypeT {
  template <class E>
  using type = std::tuple<E>;
};
template <>
struct FilterTypeT<false> {
  template <class E>
  using type = std::tuple<>;
};
template <template <class> class Pred, class... Es>
struct FilterTypes {
  using type = decltype(std::tuple_cat(
      std::declval<
          typename FilterTypeT<Pred<Es>::value>::template type<Es>>()...));
};

namespace {
/// Type trait indicating whether all template arguments are
/// trivially-destructible.
template <typename... Args>
struct all_trivially_destructible;

template <typename Arg, typename... Args>
struct all_trivially_destructible<Arg, Args...> {
  static constexpr const bool value =
      std::is_trivially_destructible<Arg>::value &&
      all_trivially_destructible<Args...>::value;
};

template <>
struct all_trivially_destructible<> {
  static constexpr const bool value = true;
};
} // namespace

/// This class provides an efficient mapping between a given `Interface` type,
/// and a particular implementation of its concept.
class InterfaceMap {
  /// Trait to check if T provides a static 'getInterfaceID' method.
  template <typename T, typename... Args>
  using has_get_interface_id = decltype(T::getInterfaceID());
  template <typename T>
  using detect_get_interface_id = llvm::is_detected<has_get_interface_id, T>;
  template <typename... Types>
  using num_interface_types = typename std::tuple_size<
      typename FilterTypes<detect_get_interface_id, Types...>::type>;

public:
  InterfaceMap(InterfaceMap &&) = default;
  ~InterfaceMap() {
    for (auto &it : interfaces)
      free(it.second);
  }

  /// Construct an InterfaceMap with the given set of template types. For
  /// convenience given that object trait lists may contain other non-interface
  /// types, not all of the types need to be interfaces. The provided types that
  /// do not represent interfaces are not added to the interface map.
  template <typename... Types>
  static std::enable_if_t<num_interface_types<Types...>::value != 0,
                          InterfaceMap>
  get() {
    // Filter the provided types for those that are interfaces.
    using FilteredTupleType =
        typename FilterTypes<detect_get_interface_id, Types...>::type;
    return getImpl((FilteredTupleType *)nullptr);
  }

  template <typename... Types>
  static std::enable_if_t<num_interface_types<Types...>::value == 0,
                          InterfaceMap>
  get() {
    return InterfaceMap();
  }

  /// Returns an instance of the concept object for the given interface if it
  /// was registered to this map, null otherwise.
  template <typename T> typename T::Concept *lookup() const {
    return reinterpret_cast<typename T::Concept *>(lookup(T::getInterfaceID()));
  }

  /// Returns true if the interface map contains an interface for the given id.
  bool contains(TypeID interfaceID) const { return lookup(interfaceID); }

  /// Create an InterfaceMap given with the implementation of the interfaces.
  /// The use of this constructor is in general discouraged in favor of
  /// 'InterfaceMap::get<InterfaceA, ...>()'.
  InterfaceMap(MutableArrayRef<std::pair<TypeID, void *>> elements)
      : interfaces(elements.begin(), elements.end()) {
    llvm::sort(interfaces, [](const auto &lhs, const auto &rhs) {
      return compare(lhs.first, rhs.first);
    });
  }

  /// Insert the given models as implementations of the corresponding interfaces
  /// for the concrete attribute class.
  template <typename... IfaceModels>
  void insert() {
    static_assert(all_trivially_destructible<IfaceModels...>::value,
                  "interface models must be trivially destructible");
    std::pair<TypeID, void *> elements[] = {
        std::make_pair(IfaceModels::Interface::getInterfaceID(),
                       new (malloc(sizeof(IfaceModels))) IfaceModels())...};
    insert(elements);
  }

private:
  /// Compare two TypeID instances by comparing the underlying pointer.
  static bool compare(TypeID lhs, TypeID rhs) {
    return lhs.getAsOpaquePointer() < rhs.getAsOpaquePointer();
  }

  InterfaceMap() = default;

  void insert(ArrayRef<std::pair<TypeID, void *>> elements);

  template <typename... Ts>
  static InterfaceMap getImpl(std::tuple<Ts...> *) {
    std::pair<TypeID, void *> elements[] = {std::make_pair(
        Ts::getInterfaceID(),
        new (malloc(sizeof(typename Ts::ModelT))) typename Ts::ModelT())...};
    return InterfaceMap(elements);
  }

  /// Returns an instance of the concept object for the given interface id if it
  /// was registered to this map, null otherwise.
  void *lookup(TypeID id) const {
    auto it = llvm::lower_bound(interfaces, id, [](const auto &it, TypeID id) {
      return compare(it.first, id);
    });
    return (it != interfaces.end() && it->first == id) ? it->second : nullptr;
  }

  /// A list of interface instances, sorted by TypeID.
  SmallVector<std::pair<TypeID, void *>> interfaces;
};

} // end namespace detail
} // end namespace mlir

#endif
