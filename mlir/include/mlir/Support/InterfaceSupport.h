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
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/TypeName.h"
#include "llvm/Support/raw_ostream.h"

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
  using InterfaceBase =
      Interface<ConcreteType, ValueT, Traits, BaseType, BaseTrait>;

  Interface(ValueT t = ValueT())
      : BaseType(t), impl(t ? ConcreteType::getInterfaceFor(t) : nullptr) {
    assert((!t || impl) &&
           "instantiating an interface with an unregistered operation");
  }

  /// Support 'classof' by checking if the given object defines the concrete
  /// interface.
  static bool classof(ValueT t) { return ConcreteType::getInterfaceFor(t); }

  /// Define an accessor for the ID of this interface.
  static TypeID getInterfaceID() { return TypeID::get<ConcreteType>(); }

  /// This is a special trait that registers a given interface with an object.
  template <typename ConcreteT>
  struct Trait : public BaseTrait<ConcreteT, Trait> {
    /// Define an accessor for the ID of this interface.
    static TypeID getInterfaceID() { return TypeID::get<ConcreteType>(); }

    /// Provide an accessor to a static instance of the interface model for the
    /// concrete T type.
    /// The implementation is inspired from Sean Parent's concept-based
    /// polymorphism. A key difference is that the set of classes erased is
    /// statically known, which alleviates the need for using dynamic memory
    /// allocation.
    /// We use a zero-sized templated class `Model<ConcreteT>` to emit the
    /// virtual table and generate a singleton object for each instantiation of
    /// this class.
    static Concept &instance() {
      static Model<ConcreteT> singleton;
      return singleton;
    }
  };

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

/// This class provides an efficient mapping between a given `Interface` type,
/// and a particular implementation of its concept.
class InterfaceMap {
public:
  /// Construct an InterfaceMap with the given set of template types. For
  /// convenience given that object trait lists may contain other non-interface
  /// types, not all of the types need to be interfaces. The provided types that
  /// do not represent interfaces are not added to the interface map.
  template <typename... Types> static InterfaceMap get() {
    return InterfaceMap(MapBuilder::create<Types...>());
  }

  /// Returns an instance of the concept object for the given interface if it
  /// was registered to this map, null otherwise.
  template <typename T> typename T::Concept *lookup() const {
    if (!interfaces)
      return nullptr;
    return reinterpret_cast<typename T::Concept *>(
        interfaces->lookup(T::getInterfaceID()));
  }

private:
  /// This struct provides support for building a map of interfaces.
  class MapBuilder {
  public:
    template <typename... Types>
    static std::unique_ptr<llvm::SmallDenseMap<TypeID, void *>> create() {
      // Filter the provided types for those that are interfaces. This reduces
      // the amount of maps that are generated.
      return createImpl((typename FilterTypes<detect_get_interface_id,
                                              Types...>::type *)nullptr);
    }

  private:
    /// Trait to check if T provides a static 'getInterfaceID' method.
    template <typename T, typename... Args>
    using has_get_interface_id = decltype(T::getInterfaceID());
    template <typename T>
    using detect_get_interface_id = llvm::is_detected<has_get_interface_id, T>;

    template <typename... Ts>
    static std::unique_ptr<llvm::SmallDenseMap<TypeID, void *>>
    createImpl(std::tuple<Ts...> *) {
      // Only create an instance of the map if there are any interface types.
      if (sizeof...(Ts) == 0)
        return std::unique_ptr<llvm::SmallDenseMap<TypeID, void *>>();

      auto map = std::make_unique<llvm::SmallDenseMap<TypeID, void *>>();
      (void)std::initializer_list<int>{
          0, (map->try_emplace(Ts::getInterfaceID(), &Ts::instance()), 0)...};
      return map;
    }
  };

private:
  InterfaceMap(std::unique_ptr<llvm::SmallDenseMap<TypeID, void *>> interfaces)
      : interfaces(std::move(interfaces)) {}

  /// The internal map of interfaces. This is constructed statically for each
  /// set of interfaces.
  std::unique_ptr<llvm::SmallDenseMap<TypeID, void *>> interfaces;
};

} // end namespace detail
} // end namespace mlir

#endif
