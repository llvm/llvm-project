//===- SideEffectInterfaces.h - SideEffect in MLIR --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains traits, interfaces, and utilities for defining and
// querying the side effects of an operation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_SIDEEFFECTINTERFACES_H
#define MLIR_INTERFACES_SIDEEFFECTINTERFACES_H

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace SideEffects {
//===----------------------------------------------------------------------===//
// Effects
//===----------------------------------------------------------------------===//

/// This class represents a base class for a specific effect type.
class Effect {
public:
  /// This base class is used for derived effects that are non-parametric.
  template <typename DerivedEffect, typename BaseEffect = Effect>
  class Base : public BaseEffect {
  public:
    using BaseT = Base<DerivedEffect>;

    /// Return the unique identifier for the base effects class.
    static TypeID getEffectID() { return TypeID::get<DerivedEffect>(); }

    /// 'classof' used to support llvm style cast functionality.
    static bool classof(const ::mlir::SideEffects::Effect *effect) {
      return effect->getEffectID() == BaseT::getEffectID();
    }

    /// Returns a unique instance for the derived effect class.
    static DerivedEffect *get() {
      return BaseEffect::template get<DerivedEffect>();
    }
    using BaseEffect::get;

  protected:
    Base() : BaseEffect(BaseT::getEffectID()) {}
  };

  /// Return the unique identifier for the base effects class.
  TypeID getEffectID() const { return id; }

  /// Returns a unique instance for the given effect class.
  template <typename DerivedEffect>
  static DerivedEffect *get() {
    static_assert(std::is_base_of<Effect, DerivedEffect>::value,
                  "expected DerivedEffect to inherit from Effect");

    static DerivedEffect instance;
    return &instance;
  }

protected:
  Effect(TypeID id) : id(id) {}

private:
  /// The id of the derived effect class.
  TypeID id;
};

//===----------------------------------------------------------------------===//
// Resources
//===----------------------------------------------------------------------===//

/// This class represents a specific resource that an effect applies to. This
/// class represents an abstract interface for a given resource.
class Resource {
public:
  virtual ~Resource() = default;

  /// This base class is used for derived effects that are non-parametric.
  template <typename DerivedResource, typename BaseResource = Resource>
  class Base : public BaseResource {
  public:
    using BaseT = Base<DerivedResource>;

    /// Returns a unique instance for the given effect class.
    static DerivedResource *get() {
      static DerivedResource instance;
      return &instance;
    }

    /// Return the unique identifier for the base resource class.
    static TypeID getResourceID() { return TypeID::get<DerivedResource>(); }

    /// 'classof' used to support llvm style cast functionality.
    static bool classof(const Resource *resource) {
      return resource->getResourceID() == BaseT::getResourceID();
    }

  protected:
    Base() : BaseResource(BaseT::getResourceID()){};
  };

  /// Return the unique identifier for the base resource class.
  TypeID getResourceID() const { return id; }

  /// Return a string name of the resource.
  virtual StringRef getName() = 0;

protected:
  Resource(TypeID id) : id(id) {}

private:
  /// The id of the derived resource class.
  TypeID id;
};

/// A conservative default resource kind.
struct DefaultResource : public Resource::Base<DefaultResource> {
  StringRef getName() final { return "<Default>"; }
};

/// An automatic allocation-scope resource that is valid in the context of a
/// parent AutomaticAllocationScope trait.
struct AutomaticAllocationScopeResource
    : public Resource::Base<AutomaticAllocationScopeResource> {
  StringRef getName() final { return "AutomaticAllocationScope"; }
};

/// This class represents a specific instance of an effect. It contains the
/// effect being applied, a resource that corresponds to where the effect is
/// applied, and an optional symbol reference or value(either operand, result,
/// or region entry argument) that the effect is applied to, and an optional
/// parameters attribute further specifying the details of the effect.
template <typename EffectT>
class EffectInstance {
public:
  EffectInstance(EffectT *effect, Resource *resource = DefaultResource::get())
      : effect(effect), resource(resource) {}
  EffectInstance(EffectT *effect, Value value,
                 Resource *resource = DefaultResource::get())
      : effect(effect), resource(resource), value(value) {}
  EffectInstance(EffectT *effect, SymbolRefAttr symbol,
                 Resource *resource = DefaultResource::get())
      : effect(effect), resource(resource), value(symbol) {}
  EffectInstance(EffectT *effect, Attribute parameters,
                 Resource *resource = DefaultResource::get())
      : effect(effect), resource(resource), parameters(parameters) {}
  EffectInstance(EffectT *effect, Value value, Attribute parameters,
                 Resource *resource = DefaultResource::get())
      : effect(effect), resource(resource), value(value),
        parameters(parameters) {}
  EffectInstance(EffectT *effect, SymbolRefAttr symbol, Attribute parameters,
                 Resource *resource = DefaultResource::get())
      : effect(effect), resource(resource), value(symbol),
        parameters(parameters) {}

  /// Return the effect being applied.
  EffectT *getEffect() const { return effect; }

  /// Return the value the effect is applied on, or nullptr if there isn't a
  /// known value being affected.
  Value getValue() const { return value ? value.dyn_cast<Value>() : Value(); }

  /// Return the symbol reference the effect is applied on, or nullptr if there
  /// isn't a known smbol being affected.
  SymbolRefAttr getSymbolRef() const {
    return value ? value.dyn_cast<SymbolRefAttr>() : SymbolRefAttr();
  }

  /// Return the resource that the effect applies to.
  Resource *getResource() const { return resource; }

  /// Return the parameters of the effect, if any.
  Attribute getParameters() const { return parameters; }

private:
  /// The specific effect being applied.
  EffectT *effect;

  /// The resource that the given value resides in.
  Resource *resource;

  /// The Symbol or Value that the effect applies to. This is optionally null.
  PointerUnion<SymbolRefAttr, Value> value;

  /// Additional parameters of the effect instance. An attribute is used for
  /// type-safe structured storage and context-based uniquing. Concrete effects
  /// can use this at their convenience. This is optionally null.
  Attribute parameters;
};
} // namespace SideEffects

//===----------------------------------------------------------------------===//
// SideEffect Traits
//===----------------------------------------------------------------------===//

namespace OpTrait {
/// This trait indicates that the side effects of an operation includes the
/// effects of operations nested within its regions. If the operation has no
/// derived effects interfaces, the operation itself can be assumed to have no
/// side effects.
template <typename ConcreteType>
class HasRecursiveSideEffects
    : public TraitBase<ConcreteType, HasRecursiveSideEffects> {};
} // namespace OpTrait

//===----------------------------------------------------------------------===//
// Operation Memory-Effect Modeling
//===----------------------------------------------------------------------===//

namespace MemoryEffects {
/// This class represents the base class used for memory effects.
struct Effect : public SideEffects::Effect {
  using SideEffects::Effect::Effect;

  /// A base class for memory effects that provides helper utilities.
  template <typename DerivedEffect>
  using Base = SideEffects::Effect::Base<DerivedEffect, Effect>;

  static bool classof(const SideEffects::Effect *effect);
};
using EffectInstance = SideEffects::EffectInstance<Effect>;

/// The following effect indicates that the operation allocates from some
/// resource. An 'allocate' effect implies only allocation of the resource, and
/// not any visible mutation or dereference.
struct Allocate : public Effect::Base<Allocate> {};

/// The following effect indicates that the operation frees some resource that
/// has been allocated. An 'allocate' effect implies only de-allocation of the
/// resource, and not any visible allocation, mutation or dereference.
struct Free : public Effect::Base<Free> {};

/// The following effect indicates that the operation reads from some resource.
/// A 'read' effect implies only dereferencing of the resource, and not any
/// visible mutation.
struct Read : public Effect::Base<Read> {};

/// The following effect indicates that the operation writes to some resource. A
/// 'write' effect implies only mutating a resource, and not any visible
/// dereference or read.
struct Write : public Effect::Base<Write> {};
} // namespace MemoryEffects

//===----------------------------------------------------------------------===//
// SideEffect Utilities
//===----------------------------------------------------------------------===//

/// Returns true if `op` has only an effect of type `EffectTy` (and of no other
/// type) on `value`.
template <typename EffectTy>
bool hasSingleEffect(Operation *op, Value value);

/// Return true if the given operation is unused, and has no side effects on
/// memory that prevent erasing.
bool isOpTriviallyDead(Operation *op);

/// Return true if the given operation would be dead if unused, and has no side
/// effects on memory that would prevent erasing. This is equivalent to checking
/// `isOpTriviallyDead` if `op` was unused.
bool wouldOpBeTriviallyDead(Operation *op);

} // namespace mlir

//===----------------------------------------------------------------------===//
// SideEffect Interfaces
//===----------------------------------------------------------------------===//

/// Include the definitions of the side effect interfaces.
#include "mlir/Interfaces/SideEffectInterfaces.h.inc"

#endif // MLIR_INTERFACES_SIDEEFFECTINTERFACES_H
