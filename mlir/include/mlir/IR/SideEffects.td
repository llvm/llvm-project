//===-- SideEffects.td - Side Effect Interfaces ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a set of interfaces that can be used to define information
// about what effects are applied by an operation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_SIDEEFFECTS
#define MLIR_IR_SIDEEFFECTS

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// EffectOpInterface
//===----------------------------------------------------------------------===//

// A base interface used to query information about the side effects applied to
// an operation. This template class takes the name of the derived interface
// class, as well as the name of the base effect class.
class EffectOpInterfaceBase<string name, string baseEffect>
    : OpInterface<name> {
  let methods = [
    InterfaceMethod<[{
        Collects all of the operation's effects into `effects`.
      }],
      "void", "getEffects",
         (ins "SmallVectorImpl<SideEffects::EffectInstance<"
              # baseEffect # ">> &":$effects)
    >,
    InterfaceMethod<[{
        Collects all of the operation's effects into `effects`.
      }],
      "void", "getEffectsOnValue",
         (ins "Value":$value,
              "SmallVectorImpl<SideEffects::EffectInstance<"
              # baseEffect # ">> &":$effects), [{
          op.getEffects(effects);
          llvm::erase_if(effects, [&](auto &it) {
            return it.getValue() != value;
          });
       }]
    >,
    InterfaceMethod<[{
        Collects all of the effects that are exhibited by this operation on the
        given resource and place them in 'effects'.
      }],
      "void", "getEffectsOnResource",
         (ins "SideEffects::Resource *":$resource,
              "SmallVectorImpl<SideEffects::EffectInstance<"
              # baseEffect # ">> &":$effects), [{
          op.getEffects(effects);
          llvm::erase_if(effects, [&](auto &it) {
            return it.getResource() != resource;
          });
       }]
    >
  ];

  let extraClassDeclaration = [{
    /// Collect all of the effect instances that correspond to the given
    /// `Effect` and place them in 'effects'.
    template <typename Effect> void getEffects(
      SmallVectorImpl<SideEffects::EffectInstance<
                                              }] # baseEffect # [{>> &effects) {
      getEffects(effects);
      llvm::erase_if(effects, [&](auto &it) {
        return !llvm::isa<Effect>(it.getEffect());
      });
    }

    /// Returns true if this operation exhibits the given effect.
    template <typename Effect> bool hasEffect() {
      SmallVector<SideEffects::EffectInstance<}] # baseEffect # [{>, 4> effects;
      getEffects(effects);
      return llvm::any_of(effects, [](const auto &it) {
        return llvm::isa<Effect>(it.getEffect());
      });
    }

    /// Returns if this operation only has the given effect.
    template <typename Effect> bool onlyHasEffect() {
      SmallVector<SideEffects::EffectInstance<}] # baseEffect # [{>, 4> effects;
      getEffects(effects);
      return !effects.empty() && llvm::all_of(effects, [](const auto &it) {
        return isa<Effect>(it.getEffect());
      });
    }

    /// Returns if this operation has no effects.
    bool hasNoEffect() {
      SmallVector<SideEffects::EffectInstance<}] # baseEffect # [{>, 4> effects;
      getEffects(effects);
      return effects.empty();
    }
  }];

  // The base effect name of this interface.
  string baseEffectName = baseEffect;
}

// This class is the general base side effect class. This is used by derived
// effect interfaces to define their effects.
class SideEffect<EffectOpInterfaceBase interface, string effectName,
                 string resourceName> {
  /// The parent interface that the effect belongs to.
  string interfaceTrait = interface.trait;

  /// The name of the base effects class.
  string baseEffect = interface.baseEffectName;

  /// The derived effect that is being applied.
  string effect = effectName;

  /// The resource that the effect is being applied to.
  string resource = resourceName;
}

// This class is the base used for specifying effects applied to an operation.
class SideEffectsTraitBase<EffectOpInterfaceBase parentInterface,
                           list<SideEffect> staticEffects>
    : OpInterfaceTrait<""> {
  /// The name of the interface trait to use.
  let trait = parentInterface.trait;

  /// The derived effects being applied.
  list<SideEffect> effects = staticEffects;
}

//===----------------------------------------------------------------------===//
// MemoryEffects
//===----------------------------------------------------------------------===//

// This def represents the definition for the memory effects interface. Users
// should generally not use this directly, and should instead use
// `MemoryEffects`.
def MemoryEffectsOpInterface
    : EffectOpInterfaceBase<"MemoryEffectOpInterface",
                            "MemoryEffects::Effect"> {
  let description = [{
    An interface used to query information about the memory effects applied by
    an operation.
  }];
}

// The base class for defining specific memory effects.
class MemoryEffect<string effectName, string resourceName>
  : SideEffect<MemoryEffectsOpInterface, effectName, resourceName>;

// This class represents the trait for memory effects that may be placed on
// operations.
class MemoryEffects<list<MemoryEffect> effects = []>
  : SideEffectsTraitBase<MemoryEffectsOpInterface, effects>;

//===----------------------------------------------------------------------===//
// Effects

// The following effect indicates that the operation allocates from some
// resource. An 'allocate' effect implies only allocation of the resource, and
// not any visible mutation or dereference.
class MemAlloc<string resourceName>
  : MemoryEffect<"MemoryEffects::Allocate", resourceName>;
def MemAlloc : MemAlloc<"">;

// The following effect indicates that the operation frees some resource that
// has been allocated. A 'free' effect implies only de-allocation of the
// resource, and not any visible allocation, mutation or dereference.
class MemFree<string resourceName>
  : MemoryEffect<"MemoryEffects::Free", resourceName>;
def MemFree : MemFree<"">;

// The following effect indicates that the operation reads from some
// resource. A 'read' effect implies only dereferencing of the resource, and
// not any visible mutation.
class MemRead<string resourceName>
  : MemoryEffect<"MemoryEffects::Read", resourceName>;
def MemRead : MemRead<"">;

// The following effect indicates that the operation writes to some
// resource. A 'write' effect implies only mutating a resource, and not any
// visible dereference or read.
class MemWrite<string resourceName>
  : MemoryEffect<"MemoryEffects::Write", resourceName>;
def MemWrite : MemWrite<"">;

#endif // MLIR_IR_SIDEEFFECTS
