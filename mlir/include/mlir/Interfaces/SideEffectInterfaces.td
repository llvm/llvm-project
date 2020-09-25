//===-- SideEffectInterfaces.td - Side Effect Interfaces ------------*- tablegen -*-===//
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

#ifndef MLIR_INTERFACES_SIDEEFFECTS
#define MLIR_INTERFACES_SIDEEFFECTS

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Resource Bindings
//===----------------------------------------------------------------------===//

// A generic resource that can be attached to a general base side effect.
class Resource<string resourceName> {
  /// The resource that the associated effect is being applied to.
  string name = resourceName;
}

// An intrinsic resource that lives in the ::mlir::SideEffects namespace.
class IntrinsicResource<string resourceName> :
  Resource<!strconcat("::mlir::SideEffects::", resourceName)> {
}

// A link to the DefaultResource class.
def DefaultResource : IntrinsicResource<"DefaultResource">;
// A link to the AutomaticAllocationScopeResource class.
def AutomaticAllocationScopeResource :
  IntrinsicResource<"AutomaticAllocationScopeResource">;

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
         (ins "SmallVectorImpl<::mlir::SideEffects::EffectInstance<"
              # baseEffect # ">> &":$effects)
    >,
    InterfaceMethod<[{
        Collects all of the operation's effects into `effects`.
      }],
      "void", "getEffectsOnValue",
         (ins "Value":$value,
              "SmallVectorImpl<::mlir::SideEffects::EffectInstance<"
              # baseEffect # ">> &":$effects), [{
          $_op.getEffects(effects);
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
              "SmallVectorImpl<::mlir::SideEffects::EffectInstance<"
              # baseEffect # ">> &":$effects), [{
          $_op.getEffects(effects);
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
      SmallVectorImpl<::mlir::SideEffects::EffectInstance<
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

    /// Returns true if this operation only has the given effect.
    template <typename Effect> bool onlyHasEffect() {
      SmallVector<SideEffects::EffectInstance<}] # baseEffect # [{>, 4> effects;
      getEffects(effects);
      return !effects.empty() && llvm::all_of(effects, [](const auto &it) {
        return isa<Effect>(it.getEffect());
      });
    }

    /// Returns true if this operation has no effects.
    bool hasNoEffect() {
      SmallVector<::mlir::SideEffects::EffectInstance<}] # baseEffect # [{>, 4> effects;
      getEffects(effects);
      return effects.empty();
    }

    /// Returns true if the given operation has no effects for this interface.
    static bool hasNoEffect(Operation *op) {
      if (auto interface = dyn_cast<}] # name # [{>(op))
        return interface.hasNoEffect();
      return op->hasTrait<::mlir::OpTrait::HasRecursiveSideEffects>();
    }
  }];

  // The base effect name of this interface.
  string baseEffectName = baseEffect;
}

// This class is the general base side effect class. This is used by derived
// effect interfaces to define their effects.
class SideEffect<EffectOpInterfaceBase interface, string effectName,
                 Resource resourceReference> : OpVariableDecorator {
  /// The name of the base effects class.
  string baseEffectName = interface.baseEffectName;

  /// The parent interface that the effect belongs to.
  string interfaceTrait = interface.trait;

  /// The cpp namespace of the interface trait.
  string cppNamespace = interface.cppNamespace;

  /// The derived effect that is being applied.
  string effect = effectName;

  /// The resource that the effect is being applied to.
  string resource = resourceReference.name;
}

// This class is the base used for specifying effects applied to an operation.
class SideEffectsTraitBase<EffectOpInterfaceBase parentInterface,
                           list<SideEffect> staticEffects>
    : OpInterfaceTrait<""> {
  /// The name of the interface trait to use.
  let trait = parentInterface.trait;

  /// The cpp namespace of the interface trait.
  string cppNamespace = parentInterface.cppNamespace;

  /// The name of the base effects class.
  string baseEffectName = parentInterface.baseEffectName;

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
                            "::mlir::MemoryEffects::Effect"> {
  let description = [{
    An interface used to query information about the memory effects applied by
    an operation.
  }];
  let cppNamespace = "::mlir";
}

// The base class for defining specific memory effects.
class MemoryEffect<string effectName, Resource resource>
  : SideEffect<MemoryEffectsOpInterface, effectName, resource>;

// This class represents the trait for memory effects that may be placed on
// operations.
class MemoryEffects<list<MemoryEffect> effects = []>
  : SideEffectsTraitBase<MemoryEffectsOpInterface, effects>;

//===----------------------------------------------------------------------===//
// Effects

// The following effect indicates that the operation allocates from some
// resource. An 'allocate' effect implies only allocation of the resource, and
// not any visible mutation or dereference.
class MemAlloc<Resource resource>
  : MemoryEffect<"MemoryEffects::Allocate", resource>;
def MemAlloc : MemAlloc<DefaultResource>;

// The following effect indicates that the operation frees some resource that
// has been allocated. A 'free' effect implies only de-allocation of the
// resource, and not any visible allocation, mutation or dereference.
class MemFree<Resource resource>
  : MemoryEffect<"MemoryEffects::Free", resource>;
def MemFree : MemFree<DefaultResource>;

// The following effect indicates that the operation reads from some
// resource. A 'read' effect implies only dereferencing of the resource, and
// not any visible mutation.
class MemRead<Resource resource>
  : MemoryEffect<"MemoryEffects::Read", resource>;
def MemRead : MemRead<DefaultResource>;

// The following effect indicates that the operation writes to some
// resource. A 'write' effect implies only mutating a resource, and not any
// visible dereference or read.
class MemWrite<Resource resource>
  : MemoryEffect<"MemoryEffects::Write", resource>;
def MemWrite : MemWrite<DefaultResource>;

//===----------------------------------------------------------------------===//
// Effect Traits
//===----------------------------------------------------------------------===//

// Op has no side effect.
def NoSideEffect : MemoryEffects<[]>;
// Op has recursively computed side effects.
def RecursiveSideEffects : NativeOpTrait<"HasRecursiveSideEffects">;

#endif // MLIR_INTERFACES_SIDEEFFECTS
