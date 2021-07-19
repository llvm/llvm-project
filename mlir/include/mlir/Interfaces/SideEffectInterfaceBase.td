//===-- SideEffectInterfaceBase.td - Side Effect Base ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains base class definitions for side effect interfaces, i.e.
// the customizable interfaces that provide information about which effects are
// applied by an operation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_SIDEEFFECTS_BASE
#define MLIR_INTERFACES_SIDEEFFECTS_BASE

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
         (ins "::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<"
              # baseEffect # ">> &":$effects)
    >,
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

    /// Collect all of the effect instances that operate on the provided value
    /// and place them in 'effects'.
    void getEffectsOnValue(::mlir::Value value,
              llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<
              }] # baseEffect # [{>> & effects) {
      getEffects(effects);
      llvm::erase_if(effects, [&](auto &it) { return it.getValue() != value; });
    }

    /// Return the effect of the given type `Effect` that is applied to the
    /// given value, or None if no effect exists.
    template <typename Effect>
    ::llvm::Optional<::mlir::SideEffects::EffectInstance<}] # baseEffect # [{>>
    getEffectOnValue(::mlir::Value value) {
      llvm::SmallVector<::mlir::SideEffects::EffectInstance<
              }] # baseEffect # [{>, 4> effects;
      getEffects(effects);
      auto it = llvm::find_if(effects, [&](auto &it) {
        return isa<Effect>(it.getEffect()) && it.getValue() == value;
      });
      if (it == effects.end())
        return llvm::None;
      return *it;
    }

    /// Collect all of the effect instances that operate on the provided symbol
    /// reference and place them in 'effects'.
    void getEffectsOnSymbol(::mlir::SymbolRefAttr value,
              llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<
              }] # baseEffect # [{>> & effects) {
      getEffects(effects);
      llvm::erase_if(effects, [&](auto &it) {
        return it.getSymbolRef() != value;
      });
    }

    /// Collect all of the effect instances that operate on the provided
    /// resource and place them in 'effects'.
    void getEffectsOnResource(::mlir::SideEffects::Resource *resource,
              llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<
              }] # baseEffect # [{>> & effects) {
      getEffects(effects);
      llvm::erase_if(effects, [&](auto &it) {
        return it.getResource() != resource;
      });
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

#endif // MLIR_INTERFACES_SIDEEFFECTS_BASE
