//===-- Traits.td - Trait definations file ------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definations for traits.
//
//===----------------------------------------------------------------------===//

#ifndef TRAITS_TD
#define TRAITS_TD

include "mlir/IR/Constraints.td"

//===----------------------------------------------------------------------===//
// Trait definitions
//===----------------------------------------------------------------------===//

// Trait represents a trait regarding an attribute, operation, or type.
class Trait;

// Define a Trait corresponding to a list of Traits, this allows for specifying
// a list of traits as trait. Avoids needing to do `[Traits, ...] # ListOfTraits
// # [Others, ...]` while still allowing providing convenient groupings.
class TraitList<list<Trait> props> : Trait {
  list<Trait> traits = props;
}

// NativeTrait corresponds to the MLIR C++ trait mechanism. The purpose to wrap
// around C++ symbol string with this class is to make traits specified for
// entities in TableGen less alien and more integrated.
// `extraConcreteClassDeclaration` and `extraConcreteClassDefinition` code
// get injected into the entities in which the NativeTrait is specified for.
class NativeTrait<string name, string entityType,
                    code extraClassDeclaration = [{}],
                    code extraClassDefinition = [{}]> : Trait {
  string trait = name;
  string cppNamespace = "::mlir::" # entityType # "Trait";

  code extraConcreteClassDeclaration = extraClassDeclaration;
  code extraConcreteClassDefinition = extraClassDefinition;
}

// ParamNativeTrait corresponds to the template-parameterized traits in the C++
// implementation. MLIR uses nested class templates to implement such traits
// leading to constructs of the form "TraitName<Parameters>::Impl". Use the
// value in `prop` as the trait name and the value in `params` as parameters to
// construct the native trait class name.
class ParamNativeTrait<string prop, string params, string entityType>
    : NativeTrait<prop # "<" # params # ">::Impl", entityType>;

// GenInternalTrait is a trait that does not have direct C++ mapping but affects
// an entities definition generator internals, like how operation builders and
// operand/attribute/result getters are generated.
class GenInternalTrait<string prop, string entityType> : Trait {
  string trait = "::mlir::" # entityType # "Trait::" # prop;
}

// PredTrait is a trait implemented by way of a predicate on an entity.
class PredTrait<string descr, Pred pred> : Trait {
  string summary = descr;
  Pred predicate = pred;
}

#endif // TRAITS_TD
