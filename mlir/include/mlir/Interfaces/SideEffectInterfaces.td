//===-- SideEffectInterfaces.td - Side Effect Interfaces ---*- tablegen -*-===//
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

include "mlir/Interfaces/SideEffectInterfaceBase.td"

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
