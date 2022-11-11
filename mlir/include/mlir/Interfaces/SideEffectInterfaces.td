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
  : MemoryEffect<"::mlir::MemoryEffects::Allocate", resource>;
def MemAlloc : MemAlloc<DefaultResource>;

// The following effect indicates that the operation frees some resource that
// has been allocated. A 'free' effect implies only de-allocation of the
// resource, and not any visible allocation, mutation or dereference.
class MemFree<Resource resource>
  : MemoryEffect<"::mlir::MemoryEffects::Free", resource>;
def MemFree : MemFree<DefaultResource>;

// The following effect indicates that the operation reads from some
// resource. A 'read' effect implies only dereferencing of the resource, and
// not any visible mutation.
class MemRead<Resource resource>
  : MemoryEffect<"::mlir::MemoryEffects::Read", resource>;
def MemRead : MemRead<DefaultResource>;

// The following effect indicates that the operation writes to some
// resource. A 'write' effect implies only mutating a resource, and not any
// visible dereference or read.
class MemWrite<Resource resource>
  : MemoryEffect<"::mlir::MemoryEffects::Write", resource>;
def MemWrite : MemWrite<DefaultResource>;

//===----------------------------------------------------------------------===//
// Effect Traits
//===----------------------------------------------------------------------===//

// Op has no effect on memory but may have undefined behavior.
def NoMemoryEffect : MemoryEffects<[]>;

// Op has recursively computed side effects.
def RecursiveMemoryEffects : NativeOpTrait<"HasRecursiveMemoryEffects">;

//===----------------------------------------------------------------------===//
// Speculation
//===----------------------------------------------------------------------===//

// Used to inject an implementation of getSpeculatability.  Users should not use
// this directly.
def RecursivelySpeculatableImplTrait
  : NativeOpTrait<"RecursivelySpeculatableImplTrait">;

// Used to inject an implementation of getSpeculatability.  Users should not use
// this directly.
def AlwaysSpeculatableImplTrait
  : NativeOpTrait<"AlwaysSpeculatableImplTrait">;

// This op interface enables Op authors to inject custom logic to determine
// whether an Operation can be speculatively executed.  Ops that implement this
// interface need to implement the custom logic in the `getSpeculatability` method.
// For instance, the `getSpeculatability` for a specific op may check the attributes
// or input types to determine whether that specific Operation is speculatable.
def ConditionallySpeculatable : OpInterface<"ConditionallySpeculatable"> {
  let description = [{
    An interface used to query information about the speculability of an
    operation.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Returns value indicating whether the specific operation in question can
        be speculatively executed.  Please see the documentation on the
        Speculatability enum to know how to interpret the return value.
      }],
      "::mlir::Speculation::Speculatability", "getSpeculatability", (ins)>
  ];
}

// Marks an Operation as always speculatable.
def AlwaysSpeculatable : TraitList<[
    ConditionallySpeculatable, AlwaysSpeculatableImplTrait]>;

// Marks an Operation as speculatable only if all the operations in all attached
// regions are also speculatable.
def RecursivelySpeculatable : TraitList<[
    ConditionallySpeculatable, RecursivelySpeculatableImplTrait]>;

// Always speculatable operation that does not touch memory.  These operations
// are always legal to hoist or sink.
def Pure : TraitList<[AlwaysSpeculatable, NoMemoryEffect]>;

#endif // MLIR_INTERFACES_SIDEEFFECTS
