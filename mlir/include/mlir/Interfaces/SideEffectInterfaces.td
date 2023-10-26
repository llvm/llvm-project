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
class MemoryEffect<string effectName, Resource resource, int stage,
                   EffectRange range>
  : SideEffect<MemoryEffectsOpInterface, effectName, resource, stage, range>;

// This class represents the trait for memory effects that may be placed on
// operations.
class MemoryEffects<list<MemoryEffect> effects = []>
  : SideEffectsTraitBase<MemoryEffectsOpInterface, effects>;

//===----------------------------------------------------------------------===//
// Effects

// The following effect indicates that the operation allocates from some
// resource. An 'allocate' effect implies only allocation of the resource, and
// not any visible mutation or dereference.
class MemAlloc<Resource resource, int stage = 0,
               EffectRange range = PartialEffect>
  : MemoryEffect<"::mlir::MemoryEffects::Allocate", resource, stage, range>;
def MemAlloc : MemAlloc<DefaultResource, 0, PartialEffect>;
class MemAllocAt<int stage, EffectRange range = PartialEffect>
  : MemAlloc<DefaultResource, stage, range>;

// The following effect indicates that the operation frees some resource that
// has been allocated. A 'free' effect implies only de-allocation of the
// resource, and not any visible allocation, mutation or dereference.
class MemFree<Resource resource, int stage = 0,
              EffectRange range = PartialEffect>
  : MemoryEffect<"::mlir::MemoryEffects::Free", resource, stage, range>;
def MemFree : MemFree<DefaultResource, 0, PartialEffect>;
class MemFreeAt<int stage, EffectRange range = PartialEffect>
  : MemFree<DefaultResource, stage, range>;

// The following effect indicates that the operation reads from some
// resource. A 'read' effect implies only dereferencing of the resource, and
// not any visible mutation.
class MemRead<Resource resource, int stage = 0,
              EffectRange range = PartialEffect>
  : MemoryEffect<"::mlir::MemoryEffects::Read", resource, stage, range>;
def MemRead : MemRead<DefaultResource, 0, PartialEffect>;
class MemReadAt<int stage, EffectRange range = PartialEffect>
  : MemRead<DefaultResource, stage, range>;

// The following effect indicates that the operation writes to some
// resource. A 'write' effect implies only mutating a resource, and not any
// visible dereference or read.
class MemWrite<Resource resource, int stage = 0,
               EffectRange range = PartialEffect>
  : MemoryEffect<"::mlir::MemoryEffects::Write", resource, stage, range>;
def MemWrite : MemWrite<DefaultResource, 0, PartialEffect>;
class MemWriteAt<int stage, EffectRange range = PartialEffect>
  : MemWrite<DefaultResource, stage, range>;

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
