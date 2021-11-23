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
// AllocationOpInterface
//===----------------------------------------------------------------------===//

def AllocationOpInterface : OpInterface<"AllocationOpInterface"> {
  let description = [{
    This interface provides general allocation-related methods that are
    designed for allocation operations. For example, it offers the ability to
    construct associated deallocation and clone operations that are compatible
    with the current allocation operation.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    StaticInterfaceMethod<[{
        Builds a deallocation operation using the provided builder and the
        current allocation value (which refers to the current Op implementing
        this interface). The allocation value is a result of the current
        operation implementing this interface. If there is no compatible
        deallocation operation, this method can return ::llvm::None.
      }],
      "::mlir::Optional<::mlir::Operation*>", "buildDealloc",
      (ins "::mlir::OpBuilder&":$opBuilder, "::mlir::Value":$alloc), [{}],
      /*defaultImplementation=*/[{ return llvm::None; }]
    >,
    StaticInterfaceMethod<[{
        Builds a clone operation using the provided builder and the current
        allocation value (which refers to the current Op implementing this
        interface). The allocation value is a result of the current operation
        implementing this interface. If there is no compatible clone operation,
        this method can return ::llvm::None.
      }],
      "::mlir::Optional<::mlir::Value>", "buildClone",
      (ins "::mlir::OpBuilder&":$opBuilder, "::mlir::Value":$alloc), [{}],
      /*defaultImplementation=*/[{ return llvm::None; }]
    >
  ];
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

// Op has no side effect.
def NoSideEffect : MemoryEffects<[]>;
// Op has recursively computed side effects.
def RecursiveSideEffects : NativeOpTrait<"HasRecursiveSideEffects">;

#endif // MLIR_INTERFACES_SIDEEFFECTS
