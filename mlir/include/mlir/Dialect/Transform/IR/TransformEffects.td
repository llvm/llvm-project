
//===- TransformEffect.td - Transform side effects ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines side effects and associated resources for operations in the
// Transform dialect and extensions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_IR_TRANSFORM_EFFECTS_TD
#define MLIR_DIALECT_TRANSFORM_IR_TRANSFORM_EFFECTS_TD

include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Effects on the mapping between Transform IR values and Payload IR ops.
//===----------------------------------------------------------------------===//

// Side effect resource corresponding to the mapping between transform IR values
// and Payload IR operations.
def TransformMappingResource
    : Resource<"::mlir::transform::TransformMappingResource">;

// Describes the creation of a new entry in the transform mapping. Should be
// accompanied by the Write effect as the entry is immediately initialized by
// any reasonable transform operation.
def TransformMappingAlloc : MemAlloc<TransformMappingResource>;

// Describes the removal of an entry in the transform mapping. Typically
// accompanied by the Read effect.
def TransformMappingFree : MemFree<TransformMappingResource>;

// Describes the access to the mapping. Read-only accesses can be reordered.
def TransformMappingRead : MemRead<TransformMappingResource>;

// Describes a modification of an existing entry in the mapping. It is rarely
// used alone, and is mostly accompanied by the Allocate effect.
def TransformMappingWrite : MemWrite<TransformMappingResource>;

//===----------------------------------------------------------------------===//
// Effects on Payload IR.
//===----------------------------------------------------------------------===//

// Side effect resource corresponding to the Payload IR itself.
def PayloadIRResource : Resource<"::mlir::transform::PayloadIRResource">;

// Corresponds to the read-only access to the Payload IR through some operation
// handles in the Transform IR.
def PayloadIRRead : MemRead<PayloadIRResource>;

// Corresponds to the mutation of the Payload IR through an operation handle in
// the Transform IR. Should be accompanied by the Read effect for most transform
// operations (only a complete overwrite of the root op of the Payload IR is a
// write-only modification).
def PayloadIRWrite : MemWrite<PayloadIRResource>;

#endif // MLIR_DIALECT_TRANSFORM_IR_TRANSFORM_EFFECTS_TD
