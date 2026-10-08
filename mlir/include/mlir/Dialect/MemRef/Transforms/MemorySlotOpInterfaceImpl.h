//===- MemorySlotOpInterfaceImpl.h - Mem2Reg for MemRef ops -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MEMREF_TRANSFORMS_MEMORYSLOTOPINTERFACEIMPL_H
#define MLIR_DIALECT_MEMREF_TRANSFORMS_MEMORYSLOTOPINTERFACEIMPL_H

#include "mlir/IR/Value.h"

namespace mlir {
class DialectRegistry;
class Location;
class OpBuilder;

namespace memref {
void registerMemorySlotOpInterfaceExternalModels(DialectRegistry &registry);

/// Whether `slotPtr` is the result of a dynamically-shaped `memref.subview`
/// that the subview aliaser exposes as an alias of the whole parent value.
bool isDynamicSubViewSlot(Value slotPtr);

/// Builds the mask of the valid region of the subview defining `slotPtr`, or
/// returns null if `isDynamicSubViewSlot(slotPtr)` does not hold.
Value buildDynamicSubViewMask(OpBuilder &builder, Location loc, Value slotPtr);
} // namespace memref
} // namespace mlir

#endif // MLIR_DIALECT_MEMREF_TRANSFORMS_MEMORYSLOTOPINTERFACEIMPL_H
