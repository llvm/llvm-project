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

/// Returns whether `slotPtr` is a dynamic `memref.subview` that can be promoted
/// as an alias of the whole parent vector.
bool isDynamicSubViewSlot(Value slotPtr);

/// Builds a mask for the subview's valid region in the parent vector, or
/// returns null if `isDynamicSubViewSlot(slotPtr)` is false.
Value buildDynamicSubViewMask(OpBuilder &builder, Location loc, Value slotPtr);
} // namespace memref
} // namespace mlir

#endif // MLIR_DIALECT_MEMREF_TRANSFORMS_MEMORYSLOTOPINTERFACEIMPL_H
