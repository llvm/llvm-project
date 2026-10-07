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

/// Whether `slotPtr` is the pointer of an alias slot that holds its whole
/// parent buffer, so that an out-of-bounds access through it is masked down to
/// the view's extent during promotion. Models for the ops that access such a
/// slot need this to know that the access is still promotable.
bool isDynamicViewSlot(Value slotPtr);

/// Builds the mask of the valid region of the dynamic view `slotPtr`, or
/// returns null if `isDynamicViewSlot(slotPtr)` does not hold.
Value buildDynamicViewMask(OpBuilder &builder, Location loc, Value slotPtr);
} // namespace memref
} // namespace mlir

#endif // MLIR_DIALECT_MEMREF_TRANSFORMS_MEMORYSLOTOPINTERFACEIMPL_H
