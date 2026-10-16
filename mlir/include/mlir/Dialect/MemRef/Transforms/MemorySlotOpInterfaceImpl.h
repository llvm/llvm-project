//===- MemorySlotOpInterfaceImpl.h - Mem2Reg for MemRef ops -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MEMREF_TRANSFORMS_MEMORYSLOTOPINTERFACEIMPL_H
#define MLIR_DIALECT_MEMREF_TRANSFORMS_MEMORYSLOTOPINTERFACEIMPL_H

#include "mlir/Interfaces/MemorySlotInterfaces.h"

namespace mlir {
class DialectRegistry;
class Location;
class OpBuilder;

namespace memref {
void registerMemorySlotOpInterfaceExternalModels(DialectRegistry &registry);

/// Returns whether `slot` represents a supported dynamic `memref.subview`,
/// optionally followed by `memref.expand_shape` or `memref.collapse_shape`.
/// Such a slot retains the whole parent vector, so accesses must mask out
/// elements outside the subview's runtime bounds.
bool isPartialViewSlot(const MemorySlot &slot);

/// Builds a mask of the subview's runtime bounds in the shape of
/// `slot.valueType`, reshaping the mask if needed. Returns a null value for
/// slots that do not require a subview mask. Expects a promotable vector slot.
Value buildPartialViewMask(OpBuilder &builder, Location loc,
                           const MemorySlot &slot);
} // namespace memref
} // namespace mlir

#endif // MLIR_DIALECT_MEMREF_TRANSFORMS_MEMORYSLOTOPINTERFACEIMPL_H
