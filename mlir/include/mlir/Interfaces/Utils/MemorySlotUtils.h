//===- MemorySlotUtils.h - Utilities for MemorySlot interfaces --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares common utilities for implementing MemorySlot interfaces,
// in particular PromotableRegionOpInterface.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_UTILS_MEMORYSLOTUTILS_H
#define MLIR_INTERFACES_UTILS_MEMORYSLOTUTILS_H

#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace memoryslot {

/// Appends the reaching definition for the given block as an operand to its
/// terminator. If the block has no entry in `reachingAtBlockEnd` (e.g. dead
/// code or the region does not use the slot), `defaultReachingDef` is used.
void updateTerminator(Block *block, Value defaultReachingDef,
                      const DenseMap<Block *, Value> &reachingAtBlockEnd);

/// Creates a shallow copy of an operation with new result types, moving the
/// regions out of the original operation and deleting the original operation.
Operation *replaceWithNewResults(RewriterBase &rewriter, Operation *op,
                                 TypeRange resultTypes);

} // namespace memoryslot
} // namespace mlir

#endif // MLIR_INTERFACES_UTILS_MEMORYSLOTUTILS_H
