//===-- Mem2Reg.h - Mem2Reg definitions -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_MEM2REG_H
#define MLIR_TRANSFORMS_MEM2REG_H

#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/Mem2RegInterfaces.h"

namespace mlir {

/// Attempts to promote the memory slots of the provided allocators. Succeeds if
/// at least one memory slot was promoted.
LogicalResult
tryToPromoteMemorySlots(ArrayRef<PromotableAllocationOpInterface> allocators,
                        OpBuilder &builder, DominanceInfo &dominance);

} // namespace mlir

#endif // MLIR_TRANSFORMS_MEM2REG_H
