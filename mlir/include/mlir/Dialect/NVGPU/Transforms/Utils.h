//===- Utils.h - Transform utilities -----------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Operation.h"

namespace mlir {
namespace nvgpu {

/// Get the indices that the given load/store operation is operating on.
Operation::operand_range getIndices(Operation *op);

/// Set the indices that the given load/store operation is operating on.
void setIndices(Operation *op, ArrayRef<Value> indices);

} // namespace nvgpu
} // namespace mlir
