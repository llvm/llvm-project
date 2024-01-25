//===- Utils.h - Transform utilities -----------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Operation.h"

namespace mlir {
namespace amdgpu {

/// Get and set the indices that the given load/store operation is operating on.
/// Preconditions:
/// - The Op must have memory affects.
/// - Considers memref::LoadOp, vector::LoadOp, and vector::TransferReadOp.
/// - Considers memref::StoreOp, vector::StoreOp, and vector::TransferWriteOp.
/// - Excludes subview op.
std::optional<Operation::operand_range> getIndices(Operation *op);
void setIndices(Operation *op, ArrayRef<Value> indices);

} // namespace amdgpu
} // namespace mlir
