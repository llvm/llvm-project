//===- Hoisting.h - Linalg hoisting transformations -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_TRANSFORMS_HOISTING_H_
#define MLIR_DIALECT_LINALG_TRANSFORMS_HOISTING_H_

namespace mlir {
class FuncOp;

namespace linalg {

/// Hoist alloc/dealloc pairs and alloca op out of immediately enclosing
/// scf::ForOp if both conditions are true:
///   1. all operands are defined outside the loop.
///   2. all uses are ViewLikeOp or DeallocOp.
// TODO: generalize on a per-need basis.
void hoistViewAllocOps(FuncOp func);

} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_TRANSFORMS_HOISTING_H_
