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
///   1. All operands are defined outside the loop.
///   2. All uses are ViewLikeOp or DeallocOp.
// TODO: generalize on a per-need basis.
void hoistViewAllocOps(FuncOp func);

/// Hoist vector.transfer_read/vector.transfer_write pairs out of immediately
/// enclosing scf::ForOp iteratively, if the following conditions are true:
///   1. The two ops access the same memref with the same indices.
///   2. All operands are invariant under the enclosing scf::ForOp.
///   3. No uses of the memref either dominate the transfer_read or are
///   dominated by the transfer_write (i.e. no aliasing between the write and
///   the read across the loop)
/// To improve hoisting opportunities, call the `moveLoopInvariantCode` helper
/// function on the candidate loop above which to hoist. Hoisting the transfers
/// results in scf::ForOp yielding the value that originally transited through
/// memory.
// TODO: generalize on a per-need basis.
void hoistRedundantVectorTransfers(FuncOp func);

} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_TRANSFORMS_HOISTING_H_
