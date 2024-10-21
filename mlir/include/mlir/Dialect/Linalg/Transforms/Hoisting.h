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
class Operation;
class RewriterBase;
namespace scf {
class ForOp;
} // namespace scf

namespace linalg {

/// Hoist vector.transfer_read/vector.transfer_write on buffers pairs out of
/// immediately enclosing scf::ForOp iteratively, if the following conditions
/// are true:
///   1. The two ops access the same memref with the same indices.
///   2. All operands are invariant under the enclosing scf::ForOp.
///   3. No uses of the memref either dominate the transfer_read or are
///   dominated by the transfer_write (i.e. no aliasing between the write and
///   the read across the loop)
///   4. The source operands for vector.transfer_{read|write} do not originate
///   from Ops implementing ViewLikeOpInterface (to reduce the risk of
///   aliasing).
///   5. If `verifyNonZeroTrip` is true, then the lower bound of the loop must
///   be statically smaller than the upper bound of the loop, guaranteeing that
///   the loop body will execute at least once.
/// To improve hoisting opportunities, call the `moveLoopInvariantCode` helper
/// function on the candidate loop above which to hoist. Hoisting the transfers
/// results in scf::ForOp yielding the value that originally transited through
/// memory.
///
/// TODO: To further improve hoisting opportunities, fold aliasing memref
/// operations into respective vector.transfer{read|write} operations and
/// avoid using ops implementing ViewLikeOpInterface as the source for transfer
/// Ops.
///
/// WARNING: This hoisting does not model parallelism and is generally incorrect
/// when used on distributed loops with memref semantics!
/// NOTE: Setting `verifyNonZeroTrip = true` makes this more stable for
/// distributed loops with memref semantics, but there could still be some
/// issues when loops are executed a different number of times for different
/// threads.
void hoistRedundantVectorTransfers(Operation *root,
                                   bool verifyNonZeroTrip = false);

/// Hoist vector.extract/vector.broadcast pairs out of immediately enclosing
/// scf::ForOp iteratively, if the following conditions are met:
///   1. The vector.extract operation is applied on an iter_argument, and no
///   other operator is using this argument in the body of the loop.
///   2. The position of the vector.extract is either a static value, or defined
///   outside of the loop.
///   3. The vector.broadcast operation is yielded by the loop.
/// To improve hoisting opportunities, call the `moveLoopInvariantCode` helper
/// function on the candidate loop above which to hoist.
void hoistRedundantVectorBroadcasts(RewriterBase &rewriter, Operation *root);

} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_TRANSFORMS_HOISTING_H_
