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
struct LogicalResult;

namespace linalg {
class PadTensorOp;

/// Hoist vector.transfer_read/vector.transfer_write on buffers pairs out of
/// immediately enclosing scf::ForOp iteratively, if the following conditions
/// are true:
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

/// Same behavior as `hoistRedundantVectorTransfers` but works on tensors
/// instead of buffers.
void hoistRedundantVectorTransfersOnTensor(FuncOp func);

/// Mechanically hoist padding operations on tensors by `nLoops` into a new,
/// generally larger tensor. This achieves packing of multiple padding ops into
/// a larger tensor. On success, `padTensorOp` is replaced by the cloned version
/// in the packing loop so the caller can continue reasoning about the padding
/// operation.
///
/// Example in pseudo-mlir:
/// =======================
///
/// If hoistPaddingOnTensors is called with `nLoops` = 2 on the following IR.
/// ```
///    scf.for (%i, %j, %k)
///      %st0 = subtensor f(%i, %k) : ... to tensor<?x?xf32>
///      %0 = linalg.pad_tensor %st0 low[0, 0] high[...] {
///      ^bb0( ... ):
///        linalg.yield %pad
///      } : tensor<?x?xf32> to tensor<4x8xf32>
///      compute(%0)
/// ```
///
/// IR resembling the following is produced:
///
/// ```
///    scf.for (%i) {
///      %packed_init = linalg.init_tensor range(%j) : tensor<?x4x8xf32>
///      %packed = scf.for (%k) iter_args(%p : %packed_init) {
///        %st0 = subtensor f(%i, %k) : ... to tensor<?x?xf32>
///        %0 = linalg.pad_tensor %st0 low[0, 0] high[...] {
///        ^bb0( ... ):
///          linalg.yield %pad
///        } : tensor<?x?xf32> to tensor<4x8xf32>
///        %1 = subtensor_insert %0 ... : tensor<4x8xf32> to tensor<?x4x8xf32>
///        scf.yield %1: tensor<?x4x8xf32>
///      } -> tensor<?x4x8xf32>
///      scf.for (%j, %k) {
///        %st0 = subtensor %packed [%k, 0, 0][1, 4, 8][1, 1, 1] :
///                 tensor<?x4x8xf32> to tensor<4x8xf32>
///        compute(%st0)
///      }
///    }
/// ```
LogicalResult hoistPaddingOnTensors(PadTensorOp &padTensorOp, unsigned nLoops);

} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_TRANSFORMS_HOISTING_H_
