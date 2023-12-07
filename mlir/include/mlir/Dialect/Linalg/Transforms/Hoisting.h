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
class RewriterBase;
namespace func {
class FuncOp;
} // namespace func
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
/// To improve hoisting opportunities, call the `moveLoopInvariantCode` helper
/// function on the candidate loop above which to hoist. Hoisting the transfers
/// results in scf::ForOp yielding the value that originally transited through
/// memory.
///
/// WARNING: This hoisting does not model parallelism and is generally incorrect
/// when used on distributed loops with memref semantics!
void hoistRedundantVectorTransfers(func::FuncOp func);

/// Greedily hoist redundant subset extract/insert operations on tensors outside
/// of `forOp`. The logic follows:
///   1. Look for a write walking back from the `forOp` yield.
///   2. Check the uses of the matching block argument and look for a matching
///      read (i.e. extract_slice of transfer_read) with matching indices.
///   3. In the case of a transfer_write, we can bypass other non-conflicting
///      operations and find more hoisting opportunities.
///   4. Hoist the read/write pair and update the tensor SSA links.
///
/// Return the unmodified `forOp` if no hoisting occured.
/// Return a new scf::ForOp if hoisting on tensors occured.
///
/// After this transformation the returned scf::ForOp may have unused arguments
/// that can be removed by application of canonicalization patterns.
///
/// Example:
/// ========
/// IR Resembling:
///
/// ```
/// %0 = scf.for %i = %l to %u step %s iter_args(%a0 = %t0)->(tensor<10xf32>) {
///  %1 = scf.for %j = %l to %u step %s iter_args(%a6 = %a0)->(tensor<10xf32>) {
///   %e = tensor.extract_slice %a6[%i][%sz][1]: tensor<10xf32> to tensor<?xf32>
///   %r = vector.transfer_read %e[%c0], %cst: tensor<?xf32>, vector<4xf32>
///   %u = "some_use"(%r) : (vector<4xf32>) -> vector<4xf32>
///   %w = vector.transfer_write %u, %e[%c0] : vector<4xf32>, tensor<?xf32>
///   %st = tensor.insert_slice %w into %a6[%i][%sz][1]
///     : tensor<?xf32> into tensor<10xf32>
///   scf.yield %st: tensor<10xf32>
///  }
///  scf.yield %1: tensor<10xf32>
/// }
/// ```
///
/// Progressively hoists to:
///
/// ```
/// %0 = scf.for %i = %l to %u step %s iter_args(%a0 = %t0) -> (tensor<10xf32>){
///  %e = tensor.extract_slice %a0[%i][%sz][1]: tensor<10xf32> to tensor<?xf32>
///  %1:2 = scf.for %j = %l to %u step %s iter_args(%a6 = a0, %a7 = %e)
///     -> (tensor<10xf32>, tensor<?xf32>) {
///   %r = vector.transfer_read %a7[%c0], %cst: tensor<?xf32>, vector<4xf32>
///   %u = "some_use"(%r) : (vector<4xf32>) -> vector<4xf32>
///   %w = vector.transfer_write %u, %a7[%c0] : vector<4xf32>, tensor<?xf32>
///   scf.yield %a6, %w: tensor<10xf32>, tensor<?xf32>
///  }
///  %st = tensor.insert_slice %1#1 into %1#0[%i][%sz][1]
///    : tensor<?xf32> into tensor<10xf32>
///  scf.yield %1: tensor<10xf32>
/// }
/// ```
///
/// and
///
/// ```
/// %0 = scf.for %i = %l to %u step %s iter_args(%a0 = %t0) -> (tensor<10xf32>){
///  %e = tensor.extract_slice %a0[%i][%sz][1]: tensor<10xf32> to tensor<?xf32>
///  %r = vector.transfer_read %a7[%c0], %cst: tensor<?xf32>, vector<4xf32>
///  %1:3 = scf.for %j = %l to %u step %s iter_args(%a6 = a0, %a7 = %e, %a7 = r)
///     -> (tensor<10xf32>, tensor<?xf32>, vector<4xf32>) {
///   %u = "some_use"(%r) : (vector<4xf32>) -> vector<4xf32>
///   scf.yield %a6, %a7, %u: tensor<10xf32>, tensor<?xf32>, vector<4xf32>
///  }
///  %w = vector.transfer_write %1#2, %1#1[%c0] : vector<4xf32>, tensor<?xf32>
///  %st = tensor.insert_slice %w into %1#0[%i][%sz][1]
///    : tensor<?xf32> into tensor<10xf32>
///  scf.yield %1: tensor<10xf32>
/// }
/// ```
///
/// It can then canonicalize to:
///
/// ```
/// %0 = scf.for %i = %l to %u step %s iter_args(%a0 = %t0) -> (tensor<10xf32>){
///  %e = tensor.extract_slice %a0[%i][%sz][1]: tensor<10xf32> to tensor<?xf32>
///  %r = vector.transfer_read %a7[%c0], %cst: tensor<?xf32>, vector<4xf32>
///  %1 = scf.for %j = %l to %u step %s iter_args(%a7 = r)
///     -> (tensor<10xf32>, tensor<?xf32>, vector<4xf32>) {
///   %u = "some_use"(%r) : (vector<4xf32>) -> vector<4xf32>
///   scf.yield %u: vector<4xf32>
///  }
///  %w = vector.transfer_write %1, %e[%c0] : vector<4xf32>, tensor<?xf32>
///  %st = tensor.insert_slice %w into %a0[%i][%sz][1]
///    : tensor<?xf32> into tensor<10xf32>
///  scf.yield %1: tensor<10xf32>
/// }
/// ```
///
// TODO: This should be further generalized along a few different axes:
//   - Other loops than scf.ForOp that operate on tensors (both sequential and
//     parallel loops).
//   - Other subset extract/insert pairs than tensor.extract/insert_slice and
//     vector.transfer_read/write.
//   - More general areSubsetDisjoint analysis/interface to work across all
//     subset op types and allow bypassing non-WAW-conflicting operations in
//     more cases.
scf::ForOp hoistRedundantSubsetExtractInsert(RewriterBase &rewriter,
                                             scf::ForOp forOp);

/// Call into `hoistRedundantSubsetInsertExtract` without a RewriterBase.
// TODO: obsolete and should be retired
void hoistRedundantVectorTransfersOnTensor(func::FuncOp func);

} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_TRANSFORMS_HOISTING_H_
