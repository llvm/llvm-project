//===- VectorToSCF.h - Utils to convert from the vector dialect -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_VECTORTOSCF_VECTORTOSCF_H_
#define MLIR_CONVERSION_VECTORTOSCF_VECTORTOSCF_H_

#include "mlir/IR/PatternMatch.h"

namespace mlir {
class MLIRContext;
class Pass;
class RewritePatternSet;
using OwningRewritePatternList = RewritePatternSet;

/// Control whether unrolling is used when lowering vector transfer ops to SCF.
///
/// Case 1:
/// =======
/// When `unroll` is false, a temporary buffer is created through which
/// individual 1-D vector are staged. this is consistent with the lack of an
/// LLVM instruction to dynamically index into an aggregate (see the Vector
/// dialect lowering to LLVM deep dive).
/// An instruction such as:
/// ```
///    vector.transfer_write %vec, %A[%base, %base] :
///      vector<17x15xf32>, memref<?x?xf32>
/// ```
/// Lowers to pseudo-IR resembling:
/// ```
///    %0 = alloc() : memref<17xvector<15xf32>>
///    %1 = vector.type_cast %0 :
///      memref<17xvector<15xf32>> to memref<vector<17x15xf32>>
///    store %vec, %1[] : memref<vector<17x15xf32>>
///    %dim = dim %A, 0 : memref<?x?xf32>
///    affine.for %I = 0 to 17 {
///      %add = affine.apply %I + %base
///      %cmp = cmpi "slt", %add, %dim : index
///      scf.if %cmp {
///        %vec_1d = load %0[%I] : memref<17xvector<15xf32>>
///        vector.transfer_write %vec_1d, %A[%add, %base] :
///          vector<15xf32>, memref<?x?xf32>
/// ```
///
/// Case 2:
/// =======
/// When `unroll` is true, the temporary buffer is skipped and static indices
/// into aggregates can be used (see the Vector dialect lowering to LLVM deep
/// dive).
/// An instruction such as:
/// ```
///    vector.transfer_write %vec, %A[%base, %base] :
///      vector<3x15xf32>, memref<?x?xf32>
/// ```
/// Lowers to pseudo-IR resembling:
/// ```
///    %0 = vector.extract %arg2[0] : vector<3x15xf32>
///    vector.transfer_write %0, %arg0[%arg1, %arg1] : vector<15xf32>,
///    memref<?x?xf32> %1 = affine.apply #map1()[%arg1] %2 = vector.extract
///    %arg2[1] : vector<3x15xf32> vector.transfer_write %2, %arg0[%1, %arg1] :
///    vector<15xf32>, memref<?x?xf32> %3 = affine.apply #map2()[%arg1] %4 =
///    vector.extract %arg2[2] : vector<3x15xf32> vector.transfer_write %4,
///    %arg0[%3, %arg1] : vector<15xf32>, memref<?x?xf32>
/// ```
struct VectorTransferToSCFOptions {
  bool unroll = false;
  VectorTransferToSCFOptions &setUnroll(bool u) {
    unroll = u;
    return *this;
  }
};

/// Implements lowering of TransferReadOp and TransferWriteOp to a
/// proper abstraction for the hardware.
///
/// There are multiple cases.
///
/// Case A: Permutation Map does not permute or broadcast.
/// ======================================================
///
/// Progressive lowering occurs to 1-D vector transfer ops according to the
/// description in `VectorTransferToSCFOptions`.
///
/// Case B: Permutation Map permutes and/or broadcasts.
/// ======================================================
///
/// This path will be progressively deprecated and folded into the case above by
/// using vector broadcast and transpose operations.
///
/// This path only emits a simple loop nest that performs clipped pointwise
/// copies from a remote to a locally allocated memory.
///
/// Consider the case:
///
/// ```mlir
///    // Read the slice `%A[%i0, %i1:%i1+256, %i2:%i2+32]` into
///    // vector<32x256xf32> and pad with %f0 to handle the boundary case:
///    %f0 = constant 0.0f : f32
///    scf.for %i0 = 0 to %0 {
///      scf.for %i1 = 0 to %1 step %c256 {
///        scf.for %i2 = 0 to %2 step %c32 {
///          %v = vector.transfer_read %A[%i0, %i1, %i2], %f0
///               {permutation_map: (d0, d1, d2) -> (d2, d1)} :
///               memref<?x?x?xf32>, vector<32x256xf32>
///    }}}
/// ```
///
/// The rewriters construct loop and indices that access MemRef A in a pattern
/// resembling the following (while guaranteeing an always full-tile
/// abstraction):
///
/// ```mlir
///    scf.for %d2 = 0 to %c256 {
///      scf.for %d1 = 0 to %c32 {
///        %s = %A[%i0, %i1 + %d1, %i2 + %d2] : f32
///        %tmp[%d2, %d1] = %s
///      }
///    }
/// ```
///
/// In the current state, only a clipping transfer is implemented by `clip`,
/// which creates individual indexing expressions of the form:
///
/// ```mlir-dsc
///    auto condMax = i + ii < N;
///    auto max = std_select(condMax, i + ii, N - one)
///    auto cond = i + ii < zero;
///    std_select(cond, zero, max);
/// ```
///
/// In the future, clipping should not be the only way and instead we should
/// load vectors + mask them. Similarly on the write side, load/mask/store for
/// implementing RMW behavior.
///
/// Lowers TransferOp into a combination of:
///   1. local memory allocation;
///   2. perfect loop nest over:
///      a. scalar load/stores from local buffers (viewed as a scalar memref);
///      a. scalar store/load to original memref (with clipping).
///   3. vector_load/store
///   4. local memory deallocation.
/// Minor variations occur depending on whether a TransferReadOp or
/// a TransferWriteOp is rewritten.
template <typename TransferOpTy>
struct VectorTransferRewriter : public RewritePattern {
  explicit VectorTransferRewriter(VectorTransferToSCFOptions options,
                                  MLIRContext *context);

  /// Used for staging the transfer in a local buffer.
  MemRefType tmpMemRefType(TransferOpTy transfer) const;

  /// Performs the rewrite.
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;

  /// See description of `VectorTransferToSCFOptions`.
  VectorTransferToSCFOptions options;
};

/// Collect a set of patterns to convert from the Vector dialect to SCF + std.
void populateVectorToSCFConversionPatterns(
    OwningRewritePatternList &patterns,
    const VectorTransferToSCFOptions &options = VectorTransferToSCFOptions());

/// Create a pass to convert a subset of vector ops to SCF.
std::unique_ptr<Pass> createConvertVectorToSCFPass(
    const VectorTransferToSCFOptions &options = VectorTransferToSCFOptions());

} // namespace mlir

#endif // MLIR_CONVERSION_VECTORTOSCF_VECTORTOSCF_H_
