//===- Transforms.h - NVGPU Dialect transformations --------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares functions that assist transformations for the nvgpu
// dialect.
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_NVGPU_TRANSFORMS_TRANSFORMS_H_
#define MLIR_DIALECT_NVGPU_TRANSFORMS_TRANSFORMS_H_

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace nvgpu {

///
/// Passes
///

/// Optimizes vectorized accesses to a shared memory buffer specified by
/// memrefValue. This transformation assumes the following:
/// 1) All relevant accesses to `memrefValue` are contained with `parentOp`.
/// 2) The function will fail precondition checks if any subviews are
/// taken of `memrefValue`. All reads/writes to `memrefValue` should occur
/// through `memrefValue` directly.
///
/// Shared memory bank conflicts occur when multiple threads attempt to read or
/// write locations assigned to the same shared memory bank. For `2^N` byte
/// vectorized accesses, we need to be concerned with conflicts among threads
/// identified as `(tid) -> tid.floordiv(2^{7-N})`. As such, this transformation
/// changes any indexed memory access (vector.load, memref.load, nvgpu.ldmatrix,
/// etc) such that the final dimension's index value is permuted such that
/// `newColIndex = oldColIndex % vectorSize +
/// perm[rowIndex](oldColIndex/vectorSize, rowIndex)` where `rowIndex` is the
/// index for the second-to last dimension and `perm[rowIndex]` is a permutation
/// function that depends on the row Index. The permutation function is chosen
/// to ensure that sequential distributed+vectorized reads/writes down a single
/// dimension of the memref have minimal conflicts.
mlir::LogicalResult optimizeSharedMemoryReadsAndWrites(Operation *parentOp,
                                                       Value memrefValue);

///
/// Rewrites patterns
///

//===----------------------------------------------------------------------===//
// NVGPU transformation options exposed as auxiliary structs.
//===----------------------------------------------------------------------===//
/// Enum to control the lowering of `nvgpu.mmasync`.
enum class MmaSyncF32Lowering { TF32 = 0, TF32x3 = 1, Unkown = 2 };

/// Collect patterns to convert mma.sync on f32 input and rewrite
/// to use tensor cores with user provided level of accuracy:
/// (a) tf32   (1 mma.sync per warp-level matrix-multiply-accumulate)
/// (b) tf32x3 (3 mma.sync per warp-level matrix-multiply-accumulate)
/// Typically, tf32 tensor core acceleration comes at a cost
/// of accuracy from missing precision bits. While f32 has 23 precision
/// bits, tf32 has only 10 precision bits. tf32x3 aims to recover the
/// precision bits by spliting each operand into two tf32 values
/// and issue three mma.sync tensor core operations.
void populateMmaSyncF32ToTF32Patterns(
    RewritePatternSet &patterns,
    nvgpu::MmaSyncF32Lowering precision = nvgpu::MmaSyncF32Lowering::TF32);

} // namespace nvgpu
} // namespace mlir

#endif // MLIR_DIALECT_NVGPU_TRANSFORMS_TRANSFORMS_H_
