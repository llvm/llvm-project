//===- Transforms.h - AMDGPU Dialect transformations -------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares functions that assist transformations for the amdgpu
// dialect.
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_AMDGPU_TRANSFORMS_TRANSFORMS_H_
#define MLIR_DIALECT_AMDGPU_TRANSFORMS_TRANSFORMS_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class RewriterBase;

namespace amdgpu {

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
/// changes any indexed memory access (vector.load, memref.load, etc)
/// such that the final dimension's index value is permuted such that
/// `newColIndex = oldColIndex % vectorSize +
/// perm[rowIndex](oldColIndex/vectorSize, rowIndex)` where `rowIndex` is the
/// index for the second-to last dimension and `perm[rowIndex]` is a permutation
/// function that depends on the row Index. The permutation function is chosen
/// to ensure that sequential distributed+vectorized reads/writes down a single
/// dimension of the memref have minimal conflicts.
LogicalResult
optimizeSharedMemoryReadsAndWrites(Operation *parentOp, Value memrefValue,
                                   int64_t sharedMemoryLineSizeBytes,
                                   int64_t defaultVectorSizeBits);

std::optional<LogicalResult>
optimizeSharedMemoryReadsAndWritesOp(func::FuncOp funcOp,
                                     int64_t sharedMemoryLineSizeBytes,
                                     int64_t defaultVectorSizeBits);

} // namespace amdgpu
} // namespace mlir

#endif // MLIR_DIALECT_AMDGPU_TRANSFORMS_TRANSFORMS_H_
