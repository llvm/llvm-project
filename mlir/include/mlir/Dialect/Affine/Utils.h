//===- Utils.h - Affine dialect utilities -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file declares a set of utilities for the affine dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFFINE_UTILS_H
#define MLIR_DIALECT_AFFINE_UTILS_H

#include "mlir/Support/LLVM.h"

namespace mlir {

class AffineForOp;
class AffineIfOp;
class AffineParallelOp;
struct LogicalResult;
class Operation;

/// Replaces parallel affine.for op with 1-d affine.parallel op.
/// mlir::isLoopParallel detect the parallel affine.for ops.
/// There is no cost model currently used to drive this parallelization.
void affineParallelize(AffineForOp forOp);

/// Hoists out affine.if/else to as high as possible, i.e., past all invariant
/// affine.fors/parallel's. Returns success if any hoisting happened; folded` is
/// set to true if the op was folded or erased. This hoisting could lead to
/// significant code expansion in some cases.
LogicalResult hoistAffineIfOp(AffineIfOp ifOp, bool *folded = nullptr);

/// Vectorizes affine loops in 'loops' using the n-D vectorization factors in
/// 'vectorSizes'. By default, each vectorization factor is applied
/// inner-to-outer to the loops of each loop nest. 'fastestVaryingPattern' can
/// be optionally used to provide a different loop vectorization order.
void vectorizeAffineLoops(
    Operation *parentOp,
    llvm::DenseSet<Operation *, DenseMapInfo<Operation *>> &loops,
    ArrayRef<int64_t> vectorSizes, ArrayRef<int64_t> fastestVaryingPattern);

} // namespace mlir

#endif // MLIR_DIALECT_AFFINE_UTILS_H
