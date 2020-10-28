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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

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

/// Holds parameters to perform n-D vectorization on a single loop nest.
/// For example, for the following loop nest:
///
/// func @vec2d(%in: memref<64x128x512xf32>, %out: memref<64x128x512xf32>) {
///   affine.for %i0 = 0 to 64 {
///     affine.for %i1 = 0 to 128 {
///       affine.for %i2 = 0 to 512 {
///         %ld = affine.load %in[%i0, %i1, %i2] : memref<64x128x512xf32>
///         affine.store %ld, %out[%i0, %i1, %i2] : memref<64x128x512xf32>
///       }
///     }
///   }
///   return
/// }
///
/// and VectorizationStrategy = 'vectorSizes = {8, 4}', 'loopToVectorDim =
/// {{i1->0}, {i2->1}}', SuperVectorizer will generate:
///
///  func @vec2d(%arg0: memref<64x128x512xf32>, %arg1: memref<64x128x512xf32>) {
///    affine.for %arg2 = 0 to 64 {
///      affine.for %arg3 = 0 to 128 step 8 {
///        affine.for %arg4 = 0 to 512 step 4 {
///          %cst = constant 0.000000e+00 : f32
///          %0 = vector.transfer_read %arg0[%arg2, %arg3, %arg4], %cst : ...
///          vector.transfer_write %0, %arg1[%arg2, %arg3, %arg4] : ...
///        }
///      }
///    }
///    return
///  }
// TODO: Hoist to a VectorizationStrategy.cpp when appropriate.
struct VectorizationStrategy {
  // Vectorization factors to apply to each target vector dimension.
  // Each factor will be applied to a different loop.
  SmallVector<int64_t, 8> vectorSizes;
  // Maps each AffineForOp vectorization candidate with its vector dimension.
  // The candidate will be vectorized using the vectorization factor in
  // 'vectorSizes' for that dimension.
  DenseMap<Operation *, unsigned> loopToVectorDim;
};

/// Vectorizes affine loops in 'loops' using the n-D vectorization factors in
/// 'vectorSizes'. By default, each vectorization factor is applied
/// inner-to-outer to the loops of each loop nest. 'fastestVaryingPattern' can
/// be optionally used to provide a different loop vectorization order.
void vectorizeAffineLoops(
    Operation *parentOp,
    llvm::DenseSet<Operation *, DenseMapInfo<Operation *>> &loops,
    ArrayRef<int64_t> vectorSizes, ArrayRef<int64_t> fastestVaryingPattern);

/// External utility to vectorize affine loops from a single loop nest using an
/// n-D vectorization strategy (see doc in VectorizationStrategy definition).
/// Loops are provided in a 2D vector container. The first dimension represents
/// the nesting level relative to the loops to be vectorized. The second
/// dimension contains the loops. This means that:
///   a) every loop in 'loops[i]' must have a parent loop in 'loops[i-1]',
///   b) a loop in 'loops[i]' may or may not have a child loop in 'loops[i+1]'.
///
/// For example, for the following loop nest:
///
///   func @vec2d(%in0: memref<64x128x512xf32>, %in1: memref<64x128x128xf32>,
///               %out0: memref<64x128x512xf32>,
///               %out1: memref<64x128x128xf32>) {
///     affine.for %i0 = 0 to 64 {
///       affine.for %i1 = 0 to 128 {
///         affine.for %i2 = 0 to 512 {
///           %ld = affine.load %in0[%i0, %i1, %i2] : memref<64x128x512xf32>
///           affine.store %ld, %out0[%i0, %i1, %i2] : memref<64x128x512xf32>
///         }
///         affine.for %i3 = 0 to 128 {
///           %ld = affine.load %in1[%i0, %i1, %i3] : memref<64x128x128xf32>
///           affine.store %ld, %out1[%i0, %i1, %i3] : memref<64x128x128xf32>
///         }
///       }
///     }
///     return
///   }
///
/// loops = {{%i0}, {%i2, %i3}}, to vectorize the outermost and the two
/// innermost loops;
/// loops = {{%i1}, {%i2, %i3}}, to vectorize the middle and the two innermost
/// loops;
/// loops = {{%i2}}, to vectorize only the first innermost loop;
/// loops = {{%i3}}, to vectorize only the second innermost loop;
/// loops = {{%i1}}, to vectorize only the middle loop.
LogicalResult
vectorizeAffineLoopNest(std::vector<SmallVector<AffineForOp, 2>> &loops,
                        const VectorizationStrategy &strategy);

/// Normalize a affine.parallel op so that lower bounds are 0 and steps are 1.
/// As currently implemented, this transformation cannot fail and will return
/// early if the op is already in a normalized form.
void normalizeAffineParallel(AffineParallelOp op);

} // namespace mlir

#endif // MLIR_DIALECT_AFFINE_UTILS_H
