//===- Transforms.h - Pass Entrypoints --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines transformations on loop operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LOOPOPS_TRANSFORMS_H_
#define MLIR_DIALECT_LOOPOPS_TRANSFORMS_H_

#include "llvm/ADT/ArrayRef.h"

namespace mlir {

class Region;

namespace loop {

class ParallelOp;

/// Fuses all adjacent loop.parallel operations with identical bounds and step
/// into one loop.parallel operations. Uses a naive aliasing and dependency
/// analysis.
void naivelyFuseParallelOps(Region &region);

/// Tile a parallel loop of the form
///   loop.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
///                                             step (%arg4, %arg5)
///
/// into
///   loop.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
///                                             step (%arg4*tileSize[0],
///                                                   %arg5*tileSize[1])
///     loop.parallel (%j0, %j1) = (0, 0) to (min(tileSize[0], %arg2-%j0)
///                                           min(tileSize[1], %arg3-%j1))
///                                        step (%arg4, %arg5)
/// The old loop is replaced with the new one.
void tileParallelLoop(ParallelOp op, ArrayRef<int64_t> tileSizes);

} // namespace loop
} // namespace mlir

#endif // MLIR_DIALECT_LOOPOPS_TRANSFORMS_H_
