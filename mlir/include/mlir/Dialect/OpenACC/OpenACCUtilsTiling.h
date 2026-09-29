//===- OpenACCUtilsTiling.h - OpenACC Loop Tiling Utilities -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains utility functions for tiling OpenACC loops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENACC_OPENACCUTILSTILING_H_
#define MLIR_DIALECT_OPENACC_OPENACCUTILSTILING_H_

#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace acc {

/// Tile a single fused acc.loop that carries all associated induction
/// variables (one IV per tile dimension).
///
/// This produces exactly two multi-IV loops, each carrying all of the tiled
/// induction variables:
///
///   - a "tile group" loop that steps over tiles: each step is the
///     original step multiplied by the tile size. It keeps the original loop's
///     gang attribute.
///   - an "element group" loop that walks the iterations inside one tile: it
///     keeps the original step, its lower bound is the current tile's starting
///     index, and its upper bound is clamped to min(original upper bound,
///     tile start + tile extent). It keeps the original loop's vector (or
///     worker) attribute.
///
/// Before Tiling:
/// \code
/// #pragma acc loop tile(tile_size1, tile_size2)
///  for (i = lb1; i < ub1; i += step1) { // original loop
///    for (j = lb2; j < ub2; j += step2) {
///      a[i,j] = i + j;
///    }
///  }
/// \endcode
///
/// After Tiling (each group is one multi-IV loop over all tiled IVs):
/// \code
///  // tile group
///  for (i = lb1; i < ub1; i += (step1 * tile_size1),
///       j = lb2; j < ub2; j += (step2 * tile_size2)) {
///    // element group
///    for (ii = i; ii < min(ub1, (step1 * tile_size1) + i); ii += step1,
///         jj = j; jj < min(ub2, (step2 * tile_size2) + j); jj += step2) {
///      a[ii,jj] = i + j;
///    }
///  }
/// \endcode
///
/// Unknown tile sizes (represented as -1 in acc dialect for `tile(*)`) are
/// resolved to the provided default tile size.
///
/// \param tileLoop The fused loop to tile.
/// \param tileSizes The tile sizes for each tiled dimension. Values of -1 are
///        treated as unknown and resolved to defaultTileSize.
/// \param defaultTileSize The default tile size to use for unknown (*) tiles.
/// \param rewriter The rewriter to use for modifications.
/// \return The tile group loop that is modified in place.
mlir::acc::LoopOp tileACCLoops(mlir::acc::LoopOp tileLoop,
                               const llvm::SmallVector<mlir::Value> &tileSizes,
                               int32_t defaultTileSize,
                               mlir::RewriterBase &rewriter);

} // namespace acc
} // namespace mlir

#endif // MLIR_DIALECT_OPENACC_OPENACCUTILSTILING_H_
