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

/// Uncollapse tile loops with multiple IVs and collapseCount < tileCount.
/// This is used to prepare loops for tiling when the collapse count is less
/// than the tile count.
///
/// \param origLoop The original loop operation to uncollapse.
/// \param tileCount The number of tile dimensions.
/// \param collapseCount The collapse count from the original loop.
/// \param rewriter The rewriter to use for modifications.
/// \return A vector of uncollapsed loop operations.
llvm::SmallVector<mlir::acc::LoopOp>
uncollapseLoops(mlir::acc::LoopOp origLoop, unsigned tileCount,
                unsigned collapseCount, mlir::RewriterBase &rewriter);

/// Tile ACC loops according to the given tile sizes.
///
/// Tiling a 2-level nested loop will create two 'tile' loops containing two
/// 'element' loops. The transformation looks like:
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
/// After Tiling:
/// \code
///  for (i = lb1; i < ub1; i += (step1 * tile_size1)) { // tile loop 1
///    for (j = lb2; j < ub2; j += (step2 * tile_size2)) { // tile loop 2
///      for (ii = i; ii < min(ub1, (step1 * tile_size1) + i); ii += step1) {
///      // element loop 1
///        for (jj = j; jj < min(ub2, (step2 * tile_size2) + j); jj += step2)
///        { // element loop 2
///          a[ii,jj] = i + j;
///        }
///      }
///    }
///  }
/// \endcode
///
/// Unknown tile sizes (represented as -1 in acc dialect for `tile(*)`) are
/// resolved to the provided default tile size.
///
/// \param tileLoops The loops to tile (outermost first).
/// \param tileSizes The tile sizes for each dimension. Values of -1 are
///        treated as unknown and resolved to defaultTileSize.
/// \param defaultTileSize The default tile size to use for unknown (*) tiles.
/// \param rewriter The rewriter to use for modifications.
/// \return The outermost loop after tiling.
mlir::acc::LoopOp tileACCLoops(llvm::SmallVector<mlir::acc::LoopOp> &tileLoops,
                               const llvm::SmallVector<mlir::Value> &tileSizes,
                               int32_t defaultTileSize,
                               mlir::RewriterBase &rewriter);

} // namespace acc
} // namespace mlir

#endif // MLIR_DIALECT_OPENACC_OPENACCUTILSTILING_H_
