//===- TileUsingInterface.h - Tiling ops using TilingInterface --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SCF_TRANSFORMS_TILEUSINGINTERFACE_H
#define MLIR_DIALECT_SCF_TRANSFORMS_TILEUSINGINTERFACE_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"

#include <deque>

namespace mlir {
class Operation;
class PatternRewriter;
class TilingInterface;
} // namespace mlir

namespace mlir {
namespace scf {

using SCFTileSizeComputationFunction =
    std::function<SmallVector<Value>(OpBuilder &, Operation *)>;

/// Options to use to control tiling.
struct SCFTilingOptions {
  /// Computation function that returns the tile sizes for each operation.
  /// Delayed construction of constant tile sizes should occur to interoperate
  /// with folding.
  SCFTileSizeComputationFunction tileSizeComputationFunction = nullptr;

  SCFTilingOptions &
  setTileSizeComputationFunction(SCFTileSizeComputationFunction fun) {
    tileSizeComputationFunction = std::move(fun);
    return *this;
  }
  /// Set the `tileSizeComputationFunction` to return the values `ts`. The
  /// values must not fold away when tiling. Otherwise, use a more robust
  /// `tileSizeComputationFunction`.
  SCFTilingOptions &setTileSizes(const SmallVector<Value, 4> &ts) {
    tileSizeComputationFunction = [=](OpBuilder &, Operation *) { return ts; };
    return *this;
  }
  /// Convenience function to set the `tileSizeComputationFunction` to a
  /// function that computes tile sizes at the point they are needed. Allows
  /// proper interaction with folding.
  SCFTilingOptions &setTileSizes(ArrayRef<int64_t> ts);

  /// The interchange vector to reorder the tiled loops.
  SmallVector<int64_t> interchangeVector = {};
  SCFTilingOptions &setInterchange(ArrayRef<int64_t> interchange) {
    interchangeVector = llvm::to_vector(interchange);
    return *this;
  }
};

/// Transformation information returned after tiling.
struct SCFTilingResult {
  /// The tiled operation generated.
  Operation *tiledOp;
  /// The `scf.for` operations that iterate over the tiles.
  SmallVector<scf::ForOp> loops;
  /// Values to use as replacements for the untiled op. Is the same size as the
  /// number of results of the untiled op.
  SmallVector<Value> replacements;
};

/// Method to tile an op that implements the `TilingInterface` using
/// `scf.for` for iterating over the tiles.
FailureOr<SCFTilingResult> tileUsingSCFForOp(RewriterBase &rewriter,
                                             TilingInterface op,
                                             const SCFTilingOptions &options);

/// Options used to control tile + fuse.
struct SCFTileAndFuseOptions {
  /// The tiling options used to control the tiling of the consumer.
  SCFTilingOptions tilingOptions;
  SCFTileAndFuseOptions &setTilingOptions(SCFTilingOptions options) {
    tilingOptions = options;
    return *this;
  }
};

/// Transformation information returned after tile and fuse.
struct SCFTileAndFuseResult {
  /// List of untiled operations that were fused with the tiled consumer.
  llvm::SetVector<Operation *> fusedProducers;
  /// List of tiled and fused operations generated. The first one in this list
  /// is guaranteed to be the tiled operations generated during tiling of the
  /// generated operation.
  llvm::SetVector<Operation *> tiledAndFusedOps;
  /// The `scf.for` operations that iterate over the tiles.
  SmallVector<scf::ForOp> loops;
  /// The replacement values to use for the tiled and fused operations.
  llvm::DenseMap<Value, Value> replacements;
};

/// Method to tile and fuse a sequence of operations, by tiling the consumer
/// and fusing its producers. Note that this assumes that it is valid to
/// tile+fuse the producer into the innermost tiled loop. Its up to the caller
/// to ensure that the tile sizes provided make this fusion valid.
///
/// For example, for the following sequence
///
/// ```mlir
/// %0 =
/// %1 = linalg.fill ... outs(%0 : ... )
/// %2 = linalg.matmul ... outs(%1 : ...) ...
/// ```
///
/// it is legal to fuse the fill with the matmul only if the matmul is tiled
/// along the parallel dimensions and not the reduction dimension, i.e. the tile
/// size for the reduction dimension should be 0. The resulting fused
/// transformation is
///
/// ```mlir
/// %1 = scf.for ... iter_args(%arg0 = %0)
///   %2 = tensor.extract_slice %arg0
///   %3 = linalg.fill .. outs(%2 : ... )
///   %4 = linalg.matmul .. outs(%3 : ...)
/// }
/// ```
FailureOr<SCFTileAndFuseResult>
tileConsumerAndFuseProducerGreedilyUsingSCFForOp(
    RewriterBase &rewriter, TilingInterface consumer,
    const SCFTileAndFuseOptions &options);

/// Method to lower an `op` that implements the `TilingInterface` to
/// loops/scalars.
FailureOr<SmallVector<scf::ForOp>>
lowerToLoopsUsingSCFForOp(RewriterBase &rewriter, TilingInterface op);

/// Transformation information returned after reduction tiling.
struct SCFReductionTilingResult {
  /// The partial reduction tiled op generated.
  Operation *parallelTiledOp;
  /// The final reduction operation merging all the partial reductions.
  Operation *mergeOp;
  /// Initial op
  Operation *initialOp;
  /// The `scf.for` operations that iterate over the tiles.
  SmallVector<scf::ForOp> loops;
};

/// Method to tile a reduction and generate a parallel op within a serial loop.
/// Each of the partial reductions are calculated in parallel. Then after the
/// loop all the partial reduction are merged into a final reduction.
/// For example for the following sequence
///
/// ```mlir
/// %0 = linalg.generic %in ["parallel", "reduction"]
///   : tensor<7x9xf32> -> tensor<7xf32>
/// ```
///
/// into:
///
/// ```mlir
/// %0 = linalg.fill ... : tensor<7x4xf32>
/// %1 = scf.for ... iter_args(%arg0 = %0)
///   %2 = tensor.extract_slice %arg0 : tensor<7x4xf32> -> tensor<7x?xf32>
///   %3 = tensor.extract_slice %in : tensor<7x9xf32> -> tensor<7x?xf32>
///   %4 = linalg.generic %2, %3 ["parallel", "parallel"]
///     : tensor<7x?xf32> -> tensor<7x?xf32>
///   %5 = tensor.insert_slice %3, %0[0, 0] : tensor<7x4xf32>
/// }
/// %6 = linalg.generic %1 ["parallel", "reduction"]
///   : tensor<7x4xf32> -> tensor<7xf32>
/// ```
FailureOr<scf::SCFReductionTilingResult>
tileReductionUsingScf(PatternRewriter &b, PartialReductionOpInterface op,
                      ArrayRef<OpFoldResult> tileSize);

} // namespace scf
} // namespace mlir

#endif // MLIR_DIALECT_SCF_TRANSFORMS_TILEUSINGINTERFACE_H
