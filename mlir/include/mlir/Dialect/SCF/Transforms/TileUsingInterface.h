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
  SmallVector<unsigned> interchangeVector = {};
  SCFTilingOptions &setInterchange(ArrayRef<unsigned> interchange) {
    interchangeVector = llvm::to_vector(interchange);
    return *this;
  }
};

struct SCFTilingResult {
  Operation *tiledOp;
  SmallVector<scf::ForOp> loops;
};

/// Pattern to tile an op that implements the `TilingInterface` using
/// `scf.for` for iterating over the tiles.
struct TileUsingSCFForOp : public OpInterfaceRewritePattern<TilingInterface> {
  /// Construct a generic pattern applied to all TilingInterface ops.
  TileUsingSCFForOp(MLIRContext *context, SCFTilingOptions options,
                    PatternBenefit benefit = 1);

  /// Construct a generic pattern applied to `opName`.
  TileUsingSCFForOp(StringRef opName, MLIRContext *context,
                    SCFTilingOptions options, PatternBenefit benefit = 1);

  /// `matchAndRewrite` implementation that returns the significant transformed
  /// pieces of IR.
  FailureOr<SCFTilingResult>
  returningMatchAndRewrite(TilingInterface op, PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(TilingInterface op,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(op, rewriter);
  }

private:
  /// Options to control tiling;
  SCFTilingOptions options;
};

/// Pattern to tile and fuse a sequence of operations, by tiling the consumer
/// and fusing its producers. Note that this assumes that it is valid to
/// tile+fuse the producer into the innermost tiled loop. Its up to the caller
/// to ensure that the tile sizes provided make this fusion valid.
///
/// For example, for the following sequence
///
/// ```mlir
/// %0 = linalg.fill ...
/// %1 = linalg.matmul ... outs(%0 : ...) ...
/// ```
///
/// it is legal to fuse the fill with the matmul only if the matmul is tiled
/// along the parallel dimensions and not the reduction dimension, i.e. the tile
/// size for the reduction dimension should be 0.
struct SCFTileAndFuseResult {
  SmallVector<Operation *> tiledAndFusedOps;
  SmallVector<scf::ForOp> loops;
};
struct TileConsumerAndFuseProducersUsingSCFForOp
    : public OpInterfaceRewritePattern<TilingInterface> {

  /// Construct a generic pattern applied to all TilingInterface ops.
  TileConsumerAndFuseProducersUsingSCFForOp(MLIRContext *context,
                                            SCFTilingOptions options,
                                            PatternBenefit benefit = 1);

  /// Construct a generic pattern applied to `opName`.
  TileConsumerAndFuseProducersUsingSCFForOp(StringRef opName,
                                            MLIRContext *context,
                                            SCFTilingOptions options,
                                            PatternBenefit benefit = 1);

  /// `matchAndRewrite` implementation that returns the significant transformed
  /// pieces of IR.
  FailureOr<SCFTileAndFuseResult>
  returningMatchAndRewrite(TilingInterface op, PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(TilingInterface op,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(op, rewriter);
  }

private:
  /// This pattern uses the tiling pattern. Instead of using inheritance, use
  /// the patterns as private object that is instantiated at the same time as
  /// this pattern.
  TileUsingSCFForOp tilingPattern;
};

} // namespace scf
} // namespace mlir

#endif // MLIR_DIALECT_SCF_TRANSFORMS_TILEUSINGINTERFACE_H
