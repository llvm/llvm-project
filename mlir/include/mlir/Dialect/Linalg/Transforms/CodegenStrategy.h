//===- CodegenStrategy.h - Linalg programmable codegen strategy -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_TRANSFORMS_CODEGENSTRATEGY_H_
#define MLIR_DIALECT_LINALG_TRANSFORMS_CODEGENSTRATEGY_H_

#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

namespace mlir {

class FuncOp;

namespace linalg {

/// Abstract Transformation class applied in a sequence that also handles state
/// through markers.
struct Transformation {
  virtual ~Transformation() = default;
  virtual OwningRewritePatternList
  buildRewritePatterns(MLIRContext *context, linalg::LinalgMarker m) = 0;
  linalg::LinalgMarker marker;
};

/// Promotion transformation enqueues a particular stage-1 pattern for
/// `Tile<LinalgOpType>`with the appropriate `options`.
template <typename LinalgOpType>
struct Tile : public Transformation {
  explicit Tile(linalg::LinalgTilingOptions options) : options(options) {}

  OwningRewritePatternList
  buildRewritePatterns(MLIRContext *context, linalg::LinalgMarker m) override {
    OwningRewritePatternList tilingPatterns;
    tilingPatterns.insert<linalg::LinalgTilingPattern<LinalgOpType>>(
        context, options, m);
    return tilingPatterns;
  }

private:
  linalg::LinalgTilingOptions options;
};

/// Promotion transformation enqueues a particular stage-1 pattern for
/// `Promote<LinalgOpType>`with the appropriate `options`.
template <typename LinalgOpType>
struct Promote : public Transformation {
  explicit Promote(linalg::LinalgPromotionOptions options) : options(options) {}

  OwningRewritePatternList
  buildRewritePatterns(MLIRContext *context, linalg::LinalgMarker m) override {
    OwningRewritePatternList promotionPatterns;
    promotionPatterns.insert<linalg::LinalgPromotionPattern<LinalgOpType>>(
        context, options, m);
    return promotionPatterns;
  }

private:
  linalg::LinalgPromotionOptions options;
};

/// Vectorization transformation enqueues a particular stage-1 pattern for
/// `LinalgVectorizationPattern<LinalgOpType>` as well as copy to vector
/// transfer rewrite forwarding patterns.
template <typename LinalgOpType>
struct Vectorize : public Transformation {
  OwningRewritePatternList
  buildRewritePatterns(MLIRContext *context, linalg::LinalgMarker m) override {
    OwningRewritePatternList vectorizationPatterns;
    // FillOp may interfere with forwarding patterns atm, so we bump up the
    // priority of LinalgCopyVTRForwardingPattern /
    // LinalgCopyVTWForwardingPattern.
    vectorizationPatterns
        .insert<linalg::LinalgVectorizationPattern<LinalgOpType>>(context, m);
    vectorizationPatterns.insert<linalg::LinalgCopyVTRForwardingPattern,
                                 linalg::LinalgCopyVTWForwardingPattern>(
        context, /*benefit=*/2);
    return vectorizationPatterns;
  }
};

/// Codegen strategy controls how a Linalg op is progressively lowered.
/// The application uses a 3-level staged patterns strategy which allows
/// ordering transformations by using the Linalg `applyStagedPatterns` function,
/// where:
///   1. The first stage consists of the successive `tile`, `promote` and
///   `vectorize` patterns, applied sequentially.
///   2. The second stage consists of common local canonicalization patterns
///   that are applied eagerly after each stage-1 pattern.
///   3. the third stage consists of more global transformation, also applied
///   eagerly, after all stage-2 patterns. Such more global transformations
struct CodegenStrategy {
  /// Append a pattern to add a level of tiling for `LinalgOpType` with tiling
  /// `options`.
  template <typename LinalgOpType>
  CodegenStrategy &tile(linalg::LinalgTilingOptions options) {
    transformationSequence.emplace_back(new Tile<LinalgOpType>(options));
    return *this;
  }
  /// Conditionally append a pattern to add a level of tiling for `LinalgOpType`
  /// with tiling `options`.
  template <typename LinalgOpType>
  CodegenStrategy &tileIf(bool b, linalg::LinalgTilingOptions options) {
    return b ? tile<LinalgOpType>(options) : *this;
  }
  /// Append a pattern to add a level of promotion for `LinalgOpType` with
  /// promotion `options`.
  template <typename LinalgOpType>
  CodegenStrategy &promote(linalg::LinalgPromotionOptions options) {
    transformationSequence.emplace_back(new Promote<LinalgOpType>(options));
    return *this;
  }
  /// Conditionally append a pattern to add a level of promotion for
  /// `LinalgOpType` with promotion `options`.
  template <typename LinalgOpType>
  CodegenStrategy &promoteIf(bool b, linalg::LinalgPromotionOptions options) {
    return b ? promote<LinalgOpType>(options) : *this;
    return *this;
  }
  /// Append a pattern to rewrite `LinalgOpType` as a vector operation.
  template <typename LinalgOpType>
  CodegenStrategy &vectorize() {
    transformationSequence.emplace_back(new Vectorize<LinalgOpType>());
    return *this;
  }
  /// Conditionally append a pattern to rewrite `LinalgOpType` as a vector
  /// operation.
  template <typename LinalgOpType>
  CodegenStrategy &vectorizeIf(bool b) {
    return b ? vectorize<LinalgOpType>() : *this;
    return *this;
  }
  /// Configure the post staged-patterns late vector transformations.
  CodegenStrategy &
  setVectorTransformsOptions(vector::VectorTransformsOptions options) {
    vectorTransformsOptions = options;
    return *this;
  }
  /// Configure the post staged-patterns late vector.transfer to scf conversion.
  CodegenStrategy &
  setVectorTransferToSCFOptions(VectorTransferToSCFOptions options) {
    vectorToSCFOptions = options;
    return *this;
  }

  /// Apply the transformation patterns in sequence with cleanup transformations
  /// interleaved.
  void transform(FuncOp func) const;

private:
  LogicalResult postPatternTransforms(Operation *func) const;

  vector::VectorTransformsOptions vectorTransformsOptions;
  VectorTransferToSCFOptions vectorToSCFOptions;
  SmallVector<std::unique_ptr<Transformation>, 4> transformationSequence;
};

} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_TRANSFORMS_CODEGENSTRATEGY_H_
