//===- CodegenStrategy.h - Linalg programmable codegen strategy -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_TRANSFORMS_CODEGENSTRATEGY_H_
#define MLIR_DIALECT_LINALG_TRANSFORMS_CODEGENSTRATEGY_H_

#include <utility>

#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {

namespace linalg {

/// Abstract Transformation class applied in a sequence that also handles state
/// through markers.
struct Transformation {
  explicit Transformation(LinalgTransformationFilter::FilterFunction f)
      : filter(std::move(f)) {}
  virtual ~Transformation() = default;
  virtual void addToPassPipeline(OpPassManager &pm,
                                 LinalgTransformationFilter m) const = 0;
  LinalgTransformationFilter::FilterFunction filter = nullptr;
};

/// Represent one application of LinalgStrategyTileAndFusePass.
struct TileAndFuse : public Transformation {
  TileAndFuse(StringRef name, linalg::LinalgTilingAndFusionOptions options,
              LinalgTransformationFilter::FilterFunction f = nullptr)
      : Transformation(std::move(f)), opName(name),
        options(std::move(options)) {}

  void addToPassPipeline(OpPassManager &pm,
                         LinalgTransformationFilter m) const override {
    pm.addPass(createLinalgStrategyTileAndFusePass(opName, options, m));
  }

private:
  std::string opName;
  linalg::LinalgTilingAndFusionOptions options;
};

/// Represent one application of LinalgStrategyTilePass.
struct Tile : public Transformation {
  Tile(StringRef name, linalg::LinalgTilingOptions options,
       LinalgTransformationFilter::FilterFunction f = nullptr)
      : Transformation(std::move(f)), opName(name),
        options(std::move(options)) {}

  void addToPassPipeline(OpPassManager &pm,
                         LinalgTransformationFilter m) const override {
    pm.addPass(createLinalgStrategyTilePass(opName, options, m));
  }

private:
  std::string opName;
  linalg::LinalgTilingOptions options;
};

/// Codegen strategy controls how a Linalg op is progressively lowered.
struct CodegenStrategy {
  /// Append a pattern to tile the Op `opName` and fuse its producers with
  /// tiling and fusion `options`.
  CodegenStrategy &
  tileAndFuse(StringRef opName, const LinalgTilingAndFusionOptions &options,
              const LinalgTransformationFilter::FilterFunction &f = nullptr) {
    transformationSequence.emplace_back(
        std::make_unique<TileAndFuse>(opName, options, f));
    return *this;
  }
  /// Conditionally append a pattern to tile the Op `opName` and fuse its
  /// producers with tiling and fusion `options`.
  CodegenStrategy &
  tileAndFuseIf(bool b, StringRef opName, LinalgTilingAndFusionOptions options,
                LinalgTransformationFilter::FilterFunction f = nullptr) {
    return b ? tileAndFuse(opName, std::move(options), std::move(f)) : *this;
  }
  /// Append a pattern to add a level of tiling for Op `opName` with tiling
  /// `options`.
  CodegenStrategy &
  tile(StringRef opName, const linalg::LinalgTilingOptions &options,
       const LinalgTransformationFilter::FilterFunction &f = nullptr) {
    transformationSequence.emplace_back(
        std::make_unique<Tile>(opName, options, f));
    return *this;
  }
  /// Conditionally append a pattern to add a level of tiling for
  /// `LinalgOpType` with tiling `options`.
  CodegenStrategy &
  tileIf(bool b, StringRef opName, linalg::LinalgTilingOptions options,
         LinalgTransformationFilter::FilterFunction f = nullptr) {
    return b ? tile(opName, std::move(options), std::move(f)) : *this;
  }
  /// Configure the post staged-patterns global enabling passes options.
  CodegenStrategy &
  setVectorTransferToSCFOptions(LinalgEnablingOptions options) {
    linalgEnablingOptions = options;
    return *this;
  }

  /// Apply the transformation patterns in sequence with cleanup
  /// transformations interleaved.
  void configurePassPipeline(OpPassManager &pm, MLIRContext *context,
                             bool addEnablePass = true) const;

private:
  LogicalResult postPatternTransforms(Operation *func) const;

  LinalgEnablingOptions linalgEnablingOptions;
  SmallVector<std::unique_ptr<Transformation>, 4> transformationSequence;
};

} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_TRANSFORMS_CODEGENSTRATEGY_H_
