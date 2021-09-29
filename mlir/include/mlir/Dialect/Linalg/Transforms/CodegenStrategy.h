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
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {

class FuncOp;

namespace linalg {

/// Abstract Transformation class applied in a sequence that also handles state
/// through markers.
struct Transformation {
  explicit Transformation(LinalgTransformationFilter::FilterFunction f)
      : filter(f) {}
  virtual ~Transformation() = default;
  virtual void addToPassPipeline(OpPassManager &pm,
                                 LinalgTransformationFilter m) const = 0;
  LinalgTransformationFilter::FilterFunction filter = nullptr;
};

/// Represent one application of LinalgStrategyTilePass.
struct Tile : public Transformation {
  Tile(StringRef name, linalg::LinalgTilingOptions options,
       LinalgTransformationFilter::FilterFunction f = nullptr)
      : Transformation(f), opName(name), options(options) {}

  void addToPassPipeline(OpPassManager &pm,
                         LinalgTransformationFilter m) const override {
    pm.addPass(createLinalgStrategyTilePass(opName, options, m));
  }

private:
  std::string opName;
  linalg::LinalgTilingOptions options;
};

/// Represent one application of createLinalgStrategyPromotePass.
struct Promote : public Transformation {
  Promote(StringRef name, linalg::LinalgPromotionOptions options,
          LinalgTransformationFilter::FilterFunction f = nullptr)
      : Transformation(f), opName(name), options(options) {}

  void addToPassPipeline(OpPassManager &pm,
                         LinalgTransformationFilter m) const override {
    pm.addPass(createLinalgStrategyPromotePass(opName, options, m));
  }

private:
  std::string opName;
  linalg::LinalgPromotionOptions options;
};

/// Represent one application of createLinalgStrategyVectorizePass.
struct Vectorize : public Transformation {
  explicit Vectorize(linalg::LinalgVectorizationOptions options,
                     LinalgTransformationFilter::FilterFunction f = nullptr)
      : Transformation(f), opName(), options(options) {}

  Vectorize(StringRef name, linalg::LinalgVectorizationOptions options,
            LinalgTransformationFilter::FilterFunction f = nullptr)
      : Transformation(f), opName(name), options(options) {}

  void addToPassPipeline(OpPassManager &pm,
                         LinalgTransformationFilter m) const override {
    pm.addPass(createLinalgStrategyVectorizePass(opName, options, m));
  }

private:
  std::string opName;
  linalg::LinalgVectorizationOptions options;
};

/// Codegen strategy controls how a Linalg op is progressively lowered.
struct CodegenStrategy {
  /// Append a pattern to add a level of tiling for Op `opName` with tiling
  /// `options`.
  CodegenStrategy &
  tile(StringRef opName, linalg::LinalgTilingOptions options,
       LinalgTransformationFilter::FilterFunction f = nullptr) {
    transformationSequence.emplace_back(
        std::make_unique<Tile>(opName, options, f));
    return *this;
  }
  /// Conditionally append a pattern to add a level of tiling for
  /// `LinalgOpType` with tiling `options`.
  CodegenStrategy &
  tileIf(bool b, StringRef opName, linalg::LinalgTilingOptions options,
         LinalgTransformationFilter::FilterFunction f = nullptr) {
    return b ? tile(opName, options) : *this;
  }
  /// Append a pattern to add a level of promotion for `LinalgOpType` with
  /// promotion `options`.
  CodegenStrategy &
  promote(StringRef opName, linalg::LinalgPromotionOptions options,
          LinalgTransformationFilter::FilterFunction f = nullptr) {
    transformationSequence.emplace_back(
        std::make_unique<Promote>(opName, options, f));
    return *this;
  }
  /// Conditionally append a pattern to add a level of promotion for
  /// `LinalgOpType` with promotion `options`.
  CodegenStrategy &
  promoteIf(bool b, StringRef opName, linalg::LinalgPromotionOptions options,
            LinalgTransformationFilter::FilterFunction f = nullptr) {
    return b ? promote(opName, options, f) : *this;
    return *this;
  }
  /// Append a pattern to rewrite `LinalgOpType` as a vector operation.
  CodegenStrategy &
  vectorize(StringRef opName,
            LinalgTransformationFilter::FilterFunction f = nullptr) {
    assert(!opName.empty() && "expected an op name");
    transformationSequence.emplace_back(std::make_unique<Vectorize>(
        opName, linalg::LinalgVectorizationOptions(), f));
    return *this;
  }
  /// Conditionally append a pattern to rewrite `LinalgOpType` as a vector
  /// operation.
  CodegenStrategy &
  vectorizeIf(bool b, StringRef opName,
              LinalgTransformationFilter::FilterFunction f = nullptr) {
    return b ? vectorize(opName, f) : *this;
    return *this;
  }
  /// Configure the post staged-patterns late vector transformations.
  CodegenStrategy &
  setVectorTransformsOptions(vector::VectorTransformsOptions options) {
    vectorTransformOptions = options;
    return *this;
  }
  /// Configure the post staged-patterns late vector.transfer to scf
  /// conversion.
  CodegenStrategy &
  setVectorTransferToSCFOptions(VectorTransferToSCFOptions options) {
    vectorToSCFOptions = options;
    return *this;
  }
  ///
  /// Configure the application of late transformations.
  ///
  CodegenStrategy &setEnableLICM(bool val) {
    this->lateCodegenStrategyOptions.enableLICM = val;
    return *this;
  }
  CodegenStrategy &setEnableHoistRedundantVectorTransfers(bool val) {
    this->lateCodegenStrategyOptions.enableHoistRedundantVectorTransfers = val;
    return *this;
  }
  CodegenStrategy &setEnableHoistRedundantVectorTransfersOnTensor(bool val) {
    this->lateCodegenStrategyOptions
        .enableHoistRedundantVectorTransfersOnTensor = val;
    return *this;
  }
  CodegenStrategy &setEnableVectorTransferPartialRewrite(bool val) {
    this->lateCodegenStrategyOptions.enableVectorTransferPartialRewrite = val;
    return *this;
  }
  CodegenStrategy &setEnableVectorContractLowering(bool val) {
    this->lateCodegenStrategyOptions.enableVectorContractLowering = val;
    return *this;
  }
  CodegenStrategy &setEnableVectorToSCFConversion(bool val) {
    this->lateCodegenStrategyOptions.enableVectorToSCFConversion = val;
    return *this;
  }

  /// Apply the transformation patterns in sequence with cleanup
  /// transformations interleaved.
  LogicalResult transform(FuncOp func) const;
  void configurePassPipeline(OpPassManager &pm, MLIRContext *context) const;

private:
  LogicalResult postPatternTransforms(Operation *func) const;

  vector::VectorTransformsOptions vectorTransformOptions;
  VectorTransferToSCFOptions vectorToSCFOptions;
  SmallVector<std::unique_ptr<Transformation>, 4> transformationSequence;
  LateCodegenStrategyOptions lateCodegenStrategyOptions;
};

} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_TRANSFORMS_CODEGENSTRATEGY_H_
