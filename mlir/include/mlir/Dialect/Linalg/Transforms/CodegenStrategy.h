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

/// Codegen strategy controls how a Linalg op is progressively lowered.
struct CodegenStrategy {
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
