//===-- Passes.h - TOSA optimization pass declarations ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the optimization passes for the TOSA Dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TOSA_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_TOSA_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/Transforms/PassesEnums.h.inc"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tosa {

#define GEN_PASS_DECL
#include "mlir/Dialect/Tosa/Transforms/Passes.h.inc"

// Expose Rewrite Functions that decompose TOSA Ops into further TOSA Ops.
// The rewrites can be selectively added to a conversion pass.
void populateTosaDecomposeConv2D(MLIRContext *ctx, RewritePatternSet &patterns);
void populateTosaDecomposeTransposeConv(MLIRContext *ctx,
                                        RewritePatternSet &patterns);
void populateTosaDecomposeDepthwise(MLIRContext *ctx,
                                    RewritePatternSet &patterns);
void populateTosaFoldConstantReciprocalPatterns(MLIRContext *ctx,
                                                RewritePatternSet &patterns);
void populateTosaFoldConstantTransposePatterns(MLIRContext *ctx,
                                               RewritePatternSet &patterns);
void populateTosaConstantReduction(MLIRContext *ctx,
                                   RewritePatternSet &patterns,
                                   bool aggressiveReduceConstant);

std::unique_ptr<Pass> createTosaLayerwiseConstantFoldPass();
std::unique_ptr<Pass> createTosaLayerwiseConstantFoldPass(
    const TosaLayerwiseConstantFoldPassOptions &options);
std::unique_ptr<Pass> createTosaInferShapesPass();
std::unique_ptr<Pass> createTosaMakeBroadcastablePass();
std::unique_ptr<Pass> createTosaTestQuantUtilAPIPass();
std::unique_ptr<Pass> createTosaOptionalDecompositions();

struct ValidationOptions {
  /// Validate if operations match for the given profile.
  TosaProfileEnum profile = TosaProfileEnum::Undefined;
  ValidationOptions &setProfile(TosaProfileEnum profile) {
    this->profile = profile;
    return *this;
  }
  /// Verify if the properties of certain operations align the spec requirement.
  bool strictOperationSpecAlignment = false;
  ValidationOptions &enableStrictOperationSpecAlignment(bool enable = true) {
    strictOperationSpecAlignment = enable;
    return *this;
  }
  /// Validate if operator parameters are within specfication for the given
  /// level.
  TosaLevelEnum level = TosaLevelEnum::EightK;
  ValidationOptions &setLevel(TosaLevelEnum level) {
    this->level = level;
    return *this;
  }
};

std::unique_ptr<Pass> createTosaValidationPass(
    ValidationOptions const &options = ValidationOptions());

#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Tosa/Transforms/Passes.h.inc"

} // namespace tosa
} // namespace mlir

#endif // MLIR_DIALECT_TOSA_TRANSFORMS_PASSES_H
