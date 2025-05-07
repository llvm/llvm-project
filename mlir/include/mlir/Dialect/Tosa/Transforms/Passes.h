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
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/PassesEnums.h.inc"
#include "mlir/Pass/Pass.h"

namespace mlir {
class TypeConverter;
namespace tosa {

#define GEN_PASS_DECL
#include "mlir/Dialect/Tosa/Transforms/Passes.h.inc"

// Expose Rewrite Functions that decompose TOSA Ops into further TOSA Ops.
// The rewrites can be selectively added to a conversion pass.
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

void populateTosaTypeConversion(TypeConverter &converter);

std::unique_ptr<Pass> createTosaTestQuantUtilAPIPass();

#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Tosa/Transforms/Passes.h.inc"

} // namespace tosa
} // namespace mlir

#endif // MLIR_DIALECT_TOSA_TRANSFORMS_PASSES_H
