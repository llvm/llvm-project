//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GPU_PASSES_H_
#define MLIR_DIALECT_GPU_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
std::unique_ptr<OperationPass<ModuleOp>> createGpuKernelOutliningPass();

/// Collect a set of patterns to rewrite ops within the GPU dialect.
void populateGpuRewritePatterns(MLIRContext *context,
                                OwningRewritePatternList &patterns);

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/GPU/Passes.h.inc"

} // namespace mlir

#endif // MLIR_DIALECT_GPU_PASSES_H_
