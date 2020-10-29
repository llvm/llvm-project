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
/// Replaces `gpu.launch` with `gpu.launch_func` by moving the region into
/// a separate kernel function.
std::unique_ptr<OperationPass<ModuleOp>> createGpuKernelOutliningPass();

/// Rewrites a function region so that GPU ops execute asynchronously.
std::unique_ptr<OperationPass<FuncOp>> createGpuAsyncRegionPass();

/// Collect a set of patterns to rewrite all-reduce ops within the GPU dialect.
void populateGpuAllReducePatterns(MLIRContext *context,
                                  OwningRewritePatternList &patterns);

/// Collect all patterns to rewrite ops within the GPU dialect.
inline void populateGpuRewritePatterns(MLIRContext *context,
                                       OwningRewritePatternList &patterns) {
  populateGpuAllReducePatterns(context, patterns);
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/GPU/Passes.h.inc"

} // namespace mlir

#endif // MLIR_DIALECT_GPU_PASSES_H_
