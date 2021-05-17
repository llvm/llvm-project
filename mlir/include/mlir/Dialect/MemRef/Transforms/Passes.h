//===- Passes.h - MemRef Patterns and Passes --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares patterns and passes on MemRef operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MEMREF_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_MEMREF_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

/// Appends patterns for folding memref.subview ops into consumer load/store ops
/// into `patterns`.
void populateFoldSubViewOpPatterns(RewritePatternSet &patterns);

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

/// Creates an operation pass to fold memref.subview ops into consumer
/// load/store ops into `patterns`.
std::unique_ptr<Pass> createFoldSubViewOpsPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"

} // namespace memref
} // namespace mlir

#endif // MLIR_DIALECT_MEMREF_TRANSFORMS_PASSES_H
