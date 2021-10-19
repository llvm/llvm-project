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

class AffineDialect;
namespace tensor {
class TensorDialect;
} // namespace tensor
namespace vector {
class VectorDialect;
} // namespace vector

namespace memref {

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

/// Appends patterns for folding memref.subview ops into consumer load/store ops
/// into `patterns`.
void populateFoldSubViewOpPatterns(RewritePatternSet &patterns);

/// Appends patterns that resolve `memref.dim` operations with values that are
/// defined by operations that implement the
/// `ReifyRankedShapeTypeShapeOpInterface`, in terms of shapes of its input
/// operands.
void populateResolveRankedShapeTypeResultDimsPatterns(
    RewritePatternSet &patterns);

/// Appends patterns that resolve `memref.dim` operations with values that are
/// defined by operations that implement the `InferShapedTypeOpInterface`, in
/// terms of shapes of its input operands.
void populateResolveShapedTypeResultDimsPatterns(RewritePatternSet &patterns);

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

/// Creates an operation pass to fold memref.subview ops into consumer
/// load/store ops into `patterns`.
std::unique_ptr<Pass> createFoldSubViewOpsPass();

/// Creates an operation pass to resolve `memref.dim` operations with values
/// that are defined by operations that implement the
/// `ReifyRankedShapeTypeShapeOpInterface`, in terms of shapes of its input
/// operands.
std::unique_ptr<Pass> createResolveRankedShapeTypeResultDimsPass();

/// Creates an operation pass to resolve `memref.dim` operations with values
/// that are defined by operations that implement the
/// `InferShapedTypeOpInterface` or the `ReifyRankedShapeTypeShapeOpInterface`,
/// in terms of shapes of its input operands.
std::unique_ptr<Pass> createResolveShapedTypeResultDimsPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"

} // namespace memref
} // namespace mlir

#endif // MLIR_DIALECT_MEMREF_TRANSFORMS_PASSES_H
