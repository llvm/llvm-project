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
class StandardOpsDialect;
namespace tensor {
class TensorDialect;
} // namespace tensor
namespace vector {
class VectorDialect;
} // namespace vector

namespace memref {
class AllocOp;
//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

/// Collects a set of patterns to rewrite ops within the memref dialect.
void populateExpandOpsPatterns(RewritePatternSet &patterns);

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

/// Transformation to do multi-buffering/array expansion to remove dependencies
/// on the temporary allocation between consecutive loop iterations.
/// It return success if the allocation was multi-buffered and returns failure()
/// otherwise.
/// Example:
/// ```
/// %0 = memref.alloc() : memref<4x128xf32>
/// scf.for %iv = %c1 to %c1024 step %c3 {
///   memref.copy %1, %0 : memref<4x128xf32> to memref<4x128xf32>
///   "some_use"(%0) : (memref<4x128xf32>) -> ()
/// }
/// ```
/// into:
/// ```
/// %0 = memref.alloc() : memref<5x4x128xf32>
/// scf.for %iv = %c1 to %c1024 step %c3 {
///   %s = arith.subi %iv, %c1 : index
///   %d = arith.divsi %s, %c3 : index
///   %i = arith.remsi %d, %c5 : index
///   %sv = memref.subview %0[%i, 0, 0] [1, 4, 128] [1, 1, 1] :
///     memref<5x4x128xf32> to memref<4x128xf32, #map0>
///   memref.copy %1, %sv : memref<4x128xf32> to memref<4x128xf32, #map0>
///   "some_use"(%sv) : (memref<4x128xf32, $map0>) -> ()
/// }
/// ```
LogicalResult multiBuffer(memref::AllocOp allocOp, unsigned multiplier);

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

/// Creates an instance of the ExpandOps pass that legalizes memref dialect ops
/// to be convertible to LLVM. For example, `memref.reshape` gets converted to
/// `memref_reinterpret_cast`.
std::unique_ptr<Pass> createExpandOpsPass();

/// Creates an operation pass to fold memref.subview ops into consumer
/// load/store ops into `patterns`.
std::unique_ptr<Pass> createFoldSubViewOpsPass();

/// Creates an interprocedural pass to normalize memrefs to have a trivial
/// (identity) layout map.
std::unique_ptr<OperationPass<ModuleOp>> createNormalizeMemRefsPass();

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
