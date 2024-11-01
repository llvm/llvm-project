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
class ModuleOp;

namespace arith {
class WideIntEmulationConverter;
} // namespace arith

namespace func {
class FuncDialect;
} // namespace func
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

/// Appends patterns for folding memref aliasing ops into consumer load/store
/// ops into `patterns`.
void populateFoldMemRefAliasOpPatterns(RewritePatternSet &patterns);

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

/// Appends patterns for expanding memref operations that modify the metadata
/// (sizes, offset, strides) of a memref into easier to analyze constructs.
void populateExpandStridedMetadataPatterns(RewritePatternSet &patterns);

/// Appends patterns for emulating wide integer memref operations with ops over
/// narrower integer types.
void populateMemRefWideIntEmulationPatterns(
    arith::WideIntEmulationConverter &typeConverter,
    RewritePatternSet &patterns);

/// Appends type converions for emulating wide integer memref operations with
/// ops over narrowe integer types.
void populateMemRefWideIntEmulationConversions(
    arith::WideIntEmulationConverter &typeConverter);

/// Transformation to do multi-buffering/array expansion to remove dependencies
/// on the temporary allocation between consecutive loop iterations.
/// It returns the new allocation if the original allocation was multi-buffered
/// and returns failure() otherwise.
/// When `skipOverrideAnalysis`, the pass will apply the transformation
/// without checking thwt the buffer is overrided at the beginning of each
/// iteration. This implies that user knows that there is no data carried across
/// loop iterations. Example:
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
///     memref<5x4x128xf32> to memref<4x128xf32, strided<[128, 1], offset: ?>>
///   memref.copy %1, %sv : memref<4x128xf32> to memref<4x128xf32, strided<...>>
///   "some_use"(%sv) : (memref<4x128xf32, strided<...>) -> ()
/// }
/// ```
FailureOr<memref::AllocOp> multiBuffer(memref::AllocOp allocOp,
                                       unsigned multiplier,
                                       bool skipOverrideAnalysis = false);

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"

/// Creates an instance of the ExpandOps pass that legalizes memref dialect ops
/// to be convertible to LLVM. For example, `memref.reshape` gets converted to
/// `memref_reinterpret_cast`.
std::unique_ptr<Pass> createExpandOpsPass();

/// Creates an operation pass to fold memref aliasing ops into consumer
/// load/store ops into `patterns`.
std::unique_ptr<Pass> createFoldMemRefAliasOpsPass();

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

/// Creates an operation pass to expand some memref operation into
/// easier to reason about operations.
std::unique_ptr<Pass> createExpandStridedMetadataPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"

} // namespace memref
} // namespace mlir

#endif // MLIR_DIALECT_MEMREF_TRANSFORMS_PASSES_H
