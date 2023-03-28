//===- Transforms.h - MemRef Dialect transformations ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// This header declares functions that assit transformations in the MemRef
/// dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MEMREF_TRANSFORMS_TRANSFORMS_H
#define MLIR_DIALECT_MEMREF_TRANSFORMS_TRANSFORMS_H

#include "mlir/Support/LogicalResult.h"

namespace mlir {
class RewritePatternSet;
class RewriterBase;

namespace arith {
class WideIntEmulationConverter;
} // namespace arith

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
FailureOr<memref::AllocOp> multiBuffer(RewriterBase &rewriter,
                                       memref::AllocOp allocOp,
                                       unsigned multiplier,
                                       bool skipOverrideAnalysis = false);
/// Call into `multiBuffer` with  locally constructed IRRewriter.
FailureOr<memref::AllocOp> multiBuffer(memref::AllocOp allocOp,
                                       unsigned multiplier,
                                       bool skipOverrideAnalysis = false);

/// Appends patterns for extracting address computations from the instructions
/// with memory accesses such that these memory accesses use only a base
/// pointer.
///
/// For instance,
/// ```mlir
/// memref.load %base[%off0, ...]
/// ```
///
/// Will be rewritten in:
/// ```mlir
/// %new_base = memref.subview %base[%off0,...][1,...][1,...]
/// memref.load %new_base[%c0,...]
/// ```
void populateExtractAddressComputationsPatterns(RewritePatternSet &patterns);

} // namespace memref
} // namespace mlir

#endif
