//===- Transforms.h - MemRef Dialect transformations ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// This header declares functions that assist transformations in the MemRef
/// dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MEMREF_TRANSFORMS_TRANSFORMS_H
#define MLIR_DIALECT_MEMREF_TRANSFORMS_TRANSFORMS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class Location;
class OpBuilder;
class RewritePatternSet;
class RewriterBase;
class Value;
class ValueRange;
class ReifyRankedShapedTypeOpInterface;

namespace arith {
class WideIntEmulationConverter;
class NarrowTypeEmulationConverter;
} // namespace arith

namespace memref {
class AllocOp;
class AllocaOp;
class CollapseShapeOp;
class DeallocOp;
class ExpandShapeOp;

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
/// `ReifyRankedShapedTypeOpInterface`, in terms of shapes of its input
/// operands.
void populateResolveRankedShapedTypeResultDimsPatterns(
    RewritePatternSet &patterns);

/// Appends patterns that resolve `memref.dim` operations with values that are
/// defined by operations that implement the `InferShapedTypeOpInterface`, in
/// terms of shapes of its input operands.
void populateResolveShapedTypeResultDimsPatterns(RewritePatternSet &patterns);

/// Appends patterns for expanding memref operations that modify the metadata
/// (sizes, offset, strides) of a memref into easier to analyze constructs.
void populateExpandStridedMetadataPatterns(RewritePatternSet &patterns);

/// Appends patterns for resolving `memref.extract_strided_metadata` into
/// `memref.extract_strided_metadata` of its source.
void populateResolveExtractStridedMetadataPatterns(RewritePatternSet &patterns);

/// Appends patterns for expanding `memref.realloc` operations.
void populateExpandReallocPatterns(RewritePatternSet &patterns,
                                   bool emitDeallocs = true);

/// Appends patterns for emulating wide integer memref operations with ops over
/// narrower integer types.
void populateMemRefWideIntEmulationPatterns(
    const arith::WideIntEmulationConverter &typeConverter,
    RewritePatternSet &patterns);

/// Appends type conversions for emulating wide integer memref operations with
/// ops over narrowe integer types.
void populateMemRefWideIntEmulationConversions(
    arith::WideIntEmulationConverter &typeConverter);

/// Appends patterns for emulating memref operations over narrow types with ops
/// over wider types.
void populateMemRefNarrowTypeEmulationPatterns(
    const arith::NarrowTypeEmulationConverter &typeConverter,
    RewritePatternSet &patterns);

/// Appends type conversions for emulating memref operations over narrow types
/// with ops over wider types.
void populateMemRefNarrowTypeEmulationConversions(
    arith::NarrowTypeEmulationConverter &typeConverter);

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

/// Patterns for flattening multi-dimensional memref operations into
/// one-dimensional memref operations.
void populateFlattenVectorOpsOnMemrefPatterns(RewritePatternSet &patterns);
void populateFlattenMemrefOpsPatterns(RewritePatternSet &patterns);
void populateFlattenMemrefsPatterns(RewritePatternSet &patterns);

/// Build a new memref::AllocaOp whose dynamic sizes are independent of all
/// given independencies. If the op is already independent of all
/// independencies, the same AllocaOp result is returned.
///
/// Failure indicates the no suitable upper bound for the dynamic sizes could be
/// found.
FailureOr<Value> buildIndependentOp(OpBuilder &b, AllocaOp allocaOp,
                                    ValueRange independencies);

/// Build a new memref::AllocaOp whose dynamic sizes are independent of all
/// given independencies. If the op is already independent of all
/// independencies, the same AllocaOp result is returned.
///
/// The original AllocaOp is replaced with the new one, wrapped in a SubviewOp.
/// The result type of the replacement is different from the original allocation
/// type: it has the same shape, but a different layout map. This function
/// updates all users that do not have a memref result or memref region block
/// argument, and some frequently used memref dialect ops (such as
/// memref.subview). It does not update other uses such as the init_arg of an
/// scf.for op. Such uses are wrapped in unrealized_conversion_cast.
///
/// Failure indicates the no suitable upper bound for the dynamic sizes could be
/// found.
///
/// Example (make independent of %iv):
/// ```
/// scf.for %iv = %c0 to %sz step %c1 {
///   %0 = memref.alloca(%iv) : memref<?xf32>
///   %1 = memref.subview %0[0][5][1] : ...
///   linalg.generic outs(%1 : ...) ...
///   %2 = scf.for ... iter_arg(%arg0 = %0) ...
///   ...
/// }
/// ```
///
/// The above IR is rewritten to:
///
/// ```
/// scf.for %iv = %c0 to %sz step %c1 {
///   %0 = memref.alloca(%sz - 1) : memref<?xf32>
///   %0_subview = memref.subview %0[0][%iv][1]
///       : memref<?xf32> to memref<?xf32, #map>
///   %1 = memref.subview %0_subview[0][5][1] : ...
///   linalg.generic outs(%1 : ...) ...
///   %cast = unrealized_conversion_cast %0_subview
///       : memref<?xf32, #map> to memref<?xf32>
///   %2 = scf.for ... iter_arg(%arg0 = %cast) ...
///  ...
/// }
/// ```
FailureOr<Value> replaceWithIndependentOp(RewriterBase &rewriter,
                                          memref::AllocaOp allocaOp,
                                          ValueRange independencies);

/// Replaces the given `alloc` with the corresponding `alloca` and returns it if
/// the following conditions are met:
///   - the corresponding dealloc is available in the same block as the alloc;
///   - the filter, if provided, succeeds on the alloc/dealloc pair.
/// Otherwise returns nullptr and leaves the IR unchanged.
memref::AllocaOp allocToAlloca(
    RewriterBase &rewriter, memref::AllocOp alloc,
    function_ref<bool(memref::AllocOp, memref::DeallocOp)> filter = nullptr);

/// Compute the expanded sizes of the given \p expandShape for the
/// \p groupId-th reassociation group.
/// \p origSizes hold the sizes of the source shape as values.
/// This is used to compute the new sizes in cases of dynamic shapes.
///
/// sizes#i =
///     baseSizes#groupId / product(expandShapeSizes#j,
///                                  for j in group excluding reassIdx#i)
/// Where reassIdx#i is the reassociation index at index i in \p groupId.
///
/// \post result.size() == expandShape.getReassociationIndices()[groupId].size()
///
/// TODO: Move this utility function directly within ExpandShapeOp. For now,
/// this is not possible because this function uses the Affine dialect and the
/// MemRef dialect cannot depend on the Affine dialect.
SmallVector<OpFoldResult> getExpandedSizes(ExpandShapeOp expandShape,
                                           OpBuilder &builder,
                                           ArrayRef<OpFoldResult> origSizes,
                                           unsigned groupId);

/// Compute the expanded strides of the given \p expandShape for the
/// \p groupId-th reassociation group.
/// \p origStrides and \p origSizes hold respectively the strides and sizes
/// of the source shape as values.
/// This is used to compute the strides in cases of dynamic shapes and/or
/// dynamic stride for this reassociation group.
///
/// strides#i =
///     origStrides#reassDim * product(expandShapeSizes#j, for j in
///                                    reassIdx#i+1..reassIdx#i+group.size-1)
///
/// Where reassIdx#i is the reassociation index for at index i in \p groupId
/// and expandShapeSizes#j is either:
/// - The constant size at dimension j, derived directly from the result type of
///   the expand_shape op, or
/// - An affine expression: baseSizes#reassDim / product of all constant sizes
///   in expandShapeSizes. (Remember expandShapeSizes has at most one dynamic
///   element.)
///
/// \post result.size() == expandShape.getReassociationIndices()[groupId].size()
///
/// TODO: Move this utility function directly within ExpandShapeOp. For now,
/// this is not possible because this function uses the Affine dialect and the
/// MemRef dialect cannot depend on the Affine dialect.
SmallVector<OpFoldResult> getExpandedStrides(ExpandShapeOp expandShape,
                                             OpBuilder &builder,
                                             ArrayRef<OpFoldResult> origSizes,
                                             ArrayRef<OpFoldResult> origStrides,
                                             unsigned groupId);

/// Produce an OpFoldResult object with \p builder at \p loc representing
/// `prod(valueOrConstant#i, for i in {indices})`,
/// where valueOrConstant#i is maybeConstant[i] when \p isDymamic is false,
/// values[i] otherwise.
///
/// \pre for all index in indices: index < values.size()
/// \pre for all index in indices: index < maybeConstants.size()
OpFoldResult getProductOfValues(ArrayRef<int64_t> indices, OpBuilder &builder,
                                Location loc, ArrayRef<int64_t> maybeConstants,
                                ArrayRef<OpFoldResult> values,
                                llvm::function_ref<bool(int64_t)> isDynamic);

/// Compute the collapsed size of the given \p collapseShape for the
/// \p groupId-th reassociation group.
/// \p origSizes hold the sizes of the source shape as values.
/// This is used to compute the new sizes in cases of dynamic shapes.
///
/// TODO: Move this utility function directly within CollapseShapeOp. For now,
/// this is not possible because this function uses the Affine dialect and the
/// MemRef dialect cannot depend on the Affine dialect.
SmallVector<OpFoldResult> getCollapsedSize(CollapseShapeOp collapseShape,
                                           OpBuilder &builder,
                                           ArrayRef<OpFoldResult> origSizes,
                                           unsigned groupId);

/// Compute the collapsed stride of the given \p collpaseShape for the
/// \p groupId-th reassociation group.
/// \p origStrides and \p origSizes hold respectively the strides and sizes
/// of the source shape as values.
/// This is used to compute the strides in cases of dynamic shapes and/or
/// dynamic stride for this reassociation group.
///
/// Conceptually this helper function returns the stride of the inner most
/// dimension of that group in the original shape.
///
/// \post result.size() == 1, in other words, each group collapse to one
/// dimension.
SmallVector<OpFoldResult> getCollapsedStride(CollapseShapeOp collapseShape,
                                             OpBuilder &builder,
                                             ArrayRef<OpFoldResult> origSizes,
                                             ArrayRef<OpFoldResult> origStrides,
                                             unsigned groupId);

} // namespace memref
} // namespace mlir

#endif
