//===- Transforms.h - Linalg transformations as patterns --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_LINALG_TRANSFORMS_TRANSFORMS_H_
#define DIALECT_LINALG_TRANSFORMS_TRANSFORMS_H_

#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/Bufferize.h"
#include "llvm/ADT/SmallBitVector.h"

namespace mlir {

class BufferAssignmentTypeConverter;

namespace linalg {

struct LinalgFusionOptions;
struct LinalgTilingOptions;

//===----------------------------------------------------------------------===//
// Transformations exposed as function calls.
//===----------------------------------------------------------------------===//
using LinalgLoops = SmallVector<Operation *, 4>;

struct TiledLinalgOp {
  LinalgOp op;
  SmallVector<Operation *, 8> loops;
  SmallVector<Value, 4> tensorResults;
};

struct TiledAndFusedLinalgOps {
  LinalgOp op;
  SmallVector<LinalgOp, 1> fusedProducers;
  SmallVector<LinalgOp, 1> originalProducers;
  SmallVector<Operation *, 4> fusedLoops;
  SmallVector<Operation *, 4> unfusedLoops;
};

/// Populates patterns for vectorization of all ConvN-D ops.
void populateConvVectorizationPatterns(
    MLIRContext *context, SmallVectorImpl<OwningRewritePatternList> &patterns,
    ArrayRef<int64_t> tileSizes);

/// Populates the given list with patterns to convert Linalg operations on
/// tensors to buffers.
void populateConvertLinalgOnTensorsToBuffersPatterns(
    MLIRContext *context, BufferAssignmentTypeConverter &converter,
    OwningRewritePatternList &patterns);

/// Performs standalone tiling of a single LinalgOp by `tileSizes`.
/// and permute the loop nest according to `interchangeVector`
/// The permutation is expressed as a list of integers that specify
/// the new ordering of the loop nest. The length of `interchangeVector`
/// must be equal to the length of `tileSizes`.
/// An empty vector is interpreted as the identity permutation and the
/// transformation returns early.
///
/// Returns a struct containing the tiled loops in the specified order
/// and the cloned op if successful, llvm::None otherwise.
///
/// E.g. the permutation `(i,j,k) -> (j,k,i)` is expressed by
/// `interchangeVector = [1,2,0]`. All values in `interchangeVector` must be
/// integers, in the range 0..`tileSizes.size()` without duplications
/// (i.e. `[1,1,2]` is an invalid permutation).
Optional<TiledLinalgOp> tileLinalgOp(OpBuilder &b, LinalgOp op,
                                     const LinalgTilingOptions &options);

/// Tile and fuse the `op` with its producers. The tile and fuse proceeds in
/// three steps
/// - Find tile loops that are fusable with its producer tile loops (a.k.a. tile
///   + fuse loops).
/// - Tile just these loops of the consumer (root operation) and fuse with
///   the producer.
/// - Tile again the tiled consumer operation produced above to do rest of
///   the tiling specified by the `tilingOptions`.
///
/// For example, consider the sequence of matmul below
///
///   linalg.matmul ins(%arg0, %arg1 : memref<256x32xf32>, memref<32x32xf32>)
///                 outs(%arg2 : memref<256x32xf32>)
///   linalg.matmul ins(%arg2, %arg3 : memref<256x32xf32>, memref<32x32xf32>)
///                 outs(%arg4 : memref<256x32xf32>)
///
/// It is legal to fuse the RAW dependence (through %arg2) by only fusing the
/// matmuls row-wise. For example, the fused computation for the above is shown
/// below. The outer `scf.parallel` loop is the "fused" loop obtained by tiling
/// along the rows of the matrix. The entire rows of the first matmul operation
/// need to be computed before they can be used for the second matmul. The
/// second matmul is further tiled (similar to normal tiling).
///
/// #map0 = affine_map<(d0, d1)[s0] -> (d0 * 32 + s0 + d1)>
/// #map1 = affine_map<(d0, d1) -> (d0 * 32 + d1)>
/// scf.parallel (%arg5) = (%c0) to (%c256) step (%c16) {
///   %0 = subview %arg2[%arg5, 0] [16, 32] [1, 1]
///     : memref<256x32xf32> to memref<16x32xf32, #map0>
///   %1 = subview %arg4[%arg5, 0] [16, 32] [1, 1]
///     : memref<256x32xf32> to memref<16x32xf32, #map0>
///   %2 = subview %arg0[%arg5, 0] [16, 32] [1, 1]
///     : memref<256x32xf32> to memref<16x32xf32, #map0>
///   %3 = subview %arg1[0, 0] [32, 32] [1, 1]
///     : memref<32x32xf32> to memref<32x32xf32, #map1>
///   linalg.matmul
///     ins(%2, %3 : memref<16x32xf32, #map0>, memref<32x32xf32, #map1>)
///     outs(%0 : memref<16x32xf32, #map0>)
///   scf.parallel (%arg6) = (%c0) to (%c32) step (%c8) {
///   scf.for %arg7 = %c0 to %c32 step %c4 {
///     %4 = subview %0[0, %arg7] [16, 4] [1, 1]
///       : memref<16x32xf32, #map0> to memref<16x4xf32, #map0>
///     %5 = subview %arg3[%arg7, %arg6] [4, 8] [1, 1]
///       : memref<32x32xf32> to memref<4x8xf32, #map0>
///     %6 = subview %1[0, %arg6] [16, 8] [1, 1]
///       : memref<16x32xf32, #map0> to memref<16x8xf32, #map0>
///     linalg.matmul
///       ins(%4, %5 : memref<16x4xf32, #map0>, memref<4x8xf32, #map0>)
///       outs(%6 : memref<16x8xf32, #map0>)
///     }
///     scf.yield
///   }
///   scf.yield
/// }
///
/// The following tiling options are handled differently in tile+fuse (compared
/// to tile only)
/// - Interchange of the tiling loops is not supported right now.
/// - Distribution is only done for the tile+fuse loops. The tiled loops
///   generated by the second tiling is not distributed.
Optional<TiledAndFusedLinalgOps>
tileAndFuseLinalgOps(PatternRewriter &rewriter, LinalgOp op,
                     const LinalgDependenceGraph &dependenceGraph,
                     const LinalgTilingOptions &tilingOptions,
                     const LinalgFusionOptions &fusionOptions);

/// Interchanges the `iterator_types` and `iterator_maps` dimensions of `op`.
/// This is an in-place transformation controlled by `interchangeVector`.
/// An empty vector is interpreted as the identity permutation and the
/// transformation returns early.
///
/// E.g. the permutation `(i,j,k) -> (j,k,i)` is expressed with
/// `interchangeVector = [1,2,0]`. All values in `interchangeVector` must be
/// integers, in the range 0..`op.rank` without duplications
/// (i.e. `[1,1,2]` is an invalid permutation).
LinalgOp interchange(LinalgOp op, ArrayRef<unsigned> interchangeVector);

/// Callback function type used to perform the allocation for the promoted
/// `subView`. In `boundingSubViewsize` a best attempt is made to find the
/// smallest constant value for the size of the buffer needed for each
/// dimension. If that is not possible, contains the dynamic size of the
/// subview. The call back should return the buffer to use.
using AllocBufferCallbackFn = std::function<Optional<Value>(
    OpBuilder &b, SubViewOp subView, ArrayRef<Value> boundingSubViewSize,
    OperationFolder *folder)>;

/// Callback function type used to deallocate the buffers used to hold the
/// promoted subview.
using DeallocBufferCallbackFn =
    std::function<LogicalResult(OpBuilder &b, Value buffer)>;

/// Callback function type used to insert copy from original subview to subview
/// of the promoted region for the read operands/subview of promoted region to
/// original subview for the results. The copy has to happen from `src` to
/// `dst`.
using CopyCallbackFn =
    std::function<LogicalResult(OpBuilder &b, Value src, Value dst)>;

struct LinalgPromotionOptions {
  /// Indices of subViews to promote. If `None`, try to promote all operands.
  Optional<DenseSet<unsigned>> operandsToPromote = None;
  LinalgPromotionOptions &setOperandsToPromote(ArrayRef<int64_t> operands) {
    operandsToPromote = DenseSet<unsigned>();
    operandsToPromote->insert(operands.begin(), operands.end());
    return *this;
  }
  /// If ith element of `useFullTiles` is true the full view should be used for
  /// the promoted buffer of the ith operand in `operandsToPromote`. Otherwise
  /// the partial view will be used.
  /// The decision is defaulted to `useFullTileBuffersDefault` when
  /// `useFullTileBuffers` is None and for operands missing from
  /// `useFullTileBuffers`.
  Optional<llvm::SmallBitVector> useFullTileBuffers = None;
  LinalgPromotionOptions &setUseFullTileBuffers(ArrayRef<bool> useFullTiles) {
    unsigned size = useFullTiles.size();
    llvm::SmallBitVector tmp(size, false);
    for (unsigned i = 0; i < size; ++i)
      tmp[i] = useFullTiles[i];
    useFullTileBuffers = tmp;
    return *this;
  }
  /// If true all operands unspecified by `useFullTileBuffers` will use the full
  /// view, otherwise the partial view.
  bool useFullTileBuffersDefault = false;
  LinalgPromotionOptions &setUseFullTileBuffersByDefault(bool use) {
    useFullTileBuffersDefault = use;
    return *this;
  }
  /// Allow the use of dynamically-sized buffers.
  bool dynamicBuffers = false;
  LinalgPromotionOptions &setDynamicBuffers(unsigned dynamic) {
    dynamicBuffers = dynamic;
    return *this;
  }
  /// Alignment of promoted buffer. If `None` do not specify alignment.
  Optional<unsigned> alignment = None;
  LinalgPromotionOptions &setAlignment(unsigned align) {
    alignment = align;
    return *this;
  }
  /// Use alloca with the default allocation scheme.
  bool useAlloca = false;
  LinalgPromotionOptions &setUseAlloca(bool use) {
    useAlloca = use;
    return *this;
  }
  /// Callback function to do the allocation of the promoted buffer. If None,
  /// then the default allocation scheme of allocating a memref<?xi8> buffer
  /// followed by a view operation is used.
  Optional<AllocBufferCallbackFn> allocationFn = None;
  Optional<DeallocBufferCallbackFn> deallocationFn = None;
  LinalgPromotionOptions &
  setAllocationDeallocationFns(AllocBufferCallbackFn const &allocFn,
                               DeallocBufferCallbackFn const &deallocFn) {
    allocationFn = allocFn;
    deallocationFn = deallocFn;
    return *this;
  }
  /// Callback function to do the copy of data to and from the promoted
  /// subview. If None then a linalg.copy is used.
  Optional<CopyCallbackFn> copyInFn = None;
  Optional<CopyCallbackFn> copyOutFn = None;
  LinalgPromotionOptions &setCopyInOutFns(CopyCallbackFn const &copyIn,
                                          CopyCallbackFn const &copyOut) {
    copyInFn = copyIn;
    copyOutFn = copyOut;
    return *this;
  }
};

/// Promotes the `subViews` into a new buffer allocated at the insertion point
/// `b`. Promotion occurs in 3 steps:
///   1. Create a new buffer for a full tile (i.e. not clipped at the boundary).
///   2. Take a full view on the buffer.
///   3. Take a partial slice of the full view in step 2. and copy into it.
/// Infers statically sized buffers from subViews unless `dynamicBuffers` is
/// true.
///
/// Returns the modified linalg op (the modification happens in place) as well
/// as all the copy ops created.
Optional<LinalgOp> promoteSubViews(OpBuilder &b, LinalgOp op,
                                   LinalgPromotionOptions options,
                                   OperationFolder *folder = nullptr);

/// Creates a number of ranges equal to the number of dimensions in the `map`.
/// The returned ranges correspond to the loop ranges, in the proper order, for
/// which new loops will be created.
/// The function supports only maps that are invertible and have results of type
/// DimExpr or (DimExpr + DimExpr - SymbolExpr floordiv ConstExpr).
/// It expects a non-inverted, concatenated map and last values in
/// allViewSizes will be applied to the symbols in the map if it contains any.
SmallVector<Range, 4> emitLoopRanges(OpBuilder &b, Location loc, AffineMap map,
                                     ValueRange viewSizes);

/// Emit a suitable vector form for a Linalg op with fully static shape.
void vectorizeLinalgOp(OpBuilder &builder, Operation *op);

/// Emits a loop nest of `LoopTy` with the proper body for `op`.
template <typename LoopTy>
Optional<LinalgLoops> linalgLowerOpToLoops(OpBuilder &builder, Operation *op);

/// Emits a loop nest of `scf.for` with the proper body for `op`.
LogicalResult linalgOpToLoops(OpBuilder &builder, Operation *op);

/// Emits a loop nest of `scf.parallel` with the proper body for `op`.
LogicalResult linalgOpToParallelLoops(OpBuilder &builder, Operation *op);

/// Emits a loop nest of `affine.for` with the proper body for `op`.
LogicalResult linalgOpToAffineLoops(OpBuilder &builder, Operation *op);

//===----------------------------------------------------------------------===//
// Preconditions that ensure the corresponding transformation succeeds and can
// be applied as a rewrite pattern.
//===----------------------------------------------------------------------===//
/// Emits a `generic` or `indexed_generic` operation with the `indexing_maps`
/// and `iterator_types` permutated according to `permutation`.
LogicalResult
interchangeGenericLinalgOpPrecondition(Operation *op,
                                       ArrayRef<unsigned> interchangeVector);

/// Promote std.subviews feeding linalg operations.
LogicalResult promoteSubviewsPrecondition(Operation *op,
                                          LinalgPromotionOptions options);

/// Rewrite a linalg.generic into a suitable vector.contraction op.
LogicalResult vectorizeLinalgOpPrecondition(Operation *op);

//===----------------------------------------------------------------------===//
// Transformations exposed as rewrite patterns.
//===----------------------------------------------------------------------===//
// Marker used as attribute name in generated Linalg rewriting transformations.
struct LinalgTransforms {
  static const StringLiteral kLinalgTransformMarker;
};

/// Helper class to control common attribute matching and setting behavior.
struct LinalgMarker {
  explicit LinalgMarker(ArrayRef<Identifier> matchDisjunction = {},
                        Optional<Identifier> replacement = None);
  LinalgMarker(LinalgMarker &&) = default;
  LinalgMarker(const LinalgMarker &) = default;
  LogicalResult checkAndNotify(PatternRewriter &rewriter, Operation *op) const;
  void replaceLinalgMarker(PatternRewriter &rewriter, Operation *op) const;

private:
  SmallVector<Identifier, 4> matchDisjunction;
  Optional<Identifier> replacement;
};

///
/// Linalg tiling patterns.
///
/// Apply the `tileLinalgOp` transformation as a pattern.
/// `marker` controls LinalgTransformMarker matching and update when specified.
/// See `tileLinalgOp` for more details.
enum class LinalgTilingLoopType {
  Loops = 0,
  AffineLoops = 1,
  ParallelLoops = 2,
};
using TileSizeComputationFunction =
    std::function<SmallVector<Value, 4>(OpBuilder &, Operation *)>;
struct LinalgTilingOptions {
  /// Computation function that returns the tile sizes for each operation.
  /// Delayed construction of constant tile sizes should occur to interoperate
  /// with folding.
  TileSizeComputationFunction tileSizeComputationFunction = nullptr;
  LinalgTilingOptions &
  setTileSizeComputationFunction(TileSizeComputationFunction &fun) {
    tileSizeComputationFunction = fun;
    return *this;
  }
  /// Set the `tileSizeComputationFunction` to return the values `ts`. The
  /// values must not fold away when tiling. Otherwise, use a more robust
  /// `tileSizeComputationFunction`.
  LinalgTilingOptions &setTileSizes(SmallVector<Value, 4> ts) {
    tileSizeComputationFunction = [=](OpBuilder &, Operation *) { return ts; };
    return *this;
  }
  /// Convenience function to set the `tileSizeComputationFunction` to a
  /// function that computes tile sizes at the point they are needed. Allows
  /// proper interaction with folding.
  LinalgTilingOptions &setTileSizes(ArrayRef<int64_t> ts);

  /// The interchange vector to reorder the tiled loops.
  SmallVector<unsigned, 4> interchangeVector{};
  LinalgTilingOptions &setInterchange(ArrayRef<unsigned> interchange) {
    interchangeVector.assign(interchange.begin(), interchange.end());
    return *this;
  }
  /// The type of tile loops to generate.
  LinalgTilingLoopType loopType{LinalgTilingLoopType::Loops};
  LinalgTilingOptions &setLoopType(LinalgTilingLoopType lt) {
    loopType = lt;
    return *this;
  }

  /// When specified, specifies distribution of generated tile loops to
  /// processors.
  Optional<LinalgLoopDistributionOptions> distribution = None;
  LinalgTilingOptions &
  setDistributionOptions(LinalgLoopDistributionOptions &distributionOptions) {
    distribution = distributionOptions;
    return *this;
  }
};

/// Canonicalization patterns relevant to apply after tiling patterns. These are
/// applied automatically by the tiling pass but need to be applied manually
/// when tiling is called programmatically.
OwningRewritePatternList
getLinalgTilingCanonicalizationPatterns(MLIRContext *ctx);

struct LinalgBaseTilingPattern : public RewritePattern {
  LinalgBaseTilingPattern(StringRef opName, MLIRContext *context,
                          LinalgTilingOptions options,
                          LinalgMarker marker = LinalgMarker(),
                          PatternBenefit benefit = 1);
  LogicalResult
  matchAndRewriteBase(Operation *op, PatternRewriter &rewriter,
                      SmallVectorImpl<Value> &tensorResults) const;

private:
  /// LinalgTransformMarker handles special attribute manipulations.
  LinalgMarker marker;
  /// Options to control tiling;
  LinalgTilingOptions options;
};

template <typename OpTy>
struct LinalgTilingPattern : public LinalgBaseTilingPattern {
  LinalgTilingPattern(MLIRContext *context, LinalgTilingOptions options,
                      LinalgMarker marker = LinalgMarker(),
                      PatternBenefit benefit = 1)
      : LinalgBaseTilingPattern(OpTy::getOperationName(), context, options,
                                marker, benefit) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value, 4> tensorResults;
    if (failed(LinalgBaseTilingPattern::matchAndRewriteBase(op, rewriter,
                                                            tensorResults)))
      return failure();
    if (tensorResults.empty())
      rewriter.eraseOp(op);
    else
      rewriter.replaceOp(op, tensorResults);
    return success();
  }
};

struct LinalgFusionOptions {
  /// Optional list of operands indices to use for fusion. When unspecified,
  /// only one fusion is done, i.e., the pattern returns after the first fusion.
  Optional<DenseSet<unsigned>> indicesToFuse = None;
  LinalgFusionOptions &setIndicesToFuse(ArrayRef<int64_t> operands) {
    indicesToFuse = DenseSet<unsigned>();
    indicesToFuse->insert(operands.begin(), operands.end());
    return *this;
  }
};

struct LinalgBaseTileAndFusePattern : public RewritePattern {
  LinalgBaseTileAndFusePattern(StringRef opName, MLIRContext *context,
                               const LinalgDependenceGraph &dependenceGraph,
                               LinalgTilingOptions tilingOptions,
                               LinalgFusionOptions fusionOptions,
                               LinalgMarker marker = LinalgMarker(),
                               LinalgMarker fusedOpMarker = LinalgMarker(),
                               LinalgMarker originalOpMarker = LinalgMarker(),
                               PatternBenefit benefit = 1);
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;

private:
  /// Dependence graph needed for fusion.
  const LinalgDependenceGraph &dependenceGraph;
  /// Options to control tiling.
  LinalgTilingOptions tilingOptions;
  /// Options to control fusion.
  LinalgFusionOptions fusionOptions;
  /// Marker to control application of the pattern.
  LinalgMarker marker;
  /// Marker set on the fused op after tile and fuse.
  LinalgMarker fusedOpMarker;
  /// The dependenceGraph is not modifiable, i.e. if the Linalg operations used
  /// to build the dependence graph changes then the dependenceGraph needs to be
  /// recomputed right now. To not invalidate the dependenceGraph as
  /// transformation happens, the original producer can be tagged with a marker
  /// that can be later used to delete the original operations.
  LinalgMarker originalOpMarker;
};

template <typename OpTy>
struct LinalgTileAndFusePattern : public LinalgBaseTileAndFusePattern {
  LinalgTileAndFusePattern(MLIRContext *context,
                           const LinalgDependenceGraph &dependenceGraph,
                           LinalgTilingOptions tilingOptions,
                           LinalgFusionOptions fusionOptions,
                           LinalgMarker marker = LinalgMarker(),
                           LinalgMarker fusedOpMarker = LinalgMarker(),
                           LinalgMarker originalOpMarker = LinalgMarker(),
                           PatternBenefit benefit = 1)
      : LinalgBaseTileAndFusePattern(
            OpTy::getOperationName(), context, dependenceGraph, tilingOptions,
            fusionOptions, marker, fusedOpMarker, originalOpMarker, benefit) {}
};

///
/// Linalg interchange patterns.
///
/// Apply the `interchange` transformation as a pattern.
/// `marker` controls LinalgTransformMarker matching and update when specified.
/// See `interchange` for more details.
struct LinalgBaseInterchangePattern : public RewritePattern {
  LinalgBaseInterchangePattern(StringRef opName, MLIRContext *context,
                               ArrayRef<unsigned> interchangeVector,
                               LinalgMarker marker = LinalgMarker(),
                               PatternBenefit benefit = 1);
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;

private:
  /// LinalgTransformMarker handles special attribute manipulations.
  LinalgMarker marker;
  /// The interchange vector to reorder the iterators and indexing_maps dims.
  SmallVector<unsigned, 8> interchangeVector;
};

template <typename OpTy>
struct LinalgInterchangePattern : public LinalgBaseInterchangePattern {
  LinalgInterchangePattern(MLIRContext *context,
                           ArrayRef<unsigned> interchangeVector,
                           LinalgMarker marker = LinalgMarker(),
                           PatternBenefit benefit = 1)
      : LinalgBaseInterchangePattern(OpTy::getOperationName(), context,
                                     interchangeVector, marker, benefit) {}
};

///
/// Linalg promotion patterns.
///
/// Apply the `promoteSubViews` transformation as a pattern.
/// `marker` controls LinalgTransformMarker matching and update when specified.
/// See `promoteSubViews` for more details.
struct LinalgBasePromotionPattern : public RewritePattern {
  LinalgBasePromotionPattern(StringRef opName, MLIRContext *context,
                             LinalgPromotionOptions options,
                             LinalgMarker marker = LinalgMarker(),
                             PatternBenefit benefit = 1);
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;

private:
  /// LinalgTransformMarker handles special attribute manipulations.
  LinalgMarker marker;
  /// Promotion options.
  LinalgPromotionOptions options;
};

template <typename OpTy>
struct LinalgPromotionPattern : public LinalgBasePromotionPattern {
  LinalgPromotionPattern(MLIRContext *context, LinalgPromotionOptions options,
                         LinalgMarker marker = LinalgMarker(),
                         PatternBenefit benefit = 1)
      : LinalgBasePromotionPattern(OpTy::getOperationName(), context, options,
                                   marker, benefit) {}
};

///
/// Linalg vectorization patterns.
///
/// Apply the `vectorizeLinalgOp` transformation as a pattern.
/// `marker` controls LinalgTransformMarker matching and update when specified.
/// See `vectorizeLinalgOp` for more details.
struct LinalgBaseVectorizationPattern : public RewritePattern {
  LinalgBaseVectorizationPattern(StringRef opName, MLIRContext *context,
                                 LinalgMarker marker = LinalgMarker(),
                                 PatternBenefit benefit = 1);
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;

private:
  /// LinalgTransformMarker handles special attribute manipulations.
  LinalgMarker marker;
};

template <typename OpTy>
struct LinalgVectorizationPattern : public LinalgBaseVectorizationPattern {
  LinalgVectorizationPattern(MLIRContext *context,
                             LinalgMarker marker = LinalgMarker(),
                             PatternBenefit benefit = 1)
      : LinalgBaseVectorizationPattern(OpTy::getOperationName(), context,
                                       marker, benefit) {}
};

///
/// Linalg lowering patterns.
///
/// Apply the `linalgLowerOpToLoops` transformation as a pattern.
/// `marker` controls LinalgTransformMarker matching and update when specified.
/// See `linalgLowerOpToLoops` for more details.
enum class LinalgLoweringType {
  LibraryCall = 0,
  Loops = 1,
  AffineLoops = 2,
  ParallelLoops = 3
};
template <typename OpTy>
struct LinalgLoweringPattern : public RewritePattern {
  LinalgLoweringPattern(MLIRContext *context, LinalgLoweringType loweringType,
                        LinalgMarker marker = LinalgMarker(),
                        PatternBenefit benefit = 1)
      : RewritePattern(OpTy::getOperationName(), {}, benefit, context),
        marker(marker), loweringType(loweringType) {}
  // TODO: Move implementation to .cpp once named ops are auto-generated.
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    LinalgOp linalgOp = dyn_cast<LinalgOp>(op);
    if (!linalgOp)
      return failure();
    if (failed(marker.checkAndNotify(rewriter, linalgOp)))
      return failure();

    if (loweringType == LinalgLoweringType::LibraryCall) {
      // TODO: Move lowering to library calls here.
      return failure();
    } else if (loweringType == LinalgLoweringType::Loops) {
      if (failed(linalgOpToLoops(rewriter, op)))
        return failure();
    } else if (loweringType == LinalgLoweringType::AffineLoops) {
      if (failed(linalgOpToAffineLoops(rewriter, op)))
        return failure();
    } else if (failed(linalgOpToParallelLoops(rewriter, op))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }

private:
  /// LinalgTransformMarker handles special attribute manipulations.
  LinalgMarker marker;
  /// Controls whether the pattern lowers to library calls, scf.for, affine.for
  /// or scf.parallel.
  LinalgLoweringType loweringType;
};

//===----------------------------------------------------------------------===//
// Op-specific patterns.
//===----------------------------------------------------------------------===//
/// Match and rewrite for the pattern:
/// ```
///    %alloc = ...
///    [optional] %view = std.view %alloc ...
///    %subView = subview %allocOrView ...
///    [optional] linalg.fill(%allocOrView, %cst) ...
///    ...
///    linalg.copy(%in, %subView) ...
///    vector.transfer_read %allocOrView[...], %cst ...
/// ```
/// into
/// ```
///    [unchanged] %alloc = ...
///    [unchanged] [optional] %view = std.view %alloc ...
///    [unchanged] [unchanged] %subView = subview %allocOrView ...
///    ...
///    vector.transfer_read %in[...], %cst ...
/// ```
/// Where there is no interleaved use between linalg.copy and transfer_read as
/// well as no interleaved use between linalg.fill and linalg.copy (if
/// linalg.fill is specified).
/// This is a custom rewrite to forward partial reads (with optional fills) to
/// vector.transfer_read.
struct LinalgCopyVTRForwardingPattern
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp xferOp,
                                PatternRewriter &rewriter) const override;
};

/// Match and rewrite for the pattern:
/// ```
///    %alloc = ...
///    [optional] %view = std.view %alloc ...
///    %subView = subview %allocOrView...
///    ...
///    vector.transfer_write %..., %allocOrView[...]
///    linalg.copy(%subView, %out)
/// ```
/// into
/// ```
///    [unchanged] %alloc = ...
///    [unchanged] [optional] %view = std.view %alloc ...
///    [unchanged] %subView = subview %allocOrView...
///    ...
///    vector.transfer_write %..., %out[...]
/// ```
/// Where there is no interleaved use between transfer_write and linalg.copy.
/// This is a custom rewrite to forward partial writes to vector.transfer_write.
struct LinalgCopyVTWForwardingPattern
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern<vector::TransferWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp xferOp,
                                PatternRewriter &rewriter) const override;
};

/// Canonicalize AffineMinOp operations in the context of enclosing scf.for and
/// scf.parallel by:
///   1. building an affine map where uses of the induction variable of a loop
///   are replaced by either the min (i.e. `%lb`) of the max
///   (i.e. `%lb + %step * floordiv(%ub -1 - %lb, %step)`) expression, depending
///   on whether the induction variable is used with a positive or negative
///   coefficient.
///   2. checking whether any of the results of this affine map is known to be
///   greater than all other results.
///   3. replacing the AffineMinOp by the result of (2).
// TODO: move to a more appropriate place when it is determined. For now Linalg
// depends both on Affine and SCF but they do not depend on each other.
struct AffineMinSCFCanonicalizationPattern
    : public OpRewritePattern<AffineMinOp> {
  using OpRewritePattern<AffineMinOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineMinOp minOp,
                                PatternRewriter &rewriter) const override;
};

/// Converts Convolution op into vector contraction.
///
/// Conversion expects ConvOp to have dimensions marked in the *mask* as
/// false of size 1. This ensures that the ConvOp can be lowered to vector
/// contraction of dimensions marked in the *mask* as true.
///
/// A good example for vectorization is ConvNHWCOp which is 2D Conv op
/// with channels as the last dimension. Let's vectorize last 3 dimensions.
/// The initial op definition looks like this:
/// ```
/// linalg.conv_2d_nhwc  %arg0, %arg1, %arg2 :
///   (memref<1x3x3x3xf32>, memref<1x3x3x3xf32>, memref<?x?x?x?xf32>)
/// ```
/// This op can be expressed as a dot product between %arg0 (input) and
/// %arg1 (kernel) which is written into first entry of %arg2 (output). This is
/// the ConvOp this pass expects and converts into:
/// ```
/// #map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
/// #map1 = affine_map<(d0, d1, d2) -> ()>
/// .....
/// %0 = vector.transfer_read %arg0[%c0, %c0, %c0, %c0], %c0_f32
///   : memref<1x3x3x3xf32>, vector<3x3x3xf32>
/// %1 = vector.transfer_read %arg1[%c0, %c0, %c0, %c0], %c0_f32
///   : memref<1x3x3x3xf32>, vector<3x3x3xf32>
/// %2 = vector.contract {indexing_maps = [#map0, #map0, #map1],
///   iterator_types = ["reduction", "reduction", "reduction"]} %0, %1,
///   %c0_f32 : vector<3x3x3xf32>, vector<3x3x3xf32> into f32
/// store %2, %arg2[%c0, %c0, %c0, %c0] : memref<?x?x?x?xf32>
/// ```
/// where first 2 operations read input and kernel memory buffers into vectors.
/// Subsequently, they are contracted together and the result is written to
/// the first entry of the output buffer.
template <typename ConvOp, int N>
class ConvOpVectorization : public OpRewritePattern<ConvOp> {
  using OpRewritePattern<ConvOp>::OpRewritePattern;
  SmallVector<bool, 4> mask;

public:
  ConvOpVectorization(MLIRContext *context, SmallVector<bool, 4> msk)
      : OpRewritePattern<ConvOp>(context) {
    assert(msk.size() == N && "Mask size does not match rank");
    this->mask = msk;
  }

  LogicalResult matchAndRewrite(ConvOp minOp,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// Patterns to convert a LinalgOp to std.call @external library implementation.
//===----------------------------------------------------------------------===//
// Create a new call to the type-canonicalized `LinalgOp::getLibraryCallName()`
// function. The implementation of the function can be either in the same module
// or in an externally linked library.
// This is a generic entry point for all LinalgOp, except for CopyOp and
// IndexedGenericOp, for which omre specialized patterns are provided.
class LinalgOpToLibraryCallRewrite : public RewritePattern {
public:
  LinalgOpToLibraryCallRewrite()
      : RewritePattern(/*benefit=*/1, MatchAnyOpTypeTag()) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;
};

/// Rewrite pattern specialization for CopyOp, kicks in when both input and
/// output permutations are left unspecified or are the identity.
class CopyOpToLibraryCallRewrite : public OpRewritePattern<CopyOp> {
public:
  using OpRewritePattern<CopyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(CopyOp op,
                                PatternRewriter &rewriter) const override;
};

/// Rewrite CopyOp with permutations into a sequence of TransposeOp and
/// permutation-free CopyOp. This interplays with TransposeOpConversion and
/// LinalgConversion<CopyOp> to create a path to the LLVM dialect.
class CopyTransposeRewrite : public OpRewritePattern<CopyOp> {
public:
  using OpRewritePattern<CopyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(CopyOp op,
                                PatternRewriter &rewriter) const override;
};

/// Conversion pattern specialization for IndexedGenericOp, has special handling
/// for the extra index operands.
class IndexedGenericOpToLibraryCallRewrite
    : public OpRewritePattern<IndexedGenericOp> {
public:
  using OpRewritePattern<IndexedGenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(IndexedGenericOp op,
                                PatternRewriter &rewriter) const override;
};

/// Populate the given list with patterns that convert from Linalg to Standard.
void populateLinalgToStandardConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx);

//===----------------------------------------------------------------------===//
// Buffer allocation patterns.
//===----------------------------------------------------------------------===//

/// Generic BufferAssignmentConversionPattern that matches any Operation* and
/// dispatches internally. This avoids template instantiating one pattern for
/// each LinalgOp op.
class LinalgOpConverter : public BufferAssignmentConversionPattern {
public:
  LinalgOpConverter(MLIRContext *context,
                    BufferAssignmentTypeConverter &converter)
      : BufferAssignmentConversionPattern(context, converter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final;
};

class TensorConstantOpConverter
    : public BufferAssignmentOpConversionPattern<ConstantOp> {
public:
  using BufferAssignmentOpConversionPattern<
      ConstantOp>::BufferAssignmentOpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final;
};

class TensorCastOpConverter
    : public BufferAssignmentOpConversionPattern<TensorCastOp> {
public:
  using BufferAssignmentOpConversionPattern<
      TensorCastOp>::BufferAssignmentOpConversionPattern;

  LogicalResult
  matchAndRewrite(TensorCastOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final;
};

//===----------------------------------------------------------------------===//
// Support for staged pattern application.
//===----------------------------------------------------------------------===//
/// Helper function to allow applying rewrite patterns, interleaved with more
/// global transformations, in a staged fashion:
///   1. the first stage consists of a list of OwningRewritePatternList. Each
///   OwningRewritePatternList in this list is applied once, in order.
///   2. the second stage consists of a single OwningRewritePattern that is
///   applied greedily until convergence.
///   3. the third stage consists of applying a lambda, generally used for
///   non-local transformation effects. This allows creating custom fused
///   transformations where patterns can be ordered and applied at a finer
///   granularity than a sequence of traditional compiler passes.
LogicalResult applyStagedPatterns(
    Operation *op, ArrayRef<OwningRewritePatternList> stage1Patterns,
    const OwningRewritePatternList &stage2Patterns,
    function_ref<LogicalResult(Operation *)> stage3Lambda = nullptr);

} // namespace linalg
} // namespace mlir

#endif // DIALECT_LINALG_TRANSFORMS_TRANSFORMS_H_
