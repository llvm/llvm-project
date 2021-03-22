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
#include "llvm/ADT/SmallSet.h"

namespace mlir {
class BufferizeTypeConverter;
class FrozenRewritePatternList;

namespace linalg {

struct LinalgFusionOptions;
struct LinalgTilingOptions;

//===----------------------------------------------------------------------===//
// Transformations exposed as function calls.
//===----------------------------------------------------------------------===//
using LinalgLoops = SmallVector<Operation *, 4>;

/// Populates patterns for vectorization of all ConvN-D ops.
void populateConvVectorizationPatterns(
    MLIRContext *context, SmallVectorImpl<OwningRewritePatternList> &patterns,
    ArrayRef<int64_t> tileSizes);

/// Populates the given list with patterns to bufferize linalg ops.
void populateLinalgBufferizePatterns(BufferizeTypeConverter &converter,
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
struct TiledLinalgOp {
  LinalgOp op;
  SmallVector<Operation *, 8> loops;
  SmallVector<Value, 4> tensorResults;
  TiledLinalgOp &operator=(const TiledLinalgOp &) = default;
};
Optional<TiledLinalgOp> tileLinalgOp(OpBuilder &b, LinalgOp op,
                                     const LinalgTilingOptions &options);

/// Fuse a sequence of linalg operations (`ops`) using tile-and-fuse. This
/// proceeds as follows:
/// - Find outer parallel loops in these ops that can be fused.
/// - Tile fusable outer parallel loops of the last operation in the sequence.
/// - Fuse the remaining operations with the tiled operation
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
///   %4 = subview %arg3[0, 0] [32, 32] [1, 1]
///     : memref<32x32xf32> to memref<32x32xf32, #map1>
///   linalg.matmul
///     ins(%2, %3 : memref<16x32xf32, #map0>, memref<32x32xf32, #map1>)
///     outs(%0 : memref<16x32xf32, #map0>)
///   linalg.matmul
///     ins(%0, %4 : memref<16x4xf32, #map0>, memref<4x8xf32, #map0>)
///     outs(%1 : memref<16x8xf32, #map0>)
/// }
///
/// `tilingOptions` are used to tile the corresponding operation in `ops` (the
/// size of the former should be same as size of the latter. Based on how
/// tile+fuse is implemented, the fused loops are generated based on the last
/// operation in the sequence. For example, the tile sizes for the fused loops
/// is obtained from `tilingOptions.back()`. The following tiling options are
/// handled differently in tile+fuse (compared to tile only)
/// - Interchange of the tiling loops is not supported right now.
/// - Only the fused loops are distributed.
struct TiledAndFusedLinalgOps {
  /// Operation obtained by tiling the last operation in sequence of `ops`
  /// passed to `tileAndFuseLinalgOps`.
  LinalgOp op;
  /// The dimension of the loops that are fused.
  std::set<unsigned> fusedLoopDims;
  /// The generated fused operations (created within the fused loops).
  SmallVector<LinalgOp, 1> fusedProducers;
  /// The fused loop generated.
  SmallVector<Operation *, 4> fusedLoops;
};
Optional<TiledAndFusedLinalgOps>
tileAndFuseLinalgOps(OpBuilder &builder, ArrayRef<LinalgOp> ops,
                     const LinalgDependenceGraph &dependenceGraph,
                     const LinalgTilingOptions &tilingOptions);

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
    OpBuilder &b, memref::SubViewOp subView,
    ArrayRef<Value> boundingSubViewSize, OperationFolder *folder)>;

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

/// Creates a new buffer using the `allocationFn` provided. The size of this
/// buffer is the smallest constant bounding size along each dimension that can
/// be computed for the size of the result of `subView`. Returns the allocated
/// buffer as `fullLocalView` and the view that matches the size of the result
/// of subview operation as `partialLocalView`.
struct PromotionInfo {
  Value fullLocalView;
  Value partialLocalView;
};
Optional<PromotionInfo>
promoteSubviewAsNewBuffer(OpBuilder &b, Location loc, memref::SubViewOp subView,
                          AllocBufferCallbackFn allocationFn,
                          OperationFolder *folder = nullptr);

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

/// Emit a suitable vector form for a Linalg op with fully static shape.
LogicalResult vectorizeLinalgOp(OpBuilder &builder, Operation *op,
                                SmallVectorImpl<Value> &newResults);

/// Emits a loop nest of `LoopTy` with the proper body for `op`.
template <typename LoopTy>
Optional<LinalgLoops>
linalgLowerOpToLoops(OpBuilder &builder, Operation *op,
                     ArrayRef<unsigned> interchangeVector = {});

/// Emits a loop nest of `scf.for` with the proper body for `op`. The generated
/// loop nest will follow the `interchangeVector`-permutated iterator order. If
/// `interchangeVector` is empty, then no permutation happens.
LogicalResult linalgOpToLoops(OpBuilder &builder, Operation *op,
                              ArrayRef<unsigned> interchangeVector = {});

/// Emits a loop nest of `scf.parallel` with the proper body for `op`. The
/// generated loop nest will follow the `interchangeVector`-permutated
// iterator order. If `interchangeVector` is empty, then no permutation happens.
LogicalResult
linalgOpToParallelLoops(OpBuilder &builder, Operation *op,
                        ArrayRef<unsigned> interchangeVector = {});

/// Emits a loop nest of `affine.for` with the proper body for `op`. The
/// generated loop nest will follow the `interchangeVector`-permutated
// iterator order. If `interchangeVector` is empty, then no permutation happens.
LogicalResult linalgOpToAffineLoops(OpBuilder &builder, Operation *op,
                                    ArrayRef<unsigned> interchangeVector = {});

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

/// Helper class to control application of linalg transformation patterns.
/// Control comes in 2 forms:
///   1. attribute matching and setting behavior using the attribute named
///      `kLinalgTransformMarker`. This can be used to build a state machine
///      using attributes and incrementally applying patterns to advance states.
///   2. filter function, which is a simple lambda on the Operation* that
///      returns a LogicalResult.
struct LinalgTransformationFilter {
  using FilterFunction = std::function<LogicalResult(Operation *)>;

  explicit LinalgTransformationFilter(
      ArrayRef<Identifier> matchDisjunction = {},
      Optional<Identifier> replacement = None);

  explicit LinalgTransformationFilter(
      FilterFunction f, ArrayRef<Identifier> matchDisjunction = {},
      Optional<Identifier> replacement = None);

  LinalgTransformationFilter(LinalgTransformationFilter &&) = default;
  LinalgTransformationFilter(const LinalgTransformationFilter &) = default;
  LogicalResult checkAndNotify(PatternRewriter &rewriter, Operation *op) const;
  void replaceLinalgTransformationFilter(PatternRewriter &rewriter,
                                         Operation *op) const;

  LinalgTransformationFilter &addFilter(FilterFunction f) {
    if (f)
      filters.push_back(f);
    return *this;
  }
  template <typename... OpTypes>
  LinalgTransformationFilter &addOpFilter() {
    return addFilter(
        [](Operation *op) { return success(isa<OpTypes...>(op)); });
  }

private:
  SmallVector<FilterFunction> filters;
  SmallVector<Identifier> matchDisjunction;
  Optional<Identifier> replacement;
};

///
/// Linalg tiling patterns.
///
/// Apply the `tileLinalgOp` transformation as a pattern.
/// `filter` controls LinalgTransformMarker matching and update when specified.
/// See `tileLinalgOp` for more details.
enum class LinalgTilingLoopType {
  Loops = 0,
  AffineLoops = 1,
  ParallelLoops = 2,
};

using TileSizeComputationFunction =
    std::function<SmallVector<Value, 4>(OpBuilder &, Operation *)>;

/// Specify the padding value for an OpOperand. This should be a function of
/// both the operation and the operand type.
using PaddingValueComputationFunction =
    std::function<Value(OpBuilder &, OpOperand &)>;

struct LinalgTilingOptions {
  /// Computation function that returns the tile sizes for each operation.
  /// Delayed construction of constant tile sizes should occur to interoperate
  /// with folding.
  TileSizeComputationFunction tileSizeComputationFunction = nullptr;

  LinalgTilingOptions &
  setTileSizeComputationFunction(TileSizeComputationFunction fun) {
    tileSizeComputationFunction = std::move(fun);
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
  SmallVector<unsigned, 4> interchangeVector = {};

  LinalgTilingOptions &setInterchange(ArrayRef<unsigned> interchange) {
    interchangeVector.assign(interchange.begin(), interchange.end());
    return *this;
  }

  /// The type of tile loops to generate.
  LinalgTilingLoopType loopType = LinalgTilingLoopType::Loops;

  LinalgTilingOptions &setLoopType(LinalgTilingLoopType lt) {
    loopType = lt;
    return *this;
  }

  /// When specified, specifies distribution of generated tile loops to
  /// processors.
  Optional<LinalgLoopDistributionOptions> distribution = None;

  LinalgTilingOptions &
  setDistributionOptions(LinalgLoopDistributionOptions distributionOptions) {
    distribution = std::move(distributionOptions);
    return *this;
  }

  /// Computation function that returns a padding value to use when padding to
  /// force static sizes. When `paddingValueComputationFunction` is set, padding
  /// operations are introduced, that guarantee the underlying op is statically
  /// shaped and can thus be vectorized.
  PaddingValueComputationFunction paddingValueComputationFunction = nullptr;

  LinalgTilingOptions &
  setPaddingValueComputationFunction(PaddingValueComputationFunction fun) {
    paddingValueComputationFunction = std::move(fun);
    return *this;
  }
};

/// Canonicalization patterns relevant to apply after tiling patterns. These are
/// applied automatically by the tiling pass but need to be applied manually
/// when tiling is called programmatically.
OwningRewritePatternList
getLinalgTilingCanonicalizationPatterns(MLIRContext *ctx);
void populateLinalgTilingCanonicalizationPatterns(
    OwningRewritePatternList &patterns);

/// Base pattern that applied the tiling transformation specified by `options`.
/// Abort and return failure in 2 cases:
///   1. if the tiling specification is invalid and tiling fails to occur.
///   2. if tiling occurs but `options.paddingValueComputationFunction` is set
///      and some operand shape cannot be bounded statically.
struct LinalgBaseTilingPattern : public RewritePattern {
  // Entry point to match any LinalgOp OpInterface.
  LinalgBaseTilingPattern(
      LinalgTilingOptions options,
      LinalgTransformationFilter filter = LinalgTransformationFilter(),
      PatternBenefit benefit = 1);
  // Entry point to match a specific Linalg op.
  LinalgBaseTilingPattern(
      StringRef opName, MLIRContext *context, LinalgTilingOptions options,
      LinalgTransformationFilter filter = LinalgTransformationFilter(),
      PatternBenefit benefit = 1);
  LogicalResult matchAndRewriteBase(Operation *op, PatternRewriter &rewriter,
                                    TiledLinalgOp &result) const;

private:
  /// LinalgTransformMarker handles special attribute manipulations.
  LinalgTransformationFilter filter;
  /// Options to control tiling;
  LinalgTilingOptions options;
};

template <typename OpTy>
struct LinalgTilingPattern : public LinalgBaseTilingPattern {
  /// SFINAE: This constructor can only trigger for concrete ops that have a
  /// static `getOperationName` method.
  template <typename ConcreateOpTy = OpTy>
  LinalgTilingPattern(
      MLIRContext *context, LinalgTilingOptions options,
      LinalgTransformationFilter filter = LinalgTransformationFilter(),
      PatternBenefit benefit = 1)
      : LinalgBaseTilingPattern(ConcreateOpTy::getOperationName(), context,
                                options, filter, benefit) {}

  /// This constructor is available to anyone.
  LinalgTilingPattern(
      StringRef opName, MLIRContext *context, LinalgTilingOptions options,
      LinalgTransformationFilter filter = LinalgTransformationFilter(),
      PatternBenefit benefit = 1)
      : LinalgBaseTilingPattern(opName, context, options, filter, benefit) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    TiledLinalgOp tiledLinalgOp;
    if (failed(LinalgBaseTilingPattern::matchAndRewriteBase(op, rewriter,
                                                            tiledLinalgOp)))
      return failure();
    if (tiledLinalgOp.tensorResults.empty())
      rewriter.eraseOp(op);
    else
      rewriter.replaceOp(op, tiledLinalgOp.tensorResults);
    return success();
  }
};

struct LinalgFusionOptions {
  /// List of operands indices to use for fusion.
  llvm::SmallSet<unsigned, 1> indicesToFuse = {};
  LinalgFusionOptions &setIndicesToFuse(ArrayRef<int64_t> operands) {
    indicesToFuse.insert(operands.begin(), operands.end());
    return *this;
  }
};

struct LinalgBaseTileAndFusePattern : public RewritePattern {
  LinalgBaseTileAndFusePattern(
      StringRef opName, MLIRContext *context,
      const LinalgDependenceGraph &dependenceGraph,
      LinalgTilingOptions tilingOptions, LinalgFusionOptions fusionOptions,
      LinalgTransformationFilter filter = LinalgTransformationFilter(),
      LinalgTransformationFilter fusedOpMarker = LinalgTransformationFilter(),
      LinalgTransformationFilter originalOpMarker =
          LinalgTransformationFilter(),
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
  LinalgTransformationFilter filter;
  /// Marker set on the fused op after tile and fuse.
  LinalgTransformationFilter fusedOpMarker;
  /// The dependenceGraph is not modifiable, i.e. if the Linalg operations used
  /// to build the dependence graph changes then the dependenceGraph needs to be
  /// recomputed right now. To not invalidate the dependenceGraph as
  /// transformation happens, the original producer can be tagged with a filter
  /// that can be later used to delete the original operations.
  LinalgTransformationFilter originalOpMarker;
};

template <typename OpTy>
struct LinalgTileAndFusePattern : public LinalgBaseTileAndFusePattern {
  LinalgTileAndFusePattern(
      MLIRContext *context, const LinalgDependenceGraph &dependenceGraph,
      LinalgTilingOptions tilingOptions, LinalgFusionOptions fusionOptions,
      LinalgTransformationFilter filter = LinalgTransformationFilter(),
      LinalgTransformationFilter fusedOpMarker = LinalgTransformationFilter(),
      LinalgTransformationFilter originalOpMarker =
          LinalgTransformationFilter(),
      PatternBenefit benefit = 1)
      : LinalgBaseTileAndFusePattern(
            OpTy::getOperationName(), context, dependenceGraph, tilingOptions,
            fusionOptions, filter, fusedOpMarker, originalOpMarker, benefit) {}
};

///
/// Linalg interchange patterns.
///
/// Apply the `interchange` transformation as a pattern.
/// `filter` controls LinalgTransformMarker matching and update when specified.
/// See `interchange` for more details.
struct LinalgBaseInterchangePattern : public RewritePattern {
  LinalgBaseInterchangePattern(
      StringRef opName, MLIRContext *context,
      ArrayRef<unsigned> interchangeVector,
      LinalgTransformationFilter filter = LinalgTransformationFilter(),
      PatternBenefit benefit = 1);
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;

private:
  /// LinalgTransformMarker handles special attribute manipulations.
  LinalgTransformationFilter filter;
  /// The interchange vector to reorder the iterators and indexing_maps dims.
  SmallVector<unsigned, 8> interchangeVector;
};

template <typename OpTy>
struct LinalgInterchangePattern : public LinalgBaseInterchangePattern {
  LinalgInterchangePattern(
      MLIRContext *context, ArrayRef<unsigned> interchangeVector,
      LinalgTransformationFilter filter = LinalgTransformationFilter(),
      PatternBenefit benefit = 1)
      : LinalgBaseInterchangePattern(OpTy::getOperationName(), context,
                                     interchangeVector, filter, benefit) {}
};

///
/// Linalg promotion patterns.
///
/// Apply the `promoteSubViews` transformation as a pattern.
/// `filter` controls LinalgTransformMarker matching and update when specified.
/// See `promoteSubViews` for more details.
struct LinalgBasePromotionPattern : public RewritePattern {
  LinalgBasePromotionPattern(
      StringRef opName, MLIRContext *context, LinalgPromotionOptions options,
      LinalgTransformationFilter filter = LinalgTransformationFilter(),
      PatternBenefit benefit = 1);

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;

private:
  /// LinalgTransformMarker handles special attribute manipulations.
  LinalgTransformationFilter filter;
  /// Promotion options.
  LinalgPromotionOptions options;
};

template <typename OpTy>
struct LinalgPromotionPattern : public LinalgBasePromotionPattern {
  /// SFINAE: This constructor can only trigger for concrete ops that have a
  /// static `getOperationName` method.
  template <typename ConcreateOpTy = OpTy>
  LinalgPromotionPattern(
      MLIRContext *context, LinalgPromotionOptions options,
      LinalgTransformationFilter filter = LinalgTransformationFilter(),
      PatternBenefit benefit = 1)
      : LinalgBasePromotionPattern(OpTy::getOperationName(), context, options,
                                   filter, benefit) {}
  /// This constructor is available to anyone.
  LinalgPromotionPattern(
      StringRef opName, MLIRContext *context, LinalgPromotionOptions options,
      LinalgTransformationFilter filter = LinalgTransformationFilter(),
      PatternBenefit benefit = 1)
      : LinalgBasePromotionPattern(opName, context, options, filter, benefit) {}
};

///
/// Linalg vectorization patterns.
///
/// Apply the `vectorizeLinalgOp` transformation as a pattern.
/// `filter` controls LinalgTransformMarker matching and update when specified.
/// See `vectorizeLinalgOp` for more details.

/// Empty for now, used for SFINAE purposes only.
struct LinalgVectorizationOptions {};

struct LinalgBaseVectorizationPattern : public RewritePattern {
  /// MatchAnyOpTag-based constructor with a mandatory `filter`.
  LinalgBaseVectorizationPattern(LinalgTransformationFilter filter,
                                 PatternBenefit benefit = 1);
  /// Name-based constructor with an optional `filter`.
  LinalgBaseVectorizationPattern(
      StringRef opName, MLIRContext *context,
      LinalgTransformationFilter filter = LinalgTransformationFilter(),
      PatternBenefit benefit = 1);
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;

private:
  /// LinalgTransformMarker handles special attribute manipulations.
  LinalgTransformationFilter filter;
};

struct LinalgVectorizationPattern : public LinalgBaseVectorizationPattern {
  /// These constructors are available to anyone.
  /// MatchAnyOpTag-based constructor with a mandatory `filter`.
  LinalgVectorizationPattern(
      LinalgTransformationFilter filter,
      LinalgVectorizationOptions options = LinalgVectorizationOptions(),
      PatternBenefit benefit = 1)
      : LinalgBaseVectorizationPattern(filter, benefit) {}
  /// Name-based constructor with an optional `filter`.
  LinalgVectorizationPattern(
      StringRef opName, MLIRContext *context,
      LinalgVectorizationOptions options = LinalgVectorizationOptions(),
      LinalgTransformationFilter filter = LinalgTransformationFilter(),
      PatternBenefit benefit = 1)
      : LinalgBaseVectorizationPattern(opName, context, filter, benefit) {}
};

/// Trait to check if T provides a `getOperationName` method.
template <typename T, typename... Args>
using has_get_operation_name = decltype(T::getOperationName());
template <typename T>
using detect_has_get_operation_name =
    llvm::is_detected<has_get_operation_name, T>;

/// SFINAE helper for single C++ op with a `getOperationName` method.
template <
    typename OpType,
    typename = std::enable_if_t<detect_has_get_operation_name<OpType>::value>,
    typename = void>
void insertVectorizationPatternImpl(OwningRewritePatternList &patternList,
                                    linalg::LinalgVectorizationOptions options,
                                    linalg::LinalgTransformationFilter f) {
  patternList.insert<linalg::LinalgVectorizationPattern>(
      OpType::getOperationName(), patternList.getContext(), options, f);
}

/// SFINAE helper for single C++ class without a `getOperationName` method (e.g.
/// an OpInterface).
template <typename OpType, typename = std::enable_if_t<
                               !detect_has_get_operation_name<OpType>::value>>
void insertVectorizationPatternImpl(OwningRewritePatternList &patternList,
                                    linalg::LinalgVectorizationOptions options,
                                    linalg::LinalgTransformationFilter f) {
  patternList.insert<linalg::LinalgVectorizationPattern>(
      f.addOpFilter<OpType>(), options);
}

/// Variadic helper function to insert vectorization patterns for C++ ops.
template <typename... OpTypes>
void insertVectorizationPatterns(OwningRewritePatternList &patternList,
                                 linalg::LinalgVectorizationOptions options,
                                 linalg::LinalgTransformationFilter f =
                                     linalg::LinalgTransformationFilter()) {
  // FIXME: In c++17 this can be simplified by using 'fold expressions'.
  (void)std::initializer_list<int>{
      0,
      (insertVectorizationPatternImpl<OpTypes>(patternList, options, f), 0)...};
}

///
/// Linalg lowering patterns.
///
/// Apply the `linalgLowerOpToLoops` transformation as a pattern.
/// `filter` controls LinalgTransformMarker matching and update when specified.
/// See `linalgLowerOpToLoops` for more details.
enum class LinalgLoweringType {
  LibraryCall = 0,
  Loops = 1,
  AffineLoops = 2,
  ParallelLoops = 3
};

template <typename OpTy>
struct LinalgLoweringPattern : public RewritePattern {
  LinalgLoweringPattern(
      MLIRContext *context, LinalgLoweringType loweringType,
      LinalgTransformationFilter filter = LinalgTransformationFilter(),
      ArrayRef<unsigned> interchangeVector = {}, PatternBenefit benefit = 1)
      : RewritePattern(OpTy::getOperationName(), {}, benefit, context),
        filter(filter), loweringType(loweringType),
        interchangeVector(interchangeVector.begin(), interchangeVector.end()) {}

  // TODO: Move implementation to .cpp once named ops are auto-generated.
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    LinalgOp linalgOp = dyn_cast<LinalgOp>(op);
    if (!linalgOp)
      return failure();
    if (failed(filter.checkAndNotify(rewriter, linalgOp)))
      return failure();

    switch (loweringType) {
    case LinalgLoweringType::LibraryCall:
      // TODO: Move lowering to library calls here.
      return failure();
    case LinalgLoweringType::Loops:
      if (failed(linalgOpToLoops(rewriter, op, interchangeVector)))
        return failure();
      break;
    case LinalgLoweringType::AffineLoops:
      if (failed(linalgOpToAffineLoops(rewriter, op, interchangeVector)))
        return failure();
      break;
    case LinalgLoweringType::ParallelLoops:
      if (failed(linalgOpToParallelLoops(rewriter, op, interchangeVector)))
        return failure();
      break;
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  /// LinalgTransformMarker handles special attribute manipulations.
  LinalgTransformationFilter filter;
  /// Controls whether the pattern lowers to library calls, scf.for, affine.for
  /// or scf.parallel.
  LinalgLoweringType loweringType;
  /// Permutated loop order in the generated loop nest.
  SmallVector<unsigned, 4> interchangeVector;
};

/// Linalg generalization patterns

/// Populates `patterns` with patterns to convert spec-generated named ops to
/// linalg.generic ops.
void populateLinalgNamedOpsGeneralizationPatterns(
    OwningRewritePatternList &patterns,
    LinalgTransformationFilter filter = LinalgTransformationFilter());

/// Populates `patterns` with patterns to convert linalg.conv ops to
/// linalg.generic ops.
void populateLinalgConvGeneralizationPatterns(
    OwningRewritePatternList &patterns,
    LinalgTransformationFilter filter = LinalgTransformationFilter());

//===----------------------------------------------------------------------===//
// Op-specific patterns.
//===----------------------------------------------------------------------===//

/// PadTensorOp does not implement the LinalgStructuredOpInterface `LinalgOp`,
/// it needs a specific pattern to vectorize.
struct PadTensorOpVectorizationPattern : public OpRewritePattern<PadTensorOp> {
  using OpRewritePattern<PadTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PadTensorOp padOp,
                                PatternRewriter &rewriter) const override;
};

/// Match and rewrite for the pattern:
/// ```
///    %alloc = ...
///    [optional] %view = memref.view %alloc ...
///    %subView = subview %allocOrView ...
///    [optional] linalg.fill(%allocOrView, %cst) ...
///    ...
///    linalg.copy(%in, %subView) ...
///    vector.transfer_read %allocOrView[...], %cst ...
/// ```
/// into
/// ```
///    [unchanged] %alloc = ...
///    [unchanged] [optional] %view = memref.view %alloc ...
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
///    [optional] %view = memref.view %alloc ...
///    %subView = subview %allocOrView...
///    ...
///    vector.transfer_write %..., %allocOrView[...]
///    linalg.copy(%subView, %out)
/// ```
/// into
/// ```
///    [unchanged] %alloc = ...
///    [unchanged] [optional] %view = memref.view %alloc ...
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

/// Helper struct to return the results of `substituteMin`.
struct AffineMapAndOperands {
  AffineMap map;
  SmallVector<Value> dims;
  SmallVector<Value> symbols;
};
/// Traverse the dims of the AffineMap of `affineMinOp` and substitute scf loop
/// induction variables by new expressions involving the lower or upper bound:
///   - If the AffineDimExpr mapped to a loop IV has a positive sign, it is
///     replaced by the loop upper bound.
///   - If the AffineDimExpr mapped to a loop IV has a negative sign, it is
///     replaced by the loop lower bound.
/// All loop induction variables are iteratively replaced, unless a
/// `substituteOperation` hook is passed to more finely determine which
/// operations are substituted.
/// This is used as an intermediate step in computing bounding boxes and
/// canonicalize AffineMinOps. All dim and symbol operands are assumed to have
/// positive values (positive orthant assumptions).
/// Return a new AffineMap, dims and symbols that have been canonicalized and
/// simplified.
AffineMapAndOperands substituteMin(
    AffineMinOp affineMinOp,
    llvm::function_ref<bool(Operation *)> substituteOperation = nullptr);

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
// Support for staged pattern application.
//===----------------------------------------------------------------------===//
/// Helper function to allow applying rewrite patterns, interleaved with more
/// global transformations, in a staged fashion:
///   1. the first stage consists of a list of FrozenRewritePatternList. Each
///   FrozenRewritePatternList in this list is applied once, in order.
///   2. the second stage consists of a single OwningRewritePattern that is
///   applied greedily until convergence.
///   3. the third stage consists of applying a lambda, generally used for
///   non-local transformation effects. This allows creating custom fused
///   transformations where patterns can be ordered and applied at a finer
///   granularity than a sequence of traditional compiler passes.
LogicalResult applyStagedPatterns(
    Operation *op, ArrayRef<FrozenRewritePatternList> stage1Patterns,
    const FrozenRewritePatternList &stage2Patterns,
    function_ref<LogicalResult(Operation *)> stage3Lambda = nullptr);

//===----------------------------------------------------------------------===//
// Support for sparse tensor code generation.
//
// The sparse compiler part of MLIR lowers a tensor expression formulated as a
// Linalg operation into a sequence of loops depending on what dimensions of the
// tensors are marked dense or sparse. The generated code distinguishes between:
// (1) for-loops that iterate over a single dense dimension,
// (2) for-loops that iterate over a single sparse dimension,
// (3) while-loops that co-iterate over several sparse dimensions.
// The for-loops may be subsequently optimized for parallel or vector execution.
//
// For more details, the Dialect/Linalg/Transforms/Sparsification.cpp file.
//===----------------------------------------------------------------------===//

/// Defines a parallelization strategy. Any implicit loop in the Linalg
/// operation that is marked "parallel" (thus not "reduction") is a candidate
/// for parallelization. The loop is made parallel if (1) allowed by the
/// strategy (e.g., AnyStorageOuterLoop considers either a dense or sparse
/// outermost loop only), and (2) the generated code is an actual for-loop
/// (and not a co-iterating while-loop).
enum class SparseParallelizationStrategy {
  kNone,
  kDenseOuterLoop,
  kAnyStorageOuterLoop,
  kDenseAnyLoop,
  kAnyStorageAnyLoop
  // TODO: support reduction parallelization too?
};

/// Defines a vectorization strategy. Any implicit inner loop in the Linalg
/// operation is a candidate (full SIMD for "parallel" loops and horizontal
/// SIMD for "reduction" loops). A loop is actually vectorized if (1) allowed
/// by the strategy, and (2) the emitted code is an actual for-loop (and not
/// a co-iterating while-loop).
enum class SparseVectorizationStrategy {
  kNone,
  kDenseInnerLoop,
  kAnyStorageInnerLoop
};

/// Defines a type for "pointer" and "index" storage in the sparse storage
/// scheme, with a choice between the native platform-dependent index width
/// or any of 64-/32-/16-/8-bit integers. A narrow width obviously reduces
/// the memory footprint of the sparse storage scheme, but the width should
/// suffice to define the total required range (viz. the maximum number of
/// stored entries per indirection level for the "pointers" and the maximum
/// value of each tensor index over all dimensions for the "indices").
enum class SparseIntType { kNative, kI64, kI32, kI16, kI8 };

/// Sparsification options.
struct SparsificationOptions {
  SparsificationOptions(SparseParallelizationStrategy p,
                        SparseVectorizationStrategy v, unsigned vl,
                        SparseIntType pt, SparseIntType it, bool fo)
      : parallelizationStrategy(p), vectorizationStrategy(v), vectorLength(vl),
        ptrType(pt), indType(it), fastOutput(fo) {
    // TODO: remove restriction when vectors with index elements are supported
    assert((v != SparseVectorizationStrategy::kAnyStorageInnerLoop ||
            (ptrType != SparseIntType::kNative &&
             indType != SparseIntType::kNative)) &&
           "This combination requires support for vectors with index elements");
  }
  SparsificationOptions()
      : SparsificationOptions(SparseParallelizationStrategy::kNone,
                              SparseVectorizationStrategy::kNone, 1u,
                              SparseIntType::kNative, SparseIntType::kNative,
                              false) {}
  SparseParallelizationStrategy parallelizationStrategy;
  SparseVectorizationStrategy vectorizationStrategy;
  unsigned vectorLength;
  SparseIntType ptrType;
  SparseIntType indType;
  bool fastOutput; // experimental: fast output buffers
};

/// Sets up sparsification rewriting rules with the given options.
void populateSparsificationPatterns(
    OwningRewritePatternList &patterns,
    const SparsificationOptions &options = SparsificationOptions());

/// Sets up sparsification conversion rules with the given options.
void populateSparsificationConversionPatterns(
    OwningRewritePatternList &patterns);

} // namespace linalg
} // namespace mlir

#endif // DIALECT_LINALG_TRANSFORMS_TRANSFORMS_H_
