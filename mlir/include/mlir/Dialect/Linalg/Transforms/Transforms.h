//===- Transforms.h - Linalg transformations as patterns --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_TRANSFORMS_TRANSFORMS_H
#define MLIR_DIALECT_LINALG_TRANSFORMS_TRANSFORMS_H

#include <utility>

#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/X86Vector/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallSet.h"

namespace mlir {
namespace bufferization {
class BufferizeTypeConverter;
} // namespace bufferization

class FrozenRewritePatternSet;

namespace linalg {

struct LinalgElementwiseFusionOptions;
struct LinalgFusionOptions;
struct LinalgTilingOptions;

//===----------------------------------------------------------------------===//
// Transformations exposed as function calls.
//===----------------------------------------------------------------------===//
using LinalgLoops = SmallVector<Operation *, 4>;

void populatePadTensorTilingPatterns(RewritePatternSet &patterns,
                                     const LinalgTilingOptions &options);

/// Populate patterns for splitting a `LinalgOp` with multiple statements within
/// its payload into multiple `GenericOp` that have a single statement.
void populateDecomposeLinalgOpsPattern(RewritePatternSet &patterns);

/// Populate patterns for vectorizing low-D convolution ops. This is a step in
/// progressive lowering for convolution ops, it assume high-D convolution ops
/// were decomposed previously.
void populateConvolutionVectorizationPatterns(RewritePatternSet &patterns,
                                              PatternBenefit benefit = 1);

/// Populate patterns that convert `ElementwiseMappable` ops to linalg
/// parallel loops.
void populateElementwiseToLinalgConversionPatterns(RewritePatternSet &patterns);

/// Populate patterns that are only useful in the context of sparse tensors.
void populateSparseTensorRewriting(RewritePatternSet &patterns);

/// Function type which is used to control when to stop fusion. It is expected
/// that OpOperand is not modified in the callback. The OpOperand is not marked
/// as const to allow callers to use non-const methods.
using ControlFusionFn = std::function<bool(OpOperand *fusedOperand)>;

/// Patterns for fusing linalg operation on tensors.

/// Pattern to fuse `linalg.generic` -> `linalg.generic` operations
/// when both operations are fusable elementwise operations.
void populateElementwiseOpsFusionPatterns(
    RewritePatternSet &patterns,
    const ControlFusionFn &controlElementwiseOpFusion);

/// Function type to control generic op dimension collapsing. It is expected
/// to return an array of `ReassociationIndices` representing dimensions that
/// should be merged.
using GetCollapsableDimensionsFn =
    std::function<SmallVector<ReassociationIndices>(linalg::GenericOp)>;

/// Pattern to collapse dimensions in a linalg.generic op. This will collapse
/// tensor operands when needed and expand back the result tensors.
void populateCollapseDimensions(
    RewritePatternSet &patterns,
    const GetCollapsableDimensionsFn &controlCollapseDimensions);

/// Patterns to fold an expanding (collapsing) tensor_reshape operation with its
/// producer (consumer) generic operation by expanding the dimensionality of the
/// loop in the generic op.
void populateFoldReshapeOpsByExpansionPatterns(
    RewritePatternSet &patterns, const ControlFusionFn &controlFoldingReshapes);

/// Patterns to fold an expanding tensor.expand_shape operation with its
/// producer generic operation by collapsing the dimensions of the generic op.
void populateFoldReshapeOpsByCollapsingPatterns(
    RewritePatternSet &patterns, const ControlFusionFn &controlFoldingReshapes);

/// Patterns to constant fold Linalg operations.
void populateConstantFoldLinalgOperations(RewritePatternSet &patterns,
                                          const ControlFusionFn &controlFn);

/// Pattern to fuse a `tensor.pad` operation with the producer of its source,
/// if the producer is a `linalg` operation with all parallel iterator types.
void populateFuseTensorPadWithProducerLinalgOpPatterns(
    RewritePatternSet &patterns);

/// Patterns to convert from one named op to another. These can be seen as
/// canonicalizations of named ops into another named op.
void populateLinalgNamedOpConversionPatterns(RewritePatternSet &patterns);

/// Patterns to fold unit-extent dimensions in operands/results of linalg ops on
/// tensors.
void populateFoldUnitExtentDimsPatterns(RewritePatternSet &patterns);

/// Patterns that are used to inline constant operands into linalg generic ops.
void populateInlineConstantOperandsPatterns(RewritePatternSet &patterns);

/// Patterns that are used to bubble up extract slice op above linalg op.
void populateBubbleUpExtractSliceOpPatterns(RewritePatternSet &patterns);

/// Adds patterns that waps tensor.extract_slice(linalg.fill(%cst, %init)) into
/// linalg.fill(%cst, tensor.extract_slice(%init)).
void populateSwapExtractSliceWithFillPatterns(RewritePatternSet &patterns);

/// Return true if two `linalg.generic` operations with producer/consumer
/// relationship through `fusedOperand` can be fused using elementwise op
/// fusion.
bool areElementwiseOpsFusable(OpOperand *fusedOperand);

/// Fuse two `linalg.generic` operations that have a producer-consumer
/// relationship captured through `fusedOperand`. The method expects
/// that `areElementwiseOpsFusable` returns true for the given `fusedOperand`.
FailureOr<Operation *> fuseElementwiseOps(RewriterBase &rewriter,
                                          OpOperand *fusedOperand);

/// Split the given `op` into two parts along the given iteration space
/// `dimension` at the specified `splitPoint`, and return the two parts.
///
/// For example, the following op:
///
///   linalg.matmul ins(%0, %1 : tensor<128x32xf32>, tensor<32x64xf32>)
///                 outs(%2 : tensor<128x64xf32>)
///
/// split along the first dimension at position 42 will result in:
///
///   %3 = tensor.extract_slice %0[0, 0][42, 32][1, 1]
///   %4 = tensor.extract_slice %2[0, 0][42, 64][1, 1]
///   %5 = linalg.matmul ins(%3, %1 : tensor<42x32xf32>, tensor<32x64xf32>)
///                      outs(%5 : tensor<42x64xf32>)
///   %6 = tensor.insert_slice %5 into %2[0, 0][42, 64][1, 1]
///
///   %7 = tensor.extract_slice %0[42, 0][86, 32][1, 1]
///   %8 = tensor.extract_slice %6[42, 0][86, 64][1, 1]
///   %9 = linalg.matmul ins(%7, %1 : tensor<86x32xf32>, tensor<32x64xf32>)
///                      outs(%8 : tensor<86x64xf32>)
///   tensor.insert_slice %5 into %6[42, 0][86, 64][1, 1]
///
/// Note that there is no simplification other than constant propagation applied
/// to slice extraction and insertion.
std::pair<TilingInterface, TilingInterface> splitOp(RewriterBase &rewriter,
                                                    TilingInterface op,
                                                    unsigned dimension,
                                                    OpFoldResult splitPoint);

/// Perform standalone tiling of a single LinalgOp by `tileSizes`.
/// and permute the loop nest according to `interchangeVector`
/// The permutation is expressed as a list of integers that specify
/// the new ordering of the loop nest. The length of `interchangeVector`
/// must be equal to the length of `tileSizes`.
/// An empty vector is interpreted as the identity permutation and the
/// transformation returns early.
///
/// Return a struct containing the tiled loops in the specified order
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
};
FailureOr<TiledLinalgOp> tileLinalgOp(RewriterBase &b, LinalgOp op,
                                      const LinalgTilingOptions &options);

/// Peel and canonicalize 'loops'.
void peelLoops(RewriterBase &rewriter, ArrayRef<scf::ForOp> loops);

/// Peel the loops of a TiledLinalgOp.
void peelTiledLinalgOp(RewriterBase &rewriter, TiledLinalgOp &res,
                       ArrayRef<int64_t> peeledLoops,
                       LinalgTilingLoopType loopType);

/// Interchange the `iterator_types` and `iterator_maps` dimensions and adapts
/// the index accesses of `op`. This is an in-place transformation controlled by
/// `interchangeVector`. An empty vector is interpreted as the identity
/// permutation and the transformation returns early.
///
/// E.g. the permutation `(i,j,k) -> (j,k,i)` is expressed with
/// `interchangeVector = [1,2,0]`. All values in `interchangeVector` must be
/// integers, in the range 0..`op.rank` without duplications
/// (i.e. `[1,1,2]` is an invalid permutation).
///
/// Return failure if the permutation is not valid.
FailureOr<GenericOp> interchangeGenericOp(RewriterBase &rewriter,
                                          GenericOp genericOp,
                                          ArrayRef<unsigned> interchangeVector);

/// Create a GenericOp from the given named operation `namedOp` and replace
/// namedOp.
/// Return failure if `namedOp` is a GenericOp or misses a region builder.
FailureOr<GenericOp> generalizeNamedOp(RewriterBase &rewriter,
                                       LinalgOp namedOp);

/// Callback function type used to perform the allocation for the promoted
/// `subView`. In `boundingSubViewsize` a best attempt is made to find the
/// smallest constant value for the size of the buffer needed for each
/// dimension. If that is not possible, contains the dynamic size of the
/// subview. The call back should return the buffer to use.
using AllocBufferCallbackFn = std::function<Optional<Value>(
    OpBuilder &b, memref::SubViewOp subView,
    ArrayRef<Value> boundingSubViewSize, DataLayout &layout)>;

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
  /// subview. If None then a memref.copy is used.
  Optional<CopyCallbackFn> copyInFn = None;
  Optional<CopyCallbackFn> copyOutFn = None;
  LinalgPromotionOptions &setCopyInOutFns(CopyCallbackFn const &copyIn,
                                          CopyCallbackFn const &copyOut) {
    copyInFn = copyIn;
    copyOutFn = copyOut;
    return *this;
  }
};

/// Create a new buffer using the `allocationFn` provided. The size of this
/// buffer is the smallest constant bounding size along each dimension that can
/// be computed for the size of the result of `subView`. Returns the allocated
/// buffer as `fullLocalView` and the view that matches the size of the result
/// of subview operation as `partialLocalView`.
struct PromotionInfo {
  Value fullLocalView;
  Value partialLocalView;
};
FailureOr<PromotionInfo>
promoteSubviewAsNewBuffer(OpBuilder &b, Location loc, memref::SubViewOp subView,
                          const AllocBufferCallbackFn &allocationFn,
                          DataLayout &layout);

/// Promote the `subViews` into a new buffer allocated at the insertion point
/// `b`. Promotion occurs in 3 steps:
///   1. Create a new buffer for a full tile (i.e. not clipped at the boundary).
///   2. Take a full view on the buffer.
///   3. Take a partial slice of the full view in step 2. and copy into it.
///
/// Return the modified linalg op (the modification happens in place) as well
/// as all the copy ops created.
FailureOr<LinalgOp> promoteSubViews(OpBuilder &b, LinalgOp op,
                                    const LinalgPromotionOptions &options);

/// Emit a suitable vector form for a Linalg op with fully static shape.
LogicalResult vectorize(RewriterBase &builder, LinalgOp linalgOp);

/// Emit a suitable vector form for a Copy op with fully static shape.
LogicalResult vectorizeCopy(RewriterBase &builder, memref::CopyOp copyOp);

/// Emit a loop nest of `scf.for` with the proper body for `linalgOp`.
FailureOr<LinalgLoops> linalgOpToLoops(PatternRewriter &rewriter,
                                       LinalgOp linalgOp);

/// Emit a loop nest of `scf.parallel` with the proper body for `linalgOp`.
FailureOr<LinalgLoops> linalgOpToParallelLoops(PatternRewriter &rewriter,
                                               LinalgOp linalgOp);

/// Emit a loop nest of `affine.for` with the proper body for `linalgOp`.
FailureOr<LinalgLoops> linalgOpToAffineLoops(PatternRewriter &rewriter,
                                             LinalgOp linalgOp);

//===----------------------------------------------------------------------===//
// Preconditions that ensure the corresponding transformation succeeds and can
// be applied as a rewrite pattern.
//===----------------------------------------------------------------------===//
/// Promote memref.subviews feeding linalg-on-buffers operations.
LogicalResult promoteSubviewsPrecondition(Operation *op,
                                          LinalgPromotionOptions options);

/// Return success if the operation can be vectorized.
LogicalResult vectorizeLinalgOpPrecondition(LinalgOp linalgOp);

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
      ArrayRef<StringAttr> matchDisjunction = {},
      Optional<StringAttr> replacement = None);

  explicit LinalgTransformationFilter(
      const FilterFunction &f, ArrayRef<StringAttr> matchDisjunction = {},
      Optional<StringAttr> replacement = None);

  LinalgTransformationFilter(LinalgTransformationFilter &&) = default;
  LinalgTransformationFilter(const LinalgTransformationFilter &) = default;
  LogicalResult checkAndNotify(PatternRewriter &rewriter, Operation *op) const;
  void replaceLinalgTransformationFilter(PatternRewriter &rewriter,
                                         Operation *op) const;
  bool hasReplacementFilter(Operation *op) const;

  LinalgTransformationFilter &addFilter(const FilterFunction &f) {
    if (f)
      filters.push_back(f);
    return *this;
  }

  template <typename... OpTypes>
  LinalgTransformationFilter &addOpFilter() {
    return addFilter(
        [](Operation *op) { return success(isa<OpTypes...>(op)); });
  }

  LinalgTransformationFilter &addOpNameFilter(StringRef opName) {
    return addFilter([opName](Operation *op) {
      return success(op->getName().getStringRef() == opName);
    });
  }

  LinalgTransformationFilter &setMatchByDefault() {
    matchByDefault = true;
    return *this;
  }

private:
  SmallVector<FilterFunction> filters;
  SmallVector<StringAttr> matchDisjunction;
  Optional<StringAttr> replacement;
  /// When set to true, if the attribute is not set, it will be treated as
  /// a match. Default is false.
  bool matchByDefault;
};

using TileSizeComputationFunction =
    std::function<SmallVector<Value, 4>(OpBuilder &, Operation *)>;

/// Creates a number of ranges equal to the number of non-zero in `tileSizes`.
/// One for each loop of the LinalgOp that is tiled. The `tileSizes` argument
/// has one entry per surrounding loop. It uses zero as the convention that a
/// particular loop is not tiled. This convention simplifies implementations by
/// avoiding affine map manipulations.
/// The returned ranges correspond to the loop ranges, in the proper order, that
/// are tiled and for which new loops will be created. Also the function returns
/// a map from loop indices of the LinalgOp to the corresponding non-empty range
/// indices of newly created loops.
using LoopIndexToRangeIndexMap = DenseMap<int, int>;
std::tuple<SmallVector<Range, 4>, LoopIndexToRangeIndexMap>
makeTiledLoopRanges(RewriterBase &b, Location loc, AffineMap map,
                    ArrayRef<OpFoldResult> allShapeSizes,
                    ArrayRef<OpFoldResult> allTileSizes);

/// A description of a multi-size tiling comprising tile sizes and numbers of
/// tiles, expressed as Values which may or may not be constant. Multi-size
/// currently means two-size.
struct MultiSizeSpecification {
  /// Tile sizes.
  Value lowTileSize, highTileSize;
  /// Number of tiles associated with each size.
  Value lowTripCount, highTripCount;
};

/// Emits the IR computing the multi-sized tiling specification with two tile
/// sizes not exceeding `targetSize`, each divisible by `sizeDivisor`, such that
/// there exist numbers of tiles with these sizes that fully cover the given
/// iteration space `dimension` of the structured `op`.
///
/// The computation is as follows:
///
///   b = originalTripCount floordiv sizeDivisor
///   t = (targetSize + sizeDivisor - 1) floordiv sizeDivisor
///   d = (b + t - 1) floordiv t
///   s = (b floordiv d) * sizeDivisor
///   v = b % d
///   u = d - v
///
/// where the tile sizes are `s` and `s` + `sizeDivisor`, and the numbers of
/// the corresponding tiles are `u` and `v`, respectively.  Alternatively,
///
///   s * u + (s + sizeDivisor) * v == original size,
///   where s mod sizeDivisor = 0.
///
/// Expects all values to be positive. In some cases with the target tile size
/// sufficiently close to the dimension shape and non-unit divisor, it is
/// impossible to compute such sizes. If `emitAssertion` is set, also emit the
/// assertion that size computation succeeded.
///
/// Returns the specification consisting of both tile values and the number of
/// tiles of each size.
FailureOr<MultiSizeSpecification>
computeMultiTileSizes(OpBuilder &builder, LinalgOp op, unsigned dimension,
                      OpFoldResult targetSize, OpFoldResult divisor,
                      bool emitAssertions = true);

/// Rewrite a TilingInterface `op` to a tiled `scf.foreach_thread`, applying
/// tiling by `numThreads`.
/// If non-empty, the `threadDimMapping` is added as an attribute to the
/// resulting `scf.foreach_thread`.
/// Zero tile sizes indicate that the dimension is not tiled, and can be thought
/// of as tiling by the full size of data.
/// It is the user's responsibility to ensure that `numThreads` is a
/// valid tiling specification (i.e. that only tiles parallel
/// dimensions, e.g. in the Linalg case).
struct ForeachThreadTilingResult {
  Operation *tileOp;
  Operation *tiledOp;
};
FailureOr<ForeachThreadTilingResult>
tileToForeachThreadOp(RewriterBase &builder, TilingInterface op,
                      ArrayRef<OpFoldResult> numThreads,
                      ArrayRef<int64_t> threadDimMapping = {});

/// Same as `tileToForeachThreadOp`, but calculate the number of threads
/// required using the given tileSizes.
FailureOr<ForeachThreadTilingResult>
tileToForeachThreadOpUsingTileSizes(RewriterBase &builder, TilingInterface op,
                                    ArrayRef<OpFoldResult> tileSizes,
                                    ArrayRef<int64_t> threadDimMapping = {});

/// All indices returned by IndexOp should be invariant with respect to tiling.
/// Therefore, if an operation is tiled, we have to transform the indices
/// accordingly, i.e. offset them by the values of the corresponding induction
/// variables that are captured implicitly in the body of the op.
///
/// Example. `linalg.generic` before tiling:
///
/// #id_2d = (i, j) -> (i, j)
/// #pointwise_2d_trait = {
///   indexing_maps = [#id_2d, #id_2d],
///   iterator_types = ["parallel", "parallel"]
/// }
/// linalg.generic #pointwise_2d_trait %operand, %result {
///   ^bb0(%operand_in: f32, %result_in: f32):
///     %i = linalg.index 0 : index
///     %j = linalg.index 1 : index
///     <some operations that use %i, %j>
/// }: memref<50x100xf32>, memref<50x100xf32>
///
/// After tiling pass with tiles sizes 10 and 25:
///
/// #strided = (i, j)[s0, s1, s2] -> (i * s1 + s0 + j * s2)
///
/// %c1 = arith.constant 1 : index
/// %c0 = arith.constant 0 : index
/// %c25 = arith.constant 25 : index
/// %c10 = arith.constant 10 : index
/// operand_dim_0 = dim %operand, 0 : memref<50x100xf32>
/// operand_dim_1 = dim %operand, 1 : memref<50x100xf32>
/// scf.for %k = %c0 to operand_dim_0 step %c10 {
///   scf.for %l = %c0 to operand_dim_1 step %c25 {
///     %4 = memref.subview %operand[%k, %l][%c10, %c25][%c1, %c1]
///       : memref<50x100xf32> to memref<?x?xf32, #strided>
///     %5 = memref.subview %result[%k, %l][%c10, %c25][%c1, %c1]
///       : memref<50x100xf32> to memref<?x?xf32, #strided>
///     linalg.generic pointwise_2d_trait %4, %5 {
///     ^bb0(%operand_in: f32, %result_in: f32):
///       %i = linalg.index 0 : index
///       %j = linalg.index 1 : index
///       // Indices `k` and `l` are implicitly captured in the body.
///       %transformed_i = arith.addi %i, %k : index // index `i` is offset by
///       %k %transformed_j = arith.addi %j, %l : index // index `j` is offset
///       by %l
///       // Every use of %i, %j is replaced with %transformed_i, %transformed_j
///       <some operations that use %transformed_i, %transformed_j>
///     }: memref<?x?xf32, #strided>, memref<?x?xf32, #strided>
///   }
/// }
///
/// TODO: Investigate whether mixing implicit and explicit indices
/// does not lead to losing information.
void transformIndexOps(RewriterBase &b, LinalgOp op,
                       SmallVectorImpl<Value> &ivs,
                       const LoopIndexToRangeIndexMap &loopIndexToRangeIndex);

struct LinalgPaddingOptions {
  /// A padding value for every operand.
  SmallVector<Attribute> paddingValues;
  LinalgPaddingOptions &setPaddingValues(ArrayRef<Attribute> pv) {
    paddingValues.assign(pv.begin(), pv.end());
    return *this;
  }
  /// A list of iterator dimensions to pad.
  SmallVector<int64_t> paddingDimensions;
  LinalgPaddingOptions &setPaddingDimensions(ArrayRef<int64_t> pd) {
    paddingDimensions.assign(pd.begin(), pd.end());
    return *this;
  }
  /// A flag for every operand to mark the PadOp as nofold which enables packing
  /// for statically shaped operands.
  SmallVector<bool> packPaddings;
  LinalgPaddingOptions &setPackPaddings(ArrayRef<bool> pp) {
    packPaddings.assign(pp.begin(), pp.end());
    return *this;
  }
  /// A number of loops to hoist the PadOp out for every operand.
  SmallVector<int64_t> hoistPaddings;
  LinalgPaddingOptions &setHoistPaddings(ArrayRef<int64_t> hp) {
    hoistPaddings.assign(hp.begin(), hp.end());
    return *this;
  }
  /// A permutation vector for every operand used to transpose the packed PadOp
  /// results.
  SmallVector<SmallVector<int64_t>> transposePaddings;
  LinalgPaddingOptions &
  setTransposePaddings(ArrayRef<SmallVector<int64_t>> tp) {
    transposePaddings.assign(tp.begin(), tp.end());
    return *this;
  }
};

struct LinalgTilingAndFusionOptions {
  /// Tile sizes used to tile the root operation.
  SmallVector<int64_t> tileSizes;
  LinalgTilingAndFusionOptions &setTileSizes(ArrayRef<int64_t> ts) {
    tileSizes.assign(ts.begin(), ts.end());
    return *this;
  }
  /// Tile interchange used to permute the tile loops.
  SmallVector<int64_t> tileInterchange;
  /// When specified, specifies distribution of generated tile loops to
  /// processors.
  Optional<LinalgLoopDistributionOptions> tileDistribution = None;
  LinalgTilingAndFusionOptions &
  setDistributionOptions(LinalgLoopDistributionOptions distributionOptions) {
    tileDistribution = std::move(distributionOptions);
    return *this;
  }
};

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
  LinalgTilingOptions &setTileSizes(const SmallVector<Value, 4> &ts) {
    tileSizeComputationFunction = [=](OpBuilder &, Operation *) { return ts; };
    return *this;
  }
  /// Convenience function to set the `tileSizeComputationFunction` to a
  /// function that computes tile sizes at the point they are needed. Allows
  /// proper interaction with folding.
  LinalgTilingOptions &setTileSizes(ArrayRef<int64_t> ts);

  /// Tile all dynamic dimensions by 1. I.e., scalarize those dimensions.
  /// Note: `scalarizeDynamicDims` and `setTileSizes` cannot be used together.
  LinalgTilingOptions &scalarizeDynamicDims();

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

  /// Specification markers of how to distribute the `linalg.tiled_loop`.
  SmallVector<StringRef, 2> distributionTypes = {};

  LinalgTilingOptions &setDistributionTypes(ArrayRef<StringRef> types) {
    distributionTypes.assign(types.begin(), types.end());
    return *this;
  }

  /// Peel the specified loops.
  SmallVector<int64_t> peeledLoops;

  LinalgTilingOptions &setPeeledLoops(ArrayRef<int64_t> loops) {
    peeledLoops.clear();
    peeledLoops.append(loops.begin(), loops.end());
    return *this;
  }
};

/// Canonicalization patterns relevant to apply after tiling patterns. These are
/// applied automatically by the tiling pass but need to be applied manually
/// when tiling is called programmatically.
RewritePatternSet getLinalgTilingCanonicalizationPatterns(MLIRContext *ctx);
void populateLinalgTilingCanonicalizationPatterns(RewritePatternSet &patterns);

/// Perform tiling using LinalgTilingOptions.
/// Note: this is on a path to deprecation that only works on LinalgOp.
/// Clients should favor using `tileUsingSCFForOp`  that more generally works on
/// TilingInterface.
FailureOr<TiledLinalgOp>
tileWithLinalgTilingOptions(RewriterBase &rewriter, LinalgOp op,
                            const LinalgTilingOptions &options);

///
/// Linalg padding pattern.
///
/// Apply the `padding` transformation as a pattern.
/// See `padding` for more details.
struct LinalgPaddingPattern : public OpInterfaceRewritePattern<LinalgOp> {
  LinalgPaddingPattern(MLIRContext *context,
                       LinalgPaddingOptions options = LinalgPaddingOptions(),
                       PatternBenefit benefit = 1);

  /// `matchAndRewrite` implementation that returns the significant
  /// transformed pieces of IR.
  FailureOr<LinalgOp> returningMatchAndRewrite(LinalgOp op,
                                               PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(LinalgOp op,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(op, rewriter);
  }

private:
  /// Options to control padding and hoisting.
  LinalgPaddingOptions options;
};

/// Rewrites 2-D convolution ops with size-1 window dimensions into 1-D
/// convolution ops.
template <typename Conv2DOp, typename Conv1DOp>
struct DownscaleSizeOneWindowed2DConvolution final
    : public OpRewritePattern<Conv2DOp> {
  using OpRewritePattern<Conv2DOp>::OpRewritePattern;

  FailureOr<Conv1DOp> returningMatchAndRewrite(Conv2DOp convOp,
                                               PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(Conv2DOp convOp,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(convOp, rewriter);
  }
};

extern template struct DownscaleSizeOneWindowed2DConvolution<Conv2DNhwcHwcfOp,
                                                             Conv1DNwcWcfOp>;
extern template struct DownscaleSizeOneWindowed2DConvolution<Conv2DNchwFchwOp,
                                                             Conv1DNcwFcwOp>;

/// Rewrites 2-D depthwise convolution ops with size-1 (w, kw) or (h, kh)
/// dimensions into 1-D depthwise convolution ops.
struct DownscaleDepthwiseConv2DNhwcHwcOp final
    : public OpRewritePattern<DepthwiseConv2DNhwcHwcOp> {
  DownscaleDepthwiseConv2DNhwcHwcOp(MLIRContext *context,
                                    PatternBenefit benefit = 1)
      : OpRewritePattern<DepthwiseConv2DNhwcHwcOp>(context, benefit) {}

  FailureOr<DepthwiseConv1DNwcWcOp>
  returningMatchAndRewrite(DepthwiseConv2DNhwcHwcOp convOp,
                           PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(DepthwiseConv2DNhwcHwcOp convOp,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(convOp, rewriter);
  }
};

///
/// Linalg generalization pattern.
///
/// Apply the `generalization` transformation as a pattern.
/// See `generalization` for more details.
//
// TODO: Automatic default pattern class that just unwraps a function returning
// FailureOr<GenericOp>.
struct LinalgGeneralizationPattern
    : public OpInterfaceRewritePattern<LinalgOp> {
  using OpInterfaceRewritePattern<LinalgOp>::OpInterfaceRewritePattern;

  /// `matchAndRewrite` implementation that returns the significant transformed
  /// pieces of IR.
  FailureOr<GenericOp>
  returningMatchAndRewrite(LinalgOp op, PatternRewriter &rewriter) const {
    return generalizeNamedOp(rewriter, op);
  }

  LogicalResult matchAndRewrite(LinalgOp op,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(op, rewriter);
  }
};

///
/// Linalg vectorization patterns.
///
/// Empty for now, used for SFINAE purposes only.
struct LinalgVectorizationOptions {};

/// `filter` controls LinalgTransformMarker matching and update when specified.
/// See `vectorizeLinalgOp` for more details.
struct CopyVectorizationPattern : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                PatternRewriter &rewriter) const override;
};

/// Return vector::CombiningKind for the given op.
llvm::Optional<vector::CombiningKind> getCombinerOpKind(Operation *combinerOp);

//===----------------------------------------------------------------------===//
// Transformation and lowering options exposed as auxiliary structs.
//===----------------------------------------------------------------------===//
/// Options to control the application of enabling transformations.
/// Hoisting transformations are always deemed beneficial and must be disabled
/// explicitly.
struct LinalgEnablingOptions {
  /// Enable LICM.
  bool licm = true;
  LinalgEnablingOptions &enableLICM(bool val = true) {
    licm = val;
    return *this;
  }
  /// Enable hoisting of redundant vector transfer ops.
  bool hoistRedundantVectorTransfers = true;
  LinalgEnablingOptions &enableHoistRedundantVectorTransfers(bool val = true) {
    hoistRedundantVectorTransfers = val;
    return *this;
  }
  /// Enable hoisting of redundant vector transfer ops on tensor.
  bool hoistRedundantVectorTransfersOnTensor = true;
  LinalgEnablingOptions &
  enableHoistRedundantVectorTransfersOnTensor(bool val = true) {
    hoistRedundantVectorTransfersOnTensor = val;
    return *this;
  }
};

//===----------------------------------------------------------------------===//
// Transformations exposed as rewrite patterns.
//===----------------------------------------------------------------------===//

/// Linalg generalization patterns

/// Populates `patterns` with patterns to convert spec-generated named ops to
/// linalg.generic ops.
void populateLinalgNamedOpsGeneralizationPatterns(RewritePatternSet &patterns);

/// Linalg decompose convolutions patterns

/// Populates patterns to decompose high-D convolution ops into low-D ones.
/// This is a step in progressive lowering for convolution ops, afterwards we
/// can vectorize the low-D convolution ops.
void populateDecomposeConvolutionPatterns(RewritePatternSet &patterns,
                                          PatternBenefit benefit = 1);

//===----------------------------------------------------------------------===//
// Op-specific patterns.
//===----------------------------------------------------------------------===//

/// tensor::PadOp is not canonicalized away yet, so we provide a transformation
/// to `linalg.generic`.
struct PadOpTransformationPattern : public OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern<tensor::PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override;
};

/// Pad the iterator dimensions `paddingDimensions` of all `opToPad` operands to
/// a static bounding box. Use `paddingValues` and `packPaddings` to set padding
/// value and nofold attribute of the created tensor::PadOps, respectively.
/// Update `paddedOp` to the cloned operation with statically shaped
/// `paddingDimensions` and return the extracted dynamically shaped results.
/// If padding fails, return failure.
FailureOr<SmallVector<Value>>
rewriteAsPaddedOp(OpBuilder &b, LinalgOp opToPad,
                  ArrayRef<int64_t> paddingDimensions,
                  ArrayRef<Attribute> paddingValues,
                  ArrayRef<bool> packPaddings, LinalgOp &paddedOp);

using OptimizeCopyFn =
    std::function<LogicalResult(PatternRewriter &, tensor::PadOp, Value)>;

/// Rewrite a tensor::PadOp into a sequence of EmptyOp, FillOp and
/// InsertSliceOp. For now, only constant padding values are supported.
/// `OptimizeCopyFn` can be used to customize copying step optimization.
struct GeneralizePadOpPattern : public OpRewritePattern<tensor::PadOp> {
  GeneralizePadOpPattern(MLIRContext *context,
                         OptimizeCopyFn optimizeCopyFn = nullptr,
                         PatternBenefit benefit = 1)
      : OpRewritePattern<tensor::PadOp>(context, benefit),
        optimizeCopyFn(std::move(optimizeCopyFn)) {}
  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override;

protected:
  OptimizeCopyFn optimizeCopyFn;
  Value createFillOrGenerateOp(PatternRewriter &rewriter, tensor::PadOp padOp,
                               Value dest,
                               const SmallVector<Value> &dynSizes) const;
};

/// Populates `patterns` with patterns that vectorize tensor.pad.
/// These patterns are meant to apply in a complementary fashion. Benefits
/// are used to encode a certain ordering of pattern application. To avoid
/// scattering magic constants throughout the code base, the patterns must be
/// added with this function. `baseBenefit` can be used to offset the benefit
/// of all tensor::PadOp vectorization patterns by a certain value.
void populatePadOpVectorizationPatterns(RewritePatternSet &patterns,
                                        PatternBenefit baseBenefit = 1);

/// Match and rewrite for the pattern:
/// ```
///    %alloc = ...
///    [optional] %view = memref.view %alloc ...
///    %subView = subview %allocOrView ...
///    [optional] linalg.fill(%allocOrView, %cst) ...
///    ...
///    memref.copy(%in, %subView) ...
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
/// Where there is no interleaved use between memref.copy and transfer_read as
/// well as no interleaved use between linalg.fill and memref.copy (if
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
///    memref.copy(%subView, %out)
/// ```
/// into
/// ```
///    [unchanged] %alloc = ...
///    [unchanged] [optional] %view = memref.view %alloc ...
///    [unchanged] %subView = subview %allocOrView...
///    ...
///    vector.transfer_write %..., %out[...]
/// ```
/// Where there is no interleaved use between transfer_write and memref.copy.
/// This is a custom rewrite to forward partial writes to vector.transfer_write.
struct LinalgCopyVTWForwardingPattern
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern<vector::TransferWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp xferOp,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// Support for staged pattern application.
//===----------------------------------------------------------------------===//
/// Helper function to allow applying rewrite patterns, interleaved with more
/// global transformations, in a staged fashion:
///   1. the first stage consists of a list of FrozenRewritePatternSet. Each
///   FrozenRewritePatternSet in this list is applied once, in order.
///   2. the second stage consists of a single RewritePattern that is applied
///      greedily until convergence.
///   3. the third stage consists of applying a lambda, generally used for
///   non-local transformation effects. This allows creating custom fused
///   transformations where patterns can be ordered and applied at a finer
///   granularity than a sequence of traditional compiler passes.
LogicalResult applyStagedPatterns(
    Operation *op, ArrayRef<FrozenRewritePatternSet> stage1Patterns,
    const FrozenRewritePatternSet &stage2Patterns,
    function_ref<LogicalResult(Operation *)> stage3Lambda = nullptr);

/// Rewrite extract_slice(tensor.pad(x)) into tensor.pad(extract_slice(x)).
struct ExtractSliceOfPadTensorSwapPattern
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  /// A function to control pattern application and rewrite logic.
  ///
  /// The function will be given the slice op and should return:
  /// -  None: to fail the match and not apply the pattern;
  /// -  true: to apply the pattern with zero slice guard;
  /// - false: to apply the pattern without zero slice guard.
  ///
  /// See the documentation for tensor::bubbleUpPadSlice regarding zero slice
  /// guard.
  using ControlFn = std::function<llvm::Optional<bool>(tensor::ExtractSliceOp)>;

  ExtractSliceOfPadTensorSwapPattern(MLIRContext *context,
                                     ControlFn controlFn = nullptr,
                                     PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), controlFn(std::move(controlFn)) {}

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override;

private:
  ControlFn controlFn;
};

//===----------------------------------------------------------------------===//
// Helper classes for type list expansion.
//===----------------------------------------------------------------------===//
template <typename... OpTypes>
class VectorizationPatterns;

template <>
class VectorizationPatterns<> {
public:
  static void insert(RewritePatternSet &patterns,
                     const LinalgVectorizationOptions &options,
                     const LinalgTransformationFilter &f) {}
};

/// Split Reduction options.
struct SplitReductionOptions {
  // Ratio used to split the reduction dimension.  If the ratio is <= 1, nothing
  // will be done.
  int64_t ratio = 0;
  // Index where the extra dimension is added to the intermediate tensor shape.
  unsigned index = 0;
  // If the inner dimension after splitting is parallel or reduction.
  bool innerParallel = false;
};

/// Function signature to control reduction splitting. This returns
/// `SplitReductionOptions`.
// TODO: don't use unsigned unless doing bit manipulation.
using ControlSplitReductionFn =
    std::function<SplitReductionOptions(LinalgOp op)>;

/// Patterns to apply `splitReduction` below.
void populateSplitReductionPattern(
    RewritePatternSet &patterns,
    const ControlSplitReductionFn &controlSplitReductionFn,
    bool useAlloc = false);

/// Apply transformation to split the single linalg op reduction into a parallel
/// and reduction dimension. Then create a new linalg.generic op doing the rest
/// of the reduction.
/// Return the new linalg op with an extra parallel dimension or failure if the
/// transformation didn't happen.
///
/// Example:
/// ```
///  %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
///                                        affine_map<(d0) -> ()>],
///       iterator_types = ["reduction"]}
///  ins(%in : tensor<32xf32>)
///  outs(%out : tensor<f32>) {
///  ^bb0(%arg1: f32, %arg2: f32):
///    %y = arith.addf %arg1, %arg2 : f32
///    linalg.yield %y : f32
///  } -> tensor<f32>
/// ```
/// To:
/// ```
///  %cst = arith.constant 0.000000e+00 : f32
///  %0 = tensor.expand_shape %in [[0, 1]] : tensor<32xf32> into tensor<4x8xf32>
///  %1 = tensor.empty [4] : tensor<4xf32>
///  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<4xf32>) -> tensor<4xf32>
///  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
///                                        affine_map<(d0, d1) -> (d0)>],
///    iterator_types = ["parallel", "reduction"]}
///    ins(%0 : tensor<4x8xf32>) outs(%2 : tensor<4xf32>) {
///    ^bb0(%arg3: f32, %arg5: f32):
///    %5 = arith.addf %arg3, %arg4 : f32
///    linalg.yield %5 : f32
///  } -> tensor<4xf32>
/// %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
///                                       affine_map<(d0) -> ()>],
///   iterator_types = ["reduction"]}
///   ins(%3 : tensor<4xf32>) outs(%out : tensor<f32>) {
///   ^bb0(%arg3: f32, %arg4: f32):
///   %5 = arith.addf %arg3, %arg4 : f32
///   linalg.yield %5 : f32
/// } -> tensor<f32>
/// ```
struct SplitReductionResult {
  Operation *initOrAlloc;
  FillOp fillOp;
  LinalgOp splitLinalgOp;
  LinalgOp resultCombiningLinalgOp;
};
FailureOr<SplitReductionResult>
splitReduction(PatternRewriter &b, LinalgOp op,
               const ControlSplitReductionFn &controlSplitReductionFn,
               bool useAlloc = false);

/// Scaling-based implementation of the split reduction transformation.
/// Instead of introducing an ExpandShapeOp, this rewrites a reduction dimension
/// `k` into `k * scale + kk`.
///
/// Example:
/// ```
///  %0 = linalg.matmul ins(%A, %B: tensor<16x256xf32>, tensor<256x32xf32>)
///    outs(%C: tensor<16x32xf32>) -> tensor<16x32xf32>
/// ```
///
/// Is transformed to:
///
/// ```
///  #map0 = affine_map<(d0, d1, d2, d3) -> (d0, d2 * 4 + d3)>
///  #map1 = affine_map<(d0, d1, d2, d3) -> (d2 * 4 + d3, d1)>
///  #map2 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
///  #map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
///  #map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
///  #map5 = affine_map<(d0, d1, d2) -> (d0, d1)>
///  %0 = tensor.empty [16, 32, 64] : tensor<16x32x64xf32>
///  %cst = arith.constant 0.000000e+00 : f32
///  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x32x64xf32>) ->
///     tensor<16x32x64xf32>
///  %2 = tensor.empty [64, 4] : tensor<64x4xi1>
///
///  %3 = linalg.generic {indexing_maps = [#map0, #map1, #map2, #map3],
///    iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
///    ins(%A, %B, %2 : tensor<16x256xf32>, tensor<256x32xf32>, tensor<64x4xi1>)
///   outs(%1 : tensor<16x32x64xf32>) {
///      ^bb0(%arg3: f32, %arg4: f32, %arg5: i1, %arg6: f32):
///        %5 = arith.mulf %arg3, %arg4 : f32
///        %6 = arith.addf %arg6, %5 : f32
///        linalg.yield %6 : f32
///  } -> tensor<16x32x64xf32>
///
///  %4 = linalg.generic {indexing_maps = [#map4, #map5],
///    iterator_types = ["parallel", "parallel", "reduction"]}
//     ins(%3 : tensor<16x32x64xf32>)
///    outs(%C : tensor<16x32xf32>) {
///      ^bb0(%arg3: f32, %arg4: f32):
///        %5 = arith.addf %arg3, %arg4 : f32
///        linalg.yield %5 : f32
///  } -> tensor<16x32xf32>
///
///  return %4 : tensor<16x32xf32>
/// ```
FailureOr<SplitReductionResult>
splitReductionByScaling(PatternRewriter &b, LinalgOp op,
                        const ControlSplitReductionFn &controlSplitReductionFn,
                        bool useAlloc = false);

} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_TRANSFORMS_TRANSFORMS_H
