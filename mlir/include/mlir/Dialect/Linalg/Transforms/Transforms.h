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

/// Default function to control reshape folding. Skips folding unit dimension
/// reshapes.
bool skipUnitDimReshape(const OpResult &producer, OpOperand &consumer);

//===----------------------------------------------------------------------===//
// Transformations exposed as function calls.
//===----------------------------------------------------------------------===//
using LinalgLoops = SmallVector<Operation *, 4>;

void populatePadTensorTilingPatterns(RewritePatternSet &patterns,
                                     const LinalgTilingOptions &options);

/// Populate patterns for vectorizing low-D convolution ops. This is a step in
/// progressive lowering for convolution ops, it assume high-D convolution ops
/// were decomposed previously.
void populateConvolutionVectorizationPatterns(RewritePatternSet &patterns,
                                              PatternBenefit benefit = 1);

/// Populate patterns that convert `ElementwiseMappable` ops to linalg
/// parallel loops.
void populateElementwiseToLinalgConversionPatterns(RewritePatternSet &patterns);

/// Function type which is used to control when to stop fusion. It is expected
/// that OpOperand is not modified in the callback. The OpOperand is not marked
/// as const to allow callers to use non-const methods.
using ControlElementwiseOpsFusionFn =
    std::function<bool(const OpResult &producer, OpOperand &consumer)>;

/// Patterns to fold an expanding (collapsing) tensor_reshape operation with its
/// producer (consumer) generic operation by expanding the dimensionality of the
/// loop in the generic op.
void populateFoldReshapeOpsByExpansionPatterns(
    RewritePatternSet &patterns,
    const ControlElementwiseOpsFusionFn &controlFoldingReshapes =
        skipUnitDimReshape);

/// Patterns to fold a collapsing (expanding) tensor_reshape operation with its
/// producer (consumer) generic operation by linearizing the indexing map used
/// to access the source (target) of the reshape operation in the generic
/// operation.
void populateFoldReshapeOpsByLinearizationPatterns(RewritePatternSet &patterns);

/// Patterns to fold a collapsing (expanding) tensor_reshape operation with its
/// producer (consumer) generic operation by linearizing the indexing map used
/// to access the source (target) of the reshape operation in the generic
/// operation. The patterns are applied only when the tensor reshape involved is
/// collapsing (introducing) unit-extent dimensions.
void populateFoldUnitDimsReshapeOpsByLinearizationPatterns(
    RewritePatternSet &patterns);

/// Pattern to fuse a `linalg.pad_tensor` operation with the producer of its
/// source, if the producer is a `linalg` operation with all parallel iterator
/// types.
void populateFusePadTensorWithProducerLinalgOpPatterns(
    RewritePatternSet &patterns);

/// Patterns to convert from one named op to another. These can be seen as
/// canonicalizations of named ops into another named op.
void populateLinalgNamedOpConversionPatterns(RewritePatternSet &patterns);

/// Populate the given list with patterns to bufferize linalg ops.
void populateLinalgBufferizePatterns(
    bufferization::BufferizeTypeConverter &converter,
    RewritePatternSet &patterns);

/// Create linalg op on buffers given the original tensor-based operation and
/// the buffers for the outputs.
LinalgOp createLinalgOpOnBuffers(ConversionPatternRewriter &rewriter,
                                 LinalgOp linalgOp, ValueRange inputs,
                                 ValueRange outputs);

/// Patterns to fold unit-extent dimensions in operands/results of linalg ops on
/// tensors.
void populateFoldUnitExtentDimsPatterns(RewritePatternSet &patterns);

/// Patterns that are used to inline constant operands into linalg generic ops.
void populateInlineConstantOperandsPatterns(RewritePatternSet &patterns);

/// Pattern to convert TiledLoopOp to SCF loops.
void populateTiledLoopToSCFPattern(RewritePatternSet &patterns);

/// Options that control fusion of elementwise operations.
struct LinalgElementwiseFusionOptions {
  /// Enable fusion of reshapes into the shape with elementwise operations. By
  /// default it is disabled for unit dimensions reshape.
  ControlElementwiseOpsFusionFn controlFoldingReshapesFn = skipUnitDimReshape;

  LinalgElementwiseFusionOptions &
  setControlFoldingReshapes(ControlElementwiseOpsFusionFn fun) {
    controlFoldingReshapesFn = std::move(fun);
    return *this;
  }

  /// Function to allow the caller to control when to stop fusion. Once a
  /// producer is deemed fusable with the consumer (structurally), this callback
  /// can be used to abort the fusion based on non-structural constraints. This
  /// is the hook for cost models to control the amount of fusion done.
  ControlElementwiseOpsFusionFn controlElementwiseOpsFusionFn =
      [](const OpResult & /*producer */, OpOperand & /*consumer */) {
        return true;
      };

  LinalgElementwiseFusionOptions &
  setControlElementwiseOpsFusionFn(ControlElementwiseOpsFusionFn fun) {
    controlElementwiseOpsFusionFn = std::move(fun);
    return *this;
  }
};

/// Patterns for fusing linalg operation on tensors.
void populateElementwiseOpsFusionPatterns(
    RewritePatternSet &patterns,
    LinalgElementwiseFusionOptions options = LinalgElementwiseFusionOptions());

/// Patterns to push reshape op towards the end of the graph in order to expose
/// more fusion opportunities.
void populatePushReshapeOpsPatterns(RewritePatternSet &patterns);

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

/// Peel the loops of a TiledLinalgOp.
void peelTiledLinalgOp(RewriterBase &rewriter, TiledLinalgOp &res,
                       ArrayRef<int64_t> peeledLoops,
                       LinalgTilingLoopType loopType);

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
FailureOr<TiledAndFusedLinalgOps>
tileAndFuseLinalgOps(OpBuilder &builder, ArrayRef<LinalgOp> ops,
                     const LinalgDependenceGraph &dependenceGraph,
                     const LinalgTilingOptions &tilingOptions);

/// Interchange the `iterator_types` and `iterator_maps` dimensions and adapts
/// the index accesses of `op`. This is an in-place transformation controlled by
/// `interchangeVector`. An empty vector is interpreted as the identity
/// permutation and the transformation returns early.
///
/// E.g. the permutation `(i,j,k) -> (j,k,i)` is expressed with
/// `interchangeVector = [1,2,0]`. All values in `interchangeVector` must be
/// integers, in the range 0..`op.rank` without duplications
/// (i.e. `[1,1,2]` is an invalid permutation).
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
/// Infers statically sized buffers from subViews unless `dynamicBuffers` is
/// true.
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
                    ValueRange allShapeSizes, ValueRange allTileSizes);

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
///     %4 = std.subview %operand[%k, %l][%c10, %c25][%c1, %c1]
///       : memref<50x100xf32> to memref<?x?xf32, #strided>
///     %5 = std.subview %result[%k, %l][%c10, %c25][%c1, %c1]
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

/// Callback returning the padding value to use for a given OpOperand or failure
/// for no padding. This should be a function of both the operation and the
/// operand type.
using PaddingValueComputationFunction =
    std::function<FailureOr<Value>(OpBuilder &, OpOperand &)>;

/// Callback returning true if the PadOp defining the given OpOperand shall be
/// marked as nofold to enable packing.
using PaddingNoFoldComputationFunction = std::function<bool(OpOperand &)>;

/// Callback returning the number of loops to hoist the PadOp defining the given
/// OpOperand.
using PaddingHoistComputationFunction = std::function<int64_t(OpOperand &)>;

/// Callback returning the transpose vector used to permute the result tensor
/// dimensions of the PadOp defining the given OpOperand.
using PaddingTransposeComputationFunction =
    std::function<SmallVector<int64_t>(OpOperand &)>;

struct LinalgPaddingOptions {
  /// Callback returning the padding value to use for a given OpOperand or
  /// failure for no padding. Padding operations are introduced if
  /// `paddingValueComputationFunction` is set and does not return failure.
  /// Padding all operands guarantees the operation is statically shaped and
  /// thus can be vectorized.
  PaddingValueComputationFunction paddingValueComputationFunction = nullptr;

  LinalgPaddingOptions &
  setPaddingValueComputationFunction(PaddingValueComputationFunction fun) {
    paddingValueComputationFunction = std::move(fun);
    return *this;
  }

  /// Callback returning true if the PadOp defining the given OpOperand shall be
  /// marked as nofold to enable packing. A padding operation is only marked
  /// nofold if `paddingNoFoldComputationFunction` is set and returns true.
  /// Otherwise, the nofold attribute is set to false.
  PaddingNoFoldComputationFunction paddingNoFoldComputationFunction = nullptr;

  LinalgPaddingOptions &
  setPaddingNoFoldComputationFunction(PaddingNoFoldComputationFunction fun) {
    paddingNoFoldComputationFunction = std::move(fun);
    return *this;
  }

  /// Callback returning the number of loops to hoist the PadOp defining the
  /// given OpOperand.
  PaddingHoistComputationFunction paddingHoistComputationFunction = nullptr;

  LinalgPaddingOptions &
  setPaddingHoistComputationFunction(PaddingHoistComputationFunction fun) {
    paddingHoistComputationFunction = std::move(fun);
    return *this;
  }

  /// Callback returning the transpose vector used to permute the result tensor
  /// dimensions of the PadOp defining the given OpOperand.
  PaddingTransposeComputationFunction paddingTransposeComputationFunction =
      nullptr;

  LinalgPaddingOptions &setPaddingTransposeComputationFunction(
      PaddingTransposeComputationFunction fun) {
    paddingTransposeComputationFunction = std::move(fun);
    return *this;
  }
};

struct LinalgTilingAndFusionOptions {
  /// Tile sizes used to tile the root operation.
  SmallVector<int64_t> tileSizes;
  /// Tile interchange used to permute the tile loops.
  SmallVector<int64_t> tileInterchange;
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

///
/// Linalg tiling pattern.
///
/// Apply the `tiling` transformation as a pattern.
/// `filter` controls LinalgTransformMarker matching and update when specified.
/// See `tiling` for more details.
// TODO: TiledOpInterface
struct LinalgTilingPattern : public OpInterfaceRewritePattern<LinalgOp> {
  /// Construct a generic pattern applied to all LinalgOp that verify `filter`.
  LinalgTilingPattern(
      MLIRContext *context, LinalgTilingOptions options,
      LinalgTransformationFilter f = LinalgTransformationFilter(),
      PatternBenefit benefit = 1);

  /// Construct a pattern specifically applied to `opName`.
  LinalgTilingPattern(
      StringRef opName, MLIRContext *context, LinalgTilingOptions options,
      LinalgTransformationFilter f = LinalgTransformationFilter(),
      PatternBenefit benefit = 1);

  /// `matchAndRewrite` implementation that returns the significant transformed
  /// pieces of IR.
  FailureOr<TiledLinalgOp>
  returningMatchAndRewrite(LinalgOp op, PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(LinalgOp op,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(op, rewriter);
  }

private:
  /// LinalgTransformMarker handles special attribute manipulations.
  LinalgTransformationFilter filter;
  /// Options to control tiling;
  LinalgTilingOptions options;
};

///
/// Linalg padding pattern.
///
/// Apply the `padding` transformation as a pattern.
/// `filter` controls LinalgTransformMarker matching and update when specified.
/// See `padding` for more details.
struct LinalgPaddingPattern : public OpInterfaceRewritePattern<LinalgOp> {
  /// Construct a generic pattern applied to all LinalgOp that verify `filter`.
  LinalgPaddingPattern(
      MLIRContext *context,
      LinalgPaddingOptions options = LinalgPaddingOptions(),
      LinalgTransformationFilter f = LinalgTransformationFilter(),
      PatternBenefit benefit = 1);

  /// Construct a pattern specifically applied to `opName`.
  LinalgPaddingPattern(
      StringRef opName, MLIRContext *context,
      LinalgPaddingOptions options = LinalgPaddingOptions(),
      LinalgTransformationFilter f = LinalgTransformationFilter(),
      PatternBenefit benefit = 1);

  /// `matchAndRewrite` implementation that returns the significant transformed
  /// pieces of IR.
  FailureOr<LinalgOp> returningMatchAndRewrite(LinalgOp op,
                                               PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(LinalgOp op,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(op, rewriter);
  }

private:
  /// LinalgTransformMarker handles special attribute manipulations.
  LinalgTransformationFilter filter;
  /// Options to control padding and hoisting.
  LinalgPaddingOptions options;
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
      LinalgTransformationFilter f = LinalgTransformationFilter(),
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
      LinalgTransformationFilter f = LinalgTransformationFilter(),
      LinalgTransformationFilter fusedOpMarker = LinalgTransformationFilter(),
      LinalgTransformationFilter originalOpMarker =
          LinalgTransformationFilter(),
      PatternBenefit benefit = 1)
      : LinalgBaseTileAndFusePattern(
            OpTy::getOperationName(), context, dependenceGraph, tilingOptions,
            fusionOptions, f, fusedOpMarker, originalOpMarker, benefit) {}
};

///
/// Linalg tile and fuse tensor ops pattern.
///
/// Apply tiling and fusion as a pattern.
/// `filter` controls LinalgTransformMarker matching and update when specified.
/// See `tileConsumerAndFuseProducers` for more details.
struct LinalgTileAndFuseTensorOpsPattern : public RewritePattern {
  // Entry point to match any LinalgOp.
  LinalgTileAndFuseTensorOpsPattern(
      MLIRContext *context, LinalgTilingAndFusionOptions options,
      LinalgTransformationFilter f = LinalgTransformationFilter(),
      PatternBenefit benefit = 1);
  // Entry point to match a specific LinalgOp.
  LinalgTileAndFuseTensorOpsPattern(
      StringRef opName, MLIRContext *context,
      LinalgTilingAndFusionOptions options,
      LinalgTransformationFilter f = LinalgTransformationFilter(),
      PatternBenefit benefit = 1);
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;

private:
  /// LinalgTransformMarker handles special attribute manipulations.
  LinalgTransformationFilter filter;
  /// Tile sizes and interchange used to tile the root operation.
  LinalgTilingAndFusionOptions options;
};

///
/// Linalg generic interchange pattern.
///
/// Apply the `interchange` transformation on a RewriterBase.
/// `filter` controls LinalgTransformMarker matching and update when specified.
/// See `interchange` for more details.
struct GenericOpInterchangePattern : public OpRewritePattern<GenericOp> {
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  /// GenericOp-specific constructor with an optional `filter`.
  GenericOpInterchangePattern(
      MLIRContext *context, ArrayRef<unsigned> interchangeVector,
      LinalgTransformationFilter f = LinalgTransformationFilter(),
      PatternBenefit benefit = 1);

  /// `matchAndRewrite` implementation that returns the significant transformed
  /// pieces of IR.
  FailureOr<GenericOp>
  returningMatchAndRewrite(GenericOp op, PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(op, rewriter);
  }

private:
  /// LinalgTransformMarker handles special attribute manipulations.
  LinalgTransformationFilter filter;
  /// The interchange vector to reorder the iterators and indexing_maps dims.
  SmallVector<unsigned, 8> interchangeVector;
};

///
/// Linalg generalization pattern.
///
/// Apply the `generalization` transformation as a pattern.
/// `filter` controls LinalgTransformMarker matching and update when specified.
/// See `generalization` for more details.
struct LinalgGeneralizationPattern
    : public OpInterfaceRewritePattern<LinalgOp> {
  /// Construct a generic pattern applied to all LinalgOp that verify `filter`.
  LinalgGeneralizationPattern(
      MLIRContext *context,
      LinalgTransformationFilter f = LinalgTransformationFilter(),
      PatternBenefit benefit = 1);

  /// Construct a pattern specifically applied to `opName`.
  LinalgGeneralizationPattern(
      StringRef opName, MLIRContext *context,
      LinalgTransformationFilter f = LinalgTransformationFilter(),
      PatternBenefit benefit = 1);

  /// `matchAndRewrite` implementation that returns the significant transformed
  /// pieces of IR.
  FailureOr<GenericOp>
  returningMatchAndRewrite(LinalgOp op, PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(LinalgOp op,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(op, rewriter);
  }

private:
  /// LinalgTransformMarker handles special attribute manipulations.
  LinalgTransformationFilter filter;
};

///
/// Linalg promotion patterns.
///
/// Apply the `promoteSubViews` transformation as a pattern.
/// `filter` controls LinalgTransformMarker matching and update when specified.
/// See `promoteSubViews` for more details.
struct LinalgBasePromotionPattern : public RewritePattern {
  /// Entry point to match any LinalgOp OpInterface.
  /// MatchAnyOpTag-based constructor with a mandatory `filter`.
  LinalgBasePromotionPattern(
      MLIRContext *context, LinalgTransformationFilter f,
      LinalgPromotionOptions options = LinalgPromotionOptions(),
      PatternBenefit benefit = 1);
  /// Entry point to match a specific Linalg op.
  LinalgBasePromotionPattern(
      StringRef opName, MLIRContext *context, LinalgPromotionOptions options,
      LinalgTransformationFilter f = LinalgTransformationFilter(),
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
      LinalgTransformationFilter f = LinalgTransformationFilter(),
      PatternBenefit benefit = 1)
      : LinalgBasePromotionPattern(OpTy::getOperationName(), context, options,
                                   f, benefit) {}
  /// This constructor is available to anyone.
  LinalgPromotionPattern(
      StringRef opName, MLIRContext *context, LinalgPromotionOptions options,
      LinalgTransformationFilter f = LinalgTransformationFilter(),
      PatternBenefit benefit = 1)
      : LinalgBasePromotionPattern(opName, context, options, f, benefit) {}
};

///
/// Linalg vectorization patterns.
///
/// Empty for now, used for SFINAE purposes only.
struct LinalgVectorizationOptions {};

/// `filter` controls LinalgTransformMarker matching and update when specified.
/// See `vectorizeLinalgOp` for more details.
struct LinalgVectorizationPattern : public OpInterfaceRewritePattern<LinalgOp> {
  /// Construct a generic pattern applied to all LinalgOp that verify `filter`.
  LinalgVectorizationPattern(
      MLIRContext *context,
      LinalgTransformationFilter f = LinalgTransformationFilter(),
      LinalgVectorizationOptions options = LinalgVectorizationOptions(),
      PatternBenefit benefit = 1);

  /// Construct a pattern specifically applied to `opName`.
  LinalgVectorizationPattern(
      StringRef opName, MLIRContext *context,
      LinalgVectorizationOptions options = LinalgVectorizationOptions(),
      LinalgTransformationFilter f = LinalgTransformationFilter(),
      PatternBenefit benefit = 1);

  LogicalResult matchAndRewrite(LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override;

private:
  /// LinalgTransformMarker handles special attribute manipulations.
  LinalgTransformationFilter filter;
};

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

/// Vector lowering options control how ops are lowered down to 1-D and scf.for
/// form.
struct LinalgVectorLoweringOptions {
  /// Enable lowering of vector.contract.
  /// In a progressive lowering of vectors, this would be the 1st step.
  bool contractionLowering = false;
  LinalgVectorLoweringOptions &enableContractionLowering(bool val = true) {
    contractionLowering = val;
    return *this;
  }
  /// Enable lowering of vector.multi_reduce.
  /// In a progressive lowering of vectors, this would be the 2nd step.
  bool multiReductionLowering = false;
  LinalgVectorLoweringOptions &enableMultiReductionLowering(bool val = true) {
    multiReductionLowering = val;
    return *this;
  }
  /// Trigger full / partial vector.transfer splits.
  /// In a progressive lowering of vectors, this would be the 3rd step.
  bool transferPartialRewrite = false;
  LinalgVectorLoweringOptions &enableTransferPartialRewrite(bool val = true) {
    transferPartialRewrite = val;
    return *this;
  }
  /// Enable lowering of vector.transfer to scf.
  /// In a progressive lowering of vectors, this would be the 4th step.
  bool transferToSCFConversion = false;
  LinalgVectorLoweringOptions &enableTransferToSCFConversion(bool val = true) {
    transferToSCFConversion = val;
    return *this;
  }
  /// Maximal transfer rank under which we do not lower further.
  int64_t maxTransferRank = 1;
  LinalgVectorLoweringOptions &setMaxTransferRank(int64_t val) {
    maxTransferRank = val;
    return *this;
  }
  /// Vector lowering operations may result in surprising behavior when
  /// composing multiple codegen strategies and must be enabled explicitly.
  /// In a progressive lowering of vectors, this would be the 5th step.
  bool transferLowering = true;
  LinalgVectorLoweringOptions &enableTransferLowering(bool val = true) {
    transferLowering = val;
    return *this;
  }
  /// Enable lowering of vector.shape_cast to insert/extract.
  /// In a progressive lowering of vectors, this would be the 6th step.
  bool shapeCastLowering = true;
  LinalgVectorLoweringOptions &enableShapeCastLowering(bool val = true) {
    shapeCastLowering = val;
    return *this;
  }
  /// Enable lowering of vector.transpose.
  /// In a progressive lowering of vectors, this would be the 7th step.
  bool transposeLowering = false;
  LinalgVectorLoweringOptions &enableVectorTransposeLowering(bool val = true) {
    transposeLowering = val;
    return *this;
  }
  /// Enable AVX2-specific lowerings.
  bool avx2Lowering = false;
  LinalgVectorLoweringOptions &enableAVX2Lowering(bool val = true) {
    avx2Lowering = val;
    return *this;
  }

  /// Configure the post staged-patterns late vector.transfer to scf
  /// conversion.
  VectorTransferToSCFOptions vectorTransferToSCFOptions;
  LinalgVectorLoweringOptions &
  setVectorTransferToSCFOptions(VectorTransferToSCFOptions options) {
    vectorTransferToSCFOptions = options;
    return *this;
  }
  /// Configure late vector transformations.
  vector::VectorTransformsOptions vectorTransformOptions;
  LinalgVectorLoweringOptions &
  setVectorTransformsOptions(vector::VectorTransformsOptions options) {
    vectorTransformOptions = options;
    return *this;
  }
  /// Configure specialized vector lowerings.
  x86vector::avx2::LoweringOptions avx2LoweringOptions;
  LinalgVectorLoweringOptions &
  setAVX2LoweringOptions(x86vector::avx2::LoweringOptions options) {
    avx2LoweringOptions = options;
    return *this;
  }
};

//===----------------------------------------------------------------------===//
// Transformations exposed as rewrite patterns.
//===----------------------------------------------------------------------===//
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
      LinalgTransformationFilter f = LinalgTransformationFilter(),
      PatternBenefit benefit = 1)
      : RewritePattern(OpTy::getOperationName(), benefit, context),
        filter(std::move(f)), loweringType(loweringType) {}

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
      if (failed(linalgOpToLoops(rewriter, op)))
        return failure();
      break;
    case LinalgLoweringType::AffineLoops:
      if (failed(linalgOpToAffineLoops(rewriter, op)))
        return failure();
      break;
    case LinalgLoweringType::ParallelLoops:
      if (failed(linalgOpToParallelLoops(rewriter, op)))
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
};

/// Linalg generalization patterns

/// Populates `patterns` with patterns to convert spec-generated named ops to
/// linalg.generic ops.
void populateLinalgNamedOpsGeneralizationPatterns(
    RewritePatternSet &patterns,
    const LinalgTransformationFilter &filter = LinalgTransformationFilter());

/// Linalg decompose convolutions patterns

/// Populates patterns to decompose high-D convolution ops into low-D ones. This
/// is a step in progressive lowering for convolution ops, afterwards we can
/// vectorize the low-D convolution ops.
void populateDecomposeConvolutionPatterns(
    RewritePatternSet &patterns,
    const LinalgTransformationFilter &filter = LinalgTransformationFilter(),
    PatternBenefit benefit = 1);

/// Linalg distribution patterns
//
/// Populates `patterns` with patterns to distribute linalg.tiled_loop.
void populateLinalgDistributeTiledLoopPattern(
    RewritePatternSet &patterns, const LinalgLoopDistributionOptions &opts,
    const LinalgTransformationFilter &marker);

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

/// Pad the operands of `opToPad` to a static bounding box. Use `paddingFunc`
/// and `nofoldFunc` to set the padding value and the nofold attribute of the
/// introduced tensor::PadOps, respectively. Update `paddedOp` to the cloned
/// statically shaped operation and return the extracted dynamically shaped
/// results. If padding fails, return failure.
FailureOr<SmallVector<Value>>
rewriteAsPaddedOp(OpBuilder &b, LinalgOp opToPad,
                  const PaddingValueComputationFunction &paddingFunc,
                  const PaddingNoFoldComputationFunction &nofoldFunc,
                  LinalgOp &paddedOp);

using OptimizeCopyFn =
    std::function<LogicalResult(PatternRewriter &, tensor::PadOp, Value)>;

/// Rewrite a tensor::PadOp into a sequence of InitTensorOp, FillOp and
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

/// Populates `patterns` with patterns that vectorize linalg.pad_tensor.
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

/// Rewrite a TiledLoopOp with bounds/step that potentially do not divide evenly
/// into a TiledLoopOp where the step divides the iteration space evenly,
/// followed by another TiledLoopOp for the last (partial) iteration (if any).
/// This transformation is called "loop peeling".
///
/// This function peels the `idx`-th loop of the TiledLoopOp. To tile all loops
/// in the loop nest, this function must be called multiple times.
///
/// After loop peeling, this function tries to simplify/canonicalize affine.min
/// and affine.max ops in the body of the two TiledLoopOps. For more details,
/// refer to `mlir::scf::peelAndCanonicalizeForLoop`.
///
/// The return value indicates whether the loop was rewritten or not. Loops are
/// not rewritten if:
/// * Loop step size is 1 or
/// * Loop bounds and step size are static, and step already divides the
///   iteration space evenly.
///
/// Note: This function rewrites the given TiledLoopOp in-place and clones the
/// TileLoopOp operation for the last iteration. It replaces all uses of the
/// unpeeled TiledLoopOp with the results of the newly generated TiledLoopOp.
LogicalResult peelAndCanonicalizeTiledLoop(RewriterBase &rewriter,
                                           TiledLoopOp loopOp, int64_t idx,
                                           TiledLoopOp &result);

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

/// Rewrite extract_slice(pad_tensor(x)) into pad_tensor(extract_slice(x)).
struct ExtractSliceOfPadTensorSwapPattern
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override;
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

template <typename OpTy, typename... OpTypes>
class VectorizationPatterns<OpTy, OpTypes...> {
public:
  static void insert(RewritePatternSet &patterns,
                     const LinalgVectorizationOptions &options,
                     const LinalgTransformationFilter &f) {
    patterns.add<LinalgVectorizationPattern>(OpTy::getOperationName(),
                                             patterns.getContext(), options, f);
    VectorizationPatterns<OpTypes...>::insert(patterns, options, f);
  }
};

template <typename... OpTypes>
class TilingPatterns;

template <>
class TilingPatterns<> {
public:
  static void insert(RewritePatternSet &patterns,
                     const LinalgTilingOptions &options,
                     const LinalgTransformationFilter &f) {}
};

template <typename OpTy, typename... OpTypes>
class TilingPatterns<OpTy, OpTypes...> {
public:
  static void insert(RewritePatternSet &patterns,
                     const LinalgTilingOptions &options,
                     const LinalgTransformationFilter &f) {
    patterns.add<LinalgTilingPattern>(OpTy::getOperationName(),
                                      patterns.getContext(), options, f);
    TilingPatterns<OpTypes...>::insert(patterns, options, f);
  }
};

} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_TRANSFORMS_TRANSFORMS_H
