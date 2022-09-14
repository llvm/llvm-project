//===- Utils.h - Utilities to support the Linalg dialect --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_UTILS_UTILS_H
#define MLIR_DIALECT_LINALG_UTILS_UTILS_H

#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
class AffineExpr;
class AffineForOp;
class AffineMap;
class PatternRewriter;

namespace tensor {
class ExtractSliceOp;
} // namespace tensor

namespace linalg {
class LinalgDependenceGraph;

//===----------------------------------------------------------------------===//
// General utilities
//===----------------------------------------------------------------------===//

/// Check if all indexing maps are projected permutations.
bool allIndexingsAreProjectedPermutation(LinalgOp op);

/// Detect whether `r` has only ConstantOp, ElementwiseMappable and YieldOp.
bool hasOnlyScalarElementwiseOp(Region &r);

/// Check if a LinalgOp is an element-wise operation.
bool isElementwise(LinalgOp op);

/// Check if `permutation` is a permutation of the range
/// `[0, permutation.size())`.
bool isPermutation(ArrayRef<int64_t> permutation);

/// Check if `attr` has "parallel" iterator type semantics.
bool isParallelIterator(Attribute attr);

/// Check if `attr` has "reduction" iterator type semantics.
bool isReductionIterator(Attribute attr);

/// Helper function that creates a memref::DimOp or tensor::DimOp depending on
/// the type of `source`.
Value createOrFoldDimOp(OpBuilder &b, Location loc, Value source, int64_t dim);
OpFoldResult createFoldedDimOp(OpBuilder &b, Location loc, Value source,
                               int64_t dim);

/// Given an operation, retrieves the value of each dynamic dimension through
/// constructing the necessary DimOp operators.
SmallVector<Value, 4> getDynOperands(Location loc, Value val, OpBuilder &b);

/// Computes an upper bound for the result `value` of an index computation.
/// Translates AffineMinOps and AffineApplyOps along the use-def chains of the
/// index computation to affine constraints and projects out intermediate
/// values. The method sets `boundMap` to an affine map that given
/// `boundOperands` evaluates to an upper bound for the index computation.
///
/// If constantRequired is true, only returns the constant bounds (potentially
/// over-approximating) and fails when not possible.
///
/// Example:
/// ```
/// %dim0 = dim %tensor, %c0
/// %dim1 = dim %tensor, %c1
/// %0 = affine.min affine.map<(d0) -> (40, d0)> (%dim0)
/// %1 = affine.apply affine.map<(d0, d1) -> (d0 + d1)> (%0, %dim1)
/// ```
/// getUpperBoundForIndex(%1, boundMap, boundOperands)
/// set the output parameters to:
/// - boundMap = affine.map<(d0) -> (d0 + 40)>
/// - boundOperands = [%dim1]
void getUpperBoundForIndex(Value value, AffineMap &boundMap,
                           SmallVectorImpl<Value> &boundOperands,
                           bool constantRequired = false);

/// Returns a constant upper bound for the result `value` of an index
/// computation. Calls `getUpperBoundForIndex` and returns a constant upper
/// bound if the result of `boundMap` is a constant expression and failure
/// otherwise.
///
/// Example:
/// ```
/// %0 = affine.min affine.map<(d0) -> (40, d0)> (%d0)
/// %1 = affine.apply affine.map<(d0) -> (d0 + 2)> (%0)
/// ```
/// getConstantUpperBoundForIndex(%1) returns 42
/// (boundsMap = affine.map<() -> (42)>)
FailureOr<int64_t> getConstantUpperBoundForIndex(Value value);

/// Create a tensor::PadOp that pads `source` to the size of the statically
/// sized `type` whose static sizes are assumed to be greater than the dynamic
/// `source` size. The padding introduces trailing `pad` values until the target
/// size is met. If `source` is defined by one or more LinalgOps that have been
/// padded with the same value and sizes, return their padded result instead of
/// creating a tensor::PadOp.
///
/// Example:
/// ```
/// %0 = tensor.extract_slice %arg0 [%iv0, %iv1] [%sz0, %sz1]
/// %1 = tensor.pad %0 low[0, 0] high[...] { tensor.yield %cst }
/// %2 = linalg.matmul ins(...) outs(%1)
/// %3 = tensor.extract_slice %2 [0, 0] [%sz0, %sz1]
/// ```
/// makeComposedPadHighOp(source=%3, pad=%cst) returns %2
/// makeComposedPadHighOp(source=%3, pad=%other_cst) returns %4
/// ```
/// %4 = tensor.pad %3 low[0, 0] high[...] { tensor.yield %other_cst }
/// ```
Value makeComposedPadHighOp(OpBuilder &b, Location loc, RankedTensorType type,
                            Value source, Value pad, bool nofold);

/// Returns a GenericOp that tansposes `inputTensor` into `outputTensor` using
/// `transposeVector` to permute the `inputTensor` dimensions.
GenericOp makeTransposeOp(OpBuilder &b, Location loc, Value inputTensor,
                          Value outputTensor,
                          ArrayRef<int64_t> transposeVector);

/// Returns GenericOp that copies an n-D memref. Unlike the current
/// implementation of memref::CopyOp, this op can further tile, lower to loops
/// or vectorize.
GenericOp makeMemRefCopyOp(OpBuilder &b, Location loc, Value from, Value to);

/// Get the reassociation maps to fold the result of a extract_slice (or source
/// of a insert_slice) operation with given offsets, and sizes to its
/// rank-reduced version. This is only done for the cases where the size is 1
/// and offset is 0. Strictly speaking the offset 0 is not required in general,
/// but non-zero offsets are not handled by SPIR-V backend at this point (and
/// potentially cannot be handled).
Optional<SmallVector<ReassociationIndices>>
getReassociationMapForFoldingUnitDims(ArrayRef<OpFoldResult> mixedSizes);

//===----------------------------------------------------------------------===//
// Fusion / Tiling utilities
//===----------------------------------------------------------------------===//

/// The type of loops to be generated during tiling.
enum class LinalgTilingLoopType {
  Loops = 0,
  AffineLoops = 1,
  ParallelLoops = 2
};

/// Checks whether the specific `producer` is the last write to exactly the
/// whole `consumedView`. This checks structural dominance, that the dependence
/// is a RAW without any interleaved write to any piece of `consumedView`.
bool isProducerLastWriteOfView(const LinalgDependenceGraph &graph,
                               LinalgOp consumer, Value consumedView,
                               LinalgOp producer);

/// Checks whether fusing the specific `producer` of the `consumedView` is
/// feasible. This checks `producer` is the last write of `consumedView` and
/// that no interleaved dependence would be violated (RAW, WAR or WAW).
bool isFusableInto(const LinalgDependenceGraph &graph, LinalgOp consumer,
                   Value consumedView, LinalgOp producer);

/// Computes tile offsets, given a list of loop `ivs` and `tileSizes`. In case a
/// tile size is zero (i.e., no tiling), the corresponding offset is also zero.
SmallVector<OpFoldResult> computeTileOffsets(OpBuilder &b, Location loc,
                                             ArrayRef<OpFoldResult> ivs,
                                             ArrayRef<OpFoldResult> tileSizes);

/// Computes tile sizes, given a list of `tileSizes` and dimension
/// sizes (`sizeBounds`). In case a tile size is zero (i.e., no tiling), the
/// corresponding result size is the corresponding value from `sizeBounds`.
/// Note: The returned tile sizes are closed intervals.
SmallVector<OpFoldResult> computeTileSizes(OpBuilder &b, Location loc,
                                           ArrayRef<OpFoldResult> tileSizes,
                                           ArrayRef<OpFoldResult> sizeBounds);

/// Returns the list of tensor output types produced when the given structured
/// operation `op` is applied to the given `operands`. Note that `operands` are
/// not necessarily the actual operands of `op`.
SmallVector<Type> getTensorOutputTypes(LinalgOp op, ValueRange operands);

/// Creates `insert_slice` ops that insert `results` back into larger tensors
/// they were originally extracted from with `extract_slice` before being passed
/// as `operands` to the given structured operation `op` or its clone. Note that
/// `operands` are not necessarily the actual operands of `op`, the operation
/// serves only as metadata container for operand types and positions.
SmallVector<Value> insertSlicesBack(OpBuilder &builder, Location loc,
                                    LinalgOp op, ValueRange operands,
                                    ValueRange results);

/// A struct containg offsets-sizes-strides arguments of the tiled shape.
struct SliceParameters {
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
};

/// Computes SliceParameters for a single `valueToTile` assuming that its user
/// is being tiled with the given loop bounds `lbs` and `ubs` and the tile sizes
/// `tileSizes`.
///
/// `omitPartialTileCheck` controls whether to omit the partial/boundary tile
/// condition check in cases where we statically know that it is unnecessary.
SliceParameters
computeSliceParameters(OpBuilder &builder, Location loc, Value valueToTile,
                       ArrayRef<OpFoldResult> tileSizes, AffineMap map,
                       ArrayRef<OpFoldResult> lbs, ArrayRef<OpFoldResult> ubs,
                       ArrayRef<OpFoldResult> subShapeSizes,
                       bool omitPartialTileCheck);

/// Computes SliceParamaters for all `valuesToTile` of the given `linalgOp`,
/// assuming `linalgOp` is being fused into a loop nest. Calls
/// `computeSliceParameters` for every individual value.
///
/// Note that a constant zero in `tileSizes` means no tiling at that implicit
/// loop. The number of non-zero values in `tileSizes` should be equal to the
/// number of values in `ivs`.
///
/// Some of the `valuesToTile` won't be affected by tiling. For these values,
/// llvm::None will be returned.
SmallVector<Optional<SliceParameters>>
computeAllSliceParameters(OpBuilder &builder, Location loc, LinalgOp linalgOp,
                          ValueRange valuesToTile, ArrayRef<OpFoldResult> ivs,
                          ArrayRef<OpFoldResult> tileSizes,
                          ArrayRef<OpFoldResult> sizeBounds,
                          bool omitPartialTileCheck);

/// Creates an extract_slice/subview op for a single `valueToTile` with
/// `builder`. This new operation extracts a tile of `valueToTile`, starting
/// at offsets `lbs` and with sizes `subShapeSizes`. `omitPartialTileCheck`
/// controls whether to omit the partial/boundary tile condition check in cases
/// where we statically know that it is unnecessary.
Value makeTiledShape(OpBuilder &builder, Location loc, Value valueToTile,
                     ArrayRef<OpFoldResult> tileSizes, AffineMap map,
                     ArrayRef<OpFoldResult> lbs, ArrayRef<OpFoldResult> ubs,
                     ArrayRef<OpFoldResult> subShapeSizes,
                     bool omitPartialTileCheck);

/// Creates extract_slice/subview ops for all `valuesToTile` of the given
/// `linalgOp` with `builder`, assuming `linalgOp` is being fused into a loop
/// nest for tiling with the given induction variables `ivs` and tile sizes
/// `tileSizes`. `sizeBounds` are the iteration space bounds for *all* the
/// implicit loops in `linalgOp`. `omitPartialTileCheck` controls whether to
/// omit the partial/boundary tile condition check in cases where we statically
/// know that it is unnecessary.
///
/// Note that a constant zero in `tileSizes` means no tiling at that implicit
/// loop. The number of non-zero values in `tileSizes` should be equal to the
/// number of values in `ivs`.
SmallVector<Value> makeTiledShapes(OpBuilder &builder, Location loc,
                                   LinalgOp linalgOp, ValueRange valuesToTile,
                                   ArrayRef<OpFoldResult> ivs,
                                   ArrayRef<OpFoldResult> tileSizes,
                                   ArrayRef<OpFoldResult> sizeBounds,
                                   bool omitPartialTileCheck);

/// Add the specified offsets to any `linalg.index` ops contained in the given
/// `linalgOp`. The offsets are provided in the same order as iteration space
/// dimensions. Null offests are assumed to be zero.
void offsetIndices(OpBuilder &b, LinalgOp linalgOp,
                   ArrayRef<OpFoldResult> offests);
void offsetIndices(RewriterBase &b, LinalgOp linalgOp,
                   ArrayRef<OpFoldResult> offests);

using FusableOpDependencesTy = llvm::MapVector<
    Operation *,
    SmallVector<LinalgDependenceGraph::LinalgDependenceGraphElem, 1>>;
FusableOpDependencesTy
findAllFusableDependences(ArrayRef<LinalgOp> ops,
                          const LinalgDependenceGraph &dependenceGraph);

/// A struct containing the Linalg producer before and after fusion.
/// When operating on tensors, `fusedProducer` may feed into a `tensor.cast` op
/// before the consumer Linalg op, until enough canonicalizations have applied.
struct FusionInfo {
  LinalgOp originalProducer;
  LinalgOp fusedProducer;
};

/// Fuses producer into consumer if the producer is structurally feasible and
/// the fusion would not violate dependencies.
/// Implements the fusion part of the "tileAndFuse on buffers" transformation
/// and thus requires the `consumerOpOperand` to be a `subview` op (generally
/// obtained by applying the tiling transformation).
FailureOr<FusionInfo> fuseProducerOfBuffer(OpBuilder &b,
                                           OpOperand &consumerOpOperand,
                                           const LinalgDependenceGraph &graph);
/// Tensor counterpart of `fuseProducerOfBuffer`.
/// This implements the fusion part of the "tileAndFuse on tensors"
/// transformation and thus requires the `consumerOpOperand` to be a
/// `extract_slice` op (generally obtained by applying the tiling
/// transformation).
FailureOr<FusionInfo> fuseProducerOfTensor(OpBuilder &b,
                                           OpOperand &consumerOpOperand);
/// Tensor counterpart of `fuseProducerOfBuffer`.
/// This implements the fusion part of the "tileAndFuse on tensors"
/// transformation and thus requires the `consumerOpOperand` to be a
/// `extract_slice` op (generally obtained by applying the tiling
/// transformation). Assumes `producerOfTensor` is a Linalg op that produces
/// `consumerOpOperand`.
FailureOr<FusionInfo> fuseProducerOfTensor(OpBuilder &b,
                                           OpResult producerOpResult,
                                           OpOperand &consumerOpOperand);

//===----------------------------------------------------------------------===//
// Distribution utilities
//===----------------------------------------------------------------------===//

/// Scheme used to distribute loops to processors.
enum class DistributionMethod {
  /// Cyclic distribution where no assumption is made about the dynamic
  /// relationship between number of processors and number of iterations of the
  /// distributed loop. Distributes the following loop
  ///
  /// scf.parallel (%iv) = (%lb) to (%ub) step (%step)
  ///
  /// to
  ///
  /// scf.parallel(%iv)= (%lb + %procId * %step) to (%ub) step (%step * %nprocs)
  Cyclic = 0,

  /// Cyclic distribution where the number of processors can be assumed to be
  /// more than or equal to the number of iterations of the distributed loop. In
  /// such cases, a simple in-bounds check is enough (instead of materializing a
  /// loop). Distributes the following loop
  ///
  /// scf.parallel (%iv) = (%lb) to (%ub) step (%step)
  ///
  /// to
  ///
  /// %iv = %lb + %procId * %step
  /// %cond = arith.cmpi "slt", %iv, %ub
  /// scf.if %cond {
  ///   ...
  /// }
  CyclicNumProcsGeNumIters = 1,

  /// Cyclic distribution where the number of processors can be assumed to be
  ///  equal to the number of iterations of the distributed loop. In such cases,
  ///  no bounds check is needed. Distributes the following loop
  ///
  /// scf.parallel (%iv) = (%lb) to (%ub) step (%step)
  ///
  /// to
  ///
  /// %iv = %lb + %procId * %step
  CyclicNumProcsEqNumIters = 2,

  /// No Distribution.
  None = 3
};

/// Callback function type used to get processor ID, and number of processors
/// used for distribution for all parallel loops generated.
struct ProcInfo {
  Value procId;
  Value nprocs;
  DistributionMethod distributionMethod;
};
using ProcInfoCallBackFn = std::function<SmallVector<ProcInfo>(
    OpBuilder &b, Location loc, ArrayRef<Range> parallelLoopRanges)>;

/// Options that allow distribution of loops generated in Linalg transforms to
/// processors while generating the loops.
struct LinalgLoopDistributionOptions {
  /// Callback function that returns the Values for processor ID (`procId`), and
  /// number of processors (`nprocs`) used to execute the parallel loops. The
  /// number of `{procId, nprocs}` pairs returned must be equal to the number of
  /// `parallelLoopRanges` passed into the callback. The `parallelLoopRanges`
  /// are ranges of the outer parallel loops of the operation that
  /// do have non-zero tile sizes specified.
  ProcInfoCallBackFn procInfo;
};

/// Update the `lb`, `ub` and `step` to get per processor `lb`, `ub` and `step`.
void updateBoundsForCyclicDistribution(OpBuilder &builder, Location loc,
                                       Value procId, Value nprocs, Value &lb,
                                       Value &ub, Value &step);

//===----------------------------------------------------------------------===//
// Fusion on tensor utilities
//===----------------------------------------------------------------------===//

/// A struct to manage the tile loop nest specific information.
class TileLoopNest {
public:
  TileLoopNest(LinalgOp rootOp) : rootOp(rootOp) {}

  /// Tile the root operation using the given `tileSizes` and `tileInterchange`,
  /// and `tileDistribution`.
  LogicalResult
  tileRootOp(OpBuilder &b, ArrayRef<int64_t> tileSizes,
             ArrayRef<int64_t> tileInterchange,
             Optional<LinalgLoopDistributionOptions> tileDistribution);

  /// Fuse the producer of `consumerOpOperand` into the tile loop nest. Returns
  /// the fused producer or fails if fusion is not possible.
  FailureOr<LinalgOp> fuseProducer(OpBuilder &b, OpOperand *consumerOpOperand);

  /// Returns the replacement results for the original untiled root operation.
  ValueRange getRootOpReplacementResults();

  /// Returns the tiled root operation.
  LinalgOp getRootOp() { return rootOp; }

  /// Returns the tiled root operation and the fused producers.
  SmallVector<LinalgOp> getAllTiledAndFusedOps();

  /// Returns the loop ops generated from tiling.
  ArrayRef<scf::ForOp> getLoopOps() { return tileLoopOps; }

  /// Returns true if the tile loop nest has no tile loops.
  bool isEmpty();

private:
  /// Returns true if the tile loop nest invariants are satisfied:
  /// - The `rootOp` has been tiled at least once.
  /// - The number of tile loop operations and dimensions match.
  /// - The innermost tile loop is the parent of `tiledOp`.
  /// - The tile loops are directly nested.
  // TODO: relax to support additional control flow, e.g., IfOp.
  bool isValid();

  /// Searches the block arguments tied to a block argument `bbArg` of the
  /// innermost tile loop. Returns the block argument from outermost to
  /// innermost or an empty vector if none are found.
  SmallVector<BlockArgument> getTiedBBArgs(BlockArgument bbArg);

  /// Returns the iteration argument of the outermost tile loop mapped to a
  /// block argument `bbArg` of the innermost tile loop.
  OpOperand *getTiedIterArg(BlockArgument bbArg);

  /// Returns true if `bbArg` has other used than `sliceOp` and its
  /// dependencies. Only if there are no other uses, the producer output
  /// iteration argument may reused to pass the producer result after fusion.
  bool hasOtherUses(BlockArgument bbArg, tensor::ExtractSliceOp sliceOp);

  LinalgOp rootOp;
  SmallVector<scf::ForOp> tileLoopOps;
  DenseMap<Operation *, SmallVector<int64_t>> tiledRootAndFusedOpsLoops;
};

/// Tiles `consumerOp` and fuses its dependencies if possible. Uses the
/// `tileSizes`, `tileInterchange`, and `tileDistribution` parameters to control
/// the tiling.
FailureOr<TileLoopNest> tileConsumerAndFuseProducers(
    OpBuilder &b, LinalgOp consumerOp, ArrayRef<int64_t> tileSizes,
    ArrayRef<int64_t> tileInterchange,
    const Optional<LinalgLoopDistributionOptions> &tileDistribution);

//===----------------------------------------------------------------------===//
// Generic op region utilities
//===----------------------------------------------------------------------===//

/// A struct containing common matchers over linalg op's region.
struct RegionMatcher {
  enum class BinaryOpKind {
    IAdd,
  };

  /// Matches the given linalg op if its body is performing binary operation on
  /// int or float scalar values and returns the binary op kind.
  ///
  /// The linalg op's region is expected to be
  /// ```
  /// {
  ///   ^bb(%a: <scalar-type>, %b: <scalar-type>):
  ///     %0 = <binary-op> %a, %b: <scalar-type>
  ///     linalg.yield %0: <scalar-type>
  /// }
  /// ```
  static Optional<BinaryOpKind> matchAsScalarBinaryOp(GenericOp op);
};

//===----------------------------------------------------------------------===//
// Loop nest utilities
//===----------------------------------------------------------------------===//

/// Utility class used to generate nested loops with ranges described by
/// `loopRanges` and loop type described by the `iteratorTypes`. `bodyBuilderFn`
/// is used to generate the body of the innermost loop. It is passed a range
/// of loop induction variables and a range of operand values to use.
template <typename LoopTy>
struct GenerateLoopNest {
  static void doit(OpBuilder &b, Location loc, ArrayRef<Range> loopRanges,
                   LinalgOp linalgOp, ArrayRef<Attribute> iteratorTypes,
                   function_ref<scf::ValueVector(OpBuilder &, Location,
                                                 ValueRange, ValueRange)>
                       bodyBuilderFn,
                   ArrayRef<linalg::ProcInfo> procInfo = {});
};

} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_UTILS_UTILS_H
