//===- Utils.h - Utilities to support the Linalg dialect --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_UTILS_H_
#define MLIR_DIALECT_LINALG_UTILS_H_

#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
class AffineExpr;
class AffineForOp;
class AffineMap;
class PatternRewriter;

namespace linalg {
class LinalgDependenceGraph;

//===----------------------------------------------------------------------===//
// General utilities
//===----------------------------------------------------------------------===//

/// Check if `permutation` is a permutation of the range
/// `[0, permutation.size())`.
bool isPermutation(ArrayRef<int64_t> permutation);

/// Apply the permutation defined by `permutation` to `inVec`.
/// Element `i` in `inVec` is mapped to location `j = permutation[i]`.
/// E.g.: for an input vector `inVec = ['a', 'b', 'c']` and a permutation vector
/// `permutation = [2, 0, 1]`, this function leaves `inVec = ['c', 'a', 'b']`.
template <typename T, unsigned N>
void applyPermutationToVector(SmallVector<T, N> &inVec,
                              ArrayRef<int64_t> permutation) {
  SmallVector<T, N> auxVec(inVec.size());
  for (auto en : enumerate(permutation))
    auxVec[en.index()] = inVec[en.value()];
  inVec = auxVec;
}

/// Helper function that creates a memref::DimOp or tensor::DimOp depending on
/// the type of `source`.
Value createOrFoldDimOp(OpBuilder &b, Location loc, Value source, int64_t dim);

/// Given an operation, retrieves the value of each dynamic dimension through
/// constructing the necessary DimOp operators.
SmallVector<Value, 4> getDynOperands(Location loc, Value val, OpBuilder &b);

/// If `size` comes from an AffineMinOp and one of the values of AffineMinOp
/// is a constant then return a new value set to the smallest such constant.
/// If `size` comes from a ConstantOp, return the constant.
/// Otherwise return nullptr.
IntegerAttr getSmallestBoundingIndex(Value size);

/// Create an ExtractSliceOp and, if `source` is defined by an ExtractSliceOp,
/// fold it by adding the offsets.
///
/// Example:
/// ```
///   %0 = tensor.extract_slice %arg0[3, 4][3, 32][1, 1] : tensor<64x64xf32> to
///                                                        tensor<3x32xf32>
///   %1 = tensor.extract_slice %0[0, 5][3, 4][1, 1] : tensor<3x32xf32> to
///                                                    tensor<3x4xf32>
/// ```
/// folds into:
/// ```
///   %1 = tensor.extract_slice %arg0[3, 9][3, 4][1, 1] : tensor<64x64xf32> to
///                                                         tensor<3x4xf32>
/// ```
tensor::ExtractSliceOp makeComposedExtractSliceOp(
    OpBuilder &b, Location loc, Value source, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides);

//===----------------------------------------------------------------------===//
// Fusion / Tiling utilities
//===----------------------------------------------------------------------===//

/// The type of loops to be generated during tiling.
enum class LinalgTilingLoopType {
  Loops = 0,
  AffineLoops = 1,
  ParallelLoops = 2,
  TiledLoops = 3,
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

/// Compute tile offsets, given a list of loop `ivs` and `tileSizes`. In case a
/// tile size is zero (i.e., no tiling), the corresponding offset is also zero.
SmallVector<Value> computeTileOffsets(OpBuilder &b, Location loc,
                                      ValueRange ivs, ValueRange tileSizes);

/// Compute tile sizes, given a list of loop `ivs`, `tileSizes` and dimension
/// sizes (`sizeBounds`). In case a tile size is zero (i.e., no tiling), the
/// corresponding result size is the corresponding value from `sizeBounds`.
/// Note: The returned tile sizes are closed intervals.
SmallVector<Value> computeTileSizes(OpBuilder &b, Location loc, ValueRange ivs,
                                    ValueRange tileSizes,
                                    ArrayRef<Value> sizeBounds);

/// Creates an extract_slice/subview op for a single `valueToTile` with
/// `builder`. This new operation extracts a tile of `valueToTile`, starting
/// at offsets `lbs` and with sizes `subShapeSizes`.
Value makeTiledShape(OpBuilder &builder, Location loc, Value valueToTile,
                     ValueRange tileSizes, AffineMap map, ValueRange lbs,
                     ValueRange ubs, ValueRange subShapeSizes);

/// Creates extract_slice/subview ops for all `valuesToTile` of the given
/// `linalgOp` with `builder`, assuming `linalgOp` is being fused into a loop
/// nest for tiling with the given induction variables `ivs` and tile sizes
/// `tileSizes`. `sizeBounds` are the iteration space bounds for *all* the
/// implicit loops in `linalgOp`.
///
/// Note that a constant zero in `tileSizes` means no tiling at that implicit
/// loop. The number of non-zero values in `tileSizes` should be equal to the
/// number of values in `ivs`.
SmallVector<Value, 4> makeTiledShapes(OpBuilder &builder, Location loc,
                                      LinalgOp linalgOp,
                                      ArrayRef<Value> valuesToTile,
                                      ValueRange ivs, ValueRange tileSizes,
                                      ArrayRef<Value> sizeBounds);

/// Add the tile loop induction variables `ivs` to the IndexOp results found in
/// the body of the `tiledOp` to account for the tile offset.
void addTileLoopIvsToIndexOpResults(OpBuilder &b, LinalgOp tiledOp,
                                    ArrayRef<Value> ivs);

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
Optional<FusionInfo> fuseProducerOfBuffer(OpBuilder &b,
                                          OpOperand &consumerOpOperand,
                                          const LinalgDependenceGraph &graph);
/// Tensor counterpart of `fuseProducerOfBuffer`.
/// This implements the fusion part of the "tileAndFuse on tensors"
/// transformation and thus requires the `consumerOpOperand` to be a
/// `extract_slice` op (generally obtained by applying the tiling
/// transformation).
Optional<FusionInfo> fuseProducerOfTensor(OpBuilder &b,
                                          OpOperand &consumerOpOperand);
/// Tensor counterpart of `fuseProducerOfBuffer`.
/// This implements the fusion part of the "tileAndFuse on tensors"
/// transformation and thus requires the `consumerOpOperand` to be a
/// `extract_slice` op (generally obtained by applying the tiling
/// transformation). Assumes `producerOfTensor` is a Linalg op that produces
/// `consumerOpOperand`.
Optional<FusionInfo> fuseProducerOfTensor(OpBuilder &b,
                                          OpResult producerOpResult,
                                          OpOperand &consumerOpOperand);

//===----------------------------------------------------------------------===//
// Fusion on tensor utilities
//===----------------------------------------------------------------------===//

/// A struct to manage the tile loop nest specific information.
class TileLoopNest {
public:
  TileLoopNest(LinalgOp rootOp) : rootOp(rootOp) {}

  /// Tile the root operation using the given `tileSizes` and `tileInterchange`.
  LogicalResult tileRootOp(OpBuilder &b, ArrayRef<int64_t> tileSizes,
                           ArrayRef<int64_t> tileInterchange);

  /// Fuse the producer of `rootOpOperand` into the tile loop nest. Returns the
  /// fused producer of fails if fusion is not possible.
  FailureOr<LinalgOp> fuseProducer(OpBuilder &b, OpOperand *rootOpOperand);

  /// Returns the replacement results for the original untiled root operation.
  ValueRange getRootOpReplacementResults();

  /// Returns the tiled root operation.
  LinalgOp getRootOp() { return rootOp; }

private:
  /// Returns true if the tile loop nest has no tile loops.
  bool isEmpty();

  /// Returns true if the tile loop nest invariants are satisfied:
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
  SmallVector<scf::ForOp> loopOps;
  SmallVector<int64_t> loopDims;
};

/// Tiles `consumerOp` and fuses its dependencies if possible. Uses the
/// `tileSizes` and `tileInterchange` parameters to control the tiling.
FailureOr<TileLoopNest>
tileConsumerAndFuseProducers(OpBuilder &b, LinalgOp consumerOp,
                             ArrayRef<int64_t> tileSizes,
                             ArrayRef<int64_t> tileInterchange);

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
  CyclicNumProcsEqNumIters = 2
};

/// Callback function type used to get processor ID, and number of processors
/// used for distribution for all parallel loops generated.
struct ProcInfo {
  Value procId;
  Value nprocs;
};
using ProcInfoCallBackFn = std::function<SmallVector<ProcInfo, 2>(
    OpBuilder &b, Location loc, ArrayRef<Range> parallelLoopRanges)>;
using OneDimProcInfoCallBackFn =
    std::function<ProcInfo(OpBuilder &b, Location loc)>;

/// Options that allow distribution of loops generated in Linalg transforms to
/// processors while generating the loops.
struct LinalgLoopDistributionOptions {
  /// Callback function that returns the Values for processor ID (`procId`), and
  /// number of processors (`nprocs`) used to execute the parallel loops. The
  /// number of `{procId, nprocs}` pairs returned must be equal to the number of
  /// `parallelLoopRanges` passed into the callback, which in-turn is same as
  /// the number of parallel loops for which the `distributionMethod` is
  /// specified below.
  ProcInfoCallBackFn procInfo;
  /// Specification of how to distribute the `scf.parallel` loops that are
  /// generated. As the `scf.parallel` loop is generated, the elements of this
  /// vector is used (from left to right) and the specified distribution is
  /// applied. If the vector is less than the number of `scf.parallel` loops
  /// generated, then no distribution is applied.
  SmallVector<DistributionMethod, 0> distributionMethod = {};

  /// The map keyed by the distribution type that contains callback functions
  /// that return the Values for processor ID (`procId`), and number of
  /// processors (`nprocs`) used to execute the parallel loops.
  DenseMap<StringRef, OneDimProcInfoCallBackFn> procInfoMap;
};

/// Update the `lb`, `ub` and `step` to get per processor `lb`, `ub` and `step`.
void updateBoundsForCyclicDistribution(OpBuilder &builder, Location loc,
                                       Value procId, Value nprocs, Value &lb,
                                       Value &ub, Value &step);

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
                   Optional<LinalgLoopDistributionOptions> = None,
                   ArrayRef<StringRef> distributionTypes = {});
};

} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_UTILS_H_
