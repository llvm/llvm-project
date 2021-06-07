//===- Vectorization.cpp - Implementation of linalg Vectorization ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the linalg dialect Vectorization transformations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <type_traits>

using namespace mlir;
using namespace mlir::linalg;

using llvm::dbgs;

#define DEBUG_TYPE "linalg-vectorization"

/// Return the unique instance of OpType in `block` if it is indeed unique.
/// Return null if none or more than 1 instances exist.
template <typename OpType>
static OpType getSingleOpOfType(Block &block) {
  OpType res;
  block.walk([&](OpType op) {
    if (res) {
      res = nullptr;
      return WalkResult::interrupt();
    }
    res = op;
    return WalkResult::advance();
  });
  return res;
}

/// Given an indexing `map` coming from a LinalgOp indexing, restricted to a
/// projectedPermutation, compress the unused dimensions to serve as a
/// permutation_map for a vector transfer operation.
/// For example, given a linalg op such as:
///
/// ```
///   %0 = linalg.generic {
///        indexing_maps = affine_map<(d0, d1, d2, d3, d4) -> (d4, d0, d2)>,
///        indexing_maps = affine_map<(d0, d1, d2, d3, d4) -> (d1, d3)>
///      }
///     ins(%0 : tensor<2x3x4xf32>)
///    outs(%1 : tensor<5x6xf32>)
/// ```
///
/// the iteration domain size of the linalg op is 3x5x4x6x2. The first affine
/// map is reindexed to `affine_map<(d0, d1, d2) -> (d2, d0, d1)>`, the second
/// affine map is reindexed to `affine_map<(d0, d1) -> (d0, d1)>`.
static AffineMap reindexIndexingMap(AffineMap map) {
  assert(map.isProjectedPermutation() && "expected projected permutation");
  auto res = compressUnusedDims(map);
  assert(res.getNumDims() == res.getNumResults() &&
         "expected reindexed map with same number of dims and results");
  return res;
}

/// Helper data structure to represent the result of vectorization.
/// In certain specific cases, like terminators, we do not want to propagate/
enum VectorizationStatus {
  /// Op failed to vectorize.
  Failure = 0,
  /// Op vectorized and custom function took care of replacement logic
  NoReplace,
  /// Op vectorized into a new Op whose results will replace original Op's
  /// results.
  NewOp
  // TODO: support values if Op vectorized to Many-Ops whose results we need to
  // aggregate for replacement.
};
struct VectorizationResult {
  /// Return status from vectorizing the current op.
  enum VectorizationStatus status = VectorizationStatus::Failure;
  /// New vectorized operation to replace the current op.
  /// Replacement behavior is specified by `status`.
  Operation *newOp;
};

/// Return a vector type of the same shape and element type as the (assumed)
/// ShapedType of `v`.
static VectorType extractVectorTypeFromShapedValue(Value v) {
  auto st = v.getType().cast<ShapedType>();
  if (st.isa<MemRefType>() && st.getShape().empty())
    return VectorType();
  return VectorType::get(st.getShape(), st.getElementType());
}

/// Given an `outputOperand` of a LinalgOp, compute the intersection of the
/// forward slice starting from `outputOperand` and the backward slice
/// starting from the corresponding linalg.yield operand.
/// This intersection is assumed to have a single binary operation that is
/// the reduction operation. Multiple reduction operations would impose an
/// ordering between reduction dimensions and is currently unsupported in
/// Linalg. This limitation is motivated by the fact that e.g.
/// min(max(X)) != max(min(X))
// TODO: use in LinalgOp verification, there is a circular dependency atm.
static Operation *getSingleBinaryOpAssumedReduction(OpOperand *outputOperand) {
  auto linalgOp = cast<LinalgOp>(outputOperand->getOwner());
  auto yieldOp = cast<YieldOp>(linalgOp->getRegion(0).front().getTerminator());
  unsigned yieldNum =
      outputOperand->getOperandNumber() - linalgOp.getNumInputs();
  llvm::SetVector<Operation *> backwardSlice, forwardSlice;
  BlockArgument bbArg = linalgOp->getRegion(0).front().getArgument(
      outputOperand->getOperandNumber());
  Value yieldVal = yieldOp->getOperand(yieldNum);
  getBackwardSlice(yieldVal, &backwardSlice, [&](Operation *op) {
    return op->getParentOp() == linalgOp;
  });
  backwardSlice.insert(yieldVal.getDefiningOp());
  getForwardSlice(bbArg, &forwardSlice,
                  [&](Operation *op) { return op->getParentOp() == linalgOp; });
  // Search for the (assumed unique) elementwiseMappable op at the intersection
  // of forward and backward slices.
  Operation *reductionOp = nullptr;
  for (Operation *op : llvm::reverse(backwardSlice)) {
    if (!forwardSlice.contains(op))
      continue;
    if (OpTrait::hasElementwiseMappableTraits(op)) {
      if (reductionOp) {
        // Reduction detection fails: found more than 1 elementwise-mappable op.
        return nullptr;
      }
      reductionOp = op;
    }
  }
  // TODO: also assert no other subsequent ops break the reduction.
  return reductionOp;
}

/// If `value` of assumed VectorType has a shape different than `shape`, try to
/// build and return a new vector.broadcast to `shape`.
/// Otherwise, just return `value`.
// TODO: this is best effort atm and there is currently no guarantee of
// correctness for the broadcast semantics.
static Value broadcastIfNeeded(OpBuilder &b, Value value,
                               ArrayRef<int64_t> shape) {
  unsigned numDimsGtOne = std::count_if(shape.begin(), shape.end(),
                                        [](int64_t val) { return val > 1; });
  auto vecType = value.getType().dyn_cast<VectorType>();
  if (shape.empty() ||
      (vecType != nullptr &&
       (vecType.getShape() == shape || vecType.getRank() > numDimsGtOne)))
    return value;
  auto newVecType = VectorType::get(shape, vecType ? vecType.getElementType()
                                                   : value.getType());
  return b.create<vector::BroadcastOp>(b.getInsertionPoint()->getLoc(),
                                       newVecType, value);
}

static llvm::Optional<vector::CombiningKind>
getKindForOp(Operation *reductionOp) {
  if (!reductionOp)
    return llvm::None;
  return llvm::TypeSwitch<Operation *, llvm::Optional<vector::CombiningKind>>(
             reductionOp)
      .Case<AddIOp, AddFOp>([&](auto op) {
        return llvm::Optional<vector::CombiningKind>{
            vector::CombiningKind::ADD};
      })
      .Default([&](auto op) { return llvm::None; });
}

/// If value of assumed VectorType has a shape different than `shape`, build and
/// return a new vector.broadcast to `shape`.
/// Otherwise, just return value.
static Value reduceIfNeeded(OpBuilder &b, VectorType targetVectorType,
                            Value value, OpOperand *outputOperand) {
  auto linalgOp = cast<LinalgOp>(outputOperand->getOwner());
  assert(targetVectorType.getShape() == linalgOp.getShape(outputOperand));
  auto vecType = value.getType().dyn_cast<VectorType>();
  if (!vecType || vecType.getShape() == targetVectorType.getShape())
    return value;
  // At this point, we know we need to reduce. Detect the reduction operator.
  // TODO: Use the generic reduction detection util.
  Operation *reductionOp = getSingleBinaryOpAssumedReduction(outputOperand);
  unsigned pos = 0;
  MLIRContext *ctx = b.getContext();
  SmallVector<AffineExpr> exprs;
  for (auto s : linalgOp.iterator_types())
    if (isParallelIterator(s))
      exprs.push_back(getAffineDimExpr(pos++, ctx));
  auto loc = value.getLoc();
  // TODO: reuse common CombiningKing logic and support more than add.
  auto maybeKind = getKindForOp(reductionOp);
  assert(maybeKind && "Failed precondition: could not get reduction kind");
  unsigned idx = 0;
  SmallVector<bool> reductionMask(linalgOp.iterator_types().size(), false);
  for (auto attr : linalgOp.iterator_types()) {
    if (isReductionIteratorType(attr))
      reductionMask[idx] = true;
    ++idx;
  }
  return b.create<vector::MultiDimReductionOp>(loc, value, reductionMask,
                                               *maybeKind);
}

/// Build a vector.transfer_read from `source` at indices set to all `0`.
/// If source has rank zero, build an memref.load.
/// Return the produced value.
static Value buildVectorRead(OpBuilder &b, Value source, VectorType vectorType,
                             AffineMap map) {
  Location loc = source.getLoc();
  auto shapedType = source.getType().cast<ShapedType>();
  SmallVector<Value> indices(shapedType.getRank(),
                             b.create<ConstantIndexOp>(loc, 0));
  return b.create<vector::TransferReadOp>(loc, vectorType, source, indices,
                                          map);
}

/// Build a vector.transfer_write of `value` into `outputOperand` at indices set
/// to all `0`; where `outputOperand` is an output operand of the LinalgOp
/// currently being vectorized. If `dest` has null rank, build an memref.store.
/// Return the produced value or null if no value is produced.
static Value buildVectorWrite(OpBuilder &b, Value value,
                              OpOperand *outputOperand) {
  Operation *write;
  Location loc = value.getLoc();
  if (VectorType vectorType =
          extractVectorTypeFromShapedValue(outputOperand->get())) {
    auto linalgOp = cast<LinalgOp>(outputOperand->getOwner());
    AffineMap map =
        reindexIndexingMap(linalgOp.getTiedIndexingMap(outputOperand));
    SmallVector<Value> indices(linalgOp.getRank(outputOperand),
                               b.create<ConstantIndexOp>(loc, 0));
    value = broadcastIfNeeded(b, value, vectorType.getShape());
    value = reduceIfNeeded(b, vectorType, value, outputOperand);
    write = b.create<vector::TransferWriteOp>(loc, value, outputOperand->get(),
                                              indices, map);
  } else {
    write = b.create<memref::StoreOp>(loc, value, outputOperand->get());
  }
  LLVM_DEBUG(dbgs() << "\n[" DEBUG_TYPE "]: vectorized op: " << *write);
  if (!write->getResults().empty())
    return write->getResult(0);
  return Value();
}

// Custom vectorization function type. Produce a vector form of Operation*
// assuming all its vectorized operands are already in the BlockAndValueMapping.
// Return nullptr if the Operation cannot be vectorized.
using CustomVectorizationHook = std::function<VectorizationResult(
    Operation *, const BlockAndValueMapping &)>;

/// Helper function to vectorize the terminator of a `linalgOp`. New result
/// vector values are appended to `newResults`. Return
/// VectorizationStatus::NoReplace to signal the vectorization algorithm that it
/// should not try to map produced operations and instead return the results
/// using the `newResults` vector making them available to the
/// vectorization algorithm for RAUW. This function is meant to be used as a
/// CustomVectorizationHook.
static VectorizationResult
vectorizeLinalgYield(OpBuilder &b, Operation *op,
                     const BlockAndValueMapping &bvm, LinalgOp linalgOp,
                     SmallVectorImpl<Value> &newResults) {
  auto yieldOp = dyn_cast<linalg::YieldOp>(op);
  if (!yieldOp)
    return VectorizationResult{VectorizationStatus::Failure, nullptr};
  for (auto outputs : llvm::enumerate(yieldOp.values())) {
    // TODO: Scan for an opportunity for reuse.
    // TODO: use a map.
    Value vectorValue = bvm.lookup(outputs.value());
    Value newResult = buildVectorWrite(
        b, vectorValue, linalgOp.getOutputOperand(outputs.index()));
    if (newResult)
      newResults.push_back(newResult);
  }
  return VectorizationResult{VectorizationStatus::NoReplace, nullptr};
}

/// Helper function to vectorize the index operations of a `linalgOp`. Return
/// VectorizationStatus::NewOp to signal the vectorization algorithm that it
/// should map the produced operations. This function is meant to be used as a
/// CustomVectorizationHook.
static VectorizationResult vectorizeLinalgIndex(OpBuilder &b, Operation *op,
                                                LinalgOp linalgOp) {
  IndexOp indexOp = dyn_cast<linalg::IndexOp>(op);
  if (!indexOp)
    return VectorizationResult{VectorizationStatus::Failure, nullptr};
  auto loc = indexOp.getLoc();
  // Compute the static loop sizes of the index op.
  auto targetShape = linalgOp.computeStaticLoopSizes();
  // Compute a one-dimensional index vector for the index op dimension.
  SmallVector<int64_t> constantSeq(
      llvm::seq<int64_t>(0, targetShape[indexOp.dim()]));
  ConstantOp constantOp =
      b.create<ConstantOp>(loc, b.getIndexVectorAttr(constantSeq));
  // Return the one-dimensional index vector if it lives in the trailing
  // dimension of the iteration space since the vectorization algorithm in this
  // case can handle the broadcast.
  if (indexOp.dim() == targetShape.size() - 1)
    return VectorizationResult{VectorizationStatus::NewOp, constantOp};
  // Otherwise permute the targetShape to move the index dimension last,
  // broadcast the one-dimensional index vector to the permuted shape, and
  // finally transpose the broadcasted index vector to undo the permutation.
  std::swap(targetShape[indexOp.dim()], targetShape.back());
  auto broadCastOp = b.create<vector::BroadcastOp>(
      loc, VectorType::get(targetShape, b.getIndexType()), constantOp);
  SmallVector<int64_t> transposition(
      llvm::seq<int64_t>(0, linalgOp.getNumLoops()));
  std::swap(transposition.back(), transposition[indexOp.dim()]);
  auto transposeOp =
      b.create<vector::TransposeOp>(loc, broadCastOp, transposition);
  return VectorizationResult{VectorizationStatus::NewOp, transposeOp};
}

/// Generic vectorization for a single operation `op`, given already vectorized
/// operands carried by `bvm`. Vectorization occurs as follows:
///   1. Try to apply any of the `customVectorizationHooks` and return its
///   result on success.
///   2. Clone any constant in the current scope without vectorization: each
///   consumer of the constant will later determine the shape to which the
///   constant needs to be broadcast to.
///   3. Fail on any remaining non `ElementwiseMappable` op. It is the purpose
///   of the `customVectorizationHooks` to cover such cases.
///   4. Clone `op` in vector form to a vector of shape prescribed by the first
///   operand of maximal rank. Other operands have smaller rank and are
///   broadcast accordingly. It is assumed this broadcast is always legal,
///   otherwise, it means one of the `customVectorizationHooks` is incorrect.
///
/// This function assumes all operands of `op` have been vectorized and are in
/// the `bvm` mapping. As a consequence, this function is meant to be called on
/// a topologically-sorted list of ops.
/// This function does not update `bvm` but returns a VectorizationStatus that
/// instructs the caller what `bvm` update needs to occur.
static VectorizationResult
vectorizeOneOp(OpBuilder &b, Operation *op, const BlockAndValueMapping &bvm,
               ArrayRef<CustomVectorizationHook> customVectorizationHooks) {
  LLVM_DEBUG(dbgs() << "\n[" DEBUG_TYPE "]: vectorize op " << *op);

  // 1. Try to apply any CustomVectorizationHook.
  if (!customVectorizationHooks.empty()) {
    for (auto &customFunc : customVectorizationHooks) {
      VectorizationResult result = customFunc(op, bvm);
      if (result.status == VectorizationStatus::Failure)
        continue;
      return result;
    }
  }

  // 2. Constant ops don't get vectorized but rather broadcasted at their users.
  // Clone so that the constant is not confined to the linalgOp block .
  if (isa<ConstantOp>(op))
    return VectorizationResult{VectorizationStatus::NewOp, b.clone(*op)};

  // 3. Only ElementwiseMappable are allowed in the generic vectorization.
  if (!OpTrait::hasElementwiseMappableTraits(op))
    return VectorizationResult{VectorizationStatus::Failure, nullptr};

  // 4. Generic vectorization path for ElementwiseMappable ops.
  //   a. first get the first max ranked shape.
  SmallVector<int64_t, 4> firstMaxRankedShape;
  for (Value operand : op->getOperands()) {
    auto vt = bvm.lookup(operand).getType().dyn_cast<VectorType>();
    if (vt && firstMaxRankedShape.size() < vt.getShape().size())
      firstMaxRankedShape.assign(vt.getShape().begin(), vt.getShape().end());
  }
  //   b. broadcast each op if needed.
  auto vectorizedOperands = llvm::map_range(op->getOperands(), [&](Value v) {
    return firstMaxRankedShape.empty()
               ? bvm.lookup(v)
               : broadcastIfNeeded(b, bvm.lookup(v), firstMaxRankedShape);
  });
  //   c. for elementwise, the result is the vector with the firstMaxRankedShape
  auto returnTypes = llvm::map_range(op->getResultTypes(), [&](Type t) {
    return firstMaxRankedShape.empty()
               ? t
               : VectorType::get(firstMaxRankedShape, t);
  });

  // Build and return the new op.
  OperationState state(op->getLoc(), op->getName());
  state.addAttributes(op->getAttrs());
  state.addOperands(llvm::to_vector<4>(vectorizedOperands));
  state.addTypes(llvm::to_vector<4>(returnTypes));
  return VectorizationResult{VectorizationStatus::NewOp,
                             b.createOperation(state)};
}

/// Detect whether `r` has only ConstantOp, ElementwiseMappable and YieldOp.
static bool hasOnlyScalarElementwiseOp(Region &r) {
  if (!llvm::hasSingleElement(r))
    return false;
  for (Operation &op : r.front()) {
    if (!(isa<ConstantOp, linalg::YieldOp, linalg::IndexOp>(op) ||
          OpTrait::hasElementwiseMappableTraits(&op)) ||
        llvm::any_of(op.getResultTypes(),
                     [](Type type) { return !type.isIntOrIndexOrFloat(); }))
      return false;
  }
  return true;
}

// Return true if the op is an element-wise linalg op.
static bool isElementwise(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp)
    return false;
  if (linalgOp.getNumLoops() != linalgOp.getNumParallelLoops())
    return false;
  // TODO: relax the restrictions on indexing map.
  for (OpOperand *opOperand : linalgOp.getOutputOperands()) {
    if (!linalgOp.getTiedIndexingMap(opOperand).isIdentity())
      return false;
  }
  if (linalgOp->getNumRegions() != 1)
    return false;
  return hasOnlyScalarElementwiseOp(linalgOp->getRegion(0));
}

/// Generic vectorization function that rewrites the body of a `linalgOp` into
/// vector form. Generic vectorization proceeds as follows:
///   1. Verify the `linalgOp` has one non-empty region.
///   2. Values defined above the region are mapped to themselves and will be
///   broadcasted on a per-need basis by their consumers.
///   3. Each region argument is vectorized into a vector.transfer_read (or 0-d
///   load).
///   TODO: Reuse opportunities for RAR dependencies.
///   4a. Register CustomVectorizationHook for YieldOp to capture the results.
///   4b. Register CustomVectorizationHook for IndexOp to access the iteration
///   indices.
///   5. Iteratively call vectorizeOneOp on the region operations.
///
/// When `broadcastToMaximalCommonShape` is set to true, eager broadcasting is
/// performed to the maximal common vector size implied by the `linalgOp`
/// iteration space. This eager broadcasting is introduced in the
/// permutation_map of the vector.transfer_read operations. The eager
/// broadcasting makes it trivial to detrmine where broadcast, transposes and
/// reductions should occur, without any bookkeeping. The tradeoff is that, in
/// the absence of good canonicalizations, the amount of work increases.
/// This is not deemed a problem as we expect canonicalizations and foldings to
/// aggressively clean up the useless work.
LogicalResult vectorizeAsLinalgGeneric(
    OpBuilder &b, LinalgOp linalgOp, SmallVectorImpl<Value> &newResults,
    bool broadcastToMaximalCommonShape = false,
    ArrayRef<CustomVectorizationHook> customVectorizationHooks = {}) {
  // 1. Fail to vectorize if the operation does not have one non-empty region.
  if (linalgOp->getNumRegions() != 1 || linalgOp->getRegion(0).empty())
    return failure();
  auto &block = linalgOp->getRegion(0).front();

  // 2. Values defined above the region can only be broadcast for now. Make them
  // map to themselves.
  BlockAndValueMapping bvm;
  SetVector<Value> valuesSet;
  mlir::getUsedValuesDefinedAbove(linalgOp->getRegion(0), valuesSet);
  bvm.map(valuesSet.getArrayRef(), valuesSet.getArrayRef());

  if (linalgOp.getNumOutputs() == 0)
    return failure();

  // TODO: the common vector shape is equal to the static loop sizes only when
  // all indexing maps are projected permutations. For convs and stencils the
  // logic will need to evolve.
  SmallVector<int64_t> commonVectorShape = linalgOp.computeStaticLoopSizes();

  // 3. Turn all BBArgs into vector.transfer_read / load.
  SmallVector<AffineMap> indexings;
  for (OpOperand *opOperand : linalgOp.getInputAndOutputOperands()) {
    BlockArgument bbarg = block.getArgument(opOperand->getOperandNumber());
    // TODO: 0-d vectors.
    if (linalgOp.getShape(opOperand).empty()) {
      Value loaded =
          b.create<memref::LoadOp>(linalgOp.getLoc(), opOperand->get());
      LLVM_DEBUG(dbgs() << "\n[" DEBUG_TYPE "]: new vectorized bbarg("
                        << bbarg.getArgNumber() << "): " << loaded);
      bvm.map(bbarg, loaded);
      bvm.map(opOperand->get(), loaded);
      continue;
    }
    AffineMap map;
    VectorType vectorType;
    if (broadcastToMaximalCommonShape) {
      map = inverseAndBroadcastProjectedPermuation(
          linalgOp.getTiedIndexingMap(opOperand));
      vectorType = VectorType::get(
          commonVectorShape, getElementTypeOrSelf(opOperand->get().getType()));
    } else {
      map = inversePermutation(
          reindexIndexingMap(linalgOp.getTiedIndexingMap(opOperand)));
      vectorType =
          VectorType::get(map.compose(linalgOp.getShape(opOperand)),
                          getElementTypeOrSelf(opOperand->get().getType()));
    }
    Value vectorRead = buildVectorRead(b, opOperand->get(), vectorType, map);
    LLVM_DEBUG(dbgs() << "\n[" DEBUG_TYPE "]: new vectorized bbarg("
                      << bbarg.getArgNumber() << "): " << vectorRead);
    bvm.map(bbarg, vectorRead);
    bvm.map(opOperand->get(), vectorRead);
  }

  auto hooks = llvm::to_vector<4>(customVectorizationHooks);
  // 4a. Register CustomVectorizationHook for yieldOp.
  CustomVectorizationHook vectorizeYield =
      [&](Operation *op,
          const BlockAndValueMapping &bvm) -> VectorizationResult {
    return vectorizeLinalgYield(b, op, bvm, linalgOp, newResults);
  };
  hooks.push_back(vectorizeYield);

  // 4b. Register CustomVectorizationHook for indexOp.
  CustomVectorizationHook vectorizeIndex =
      [&](Operation *op,
          const BlockAndValueMapping &bvm) -> VectorizationResult {
    return vectorizeLinalgIndex(b, op, linalgOp);
  };
  hooks.push_back(vectorizeIndex);

  // 5. Iteratively call `vectorizeOneOp` to each op in the slice.
  for (Operation &op : block.getOperations()) {
    VectorizationResult result = vectorizeOneOp(b, &op, bvm, hooks);
    if (result.status == VectorizationStatus::Failure) {
      LLVM_DEBUG(dbgs() << "\n[" DEBUG_TYPE "]: failed to vectorize: " << op);
      return failure();
    }
    if (result.status == VectorizationStatus::NewOp) {
      LLVM_DEBUG(dbgs() << "\n[" DEBUG_TYPE "]: new vector op: "
                        << *result.newOp;);
      bvm.map(op.getResults(), result.newOp->getResults());
    }
  }

  return success();
}

static LogicalResult vectorizeContraction(OpBuilder &b, LinalgOp linalgOp,
                                          SmallVectorImpl<Value> &newResults) {
  assert(isaContractionOpInterface(linalgOp) &&
         "expected vectorizeContraction preconditions to be met");
  Location loc = linalgOp.getLoc();
  // Vectorize other ops as vector contraction.
  // TODO: interface.
  LLVM_DEBUG(dbgs() << "\n[" DEBUG_TYPE "]: "
                    << "Rewrite linalg op as vector.contract: ";
             linalgOp.dump());
  // Special function that describes how to vectorize the multiplication op in a
  // linalg contraction.
  CustomVectorizationHook vectorizeContraction =
      [&](Operation *op,
          const BlockAndValueMapping &bvm) -> VectorizationResult {
    if (!isa<MulIOp, MulFOp>(op))
      return VectorizationResult{VectorizationStatus::Failure, nullptr};
    ArrayRef<int64_t> outShape =
        linalgOp.getShape(linalgOp.getOutputOperand(0));
    auto vType = outShape.empty()
                     ? op->getResult(0).getType()
                     : VectorType::get(outShape, op->getResult(0).getType());
    auto zero = b.create<ConstantOp>(loc, vType, b.getZeroAttr(vType));
    // Indexing maps at the time of vector.transfer_read are adjusted to order
    // vector dimensions in the same order as the canonical linalg op iteration
    // space order.
    // The indexings for the contraction therefore need to be adjusted.
    // TODO: consider dropping contraction special casing altogether, this will
    // require more advanced canonicalizations involving vector.multi_reduction
    // that are not yet available.
    SmallVector<AffineMap> indexingMaps;
    indexingMaps.reserve(linalgOp.getNumInputsAndOutputs());
    llvm::transform(linalgOp.getIndexingMaps(),
                    std::back_inserter(indexingMaps),
                    [](AffineMap indexingMap) {
                      return inversePermutation(reindexIndexingMap(indexingMap))
                          .compose(indexingMap);
                    });
    Operation *contract = b.create<vector::ContractionOp>(
        loc, bvm.lookup(op->getOperand(0)), bvm.lookup(op->getOperand(1)), zero,
        b.getAffineMapArrayAttr(indexingMaps), linalgOp.iterator_types());
    return VectorizationResult{VectorizationStatus::NewOp, contract};
  };
  return vectorizeAsLinalgGeneric(b, linalgOp, newResults,
                                  /*broadcastToMaximalCommonShape=*/false,
                                  {vectorizeContraction});
}

static bool allIndexingsAreProjectedPermutation(LinalgOp op) {
  return llvm::all_of(op.getIndexingMaps(),
                      [](AffineMap m) { return m.isProjectedPermutation(); });
}

// TODO: probably need some extra checks for reduction followed by consumer
// ops that may not commute (e.g. linear reduction + non-linear instructions).
static LogicalResult reductionPreconditions(LinalgOp op) {
  if (llvm::none_of(op.iterator_types(), isReductionIteratorType))
    return failure();
  for (OpOperand *opOperand : op.getOutputOperands()) {
    Operation *reductionOp = getSingleBinaryOpAssumedReduction(opOperand);
    if (!getKindForOp(reductionOp))
      return failure();
  }
  return success();
}

LogicalResult mlir::linalg::vectorizeLinalgOpPrecondition(Operation *op) {
  auto linalgOp = cast<linalg::LinalgOp>(op);
  // All types must be static shape to go to vector.
  if (linalgOp.hasDynamicShape())
    return failure();
  if (isElementwise(op))
    return success();
  if (isaContractionOpInterface(linalgOp))
    return success();
  // TODO: the common vector shape is equal to the static loop sizes only when
  // all indexing maps are projected permutations. For convs and stencils the
  // logic will need to evolve.
  if (allIndexingsAreProjectedPermutation(linalgOp) &&
      succeeded(reductionPreconditions(linalgOp)))
    return success();
  return failure();
}

LogicalResult
mlir::linalg::vectorizeLinalgOp(OpBuilder &b, Operation *op,
                                SmallVectorImpl<Value> &newResults) {
  if (failed(vectorizeLinalgOpPrecondition(op)))
    return failure();

  auto linalgOp = cast<LinalgOp>(op);
  if (isaContractionOpInterface(linalgOp))
    return vectorizeContraction(b, linalgOp, newResults);

  LLVM_DEBUG(dbgs() << "\n[" DEBUG_TYPE "]: "
                    << "Vectorize linalg op as a generic by broadcasting to "
                       "maximal common shape: "
                    << *op);
  return vectorizeAsLinalgGeneric(b, linalgOp, newResults,
                                  /*broadcastToMaximalCommonShape=*/true);
}

//----------------------------------------------------------------------------//
// Misc. vectorization patterns.
//----------------------------------------------------------------------------//

/// Given a block, return the Value that the block yields if that Value is
/// constant. In this context, "constant" means "defined outside of the block".
/// Should not be called on blocks that yield more than one value.
///
/// Values are considered constant in two cases:
///  - A basic block argument from a different block.
///  - A value defined outside of the block.
///
/// If the yielded value is not constant, an empty Value is returned.
static Value getConstantYieldValueFromBlock(Block &block) {
  auto yieldOp = cast<YieldOp>(block.getTerminator());
  assert(yieldOp.getNumOperands() == 1 && "expected single operand yield");
  Value result = yieldOp.values().front();
  Operation *definingOp = result.getDefiningOp();

  // Check if yield value is defined inside the block.
  if (definingOp && definingOp->getBlock() == &block)
    return Value();
  // Check if the yield value is a BB arg of the block.
  if (!definingOp && result.cast<BlockArgument>().getOwner() == &block)
    return Value();

  return result;
}

/// Rewrite a PadTensorOp into a sequence of InitTensorOp, TransferReadOp and
/// TransferWriteOp. For now, this only applies when all low and high paddings
/// are determined to be zero.
struct GenericPadTensorOpVectorizationPattern
    : public OpRewritePattern<PadTensorOp> {
  using OpRewritePattern<PadTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PadTensorOp padOp,
                                PatternRewriter &rewriter) const override {
    /// Given an OpFoldResult, return true if its value is guaranteed to be a
    /// zero integer.
    auto isZeroInt = [&](OpFoldResult ofr) {
      return isEqualConstantIntOrValue(ofr, rewriter.getIndexAttr(0)); };
    // Low padding must be static 0.
    if (!llvm::all_of(padOp.getMixedLowPad(), isZeroInt)) return failure();
    // High padding must be static 0.
    if (!llvm::all_of(padOp.getMixedHighPad(), isZeroInt)) return failure();
    // Pad value must be a constant.
    auto padValue = getConstantYieldValueFromBlock(padOp.region().front());
    if (!padValue) return failure();

    // Bail on non-static shapes.
    auto resultShapedType = padOp.result().getType().cast<ShapedType>();
    if (!resultShapedType.hasStaticShape())
      return failure();
    VectorType vectorType = extractVectorTypeFromShapedValue(padOp.result());
    if (!vectorType)
      return failure();

    // Now we can rewrite as InitTensorOp + TransferReadOp@[0..0] +
    // TransferWriteOp@[0..0].
    SmallVector<Value> indices(
        resultShapedType.getRank(),
        rewriter.create<ConstantIndexOp>(padOp.getLoc(), 0));
    Value read = rewriter.create<vector::TransferReadOp>(
        padOp.getLoc(), vectorType, padOp.source(), indices, padValue);
    Value init = rewriter.create<InitTensorOp>(
        padOp.getLoc(), resultShapedType.getShape(),
        resultShapedType.getElementType());
    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(padOp, read, init,
                                                         indices);

    return success();
  }
};

void mlir::linalg::populatePadTensorOpVectorizationPatterns(
    RewritePatternSet &patterns, PatternBenefit baseBenefit) {
  patterns.add<GenericPadTensorOpVectorizationPattern>(
      patterns.getContext(), baseBenefit);
}

// TODO: cleanup all the convolution vectorization patterns.
template <class ConvOp, int N>
LogicalResult ConvOpVectorization<ConvOp, N>::matchAndRewrite(
    ConvOp op, PatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  MLIRContext *context = op.getContext();

  OpOperand *input = op.getInputOperand(0);
  OpOperand *kernel = op.getInputOperand(1);
  OpOperand *output = op.getOutputOperand(0);
  ArrayRef<int64_t> inShape = op.getShape(input);
  ArrayRef<int64_t> kShape = op.getShape(kernel);

  if (llvm::any_of(inShape, ShapedType::isDynamic) ||
      llvm::any_of(kShape, ShapedType::isDynamic))
    return failure();

  SmallVector<AffineExpr, 4> mapping;
  SmallVector<int64_t, 4> vectorDims;
  // Fail to apply when the size of not vectorized dimension is not 1.
  for (unsigned i = 0; i < N; i++) {
    if (!mask[i] && (inShape[i] != 1 || kShape[i] != 1))
      return failure();

    if (mask[i] && inShape[i] != kShape[i])
      return failure();

    if (mask[i]) {
      mapping.push_back(getAffineDimExpr(i, context));
      vectorDims.push_back(inShape[i]);
    }
  }

  int64_t rank = op.getRank(input);
  int64_t numDims = mapping.size();
  Type elemType = getElementTypeOrSelf(input->get().getType());

  auto map = AffineMap::get(rank, 0, mapping, context);
  SmallVector<Value, 4> zeros(rank, rewriter.create<ConstantIndexOp>(loc, 0));
  auto vecType = VectorType::get(vectorDims, elemType);

  auto inputVec = rewriter.create<vector::TransferReadOp>(
      loc, vecType, input->get(), zeros, map);
  auto kernelVec = rewriter.create<vector::TransferReadOp>(
      loc, vecType, kernel->get(), zeros, map);

  auto acc = rewriter.create<ConstantOp>(loc, elemType,
                                         rewriter.getZeroAttr(elemType));

  std::array<AffineMap, 3> indexingMaps{
      AffineMap::getMultiDimIdentityMap(numDims, context),
      AffineMap::getMultiDimIdentityMap(numDims, context),
      AffineMap::get(numDims, 0, {}, context)};

  std::vector<StringRef> iteratorTypes(numDims, "reduction");

  auto result = rewriter.create<vector::ContractionOp>(
      loc, inputVec, kernelVec, acc,
      rewriter.getAffineMapArrayAttr(indexingMaps),
      rewriter.getStrArrayAttr(iteratorTypes));

  rewriter.create<memref::StoreOp>(loc, result, output->get(),
                                   ValueRange(zeros));
  rewriter.eraseOp(op);
  return success();
}

using ConvOpConst = ConvOpVectorization<ConvWOp, 1>;

/// Inserts tiling, promotion and vectorization pattern for ConvOp
/// conversion into corresponding pattern lists.
template <typename ConvOp, unsigned N>
static void populateVectorizationPatterns(
    RewritePatternSet &tilingPatterns, RewritePatternSet &promotionPatterns,
    RewritePatternSet &vectorizationPatterns, ArrayRef<int64_t> tileSizes) {
  auto *context = tilingPatterns.getContext();
  if (tileSizes.size() < N)
    return;

  constexpr static StringRef kTiledMarker = "TILED";
  constexpr static StringRef kPromotedMarker = "PROMOTED";
  tilingPatterns.add<LinalgTilingPattern<ConvOp>>(
      context, LinalgTilingOptions().setTileSizes(tileSizes),
      LinalgTransformationFilter(ArrayRef<Identifier>{},
                                 Identifier::get(kTiledMarker, context)));

  promotionPatterns.add<LinalgPromotionPattern<ConvOp>>(
      context, LinalgPromotionOptions().setUseFullTileBuffersByDefault(true),
      LinalgTransformationFilter(Identifier::get(kTiledMarker, context),
                                 Identifier::get(kPromotedMarker, context)));

  SmallVector<bool, 4> mask(N);
  int offset = tileSizes.size() - N;
  std::transform(tileSizes.begin() + offset, tileSizes.end(), mask.begin(),
                 [](int64_t i) -> bool { return i > 1; });

  vectorizationPatterns.add<ConvOpVectorization<ConvOp, N>>(context, mask);
}

void mlir::linalg::populateConvVectorizationPatterns(
    MLIRContext *context, SmallVectorImpl<RewritePatternSet> &patterns,
    ArrayRef<int64_t> tileSizes) {
  RewritePatternSet tiling(context);
  RewritePatternSet promotion(context);
  RewritePatternSet vectorization(context);
  populateVectorizationPatterns<ConvWOp, 1>(tiling, promotion, vectorization,
                                            tileSizes);

  populateVectorizationPatterns<ConvNWCOp, 3>(tiling, promotion, vectorization,
                                              tileSizes);
  populateVectorizationPatterns<ConvInputNWCFilterWCFOp, 3>(
      tiling, promotion, vectorization, tileSizes);

  populateVectorizationPatterns<ConvNCWOp, 3>(tiling, promotion, vectorization,
                                              tileSizes);
  populateVectorizationPatterns<ConvInputNCWFilterWCFOp, 3>(
      tiling, promotion, vectorization, tileSizes);

  populateVectorizationPatterns<ConvHWOp, 2>(tiling, promotion, vectorization,
                                             tileSizes);

  populateVectorizationPatterns<ConvNHWCOp, 4>(tiling, promotion, vectorization,
                                               tileSizes);
  populateVectorizationPatterns<ConvInputNHWCFilterHWCFOp, 4>(
      tiling, promotion, vectorization, tileSizes);

  populateVectorizationPatterns<ConvNCHWOp, 4>(tiling, promotion, vectorization,
                                               tileSizes);
  populateVectorizationPatterns<ConvInputNCHWFilterHWCFOp, 4>(
      tiling, promotion, vectorization, tileSizes);

  populateVectorizationPatterns<ConvDHWOp, 3>(tiling, promotion, vectorization,
                                              tileSizes);

  populateVectorizationPatterns<ConvNDHWCOp, 5>(tiling, promotion,
                                                vectorization, tileSizes);
  populateVectorizationPatterns<ConvInputNDHWCFilterDHWCFOp, 5>(
      tiling, promotion, vectorization, tileSizes);

  populateVectorizationPatterns<ConvNCDHWOp, 5>(tiling, promotion,
                                                vectorization, tileSizes);
  populateVectorizationPatterns<ConvInputNCDHWFilterDHWCFOp, 5>(
      tiling, promotion, vectorization, tileSizes);

  patterns.push_back(std::move(tiling));
  patterns.push_back(std::move(promotion));
  patterns.push_back(std::move(vectorization));
}

//----------------------------------------------------------------------------//
// Forwarding patterns
//----------------------------------------------------------------------------//

/// Check whether there is any interleaved use of any `values` between `firstOp`
/// and `secondOp`. Conservatively return `true` if any op or value is in a
/// different block.
static bool mayExistInterleavedUses(Operation *firstOp, Operation *secondOp,
                                    ValueRange values) {
  if (firstOp->getBlock() != secondOp->getBlock() ||
      !firstOp->isBeforeInBlock(secondOp)) {
    LLVM_DEBUG(llvm::dbgs() << "\n[" DEBUG_TYPE "]: "
                            << "interleavedUses precondition failed, firstOp: "
                            << *firstOp << ", second op: " << *secondOp);
    return true;
  }
  for (auto v : values) {
    for (auto &u : v.getUses()) {
      Operation *owner = u.getOwner();
      if (owner == firstOp || owner == secondOp)
        continue;
      // TODO: this is too conservative, use dominance info in the future.
      if (owner->getBlock() == firstOp->getBlock() &&
          (owner->isBeforeInBlock(firstOp) || secondOp->isBeforeInBlock(owner)))
        continue;
      LLVM_DEBUG(llvm::dbgs()
                 << "\n[" DEBUG_TYPE "]: "
                 << " found interleaved op " << *owner
                 << ", firstOp: " << *firstOp << ", second op: " << *secondOp);
      return true;
    }
  }
  return false;
}

/// Return the unique subview use of `v` if it is indeed unique, null otherwise.
static memref::SubViewOp getSubViewUseIfUnique(Value v) {
  memref::SubViewOp subViewOp;
  for (auto &u : v.getUses()) {
    if (auto newSubViewOp = dyn_cast<memref::SubViewOp>(u.getOwner())) {
      if (subViewOp)
        return memref::SubViewOp();
      subViewOp = newSubViewOp;
    }
  }
  return subViewOp;
}

/// TODO: use interfaces, side-effects and aliasing analysis as appropriate,
/// when available.
LogicalResult LinalgCopyVTRForwardingPattern::matchAndRewrite(
    vector::TransferReadOp xferOp, PatternRewriter &rewriter) const {

  // Transfer into `view`.
  Value viewOrAlloc = xferOp.source();
  if (!viewOrAlloc.getDefiningOp<memref::ViewOp>() &&
      !viewOrAlloc.getDefiningOp<memref::AllocOp>())
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "\n[" DEBUG_TYPE "]: " << viewOrAlloc);

  // Ensure there is exactly one subview of `viewOrAlloc` defining `subView`.
  memref::SubViewOp subViewOp = getSubViewUseIfUnique(viewOrAlloc);
  if (!subViewOp)
    return failure();
  Value subView = subViewOp.getResult();
  LLVM_DEBUG(llvm::dbgs() << "\n[" DEBUG_TYPE "]: "
                          << "with subView " << subView);

  // Find the copy into `subView` without interleaved uses.
  CopyOp copyOp;
  for (auto &u : subView.getUses()) {
    if (auto newCopyOp = dyn_cast<CopyOp>(u.getOwner())) {
      assert(newCopyOp.output().getType().isa<MemRefType>());
      if (newCopyOp.output() != subView)
        continue;
      LLVM_DEBUG(llvm::dbgs() << "\n[" DEBUG_TYPE "]: "
                              << "copy candidate " << *newCopyOp);
      if (mayExistInterleavedUses(newCopyOp, xferOp, {viewOrAlloc, subView}))
        continue;
      copyOp = newCopyOp;
      break;
    }
  }
  if (!copyOp)
    return failure();
  LLVM_DEBUG(llvm::dbgs() << "\n[" DEBUG_TYPE "]: "
                          << "with copy " << *copyOp);

  // Find the fill into `viewOrAlloc` without interleaved uses before the copy.
  FillOp maybeFillOp;
  for (auto &u : viewOrAlloc.getUses()) {
    if (auto newFillOp = dyn_cast<FillOp>(u.getOwner())) {
      assert(newFillOp.output().getType().isa<MemRefType>());
      if (newFillOp.output() != viewOrAlloc)
        continue;
      LLVM_DEBUG(llvm::dbgs() << "\n[" DEBUG_TYPE "]: "
                              << "fill candidate " << *newFillOp);
      if (mayExistInterleavedUses(newFillOp, copyOp, {viewOrAlloc, subView}))
        continue;
      maybeFillOp = newFillOp;
      break;
    }
  }
  // Ensure padding matches.
  if (maybeFillOp && xferOp.padding() != maybeFillOp.value())
    return failure();
  if (maybeFillOp)
    LLVM_DEBUG(llvm::dbgs() << "\n[" DEBUG_TYPE "]: "
                            << "with maybeFillOp " << *maybeFillOp);

  // `in` is the subview that linalg.copy reads. Replace it.
  Value in = copyOp.input();

  // linalg.copy + linalg.fill can be used to create a padded local buffer.
  // The `masked` attribute is only valid on this padded buffer.
  // When forwarding to vector.transfer_read, the attribute must be reset
  // conservatively.
  Value res = rewriter.create<vector::TransferReadOp>(
      xferOp.getLoc(), xferOp.getVectorType(), in, xferOp.indices(),
      xferOp.permutation_map(), xferOp.padding(), ArrayAttr());

  if (maybeFillOp)
    rewriter.eraseOp(maybeFillOp);
  rewriter.eraseOp(copyOp);
  rewriter.replaceOp(xferOp, res);

  return success();
}

/// TODO: use interfaces, side-effects and aliasing analysis as appropriate,
/// when available.
LogicalResult LinalgCopyVTWForwardingPattern::matchAndRewrite(
    vector::TransferWriteOp xferOp, PatternRewriter &rewriter) const {
  // Transfer into `viewOrAlloc`.
  Value viewOrAlloc = xferOp.source();
  if (!viewOrAlloc.getDefiningOp<memref::ViewOp>() &&
      !viewOrAlloc.getDefiningOp<memref::AllocOp>())
    return failure();

  // Ensure there is exactly one subview of `viewOrAlloc` defining `subView`.
  memref::SubViewOp subViewOp = getSubViewUseIfUnique(viewOrAlloc);
  if (!subViewOp)
    return failure();
  Value subView = subViewOp.getResult();

  // Find the copy from `subView` without interleaved uses.
  CopyOp copyOp;
  for (auto &u : subViewOp.getResult().getUses()) {
    if (auto newCopyOp = dyn_cast<CopyOp>(u.getOwner())) {
      if (newCopyOp.getInputOperand(0)->get() != subView)
        continue;
      if (mayExistInterleavedUses(xferOp, newCopyOp, {viewOrAlloc, subView}))
        continue;
      copyOp = newCopyOp;
      break;
    }
  }
  if (!copyOp)
    return failure();

  // `out` is the subview copied into that we replace.
  assert(copyOp.output().getType().isa<MemRefType>());
  Value out = copyOp.output();

  // Forward vector.transfer into copy.
  // linalg.copy + linalg.fill can be used to create a padded local buffer.
  // The `masked` attribute is only valid on this padded buffer.
  // When forwarding to vector.transfer_write, the attribute must be reset
  // conservatively.
  rewriter.create<vector::TransferWriteOp>(
      xferOp.getLoc(), xferOp.vector(), out, xferOp.indices(),
      xferOp.permutation_map(), ArrayAttr());

  rewriter.eraseOp(copyOp);
  rewriter.eraseOp(xferOp);

  return success();
}
