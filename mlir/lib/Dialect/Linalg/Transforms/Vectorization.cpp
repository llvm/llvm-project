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
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <type_traits>

using namespace mlir;
using namespace mlir::linalg;

#define DEBUG_TYPE "linalg-vectorization"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X)

/// Try to vectorize `convOp` as a convolution.
static FailureOr<Operation *> vectorizeConvolution(OpBuilder &b,
                                                   LinalgOp convOp);

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
  assert(map.isProjectedPermutation(/*allowZeroInResults=*/true) &&
         "expected projected permutation");
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

llvm::Optional<vector::CombiningKind>
mlir::linalg::getCombinerOpKind(Operation *combinerOp) {
  using ::mlir::vector::CombiningKind;

  if (!combinerOp)
    return llvm::None;
  return llvm::TypeSwitch<Operation *, llvm::Optional<CombiningKind>>(
             combinerOp)
      .Case<arith::AddIOp, arith::AddFOp>(
          [&](auto op) { return CombiningKind::ADD; })
      .Case<arith::AndIOp>([&](auto op) { return CombiningKind::AND; })
      .Case<arith::MaxSIOp>([&](auto op) { return CombiningKind::MAXSI; })
      .Case<arith::MaxFOp>([&](auto op) { return CombiningKind::MAXF; })
      .Case<arith::MinSIOp>([&](auto op) { return CombiningKind::MINSI; })
      .Case<arith::MinFOp>([&](auto op) { return CombiningKind::MINF; })
      .Case<arith::MulIOp, arith::MulFOp>(
          [&](auto op) { return CombiningKind::MUL; })
      .Case<arith::OrIOp>([&](auto op) { return CombiningKind::OR; })
      .Case<arith::XOrIOp>([&](auto op) { return CombiningKind::XOR; })
      .Default([&](auto op) { return llvm::None; });
}

/// Check whether `outputOperand` is a reduction with a single combiner
/// operation. Return the combiner operation of the reduction. Return
/// nullptr otherwise. Multiple reduction operations would impose an
/// ordering between reduction dimensions and is currently unsupported in
/// Linalg. This limitation is motivated by the fact that e.g. min(max(X)) !=
/// max(min(X))
// TODO: use in LinalgOp verification, there is a circular dependency atm.
static Operation *matchLinalgReduction(OpOperand *outputOperand) {
  auto linalgOp = cast<LinalgOp>(outputOperand->getOwner());
  unsigned outputPos =
      outputOperand->getOperandNumber() - linalgOp.getNumInputs();
  // Only single combiner operations are supported for now.
  SmallVector<Operation *, 4> combinerOps;
  if (!matchReduction(linalgOp.getRegionOutputArgs(), outputPos, combinerOps) ||
      combinerOps.size() != 1)
    return nullptr;

  // Return the combiner operation.
  return combinerOps[0];
}

/// Broadcast `value` to a vector of `shape` if possible. Return value
/// otherwise.
static Value broadcastIfNeeded(OpBuilder &b, Value value,
                               ArrayRef<int64_t> shape) {
  // If no shape to broadcast to, just return `value`.
  if (shape.empty())
    return value;
  VectorType targetVectorType =
      VectorType::get(shape, getElementTypeOrSelf(value));
  if (vector::isBroadcastableTo(value.getType(), targetVectorType) !=
      vector::BroadcastableToResult::Success)
    return value;
  Location loc = b.getInsertionPoint()->getLoc();
  return b.createOrFold<vector::BroadcastOp>(loc, targetVectorType, value);
}

/// Create MultiDimReductionOp to compute the reduction for `reductionOp`. This
/// assumes that `reductionOp` has two operands and one of them is the reduction
/// initial value.
static Value buildMultiDimReduce(OpBuilder &b, Operation *reduceOp,
                                 Value valueToReduce,
                                 const SmallVector<bool> &reductionMask) {
  auto maybeKind = getCombinerOpKind(reduceOp);
  assert(maybeKind && "Failed precondition: could not get reduction kind");
  return b.create<vector::MultiDimReductionOp>(
      reduceOp->getLoc(), valueToReduce, reductionMask, *maybeKind);
}

static SmallVector<bool> getReductionMask(LinalgOp linalgOp) {
  unsigned idx = 0;
  SmallVector<bool> reductionMask(linalgOp.iterator_types().size(), false);
  for (auto attr : linalgOp.iterator_types()) {
    if (isReductionIterator(attr))
      reductionMask[idx] = true;
    ++idx;
  }
  return reductionMask;
}

/// Build a vector.transfer_write of `value` into `outputOperand` at indices set
/// to all `0`; where `outputOperand` is an output operand of the LinalgOp
/// currently being vectorized. If `dest` has null rank, build an memref.store.
/// Return the produced value or null if no value is produced.
static Value buildVectorWrite(OpBuilder &b, Value value,
                              OpOperand *outputOperand) {
  Operation *write;
  Location loc = value.getLoc();
  auto linalgOp = cast<LinalgOp>(outputOperand->getOwner());
  ArrayRef<int64_t> shape = linalgOp.getShape(outputOperand);
  auto vectorType = VectorType::get(
      shape, getElementTypeOrSelf(outputOperand->get().getType()));
  if (vectorType.getRank() > 0) {
    // 0-d case is still special: do not invert the reindexing map.
    AffineMap map =
        reindexIndexingMap(linalgOp.getTiedIndexingMap(outputOperand));
    SmallVector<int64_t> transposeShape =
        applyPermutationMap(inversePermutation(map), vectorType.getShape());
    assert(!transposeShape.empty() && "unexpected empty transpose shape");
    vectorType = VectorType::get(transposeShape, vectorType.getElementType());
    SmallVector<Value> indices(linalgOp.getRank(outputOperand),
                               b.create<arith::ConstantIndexOp>(loc, 0));
    value = broadcastIfNeeded(b, value, vectorType.getShape());
    write = b.create<vector::TransferWriteOp>(loc, value, outputOperand->get(),
                                              indices, map);
  } else {
    if (!value.getType().isa<VectorType>())
      value = b.create<vector::BroadcastOp>(loc, vectorType, value);
    assert(value.getType() == vectorType && "incorrect type");
    write = b.create<vector::TransferWriteOp>(loc, value, outputOperand->get(),
                                              ValueRange{});
  }
  LDBG("vectorized op: " << *write);
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
  for (const auto &outputs : llvm::enumerate(yieldOp.values())) {
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
  SmallVector<int64_t> constantSeq =
      llvm::to_vector<16>(llvm::seq<int64_t>(0, targetShape[indexOp.dim()]));
  auto constantOp =
      b.create<arith::ConstantOp>(loc, b.getIndexVectorAttr(constantSeq));
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
  SmallVector<int64_t> transposition =
      llvm::to_vector<16>(llvm::seq<int64_t>(0, linalgOp.getNumLoops()));
  std::swap(transposition.back(), transposition[indexOp.dim()]);
  auto transposeOp =
      b.create<vector::TransposeOp>(loc, broadCastOp, transposition);
  return VectorizationResult{VectorizationStatus::NewOp, transposeOp};
}

/// Create a new vectorized verstion of `op` with the given operands and types.
static Operation *createVectorizedOp(OpBuilder &b, Operation *op,
                                     ValueRange newOperands,
                                     ArrayRef<Type> types) {
  OperationState state(op->getLoc(), op->getName());
  state.addAttributes(op->getAttrs());
  state.addOperands(newOperands);
  state.addTypes(types);
  return b.createOperation(state);
}

/// Emit reduction operations if the shapes of the value to reduce is different
/// that the result shape.
static Operation *reduceIfNeeded(OpBuilder &b, LinalgOp linalgOp, Operation *op,
                                 Value reduceValue, Value initialValue,
                                 const BlockAndValueMapping &bvm) {
  Value reduceVec = bvm.lookup(reduceValue);
  Value outputVec = bvm.lookup(initialValue);
  auto reduceType = reduceVec.getType().dyn_cast<VectorType>();
  auto outputType = outputVec.getType().dyn_cast<VectorType>();
  // Reduce only if needed as the value may already have been reduce for
  // contraction vectorization.
  if (!reduceType ||
      (outputType && reduceType.getShape() == outputType.getShape()))
    return nullptr;
  SmallVector<bool> reductionMask = getReductionMask(linalgOp);
  Value reduce = buildMultiDimReduce(b, op, reduceVec, reductionMask);
  return createVectorizedOp(b, op, {reduce, outputVec}, reduce.getType());
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
vectorizeOneOp(OpBuilder &b, LinalgOp linalgOp, Operation *op,
               const BlockAndValueMapping &bvm,
               ArrayRef<CustomVectorizationHook> customVectorizationHooks) {
  LDBG("vectorize op " << *op);

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
  if (isa<arith::ConstantOp, ConstantOp>(op))
    return VectorizationResult{VectorizationStatus::NewOp, b.clone(*op)};

  // 3. Only ElementwiseMappable are allowed in the generic vectorization.
  if (!OpTrait::hasElementwiseMappableTraits(op))
    return VectorizationResult{VectorizationStatus::Failure, nullptr};

  // 4 . Check if the operation is a reduction.
  SmallVector<std::pair<Value, Value>> reductionOperands;
  for (Value operand : op->getOperands()) {
    auto arg = operand.dyn_cast<BlockArgument>();
    if (!arg || arg.getArgNumber() < linalgOp.getNumInputs())
      continue;
    SmallVector<Operation *> reductionOps;
    Value reduceValue = matchReduction(
        linalgOp.getRegionOutputArgs(),
        arg.getArgNumber() - linalgOp.getNumInputs(), reductionOps);
    if (!reduceValue)
      continue;
    reductionOperands.push_back(std::make_pair(reduceValue, operand));
  }
  if (!reductionOperands.empty()) {
    assert(reductionOperands.size() == 1);
    Operation *reduceOp =
        reduceIfNeeded(b, linalgOp, op, reductionOperands[0].first,
                       reductionOperands[0].second, bvm);
    if (reduceOp)
      return VectorizationResult{VectorizationStatus::NewOp, reduceOp};
  }

  // 5. Generic vectorization path for ElementwiseMappable ops.
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
  return VectorizationResult{
      VectorizationStatus::NewOp,
      createVectorizedOp(b, op, llvm::to_vector<4>(vectorizedOperands),
                         llvm::to_vector<4>(returnTypes))};
}

/// Detect whether `r` has only ConstantOp, ElementwiseMappable and YieldOp.
static bool hasOnlyScalarElementwiseOp(Region &r) {
  if (!llvm::hasSingleElement(r))
    return false;
  for (Operation &op : r.front()) {
    if (!(isa<arith::ConstantOp, ConstantOp, linalg::YieldOp, linalg::IndexOp>(
              op) ||
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
static LogicalResult
vectorizeAsLinalgGeneric(OpBuilder &b, LinalgOp linalgOp,
                         SmallVectorImpl<Value> &newResults) {
  Block *block = linalgOp.getBlock();

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
  Location loc = linalgOp.getLoc();
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  for (OpOperand *opOperand : linalgOp.getInputAndOutputOperands()) {
    BlockArgument bbarg = block->getArgument(opOperand->getOperandNumber());
    if (linalgOp.isScalar(opOperand)) {
      bvm.map(bbarg, opOperand->get());
      continue;
    }
    VectorType readType;
    AffineMap map;
    // TODO: can we keep this simplification?
    // if (linalgOp.getShape(opOperand).empty()) {
    //   readType = VectorType::get({}, bbarg.getType());
    // } else {
    if (opOperand->getOperandNumber() < linalgOp.getNumInputs()) {
      map = inverseAndBroadcastProjectedPermuation(
          linalgOp.getTiedIndexingMap(opOperand));
      readType = VectorType::get(commonVectorShape,
                                 getElementTypeOrSelf(opOperand->get()));
    } else {
      map = inversePermutation(
          reindexIndexingMap(linalgOp.getTiedIndexingMap(opOperand)));
      readType = VectorType::get(map.compose(linalgOp.getShape(opOperand)),
                                 getElementTypeOrSelf(opOperand->get()));
    }
    // }

    auto shape = linalgOp.getShape(opOperand);
    SmallVector<Value> indices(shape.size(), zero);
    Value readValue = b.create<vector::TransferReadOp>(
        loc, readType, opOperand->get(), indices, map);
    // Not all ops support 0-d vectors, extract the scalar for now.
    // TODO: remove this.
    if (readValue.getType().cast<VectorType>().getRank() == 0)
      readValue = b.create<vector::ExtractElementOp>(loc, readValue);

    LDBG("new vectorized bbarg(" << bbarg.getArgNumber() << "): " << readValue);
    bvm.map(bbarg, readValue);
    bvm.map(opOperand->get(), readValue);
  }

  SmallVector<CustomVectorizationHook> hooks;
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
  for (Operation &op : block->getOperations()) {
    VectorizationResult result = vectorizeOneOp(b, linalgOp, &op, bvm, hooks);
    if (result.status == VectorizationStatus::Failure) {
      LDBG("failed to vectorize: " << op);
      return failure();
    }
    if (result.status == VectorizationStatus::NewOp) {
      LDBG("new vector op: " << *result.newOp;);
      bvm.map(op.getResults(), result.newOp->getResults());
    }
  }

  return success();
}

/// Helper function to vectorize a `linalgOp` with contraction semantics in a
/// generic fashion.
/// This helper is needed atm because the truly generic implementation requires
/// good vector.multi_reduce folding patterns that are currently NYI.
// TODO: drop reliance on a specific pattern.
static bool allIndexingsAreProjectedPermutation(LinalgOp op) {
  return llvm::all_of(op.getIndexingMaps(), [](AffineMap m) {
    return m.isProjectedPermutation(/*allowZeroInResults=*/true);
  });
}

// TODO: probably need some extra checks for reduction followed by consumer
// ops that may not commute (e.g. linear reduction + non-linear instructions).
static LogicalResult reductionPreconditions(LinalgOp op) {
  if (llvm::none_of(op.iterator_types(), isReductionIterator)) {
    LDBG("reduction precondition failed: no reduction iterator");
    return failure();
  }
  for (OpOperand *opOperand : op.getOutputOperands()) {
    Operation *reduceOp = matchLinalgReduction(opOperand);
    if (!reduceOp || !getCombinerOpKind(reduceOp)) {
      LDBG("reduction precondition failed: reduction detection failed");
      return failure();
    }
  }
  return success();
}

static LogicalResult vectorizeStaticLinalgOpPrecondition(linalg::LinalgOp op) {
  if (isElementwise(op))
    return success();
  // TODO: isaConvolutionOpInterface that can also infer from generic features.
  // But we will still need stride/dilation attributes that will be annoying to
  // reverse-engineer...
  if (isa<ConvolutionOpInterface>(op.getOperation()))
    return success();
  // TODO: the common vector shape is equal to the static loop sizes only when
  // all indexing maps are projected permutations. For convs and stencils the
  // logic will need to evolve.
  if (!allIndexingsAreProjectedPermutation(op)) {
    LDBG("precondition failed: not projected permutations");
    return failure();
  }
  if (failed(reductionPreconditions(op))) {
    LDBG("precondition failed: reduction preconditions");
    return failure();
  }
  return success();
}

static LogicalResult vectorizeLinalgOpPrecondition(LinalgOp linalgOp) {
  // All types must be static shape to go to vector.
  if (linalgOp.hasDynamicShape()) {
    LDBG("precondition failed: dynamic shape");
    return failure();
  }
  return vectorizeStaticLinalgOpPrecondition(linalgOp);
}

LogicalResult mlir::linalg::vectorize(RewriterBase &rewriter,
                                      LinalgOp linalgOp) {
  if (failed(vectorizeLinalgOpPrecondition(linalgOp)))
    return failure();

  SmallVector<Value> results;
  // TODO: isaConvolutionOpInterface that can also infer from generic
  // features. Will require stride/dilation attributes inference.
  FailureOr<Operation *> convOr = vectorizeConvolution(rewriter, linalgOp);
  if (succeeded(convOr)) {
    llvm::append_range(results, (*convOr)->getResults());
  } else {
    if (failed(vectorizeLinalgOpPrecondition(linalgOp)))
      return failure();
    LDBG("Vectorize generic by broadcasting to a common shape: " << linalgOp);
    if (failed(vectorizeAsLinalgGeneric(rewriter, linalgOp, results)))
      return failure();
  }

  if (!results.empty())
    rewriter.replaceOp(linalgOp, results);
  else
    rewriter.eraseOp(linalgOp);

  return success();
}

LogicalResult mlir::linalg::vectorizeCopy(RewriterBase &rewriter,
                                          memref::CopyOp copyOp) {

  auto srcType = copyOp.source().getType().cast<MemRefType>();
  auto dstType = copyOp.target().getType().cast<MemRefType>();
  if (!srcType.hasStaticShape() || !dstType.hasStaticShape())
    return failure();

  auto readType =
      VectorType::get(srcType.getShape(), getElementTypeOrSelf(srcType));
  auto writeType =
      VectorType::get(dstType.getShape(), getElementTypeOrSelf(dstType));

  Location loc = copyOp->getLoc();
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> indices(srcType.getRank(), zero);

  Value readValue = rewriter.create<vector::TransferReadOp>(
      loc, readType, copyOp.source(), indices,
      rewriter.getMultiDimIdentityMap(srcType.getRank()));
  if (readValue.getType().cast<VectorType>().getRank() == 0) {
    readValue = rewriter.create<vector::ExtractElementOp>(loc, readValue);
    readValue = rewriter.create<vector::BroadcastOp>(loc, writeType, readValue);
  }
  Operation *writeValue = rewriter.create<vector::TransferWriteOp>(
      loc, readValue, copyOp.target(), indices,
      rewriter.getMultiDimIdentityMap(srcType.getRank()));
  copyOp->getParentOfType<FuncOp>().dump();
  rewriter.replaceOp(copyOp, writeValue->getResults());
  return success();
}

//----------------------------------------------------------------------------//
// Misc. vectorization patterns.
//----------------------------------------------------------------------------//

/// Helper function that retrieves the value of an IntegerAttr.
static int64_t getIntFromAttr(Attribute attr) {
  return attr.cast<IntegerAttr>().getInt();
}

/// Given an ArrayRef of OpFoldResults, return a vector of Values.
/// IntegerAttrs are converted to ConstantIndexOps. Other attribute types are
/// not supported.
static SmallVector<Value> ofrToIndexValues(OpBuilder &builder, Location loc,
                                           ArrayRef<OpFoldResult> ofrs) {
  SmallVector<Value> result;
  llvm::for_each(ofrs, [&](auto o) {
    if (auto val = o.template dyn_cast<Value>()) {
      result.push_back(val);
    } else {
      result.push_back(builder.create<arith::ConstantIndexOp>(
          loc, getIntFromAttr(o.template get<Attribute>())));
    }
  });
  return result;
}

/// Rewrite a tensor::PadOp into a sequence of InitTensorOp, FillOp and
/// InsertSliceOp. For now, only constant padding values are supported.
/// If there is enough static type information, TransferReadOps and
/// TransferWriteOps may be generated instead of InsertSliceOps.
struct GenericPadOpVectorizationPattern : public GeneralizePadOpPattern {
  GenericPadOpVectorizationPattern(MLIRContext *context,
                                   PatternBenefit benefit = 1)
      : GeneralizePadOpPattern(context, tryVectorizeCopy, benefit) {}
  /// Vectorize the copying of a tensor::PadOp's source. This is possible if
  /// each dimension size is statically know in the source type or the result
  /// type (or both).
  static LogicalResult tryVectorizeCopy(PatternRewriter &rewriter,
                                        tensor::PadOp padOp, Value dest) {
    auto sourceType = padOp.getSourceType();
    auto resultType = padOp.getResultType();

    // Copy cannot be vectorized if pad value is non-constant and source shape
    // is dynamic. In case of a dynamic source shape, padding must be appended
    // by TransferReadOp, but TransferReadOp supports only constant padding.
    auto padValue = padOp.getConstantPaddingValue();
    if (!padValue) {
      if (!sourceType.hasStaticShape())
        return failure();
      // Create dummy padding value.
      auto elemType = sourceType.getElementType();
      padValue = rewriter.create<arith::ConstantOp>(
          padOp.getLoc(), elemType, rewriter.getZeroAttr(elemType));
    }

    SmallVector<int64_t> vecShape;
    SmallVector<bool> readInBounds;
    SmallVector<bool> writeInBounds;
    for (unsigned i = 0; i < sourceType.getRank(); ++i) {
      if (!sourceType.isDynamicDim(i)) {
        vecShape.push_back(sourceType.getDimSize(i));
        // Source shape is statically known: Neither read nor write are
        // out-of- bounds.
        readInBounds.push_back(true);
        writeInBounds.push_back(true);
      } else if (!resultType.isDynamicDim(i)) {
        // Source shape is not statically known, but result shape is.
        // Vectorize with size of result shape. This may be larger than the
        // source size.
        vecShape.push_back(resultType.getDimSize(i));
        // Read may be out-of-bounds because the result size could be larger
        // than the source size.
        readInBounds.push_back(false);
        // Write is out-of-bounds if low padding > 0.
        writeInBounds.push_back(
            getConstantIntValue(padOp.getMixedLowPad()[i]) ==
            static_cast<int64_t>(0));
      } else {
        // Neither source nor result dim of padOp is static. Cannot vectorize
        // the copy.
        return failure();
      }
    }
    auto vecType = VectorType::get(vecShape, sourceType.getElementType());

    // Generate TransferReadOp.
    SmallVector<Value> readIndices(
        vecType.getRank(),
        rewriter.create<arith::ConstantIndexOp>(padOp.getLoc(), 0));
    auto read = rewriter.create<vector::TransferReadOp>(
        padOp.getLoc(), vecType, padOp.source(), readIndices, padValue,
        ArrayRef<bool>{readInBounds});

    // If `dest` is a FillOp and the TransferWriteOp would overwrite the
    // entire tensor, write directly to the FillOp's operand.
    if (llvm::equal(vecShape, resultType.getShape()) &&
        llvm::all_of(writeInBounds, [](bool b) { return b; }))
      if (auto fill = dest.getDefiningOp<FillOp>())
        dest = fill.output();

    // Generate TransferWriteOp.
    auto writeIndices =
        ofrToIndexValues(rewriter, padOp.getLoc(), padOp.getMixedLowPad());
    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        padOp, read, dest, writeIndices, ArrayRef<bool>{writeInBounds});

    return success();
  }
};

/// Base pattern for rewriting tensor::PadOps whose result is consumed by a
/// given operation type OpTy.
template <typename OpTy>
struct VectorizePadOpUserPattern : public OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern<tensor::PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const final {
    bool changed = false;
    // Insert users in vector, because some users may be replaced/removed.
    for (auto *user : llvm::to_vector<4>(padOp->getUsers()))
      if (auto op = dyn_cast<OpTy>(user))
        changed |= rewriteUser(rewriter, padOp, op).succeeded();
    return success(changed);
  }

protected:
  virtual LogicalResult rewriteUser(PatternRewriter &rewriter,
                                    tensor::PadOp padOp, OpTy op) const = 0;
};

/// Rewrite use of tensor::PadOp result in TransferReadOp. E.g.:
/// ```
/// %0 = linalg.pad_tensor %src ... : tensor<?x?xf32> to tensor<17x5xf32>
/// %r = vector.transfer_read %0[%c0, %c0], %cst
///     {in_bounds = [true, true]} : tensor<17x5xf32>, vector<17x5xf32>
/// ```
/// is rewritten to:
/// ```
/// %r = vector.transfer_read %src[%c0, %c0], %padding
///     {in_bounds = [true, true]}
///     : tensor<?x?xf32>, vector<17x5xf32>
/// ```
/// Note: By restricting this pattern to in-bounds TransferReadOps, we can be
/// sure that the original padding value %cst was never used.
///
/// This rewrite is possible if:
/// - `xferOp` has no out-of-bounds dims or mask.
/// - Low padding is static 0.
/// - Single, scalar padding value.
struct PadOpVectorizationWithTransferReadPattern
    : public VectorizePadOpUserPattern<vector::TransferReadOp> {
  using VectorizePadOpUserPattern<
      vector::TransferReadOp>::VectorizePadOpUserPattern;

  LogicalResult rewriteUser(PatternRewriter &rewriter, tensor::PadOp padOp,
                            vector::TransferReadOp xferOp) const override {
    // Low padding must be static 0.
    if (!padOp.hasZeroLowPad())
      return failure();
    // Pad value must be a constant.
    auto padValue = padOp.getConstantPaddingValue();
    if (!padValue)
      return failure();
    // Padding value of existing `xferOp` is unused.
    if (xferOp.hasOutOfBoundsDim() || xferOp.mask())
      return failure();

    rewriter.updateRootInPlace(xferOp, [&]() {
      SmallVector<bool> inBounds(xferOp.getVectorType().getRank(), false);
      xferOp->setAttr(xferOp.getInBoundsAttrName(),
                      rewriter.getBoolArrayAttr(inBounds));
      xferOp.sourceMutable().assign(padOp.source());
      xferOp.paddingMutable().assign(padValue);
    });

    return success();
  }
};

/// Rewrite use of tensor::PadOp result in TransferWriteOp.
/// This pattern rewrites TransferWriteOps that write to a padded tensor
/// value, where the same amount of padding is immediately removed again after
/// the write. In such cases, the TransferWriteOp can write to the non-padded
/// tensor value and apply out-of-bounds masking. E.g.:
/// ```
/// %0 = tensor.extract_slice ...[...] [%s0, %s1] [1, 1]
///     : tensor<...> to tensor<?x?xf32>
/// %1 = linalg.pad_tensor %0 ... : tensor<?x?xf32> to tensor<17x5xf32>
/// %2 = vector.transfer_write %vec, %1[...]
///     : vector<17x5xf32>, tensor<17x5xf32>
/// %r = tensor.extract_slice %2[0, 0] [%s0, %s1] [1, 1]
///     : tensor<17x5xf32> to tensor<?x?xf32>
/// ```
/// is rewritten to:
/// ```
/// %0 = tensor.extract_slice ...[...] [%s0, %s1] [1, 1]
///     : tensor<...> to tensor<?x?xf32>
/// %r = vector.transfer_write %vec, %0[...] : vector<17x5xf32>,
/// tensor<?x?xf32>
/// ```
/// Note: It is important that the ExtractSliceOp %r resizes the result of the
/// TransferWriteOp to the same size as the input of the TensorPadOp (or an
/// even smaller size). Otherwise, %r's new (dynamic) dimensions would differ
/// from %r's old dimensions.
///
/// This rewrite is possible if:
/// - Low padding is static 0.
/// - `xferOp` has exactly one use, which is an ExtractSliceOp. This
///   ExtractSliceOp trims the same amount of padding that was added
///   beforehand.
/// - Single, scalar padding value.
struct PadOpVectorizationWithTransferWritePattern
    : public VectorizePadOpUserPattern<vector::TransferWriteOp> {
  using VectorizePadOpUserPattern<
      vector::TransferWriteOp>::VectorizePadOpUserPattern;

  LogicalResult rewriteUser(PatternRewriter &rewriter, tensor::PadOp padOp,
                            vector::TransferWriteOp xferOp) const override {
    // TODO: support 0-d corner case.
    if (xferOp.getTransferRank() == 0)
      return failure();

    // Low padding must be static 0.
    if (!padOp.hasZeroLowPad())
      return failure();
    // Pad value must be a constant.
    auto padValue = padOp.getConstantPaddingValue();
    if (!padValue)
      return failure();
    // TransferWriteOp result must be directly consumed by an ExtractSliceOp.
    if (!xferOp->hasOneUse())
      return failure();
    auto trimPadding = dyn_cast<tensor::ExtractSliceOp>(*xferOp->user_begin());
    if (!trimPadding)
      return failure();
    // Only static zero offsets supported when trimming padding.
    if (!trimPadding.hasZeroOffset())
      return failure();
    // trimPadding must remove the amount of padding that was added earlier.
    if (!hasSameTensorSize(padOp.source(), trimPadding))
      return failure();

    // Insert the new TransferWriteOp at position of the old TransferWriteOp.
    rewriter.setInsertionPoint(xferOp);

    SmallVector<bool> inBounds(xferOp.getVectorType().getRank(), false);
    auto newXferOp = rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        xferOp, padOp.source().getType(), xferOp.vector(), padOp.source(),
        xferOp.indices(), xferOp.permutation_mapAttr(), xferOp.mask(),
        rewriter.getBoolArrayAttr(inBounds));
    rewriter.replaceOp(trimPadding, newXferOp->getResult(0));

    return success();
  }

  /// Check if `beforePadding` and `afterTrimming` have the same tensor size,
  /// i.e., same dimensions.
  ///
  /// Dimensions may be static, dynamic or mix of both. In case of dynamic
  /// dimensions, this function tries to infer the (static) tensor size by
  /// looking at the defining op and utilizing op-specific knowledge.
  ///
  /// This is a conservative analysis. In case equal tensor sizes cannot be
  /// proven statically, this analysis returns `false` even though the tensor
  /// sizes may turn out to be equal at runtime.
  bool hasSameTensorSize(Value beforePadding,
                         tensor::ExtractSliceOp afterTrimming) const {
    // If the input to tensor::PadOp is a CastOp, try with with both CastOp
    // result and CastOp operand.
    if (auto castOp = beforePadding.getDefiningOp<tensor::CastOp>())
      if (hasSameTensorSize(castOp.source(), afterTrimming))
        return true;

    auto t1 = beforePadding.getType().dyn_cast<RankedTensorType>();
    auto t2 = afterTrimming.getType().dyn_cast<RankedTensorType>();
    // Only RankedTensorType supported.
    if (!t1 || !t2)
      return false;
    // Rank of both values must be the same.
    if (t1.getRank() != t2.getRank())
      return false;

    // All static dimensions must be the same. Mixed cases (e.g., dimension
    // static in `t1` but dynamic in `t2`) are not supported.
    for (unsigned i = 0; i < t1.getRank(); ++i) {
      if (t1.isDynamicDim(i) != t2.isDynamicDim(i))
        return false;
      if (!t1.isDynamicDim(i) && t1.getDimSize(i) != t2.getDimSize(i))
        return false;
    }

    // Nothing more to check if all dimensions are static.
    if (t1.getNumDynamicDims() == 0)
      return true;

    // All dynamic sizes must be the same. The only supported case at the
    // moment is when `beforePadding` is an ExtractSliceOp (or a cast
    // thereof).

    // Apart from CastOp, only ExtractSliceOp is supported.
    auto beforeSlice = beforePadding.getDefiningOp<tensor::ExtractSliceOp>();
    if (!beforeSlice)
      return false;

    assert(static_cast<size_t>(t1.getRank()) ==
           beforeSlice.getMixedSizes().size());
    assert(static_cast<size_t>(t2.getRank()) ==
           afterTrimming.getMixedSizes().size());

    for (unsigned i = 0; i < t1.getRank(); ++i) {
      // Skip static dimensions.
      if (!t1.isDynamicDim(i))
        continue;
      auto size1 = beforeSlice.getMixedSizes()[i];
      auto size2 = afterTrimming.getMixedSizes()[i];

      // Case 1: Same value or same constant int.
      if (isEqualConstantIntOrValue(size1, size2))
        continue;

      // Other cases: Take a deeper look at defining ops of values.
      auto v1 = size1.dyn_cast<Value>();
      auto v2 = size2.dyn_cast<Value>();
      if (!v1 || !v2)
        return false;

      // Case 2: Both values are identical AffineMinOps. (Should not happen if
      // CSE is run.)
      auto minOp1 = v1.getDefiningOp<AffineMinOp>();
      auto minOp2 = v2.getDefiningOp<AffineMinOp>();
      if (minOp1 && minOp2 && minOp1.getAffineMap() == minOp2.getAffineMap() &&
          minOp1.operands() == minOp2.operands())
        continue;

      // Add additional cases as needed.
    }

    // All tests passed.
    return true;
  }
};

/// Rewrite use of tensor::PadOp result in InsertSliceOp. E.g.:
/// ```
/// %0 = linalg.pad_tensor %src ... : tensor<?x?xf32> to tensor<17x5xf32>
/// %r = tensor.insert_slice %0
///     into %dest[%a, %b, 0, 0] [1, 1, 17, 5] [1, 1, 1, 1]
///     : tensor<17x5xf32> into tensor<?x?x17x5xf32>
/// ```
/// is rewritten to:
/// ```
/// %0 = vector.transfer_read %src[%c0, %c0], %padding
///     : tensor<?x?xf32>, vector<17x5xf32>
/// %r = vector.transfer_write %0, %dest[%a, %b, %c0, %c0]
///     {in_bounds = [true, true]} : vector<17x5xf32>, tensor<?x?x17x5xf32>
/// ```
///
/// This rewrite is possible if:
/// - Low padding is static 0.
/// - `padOp` result shape is static.
/// - The entire padded tensor is inserted.
///   (Implies that sizes of `insertOp` are all static.)
/// - Only unit strides in `insertOp`.
/// - Single, scalar padding value.
/// - `padOp` result not used as destination.
struct PadOpVectorizationWithInsertSlicePattern
    : public VectorizePadOpUserPattern<tensor::InsertSliceOp> {
  using VectorizePadOpUserPattern<
      tensor::InsertSliceOp>::VectorizePadOpUserPattern;

  LogicalResult rewriteUser(PatternRewriter &rewriter, tensor::PadOp padOp,
                            tensor::InsertSliceOp insertOp) const override {
    // Low padding must be static 0.
    if (!padOp.hasZeroLowPad())
      return failure();
    // Only unit stride supported.
    if (!insertOp.hasUnitStride())
      return failure();
    // Pad value must be a constant.
    auto padValue = padOp.getConstantPaddingValue();
    if (!padValue)
      return failure();
    // Dynamic shapes not supported.
    if (!padOp.result().getType().cast<ShapedType>().hasStaticShape())
      return failure();
    // Pad result not used as destination.
    if (insertOp.dest() == padOp.result())
      return failure();

    auto vecType = VectorType::get(padOp.getType().getShape(),
                                   padOp.getType().getElementType());
    unsigned vecRank = vecType.getRank();
    unsigned tensorRank = insertOp.getType().getRank();

    // Check if sizes match: Insert the entire tensor into most minor dims.
    // (No permutations allowed.)
    SmallVector<int64_t> expectedSizes(tensorRank - vecRank, 1);
    expectedSizes.append(vecType.getShape().begin(), vecType.getShape().end());
    if (!llvm::all_of(
            llvm::zip(insertOp.getMixedSizes(), expectedSizes), [](auto it) {
              return getConstantIntValue(std::get<0>(it)) == std::get<1>(it);
            }))
      return failure();

    // Insert the TransferReadOp and TransferWriteOp at the position of the
    // InsertSliceOp.
    rewriter.setInsertionPoint(insertOp);

    // Generate TransferReadOp: Read entire source tensor and add high
    // padding.
    SmallVector<Value> readIndices(
        vecRank, rewriter.create<arith::ConstantIndexOp>(padOp.getLoc(), 0));
    auto read = rewriter.create<vector::TransferReadOp>(
        padOp.getLoc(), vecType, padOp.source(), readIndices, padValue);

    // Generate TransferWriteOp: Write to InsertSliceOp's dest tensor at
    // specified offsets. Write is fully in-bounds because a InsertSliceOp's
    // source must fit into the destination at the specified offsets.
    auto writeIndices =
        ofrToIndexValues(rewriter, padOp.getLoc(), insertOp.getMixedOffsets());
    SmallVector<bool> inBounds(vecRank, true);
    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        insertOp, read, insertOp.dest(), writeIndices,
        ArrayRef<bool>{inBounds});

    return success();
  }
};

void mlir::linalg::populatePadOpVectorizationPatterns(
    RewritePatternSet &patterns, PatternBenefit baseBenefit) {
  patterns.add<GenericPadOpVectorizationPattern>(patterns.getContext(),
                                                 baseBenefit);
  // Try these specialized patterns first before resorting to the generic one.
  patterns.add<PadOpVectorizationWithTransferReadPattern,
               PadOpVectorizationWithTransferWritePattern,
               PadOpVectorizationWithInsertSlicePattern>(
      patterns.getContext(), baseBenefit.getBenefit() + 1);
}

//----------------------------------------------------------------------------//
// Forwarding patterns
//----------------------------------------------------------------------------//

/// Check whether there is any interleaved use of any `values` between
/// `firstOp` and `secondOp`. Conservatively return `true` if any op or value
/// is in a different block.
static bool mayExistInterleavedUses(Operation *firstOp, Operation *secondOp,
                                    ValueRange values) {
  if (firstOp->getBlock() != secondOp->getBlock() ||
      !firstOp->isBeforeInBlock(secondOp)) {
    LDBG("interleavedUses precondition failed, firstOp: "
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
      LDBG(" found interleaved op " << *owner << ", firstOp: " << *firstOp
                                    << ", second op: " << *secondOp);
      return true;
    }
  }
  return false;
}

/// Return the unique subview use of `v` if it is indeed unique, null
/// otherwise.
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

  // TODO: support mask.
  if (xferOp.mask())
    return failure();

  // Transfer into `view`.
  Value viewOrAlloc = xferOp.source();
  if (!viewOrAlloc.getDefiningOp<memref::ViewOp>() &&
      !viewOrAlloc.getDefiningOp<memref::AllocOp>())
    return failure();

  LDBG(viewOrAlloc);

  // Ensure there is exactly one subview of `viewOrAlloc` defining `subView`.
  memref::SubViewOp subViewOp = getSubViewUseIfUnique(viewOrAlloc);
  if (!subViewOp)
    return failure();
  Value subView = subViewOp.getResult();
  LDBG("with subView " << subView);

  // Find the copy into `subView` without interleaved uses.
  memref::CopyOp copyOp;
  for (auto &u : subView.getUses()) {
    if (auto newCopyOp = dyn_cast<memref::CopyOp>(u.getOwner())) {
      assert(newCopyOp.target().getType().isa<MemRefType>());
      if (newCopyOp.target() != subView)
        continue;
      LDBG("copy candidate " << *newCopyOp);
      if (mayExistInterleavedUses(newCopyOp, xferOp, {viewOrAlloc, subView}))
        continue;
      copyOp = newCopyOp;
      break;
    }
  }
  if (!copyOp)
    return failure();
  LDBG("with copy " << *copyOp);

  // Find the fill into `viewOrAlloc` without interleaved uses before the
  // copy.
  FillOp maybeFillOp;
  for (auto &u : viewOrAlloc.getUses()) {
    if (auto newFillOp = dyn_cast<FillOp>(u.getOwner())) {
      assert(newFillOp.output().getType().isa<MemRefType>());
      if (newFillOp.output() != viewOrAlloc)
        continue;
      LDBG("fill candidate " << *newFillOp);
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
    LDBG("with maybeFillOp " << *maybeFillOp);

  // `in` is the subview that memref.copy reads. Replace it.
  Value in = copyOp.source();

  // memref.copy + linalg.fill can be used to create a padded local buffer.
  // The `masked` attribute is only valid on this padded buffer.
  // When forwarding to vector.transfer_read, the attribute must be reset
  // conservatively.
  Value res = rewriter.create<vector::TransferReadOp>(
      xferOp.getLoc(), xferOp.getVectorType(), in, xferOp.indices(),
      xferOp.permutation_mapAttr(), xferOp.padding(), xferOp.mask(),
      // in_bounds is explicitly reset
      /*inBoundsAttr=*/ArrayAttr());

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
  // TODO: support mask.
  if (xferOp.mask())
    return failure();

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
  memref::CopyOp copyOp;
  for (auto &u : subViewOp.getResult().getUses()) {
    if (auto newCopyOp = dyn_cast<memref::CopyOp>(u.getOwner())) {
      if (newCopyOp.source() != subView)
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
  assert(copyOp.target().getType().isa<MemRefType>());
  Value out = copyOp.target();

  // Forward vector.transfer into copy.
  // memref.copy + linalg.fill can be used to create a padded local buffer.
  // The `masked` attribute is only valid on this padded buffer.
  // When forwarding to vector.transfer_write, the attribute must be reset
  // conservatively.
  rewriter.create<vector::TransferWriteOp>(
      xferOp.getLoc(), xferOp.vector(), out, xferOp.indices(),
      xferOp.permutation_mapAttr(), xferOp.mask(),
      // in_bounds is explicitly reset
      /*inBoundsAttr=*/ArrayAttr());

  rewriter.eraseOp(copyOp);
  rewriter.eraseOp(xferOp);

  return success();
}

//===----------------------------------------------------------------------===//
// Convolution vectorization patterns
//===----------------------------------------------------------------------===//

template <int N>
static void bindShapeDims(ShapedType shapedType) {}

template <int N, typename IntTy, typename... IntTy2>
static void bindShapeDims(ShapedType shapedType, IntTy &val, IntTy2 &...vals) {
  val = shapedType.getShape()[N];
  bindShapeDims<N + 1, IntTy2 &...>(shapedType, vals...);
}

/// Bind a pack of int& to the leading dimensions of shapedType.getShape().
template <typename... IntTy>
static void bindShapeDims(ShapedType shapedType, IntTy &...vals) {
  bindShapeDims<0>(shapedType, vals...);
}

namespace {
/// Generate a vector implementation for either:
/// ```
///   Op def: (     n,     w,     c,    kw,    f  )
///    Iters: ({Par(), Par(), Par(), Red(), Red()})
///   Layout: {{n, strideW * w + dilationW * kw, c}, {kw, c, f}, {n, w, f}}
/// ```
/// kw is unrolled, w is unrolled iff dilationW > 1.
///
/// or
///
/// ```
///   Op def: (     n,     w,     c,    kw )
///    Iters: ({Par(), Par(), Par(), Red()})
///   Layout: {{n, strideW * w + dilationW * kw, c}, {kw, c}, {n, w, c}}
/// ```
/// kw is unrolled, w is unrolled iff dilationW > 1.
struct Conv1DNwcGenerator : public StructuredGenerator<LinalgOp> {
  Conv1DNwcGenerator(OpBuilder &builder, LinalgOp linalgOp, int strideW,
                     int dilationW)
      : StructuredGenerator<LinalgOp>(builder, linalgOp), valid(false),
        strideW(strideW), dilationW(dilationW) {
    // Determine whether `linalgOp` can be generated with this generator
    if (linalgOp.getNumInputs() != 2 || linalgOp.getNumOutputs() != 1)
      return;
    lhsShaped = linalgOp.inputs()[0];
    rhsShaped = linalgOp.inputs()[1];
    resShaped = linalgOp.outputs()[0];
    lhsShapedType = lhsShaped.getType().dyn_cast<ShapedType>();
    rhsShapedType = rhsShaped.getType().dyn_cast<ShapedType>();
    resShapedType = resShaped.getType().dyn_cast<ShapedType>();
    if (!lhsShapedType || !rhsShapedType || !resShapedType)
      return;
    if (lhsShapedType.getRank() != 3 ||
        (rhsShapedType.getRank() != 2 && rhsShapedType.getRank() != 3) ||
        resShapedType.getRank() != 3)
      return;

    // Check for reduction `add` preceded by `mul`.
    Operation *reduceOp = matchLinalgReduction(linalgOp.getOutputOperand(0));
    if (!reduceOp)
      return;
    llvm::Optional<vector::CombiningKind> maybeKind;
    maybeKind = getCombinerOpKind(reduceOp);
    if (!maybeKind || *maybeKind != vector::CombiningKind::ADD)
      return;
    maybeKind = getCombinerOpKind(&(linalgOp->getRegion(0).front().front()));
    if (!maybeKind || *maybeKind != vector::CombiningKind::MUL)
      return;

    // The op is now known to be valid.
    valid = true;
  }

  /// Generate a vector implementation for:
  /// ```
  ///   Op def: (     n,     w,     c,    kw,    f  )
  ///    Iters: ({Par(), Par(), Par(), Red(), Red()})
  ///   Layout: {{n, strideW * w + dilationW * kw, c}, {kw, c, f}, {n, w, f}}
  /// ```
  /// kw is always unrolled.
  /// TODO: w (resp. kw) is unrolled when the strideW ( resp. dilationW) is
  /// > 1.
  FailureOr<Operation *> conv() {
    if (!valid)
      return failure();

    int64_t nSize, wSize, cSize, kwSize, fSize;
    // kernel{kw, c, f}
    bindShapeDims(rhsShapedType, kwSize, cSize, fSize);
    // out{n, w, f}
    bindShapeDims(resShapedType, nSize, wSize);

    vector::TransferWriteOp write;
    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);

    // w is unrolled (i.e. wSizeStep == 1) iff strideW > 1.
    // When strideW == 1, we can batch the contiguous loads and avoid
    // unrolling
    int64_t wSizeStep = strideW == 1 ? wSize : 1;

    Type lhsEltType = lhsShapedType.getElementType();
    Type rhsEltType = rhsShapedType.getElementType();
    Type resEltType = resShapedType.getElementType();
    VectorType lhsType = VectorType::get(
        {nSize,
         // iw = ow * sw + kw *  dw - 1
         //   (i.e. 16 convolved with 3 (@stride 1 dilation 1) -> 14)
         // Perform the proper inclusive -> exclusive -> inclusive.
         ((wSize - 1) * strideW + 1) + ((kwSize - 1) * dilationW + 1) - 1,
         cSize},
        lhsEltType);
    VectorType rhsType = VectorType::get({kwSize, cSize, fSize}, rhsEltType);
    VectorType resType = VectorType::get({nSize, wSize, fSize}, resEltType);

    // Read lhs slice of size {w * strideW + kw * dilationW, c, f} @ [0, 0,
    // 0].
    Value lhs = builder.create<vector::TransferReadOp>(
        loc, lhsType, lhsShaped, ValueRange{zero, zero, zero});
    // Read rhs slice of size {kw, c, f} @ [0, 0, 0].
    Value rhs = builder.create<vector::TransferReadOp>(
        loc, rhsType, rhsShaped, ValueRange{zero, zero, zero});
    // Read res slice of size {n, w, f} @ [0, 0, 0].
    Value res = builder.create<vector::TransferReadOp>(
        loc, resType, resShaped, ValueRange{zero, zero, zero});

    //===------------------------------------------------------------------===//
    // Begin vector-only rewrite part
    //===------------------------------------------------------------------===//
    // Unroll along kw and read slices of lhs and rhs.
    SmallVector<Value> lhsVals, rhsVals, resVals;
    // Extract lhs slice of size {n, wSizeStep, c} @ [0, sw * w + dw * kw, 0].
    for (int64_t kw = 0; kw < kwSize; ++kw) {
      for (int64_t w = 0; w < wSize; w += wSizeStep) {
        lhsVals.push_back(builder.create<vector::ExtractStridedSliceOp>(
            loc, lhs,
            /*offsets=*/ArrayRef<int64_t>{0, w * strideW + kw * dilationW, 0},
            /*sizes=*/ArrayRef<int64_t>{nSize, wSizeStep, cSize},
            /*strides=*/ArrayRef<int64_t>{1, 1, 1}));
      }
    }
    // Extract rhs slice of size {c, f} @ [kw].
    for (int64_t kw = 0; kw < kwSize; ++kw) {
      rhsVals.push_back(builder.create<vector::ExtractOp>(
          loc, rhs, /*offsets=*/ArrayRef<int64_t>{kw}));
    }
    // Extract res slice: {n, wSizeStep, f} @ [0, w, 0].
    for (int64_t w = 0; w < wSize; w += wSizeStep) {
      resVals.push_back(builder.create<vector::ExtractStridedSliceOp>(
          loc, res,
          /*offsets=*/ArrayRef<int64_t>{0, w, 0},
          /*sizes=*/ArrayRef<int64_t>{nSize, wSizeStep, fSize},
          /*strides=*/ArrayRef<int64_t>{1, 1, 1}));
    }

    auto linearIndex = [&](int64_t kw, int64_t w) {
      return kw * (wSize / wSizeStep) + w;
    };

    // Compute contraction: O{n, w, f} += I{n, sw * w + dw * kw, c} * F{c, f}
    for (int64_t kw = 0; kw < kwSize; ++kw) {
      for (int64_t w = 0; w < wSize; w += wSizeStep) {
        resVals[w] = conv1dSliceAsContraction(
            builder, loc, lhsVals[linearIndex(kw, w)], rhsVals[kw], resVals[w]);
      }
    }

    // Write back res slice: {n, wSizeStep, f} @ [0, w, 0].
    // This does not depend on kw.
    for (int64_t w = 0; w < wSize; w += wSizeStep) {
      res = builder.create<vector::InsertStridedSliceOp>(
          loc, resVals[w], res,
          /*offsets=*/ArrayRef<int64_t>{0, w, 0},
          /*strides=*/ArrayRef<int64_t>{1, 1, 1});
    }
    //===------------------------------------------------------------------===//
    // End vector-only rewrite part
    //===------------------------------------------------------------------===//

    // Write back res slice of size {n, w, f} @ [0, 0, 0].
    return builder
        .create<vector::TransferWriteOp>(loc, res, resShaped,
                                         ValueRange{zero, zero, zero})
        .getOperation();
  }

  // Create a contraction: lhs{n, w, c} * rhs{c, f} -> res{n, w, f}
  Value conv1dSliceAsContraction(OpBuilder &b, Location loc, Value lhs,
                                 Value rhs, Value res) {
    StringRef par = Par().strRef, red = Red().strRef;
    AffineExpr n, w, f, c;
    bindDims(ctx, n, w, f, c);
    return builder.create<vector::ContractionOp>(
        loc, lhs, rhs, res,
        /*indexingMaps=*/MapList{{n, w, c}, {c, f}, {n, w, f}},
        /*iteratorTypes=*/ArrayRef<StringRef>{par, par, par, red});
  }

  /// Generate a vector implementation for:
  /// ```
  ///   Op def: (     n,     w,     c,    kw)
  ///    Iters: ({Par(), Par(), Par(), Red()})
  ///   Layout: {{n, strideW * w + dilationW * kw, c}, {kw, c}, {n, w, c}}
  /// ```
  /// kw is always unrolled.
  /// TODO: w (resp. kw) is unrolled when the strideW ( resp. dilationW) is
  /// > 1.
  FailureOr<Operation *> depthwiseConv() {
    if (!valid)
      return failure();

    int64_t nSize, wSize, cSize, kwSize;
    // kernel{kw, c}
    bindShapeDims(rhsShapedType, kwSize, cSize);
    // out{n, w, c}
    bindShapeDims(resShapedType, nSize, wSize);

    vector::TransferWriteOp write;
    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);

    // w is unrolled (i.e. wSizeStep == 1) iff strideW > 1.
    // When strideW == 1, we can batch the contiguous loads and avoid
    // unrolling
    int64_t wSizeStep = strideW == 1 ? wSize : 1;

    Type lhsEltType = lhsShapedType.getElementType();
    Type rhsEltType = rhsShapedType.getElementType();
    Type resEltType = resShapedType.getElementType();
    VectorType lhsType = VectorType::get(
        {nSize,
         // iw = ow * sw + kw *  dw - 1
         //   (i.e. 16 convolved with 3 (@stride 1 dilation 1) -> 14)
         ((wSize - 1) * strideW + 1) + ((kwSize - 1) * dilationW + 1) - 1,
         cSize},
        lhsEltType);
    VectorType rhsType = VectorType::get({kwSize, cSize}, rhsEltType);
    VectorType resType = VectorType::get({nSize, wSize, cSize}, resEltType);

    // Read lhs slice of size {n, w * strideW + kw * dilationW, c} @ [0, 0,
    // 0].
    Value lhs = builder.create<vector::TransferReadOp>(
        loc, lhsType, lhsShaped, ValueRange{zero, zero, zero});
    // Read rhs slice of size {kw, c} @ [0, 0].
    Value rhs = builder.create<vector::TransferReadOp>(loc, rhsType, rhsShaped,
                                                       ValueRange{zero, zero});
    // Read res slice of size {n, w, c} @ [0, 0, 0].
    Value res = builder.create<vector::TransferReadOp>(
        loc, resType, resShaped, ValueRange{zero, zero, zero});

    //===------------------------------------------------------------------===//
    // Begin vector-only rewrite part
    //===------------------------------------------------------------------===//
    // Unroll along kw and read slices of lhs and rhs.
    SmallVector<Value> lhsVals, rhsVals, resVals;
    // Extract lhs slice of size {n, wSizeStep, c}
    //   @ [0, sw * w + dw * kw, 0].
    for (int64_t kw = 0; kw < kwSize; ++kw) {
      for (int64_t w = 0; w < wSize; w += wSizeStep) {
        lhsVals.push_back(builder.create<vector::ExtractStridedSliceOp>(
            loc, lhs,
            /*offsets=*/ArrayRef<int64_t>{0, w * strideW + kw * dilationW, 0},
            /*sizes=*/ArrayRef<int64_t>{nSize, wSizeStep, cSize},
            /*strides=*/ArrayRef<int64_t>{1, 1, 1}));
      }
    }
    // Extract rhs slice of size {c} @ [kw].
    for (int64_t kw = 0; kw < kwSize; ++kw) {
      rhsVals.push_back(builder.create<vector::ExtractOp>(
          loc, rhs, /*offsets=*/ArrayRef<int64_t>{kw}));
    }
    // Extract res slice: {n, wSizeStep, c} @ [0, w, 0].
    for (int64_t w = 0; w < wSize; w += wSizeStep) {
      resVals.push_back(builder.create<vector::ExtractStridedSliceOp>(
          loc, res,
          /*offsets=*/ArrayRef<int64_t>{0, w, 0},
          /*sizes=*/ArrayRef<int64_t>{nSize, wSizeStep, cSize},
          /*strides=*/ArrayRef<int64_t>{1, 1, 1}));
    }

    auto linearIndex = [&](int64_t kw, int64_t w) {
      return kw * (wSize / wSizeStep) + w;
    };

    // Compute contraction: O{n, w, c} += I{n, sw * w + dw * kw, c} * F{c}
    for (int64_t kw = 0; kw < kwSize; ++kw) {
      for (int64_t w = 0; w < wSize; w += wSizeStep) {
        resVals[w] = depthwiseConv1dSliceAsFma(
            builder, loc, lhsVals[linearIndex(kw, w)], rhsVals[kw], resVals[w]);
      }
    }

    // Write back res slice: {n, wSizeStep, c} @ [0, w, 0].
    // This does not depend on kw.
    for (int64_t w = 0; w < wSize; w += wSizeStep) {
      res = builder.create<vector::InsertStridedSliceOp>(
          loc, resVals[w], res,
          /*offsets=*/ArrayRef<int64_t>{0, w, 0},
          /*strides=*/ArrayRef<int64_t>{1, 1, 1});
    }
    //===------------------------------------------------------------------===//
    // End vector-only rewrite part
    //===------------------------------------------------------------------===//

    // Write back res slice of size {n, w, c} @ [0, 0, 0].
    return builder
        .create<vector::TransferWriteOp>(loc, res, resShaped,
                                         ValueRange{zero, zero, zero})
        .getOperation();
  }

  /// Lower lhs{n, w, c} * rhs{c} -> res{n, w, c} to fma.
  Value depthwiseConv1dSliceAsFma(OpBuilder &b, Location loc, Value lhs,
                                  Value rhs, Value res) {
    Value bcast = builder.create<vector::BroadcastOp>(loc, res.getType(), rhs);
    return b.create<vector::FMAOp>(loc, lhs, bcast, res);
  }

  /// Entry point that transposes into the common form:
  ///   {{n, strideW * w + dilationW * kw, c}, {kw, c, f}, {n, w, f}}
  FailureOr<Operation *> generateConv() {
    AffineExpr n, w, f, kw, c;
    bindDims(ctx, n, w, f, kw, c);
    if (!iters({Par(), Par(), Par(), Red(), Red()}))
      return failure();

    // No transposition needed.
    if (layout({/*lhsIndex*/ {n, strideW * w + dilationW * kw, c},
                /*rhsIndex*/ {kw, c, f},
                /*resIndex*/ {n, w, f}}))
      return conv();
    return failure();
  }

  /// Entry point that transposes into the common form:
  ///   {{n, strideW * w + dilationW * kw, c}, {kw, c}, {n, w, c}}
  FailureOr<Operation *> generateDilatedConv() {
    AffineExpr n, w, c, kw;
    bindDims(ctx, n, w, c, kw);
    if (!iters({Par(), Par(), Par(), Red()}))
      return failure();

    // No transposition needed.
    if (layout({/*lhsIndex*/ {n, strideW * w + dilationW * kw, c},
                /*rhsIndex*/ {kw, c},
                /*resIndex*/ {n, w, c}}))
      return depthwiseConv();
    return failure();
  }

private:
  bool valid;
  int strideW, dilationW;
  Value lhsShaped, rhsShaped, resShaped;
  ShapedType lhsShapedType, rhsShapedType, resShapedType;
};
} // namespace

/// Helper function to vectorize a LinalgOp with convolution semantics.
// TODO: extend the generic vectorization to support windows and drop this.
static FailureOr<Operation *> vectorizeConvolution(OpBuilder &b, LinalgOp op) {
  // The ConvolutionOpInterface gives us guarantees of existence for
  // strides/dilations. However, we do not need to rely on those, we can simply
  // use them if present, otherwise use the default and let the generic conv.
  // matcher in the ConvGenerator succeed or fail.
  auto strides = op->getAttrOfType<DenseIntElementsAttr>("strides");
  auto dilations = op->getAttrOfType<DenseIntElementsAttr>("dilations");
  auto stride = strides ? *strides.getValues<uint64_t>().begin() : 1;
  auto dilation = dilations ? *dilations.getValues<uint64_t>().begin() : 1;
  Conv1DNwcGenerator e(b, op, stride, dilation);
  auto res = e.generateConv();
  if (succeeded(res))
    return res;
  return e.generateDilatedConv();
}

struct VectorizeConvolution : public OpInterfaceRewritePattern<LinalgOp> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(LinalgOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<Operation *> resultOrFail = vectorizeConvolution(rewriter, op);
    if (failed(resultOrFail))
      return failure();
    Operation *newOp = *resultOrFail;
    if (newOp->getNumResults() == 0) {
      rewriter.eraseOp(op.getOperation());
      return success();
    }
    assert(newOp->getNumResults() == 1 && "expected single result");
    rewriter.replaceOp(op.getOperation(), newOp->getResult(0));
    return success();
  }
};

void mlir::linalg::populateConvolutionVectorizationPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<VectorizeConvolution>(patterns.getContext(), benefit);
}
