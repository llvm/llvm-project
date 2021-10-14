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

#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/VectorOps.h"
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

using llvm::dbgs;

#define DEBUG_TYPE "linalg-vectorization"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X)

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
  assert(map.isProjectedPermutation(/*allowZerosInResults=*/true) &&
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

/// Return a vector type of the same shape and element type as the (assumed)
/// ShapedType of `v`.
static VectorType extractVectorTypeFromShapedValue(Value v) {
  auto st = v.getType().cast<ShapedType>();
  if (st.getShape().empty())
    return VectorType();
  return VectorType::get(st.getShape(), st.getElementType());
}

static llvm::Optional<vector::CombiningKind>
getKindForOp(Operation *reductionOp) {
  if (!reductionOp)
    return llvm::None;
  return llvm::TypeSwitch<Operation *, llvm::Optional<vector::CombiningKind>>(
             reductionOp)
      .Case<arith::AddIOp, arith::AddFOp>(
          [&](auto op) { return vector::CombiningKind::ADD; })
      .Case<arith::AndIOp>([&](auto op) { return vector::CombiningKind::AND; })
      .Case<MaxSIOp>([&](auto op) { return vector::CombiningKind::MAXSI; })
      .Case<MaxFOp>([&](auto op) { return vector::CombiningKind::MAXF; })
      .Case<MinSIOp>([&](auto op) { return vector::CombiningKind::MINSI; })
      .Case<MinFOp>([&](auto op) { return vector::CombiningKind::MINF; })
      .Case<arith::MulIOp, arith::MulFOp>(
          [&](auto op) { return vector::CombiningKind::MUL; })
      .Case<arith::OrIOp>([&](auto op) { return vector::CombiningKind::OR; })
      .Case<arith::XOrIOp>([&](auto op) { return vector::CombiningKind::XOR; })
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
  // Only single combiner operatios are supported for now.
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

/// Build a vector.transfer_read from `source` at indices set to all `0`.
/// If source has rank zero, build a `vector<1xt> transfer_read + extract`.
/// Return the produced value.
static Value buildVectorRead(OpBuilder &b, Value source, Type readType,
                             AffineMap map) {
  Location loc = source.getLoc();
  auto shapedType = source.getType().cast<ShapedType>();
  SmallVector<Value> indices(shapedType.getRank(),
                             b.create<arith::ConstantIndexOp>(loc, 0));
  if (auto vectorType = readType.dyn_cast<VectorType>())
    return b.create<vector::TransferReadOp>(loc, vectorType, source, indices,
                                            map);
  return vector::TransferReadOp::createScalarOp(b, loc, source, indices);
}

/// Create MultiDimReductionOp to compute the reduction for `reductionOp`. This
/// assumes that `reductionOp` has tow operands and one of them is the reduction
/// initial value.
static Value buildMultiDimReduce(OpBuilder &b, Operation *reduceOp,
                                 Value outputArg,
                                 const SmallVector<bool> &reductionMask,
                                 const BlockAndValueMapping &bvm) {
  auto maybeKind = getKindForOp(reduceOp);
  assert(maybeKind && "Failed precondition: could not get reduction kind");
  Value operandToReduce = reduceOp->getOperand(0) == outputArg
                              ? reduceOp->getOperand(1)
                              : reduceOp->getOperand(0);
  Value vec = bvm.lookup(operandToReduce);
  return b.create<vector::MultiDimReductionOp>(reduceOp->getLoc(), vec,
                                               reductionMask, *maybeKind);
}

/// Read the initial value associated to the given `outputOperand`.
static Value readInitialValue(OpBuilder &b, LinalgOp linalgOp,
                              OpOperand *outputOperand) {
  AffineMap map = inversePermutation(
      reindexIndexingMap(linalgOp.getTiedIndexingMap(outputOperand)));
  Type readType;
  if (linalgOp.getShape(outputOperand).empty()) {
    readType = getElementTypeOrSelf(outputOperand->get());
  } else {
    readType = VectorType::get(map.compose(linalgOp.getShape(outputOperand)),
                               getElementTypeOrSelf(outputOperand->get()));
  }
  Value vectorRead = buildVectorRead(b, outputOperand->get(), readType, map);
  return vectorRead;
}

/// Assuming `outputOperand` is an output operand of a LinalgOp, determine
/// whether a reduction is needed to produce a `targetType` and create that
/// reduction if it is the case.
static Value reduceIfNeeded(OpBuilder &b, Type targetType, Value value,
                            OpOperand *outputOperand,
                            const BlockAndValueMapping &bvm) {
  LDBG("Reduce " << value << " to type " << targetType);
  LDBG("In LinalgOp operand #" << outputOperand->getOperandNumber() << "\n"
                               << *(outputOperand->getOwner()));
  auto linalgOp = cast<LinalgOp>(outputOperand->getOwner());
  auto vecType = value.getType().dyn_cast<VectorType>();
  VectorType targetVectorType = targetType.dyn_cast<VectorType>();
  if (!vecType)
    return value;
  if (targetVectorType && vecType.getShape() == targetVectorType.getShape())
    return value;

  // At this point, we know we need to reduce. Detect the reduction operator.
  unsigned pos = 0;
  MLIRContext *ctx = b.getContext();
  SmallVector<AffineExpr> exprs;
  for (auto s : linalgOp.iterator_types())
    if (isParallelIterator(s))
      exprs.push_back(getAffineDimExpr(pos++, ctx));

  Operation *reduceOp = matchLinalgReduction(outputOperand);
  assert(reduceOp && "Failed precondition: could not math a reduction");
  unsigned idx = 0;
  SmallVector<bool> reductionMask(linalgOp.iterator_types().size(), false);
  for (auto attr : linalgOp.iterator_types()) {
    if (isReductionIterator(attr))
      reductionMask[idx] = true;
    ++idx;
  }
  assert(reduceOp->getNumOperands() == 2 &&
         "Only support binary reduce op right now");
  unsigned outputPos =
      outputOperand->getOperandNumber() - linalgOp.getNumInputs();
  Value outputArg = linalgOp.getRegionOutputArgs()[outputPos];
  // Reduce across the iteration space.
  Value reduce =
      buildMultiDimReduce(b, reduceOp, outputArg, reductionMask, bvm);

  // Read the original output value.
  Value initialValue = readInitialValue(b, linalgOp, outputOperand);

  // Combine the output argument with the reduced value.
  OperationState state(reduceOp->getLoc(), reduceOp->getName());
  state.addAttributes(reduceOp->getAttrs());
  state.addOperands({reduce, initialValue});
  state.addTypes(initialValue.getType());
  return b.createOperation(state)->getResult(0);
}

/// Build a vector.transfer_write of `value` into `outputOperand` at indices set
/// to all `0`; where `outputOperand` is an output operand of the LinalgOp
/// currently being vectorized. If `dest` has null rank, build an memref.store.
/// Return the produced value or null if no value is produced.
static Value buildVectorWrite(OpBuilder &b, Value value,
                              OpOperand *outputOperand,
                              const BlockAndValueMapping &bvm) {
  Operation *write;
  Location loc = value.getLoc();
  auto linalgOp = cast<LinalgOp>(outputOperand->getOwner());
  if (VectorType vectorType =
          extractVectorTypeFromShapedValue(outputOperand->get())) {
    AffineMap map =
        reindexIndexingMap(linalgOp.getTiedIndexingMap(outputOperand));
    SmallVector<int64_t> transposeShape =
        applyPermutationMap(inversePermutation(map), vectorType.getShape());
    assert(!transposeShape.empty() && "unexpected empty transpose shape");
    vectorType = VectorType::get(transposeShape, vectorType.getElementType());
    SmallVector<Value> indices(linalgOp.getRank(outputOperand),
                               b.create<arith::ConstantIndexOp>(loc, 0));
    value = broadcastIfNeeded(b, value, vectorType.getShape());
    value = reduceIfNeeded(b, vectorType, value, outputOperand, bvm);
    write = b.create<vector::TransferWriteOp>(loc, value, outputOperand->get(),
                                              indices, map);
  } else {
    value = reduceIfNeeded(b, getElementTypeOrSelf(value), value, outputOperand,
                           bvm);
    write = vector::TransferWriteOp::createScalarOp(
        b, loc, value, outputOperand->get(), ValueRange{});
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
  for (auto outputs : llvm::enumerate(yieldOp.values())) {
    // TODO: Scan for an opportunity for reuse.
    // TODO: use a map.
    Value vectorValue = bvm.lookup(outputs.value());
    Value newResult = buildVectorWrite(
        b, vectorValue, linalgOp.getOutputOperand(outputs.index()), bvm);
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
    if (linalgOp.isScalar(opOperand)) {
      bvm.map(bbarg, opOperand->get());
      continue;
    }
    // TODO: 0-d vectors.
    Type readType;
    AffineMap map;
    if (linalgOp.getShape(opOperand).empty()) {
      readType = bbarg.getType();
    } else {
      if (broadcastToMaximalCommonShape) {
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
    }
    Value readValue = buildVectorRead(b, opOperand->get(), readType, map);
    LDBG("new vectorized bbarg(" << bbarg.getArgNumber() << "): " << readValue);
    bvm.map(bbarg, readValue);
    bvm.map(opOperand->get(), readValue);
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

static LogicalResult vectorizeContraction(OpBuilder &b, LinalgOp linalgOp,
                                          SmallVectorImpl<Value> &newResults) {
  assert(isaContractionOpInterface(linalgOp) &&
         "expected vectorizeContraction preconditions to be met");
  Location loc = linalgOp.getLoc();
  // Vectorize other ops as vector contraction.
  // TODO: interface.
  LDBG(""
           << "Rewrite linalg op as vector.contract: ";
       linalgOp.dump());
  // Special function that describes how to vectorize the multiplication op in a
  // linalg contraction.
  CustomVectorizationHook vectorizeContraction =
      [&](Operation *op,
          const BlockAndValueMapping &bvm) -> VectorizationResult {
    if (!isa<arith::MulIOp, arith::MulFOp>(op))
      return VectorizationResult{VectorizationStatus::Failure, nullptr};
    ArrayRef<int64_t> outShape =
        linalgOp.getShape(linalgOp.getOutputOperand(0));
    Type vType;
    if (outShape.empty()) {
      vType = op->getResult(0).getType();
    } else {
      SmallVector<int64_t> resultShape = applyPermutationMap(
          inversePermutation(reindexIndexingMap(
              linalgOp.getTiedIndexingMap(linalgOp.getOutputOperand(0)))),
          outShape);
      vType = VectorType::get(resultShape, op->getResult(0).getType());
    }
    auto zero = b.create<arith::ConstantOp>(loc, vType, b.getZeroAttr(vType));
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
  return llvm::all_of(op.getIndexingMaps(), [](AffineMap m) {
    return m.isProjectedPermutation(/*allowZerosInResults=*/true);
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
    if (!reduceOp || !getKindForOp(reduceOp)) {
      LDBG("reduction precondition failed: reduction detection failed");
      return failure();
    }
  }
  return success();
}

LogicalResult mlir::linalg::vectorizeLinalgOpPrecondition(Operation *op) {
  auto linalgOp = cast<linalg::LinalgOp>(op);
  // All types must be static shape to go to vector.
  if (linalgOp.hasDynamicShape()) {
    LDBG("precondition failed: dynamic shape");
    return failure();
  }
  if (isElementwise(op))
    return success();
  if (isaContractionOpInterface(linalgOp))
    return success();
  // TODO: the common vector shape is equal to the static loop sizes only when
  // all indexing maps are projected permutations. For convs and stencils the
  // logic will need to evolve.
  if (!allIndexingsAreProjectedPermutation(linalgOp)) {
    LDBG("precondition failed: not projected permutations");
    return failure();
  }
  if (failed(reductionPreconditions(linalgOp))) {
    LDBG("precondition failed: reduction preconditions");
    return failure();
  }
  return success();
}

LogicalResult
mlir::linalg::vectorizeLinalgOp(OpBuilder &b, Operation *op,
                                SmallVectorImpl<Value> &newResults) {
  if (failed(vectorizeLinalgOpPrecondition(op)))
    return failure();

  auto linalgOp = cast<LinalgOp>(op);
  if (isaContractionOpInterface(linalgOp))
    return vectorizeContraction(b, linalgOp, newResults);

  LDBG(""
       << "Vectorize linalg op as a generic by broadcasting to "
          "maximal common shape: "
       << *op);
  return vectorizeAsLinalgGeneric(b, linalgOp, newResults,
                                  /*broadcastToMaximalCommonShape=*/true);
}

//----------------------------------------------------------------------------//
// Misc. vectorization patterns.
//----------------------------------------------------------------------------//

/// Helper function that retrieves the value of an IntegerAttr.
static int64_t getIntFromAttr(Attribute attr) {
  return attr.cast<IntegerAttr>().getInt();
}

/// Given an ArrayRef of OpFoldResults, return a vector of Values. IntegerAttrs
/// are converted to ConstantIndexOps. Other attribute types are not supported.
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

/// Rewrite a PadTensorOp into a sequence of InitTensorOp, FillOp and
/// InsertSliceOp. For now, only constant padding values are supported.
/// If there is enough static type information, TransferReadOps and
/// TransferWriteOps may be generated instead of InsertSliceOps.
struct GenericPadTensorOpVectorizationPattern
    : public GeneralizePadTensorOpPattern {
  GenericPadTensorOpVectorizationPattern(MLIRContext *context,
                                         PatternBenefit benefit = 1)
      : GeneralizePadTensorOpPattern(context, tryVectorizeCopy, benefit) {}
  /// Vectorize the copying of a PadTensorOp's source. This is possible if each
  /// dimension size is statically know in the source type or the result type
  /// (or both).
  static LogicalResult tryVectorizeCopy(PatternRewriter &rewriter,
                                        PadTensorOp padOp, Value dest) {
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
        // Source shape is statically known: Neither read nor write are out-of-
        // bounds.
        readInBounds.push_back(true);
        writeInBounds.push_back(true);
      } else if (!resultType.isDynamicDim(i)) {
        // Source shape is not statically known, but result shape is. Vectorize
        // with size of result shape. This may be larger than the source size.
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
        readInBounds);

    // If `dest` is a FillOp and the TransferWriteOp would overwrite the entire
    // tensor, write directly to the FillOp's operand.
    if (llvm::equal(vecShape, resultType.getShape()) &&
        llvm::all_of(writeInBounds, [](bool b) { return b; }))
      if (auto fill = dest.getDefiningOp<FillOp>())
        dest = fill.output();

    // Generate TransferWriteOp.
    auto writeIndices =
        ofrToIndexValues(rewriter, padOp.getLoc(), padOp.getMixedLowPad());
    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        padOp, read, dest, writeIndices, writeInBounds);

    return success();
  }
};

/// Base pattern for rewriting PadTensorOps whose result is consumed by a given
/// operation type OpTy.
template <typename OpTy>
struct VectorizePadTensorOpUserPattern : public OpRewritePattern<PadTensorOp> {
  using OpRewritePattern<PadTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PadTensorOp padOp,
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
                                    PadTensorOp padOp, OpTy op) const = 0;
};

/// Rewrite use of PadTensorOp result in TransferReadOp. E.g.:
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
struct PadTensorOpVectorizationWithTransferReadPattern
    : public VectorizePadTensorOpUserPattern<vector::TransferReadOp> {
  using VectorizePadTensorOpUserPattern<
      vector::TransferReadOp>::VectorizePadTensorOpUserPattern;

  LogicalResult rewriteUser(PatternRewriter &rewriter, PadTensorOp padOp,
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

/// Rewrite use of PadTensorOp result in TransferWriteOp.
/// This pattern rewrites TransferWriteOps that write to a padded tensor value,
/// where the same amount of padding is immediately removed again after the
/// write. In such cases, the TransferWriteOp can write to the non-padded tensor
/// value and apply out-of-bounds masking. E.g.:
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
/// %r = vector.transfer_write %vec, %0[...] : vector<17x5xf32>, tensor<?x?xf32>
/// ```
/// Note: It is important that the ExtractSliceOp %r resizes the result of the
/// TransferWriteOp to the same size as the input of the TensorPadOp (or an even
/// smaller size). Otherwise, %r's new (dynamic) dimensions would differ from
/// %r's old dimensions.
///
/// This rewrite is possible if:
/// - Low padding is static 0.
/// - `xferOp` has exactly one use, which is an ExtractSliceOp. This
///   ExtractSliceOp trims the same amount of padding that was added beforehand.
/// - Single, scalar padding value.
struct PadTensorOpVectorizationWithTransferWritePattern
    : public VectorizePadTensorOpUserPattern<vector::TransferWriteOp> {
  using VectorizePadTensorOpUserPattern<
      vector::TransferWriteOp>::VectorizePadTensorOpUserPattern;

  LogicalResult rewriteUser(PatternRewriter &rewriter, PadTensorOp padOp,
                            vector::TransferWriteOp xferOp) const override {
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
    // If the input to PadTensorOp is a CastOp, try with with both CastOp result
    // and CastOp operand.
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

    // All dynamic sizes must be the same. The only supported case at the moment
    // is when `beforePadding` is an ExtractSliceOp (or a cast thereof).

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

/// Rewrite use of PadTensorOp result in InsertSliceOp. E.g.:
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
struct PadTensorOpVectorizationWithInsertSlicePattern
    : public VectorizePadTensorOpUserPattern<tensor::InsertSliceOp> {
  using VectorizePadTensorOpUserPattern<
      tensor::InsertSliceOp>::VectorizePadTensorOpUserPattern;

  LogicalResult rewriteUser(PatternRewriter &rewriter, PadTensorOp padOp,
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

    // Generate TransferReadOp: Read entire source tensor and add high padding.
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
        insertOp, read, insertOp.dest(), writeIndices, inBounds);

    return success();
  }
};

void mlir::linalg::populatePadTensorOpVectorizationPatterns(
    RewritePatternSet &patterns, PatternBenefit baseBenefit) {
  patterns.add<GenericPadTensorOpVectorizationPattern>(patterns.getContext(),
                                                       baseBenefit);
  // Try these specialized patterns first before resorting to the generic one.
  patterns.add<PadTensorOpVectorizationWithTransferReadPattern,
               PadTensorOpVectorizationWithTransferWritePattern,
               PadTensorOpVectorizationWithInsertSlicePattern>(
      patterns.getContext(), baseBenefit.getBenefit() + 1);
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
  Type elemType = getElementTypeOrSelf(input->get());

  auto map = AffineMap::get(rank, 0, mapping, context);
  SmallVector<Value, 4> zeros(rank,
                              rewriter.create<arith::ConstantIndexOp>(loc, 0));
  auto vecType = VectorType::get(vectorDims, elemType);

  auto inputVec = rewriter.create<vector::TransferReadOp>(
      loc, vecType, input->get(), zeros, map);
  auto kernelVec = rewriter.create<vector::TransferReadOp>(
      loc, vecType, kernel->get(), zeros, map);

  auto acc = rewriter.create<arith::ConstantOp>(loc, elemType,
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
  populateVectorizationPatterns<Conv1DOp, 1>(tiling, promotion, vectorization,
                                             tileSizes);

  populateVectorizationPatterns<Conv2DOp, 2>(tiling, promotion, vectorization,
                                             tileSizes);

  populateVectorizationPatterns<Conv3DOp, 3>(tiling, promotion, vectorization,
                                             tileSizes);

  populateVectorizationPatterns<Conv1DNwcWcfOp, 3>(tiling, promotion,
                                                   vectorization, tileSizes);

  populateVectorizationPatterns<Conv2DNhwcHwcfOp, 4>(tiling, promotion,
                                                     vectorization, tileSizes);

  populateVectorizationPatterns<Conv3DNdhwcDhwcfOp, 5>(
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

  LDBG(viewOrAlloc);

  // Ensure there is exactly one subview of `viewOrAlloc` defining `subView`.
  memref::SubViewOp subViewOp = getSubViewUseIfUnique(viewOrAlloc);
  if (!subViewOp)
    return failure();
  Value subView = subViewOp.getResult();
  LDBG("with subView " << subView);

  // Find the copy into `subView` without interleaved uses.
  CopyOp copyOp;
  for (auto &u : subView.getUses()) {
    if (auto newCopyOp = dyn_cast<CopyOp>(u.getOwner())) {
      assert(newCopyOp.output().getType().isa<MemRefType>());
      if (newCopyOp.output() != subView)
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

  // Find the fill into `viewOrAlloc` without interleaved uses before the copy.
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
