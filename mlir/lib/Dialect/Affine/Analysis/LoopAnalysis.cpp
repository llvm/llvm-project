//===- LoopAnalysis.cpp - Misc loop analysis routines //-------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements miscellaneous loop analysis routines.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/NestedMatcher.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Support/MathExtras.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include <numeric>
#include <optional>
#include <type_traits>

using namespace mlir;
using namespace mlir::affine;

/// Returns the trip count of the loop as an affine expression if the latter is
/// expressible as an affine expression, and nullptr otherwise. The trip count
/// expression is simplified before returning. This method only utilizes map
/// composition to construct lower and upper bounds before computing the trip
/// count expressions.
void mlir::affine::getTripCountMapAndOperands(
    AffineForOp forOp, AffineMap *tripCountMap,
    SmallVectorImpl<Value> *tripCountOperands) {
  MLIRContext *context = forOp.getContext();
  int64_t step = forOp.getStepAsInt();
  int64_t loopSpan;
  if (forOp.hasConstantBounds()) {
    int64_t lb = forOp.getConstantLowerBound();
    int64_t ub = forOp.getConstantUpperBound();
    loopSpan = ub - lb;
    if (loopSpan < 0)
      loopSpan = 0;
    *tripCountMap = AffineMap::getConstantMap(ceilDiv(loopSpan, step), context);
    tripCountOperands->clear();
    return;
  }
  auto lbMap = forOp.getLowerBoundMap();
  auto ubMap = forOp.getUpperBoundMap();
  if (lbMap.getNumResults() != 1) {
    *tripCountMap = AffineMap();
    return;
  }

  // Difference of each upper bound expression from the single lower bound
  // expression (divided by the step) provides the expressions for the trip
  // count map.
  AffineValueMap ubValueMap(ubMap, forOp.getUpperBoundOperands());

  SmallVector<AffineExpr, 4> lbSplatExpr(ubValueMap.getNumResults(),
                                         lbMap.getResult(0));
  auto lbMapSplat = AffineMap::get(lbMap.getNumDims(), lbMap.getNumSymbols(),
                                   lbSplatExpr, context);
  AffineValueMap lbSplatValueMap(lbMapSplat, forOp.getLowerBoundOperands());

  AffineValueMap tripCountValueMap;
  AffineValueMap::difference(ubValueMap, lbSplatValueMap, &tripCountValueMap);
  for (unsigned i = 0, e = tripCountValueMap.getNumResults(); i < e; ++i)
    tripCountValueMap.setResult(i,
                                tripCountValueMap.getResult(i).ceilDiv(step));

  *tripCountMap = tripCountValueMap.getAffineMap();
  tripCountOperands->assign(tripCountValueMap.getOperands().begin(),
                            tripCountValueMap.getOperands().end());
}

/// Returns the trip count of the loop if it's a constant, std::nullopt
/// otherwise. This method uses affine expression analysis (in turn using
/// getTripCount) and is able to determine constant trip count in non-trivial
/// cases.
std::optional<uint64_t> mlir::affine::getConstantTripCount(AffineForOp forOp) {
  SmallVector<Value, 4> operands;
  AffineMap map;
  getTripCountMapAndOperands(forOp, &map, &operands);

  if (!map)
    return std::nullopt;

  // Take the min if all trip counts are constant.
  std::optional<uint64_t> tripCount;
  for (auto resultExpr : map.getResults()) {
    if (auto constExpr = resultExpr.dyn_cast<AffineConstantExpr>()) {
      if (tripCount.has_value())
        tripCount =
            std::min(*tripCount, static_cast<uint64_t>(constExpr.getValue()));
      else
        tripCount = constExpr.getValue();
    } else
      return std::nullopt;
  }
  return tripCount;
}

/// Returns the greatest known integral divisor of the trip count. Affine
/// expression analysis is used (indirectly through getTripCount), and
/// this method is thus able to determine non-trivial divisors.
uint64_t mlir::affine::getLargestDivisorOfTripCount(AffineForOp forOp) {
  SmallVector<Value, 4> operands;
  AffineMap map;
  getTripCountMapAndOperands(forOp, &map, &operands);

  if (!map)
    return 1;

  // The largest divisor of the trip count is the GCD of the individual largest
  // divisors.
  assert(map.getNumResults() >= 1 && "expected one or more results");
  std::optional<uint64_t> gcd;
  for (auto resultExpr : map.getResults()) {
    uint64_t thisGcd;
    if (auto constExpr = resultExpr.dyn_cast<AffineConstantExpr>()) {
      uint64_t tripCount = constExpr.getValue();
      // 0 iteration loops (greatest divisor is 2^64 - 1).
      if (tripCount == 0)
        thisGcd = std::numeric_limits<uint64_t>::max();
      else
        // The greatest divisor is the trip count.
        thisGcd = tripCount;
    } else {
      // Trip count is not a known constant; return its largest known divisor.
      thisGcd = resultExpr.getLargestKnownDivisor();
    }
    if (gcd.has_value())
      gcd = std::gcd(*gcd, thisGcd);
    else
      gcd = thisGcd;
  }
  assert(gcd.has_value() && "value expected per above logic");
  return *gcd;
}

/// Given an induction variable `iv` of type AffineForOp and an access `index`
/// of type index, returns `true` if `index` is independent of `iv` and
/// false otherwise. The determination supports composition with at most one
/// AffineApplyOp. The 'at most one AffineApplyOp' comes from the fact that
/// the composition of AffineApplyOp needs to be canonicalized by construction
/// to avoid writing code that composes arbitrary numbers of AffineApplyOps
/// everywhere. To achieve this, at the very least, the compose-affine-apply
/// pass must have been run.
///
/// Prerequisites:
///   1. `iv` and `index` of the proper type;
///   2. at most one reachable AffineApplyOp from index;
///
/// Returns false in cases with more than one AffineApplyOp, this is
/// conservative.
static bool isAccessIndexInvariant(Value iv, Value index) {
  assert(isAffineForInductionVar(iv) && "iv must be a AffineForOp");
  assert(isa<IndexType>(index.getType()) && "index must be of IndexType");
  SmallVector<Operation *, 4> affineApplyOps;
  getReachableAffineApplyOps({index}, affineApplyOps);

  if (affineApplyOps.empty()) {
    // Pointer equality test because of Value pointer semantics.
    return index != iv;
  }

  if (affineApplyOps.size() > 1) {
    affineApplyOps[0]->emitRemark(
        "CompositionAffineMapsPass must have been run: there should be at most "
        "one AffineApplyOp, returning false conservatively.");
    return false;
  }

  auto composeOp = cast<AffineApplyOp>(affineApplyOps[0]);
  // We need yet another level of indirection because the `dim` index of the
  // access may not correspond to the `dim` index of composeOp.
  return !composeOp.getAffineValueMap().isFunctionOf(0, iv);
}

DenseSet<Value> mlir::affine::getInvariantAccesses(Value iv,
                                                   ArrayRef<Value> indices) {
  DenseSet<Value> res;
  for (auto val : indices) {
    if (isAccessIndexInvariant(iv, val)) {
      res.insert(val);
    }
  }
  return res;
}

/// Given:
///   1. an induction variable `iv` of type AffineForOp;
///   2. a `memoryOp` of type const LoadOp& or const StoreOp&;
/// determines whether `memoryOp` has a contiguous access along `iv`. Contiguous
/// is defined as either invariant or varying only along a unique MemRef dim.
/// Upon success, the unique MemRef dim is written in `memRefDim` (or -1 to
/// convey the memRef access is invariant along `iv`).
///
/// Prerequisites:
///   1. `memRefDim` ~= nullptr;
///   2. `iv` of the proper type;
///   3. the MemRef accessed by `memoryOp` has no layout map or at most an
///      identity layout map.
///
/// Currently only supports no layoutMap or identity layoutMap in the MemRef.
/// Returns false if the MemRef has a non-identity layoutMap or more than 1
/// layoutMap. This is conservative.
///
// TODO: check strides.
template <typename LoadOrStoreOp>
static bool isContiguousAccess(Value iv, LoadOrStoreOp memoryOp,
                               int *memRefDim) {
  static_assert(
      llvm::is_one_of<LoadOrStoreOp, AffineLoadOp, AffineStoreOp>::value,
      "Must be called on either LoadOp or StoreOp");
  assert(memRefDim && "memRefDim == nullptr");
  auto memRefType = memoryOp.getMemRefType();

  if (!memRefType.getLayout().isIdentity())
    return memoryOp.emitError("NYI: non-trivial layoutMap"), false;

  int uniqueVaryingIndexAlongIv = -1;
  auto accessMap = memoryOp.getAffineMap();
  SmallVector<Value, 4> mapOperands(memoryOp.getMapOperands());
  unsigned numDims = accessMap.getNumDims();
  for (unsigned i = 0, e = memRefType.getRank(); i < e; ++i) {
    // Gather map operands used result expr 'i' in 'exprOperands'.
    SmallVector<Value, 4> exprOperands;
    auto resultExpr = accessMap.getResult(i);
    resultExpr.walk([&](AffineExpr expr) {
      if (auto dimExpr = expr.dyn_cast<AffineDimExpr>())
        exprOperands.push_back(mapOperands[dimExpr.getPosition()]);
      else if (auto symExpr = expr.dyn_cast<AffineSymbolExpr>())
        exprOperands.push_back(mapOperands[numDims + symExpr.getPosition()]);
    });
    // Check access invariance of each operand in 'exprOperands'.
    for (auto exprOperand : exprOperands) {
      if (!isAccessIndexInvariant(iv, exprOperand)) {
        if (uniqueVaryingIndexAlongIv != -1) {
          // 2+ varying indices -> do not vectorize along iv.
          return false;
        }
        uniqueVaryingIndexAlongIv = i;
      }
    }
  }

  if (uniqueVaryingIndexAlongIv == -1)
    *memRefDim = -1;
  else
    *memRefDim = memRefType.getRank() - (uniqueVaryingIndexAlongIv + 1);
  return true;
}

template <typename LoadOrStoreOp>
static bool isVectorElement(LoadOrStoreOp memoryOp) {
  auto memRefType = memoryOp.getMemRefType();
  return isa<VectorType>(memRefType.getElementType());
}

using VectorizableOpFun = std::function<bool(AffineForOp, Operation &)>;

static bool
isVectorizableLoopBodyWithOpCond(AffineForOp loop,
                                 const VectorizableOpFun &isVectorizableOp,
                                 NestedPattern &vectorTransferMatcher) {
  auto *forOp = loop.getOperation();

  // No vectorization across conditionals for now.
  auto conditionals = matcher::If();
  SmallVector<NestedMatch, 8> conditionalsMatched;
  conditionals.match(forOp, &conditionalsMatched);
  if (!conditionalsMatched.empty()) {
    return false;
  }

  // No vectorization for ops with operand or result types that are not
  // vectorizable.
  auto types = matcher::Op([](Operation &op) -> bool {
    if (llvm::any_of(op.getOperandTypes(), [](Type type) {
          if (MemRefType t = dyn_cast<MemRefType>(type))
            return !VectorType::isValidElementType(t.getElementType());
          return !VectorType::isValidElementType(type);
        }))
      return true;
    return llvm::any_of(op.getResultTypes(), [](Type type) {
      return !VectorType::isValidElementType(type);
    });
  });
  SmallVector<NestedMatch, 8> opsMatched;
  types.match(forOp, &opsMatched);
  if (!opsMatched.empty()) {
    return false;
  }

  // No vectorization across unknown regions.
  auto regions = matcher::Op([](Operation &op) -> bool {
    return op.getNumRegions() != 0 && !isa<AffineIfOp, AffineForOp>(op);
  });
  SmallVector<NestedMatch, 8> regionsMatched;
  regions.match(forOp, &regionsMatched);
  if (!regionsMatched.empty()) {
    return false;
  }

  SmallVector<NestedMatch, 8> vectorTransfersMatched;
  vectorTransferMatcher.match(forOp, &vectorTransfersMatched);
  if (!vectorTransfersMatched.empty()) {
    return false;
  }

  auto loadAndStores = matcher::Op(matcher::isLoadOrStore);
  SmallVector<NestedMatch, 8> loadAndStoresMatched;
  loadAndStores.match(forOp, &loadAndStoresMatched);
  for (auto ls : loadAndStoresMatched) {
    auto *op = ls.getMatchedOperation();
    auto load = dyn_cast<AffineLoadOp>(op);
    auto store = dyn_cast<AffineStoreOp>(op);
    // Only scalar types are considered vectorizable, all load/store must be
    // vectorizable for a loop to qualify as vectorizable.
    // TODO: ponder whether we want to be more general here.
    bool vector = load ? isVectorElement(load) : isVectorElement(store);
    if (vector) {
      return false;
    }
    if (isVectorizableOp && !isVectorizableOp(loop, *op)) {
      return false;
    }
  }
  return true;
}

bool mlir::affine::isVectorizableLoopBody(
    AffineForOp loop, int *memRefDim, NestedPattern &vectorTransferMatcher) {
  *memRefDim = -1;
  VectorizableOpFun fun([memRefDim](AffineForOp loop, Operation &op) {
    auto load = dyn_cast<AffineLoadOp>(op);
    auto store = dyn_cast<AffineStoreOp>(op);
    int thisOpMemRefDim = -1;
    bool isContiguous = load ? isContiguousAccess(loop.getInductionVar(), load,
                                                  &thisOpMemRefDim)
                             : isContiguousAccess(loop.getInductionVar(), store,
                                                  &thisOpMemRefDim);
    if (thisOpMemRefDim != -1) {
      // If memory accesses vary across different dimensions then the loop is
      // not vectorizable.
      if (*memRefDim != -1 && *memRefDim != thisOpMemRefDim)
        return false;
      *memRefDim = thisOpMemRefDim;
    }
    return isContiguous;
  });
  return isVectorizableLoopBodyWithOpCond(loop, fun, vectorTransferMatcher);
}

bool mlir::affine::isVectorizableLoopBody(
    AffineForOp loop, NestedPattern &vectorTransferMatcher) {
  return isVectorizableLoopBodyWithOpCond(loop, nullptr, vectorTransferMatcher);
}

/// Checks whether SSA dominance would be violated if a for op's body
/// operations are shifted by the specified shifts. This method checks if a
/// 'def' and all its uses have the same shift factor.
// TODO: extend this to check for memory-based dependence violation when we have
// the support.
bool mlir::affine::isOpwiseShiftValid(AffineForOp forOp,
                                      ArrayRef<uint64_t> shifts) {
  auto *forBody = forOp.getBody();
  assert(shifts.size() == forBody->getOperations().size());

  // Work backwards over the body of the block so that the shift of a use's
  // ancestor operation in the block gets recorded before it's looked up.
  DenseMap<Operation *, uint64_t> forBodyShift;
  for (const auto &it :
       llvm::enumerate(llvm::reverse(forBody->getOperations()))) {
    auto &op = it.value();

    // Get the index of the current operation, note that we are iterating in
    // reverse so we need to fix it up.
    size_t index = shifts.size() - it.index() - 1;

    // Remember the shift of this operation.
    uint64_t shift = shifts[index];
    forBodyShift.try_emplace(&op, shift);

    // Validate the results of this operation if it were to be shifted.
    for (unsigned i = 0, e = op.getNumResults(); i < e; ++i) {
      Value result = op.getResult(i);
      for (auto *user : result.getUsers()) {
        // If an ancestor operation doesn't lie in the block of forOp,
        // there is no shift to check.
        if (auto *ancOp = forBody->findAncestorOpInBlock(*user)) {
          assert(forBodyShift.count(ancOp) > 0 && "ancestor expected in map");
          if (shift != forBodyShift[ancOp])
            return false;
        }
      }
    }
  }
  return true;
}
