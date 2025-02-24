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
#include "llvm/Support/MathExtras.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Debug.h"
#include <numeric>
#include <optional>
#include <type_traits>

using namespace mlir;
using namespace mlir::affine;

#define DEBUG_TYPE "affine-loop-analysis"

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
    *tripCountMap = AffineMap::getConstantMap(
        llvm::divideCeilSigned(loopSpan, step), context);
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
    if (auto constExpr = dyn_cast<AffineConstantExpr>(resultExpr)) {
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
    if (auto constExpr = dyn_cast<AffineConstantExpr>(resultExpr)) {
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

/// Given an affine.for `iv` and an access `index` of type index, returns `true`
/// if `index` is independent of `iv` and false otherwise.
///
/// Prerequisites: `iv` and `index` of the proper type;
static bool isAccessIndexInvariant(Value iv, Value index) {
  assert(isAffineForInductionVar(iv) && "iv must be an affine.for iv");
  assert(isa<IndexType>(index.getType()) && "index must be of 'index' type");
  auto map = AffineMap::getMultiDimIdentityMap(/*numDims=*/1, iv.getContext());
  SmallVector<Value> operands = {index};
  AffineValueMap avm(map, operands);
  avm.composeSimplifyAndCanonicalize();
  return !avm.isFunctionOf(0, iv);
}

// Pre-requisite: Loop bounds should be in canonical form.
template <typename LoadOrStoreOp>
bool mlir::affine::isInvariantAccess(LoadOrStoreOp memOp, AffineForOp forOp) {
  AffineValueMap avm(memOp.getAffineMap(), memOp.getMapOperands());
  avm.composeSimplifyAndCanonicalize();
  return !llvm::is_contained(avm.getOperands(), forOp.getInductionVar());
}

// Explicitly instantiate the template so that the compiler knows we need them.
template bool mlir::affine::isInvariantAccess(AffineReadOpInterface,
                                              AffineForOp);
template bool mlir::affine::isInvariantAccess(AffineWriteOpInterface,
                                              AffineForOp);
template bool mlir::affine::isInvariantAccess(AffineLoadOp, AffineForOp);
template bool mlir::affine::isInvariantAccess(AffineStoreOp, AffineForOp);

DenseSet<Value> mlir::affine::getInvariantAccesses(Value iv,
                                                   ArrayRef<Value> indices) {
  DenseSet<Value> res;
  for (Value index : indices) {
    if (isAccessIndexInvariant(iv, index))
      res.insert(index);
  }
  return res;
}

// TODO: check access stride.
template <typename LoadOrStoreOp>
bool mlir::affine::isContiguousAccess(Value iv, LoadOrStoreOp memoryOp,
                                      int *memRefDim) {
  static_assert(llvm::is_one_of<LoadOrStoreOp, AffineReadOpInterface,
                                AffineWriteOpInterface>::value,
                "Must be called on either an affine read or write op");
  assert(memRefDim && "memRefDim == nullptr");
  auto memRefType = memoryOp.getMemRefType();

  if (!memRefType.getLayout().isIdentity())
    return memoryOp.emitError("NYI: non-trivial layout map"), false;

  int uniqueVaryingIndexAlongIv = -1;
  auto accessMap = memoryOp.getAffineMap();
  SmallVector<Value, 4> mapOperands(memoryOp.getMapOperands());
  unsigned numDims = accessMap.getNumDims();
  for (unsigned i = 0, e = memRefType.getRank(); i < e; ++i) {
    // Gather map operands used in result expr 'i' in 'exprOperands'.
    SmallVector<Value, 4> exprOperands;
    auto resultExpr = accessMap.getResult(i);
    resultExpr.walk([&](AffineExpr expr) {
      if (auto dimExpr = dyn_cast<AffineDimExpr>(expr))
        exprOperands.push_back(mapOperands[dimExpr.getPosition()]);
      else if (auto symExpr = dyn_cast<AffineSymbolExpr>(expr))
        exprOperands.push_back(mapOperands[numDims + symExpr.getPosition()]);
    });
    // Check access invariance of each operand in 'exprOperands'.
    for (Value exprOperand : exprOperands) {
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

template bool mlir::affine::isContiguousAccess(Value iv,
                                               AffineReadOpInterface loadOp,
                                               int *memRefDim);
template bool mlir::affine::isContiguousAccess(Value iv,
                                               AffineWriteOpInterface loadOp,
                                               int *memRefDim);

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
    bool isContiguous =
        load ? isContiguousAccess(loop.getInductionVar(),
                                  cast<AffineReadOpInterface>(*load),
                                  &thisOpMemRefDim)
             : isContiguousAccess(loop.getInductionVar(),
                                  cast<AffineWriteOpInterface>(*store),
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

bool mlir::affine::isTilingValid(ArrayRef<AffineForOp> loops) {
  assert(!loops.empty() && "no original loops provided");

  // We first find out all dependences we intend to check.
  SmallVector<Operation *, 8> loadAndStoreOps;
  loops[0]->walk([&](Operation *op) {
    if (isa<AffineReadOpInterface, AffineWriteOpInterface>(op))
      loadAndStoreOps.push_back(op);
  });

  unsigned numOps = loadAndStoreOps.size();
  unsigned numLoops = loops.size();
  for (unsigned d = 1; d <= numLoops + 1; ++d) {
    for (unsigned i = 0; i < numOps; ++i) {
      Operation *srcOp = loadAndStoreOps[i];
      MemRefAccess srcAccess(srcOp);
      for (unsigned j = 0; j < numOps; ++j) {
        Operation *dstOp = loadAndStoreOps[j];
        MemRefAccess dstAccess(dstOp);

        SmallVector<DependenceComponent, 2> depComps;
        DependenceResult result = checkMemrefAccessDependence(
            srcAccess, dstAccess, d, /*dependenceConstraints=*/nullptr,
            &depComps);

        // Skip if there is no dependence in this case.
        if (!hasDependence(result))
          continue;

        // Check whether there is any negative direction vector in the
        // dependence components found above, which means that dependence is
        // violated by the default hyper-rect tiling method.
        LLVM_DEBUG(llvm::dbgs() << "Checking whether tiling legality violated "
                                   "for dependence at depth: "
                                << Twine(d) << " between:\n";);
        LLVM_DEBUG(srcAccess.opInst->dump());
        LLVM_DEBUG(dstAccess.opInst->dump());
        for (const DependenceComponent &depComp : depComps) {
          if (depComp.lb.has_value() && depComp.ub.has_value() &&
              *depComp.lb < *depComp.ub && *depComp.ub < 0) {
            LLVM_DEBUG(llvm::dbgs()
                       << "Dependence component lb = " << Twine(*depComp.lb)
                       << " ub = " << Twine(*depComp.ub)
                       << " is negative  at depth: " << Twine(d)
                       << " and thus violates the legality rule.\n");
            return false;
          }
        }
      }
    }
  }

  return true;
}
