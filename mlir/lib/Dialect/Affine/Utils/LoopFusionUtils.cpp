//===- LoopFusionUtils.cpp ---- Utilities for loop fusion ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements loop fusion transformation utility functions.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/LoopFusionUtils.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "loop-fusion-utils"

using namespace mlir;

// Gathers all load and store memref accesses in 'opA' into 'values', where
// 'values[memref] == true' for each store operation.
static void getLoadAndStoreMemRefAccesses(Operation *opA,
                                          DenseMap<Value, bool> &values) {
  opA->walk([&](Operation *op) {
    if (auto loadOp = dyn_cast<AffineReadOpInterface>(op)) {
      if (values.count(loadOp.getMemRef()) == 0)
        values[loadOp.getMemRef()] = false;
    } else if (auto storeOp = dyn_cast<AffineWriteOpInterface>(op)) {
      values[storeOp.getMemRef()] = true;
    }
  });
}

/// Returns true if 'op' is a load or store operation which access a memref
/// accessed 'values' and at least one of the access is a store operation.
/// Returns false otherwise.
static bool isDependentLoadOrStoreOp(Operation *op,
                                     DenseMap<Value, bool> &values) {
  if (auto loadOp = dyn_cast<AffineReadOpInterface>(op)) {
    return values.count(loadOp.getMemRef()) > 0 && values[loadOp.getMemRef()];
  }
  if (auto storeOp = dyn_cast<AffineWriteOpInterface>(op)) {
    return values.count(storeOp.getMemRef()) > 0;
  }
  return false;
}

// Returns the first operation in range ('opA', 'opB') which has a data
// dependence on 'opA'. Returns 'nullptr' of no dependence exists.
static Operation *getFirstDependentOpInRange(Operation *opA, Operation *opB) {
  // Record memref values from all loads/store in loop nest rooted at 'opA'.
  // Map from memref value to bool which is true if store, false otherwise.
  DenseMap<Value, bool> values;
  getLoadAndStoreMemRefAccesses(opA, values);

  // For each 'opX' in block in range ('opA', 'opB'), check if there is a data
  // dependence from 'opA' to 'opX' ('opA' and 'opX' access the same memref
  // and at least one of the accesses is a store).
  Operation *firstDepOp = nullptr;
  for (Block::iterator it = std::next(Block::iterator(opA));
       it != Block::iterator(opB); ++it) {
    Operation *opX = &(*it);
    opX->walk([&](Operation *op) {
      if (!firstDepOp && isDependentLoadOrStoreOp(op, values))
        firstDepOp = opX;
    });
    if (firstDepOp)
      break;
  }
  return firstDepOp;
}

// Returns the last operation 'opX' in range ('opA', 'opB'), for which there
// exists a data dependence from 'opX' to 'opB'.
// Returns 'nullptr' of no dependence exists.
static Operation *getLastDependentOpInRange(Operation *opA, Operation *opB) {
  // Record memref values from all loads/store in loop nest rooted at 'opB'.
  // Map from memref value to bool which is true if store, false otherwise.
  DenseMap<Value, bool> values;
  getLoadAndStoreMemRefAccesses(opB, values);

  // For each 'opX' in block in range ('opA', 'opB') in reverse order,
  // check if there is a data dependence from 'opX' to 'opB':
  // *) 'opX' and 'opB' access the same memref and at least one of the accesses
  //    is a store.
  // *) 'opX' produces an SSA Value which is used by 'opB'.
  Operation *lastDepOp = nullptr;
  for (Block::reverse_iterator it = std::next(Block::reverse_iterator(opB));
       it != Block::reverse_iterator(opA); ++it) {
    Operation *opX = &(*it);
    opX->walk([&](Operation *op) {
      if (isa<AffineReadOpInterface, AffineWriteOpInterface>(op)) {
        if (isDependentLoadOrStoreOp(op, values)) {
          lastDepOp = opX;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      for (auto value : op->getResults()) {
        for (Operation *user : value.getUsers()) {
          SmallVector<AffineForOp, 4> loops;
          // Check if any loop in loop nest surrounding 'user' is 'opB'.
          getLoopIVs(*user, &loops);
          if (llvm::is_contained(loops, cast<AffineForOp>(opB))) {
            lastDepOp = opX;
            return WalkResult::interrupt();
          }
        }
      }
      return WalkResult::advance();
    });
    if (lastDepOp)
      break;
  }
  return lastDepOp;
}

// Computes and returns an insertion point operation, before which the
// the fused <srcForOp, dstForOp> loop nest can be inserted while preserving
// dependences. Returns nullptr if no such insertion point is found.
static Operation *getFusedLoopNestInsertionPoint(AffineForOp srcForOp,
                                                 AffineForOp dstForOp) {
  bool isSrcForOpBeforeDstForOp =
      srcForOp->isBeforeInBlock(dstForOp.getOperation());
  auto forOpA = isSrcForOpBeforeDstForOp ? srcForOp : dstForOp;
  auto forOpB = isSrcForOpBeforeDstForOp ? dstForOp : srcForOp;

  auto *firstDepOpA =
      getFirstDependentOpInRange(forOpA.getOperation(), forOpB.getOperation());
  auto *lastDepOpB =
      getLastDependentOpInRange(forOpA.getOperation(), forOpB.getOperation());
  // Block:
  //      ...
  //  |-- opA
  //  |   ...
  //  |   lastDepOpB --|
  //  |   ...          |
  //  |-> firstDepOpA  |
  //      ...          |
  //      opB <---------
  //
  // Valid insertion point range: (lastDepOpB, firstDepOpA)
  //
  if (firstDepOpA != nullptr) {
    if (lastDepOpB != nullptr) {
      if (firstDepOpA->isBeforeInBlock(lastDepOpB) || firstDepOpA == lastDepOpB)
        // No valid insertion point exists which preserves dependences.
        return nullptr;
    }
    // Return insertion point in valid range closest to 'opB'.
    // TODO: Consider other insertion points in valid range.
    return firstDepOpA;
  }
  // No dependences from 'opA' to operation in range ('opA', 'opB'), return
  // 'opB' insertion point.
  return forOpB.getOperation();
}

// Gathers all load and store ops in loop nest rooted at 'forOp' into
// 'loadAndStoreOps'.
static bool
gatherLoadsAndStores(AffineForOp forOp,
                     SmallVectorImpl<Operation *> &loadAndStoreOps) {
  bool hasIfOp = false;
  forOp.walk([&](Operation *op) {
    if (isa<AffineReadOpInterface, AffineWriteOpInterface>(op))
      loadAndStoreOps.push_back(op);
    else if (isa<AffineIfOp>(op))
      hasIfOp = true;
  });
  return !hasIfOp;
}

/// Returns the maximum loop depth at which we could fuse producer loop
/// 'srcForOp' into consumer loop 'dstForOp' without violating data dependences.
// TODO: Generalize this check for sibling and more generic fusion scenarios.
// TODO: Support forward slice fusion.
static unsigned getMaxLoopDepth(ArrayRef<Operation *> srcOps,
                                ArrayRef<Operation *> dstOps) {
  if (dstOps.empty())
    // Expected at least one memory operation.
    // TODO: Revisit this case with a specific example.
    return 0;

  // Filter out ops in 'dstOps' that do not use the producer-consumer memref so
  // that they are not considered for analysis.
  DenseSet<Value> producerConsumerMemrefs;
  gatherProducerConsumerMemrefs(srcOps, dstOps, producerConsumerMemrefs);
  SmallVector<Operation *, 4> targetDstOps;
  for (Operation *dstOp : dstOps) {
    auto loadOp = dyn_cast<AffineReadOpInterface>(dstOp);
    Value memref = loadOp ? loadOp.getMemRef()
                          : cast<AffineWriteOpInterface>(dstOp).getMemRef();
    if (producerConsumerMemrefs.count(memref) > 0)
      targetDstOps.push_back(dstOp);
  }

  assert(!targetDstOps.empty() &&
         "No dependences between 'srcForOp' and 'dstForOp'?");

  // Compute the innermost common loop depth for loads and stores.
  unsigned loopDepth = getInnermostCommonLoopDepth(targetDstOps);

  // Return common loop depth for loads if there are no store ops.
  if (all_of(targetDstOps,
             [&](Operation *op) { return isa<AffineReadOpInterface>(op); }))
    return loopDepth;

  // Check dependences on all pairs of ops in 'targetDstOps' and store the
  // minimum loop depth at which a dependence is satisfied.
  for (unsigned i = 0, e = targetDstOps.size(); i < e; ++i) {
    auto *srcOpInst = targetDstOps[i];
    MemRefAccess srcAccess(srcOpInst);
    for (unsigned j = 0; j < e; ++j) {
      auto *dstOpInst = targetDstOps[j];
      MemRefAccess dstAccess(dstOpInst);

      unsigned numCommonLoops =
          getNumCommonSurroundingLoops(*srcOpInst, *dstOpInst);
      for (unsigned d = 1; d <= numCommonLoops + 1; ++d) {
        FlatAffineValueConstraints dependenceConstraints;
        // TODO: Cache dependence analysis results, check cache here.
        DependenceResult result = checkMemrefAccessDependence(
            srcAccess, dstAccess, d, &dependenceConstraints,
            /*dependenceComponents=*/nullptr);
        if (hasDependence(result)) {
          // Store minimum loop depth and break because we want the min 'd' at
          // which there is a dependence.
          loopDepth = std::min(loopDepth, d - 1);
          break;
        }
      }
    }
  }

  return loopDepth;
}

// TODO: Prevent fusion of loop nests with side-effecting operations.
// TODO: This pass performs some computation that is the same for all the depths
// (e.g., getMaxLoopDepth). Implement a version of this utility that processes
// all the depths at once or only the legal maximal depth for maximal fusion.
FusionResult mlir::canFuseLoops(AffineForOp srcForOp, AffineForOp dstForOp,
                                unsigned dstLoopDepth,
                                ComputationSliceState *srcSlice,
                                FusionStrategy fusionStrategy) {
  // Return 'failure' if 'dstLoopDepth == 0'.
  if (dstLoopDepth == 0) {
    LLVM_DEBUG(llvm::dbgs() << "Cannot fuse loop nests at depth 0\n");
    return FusionResult::FailPrecondition;
  }
  // Return 'failure' if 'srcForOp' and 'dstForOp' are not in the same block.
  auto *block = srcForOp->getBlock();
  if (block != dstForOp->getBlock()) {
    LLVM_DEBUG(llvm::dbgs() << "Cannot fuse loop nests in different blocks\n");
    return FusionResult::FailPrecondition;
  }

  // Return 'failure' if no valid insertion point for fused loop nest in 'block'
  // exists which would preserve dependences.
  if (!getFusedLoopNestInsertionPoint(srcForOp, dstForOp)) {
    LLVM_DEBUG(llvm::dbgs() << "Fusion would violate dependences in block\n");
    return FusionResult::FailBlockDependence;
  }

  // Check if 'srcForOp' precedes 'dstForOp' in 'block'.
  bool isSrcForOpBeforeDstForOp =
      srcForOp->isBeforeInBlock(dstForOp.getOperation());
  // 'forOpA' executes before 'forOpB' in 'block'.
  auto forOpA = isSrcForOpBeforeDstForOp ? srcForOp : dstForOp;
  auto forOpB = isSrcForOpBeforeDstForOp ? dstForOp : srcForOp;

  // Gather all load and store from 'forOpA' which precedes 'forOpB' in 'block'.
  SmallVector<Operation *, 4> opsA;
  if (!gatherLoadsAndStores(forOpA, opsA)) {
    LLVM_DEBUG(llvm::dbgs() << "Fusing loops with affine.if unsupported\n");
    return FusionResult::FailPrecondition;
  }

  // Gather all load and store from 'forOpB' which succeeds 'forOpA' in 'block'.
  SmallVector<Operation *, 4> opsB;
  if (!gatherLoadsAndStores(forOpB, opsB)) {
    LLVM_DEBUG(llvm::dbgs() << "Fusing loops with affine.if unsupported\n");
    return FusionResult::FailPrecondition;
  }

  // Return 'failure' if fusing loops at depth 'dstLoopDepth' wouldn't preserve
  // loop dependences.
  // TODO: Enable this check for sibling and more generic loop fusion
  // strategies.
  if (fusionStrategy.getStrategy() == FusionStrategy::ProducerConsumer) {
    // TODO: 'getMaxLoopDepth' does not support forward slice fusion.
    assert(isSrcForOpBeforeDstForOp && "Unexpected forward slice fusion");
    if (getMaxLoopDepth(opsA, opsB) < dstLoopDepth) {
      LLVM_DEBUG(llvm::dbgs() << "Fusion would violate loop dependences\n");
      return FusionResult::FailFusionDependence;
    }
  }

  // Calculate the number of common loops surrounding 'srcForOp' and 'dstForOp'.
  unsigned numCommonLoops = mlir::getNumCommonSurroundingLoops(
      *srcForOp.getOperation(), *dstForOp.getOperation());

  // Filter out ops in 'opsA' to compute the slice union based on the
  // assumptions made by the fusion strategy.
  SmallVector<Operation *, 4> strategyOpsA;
  switch (fusionStrategy.getStrategy()) {
  case FusionStrategy::Generic:
    // Generic fusion. Take into account all the memory operations to compute
    // the slice union.
    strategyOpsA.append(opsA.begin(), opsA.end());
    break;
  case FusionStrategy::ProducerConsumer:
    // Producer-consumer fusion (AffineLoopFusion pass) only takes into
    // account stores in 'srcForOp' to compute the slice union.
    for (Operation *op : opsA) {
      if (isa<AffineWriteOpInterface>(op))
        strategyOpsA.push_back(op);
    }
    break;
  case FusionStrategy::Sibling:
    // Sibling fusion (AffineLoopFusion pass) only takes into account the loads
    // to 'memref' in 'srcForOp' to compute the slice union.
    for (Operation *op : opsA) {
      auto load = dyn_cast<AffineReadOpInterface>(op);
      if (load && load.getMemRef() == fusionStrategy.getSiblingFusionMemRef())
        strategyOpsA.push_back(op);
    }
    break;
  }

  // Compute union of computation slices computed between all pairs of ops
  // from 'forOpA' and 'forOpB'.
  SliceComputationResult sliceComputationResult =
      mlir::computeSliceUnion(strategyOpsA, opsB, dstLoopDepth, numCommonLoops,
                              isSrcForOpBeforeDstForOp, srcSlice);
  if (sliceComputationResult.value == SliceComputationResult::GenericFailure) {
    LLVM_DEBUG(llvm::dbgs() << "computeSliceUnion failed\n");
    return FusionResult::FailPrecondition;
  }
  if (sliceComputationResult.value ==
      SliceComputationResult::IncorrectSliceFailure) {
    LLVM_DEBUG(llvm::dbgs() << "Incorrect slice computation\n");
    return FusionResult::FailIncorrectSlice;
  }

  return FusionResult::Success;
}

/// Patch the loop body of a forOp that is a single iteration reduction loop
/// into its containing block.
LogicalResult promoteSingleIterReductionLoop(AffineForOp forOp,
                                             bool siblingFusionUser) {
  // Check if the reduction loop is a single iteration loop.
  Optional<uint64_t> tripCount = getConstantTripCount(forOp);
  if (!tripCount || tripCount.getValue() != 1)
    return failure();
  auto iterOperands = forOp.getIterOperands();
  auto *parentOp = forOp->getParentOp();
  if (!isa<AffineForOp>(parentOp))
    return failure();
  auto newOperands = forOp.getBody()->getTerminator()->getOperands();
  OpBuilder b(parentOp);
  // Replace the parent loop and add iteroperands and results from the `forOp`.
  AffineForOp parentForOp = forOp->getParentOfType<AffineForOp>();
  AffineForOp newLoop = replaceForOpWithNewYields(
      b, parentForOp, iterOperands, newOperands, forOp.getRegionIterArgs());

  // For sibling-fusion users, collect operations that use the results of the
  // `forOp` outside the new parent loop that has absorbed all its iter args
  // and operands. These operations will be moved later after the results
  // have been replaced.
  SetVector<Operation *> forwardSlice;
  if (siblingFusionUser) {
    for (unsigned i = 0, e = forOp.getNumResults(); i != e; ++i) {
      SetVector<Operation *> tmpForwardSlice;
      getForwardSlice(forOp.getResult(i), &tmpForwardSlice);
      forwardSlice.set_union(tmpForwardSlice);
    }
  }
  // Update the results of the `forOp` in the new loop.
  for (unsigned i = 0, e = forOp.getNumResults(); i != e; ++i) {
    forOp.getResult(i).replaceAllUsesWith(
        newLoop.getResult(i + parentOp->getNumResults()));
  }
  // For sibling-fusion users, move operations that use the results of the
  // `forOp` outside the new parent loop
  if (siblingFusionUser) {
    topologicalSort(forwardSlice);
    for (Operation *op : llvm::reverse(forwardSlice))
      op->moveAfter(newLoop);
  }
  // Replace the induction variable.
  auto iv = forOp.getInductionVar();
  iv.replaceAllUsesWith(newLoop.getInductionVar());
  // Replace the iter args.
  auto forOpIterArgs = forOp.getRegionIterArgs();
  for (auto it : llvm::zip(forOpIterArgs, newLoop.getRegionIterArgs().take_back(
                                              forOpIterArgs.size()))) {
    std::get<0>(it).replaceAllUsesWith(std::get<1>(it));
  }
  // Move the loop body operations, except for its terminator, to the loop's
  // containing block.
  forOp.getBody()->back().erase();
  auto *parentBlock = forOp->getBlock();
  parentBlock->getOperations().splice(Block::iterator(forOp),
                                      forOp.getBody()->getOperations());
  forOp.erase();
  parentForOp.erase();
  return success();
}

/// Fuses 'srcForOp' into 'dstForOp' with destination loop block insertion point
/// and source slice loop bounds specified in 'srcSlice'.
void mlir::fuseLoops(AffineForOp srcForOp, AffineForOp dstForOp,
                     const ComputationSliceState &srcSlice,
                     bool isInnermostSiblingInsertion) {
  // Clone 'srcForOp' into 'dstForOp' at 'srcSlice->insertPoint'.
  OpBuilder b(srcSlice.insertPoint->getBlock(), srcSlice.insertPoint);
  BlockAndValueMapping mapper;
  b.clone(*srcForOp, mapper);

  // Update 'sliceLoopNest' upper and lower bounds from computed 'srcSlice'.
  SmallVector<AffineForOp, 4> sliceLoops;
  for (unsigned i = 0, e = srcSlice.ivs.size(); i < e; ++i) {
    auto loopIV = mapper.lookupOrNull(srcSlice.ivs[i]);
    if (!loopIV)
      continue;
    auto forOp = getForInductionVarOwner(loopIV);
    sliceLoops.push_back(forOp);
    if (AffineMap lbMap = srcSlice.lbs[i]) {
      auto lbOperands = srcSlice.lbOperands[i];
      canonicalizeMapAndOperands(&lbMap, &lbOperands);
      forOp.setLowerBound(lbOperands, lbMap);
    }
    if (AffineMap ubMap = srcSlice.ubs[i]) {
      auto ubOperands = srcSlice.ubOperands[i];
      canonicalizeMapAndOperands(&ubMap, &ubOperands);
      forOp.setUpperBound(ubOperands, ubMap);
    }
  }

  llvm::SmallDenseMap<Operation *, uint64_t, 8> sliceTripCountMap;
  auto srcIsUnitSlice = [&]() {
    return (buildSliceTripCountMap(srcSlice, &sliceTripCountMap) &&
            (getSliceIterationCount(sliceTripCountMap) == 1));
  };
  // Fix up and if possible, eliminate single iteration loops.
  for (AffineForOp forOp : sliceLoops) {
    if (isLoopParallelAndContainsReduction(forOp) &&
        isInnermostSiblingInsertion && srcIsUnitSlice())
      // Patch reduction loop - only ones that are sibling-fused with the
      // destination loop - into the parent loop.
      (void)promoteSingleIterReductionLoop(forOp, true);
    else
      // Promote any single iteration slice loops.
      (void)promoteIfSingleIteration(forOp);
  }
}

/// Collect loop nest statistics (eg. loop trip count and operation count)
/// in 'stats' for loop nest rooted at 'forOp'. Returns true on success,
/// returns false otherwise.
bool mlir::getLoopNestStats(AffineForOp forOpRoot, LoopNestStats *stats) {
  auto walkResult = forOpRoot.walk([&](AffineForOp forOp) {
    auto *childForOp = forOp.getOperation();
    auto *parentForOp = forOp->getParentOp();
    if (!llvm::isa<func::FuncOp>(parentForOp)) {
      if (!isa<AffineForOp>(parentForOp)) {
        LLVM_DEBUG(llvm::dbgs() << "Expected parent AffineForOp\n");
        return WalkResult::interrupt();
      }
      // Add mapping to 'forOp' from its parent AffineForOp.
      stats->loopMap[parentForOp].push_back(forOp);
    }

    // Record the number of op operations in the body of 'forOp'.
    unsigned count = 0;
    stats->opCountMap[childForOp] = 0;
    for (auto &op : *forOp.getBody()) {
      if (!isa<AffineForOp, AffineIfOp>(op))
        ++count;
    }
    stats->opCountMap[childForOp] = count;

    // Record trip count for 'forOp'. Set flag if trip count is not
    // constant.
    Optional<uint64_t> maybeConstTripCount = getConstantTripCount(forOp);
    if (!maybeConstTripCount) {
      // Currently only constant trip count loop nests are supported.
      LLVM_DEBUG(llvm::dbgs() << "Non-constant trip count unsupported\n");
      return WalkResult::interrupt();
    }

    stats->tripCountMap[childForOp] = maybeConstTripCount.getValue();
    return WalkResult::advance();
  });
  return !walkResult.wasInterrupted();
}

// Computes the total cost of the loop nest rooted at 'forOp'.
// Currently, the total cost is computed by counting the total operation
// instance count (i.e. total number of operations in the loop bodyloop
// operation count * loop trip count) for the entire loop nest.
// If 'tripCountOverrideMap' is non-null, overrides the trip count for loops
// specified in the map when computing the total op instance count.
// NOTEs: 1) This is used to compute the cost of computation slices, which are
// sliced along the iteration dimension, and thus reduce the trip count.
// If 'computeCostMap' is non-null, the total op count for forOps specified
// in the map is increased (not overridden) by adding the op count from the
// map to the existing op count for the for loop. This is done before
// multiplying by the loop's trip count, and is used to model the cost of
// inserting a sliced loop nest of known cost into the loop's body.
// 2) This is also used to compute the cost of fusing a slice of some loop nest
// within another loop.
static int64_t getComputeCostHelper(
    Operation *forOp, LoopNestStats &stats,
    llvm::SmallDenseMap<Operation *, uint64_t, 8> *tripCountOverrideMap,
    DenseMap<Operation *, int64_t> *computeCostMap) {
  // 'opCount' is the total number operations in one iteration of 'forOp' body,
  // minus terminator op which is a no-op.
  int64_t opCount = stats.opCountMap[forOp] - 1;
  if (stats.loopMap.count(forOp) > 0) {
    for (auto childForOp : stats.loopMap[forOp]) {
      opCount += getComputeCostHelper(childForOp.getOperation(), stats,
                                      tripCountOverrideMap, computeCostMap);
    }
  }
  // Add in additional op instances from slice (if specified in map).
  if (computeCostMap != nullptr) {
    auto it = computeCostMap->find(forOp);
    if (it != computeCostMap->end()) {
      opCount += it->second;
    }
  }
  // Override trip count (if specified in map).
  int64_t tripCount = stats.tripCountMap[forOp];
  if (tripCountOverrideMap != nullptr) {
    auto it = tripCountOverrideMap->find(forOp);
    if (it != tripCountOverrideMap->end()) {
      tripCount = it->second;
    }
  }
  // Returns the total number of dynamic instances of operations in loop body.
  return tripCount * opCount;
}

/// Computes the total cost of the loop nest rooted at 'forOp' using 'stats'.
/// Currently, the total cost is computed by counting the total operation
/// instance count (i.e. total number of operations in the loop body * loop
/// trip count) for the entire loop nest.
int64_t mlir::getComputeCost(AffineForOp forOp, LoopNestStats &stats) {
  return getComputeCostHelper(forOp.getOperation(), stats,
                              /*tripCountOverrideMap=*/nullptr,
                              /*computeCostMap=*/nullptr);
}

/// Computes and returns in 'computeCost', the total compute cost of fusing the
/// 'slice' of the loop nest rooted at 'srcForOp' into 'dstForOp'. Currently,
/// the total cost is computed by counting the total operation instance count
/// (i.e. total number of operations in the loop body * loop trip count) for
/// the entire loop nest.
bool mlir::getFusionComputeCost(AffineForOp srcForOp, LoopNestStats &srcStats,
                                AffineForOp dstForOp, LoopNestStats &dstStats,
                                const ComputationSliceState &slice,
                                int64_t *computeCost) {
  llvm::SmallDenseMap<Operation *, uint64_t, 8> sliceTripCountMap;
  DenseMap<Operation *, int64_t> computeCostMap;

  // Build trip count map for computation slice.
  if (!buildSliceTripCountMap(slice, &sliceTripCountMap))
    return false;
  // Checks whether a store to load forwarding will happen.
  int64_t sliceIterationCount = getSliceIterationCount(sliceTripCountMap);
  assert(sliceIterationCount > 0);
  bool storeLoadFwdGuaranteed = (sliceIterationCount == 1);
  auto *insertPointParent = slice.insertPoint->getParentOp();

  // The store and loads to this memref will disappear.
  // TODO: Add load coalescing to memref data flow opt pass.
  if (storeLoadFwdGuaranteed) {
    // Subtract from operation count the loads/store we expect load/store
    // forwarding to remove.
    unsigned storeCount = 0;
    llvm::SmallDenseSet<Value, 4> storeMemrefs;
    srcForOp.walk([&](Operation *op) {
      if (auto storeOp = dyn_cast<AffineWriteOpInterface>(op)) {
        storeMemrefs.insert(storeOp.getMemRef());
        ++storeCount;
      }
    });
    // Subtract out any store ops in single-iteration src slice loop nest.
    if (storeCount > 0)
      computeCostMap[insertPointParent] = -storeCount;
    // Subtract out any load users of 'storeMemrefs' nested below
    // 'insertPointParent'.
    for (auto value : storeMemrefs) {
      for (auto *user : value.getUsers()) {
        if (auto loadOp = dyn_cast<AffineReadOpInterface>(user)) {
          SmallVector<AffineForOp, 4> loops;
          // Check if any loop in loop nest surrounding 'user' is
          // 'insertPointParent'.
          getLoopIVs(*user, &loops);
          if (llvm::is_contained(loops, cast<AffineForOp>(insertPointParent))) {
            if (auto forOp =
                    dyn_cast_or_null<AffineForOp>(user->getParentOp())) {
              if (computeCostMap.count(forOp) == 0)
                computeCostMap[forOp] = 0;
              computeCostMap[forOp] -= 1;
            }
          }
        }
      }
    }
  }

  // Compute op instance count for the src loop nest with iteration slicing.
  int64_t sliceComputeCost = getComputeCostHelper(
      srcForOp.getOperation(), srcStats, &sliceTripCountMap, &computeCostMap);

  // Compute cost of fusion for this depth.
  computeCostMap[insertPointParent] = sliceComputeCost;

  *computeCost =
      getComputeCostHelper(dstForOp.getOperation(), dstStats,
                           /*tripCountOverrideMap=*/nullptr, &computeCostMap);
  return true;
}

/// Returns in 'producerConsumerMemrefs' the memrefs involved in a
/// producer-consumer dependence between write ops in 'srcOps' and read ops in
/// 'dstOps'.
void mlir::gatherProducerConsumerMemrefs(
    ArrayRef<Operation *> srcOps, ArrayRef<Operation *> dstOps,
    DenseSet<Value> &producerConsumerMemrefs) {
  // Gather memrefs from stores in 'srcOps'.
  DenseSet<Value> srcStoreMemRefs;
  for (Operation *op : srcOps)
    if (auto storeOp = dyn_cast<AffineWriteOpInterface>(op))
      srcStoreMemRefs.insert(storeOp.getMemRef());

  // Compute the intersection between memrefs from stores in 'srcOps' and
  // memrefs from loads in 'dstOps'.
  for (Operation *op : dstOps)
    if (auto loadOp = dyn_cast<AffineReadOpInterface>(op))
      if (srcStoreMemRefs.count(loadOp.getMemRef()) > 0)
        producerConsumerMemrefs.insert(loadOp.getMemRef());
}
