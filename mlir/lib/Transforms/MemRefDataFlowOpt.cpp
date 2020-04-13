//===- MemRefDataFlowOpt.cpp - MemRef DataFlow Optimization pass ------ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to forward memref stores to loads, thereby
// potentially getting rid of intermediate memref's entirely.
// TODO(mlir-team): In the future, similar techniques could be used to eliminate
// dead memref store's and perform more complex forwarding when support for
// SSA scalars live out of 'affine.for'/'affine.if' statements is available.
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/Dominance.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <algorithm>

#define DEBUG_TYPE "memref-dataflow-opt"

using namespace mlir;

namespace {
// The store to load forwarding relies on three conditions:
//
// 1) they need to have mathematically equivalent affine access functions
// (checked after full composition of load/store operands); this implies that
// they access the same single memref element for all iterations of the common
// surrounding loop,
//
// 2) the store op should dominate the load op,
//
// 3) among all op's that satisfy both (1) and (2), the one that postdominates
// all store op's that have a dependence into the load, is provably the last
// writer to the particular memref location being loaded at the load op, and its
// store value can be forwarded to the load. Note that the only dependences
// that are to be considered are those that are satisfied at the block* of the
// innermost common surrounding loop of the <store, load> being considered.
//
// (* A dependence being satisfied at a block: a dependence that is satisfied by
// virtue of the destination operation appearing textually / lexically after
// the source operation within the body of a 'affine.for' operation; thus, a
// dependence is always either satisfied by a loop or by a block).
//
// The above conditions are simple to check, sufficient, and powerful for most
// cases in practice - they are sufficient, but not necessary --- since they
// don't reason about loops that are guaranteed to execute at least once or
// multiple sources to forward from.
//
// TODO(mlir-team): more forwarding can be done when support for
// loop/conditional live-out SSA values is available.
// TODO(mlir-team): do general dead store elimination for memref's. This pass
// currently only eliminates the stores only if no other loads/uses (other
// than dealloc) remain.
//
struct MemRefDataFlowOpt : public MemRefDataFlowOptBase<MemRefDataFlowOpt> {
  void runOnFunction() override;

  void forwardStoreToLoad(AffineLoadOp loadOp);

  // A list of memref's that are potentially dead / could be eliminated.
  SmallPtrSet<Value, 4> memrefsToErase;
  // Load op's whose results were replaced by those forwarded from stores.
  SmallVector<Operation *, 8> loadOpsToErase;

  DominanceInfo *domInfo = nullptr;
  PostDominanceInfo *postDomInfo = nullptr;
};

} // end anonymous namespace

/// Creates a pass to perform optimizations relying on memref dataflow such as
/// store to load forwarding, elimination of dead stores, and dead allocs.
std::unique_ptr<OperationPass<FuncOp>> mlir::createMemRefDataFlowOptPass() {
  return std::make_unique<MemRefDataFlowOpt>();
}

// This is a straightforward implementation not optimized for speed. Optimize
// if needed.
void MemRefDataFlowOpt::forwardStoreToLoad(AffineLoadOp loadOp) {
  // First pass over the use list to get the minimum number of surrounding
  // loops common between the load op and the store op, with min taken across
  // all store ops.
  SmallVector<Operation *, 8> storeOps;
  unsigned minSurroundingLoops = getNestingDepth(loadOp);
  for (auto *user : loadOp.getMemRef().getUsers()) {
    auto storeOp = dyn_cast<AffineStoreOp>(user);
    if (!storeOp)
      continue;
    unsigned nsLoops = getNumCommonSurroundingLoops(*loadOp, *storeOp);
    minSurroundingLoops = std::min(nsLoops, minSurroundingLoops);
    storeOps.push_back(storeOp);
  }

  // The list of store op candidates for forwarding that satisfy conditions
  // (1) and (2) above - they will be filtered later when checking (3).
  SmallVector<Operation *, 8> fwdingCandidates;

  // Store ops that have a dependence into the load (even if they aren't
  // forwarding candidates). Each forwarding candidate will be checked for a
  // post-dominance on these. 'fwdingCandidates' are a subset of depSrcStores.
  SmallVector<Operation *, 8> depSrcStores;

  for (auto *storeOp : storeOps) {
    MemRefAccess srcAccess(storeOp);
    MemRefAccess destAccess(loadOp);
    // Find stores that may be reaching the load.
    FlatAffineConstraints dependenceConstraints;
    unsigned nsLoops = getNumCommonSurroundingLoops(*loadOp, *storeOp);
    unsigned d;
    // Dependences at loop depth <= minSurroundingLoops do NOT matter.
    for (d = nsLoops + 1; d > minSurroundingLoops; d--) {
      DependenceResult result = checkMemrefAccessDependence(
          srcAccess, destAccess, d, &dependenceConstraints,
          /*dependenceComponents=*/nullptr);
      if (hasDependence(result))
        break;
    }
    if (d == minSurroundingLoops)
      continue;

    // Stores that *may* be reaching the load.
    depSrcStores.push_back(storeOp);

    // 1. Check if the store and the load have mathematically equivalent
    // affine access functions; this implies that they statically refer to the
    // same single memref element. As an example this filters out cases like:
    //     store %A[%i0 + 1]
    //     load %A[%i0]
    //     store %A[%M]
    //     load %A[%N]
    // Use the AffineValueMap difference based memref access equality checking.
    if (srcAccess != destAccess)
      continue;

    // 2. The store has to dominate the load op to be candidate.
    if (!domInfo->dominates(storeOp, loadOp))
      continue;

    // We now have a candidate for forwarding.
    fwdingCandidates.push_back(storeOp);
  }

  // 3. Of all the store op's that meet the above criteria, the store that
  // postdominates all 'depSrcStores' (if one exists) is the unique store
  // providing the value to the load, i.e., provably the last writer to that
  // memref loc.
  // Note: this can be implemented in a cleaner way with postdominator tree
  // traversals. Consider this for the future if needed.
  Operation *lastWriteStoreOp = nullptr;
  for (auto *storeOp : fwdingCandidates) {
    if (llvm::all_of(depSrcStores, [&](Operation *depStore) {
          return postDomInfo->postDominates(storeOp, depStore);
        })) {
      lastWriteStoreOp = storeOp;
      break;
    }
  }
  if (!lastWriteStoreOp)
    return;

  // Perform the actual store to load forwarding.
  Value storeVal = cast<AffineStoreOp>(lastWriteStoreOp).getValueToStore();
  loadOp.replaceAllUsesWith(storeVal);
  // Record the memref for a later sweep to optimize away.
  memrefsToErase.insert(loadOp.getMemRef());
  // Record this to erase later.
  loadOpsToErase.push_back(loadOp);
}

void MemRefDataFlowOpt::runOnFunction() {
  // Only supports single block functions at the moment.
  FuncOp f = getFunction();
  if (f.getBlocks().size() != 1) {
    markAllAnalysesPreserved();
    return;
  }

  domInfo = &getAnalysis<DominanceInfo>();
  postDomInfo = &getAnalysis<PostDominanceInfo>();

  loadOpsToErase.clear();
  memrefsToErase.clear();

  // Walk all load's and perform store to load forwarding.
  f.walk([&](AffineLoadOp loadOp) { forwardStoreToLoad(loadOp); });

  // Erase all load op's whose results were replaced with store fwd'ed ones.
  for (auto *loadOp : loadOpsToErase)
    loadOp->erase();

  // Check if the store fwd'ed memrefs are now left with only stores and can
  // thus be completely deleted. Note: the canonicalize pass should be able
  // to do this as well, but we'll do it here since we collected these anyway.
  for (auto memref : memrefsToErase) {
    // If the memref hasn't been alloc'ed in this function, skip.
    Operation *defOp = memref.getDefiningOp();
    if (!defOp || !isa<AllocOp>(defOp))
      // TODO(mlir-team): if the memref was returned by a 'call' operation, we
      // could still erase it if the call had no side-effects.
      continue;
    if (llvm::any_of(memref.getUsers(), [&](Operation *ownerOp) {
          return (!isa<AffineStoreOp>(ownerOp) && !isa<DeallocOp>(ownerOp));
        }))
      continue;

    // Erase all stores, the dealloc, and the alloc on the memref.
    for (auto *user : llvm::make_early_inc_range(memref.getUsers()))
      user->erase();
    defOp->erase();
  }
}
