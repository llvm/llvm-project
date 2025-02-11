//===- LoopFusion.cpp - Code to perform loop fusion -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements affine fusion.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Passes.h"

#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopFusionUtils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <iomanip>
#include <optional>
#include <sstream>

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_AFFINELOOPFUSION
#include "mlir/Dialect/Affine/Passes.h.inc"
} // namespace affine
} // namespace mlir

#define DEBUG_TYPE "affine-loop-fusion"

using namespace mlir;
using namespace mlir::affine;

namespace {
/// Loop fusion pass. This pass currently supports a greedy fusion policy,
/// which fuses loop nests with single-writer/single-reader memref dependences
/// with the goal of improving locality.
// TODO: Support fusion of source loop nests which write to multiple
// memrefs, where each memref can have multiple users (if profitable).
struct LoopFusion : public affine::impl::AffineLoopFusionBase<LoopFusion> {
  LoopFusion() = default;
  LoopFusion(unsigned fastMemorySpace, uint64_t localBufSizeThresholdBytes,
             bool maximalFusion, enum FusionMode affineFusionMode) {
    this->fastMemorySpace = fastMemorySpace;
    this->localBufSizeThreshold = localBufSizeThresholdBytes / 1024;
    this->maximalFusion = maximalFusion;
    this->affineFusionMode = affineFusionMode;
  }

  void runOnBlock(Block *block);
  void runOnOperation() override;
};

} // namespace

/// Returns true if node 'srcId' can be removed after fusing it with node
/// 'dstId'. The node can be removed if any of the following conditions are met:
///   1. 'srcId' has no output dependences after fusion and no escaping memrefs.
///   2. 'srcId' has no output dependences after fusion, has escaping memrefs
///       and the fusion slice is maximal.
///   3. 'srcId' has output dependences after fusion, the fusion slice is
///      maximal and the fusion insertion point dominates all the dependences.
static bool canRemoveSrcNodeAfterFusion(
    unsigned srcId, unsigned dstId, const ComputationSliceState &fusionSlice,
    Operation *fusedLoopInsPoint, const DenseSet<Value> &escapingMemRefs,
    MemRefDependenceGraph *mdg) {

  Operation *dstNodeOp = mdg->getNode(dstId)->op;
  bool hasOutDepsAfterFusion = false;

  for (auto &outEdge : mdg->outEdges[srcId]) {
    Operation *depNodeOp = mdg->getNode(outEdge.id)->op;
    // Skip dependence with dstOp since it will be removed after fusion.
    if (depNodeOp == dstNodeOp)
      continue;

    // Only fusion within the same block is supported. Use domination analysis
    // when needed.
    if (depNodeOp->getBlock() != dstNodeOp->getBlock())
      return false;

    // Check if the insertion point of the fused loop dominates the dependence.
    // Otherwise, the src loop can't be removed.
    if (fusedLoopInsPoint != depNodeOp &&
        !fusedLoopInsPoint->isBeforeInBlock(depNodeOp)) {
      LLVM_DEBUG(llvm::dbgs() << "Src loop can't be removed: dst loop doesn't "
                                 "dominate dependence\n");
      return false;
    }

    hasOutDepsAfterFusion = true;
  }

  // If src loop has dependences after fusion or it writes to an live-out or
  // escaping memref, we can only remove it if the fusion slice is maximal so
  // that all the dependences are preserved.
  if (hasOutDepsAfterFusion || !escapingMemRefs.empty()) {
    std::optional<bool> isMaximal = fusionSlice.isMaximal();
    if (!isMaximal) {
      LLVM_DEBUG(llvm::dbgs() << "Src loop can't be removed: can't determine "
                                 "if fusion is maximal\n");
      return false;
    }

    if (!*isMaximal) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Src loop can't be removed: fusion is not maximal\n");
      return false;
    }
  }

  return true;
}

/// Returns in 'srcIdCandidates' the producer fusion candidates for consumer
/// 'dstId'. Candidates are sorted by node id order. This order corresponds to
/// the program order when the 'mdg' is created. However, program order is not
/// guaranteed and must not be required by the client. Program order won't be
/// held if the 'mdg' is reused from a previous fusion step or if the node
/// creation order changes in the future to support more advance cases.
// TODO: Move this to a loop fusion utility once 'mdg' is also moved.
static void getProducerCandidates(unsigned dstId, MemRefDependenceGraph *mdg,
                                  SmallVectorImpl<unsigned> &srcIdCandidates) {
  // Skip if no input edges along which to fuse.
  if (mdg->inEdges.count(dstId) == 0)
    return;

  // Gather memrefs from loads in 'dstId'.
  auto *dstNode = mdg->getNode(dstId);
  DenseSet<Value> consumedMemrefs;
  for (Operation *load : dstNode->loads)
    consumedMemrefs.insert(cast<AffineReadOpInterface>(load).getMemRef());

  // Traverse 'dstId' incoming edges and gather the nodes that contain a store
  // to one of the consumed memrefs.
  for (auto &srcEdge : mdg->inEdges[dstId]) {
    auto *srcNode = mdg->getNode(srcEdge.id);
    // Skip if 'srcNode' is not a loop nest.
    if (!isa<AffineForOp>(srcNode->op))
      continue;

    if (any_of(srcNode->stores, [&](Operation *op) {
          auto storeOp = cast<AffineWriteOpInterface>(op);
          return consumedMemrefs.count(storeOp.getMemRef()) > 0;
        }))
      srcIdCandidates.push_back(srcNode->id);
  }

  llvm::sort(srcIdCandidates);
  srcIdCandidates.erase(llvm::unique(srcIdCandidates), srcIdCandidates.end());
}

/// Returns in 'producerConsumerMemrefs' the memrefs involved in a
/// producer-consumer dependence between 'srcId' and 'dstId'.
static void
gatherProducerConsumerMemrefs(unsigned srcId, unsigned dstId,
                              MemRefDependenceGraph *mdg,
                              DenseSet<Value> &producerConsumerMemrefs) {
  auto *dstNode = mdg->getNode(dstId);
  auto *srcNode = mdg->getNode(srcId);
  gatherProducerConsumerMemrefs(srcNode->stores, dstNode->loads,
                                producerConsumerMemrefs);
}

/// A memref escapes in the context of the fusion pass if either:
///   1. it (or its alias) is a block argument, or
///   2. created by an op not known to guarantee alias freedom,
///   3. it (or its alias) are used by ops other than affine dereferencing ops
///   (e.g., by call op, memref load/store ops, alias creating ops, unknown ops,
///   terminator ops, etc.); such ops do not deference the memref in an affine
///   way.
static bool isEscapingMemref(Value memref, Block *block) {
  Operation *defOp = memref.getDefiningOp();
  // Check if 'memref' is a block argument.
  if (!defOp)
    return true;

  // Check if this is defined to be an alias of another memref.
  if (auto viewOp = dyn_cast<mlir::ViewLikeOpInterface>(defOp))
    if (isEscapingMemref(viewOp.getViewSource(), block))
      return true;

  // Any op besides allocating ops wouldn't guarantee alias freedom
  if (!hasSingleEffect<mlir::MemoryEffects::Allocate>(defOp, memref))
    return true;

  // Check if 'memref' is used by a non-deferencing op (including unknown ones)
  // (e.g., call ops, alias creating ops, etc.).
  return llvm::any_of(memref.getUsers(), [&](Operation *user) {
    // Ignore users outside of `block`.
    Operation *ancestorOp = block->getParent()->findAncestorOpInRegion(*user);
    if (!ancestorOp)
      return true;
    if (ancestorOp->getBlock() != block)
      return false;
    return !isa<AffineMapAccessInterface>(*user);
  });
}

/// Returns in 'escapingMemRefs' the memrefs from affine store ops in node 'id'
/// that escape the block or are accessed in a non-affine way.
static void gatherEscapingMemrefs(unsigned id, MemRefDependenceGraph *mdg,
                                  DenseSet<Value> &escapingMemRefs) {
  auto *node = mdg->getNode(id);
  for (Operation *storeOp : node->stores) {
    auto memref = cast<AffineWriteOpInterface>(storeOp).getMemRef();
    if (escapingMemRefs.count(memref))
      continue;
    if (isEscapingMemref(memref, &mdg->block))
      escapingMemRefs.insert(memref);
  }
}

// Sinks all sequential loops to the innermost levels (while preserving
// relative order among them) and moves all parallel loops to the
// outermost (while again preserving relative order among them).
// This can increase the loop depth at which we can fuse a slice, since we are
// pushing loop carried dependence to a greater depth in the loop nest.
static void sinkSequentialLoops(MemRefDependenceGraph::Node *node) {
  assert(isa<AffineForOp>(node->op));
  AffineForOp newRootForOp = sinkSequentialLoops(cast<AffineForOp>(node->op));
  node->op = newRootForOp;
}

// Creates and returns a private (single-user) memref for fused loop rooted
// at 'forOp', with (potentially reduced) memref size based on the
// MemRefRegion written to by 'srcStoreOpInst' at depth 'dstLoopDepth'.
// TODO: consider refactoring the common code from generateDma and
// this one.
static Value createPrivateMemRef(AffineForOp forOp, Operation *srcStoreOpInst,
                                 unsigned dstLoopDepth,
                                 std::optional<unsigned> fastMemorySpace,
                                 uint64_t localBufSizeThreshold) {
  Operation *forInst = forOp.getOperation();

  // Create builder to insert alloc op just before 'forOp'.
  OpBuilder b(forInst);
  // Builder to create constants at the top level.
  OpBuilder top(forInst->getParentRegion());
  // Create new memref type based on slice bounds.
  auto oldMemRef = cast<AffineWriteOpInterface>(srcStoreOpInst).getMemRef();
  auto oldMemRefType = cast<MemRefType>(oldMemRef.getType());
  unsigned rank = oldMemRefType.getRank();

  // Compute MemRefRegion for 'srcStoreOpInst' at depth 'dstLoopDepth'.
  MemRefRegion region(srcStoreOpInst->getLoc());
  bool validRegion = succeeded(region.compute(srcStoreOpInst, dstLoopDepth));
  (void)validRegion;
  assert(validRegion && "unexpected memref region failure");
  SmallVector<int64_t, 4> newShape;
  std::vector<SmallVector<int64_t, 4>> lbs;
  SmallVector<int64_t, 8> lbDivisors;
  lbs.reserve(rank);
  // Query 'region' for 'newShape' and lower bounds of MemRefRegion accessed
  // by 'srcStoreOpInst' at depth 'dstLoopDepth'.
  std::optional<int64_t> numElements =
      region.getConstantBoundingSizeAndShape(&newShape, &lbs, &lbDivisors);
  assert(numElements && "non-constant number of elts in local buffer");

  const FlatAffineValueConstraints *cst = region.getConstraints();
  // 'outerIVs' holds the values that this memory region is symbolic/parametric
  // on; this would correspond to loop IVs surrounding the level at which the
  // slice is being materialized.
  SmallVector<Value, 8> outerIVs;
  cst->getValues(rank, cst->getNumVars(), &outerIVs);

  // Build 'rank' AffineExprs from MemRefRegion 'lbs'
  SmallVector<AffineExpr, 4> offsets;
  offsets.reserve(rank);
  for (unsigned d = 0; d < rank; ++d) {
    assert(lbs[d].size() == cst->getNumCols() - rank && "incorrect bound size");

    AffineExpr offset = top.getAffineConstantExpr(0);
    for (unsigned j = 0, e = cst->getNumCols() - rank - 1; j < e; j++) {
      offset = offset + lbs[d][j] * top.getAffineDimExpr(j);
    }
    assert(lbDivisors[d] > 0);
    offset =
        (offset + lbs[d][cst->getNumCols() - 1 - rank]).floorDiv(lbDivisors[d]);
    offsets.push_back(offset);
  }

  // Create 'newMemRefType' using 'newShape' from MemRefRegion accessed
  // by 'srcStoreOpInst'.
  auto eltSize = getMemRefIntOrFloatEltSizeInBytes(oldMemRefType);
  assert(eltSize && "memrefs with size elt types expected");
  uint64_t bufSize = *eltSize * *numElements;
  unsigned newMemSpace;
  if (bufSize <= localBufSizeThreshold && fastMemorySpace.has_value()) {
    newMemSpace = *fastMemorySpace;
  } else {
    newMemSpace = oldMemRefType.getMemorySpaceAsInt();
  }
  auto newMemRefType = MemRefType::get(newShape, oldMemRefType.getElementType(),
                                       {}, newMemSpace);

  // Create new private memref for fused loop 'forOp'. 'newShape' is always
  // a constant shape.
  // TODO: Create/move alloc ops for private memrefs closer to their
  // consumer loop nests to reduce their live range. Currently they are added
  // at the beginning of the block, because loop nests can be reordered
  // during the fusion pass.
  Value newMemRef = top.create<memref::AllocOp>(forOp.getLoc(), newMemRefType);

  // Build an AffineMap to remap access functions based on lower bound offsets.
  SmallVector<AffineExpr, 4> remapExprs;
  remapExprs.reserve(rank);
  for (unsigned i = 0; i < rank; i++) {
    auto dimExpr = b.getAffineDimExpr(outerIVs.size() + i);

    auto remapExpr =
        simplifyAffineExpr(dimExpr - offsets[i], outerIVs.size() + rank, 0);
    remapExprs.push_back(remapExpr);
  }

  auto indexRemap =
      AffineMap::get(outerIVs.size() + rank, 0, remapExprs, forOp.getContext());

  // Replace all users of 'oldMemRef' with 'newMemRef'.
  LogicalResult res =
      replaceAllMemRefUsesWith(oldMemRef, newMemRef, {}, indexRemap,
                               /*extraOperands=*/outerIVs,
                               /*symbolOperands=*/{},
                               /*domOpFilter=*/&*forOp.getBody()->begin());
  assert(succeeded(res) &&
         "replaceAllMemrefUsesWith should always succeed here");
  (void)res;
  return newMemRef;
}

// Checks the profitability of fusing a backwards slice of the loop nest
// surrounding 'srcOpInst' into the loop nest surrounding 'dstLoadOpInsts'.
// The argument 'srcStoreOpInst' is used to calculate the storage reduction on
// the memref being produced and consumed, which is an input to the cost model.
// For producer-consumer fusion, 'srcStoreOpInst' will be the same as
// 'srcOpInst', as we are slicing w.r.t to that producer. For input-reuse
// fusion, 'srcOpInst' will be the src loop nest LoadOp which reads from the
// same memref as dst loop nest load ops, and 'srcStoreOpInst' will be the
// unique store op in the src node, which will be used to check that the write
// region is the same after input-reuse fusion. Computation slices are provided
// in 'depthSliceUnions' for each legal fusion depth. The maximal depth at which
// fusion is legal is provided in 'maxLegalFusionDepth'. Returns true if it is
// profitable to fuse the candidate loop nests. Returns false otherwise.
// `dstLoopDepth` is set to the most profitable depth at which to materialize
// the source loop nest slice.
// The profitability model executes the following steps:
// *) Computes the backward computation slice at 'srcOpInst'. This
//    computation slice of the loop nest surrounding 'srcOpInst' is
//    represented by modified src loop bounds in 'sliceState', which are
//    functions of loop IVs in the loop nest surrounding 'srcOpInst'.
// *) Computes the cost of unfused src/dst loop nests (currently the cost of a
//    loop nest is the total number of dynamic operation instances in the loop
//    nest).
// *) Computes the cost of fusing a slice of the src loop nest into the dst
//    loop nest at various values of dst loop depth, attempting to fuse
//    the largest computation slice at the maximal dst loop depth (closest to
//    the load) to minimize reuse distance and potentially enable subsequent
//    load/store forwarding.
//    NOTE: 'dstLoopDepth' refers to the loop depth within the destination loop
//    nest, at which the src computation slice is inserted/fused.
//    NOTE: We attempt to maximize the dst loop depth, but there are cases
//    where a particular setting for 'dstLoopNest' might fuse an unsliced
//    loop (within the src computation slice) at a depth which results in
//    excessive recomputation (see unit tests for examples).
// *) Compares the total cost of the unfused loop nests to the min cost fused
//    loop nest computed in the previous step, and returns true if the latter
//    is lower.
// TODO: Extend profitability analysis to support scenarios with multiple
// stores.
static bool isFusionProfitable(Operation *srcOpInst, Operation *srcStoreOpInst,
                               AffineForOp dstForOp,
                               ArrayRef<ComputationSliceState> depthSliceUnions,
                               unsigned maxLegalFusionDepth,
                               unsigned *dstLoopDepth,
                               double computeToleranceThreshold) {
  LLVM_DEBUG({
    llvm::dbgs() << "Checking whether fusion is profitable between src op:\n";
    llvm::dbgs() << ' ' << *srcOpInst << " and destination loop:\n";
    llvm::dbgs() << dstForOp << "\n";
  });

  if (maxLegalFusionDepth == 0) {
    LLVM_DEBUG(llvm::dbgs() << "Can't fuse: maxLegalFusionDepth is 0\n");
    return false;
  }

  // Compute cost of sliced and unsliced src loop nest.
  SmallVector<AffineForOp, 4> srcLoopIVs;
  getAffineForIVs(*srcOpInst, &srcLoopIVs);

  // Walk src loop nest and collect stats.
  LoopNestStats srcLoopNestStats;
  if (!getLoopNestStats(srcLoopIVs[0], &srcLoopNestStats))
    return false;

  // Compute cost of dst loop nest.
  LoopNestStats dstLoopNestStats;
  if (!getLoopNestStats(dstForOp, &dstLoopNestStats))
    return false;

  // Search for min cost value for 'dstLoopDepth'. At each value of
  // 'dstLoopDepth' from 'maxLegalLoopDepth' to '1', compute computation slice
  // bounds between 'srcOpInst' and each op in 'dstOpinsts' (taking the union
  // of these bounds). Next the union slice bounds are used to calculate
  // the cost of the slice and the cost of the slice inserted into the dst
  // loop nest at 'dstLoopDepth'.
  uint64_t minFusedLoopNestComputeCost = std::numeric_limits<uint64_t>::max();
  double maxStorageReduction = 0.0;
  std::optional<uint64_t> sliceMemEstimate;

  // The best loop depth at which to materialize the slice.
  std::optional<unsigned> bestDstLoopDepth;

  // Compute op instance count for the src loop nest without iteration slicing.
  uint64_t srcLoopNestCost = getComputeCost(srcLoopIVs[0], srcLoopNestStats);

  // Compute src loop nest write region size.
  MemRefRegion srcWriteRegion(srcStoreOpInst->getLoc());
  if (failed(srcWriteRegion.compute(srcStoreOpInst, /*loopDepth=*/0))) {
    LLVM_DEBUG(llvm::dbgs()
               << "Unable to compute MemRefRegion for source operation\n");
    return false;
  }

  std::optional<int64_t> maybeSrcWriteRegionSizeBytes =
      srcWriteRegion.getRegionSize();
  if (!maybeSrcWriteRegionSizeBytes.has_value())
    return false;
  int64_t srcWriteRegionSizeBytes = *maybeSrcWriteRegionSizeBytes;

  // Compute op instance count for the src loop nest.
  uint64_t dstLoopNestCost = getComputeCost(dstForOp, dstLoopNestStats);

  // Evaluate all depth choices for materializing the slice in the destination
  // loop nest.
  for (unsigned i = maxLegalFusionDepth; i >= 1; --i) {
    const ComputationSliceState &slice = depthSliceUnions[i - 1];
    // Skip slice union if it wasn't computed for this depth.
    if (slice.isEmpty())
      continue;

    int64_t fusedLoopNestComputeCost;
    if (!getFusionComputeCost(srcLoopIVs[0], srcLoopNestStats, dstForOp,
                              dstLoopNestStats, slice,
                              &fusedLoopNestComputeCost)) {
      LLVM_DEBUG(llvm::dbgs() << "Unable to compute fusion compute cost\n");
      continue;
    }

    double additionalComputeFraction =
        fusedLoopNestComputeCost /
            (static_cast<double>(srcLoopNestCost) + dstLoopNestCost) -
        1;

    // Determine what the slice write MemRefRegion would be, if the src loop
    // nest slice 'slice' were to be inserted into the dst loop nest at loop
    // depth 'i'.
    MemRefRegion sliceWriteRegion(srcStoreOpInst->getLoc());
    if (failed(sliceWriteRegion.compute(srcStoreOpInst, /*loopDepth=*/0,
                                        &slice))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to compute slice write region at loopDepth: " << i
                 << "\n");
      continue;
    }

    std::optional<int64_t> maybeSliceWriteRegionSizeBytes =
        sliceWriteRegion.getRegionSize();
    if (!maybeSliceWriteRegionSizeBytes.has_value() ||
        *maybeSliceWriteRegionSizeBytes == 0) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to get slice write region size at loopDepth: " << i
                 << "\n");
      continue;
    }
    int64_t sliceWriteRegionSizeBytes = *maybeSliceWriteRegionSizeBytes;

    // If we are fusing for reuse, check that write regions remain the same.
    // TODO: Write region check should check sizes and offsets in
    // each dimension, so that we are sure they are covering the same memref
    // region. Also, move this out to a isMemRefRegionSuperSet helper function.
    if (srcOpInst != srcStoreOpInst &&
        sliceWriteRegionSizeBytes != srcWriteRegionSizeBytes)
      continue;

    double storageReduction = static_cast<double>(srcWriteRegionSizeBytes) /
                              static_cast<double>(sliceWriteRegionSizeBytes);

    LLVM_DEBUG({
      std::stringstream msg;
      msg << "  evaluating fusion profitability at depth : " << i << "\n"
          << std::fixed << std::setprecision(2)
          << "   additional compute fraction: "
          << 100.0 * additionalComputeFraction << "%\n"
          << "   storage reduction factor: " << storageReduction << "x\n"
          << "   fused nest cost: " << fusedLoopNestComputeCost << "\n"
          << "   src write region size: " << srcWriteRegionSizeBytes << "\n"
          << "   slice write region size: " << sliceWriteRegionSizeBytes
          << "\n";
      llvm::dbgs() << msg.str();
    });

    // TODO: This is a placeholder cost model.
    // Among all choices that add an acceptable amount of redundant computation
    // (as per computeToleranceThreshold), we will simply pick the one that
    // reduces the intermediary size the most.
    if ((storageReduction > maxStorageReduction) &&
        (additionalComputeFraction < computeToleranceThreshold)) {
      maxStorageReduction = storageReduction;
      bestDstLoopDepth = i;
      minFusedLoopNestComputeCost = fusedLoopNestComputeCost;
      sliceMemEstimate = sliceWriteRegionSizeBytes;
    }
  }

  // A simple cost model: fuse if it reduces the memory footprint.

  if (!bestDstLoopDepth) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "All fusion choices involve more than the threshold amount of "
           "redundant computation; NOT fusing.\n");
    return false;
  }

  if (!bestDstLoopDepth) {
    LLVM_DEBUG(llvm::dbgs() << "no fusion depth could be evaluated.\n");
    return false;
  }

  // Set dstLoopDepth based on best values from search.
  *dstLoopDepth = *bestDstLoopDepth;

  LLVM_DEBUG(
      llvm::dbgs() << " LoopFusion fusion stats:"
                   << "\n  best loop depth: " << bestDstLoopDepth
                   << "\n  src loop nest compute cost: " << srcLoopNestCost
                   << "\n  dst loop nest compute cost: " << dstLoopNestCost
                   << "\n  fused loop nest compute cost: "
                   << minFusedLoopNestComputeCost << "\n");

  auto dstMemSize = getMemoryFootprintBytes(dstForOp);
  auto srcMemSize = getMemoryFootprintBytes(srcLoopIVs[0]);

  std::optional<double> storageReduction;

  if (!dstMemSize || !srcMemSize) {
    LLVM_DEBUG(llvm::dbgs()
               << "  fusion memory benefit cannot be evaluated; NOT fusing.\n");
    return false;
  }

  auto srcMemSizeVal = *srcMemSize;
  auto dstMemSizeVal = *dstMemSize;

  assert(sliceMemEstimate && "expected value");
  auto fusedMem = dstMemSizeVal + *sliceMemEstimate;

  LLVM_DEBUG(llvm::dbgs() << "   src mem: " << srcMemSizeVal << "\n"
                          << "   dst mem: " << dstMemSizeVal << "\n"
                          << "   fused mem: " << fusedMem << "\n"
                          << "   slice mem: " << sliceMemEstimate << "\n");

  if (static_cast<long>(fusedMem) > srcMemSizeVal + dstMemSizeVal) {
    LLVM_DEBUG(llvm::dbgs() << "Fusion is not profitable; NOT fusing.\n");
    return false;
  }
  storageReduction =
      100.0 *
      (1.0 - fusedMem / (static_cast<double>(srcMemSizeVal) + dstMemSizeVal));

  double additionalComputeFraction =
      100.0 * (minFusedLoopNestComputeCost /
                   (static_cast<double>(srcLoopNestCost) + dstLoopNestCost) -
               1);
  (void)additionalComputeFraction;
  LLVM_DEBUG({
    std::stringstream msg;
    msg << " fusion is most profitable at depth " << *dstLoopDepth << " with "
        << std::setprecision(2) << additionalComputeFraction
        << "% redundant computation and a ";
    msg << (storageReduction ? std::to_string(*storageReduction) : "<unknown>");
    msg << "% storage reduction.\n";
    llvm::dbgs() << msg.str();
  });

  return true;
}

namespace {

// GreedyFusion greedily fuses loop nests which have a producer/consumer or
// input-reuse relationship on a memref, with the goal of improving locality.
//
// The steps of the producer-consumer fusion algorithm are as follows:
//
// *) A worklist is initialized with node ids from the dependence graph.
// *) For each node id in the worklist:
//   *) Pop an AffineForOp of the worklist. This 'dstAffineForOp' will be a
//      candidate destination AffineForOp into which fusion will be attempted.
//   *) Add each LoadOp currently in 'dstAffineForOp' into list 'dstLoadOps'.
//   *) For each LoadOp in 'dstLoadOps' do:
//      *) Look up dependent loop nests which have a single store op to the same
//         memref.
//      *) Check if dependences would be violated by the fusion.
//      *) Get a computation slice of 'srcLoopNest', which adjusts its loop
//         bounds to be functions of 'dstLoopNest' IVs and symbols.
//      *) Fuse the 'srcLoopNest' computation slice into the 'dstLoopNest',
//         at a loop depth determined by the cost model in 'isFusionProfitable'.
//      *) Add the newly fused load/store operations to the state,
//         and also add newly fused load ops to 'dstLoopOps' to be considered
//         as fusion dst load ops in another iteration.
//      *) Remove old src loop nest and its associated state.
//
// The steps of the input-reuse fusion algorithm are as follows:
//
// *) Initialize 'worklist' with node ids from the dependence graph.
// *) For each 'dstNode' in the worklist:
//   *) Find a candidate sibling node 'sibNode' to fuse with 'dstNode' which
//      loads from the same memref, but which has no dependence paths to/from.
//   *) Get a computation slice of 'sibLoopNest', which adjusts its loop
//      bounds to be functions of 'dstLoopNest' IVs and symbols.
//   *) Fuse the 'sibLoopNest' computation slice into the 'dstLoopNest',
//      at a loop depth determined by the cost model in 'isFusionProfitable'.
//      This function also checks that the memref write region of 'sibLoopNest',
//      is preserved in the fused loop nest.
//   *) Update graph state to reflect the fusion of 'sibNode' into 'dstNode'.
//
// Given a graph where top-level operations are vertices in the set 'V' and
// edges in the set 'E' are dependences between vertices, this algorithm
// takes O(V) time for initialization, and has runtime O(V + E).
//
// This greedy algorithm is not 'maximal' due to the current restriction of
// fusing along single producer consumer edges, but there is a TODO: to fix
// this.
//
// TODO: Experiment with other fusion policies.
struct GreedyFusion {
public:
  // The data dependence graph to traverse during fusion.
  MemRefDependenceGraph *mdg;
  // Worklist of graph nodes visited during the fusion pass.
  SmallVector<unsigned, 8> worklist;
  // Parameter for local buffer size threshold.
  unsigned localBufSizeThreshold;
  // Parameter for fast memory space.
  std::optional<unsigned> fastMemorySpace;
  // If true, ignore any additional (redundant) computation tolerance threshold
  // that would have prevented fusion.
  bool maximalFusion;
  // The amount of additional computation that is tolerated while fusing
  // pair-wise as a fraction of the total computation.
  double computeToleranceThreshold;

  using Node = MemRefDependenceGraph::Node;

  GreedyFusion(MemRefDependenceGraph *mdg, unsigned localBufSizeThreshold,
               std::optional<unsigned> fastMemorySpace, bool maximalFusion,
               double computeToleranceThreshold)
      : mdg(mdg), localBufSizeThreshold(localBufSizeThreshold),
        fastMemorySpace(fastMemorySpace), maximalFusion(maximalFusion),
        computeToleranceThreshold(computeToleranceThreshold) {}

  /// Initializes 'worklist' with nodes from 'mdg'.
  void init() {
    // TODO: Add a priority queue for prioritizing nodes by different
    // metrics (e.g. arithmetic intensity/flops-to-bytes ratio).
    worklist.clear();
    for (auto &idAndNode : mdg->nodes) {
      const Node &node = idAndNode.second;
      worklist.push_back(node.id);
    }
  }
  /// Run only sibling fusion on the `mdg`.
  void runSiblingFusionOnly() {
    fuseSiblingNodes();
    eraseUnusedMemRefAllocations();
  }

  /// Run only producer/consumer fusion on the `mdg`.
  void runProducerConsumerFusionOnly() {
    fuseProducerConsumerNodes(
        /*maxSrcUserCount=*/std::numeric_limits<unsigned>::max());
    eraseUnusedMemRefAllocations();
  }

  // Run the GreedyFusion pass.
  // *) First pass through the nodes fuses single-use producer nodes into their
  //    unique consumer.
  // *) Second pass fuses sibling nodes which share no dependence edges.
  // *) Third pass fuses any remaining producer nodes into their users.
  void runGreedyFusion() {
    // TODO: Run this repeatedly until a fixed-point is reached.
    fuseProducerConsumerNodes(/*maxSrcUserCount=*/1);
    fuseSiblingNodes();
    fuseProducerConsumerNodes(
        /*maxSrcUserCount=*/std::numeric_limits<unsigned>::max());
    eraseUnusedMemRefAllocations();
  }

  /// Returns true if a private memref can be created for `memref` given
  /// the fusion scenario reflected by the other arguments.
  bool canCreatePrivateMemRef(Value memref,
                              const DenseSet<Value> &srcEscapingMemRefs,
                              unsigned producerId, unsigned consumerId,
                              bool removeSrcNode) {
    const Node *consumerNode = mdg->getNode(consumerId);
    // If `memref` is an escaping one, do not create a private memref
    // for the below scenarios, since doing so will leave the escaping
    // memref unmodified as all the writes originally meant for the
    // escaping memref would be performed on the private memref:
    // 1. The source is to be removed after fusion,
    // OR
    // 2. The destination writes to `memref`.
    if (srcEscapingMemRefs.count(memref) > 0 &&
        (removeSrcNode || consumerNode->getStoreOpCount(memref) > 0))
      return false;

    // Don't create a private memref if 'srcNode' has in edges on
    // 'memref' or 'dstNode' has out edges on 'memref'.
    if (mdg->getIncomingMemRefAccesses(producerId, memref) > 0 ||
        mdg->getOutEdgeCount(consumerId, memref) > 0)
      return false;

    // If 'srcNode' will be removed but it has out edges on 'memref' to
    // nodes other than 'dstNode', we have to preserve dependences and
    // cannot create a private memref.
    if (removeSrcNode &&
        any_of(mdg->outEdges[producerId], [&](const auto &edge) {
          return edge.value == memref && edge.id != consumerId;
        }))
      return false;

    return true;
  }

  /// Perform fusions with node `dstId` as the destination of fusion, with
  /// No fusion is performed when producers with a user count greater than
  /// `maxSrcUserCount` for any of the memrefs involved.
  void performFusionsIntoDest(unsigned dstId, unsigned maxSrcUserCount) {
    LLVM_DEBUG(llvm::dbgs() << "Evaluating dst loop " << dstId << "\n");
    // Skip if this node was removed (fused into another node).
    if (mdg->nodes.count(dstId) == 0)
      return;
    // Get 'dstNode' into which to attempt fusion.
    auto *dstNode = mdg->getNode(dstId);
    // Skip if 'dstNode' is not a loop nest.
    if (!isa<AffineForOp>(dstNode->op))
      return;
    // Skip if 'dstNode' is a loop nest returning values.
    // TODO: support loop nests that return values.
    if (dstNode->op->getNumResults() > 0)
      return;

    LLVM_DEBUG(llvm::dbgs() << "Evaluating dst loop " << dstId << "\n");

    // Sink sequential loops in 'dstNode' (and thus raise parallel loops)
    // while preserving relative order. This can increase the maximum loop
    // depth at which we can fuse a slice of a producer loop nest into a
    // consumer loop nest.
    sinkSequentialLoops(dstNode);
    auto dstAffineForOp = cast<AffineForOp>(dstNode->op);

    // Try to fuse 'dstNode' with candidate producer loops until a fixed point
    // is reached. Fusing two loops may expose new fusion opportunities.
    bool dstNodeChanged;
    do {
      // Gather src loop candidates for 'dstNode' and visit them in "quasi"
      // reverse program order to minimize the number of iterations needed to
      // reach the fixed point. Note that this is a best effort approach since
      // 'getProducerCandidates' does not always guarantee that program order
      // in 'srcIdCandidates'.
      dstNodeChanged = false;
      SmallVector<unsigned, 16> srcIdCandidates;
      getProducerCandidates(dstId, mdg, srcIdCandidates);

      for (unsigned srcId : llvm::reverse(srcIdCandidates)) {
        // Get 'srcNode' from which to attempt fusion into 'dstNode'.
        auto *srcNode = mdg->getNode(srcId);
        auto srcAffineForOp = cast<AffineForOp>(srcNode->op);
        LLVM_DEBUG(llvm::dbgs() << "Evaluating src loop " << srcId
                                << " for dst loop " << dstId << "\n");

        // Skip if 'srcNode' is a loop nest returning values.
        // TODO: support loop nests that return values.
        if (isa<AffineForOp>(srcNode->op) && srcNode->op->getNumResults() > 0)
          continue;

        DenseSet<Value> producerConsumerMemrefs;
        gatherProducerConsumerMemrefs(srcId, dstId, mdg,
                                      producerConsumerMemrefs);

        // Skip if 'srcNode' out edge count on any memref is greater than
        // 'maxSrcUserCount'.
        if (any_of(producerConsumerMemrefs, [&](Value memref) {
              return mdg->getOutEdgeCount(srcNode->id, memref) >
                     maxSrcUserCount;
            }))
          continue;

        // Gather memrefs in 'srcNode' that are written and escape out of the
        // block (e.g., memref block arguments, returned memrefs,
        // memrefs passed to function calls, etc.).
        DenseSet<Value> srcEscapingMemRefs;
        gatherEscapingMemrefs(srcNode->id, mdg, srcEscapingMemRefs);

        // Compute an operation list insertion point for the fused loop
        // nest which preserves dependences.
        Operation *fusedLoopInsPoint =
            mdg->getFusedLoopNestInsertionPoint(srcNode->id, dstNode->id);
        if (fusedLoopInsPoint == nullptr)
          continue;

        // It's possible this fusion is at an inner depth (i.e., there are
        // common surrounding affine loops for the source and destination for
        // ops). We need to get this number because the call to canFuseLoops
        // needs to be passed the absolute depth. The max legal depth and the
        // depths we try below are however *relative* and as such don't include
        // the common depth.
        SmallVector<AffineForOp, 4> surroundingLoops;
        getAffineForIVs(*dstAffineForOp, &surroundingLoops);
        unsigned numSurroundingLoops = surroundingLoops.size();

        // Compute the innermost common loop depth for dstNode
        // producer-consumer loads/stores.
        SmallVector<Operation *, 2> dstMemrefOps;
        for (Operation *op : dstNode->loads)
          if (producerConsumerMemrefs.count(
                  cast<AffineReadOpInterface>(op).getMemRef()) > 0)
            dstMemrefOps.push_back(op);
        for (Operation *op : dstNode->stores)
          if (producerConsumerMemrefs.count(
                  cast<AffineWriteOpInterface>(op).getMemRef()))
            dstMemrefOps.push_back(op);
        unsigned dstLoopDepthTest =
            getInnermostCommonLoopDepth(dstMemrefOps) - numSurroundingLoops;

        // Check the feasibility of fusing src loop nest into dst loop nest
        // at loop depths in range [1, dstLoopDepthTest].
        unsigned maxLegalFusionDepth = 0;
        SmallVector<ComputationSliceState, 8> depthSliceUnions;
        depthSliceUnions.resize(dstLoopDepthTest);
        FusionStrategy strategy(FusionStrategy::ProducerConsumer);
        for (unsigned i = 1; i <= dstLoopDepthTest; ++i) {
          FusionResult result =
              affine::canFuseLoops(srcAffineForOp, dstAffineForOp,
                                   /*dstLoopDepth=*/i + numSurroundingLoops,
                                   &depthSliceUnions[i - 1], strategy);

          if (result.value == FusionResult::Success)
            maxLegalFusionDepth = i;
        }

        if (maxLegalFusionDepth == 0) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Can't fuse: fusion is not legal at any depth\n");
          continue;
        }

        // Check if fusion would be profitable. We skip profitability analysis
        // for maximal fusion since we already know the maximal legal depth to
        // fuse.
        unsigned bestDstLoopDepth = maxLegalFusionDepth;
        if (!maximalFusion) {
          // Retrieve producer stores from the src loop.
          SmallVector<Operation *, 2> producerStores;
          for (Operation *op : srcNode->stores)
            if (producerConsumerMemrefs.count(
                    cast<AffineWriteOpInterface>(op).getMemRef()))
              producerStores.push_back(op);

          // TODO: Suppport multiple producer stores in profitability
          // analysis. We limit profitability analysis to only scenarios with
          // a single producer store for now. Note that some multi-store
          // producer scenarios will still go through profitability analysis
          // if only one of the stores is involved the producer-consumer
          // relationship of the candidate loops.
          assert(!producerStores.empty() && "Expected producer store");
          if (producerStores.size() > 1)
            LLVM_DEBUG(llvm::dbgs() << "Skipping profitability analysis. Not "
                                       "supported for this case\n");
          else if (!isFusionProfitable(producerStores[0], producerStores[0],
                                       dstAffineForOp, depthSliceUnions,
                                       maxLegalFusionDepth, &bestDstLoopDepth,
                                       computeToleranceThreshold))
            continue;
        }

        assert(bestDstLoopDepth > 0 && "Unexpected loop fusion depth");
        ComputationSliceState &bestSlice =
            depthSliceUnions[bestDstLoopDepth - 1];
        assert(!bestSlice.isEmpty() && "Missing slice union for depth");

        // Determine if 'srcId' can be removed after fusion, taking into
        // account remaining dependences, escaping memrefs and the fusion
        // insertion point.
        bool removeSrcNode = canRemoveSrcNodeAfterFusion(
            srcId, dstId, bestSlice, fusedLoopInsPoint, srcEscapingMemRefs,
            mdg);

        DenseSet<Value> privateMemrefs;
        for (Value memref : producerConsumerMemrefs) {
          if (canCreatePrivateMemRef(memref, srcEscapingMemRefs, srcId, dstId,
                                     removeSrcNode)) {
            // Create a private version of this memref.
            LLVM_DEBUG(llvm::dbgs()
                       << "Creating private memref for " << memref << '\n');
            // Create a private version of this memref.
            privateMemrefs.insert(memref);
          }
        }

        // Fuse computation slice of 'srcLoopNest' into 'dstLoopNest'.
        fuseLoops(srcAffineForOp, dstAffineForOp, bestSlice);
        dstNodeChanged = true;

        LLVM_DEBUG(llvm::dbgs()
                   << "Fused src loop " << srcId << " into dst loop " << dstId
                   << " at depth " << bestDstLoopDepth << ":\n"
                   << dstAffineForOp << "\n");

        // Move 'dstAffineForOp' before 'insertPointInst' if needed.
        if (fusedLoopInsPoint != dstAffineForOp)
          dstAffineForOp->moveBefore(fusedLoopInsPoint);

        // Update edges between 'srcNode' and 'dstNode'.
        mdg->updateEdges(srcNode->id, dstNode->id, privateMemrefs,
                         removeSrcNode);

        // Create private memrefs.
        if (!privateMemrefs.empty()) {
          // Gather stores for all the private-to-be memrefs.
          DenseMap<Value, SmallVector<Operation *, 4>> privateMemRefToStores;
          dstAffineForOp.walk([&](AffineWriteOpInterface storeOp) {
            Value storeMemRef = storeOp.getMemRef();
            if (privateMemrefs.count(storeMemRef) > 0)
              privateMemRefToStores[storeMemRef].push_back(storeOp);
          });

          // Replace original memrefs with private memrefs. Note that all the
          // loads and stores on these memrefs will be replaced with a new
          // loads and stores. Any reference to the original ones becomes
          // invalid after this point.
          for (auto &memrefToStoresPair : privateMemRefToStores) {
            // TODO: Use union of memref write regions to compute
            // private memref footprint.
            SmallVector<Operation *, 4> &storesForMemref =
                memrefToStoresPair.second;
            Value newMemRef = createPrivateMemRef(
                dstAffineForOp, storesForMemref[0], bestDstLoopDepth,
                fastMemorySpace, localBufSizeThreshold);
            // Create new node in dependence graph for 'newMemRef' alloc op.
            unsigned newMemRefNodeId = mdg->addNode(newMemRef.getDefiningOp());
            // Add edge from 'newMemRef' node to dstNode.
            mdg->addEdge(newMemRefNodeId, dstId, newMemRef);
          }
          // One or more entries for 'newMemRef' alloc op are inserted into
          // the DenseMap mdg->nodes. Since an insertion may cause DenseMap to
          // reallocate, update dstNode.
          dstNode = mdg->getNode(dstId);
        }

        // Collect dst loop stats after memref privatization transformation.
        LoopNestStateCollector dstLoopCollector;
        dstLoopCollector.collect(dstAffineForOp);

        // Clear and add back loads and stores.
        mdg->clearNodeLoadAndStores(dstNode->id);
        mdg->addToNode(
            dstId, dstLoopCollector.loadOpInsts, dstLoopCollector.storeOpInsts,
            dstLoopCollector.memrefLoads, dstLoopCollector.memrefStores,
            dstLoopCollector.memrefFrees);

        if (removeSrcNode) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Removing src loop " << srcId << " after fusion\n");
          // srcNode is no longer valid after it is removed from mdg.
          srcAffineForOp.erase();
          mdg->removeNode(srcId);
          srcNode = nullptr;
        }
      }
    } while (dstNodeChanged);
  }

  /// Visit each node in the graph, and for each node, attempt to fuse it with
  /// producer-consumer candidates. No fusion is performed when producers with a
  /// user count greater than `maxSrcUserCount` for any of the memrefs involved
  /// are encountered.
  void fuseProducerConsumerNodes(unsigned maxSrcUserCount) {
    LLVM_DEBUG(llvm::dbgs() << "--- Producer/Consumer Fusion ---\n");
    init();
    while (!worklist.empty()) {
      unsigned dstId = worklist.back();
      worklist.pop_back();
      performFusionsIntoDest(dstId, maxSrcUserCount);
    }
  }

  // Visits each node in the graph, and for each node, attempts to fuse it with
  // its sibling nodes (nodes which share a parent, but no dependence edges).
  void fuseSiblingNodes() {
    LLVM_DEBUG(llvm::dbgs() << "--- Sibling Fusion ---\n");
    init();
    while (!worklist.empty()) {
      unsigned dstId = worklist.back();
      worklist.pop_back();

      // Skip if this node was removed (fused into another node).
      if (mdg->nodes.count(dstId) == 0)
        continue;
      // Get 'dstNode' into which to attempt fusion.
      auto *dstNode = mdg->getNode(dstId);
      // Skip if 'dstNode' is not a loop nest.
      if (!isa<AffineForOp>(dstNode->op))
        continue;
      // Attempt to fuse 'dstNode' with its sibling nodes in the graph.
      fuseWithSiblingNodes(dstNode);
    }
  }

  // Attempt to fuse 'dstNode' with sibling nodes in the graph.
  void fuseWithSiblingNodes(Node *dstNode) {
    DenseSet<unsigned> visitedSibNodeIds;
    std::pair<unsigned, Value> idAndMemref;
    auto dstAffineForOp = cast<AffineForOp>(dstNode->op);

    while (findSiblingNodeToFuse(dstNode, &visitedSibNodeIds, &idAndMemref)) {
      unsigned sibId = idAndMemref.first;
      Value memref = idAndMemref.second;
      // TODO: Check that 'sibStoreOpInst' post-dominates all other
      // stores to the same memref in 'sibNode' loop nest.
      auto *sibNode = mdg->getNode(sibId);
      // Compute an operation list insertion point for the fused loop
      // nest which preserves dependences.
      assert(sibNode->op->getBlock() == dstNode->op->getBlock());
      Operation *insertPointInst =
          sibNode->op->isBeforeInBlock(dstNode->op)
              ? mdg->getFusedLoopNestInsertionPoint(sibNode->id, dstNode->id)
              : mdg->getFusedLoopNestInsertionPoint(dstNode->id, sibNode->id);
      if (insertPointInst == nullptr)
        continue;

      // Check if fusion would be profitable and at what depth.

      // Get unique 'sibNode' load op to 'memref'.
      SmallVector<Operation *, 2> sibLoadOpInsts;
      sibNode->getLoadOpsForMemref(memref, &sibLoadOpInsts);
      // Currently findSiblingNodeToFuse searches for siblings with one load.
      assert(sibLoadOpInsts.size() == 1);
      Operation *sibLoadOpInst = sibLoadOpInsts[0];

      // Gather 'dstNode' load ops to 'memref'.
      SmallVector<Operation *, 2> dstLoadOpInsts;
      dstNode->getLoadOpsForMemref(memref, &dstLoadOpInsts);

      // It's possible this fusion is at an inner depth (i.e., there are common
      // surrounding affine loops for the source and destination for ops). We
      // need to get this number because the call to canFuseLoops needs to be
      // passed the absolute depth. The max legal depth and the depths we try
      // below are however *relative* and as such don't include the common
      // depth.
      SmallVector<AffineForOp, 4> surroundingLoops;
      getAffineForIVs(*dstAffineForOp, &surroundingLoops);
      unsigned numSurroundingLoops = surroundingLoops.size();
      SmallVector<AffineForOp, 4> dstLoopIVs;
      getAffineForIVs(*dstLoadOpInsts[0], &dstLoopIVs);
      unsigned dstLoopDepthTest = dstLoopIVs.size() - numSurroundingLoops;
      auto sibAffineForOp = cast<AffineForOp>(sibNode->op);

      // Compute loop depth and slice union for fusion.
      SmallVector<ComputationSliceState, 8> depthSliceUnions;
      depthSliceUnions.resize(dstLoopDepthTest);
      unsigned maxLegalFusionDepth = 0;
      FusionStrategy strategy(memref);
      for (unsigned i = 1; i <= dstLoopDepthTest; ++i) {
        FusionResult result =
            affine::canFuseLoops(sibAffineForOp, dstAffineForOp,
                                 /*dstLoopDepth=*/i + numSurroundingLoops,
                                 &depthSliceUnions[i - 1], strategy);

        if (result.value == FusionResult::Success)
          maxLegalFusionDepth = i;
      }

      LLVM_DEBUG(llvm::dbgs() << "Max legal depth for fusion: "
                              << maxLegalFusionDepth << '\n');

      // Skip if fusion is not feasible at any loop depths.
      if (maxLegalFusionDepth == 0)
        continue;

      unsigned bestDstLoopDepth = maxLegalFusionDepth;
      if (!maximalFusion) {
        // Check if fusion would be profitable. For sibling fusion, the sibling
        // load op is treated as the src "store" op for fusion profitability
        // purposes. The footprint of the load in the slice relative to the
        // unfused source's determines reuse.
        if (!isFusionProfitable(sibLoadOpInst, sibLoadOpInst, dstAffineForOp,
                                depthSliceUnions, maxLegalFusionDepth,
                                &bestDstLoopDepth, computeToleranceThreshold))
          continue;
      }

      assert(bestDstLoopDepth > 0 && "Unexpected loop fusion depth");
      assert(!depthSliceUnions[bestDstLoopDepth - 1].isEmpty() &&
             "Fusion depth has no computed slice union");
      // Check if source loop is being inserted in the innermost
      // destination loop. Based on this, the fused loop may be optimized
      // further inside `fuseLoops`.
      bool isInnermostInsertion = (bestDstLoopDepth == dstLoopDepthTest);
      // Fuse computation slice of 'sibLoopNest' into 'dstLoopNest'.
      affine::fuseLoops(sibAffineForOp, dstAffineForOp,
                        depthSliceUnions[bestDstLoopDepth - 1],
                        isInnermostInsertion);

      auto dstForInst = cast<AffineForOp>(dstNode->op);
      // Update operation position of fused loop nest (if needed).
      if (insertPointInst != dstForInst) {
        dstForInst->moveBefore(insertPointInst);
      }
      // Update data dependence graph state post fusion.
      updateStateAfterSiblingFusion(sibNode, dstNode);
    }
  }

  // Searches block argument uses and the graph from 'dstNode' looking for a
  // fusion candidate sibling node which shares no dependences with 'dstNode'
  // but which loads from the same memref. Returns true and sets
  // 'idAndMemrefToFuse' on success. Returns false otherwise.
  bool findSiblingNodeToFuse(Node *dstNode,
                             DenseSet<unsigned> *visitedSibNodeIds,
                             std::pair<unsigned, Value> *idAndMemrefToFuse) {
    // Returns true if 'sibNode' can be fused with 'dstNode' for input reuse
    // on 'memref'.
    auto canFuseWithSibNode = [&](Node *sibNode, Value memref) {
      // Skip if 'outEdge' is not a read-after-write dependence.
      // TODO: Remove restrict to single load op restriction.
      if (sibNode->getLoadOpCount(memref) != 1)
        return false;
      // Skip if there exists a path of dependent edges between
      // 'sibNode' and 'dstNode'.
      if (mdg->hasDependencePath(sibNode->id, dstNode->id) ||
          mdg->hasDependencePath(dstNode->id, sibNode->id))
        return false;
      // Skip sib node if it loads to (and stores from) the same memref on
      // which it also has an input dependence edge.
      DenseSet<Value> loadAndStoreMemrefSet;
      sibNode->getLoadAndStoreMemrefSet(&loadAndStoreMemrefSet);
      if (llvm::any_of(loadAndStoreMemrefSet, [=](Value memref) {
            return mdg->getIncomingMemRefAccesses(sibNode->id, memref) > 0;
          }))
        return false;

      // Check that all stores are to the same memref if any.
      DenseSet<Value> storeMemrefs;
      for (auto *storeOpInst : sibNode->stores) {
        storeMemrefs.insert(
            cast<AffineWriteOpInterface>(storeOpInst).getMemRef());
      }
      return storeMemrefs.size() <= 1;
    };

    // Search for siblings which load the same memref block argument.
    Block *block = dstNode->op->getBlock();
    for (unsigned i = 0, e = block->getNumArguments(); i != e; ++i) {
      for (Operation *user : block->getArgument(i).getUsers()) {
        auto loadOp = dyn_cast<AffineReadOpInterface>(user);
        if (!loadOp)
          continue;
        // Gather loops surrounding 'use'.
        SmallVector<AffineForOp, 4> loops;
        getAffineForIVs(*user, &loops);
        // Skip 'use' if it is not within a loop nest.
        // Find the surrounding affine.for nested immediately within the
        // block.
        auto *it = llvm::find_if(loops, [&](AffineForOp loop) {
          return loop->getBlock() == &mdg->block;
        });
        // Skip 'use' if it is not within a loop nest in `block`.
        if (it == loops.end())
          continue;
        Node *sibNode = mdg->getForOpNode(*it);
        assert(sibNode != nullptr);
        // Skip 'use' if it not a sibling to 'dstNode'.
        if (sibNode->id == dstNode->id)
          continue;
        // Skip 'use' if it has been visited.
        if (visitedSibNodeIds->count(sibNode->id) > 0)
          continue;
        // Skip 'use' if it does not load from the same memref as 'dstNode'.
        auto memref = loadOp.getMemRef();
        if (dstNode->getLoadOpCount(memref) == 0)
          continue;
        // Check if 'sibNode/dstNode' can be input-reuse fused on 'memref'.
        if (canFuseWithSibNode(sibNode, memref)) {
          visitedSibNodeIds->insert(sibNode->id);
          idAndMemrefToFuse->first = sibNode->id;
          idAndMemrefToFuse->second = memref;
          return true;
        }
      }
    }

    // Search for siblings by following edges through an intermediate src node.
    // Collect candidate 'dstNode' input edges in 'inEdges'.
    SmallVector<MemRefDependenceGraph::Edge, 2> inEdges;
    mdg->forEachMemRefInputEdge(
        dstNode->id, [&](MemRefDependenceGraph::Edge inEdge) {
          // Add 'inEdge' if it is a read-after-write dependence.
          if (dstNode->getLoadOpCount(inEdge.value) > 0 &&
              mdg->getNode(inEdge.id)->getStoreOpCount(inEdge.value) > 0)
            inEdges.push_back(inEdge);
        });

    // Search for sibling nodes to fuse by visiting output edges from each input
    // edge in 'inEdges'.
    for (auto &inEdge : inEdges) {
      // Collect candidate output edges from each node 'inEdge.id' in 'inEdges'.
      SmallVector<MemRefDependenceGraph::Edge, 2> outEdges;
      mdg->forEachMemRefOutputEdge(
          inEdge.id, [&](MemRefDependenceGraph::Edge outEdge) {
            unsigned sibNodeId = outEdge.id;
            if (visitedSibNodeIds->count(sibNodeId) > 0)
              return;
            // Skip output edge if not a sibling using the same memref.
            if (outEdge.id == dstNode->id || outEdge.value != inEdge.value)
              return;
            auto *sibNode = mdg->getNode(sibNodeId);
            if (!isa<AffineForOp>(sibNode->op))
              return;
            // Check if 'sibNode/dstNode' can be input-reuse fused on 'memref'.
            if (canFuseWithSibNode(sibNode, outEdge.value)) {
              // Add candidate 'outEdge' to sibling node.
              outEdges.push_back(outEdge);
            }
          });

      // Add first candidate if any were returned.
      if (!outEdges.empty()) {
        visitedSibNodeIds->insert(outEdges[0].id);
        idAndMemrefToFuse->first = outEdges[0].id;
        idAndMemrefToFuse->second = outEdges[0].value;
        return true;
      }
    }
    return false;
  }

  /// Update data dependence graph state to reflect sibling fusion of 'sibNode'
  /// into 'dstNode'.
  void updateStateAfterSiblingFusion(Node *sibNode, Node *dstNode) {
    // Update 'sibNode' and 'dstNode' input/output edges to reflect fusion.
    mdg->updateEdges(sibNode->id, dstNode->id);

    // Collect dst loop stats after memref privatization transformation.
    auto dstForInst = cast<AffineForOp>(dstNode->op);
    LoopNestStateCollector dstLoopCollector;
    dstLoopCollector.collect(dstForInst);
    // Clear and add back loads and stores
    mdg->clearNodeLoadAndStores(dstNode->id);
    mdg->addToNode(dstNode->id, dstLoopCollector.loadOpInsts,
                   dstLoopCollector.storeOpInsts, dstLoopCollector.memrefLoads,
                   dstLoopCollector.memrefStores, dstLoopCollector.memrefFrees);
    // Remove old sibling loop nest if it no longer has outgoing dependence
    // edges, and it does not write to a memref which escapes the block.
    if (mdg->getOutEdgeCount(sibNode->id) == 0) {
      Operation *op = sibNode->op;
      mdg->removeNode(sibNode->id);
      op->erase();
    }
  }

  // Clean up any allocs with no users.
  void eraseUnusedMemRefAllocations() {
    for (auto &pair : mdg->memrefEdgeCount) {
      if (pair.second > 0)
        continue;
      auto memref = pair.first;
      // Skip if there exist other uses (return operation or function calls).
      if (!memref.use_empty())
        continue;
      // Use list expected to match the dep graph info.
      auto *op = memref.getDefiningOp();
      if (isa_and_nonnull<memref::AllocOp>(op))
        op->erase();
    }
  }
};

} // namespace

/// Run fusion on `block`.
void LoopFusion::runOnBlock(Block *block) {
  MemRefDependenceGraph g(*block);
  if (!g.init()) {
    LLVM_DEBUG(llvm::dbgs() << "MDG init failed\n");
    return;
  }

  std::optional<unsigned> fastMemorySpaceOpt;
  if (fastMemorySpace.hasValue())
    fastMemorySpaceOpt = fastMemorySpace;
  unsigned localBufSizeThresholdBytes = localBufSizeThreshold * 1024;
  GreedyFusion fusion(&g, localBufSizeThresholdBytes, fastMemorySpaceOpt,
                      maximalFusion, computeToleranceThreshold);

  if (affineFusionMode == FusionMode::ProducerConsumer)
    fusion.runProducerConsumerFusionOnly();
  else if (affineFusionMode == FusionMode::Sibling)
    fusion.runSiblingFusionOnly();
  else
    fusion.runGreedyFusion();
}

void LoopFusion::runOnOperation() {
  // Call fusion on every op that has at least two affine.for nests (in post
  // order).
  getOperation()->walk([&](Operation *op) {
    for (Region &region : op->getRegions()) {
      for (Block &block : region.getBlocks()) {
        auto affineFors = block.getOps<AffineForOp>();
        if (!affineFors.empty() && !llvm::hasSingleElement(affineFors))
          runOnBlock(&block);
      }
    }
  });
}

std::unique_ptr<Pass> mlir::affine::createLoopFusionPass(
    unsigned fastMemorySpace, uint64_t localBufSizeThreshold,
    bool maximalFusion, enum FusionMode affineFusionMode) {
  return std::make_unique<LoopFusion>(fastMemorySpace, localBufSizeThreshold,
                                      maximalFusion, affineFusionMode);
}
