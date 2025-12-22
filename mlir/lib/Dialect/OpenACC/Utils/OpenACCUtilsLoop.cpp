//===- OpenACCUtilsLoop.cpp - OpenACC Loop Utilities ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains utility functions for converting OpenACC loops to SCF.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/OpenACCUtilsLoop.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;

namespace {

/// Calculate trip count for a loop: (ub - lb + step) / step
/// If inclusiveUpperbound is false, subtracts 1 from ub first.
static Value calculateTripCount(OpBuilder &b, Location loc, Value lb, Value ub,
                                Value step, bool inclusiveUpperbound) {
  Type type = b.getIndexType();

  // Convert original loop arguments to index type
  lb = getValueOrCreateCastToIndexLike(b, loc, type, lb);
  ub = getValueOrCreateCastToIndexLike(b, loc, type, ub);
  step = getValueOrCreateCastToIndexLike(b, loc, type, step);

  if (!inclusiveUpperbound) {
    Value one = arith::ConstantIndexOp::create(b, loc, 1);
    ub = b.createOrFold<arith::SubIOp>(loc, ub, one);
  }

  Value sub = b.createOrFold<arith::SubIOp>(loc, ub, lb);
  Value add = b.createOrFold<arith::AddIOp>(loc, sub, step);
  return b.createOrFold<arith::DivSIOp>(loc, add, step);
}

/// Get exclusive upper bound from acc.loop (add 1 if inclusive).
/// The result is always in index type.
static Value getExclusiveUpperBoundAsIndex(acc::LoopOp loopOp, size_t ivPos,
                                           OpBuilder &b) {
  bool isInclusive = false;
  if (loopOp.getInclusiveUpperbound().has_value())
    isInclusive = loopOp.getInclusiveUpperboundAttr().asArrayRef()[ivPos];

  Value origUB = loopOp.getUpperbound()[ivPos];
  Location loc = origUB.getLoc();
  Type indexType = b.getIndexType();

  // Cast to index first, then add if inclusive
  Value ub = getValueOrCreateCastToIndexLike(b, loc, indexType, origUB);
  if (isInclusive) {
    Value one = arith::ConstantIndexOp::create(b, loc, 1);
    ub = b.createOrFold<arith::AddIOp>(loc, ub, one);
  }
  return ub;
}

/// Handle differing types between SCF (index) and ACC loops.
/// Creates casts from the new SCF IVs to the original ACC IV types and updates
/// the mapping. The newIVs should correspond 1:1 with the ACC loop's IVs.
static void mapACCLoopIVsToSCFIVs(acc::LoopOp accLoop, ValueRange newIVs,
                                  OpBuilder &b, IRMapping &mapping) {
  for (auto [origIV, newIV] :
       llvm::zip(accLoop.getBody().getArguments(), newIVs)) {
    Value replacementIV = getValueOrCreateCastToIndexLike(
        b, accLoop->getLoc(), origIV.getType(), newIV);
    mapping.map(origIV, replacementIV);
  }
}

/// Normalize IV uses after converting to normalized loop form.
/// For normalized loops (lb=0, step=1), we need to denormalize the IV:
/// original_iv = new_iv * orig_step + orig_lb
static void normalizeIVUses(OpBuilder &b, Location loc, Value iv, Value origLB,
                            Value origStep) {
  Type indexType = b.getIndexType();
  Value lb = getValueOrCreateCastToIndexLike(b, loc, indexType, origLB);
  Value step = getValueOrCreateCastToIndexLike(b, loc, indexType, origStep);

  // new_iv * step + lb
  Value scaled = arith::MulIOp::create(b, loc, iv, step);
  Value denormalized = arith::AddIOp::create(b, loc, scaled, lb);

  // Replace uses of iv with denormalized value, except for the ops that
  // compute the denormalized value itself (muli and addi)
  llvm::SmallPtrSet<Operation *, 2> exceptions;
  exceptions.insert(scaled.getDefiningOp());
  exceptions.insert(denormalized.getDefiningOp());
  iv.replaceAllUsesExcept(denormalized, exceptions);
}

/// Clone an ACC region into a destination block, handling the ACC terminators.
/// Returns the insertion point after the cloned operations.
static Block::iterator cloneACCRegionInto(Region *src, Block *dest,
                                          Block::iterator insertionPoint,
                                          IRMapping &mapping) {
  assert(src->hasOneBlock() && "expected single-block region");

  Region *insertRegion = dest->getParent();
  Block *postInsertBlock = dest->splitBlock(insertionPoint);
  src->cloneInto(insertRegion, postInsertBlock->getIterator(), mapping);

  auto lastNewBlock = std::prev(postInsertBlock->getIterator());

  Block::iterator newInsertionPoint;
  Operation *terminator = lastNewBlock->getTerminator();

  if (auto yieldOp = dyn_cast<acc::YieldOp>(terminator)) {
    newInsertionPoint = std::prev(yieldOp->getIterator());
    yieldOp.erase();
  } else if (auto terminatorOp = dyn_cast<acc::TerminatorOp>(terminator)) {
    newInsertionPoint = std::prev(terminatorOp->getIterator());
    terminatorOp.erase();
  } else {
    llvm_unreachable("unexpected terminator in ACC region");
  }

  // Merge last block with the postInsertBlock
  lastNewBlock->getOperations().splice(lastNewBlock->end(),
                                       postInsertBlock->getOperations());
  postInsertBlock->erase();

  // Merge first block with original dest block
  auto firstNewBlock = std::next(dest->getIterator());
  dest->getOperations().splice(dest->end(), firstNewBlock->getOperations());
  firstNewBlock->erase();

  return newInsertionPoint;
}

/// Wrap a multi-block region with scf.execute_region.
static scf::ExecuteRegionOp
wrapMultiBlockRegionWithSCFExecuteRegion(Region &region, IRMapping &mapping,
                                         Location loc, OpBuilder &b) {
  auto exeRegionOp = scf::ExecuteRegionOp::create(b, loc, TypeRange{});

  b.cloneRegionBefore(region, exeRegionOp.getRegion(),
                      exeRegionOp.getRegion().end(), mapping);

  // Find and replace the ACC terminator with scf.yield
  Operation *terminator = exeRegionOp.getRegion().back().getTerminator();
  if (auto yieldOp = dyn_cast<acc::YieldOp>(terminator)) {
    if (yieldOp.getNumOperands() > 0) {
      region.getParentOp()->emitError(
          "acc.loop with results not yet supported");
      return nullptr;
    }
    terminator->erase();
  } else if (auto accTerminator = dyn_cast<acc::TerminatorOp>(terminator)) {
    terminator->erase();
  } else {
    llvm_unreachable("unexpected terminator in ACC region");
  }

  b.setInsertionPointToEnd(&exeRegionOp.getRegion().back());
  scf::YieldOp::create(b, loc);
  return exeRegionOp;
}

} // namespace

namespace mlir {
namespace acc {

scf::ForOp convertACCLoopToSCFFor(LoopOp loopOp, bool enableCollapse) {
  assert(!loopOp.getUnstructured() &&
         "use convertUnstructuredACCLoopToSCFExecuteRegion for unstructured "
         "loops");

  OpBuilder b(loopOp);

  // Lambda to create an scf::ForOp for a single dimension of the acc.loop
  auto createSCFForOp = [&](acc::LoopOp accLoopOp, size_t idx, OpBuilder &b,
                            OpBuilder &nestBuilder) -> scf::ForOp {
    assert(idx < accLoopOp.getBody().getNumArguments());

    Location loc = accLoopOp->getLoc();
    Type indexType = b.getIndexType();

    Value newLowerBound = getValueOrCreateCastToIndexLike(
        b, loc, indexType, accLoopOp.getLowerbound()[idx]);
    Value newUpperBound = getExclusiveUpperBoundAsIndex(accLoopOp, idx, b);
    Value newStep = getValueOrCreateCastToIndexLike(b, loc, indexType,
                                                    accLoopOp.getStep()[idx]);

    return scf::ForOp::create(nestBuilder, loc, newLowerBound, newUpperBound,
                              newStep);
  };

  // Create nested scf.for loops and build IR mapping for IVs
  IRMapping mapping;
  SmallVector<scf::ForOp> forOps;
  b.setInsertionPoint(loopOp);
  OpBuilder nestBuilder(loopOp);

  for (BlockArgument iv : loopOp.getBody().getArguments()) {
    size_t idx = iv.getArgNumber();
    scf::ForOp forOp = createSCFForOp(loopOp, idx, b, nestBuilder);
    forOps.push_back(forOp);
    mapping.map(iv, forOp.getInductionVar());

    // The "outside" builder stays before the outer loop
    if (idx == 0)
      b.setInsertionPoint(forOp);

    // The "inside" builder moves into each new loop
    nestBuilder.setInsertionPointToStart(forOp.getBody());
  }

  // Handle IV type conversion (index -> original type)
  SmallVector<Value> scfIVs;
  for (scf::ForOp forOp : forOps)
    scfIVs.push_back(forOp.getInductionVar());
  mapACCLoopIVsToSCFIVs(loopOp, scfIVs, nestBuilder, mapping);

  // Clone the loop body into the innermost scf.for
  cloneACCRegionInto(&loopOp.getRegion(), forOps.back().getBody(),
                     nestBuilder.getInsertionPoint(), mapping);

  // Optionally collapse nested loops
  if (enableCollapse && forOps.size() > 1)
    if (failed(coalesceLoops(forOps)))
      loopOp.emitError("failed to collapse acc.loop");

  return forOps.front();
}

scf::ParallelOp convertACCLoopToSCFParallel(LoopOp loopOp, OpBuilder &b) {
  assert(!loopOp.getUnstructured() &&
         "use convertUnstructuredACCLoopToSCFExecuteRegion for unstructured "
         "loops");
  assert(b.getInsertionBlock() &&
         !loopOp->isProperAncestor(b.getInsertionBlock()->getParentOp()) &&
         "builder insertion point must not be inside the loop being converted");

  Location loc = loopOp->getLoc();

  SmallVector<Value> lowerBounds, upperBounds, steps;

  // Normalize all loops: lb=0, step=1, ub=tripCount
  Value lb = arith::ConstantIndexOp::create(b, loc, 0);
  Value step = arith::ConstantIndexOp::create(b, loc, 1);

  for (auto [idx, iv] : llvm::enumerate(loopOp.getBody().getArguments())) {
    bool inclusiveUpperbound = false;
    if (loopOp.getInclusiveUpperbound().has_value())
      inclusiveUpperbound = loopOp.getInclusiveUpperbound().value()[idx];

    Value ub = calculateTripCount(b, loc, loopOp.getLowerbound()[idx],
                                  loopOp.getUpperbound()[idx],
                                  loopOp.getStep()[idx], inclusiveUpperbound);

    lowerBounds.push_back(lb);
    upperBounds.push_back(ub);
    steps.push_back(step);
  }

  auto parallelOp =
      scf::ParallelOp::create(b, loc, lowerBounds, upperBounds, steps);

  // Create IV type conversions
  IRMapping mapping;
  b.setInsertionPointToStart(parallelOp.getBody());
  mapACCLoopIVsToSCFIVs(loopOp, parallelOp.getInductionVars(), b, mapping);

  if (!loopOp.getRegion().hasOneBlock()) {
    auto exeRegion = wrapMultiBlockRegionWithSCFExecuteRegion(
        loopOp.getRegion(), mapping, loc, b);
    if (!exeRegion) {
      parallelOp.erase();
      return nullptr;
    }
  } else {
    cloneACCRegionInto(&loopOp.getRegion(), parallelOp.getBody(),
                       b.getInsertionPoint(), mapping);
  }

  // Denormalize IV uses
  b.setInsertionPointToStart(parallelOp.getBody());
  for (auto [idx, iv] : llvm::enumerate(parallelOp.getBody()->getArguments()))
    if (!iv.use_empty())
      normalizeIVUses(b, loc, iv, loopOp.getLowerbound()[idx],
                      loopOp.getStep()[idx]);

  return parallelOp;
}

scf::ExecuteRegionOp
convertUnstructuredACCLoopToSCFExecuteRegion(LoopOp loopOp, OpBuilder &b) {
  assert(loopOp.getUnstructured() &&
         "use convertACCLoopToSCFFor for structured loops");
  assert(b.getInsertionBlock() &&
         !loopOp->isProperAncestor(b.getInsertionBlock()->getParentOp()) &&
         "builder insertion point must not be inside the loop being converted");

  IRMapping mapping;
  return wrapMultiBlockRegionWithSCFExecuteRegion(loopOp.getRegion(), mapping,
                                                  loopOp->getLoc(), b);
}

} // namespace acc
} // namespace mlir
