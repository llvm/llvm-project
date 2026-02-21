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
    ub = b.createOrFold<arith::SubIOp>(loc, ub, one,
                                       arith::IntegerOverflowFlags::nsw);
  }

  Value sub = b.createOrFold<arith::SubIOp>(loc, ub, lb,
                                            arith::IntegerOverflowFlags::nsw);
  Value add = b.createOrFold<arith::AddIOp>(loc, sub, step,
                                            arith::IntegerOverflowFlags::nsw);
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
    ub = b.createOrFold<arith::AddIOp>(loc, ub, one,
                                       arith::IntegerOverflowFlags::nsw);
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
  Value scaled =
      arith::MulIOp::create(b, loc, iv, step, arith::IntegerOverflowFlags::nsw);
  Value denormalized = arith::AddIOp::create(b, loc, scaled, lb,
                                             arith::IntegerOverflowFlags::nsw);

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
                                          IRMapping &mapping,
                                          RewriterBase &rewriter) {
  assert(src->hasOneBlock() && "expected single-block region");

  Region *insertRegion = dest->getParent();
  Block *postInsertBlock = rewriter.splitBlock(dest, insertionPoint);
  rewriter.cloneRegionBefore(*src, *insertRegion,
                             postInsertBlock->getIterator(), mapping);

  auto lastNewBlock = std::prev(postInsertBlock->getIterator());

  Block::iterator newInsertionPoint;
  Operation *terminator = lastNewBlock->getTerminator();

  if (auto yieldOp = dyn_cast<acc::YieldOp>(terminator)) {
    newInsertionPoint = std::prev(yieldOp->getIterator());
    rewriter.eraseOp(yieldOp);
  } else if (auto terminatorOp = dyn_cast<acc::TerminatorOp>(terminator)) {
    newInsertionPoint = std::prev(terminatorOp->getIterator());
    rewriter.eraseOp(terminatorOp);
  } else {
    llvm_unreachable("unexpected terminator in ACC region");
  }

  // Merge last block with the postInsertBlock
  rewriter.mergeBlocks(postInsertBlock, &*lastNewBlock);

  // Merge first block with original dest block
  Block *firstNewBlock = &*std::next(dest->getIterator());
  rewriter.mergeBlocks(firstNewBlock, dest);

  return newInsertionPoint;
}

/// Wrap a multi-block region with scf.execute_region.
static scf::ExecuteRegionOp
wrapMultiBlockRegionWithSCFExecuteRegion(Region &region, IRMapping &mapping,
                                         Location loc, RewriterBase &rewriter) {
  auto exeRegionOp = scf::ExecuteRegionOp::create(rewriter, loc, TypeRange{});

  rewriter.cloneRegionBefore(region, exeRegionOp.getRegion(),
                             exeRegionOp.getRegion().end(), mapping);

  // Find and replace the ACC terminator with scf.yield
  Operation *terminator = exeRegionOp.getRegion().back().getTerminator();
  if (auto yieldOp = dyn_cast<acc::YieldOp>(terminator)) {
    if (yieldOp.getNumOperands() > 0) {
      region.getParentOp()->emitError(
          "acc.loop with results not yet supported");
      return nullptr;
    }
  } else if (!isa<acc::TerminatorOp>(terminator)) {
    llvm_unreachable("unexpected terminator in ACC region");
  }

  rewriter.eraseOp(terminator);
  rewriter.setInsertionPointToEnd(&exeRegionOp.getRegion().back());
  scf::YieldOp::create(rewriter, loc);
  return exeRegionOp;
}

} // namespace

namespace mlir {
namespace acc {

scf::ForOp convertACCLoopToSCFFor(LoopOp loopOp, RewriterBase &rewriter,
                                  bool enableCollapse) {
  assert(!loopOp.getUnstructured() &&
         "use convertUnstructuredACCLoopToSCFExecuteRegion for unstructured "
         "loops");

  Location loc = loopOp->getLoc();
  Type indexType = rewriter.getIndexType();

  // Create nested scf.for loops and build IR mapping for IVs
  IRMapping mapping;
  SmallVector<scf::ForOp> forOps;

  // Save the original insertion point
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(loopOp);

  // First, compute ALL loop bounds at the current insertion point (before
  // any ForOp). This ensures all bounds are defined in the outer scope,
  // which is required for coalesceLoops to work correctly.
  SmallVector<Value> lowerBounds, upperBounds, steps;
  for (BlockArgument iv : loopOp.getBody().getArguments()) {
    size_t idx = iv.getArgNumber();
    Value newLowerBound = getValueOrCreateCastToIndexLike(
        rewriter, loc, indexType, loopOp.getLowerbound()[idx]);
    Value newUpperBound = getExclusiveUpperBoundAsIndex(loopOp, idx, rewriter);
    Value newStep = getValueOrCreateCastToIndexLike(rewriter, loc, indexType,
                                                    loopOp.getStep()[idx]);
    lowerBounds.push_back(newLowerBound);
    upperBounds.push_back(newUpperBound);
    steps.push_back(newStep);
  }

  // Now create the nested ForOps using the pre-computed bounds
  for (BlockArgument iv : loopOp.getBody().getArguments()) {
    size_t idx = iv.getArgNumber();

    // For nested loops, insert inside the previous loop's body
    if (idx > 0)
      rewriter.setInsertionPointToStart(forOps.back().getBody());

    scf::ForOp forOp = scf::ForOp::create(rewriter, loc, lowerBounds[idx],
                                          upperBounds[idx], steps[idx]);
    forOps.push_back(forOp);
    mapping.map(iv, forOp.getInductionVar());
  }

  // Set insertion point inside the innermost loop for IV casts and body cloning
  rewriter.setInsertionPointToStart(forOps.back().getBody());

  // Handle IV type conversion (index -> original type)
  SmallVector<Value> scfIVs;
  for (scf::ForOp forOp : forOps)
    scfIVs.push_back(forOp.getInductionVar());
  mapACCLoopIVsToSCFIVs(loopOp, scfIVs, rewriter, mapping);

  // Clone the loop body into the innermost scf.for
  cloneACCRegionInto(&loopOp.getRegion(), forOps.back().getBody(),
                     rewriter.getInsertionPoint(), mapping, rewriter);

  // Optionally collapse nested loops
  if (enableCollapse && forOps.size() > 1)
    if (failed(coalesceLoops(rewriter, forOps)))
      loopOp.emitError("failed to collapse acc.loop");

  return forOps.front();
}

scf::ParallelOp convertACCLoopToSCFParallel(LoopOp loopOp,
                                            RewriterBase &rewriter) {
  assert(!loopOp.getUnstructured() &&
         "use convertUnstructuredACCLoopToSCFExecuteRegion for unstructured "
         "loops");
  assert(
      rewriter.getInsertionBlock() &&
      !loopOp->isProperAncestor(rewriter.getInsertionBlock()->getParentOp()) &&
      "builder insertion point must not be inside the loop being converted");

  Location loc = loopOp->getLoc();

  SmallVector<Value> lowerBounds, upperBounds, steps;

  // Normalize all loops: lb=0, step=1, ub=tripCount
  Value lb = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value step = arith::ConstantIndexOp::create(rewriter, loc, 1);

  for (auto [idx, iv] : llvm::enumerate(loopOp.getBody().getArguments())) {
    bool inclusiveUpperbound = false;
    if (loopOp.getInclusiveUpperbound().has_value())
      inclusiveUpperbound = loopOp.getInclusiveUpperbound().value()[idx];

    Value ub = calculateTripCount(rewriter, loc, loopOp.getLowerbound()[idx],
                                  loopOp.getUpperbound()[idx],
                                  loopOp.getStep()[idx], inclusiveUpperbound);

    lowerBounds.push_back(lb);
    upperBounds.push_back(ub);
    steps.push_back(step);
  }

  auto parallelOp =
      scf::ParallelOp::create(rewriter, loc, lowerBounds, upperBounds, steps);

  // Create IV type conversions
  IRMapping mapping;
  rewriter.setInsertionPointToStart(parallelOp.getBody());
  mapACCLoopIVsToSCFIVs(loopOp, parallelOp.getInductionVars(), rewriter,
                        mapping);

  if (!loopOp.getRegion().hasOneBlock()) {
    auto exeRegion = wrapMultiBlockRegionWithSCFExecuteRegion(
        loopOp.getRegion(), mapping, loc, rewriter);
    if (!exeRegion) {
      rewriter.eraseOp(parallelOp);
      return nullptr;
    }
  } else {
    cloneACCRegionInto(&loopOp.getRegion(), parallelOp.getBody(),
                       rewriter.getInsertionPoint(), mapping, rewriter);
  }

  // Denormalize IV uses
  rewriter.setInsertionPointToStart(parallelOp.getBody());
  for (auto [idx, iv] : llvm::enumerate(parallelOp.getBody()->getArguments()))
    if (!iv.use_empty())
      normalizeIVUses(rewriter, loc, iv, loopOp.getLowerbound()[idx],
                      loopOp.getStep()[idx]);

  return parallelOp;
}

scf::ExecuteRegionOp
convertUnstructuredACCLoopToSCFExecuteRegion(LoopOp loopOp,
                                             RewriterBase &rewriter) {
  assert(loopOp.getUnstructured() &&
         "use convertACCLoopToSCFFor for structured loops");
  assert(
      rewriter.getInsertionBlock() &&
      !loopOp->isProperAncestor(rewriter.getInsertionBlock()->getParentOp()) &&
      "builder insertion point must not be inside the loop being converted");

  IRMapping mapping;
  return wrapMultiBlockRegionWithSCFExecuteRegion(loopOp.getRegion(), mapping,
                                                  loopOp->getLoc(), rewriter);
}

} // namespace acc
} // namespace mlir
