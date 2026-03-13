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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/ErrorHandling.h"

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

/// Helper used by loop conversion: clone region and return insertion point
/// only.
static Block::iterator cloneACCRegionIntoForLoop(Region *src, Block *dest,
                                                 Block::iterator insertionPoint,
                                                 IRMapping &mapping,
                                                 RewriterBase &rewriter) {
  auto [replacements, ip] =
      acc::cloneACCRegionInto(src, dest, insertionPoint, mapping, ValueRange{});
  (void)replacements;
  return ip;
}

} // namespace

namespace mlir {
namespace acc {

std::pair<SmallVector<Value>, Block::iterator>
cloneACCRegionInto(Region *src, Block *dest, Block::iterator inlinePoint,
                   IRMapping &mapping, ValueRange resultsToReplace) {
  if (!src->hasOneBlock())
    llvm_unreachable("cloneACCRegionInto: multi-block region not supported "
                     "(requires scf.execute_region)");

  Region *insertRegion = dest->getParent();
  Block *postInsertBlock = dest->splitBlock(inlinePoint);
  src->cloneInto(insertRegion, postInsertBlock->getIterator(), mapping);

  SmallVector<Value> replacements;
  Block *lastNewBlock = &*std::prev(postInsertBlock->getIterator());

  Block::iterator ip;
  if (auto yieldOp = dyn_cast<acc::YieldOp>(lastNewBlock->getTerminator())) {
    for (auto [replacement, orig] :
         llvm::zip(yieldOp.getOperands(), resultsToReplace)) {
      replaceAllUsesInRegionWith(orig, replacement, *dest->getParent());
      replacements.push_back(replacement);
    }
    ip = std::prev(yieldOp->getIterator());
    yieldOp.erase();
  } else {
    auto terminatorOp =
        dyn_cast<acc::TerminatorOp>(lastNewBlock->getTerminator());
    if (!terminatorOp)
      llvm_unreachable(
          "cloneACCRegionInto: expected acc.yield or acc.terminator");
    ip = std::prev(terminatorOp->getIterator());
    terminatorOp.erase();
  }

  lastNewBlock->getOperations().splice(lastNewBlock->end(),
                                       postInsertBlock->getOperations());
  postInsertBlock->erase();

  Block *firstNewBlock = &*std::next(dest->getIterator());
  dest->getOperations().splice(dest->end(), firstNewBlock->getOperations());
  firstNewBlock->erase();
  return {replacements, ip};
}

/// Wrap a multi-block region with scf.execute_region.
scf::ExecuteRegionOp
wrapMultiBlockRegionWithSCFExecuteRegion(Region &region, IRMapping &mapping,
                                         Location loc, RewriterBase &rewriter,
                                         bool convertFuncReturn) {
  SmallVector<Operation *> terminators;
  for (Block &block : region.getBlocks()) {
    if (block.empty())
      continue;
    Operation *term = block.getTerminator();
    if ((convertFuncReturn && isa<func::ReturnOp>(*term)) ||
        isa<acc::YieldOp>(*term))
      terminators.push_back(term);
  }
  SmallVector<Type> resultTypes;
  if (!terminators.empty())
    for (Value operand : terminators.front()->getOperands())
      resultTypes.push_back(operand.getType());

  auto exeRegionOp =
      scf::ExecuteRegionOp::create(rewriter, loc, TypeRange(resultTypes));

  rewriter.cloneRegionBefore(region, exeRegionOp.getRegion(),
                             exeRegionOp.getRegion().end(), mapping);

  for (Operation *term : terminators) {
    Operation *blockTerminator = mapping.lookup(term);
    assert(blockTerminator && "expected terminator to be in mapping");
    rewriter.setInsertionPoint(blockTerminator);
    (void)scf::YieldOp::create(rewriter, blockTerminator->getLoc(),
                               blockTerminator->getOperands());
    rewriter.eraseOp(blockTerminator);
  }

  return exeRegionOp;
}

scf::ForOp convertACCLoopToSCFFor(LoopOp loopOp, RewriterBase &rewriter,
                                  bool enableCollapse) {
  assert(!loopOp.getUnstructured() &&
         "use convertUnstructuredACCLoopToSCFExecuteRegion for unstructured "
         "loops");

  Location loc = loopOp->getLoc();

  IRMapping mapping;
  SmallVector<scf::ForOp> forOps;

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(loopOp);

  // Normalize all loops: lb=0, step=1, ub=tripCount.
  // scf.for requires a positive step, but acc.loop may have arbitrary steps
  // (including negative). Normalizing unconditionally keeps this consistent
  // with convertACCLoopToSCFParallel and lets later passes fold constants.
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);

  SmallVector<Value> tripCounts;
  for (auto [idx, iv] : llvm::enumerate(loopOp.getBody().getArguments())) {
    bool inclusiveUpperbound = false;
    if (loopOp.getInclusiveUpperbound().has_value())
      inclusiveUpperbound =
          loopOp.getInclusiveUpperboundAttr().asArrayRef()[idx];

    Value tc = calculateTripCount(rewriter, loc, loopOp.getLowerbound()[idx],
                                  loopOp.getUpperbound()[idx],
                                  loopOp.getStep()[idx], inclusiveUpperbound);
    tripCounts.push_back(tc);
  }

  for (auto [idx, iv] : llvm::enumerate(loopOp.getBody().getArguments())) {
    // For nested loops, insert inside the previous loop's body
    if (idx > 0)
      rewriter.setInsertionPointToStart(forOps.back().getBody());

    scf::ForOp forOp =
        scf::ForOp::create(rewriter, loc, zero, tripCounts[idx], one);
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
  cloneACCRegionIntoForLoop(&loopOp.getRegion(), forOps.back().getBody(),
                            rewriter.getInsertionPoint(), mapping, rewriter);

  // Denormalize IV uses: original_iv = normalized_iv * orig_step + orig_lb
  for (size_t idx = 0; idx < forOps.size(); ++idx) {
    Value iv = forOps[idx].getInductionVar();
    if (!iv.use_empty()) {
      rewriter.setInsertionPointToStart(forOps[idx].getBody());
      normalizeIVUses(rewriter, loc, iv, loopOp.getLowerbound()[idx],
                      loopOp.getStep()[idx]);
    }
  }

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
    cloneACCRegionIntoForLoop(&loopOp.getRegion(), parallelOp.getBody(),
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
