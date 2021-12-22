//===- Utils.cpp ---- Utilities for affine dialect transformation ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements miscellaneous transformation utilities for the Affine
// dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Utils.h"

#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopUtils.h"

using namespace mlir;

/// Promotes the `then` or the `else` block of `ifOp` (depending on whether
/// `elseBlock` is false or true) into `ifOp`'s containing block, and discards
/// the rest of the op.
static void promoteIfBlock(AffineIfOp ifOp, bool elseBlock) {
  if (elseBlock)
    assert(ifOp.hasElse() && "else block expected");

  Block *destBlock = ifOp->getBlock();
  Block *srcBlock = elseBlock ? ifOp.getElseBlock() : ifOp.getThenBlock();
  destBlock->getOperations().splice(
      Block::iterator(ifOp), srcBlock->getOperations(), srcBlock->begin(),
      std::prev(srcBlock->end()));
  ifOp.erase();
}

/// Returns the outermost affine.for/parallel op that the `ifOp` is invariant
/// on. The `ifOp` could be hoisted and placed right before such an operation.
/// This method assumes that the ifOp has been canonicalized (to be correct and
/// effective).
static Operation *getOutermostInvariantForOp(AffineIfOp ifOp) {
  // Walk up the parents past all for op that this conditional is invariant on.
  auto ifOperands = ifOp.getOperands();
  auto *res = ifOp.getOperation();
  while (!isa<FuncOp>(res->getParentOp())) {
    auto *parentOp = res->getParentOp();
    if (auto forOp = dyn_cast<AffineForOp>(parentOp)) {
      if (llvm::is_contained(ifOperands, forOp.getInductionVar()))
        break;
    } else if (auto parallelOp = dyn_cast<AffineParallelOp>(parentOp)) {
      for (auto iv : parallelOp.getIVs())
        if (llvm::is_contained(ifOperands, iv))
          break;
    } else if (!isa<AffineIfOp>(parentOp)) {
      // Won't walk up past anything other than affine.for/if ops.
      break;
    }
    // You can always hoist up past any affine.if ops.
    res = parentOp;
  }
  return res;
}

/// A helper for the mechanics of mlir::hoistAffineIfOp. Hoists `ifOp` just over
/// `hoistOverOp`. Returns the new hoisted op if any hoisting happened,
/// otherwise the same `ifOp`.
static AffineIfOp hoistAffineIfOp(AffineIfOp ifOp, Operation *hoistOverOp) {
  // No hoisting to do.
  if (hoistOverOp == ifOp)
    return ifOp;

  // Create the hoisted 'if' first. Then, clone the op we are hoisting over for
  // the else block. Then drop the else block of the original 'if' in the 'then'
  // branch while promoting its then block, and analogously drop the 'then'
  // block of the original 'if' from the 'else' branch while promoting its else
  // block.
  BlockAndValueMapping operandMap;
  OpBuilder b(hoistOverOp);
  auto hoistedIfOp = b.create<AffineIfOp>(ifOp.getLoc(), ifOp.getIntegerSet(),
                                          ifOp.getOperands(),
                                          /*elseBlock=*/true);

  // Create a clone of hoistOverOp to use for the else branch of the hoisted
  // conditional. The else block may get optimized away if empty.
  Operation *hoistOverOpClone = nullptr;
  // We use this unique name to identify/find  `ifOp`'s clone in the else
  // version.
  StringAttr idForIfOp = b.getStringAttr("__mlir_if_hoisting");
  operandMap.clear();
  b.setInsertionPointAfter(hoistOverOp);
  // We'll set an attribute to identify this op in a clone of this sub-tree.
  ifOp->setAttr(idForIfOp, b.getBoolAttr(true));
  hoistOverOpClone = b.clone(*hoistOverOp, operandMap);

  // Promote the 'then' block of the original affine.if in the then version.
  promoteIfBlock(ifOp, /*elseBlock=*/false);

  // Move the then version to the hoisted if op's 'then' block.
  auto *thenBlock = hoistedIfOp.getThenBlock();
  thenBlock->getOperations().splice(thenBlock->begin(),
                                    hoistOverOp->getBlock()->getOperations(),
                                    Block::iterator(hoistOverOp));

  // Find the clone of the original affine.if op in the else version.
  AffineIfOp ifCloneInElse;
  hoistOverOpClone->walk([&](AffineIfOp ifClone) {
    if (!ifClone->getAttr(idForIfOp))
      return WalkResult::advance();
    ifCloneInElse = ifClone;
    return WalkResult::interrupt();
  });
  assert(ifCloneInElse && "if op clone should exist");
  // For the else block, promote the else block of the original 'if' if it had
  // one; otherwise, the op itself is to be erased.
  if (!ifCloneInElse.hasElse())
    ifCloneInElse.erase();
  else
    promoteIfBlock(ifCloneInElse, /*elseBlock=*/true);

  // Move the else version into the else block of the hoisted if op.
  auto *elseBlock = hoistedIfOp.getElseBlock();
  elseBlock->getOperations().splice(
      elseBlock->begin(), hoistOverOpClone->getBlock()->getOperations(),
      Block::iterator(hoistOverOpClone));

  return hoistedIfOp;
}

LogicalResult
mlir::affineParallelize(AffineForOp forOp,
                        ArrayRef<LoopReduction> parallelReductions) {
  // Fail early if there are iter arguments that are not reductions.
  unsigned numReductions = parallelReductions.size();
  if (numReductions != forOp.getNumIterOperands())
    return failure();

  Location loc = forOp.getLoc();
  OpBuilder outsideBuilder(forOp);
  AffineMap lowerBoundMap = forOp.getLowerBoundMap();
  ValueRange lowerBoundOperands = forOp.getLowerBoundOperands();
  AffineMap upperBoundMap = forOp.getUpperBoundMap();
  ValueRange upperBoundOperands = forOp.getUpperBoundOperands();

  // Creating empty 1-D affine.parallel op.
  auto reducedValues = llvm::to_vector<4>(llvm::map_range(
      parallelReductions, [](const LoopReduction &red) { return red.value; }));
  auto reductionKinds = llvm::to_vector<4>(llvm::map_range(
      parallelReductions, [](const LoopReduction &red) { return red.kind; }));
  AffineParallelOp newPloop = outsideBuilder.create<AffineParallelOp>(
      loc, ValueRange(reducedValues).getTypes(), reductionKinds,
      llvm::makeArrayRef(lowerBoundMap), lowerBoundOperands,
      llvm::makeArrayRef(upperBoundMap), upperBoundOperands,
      llvm::makeArrayRef(forOp.getStep()));
  // Steal the body of the old affine for op.
  newPloop.region().takeBody(forOp.region());
  Operation *yieldOp = &newPloop.getBody()->back();

  // Handle the initial values of reductions because the parallel loop always
  // starts from the neutral value.
  SmallVector<Value> newResults;
  newResults.reserve(numReductions);
  for (unsigned i = 0; i < numReductions; ++i) {
    Value init = forOp.getIterOperands()[i];
    // This works because we are only handling single-op reductions at the
    // moment. A switch on reduction kind or a mechanism to collect operations
    // participating in the reduction will be necessary for multi-op reductions.
    Operation *reductionOp = yieldOp->getOperand(i).getDefiningOp();
    assert(reductionOp && "yielded value is expected to be produced by an op");
    outsideBuilder.getInsertionBlock()->getOperations().splice(
        outsideBuilder.getInsertionPoint(), newPloop.getBody()->getOperations(),
        reductionOp);
    reductionOp->setOperands({init, newPloop->getResult(i)});
    forOp->getResult(i).replaceAllUsesWith(reductionOp->getResult(0));
  }

  // Update the loop terminator to yield reduced values bypassing the reduction
  // operation itself (now moved outside of the loop) and erase the block
  // arguments that correspond to reductions. Note that the loop always has one
  // "main" induction variable whenc coming from a non-parallel for.
  unsigned numIVs = 1;
  yieldOp->setOperands(reducedValues);
  newPloop.getBody()->eraseArguments(
      llvm::to_vector<4>(llvm::seq<unsigned>(numIVs, numReductions + numIVs)));

  forOp.erase();
  return success();
}

// Returns success if any hoisting happened.
LogicalResult mlir::hoistAffineIfOp(AffineIfOp ifOp, bool *folded) {
  // Bail out early if the ifOp returns a result.  TODO: Consider how to
  // properly support this case.
  if (ifOp.getNumResults() != 0)
    return failure();

  // Apply canonicalization patterns and folding - this is necessary for the
  // hoisting check to be correct (operands should be composed), and to be more
  // effective (no unused operands). Since the pattern rewriter's folding is
  // entangled with application of patterns, we may fold/end up erasing the op,
  // in which case we return with `folded` being set.
  RewritePatternSet patterns(ifOp.getContext());
  AffineIfOp::getCanonicalizationPatterns(patterns, ifOp.getContext());
  bool erased;
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  (void)applyOpPatternsAndFold(ifOp, frozenPatterns, &erased);
  if (erased) {
    if (folded)
      *folded = true;
    return failure();
  }
  if (folded)
    *folded = false;

  // The folding above should have ensured this, but the affine.if's
  // canonicalization is missing composition of affine.applys into it.
  assert(llvm::all_of(ifOp.getOperands(),
                      [](Value v) {
                        return isTopLevelValue(v) || isForInductionVar(v);
                      }) &&
         "operands not composed");

  // We are going hoist as high as possible.
  // TODO: this could be customized in the future.
  auto *hoistOverOp = getOutermostInvariantForOp(ifOp);

  AffineIfOp hoistedIfOp = ::hoistAffineIfOp(ifOp, hoistOverOp);
  // Nothing to hoist over.
  if (hoistedIfOp == ifOp)
    return failure();

  // Canonicalize to remove dead else blocks (happens whenever an 'if' moves up
  // a sequence of affine.fors that are all perfectly nested).
  (void)applyPatternsAndFoldGreedily(
      hoistedIfOp->getParentWithTrait<OpTrait::IsIsolatedFromAbove>(),
      frozenPatterns);

  return success();
}

// Return the min expr after replacing the given dim.
AffineExpr mlir::substWithMin(AffineExpr e, AffineExpr dim, AffineExpr min,
                              AffineExpr max, bool positivePath) {
  if (e == dim)
    return positivePath ? min : max;
  if (auto bin = e.dyn_cast<AffineBinaryOpExpr>()) {
    AffineExpr lhs = bin.getLHS();
    AffineExpr rhs = bin.getRHS();
    if (bin.getKind() == mlir::AffineExprKind::Add)
      return substWithMin(lhs, dim, min, max, positivePath) +
             substWithMin(rhs, dim, min, max, positivePath);

    auto c1 = bin.getLHS().dyn_cast<AffineConstantExpr>();
    auto c2 = bin.getRHS().dyn_cast<AffineConstantExpr>();
    if (c1 && c1.getValue() < 0)
      return getAffineBinaryOpExpr(
          bin.getKind(), c1, substWithMin(rhs, dim, min, max, !positivePath));
    if (c2 && c2.getValue() < 0)
      return getAffineBinaryOpExpr(
          bin.getKind(), substWithMin(lhs, dim, min, max, !positivePath), c2);
    return getAffineBinaryOpExpr(
        bin.getKind(), substWithMin(lhs, dim, min, max, positivePath),
        substWithMin(rhs, dim, min, max, positivePath));
  }
  return e;
}

void mlir::normalizeAffineParallel(AffineParallelOp op) {
  // Loops with min/max in bounds are not normalized at the moment.
  if (op.hasMinMaxBounds())
    return;

  AffineMap lbMap = op.lowerBoundsMap();
  SmallVector<int64_t, 8> steps = op.getSteps();
  // No need to do any work if the parallel op is already normalized.
  bool isAlreadyNormalized =
      llvm::all_of(llvm::zip(steps, lbMap.getResults()), [](auto tuple) {
        int64_t step = std::get<0>(tuple);
        auto lbExpr =
            std::get<1>(tuple).template dyn_cast<AffineConstantExpr>();
        return lbExpr && lbExpr.getValue() == 0 && step == 1;
      });
  if (isAlreadyNormalized)
    return;

  AffineValueMap ranges;
  AffineValueMap::difference(op.getUpperBoundsValueMap(),
                             op.getLowerBoundsValueMap(), &ranges);
  auto builder = OpBuilder::atBlockBegin(op.getBody());
  auto zeroExpr = builder.getAffineConstantExpr(0);
  SmallVector<AffineExpr, 8> lbExprs;
  SmallVector<AffineExpr, 8> ubExprs;
  for (unsigned i = 0, e = steps.size(); i < e; ++i) {
    int64_t step = steps[i];

    // Adjust the lower bound to be 0.
    lbExprs.push_back(zeroExpr);

    // Adjust the upper bound expression: 'range / step'.
    AffineExpr ubExpr = ranges.getResult(i).ceilDiv(step);
    ubExprs.push_back(ubExpr);

    // Adjust the corresponding IV: 'lb + i * step'.
    BlockArgument iv = op.getBody()->getArgument(i);
    AffineExpr lbExpr = lbMap.getResult(i);
    unsigned nDims = lbMap.getNumDims();
    auto expr = lbExpr + builder.getAffineDimExpr(nDims) * step;
    auto map = AffineMap::get(/*dimCount=*/nDims + 1,
                              /*symbolCount=*/lbMap.getNumSymbols(), expr);

    // Use an 'affine.apply' op that will be simplified later in subsequent
    // canonicalizations.
    OperandRange lbOperands = op.getLowerBoundsOperands();
    OperandRange dimOperands = lbOperands.take_front(nDims);
    OperandRange symbolOperands = lbOperands.drop_front(nDims);
    SmallVector<Value, 8> applyOperands{dimOperands};
    applyOperands.push_back(iv);
    applyOperands.append(symbolOperands.begin(), symbolOperands.end());
    auto apply = builder.create<AffineApplyOp>(op.getLoc(), map, applyOperands);
    iv.replaceAllUsesExcept(apply, apply);
  }

  SmallVector<int64_t, 8> newSteps(op.getNumDims(), 1);
  op.setSteps(newSteps);
  auto newLowerMap = AffineMap::get(
      /*dimCount=*/0, /*symbolCount=*/0, lbExprs, op.getContext());
  op.setLowerBounds({}, newLowerMap);
  auto newUpperMap = AffineMap::get(ranges.getNumDims(), ranges.getNumSymbols(),
                                    ubExprs, op.getContext());
  op.setUpperBounds(ranges.getOperands(), newUpperMap);
}

/// Normalizes affine.for ops. If the affine.for op has only a single iteration
/// only then it is simply promoted, else it is normalized in the traditional
/// way, by converting the lower bound to zero and loop step to one. The upper
/// bound is set to the trip count of the loop. For now, original loops must
/// have lower bound with a single result only. There is no such restriction on
/// upper bounds.
void mlir::normalizeAffineFor(AffineForOp op) {
  if (succeeded(promoteIfSingleIteration(op)))
    return;

  // Check if the forop is already normalized.
  if (op.hasConstantLowerBound() && (op.getConstantLowerBound() == 0) &&
      (op.getStep() == 1))
    return;

  // Check if the lower bound has a single result only. Loops with a max lower
  // bound can't be normalized without additional support like
  // affine.execute_region's. If the lower bound does not have a single result
  // then skip this op.
  if (op.getLowerBoundMap().getNumResults() != 1)
    return;

  Location loc = op.getLoc();
  OpBuilder opBuilder(op);
  int64_t origLoopStep = op.getStep();

  // Calculate upperBound for normalized loop.
  SmallVector<Value, 4> ubOperands;
  AffineBound lb = op.getLowerBound();
  AffineBound ub = op.getUpperBound();
  ubOperands.reserve(ub.getNumOperands() + lb.getNumOperands());
  AffineMap origLbMap = lb.getMap();
  AffineMap origUbMap = ub.getMap();

  // Add dimension operands from upper/lower bound.
  for (unsigned j = 0, e = origUbMap.getNumDims(); j < e; ++j)
    ubOperands.push_back(ub.getOperand(j));
  for (unsigned j = 0, e = origLbMap.getNumDims(); j < e; ++j)
    ubOperands.push_back(lb.getOperand(j));

  // Add symbol operands from upper/lower bound.
  for (unsigned j = 0, e = origUbMap.getNumSymbols(); j < e; ++j)
    ubOperands.push_back(ub.getOperand(origUbMap.getNumDims() + j));
  for (unsigned j = 0, e = origLbMap.getNumSymbols(); j < e; ++j)
    ubOperands.push_back(lb.getOperand(origLbMap.getNumDims() + j));

  // Add original result expressions from lower/upper bound map.
  SmallVector<AffineExpr, 1> origLbExprs(origLbMap.getResults().begin(),
                                         origLbMap.getResults().end());
  SmallVector<AffineExpr, 2> origUbExprs(origUbMap.getResults().begin(),
                                         origUbMap.getResults().end());
  SmallVector<AffineExpr, 4> newUbExprs;

  // The original upperBound can have more than one result. For the new
  // upperBound of this loop, take difference of all possible combinations of
  // the ub results and lb result and ceildiv with the loop step. For e.g.,
  //
  //  affine.for %i1 = 0 to min affine_map<(d0)[] -> (d0 + 32, 1024)>(%i0)
  //  will have an upperBound map as,
  //  affine_map<(d0)[] -> (((d0 + 32) - 0) ceildiv 1, (1024 - 0) ceildiv
  //  1)>(%i0)
  //
  // Insert all combinations of upper/lower bound results.
  for (unsigned i = 0, e = origUbExprs.size(); i < e; ++i) {
    newUbExprs.push_back(
        (origUbExprs[i] - origLbExprs[0]).ceilDiv(origLoopStep));
  }

  // Construct newUbMap.
  AffineMap newUbMap =
      AffineMap::get(origLbMap.getNumDims() + origUbMap.getNumDims(),
                     origLbMap.getNumSymbols() + origUbMap.getNumSymbols(),
                     newUbExprs, opBuilder.getContext());

  // Normalize the loop.
  op.setUpperBound(ubOperands, newUbMap);
  op.setLowerBound({}, opBuilder.getConstantAffineMap(0));
  op.setStep(1);

  // Calculate the Value of new loopIV. Create affine.apply for the value of
  // the loopIV in normalized loop.
  opBuilder.setInsertionPointToStart(op.getBody());
  SmallVector<Value, 4> lbOperands(lb.getOperands().begin(),
                                   lb.getOperands().begin() +
                                       lb.getMap().getNumDims());
  // Add an extra dim operand for loopIV.
  lbOperands.push_back(op.getInductionVar());
  // Add symbol operands from lower bound.
  for (unsigned j = 0, e = origLbMap.getNumSymbols(); j < e; ++j)
    lbOperands.push_back(lb.getOperand(origLbMap.getNumDims() + j));

  AffineExpr origIVExpr = opBuilder.getAffineDimExpr(lb.getMap().getNumDims());
  AffineExpr newIVExpr = origIVExpr * origLoopStep + origLbMap.getResult(0);
  AffineMap ivMap = AffineMap::get(origLbMap.getNumDims() + 1,
                                   origLbMap.getNumSymbols(), newIVExpr);
  Operation *newIV = opBuilder.create<AffineApplyOp>(loc, ivMap, lbOperands);
  op.getInductionVar().replaceAllUsesExcept(newIV->getResult(0), newIV);
}

/// Ensure that all operations that could be executed after `start`
/// (noninclusive) and prior to `memOp` (e.g. on a control flow/op path
/// between the operations) do not have the potential memory effect
/// `EffectType` on `memOp`. `memOp`  is an operation that reads or writes to
/// a memref. For example, if `EffectType` is MemoryEffects::Write, this method
/// will check if there is no write to the memory between `start` and `memOp`
/// that would change the read within `memOp`.
template <typename EffectType, typename T>
static bool hasNoInterveningEffect(Operation *start, T memOp) {
  Value memref = memOp.getMemRef();
  bool isOriginalAllocation = memref.getDefiningOp<memref::AllocaOp>() ||
                              memref.getDefiningOp<memref::AllocOp>();

  // A boolean representing whether an intervening operation could have impacted
  // memOp.
  bool hasSideEffect = false;

  // Check whether the effect on memOp can be caused by a given operation op.
  std::function<void(Operation *)> checkOperation = [&](Operation *op) {
    // If the effect has alreay been found, early exit,
    if (hasSideEffect)
      return;

    if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(op)) {
      SmallVector<MemoryEffects::EffectInstance, 1> effects;
      memEffect.getEffects(effects);

      bool opMayHaveEffect = false;
      for (auto effect : effects) {
        // If op causes EffectType on a potentially aliasing location for
        // memOp, mark as having the effect.
        if (isa<EffectType>(effect.getEffect())) {
          if (isOriginalAllocation && effect.getValue() &&
              (effect.getValue().getDefiningOp<memref::AllocaOp>() ||
               effect.getValue().getDefiningOp<memref::AllocOp>())) {
            if (effect.getValue() != memref)
              continue;
          }
          opMayHaveEffect = true;
          break;
        }
      }

      if (!opMayHaveEffect)
        return;

      // If the side effect comes from an affine read or write, try to
      // prove the side effecting `op` cannot reach `memOp`.
      if (isa<AffineReadOpInterface, AffineWriteOpInterface>(op)) {
        MemRefAccess srcAccess(op);
        MemRefAccess destAccess(memOp);
        // Dependence analysis is only correct if both ops operate on the same
        // memref.
        if (srcAccess.memref == destAccess.memref) {
          FlatAffineValueConstraints dependenceConstraints;

          // Number of loops containing the start op and the ending operation.
          unsigned minSurroundingLoops =
              getNumCommonSurroundingLoops(*start, *memOp);

          // Number of loops containing the operation `op` which has the
          // potential memory side effect and can occur on a path between
          // `start` and `memOp`.
          unsigned nsLoops = getNumCommonSurroundingLoops(*op, *memOp);

          // For ease, let's consider the case that `op` is a store and we're
          // looking for other potential stores (e.g `op`) that overwrite memory
          // after `start`, and before being read in `memOp`. In this case, we
          // only need to consider other potential stores with depth >
          // minSurrounding loops since `start` would overwrite any store with a
          // smaller number of surrounding loops before.
          unsigned d;
          for (d = nsLoops + 1; d > minSurroundingLoops; d--) {
            DependenceResult result = checkMemrefAccessDependence(
                srcAccess, destAccess, d, &dependenceConstraints,
                /*dependenceComponents=*/nullptr);
            if (hasDependence(result)) {
              hasSideEffect = true;
              return;
            }
          }

          // No side effect was seen, simply return.
          return;
        }
      }
      hasSideEffect = true;
      return;
    }

    if (op->hasTrait<OpTrait::HasRecursiveSideEffects>()) {
      // Recurse into the regions for this op and check whether the internal
      // operations may have the side effect `EffectType` on memOp.
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (Operation &op : block)
            checkOperation(&op);
      return;
    }

    // Otherwise, conservatively assume generic operations have the effect
    // on the operation
    hasSideEffect = true;
    return;
  };

  // Check all paths from ancestor op `parent` to the operation `to` for the
  // effect. It is known that `to` must be contained within `parent`.
  auto until = [&](Operation *parent, Operation *to) {
    // TODO check only the paths from `parent` to `to`.
    // Currently we fallback and check the entire parent op, rather than
    // just the paths from the parent path, stopping after reaching `to`.
    // This is conservatively correct, but could be made more aggressive.
    assert(parent->isAncestor(to));
    checkOperation(parent);
  };

  // Check for all paths from operation `from` to operation `untilOp` for the
  // given memory effect.
  std::function<void(Operation *, Operation *)> recur =
      [&](Operation *from, Operation *untilOp) {
        assert(
            from->getParentRegion()->isAncestor(untilOp->getParentRegion()) &&
            "Checking for side effect between two operations without a common "
            "ancestor");

        // If the operations are in different regions, recursively consider all
        // path from `from` to the parent of `to` and all paths from the parent
        // of `to` to `to`.
        if (from->getParentRegion() != untilOp->getParentRegion()) {
          recur(from, untilOp->getParentOp());
          until(untilOp->getParentOp(), untilOp);
          return;
        }

        // Now, assuming that `from` and `to` exist in the same region, perform
        // a CFG traversal to check all the relevant operations.

        // Additional blocks to consider.
        SmallVector<Block *, 2> todoBlocks;
        {
          // First consider the parent block of `from` an check all operations
          // after `from`.
          for (auto iter = ++from->getIterator(), end = from->getBlock()->end();
               iter != end && &*iter != untilOp; ++iter) {
            checkOperation(&*iter);
          }

          // If the parent of `from` doesn't contain `to`, add the successors
          // to the list of blocks to check.
          if (untilOp->getBlock() != from->getBlock())
            for (Block *succ : from->getBlock()->getSuccessors())
              todoBlocks.push_back(succ);
        }

        SmallPtrSet<Block *, 4> done;
        // Traverse the CFG until hitting `to`.
        while (!todoBlocks.empty()) {
          Block *blk = todoBlocks.pop_back_val();
          if (done.count(blk))
            continue;
          done.insert(blk);
          for (auto &op : *blk) {
            if (&op == untilOp)
              break;
            checkOperation(&op);
            if (&op == blk->getTerminator())
              for (Block *succ : blk->getSuccessors())
                todoBlocks.push_back(succ);
          }
        }
      };
  recur(start, memOp);
  return !hasSideEffect;
}

/// Attempt to eliminate loadOp by replacing it with a value stored into memory
/// which the load is guaranteed to retrieve. This check involves three
/// components: 1) The store and load must be on the same location 2) The store
/// must dominate (and therefore must always occur prior to) the load 3) No
/// other operations will overwrite the memory loaded between the given load
/// and store.  If such a value exists, the replaced `loadOp` will be added to
/// `loadOpsToErase` and its memref will be added to `memrefsToErase`.
static LogicalResult forwardStoreToLoad(
    AffineReadOpInterface loadOp, SmallVectorImpl<Operation *> &loadOpsToErase,
    SmallPtrSetImpl<Value> &memrefsToErase, DominanceInfo &domInfo) {

  // The store op candidate for forwarding that satisfies all conditions
  // to replace the load, if any.
  Operation *lastWriteStoreOp = nullptr;

  for (auto *user : loadOp.getMemRef().getUsers()) {
    auto storeOp = dyn_cast<AffineWriteOpInterface>(user);
    if (!storeOp)
      continue;
    MemRefAccess srcAccess(storeOp);
    MemRefAccess destAccess(loadOp);

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
    if (!domInfo.dominates(storeOp, loadOp))
      continue;

    // 3. Ensure there is no intermediate operation which could replace the
    // value in memory.
    if (!hasNoInterveningEffect<MemoryEffects::Write>(storeOp, loadOp))
      continue;

    // We now have a candidate for forwarding.
    assert(lastWriteStoreOp == nullptr &&
           "multiple simulataneous replacement stores");
    lastWriteStoreOp = storeOp;
  }

  if (!lastWriteStoreOp)
    return failure();

  // Perform the actual store to load forwarding.
  Value storeVal =
      cast<AffineWriteOpInterface>(lastWriteStoreOp).getValueToStore();
  // Check if 2 values have the same shape. This is needed for affine vector
  // loads and stores.
  if (storeVal.getType() != loadOp.getValue().getType())
    return failure();
  loadOp.getValue().replaceAllUsesWith(storeVal);
  // Record the memref for a later sweep to optimize away.
  memrefsToErase.insert(loadOp.getMemRef());
  // Record this to erase later.
  loadOpsToErase.push_back(loadOp);
  return success();
}

// This attempts to find stores which have no impact on the final result.
// A writing op writeA will be eliminated if there exists an op writeB if
// 1) writeA and writeB have mathematically equivalent affine access functions.
// 2) writeB postdominates writeA.
// 3) There is no potential read between writeA and writeB.
static void findUnusedStore(AffineWriteOpInterface writeA,
                            SmallVectorImpl<Operation *> &opsToErase,
                            SmallPtrSetImpl<Value> &memrefsToErase,
                            PostDominanceInfo &postDominanceInfo) {

  for (Operation *user : writeA.getMemRef().getUsers()) {
    // Only consider writing operations.
    auto writeB = dyn_cast<AffineWriteOpInterface>(user);
    if (!writeB)
      continue;

    // The operations must be distinct.
    if (writeB == writeA)
      continue;

    // Both operations must lie in the same region.
    if (writeB->getParentRegion() != writeA->getParentRegion())
      continue;

    // Both operations must write to the same memory.
    MemRefAccess srcAccess(writeB);
    MemRefAccess destAccess(writeA);

    if (srcAccess != destAccess)
      continue;

    // writeB must postdominate writeA.
    if (!postDominanceInfo.postDominates(writeB, writeA))
      continue;

    // There cannot be an operation which reads from memory between
    // the two writes.
    if (!hasNoInterveningEffect<MemoryEffects::Read>(writeA, writeB))
      continue;

    opsToErase.push_back(writeA);
    break;
  }
}

// The load to load forwarding / redundant load elimination is similar to the
// store to load forwarding.
// loadA will be be replaced with loadB if:
// 1) loadA and loadB have mathematically equivalent affine access functions.
// 2) loadB dominates loadA.
// 3) There is no write between loadA and loadB.
static void loadCSE(AffineReadOpInterface loadA,
                    SmallVectorImpl<Operation *> &loadOpsToErase,
                    DominanceInfo &domInfo) {
  SmallVector<AffineReadOpInterface, 4> loadCandidates;
  for (auto *user : loadA.getMemRef().getUsers()) {
    auto loadB = dyn_cast<AffineReadOpInterface>(user);
    if (!loadB || loadB == loadA)
      continue;

    MemRefAccess srcAccess(loadB);
    MemRefAccess destAccess(loadA);

    // 1. The accesses have to be to the same location.
    if (srcAccess != destAccess) {
      continue;
    }

    // 2. The store has to dominate the load op to be candidate.
    if (!domInfo.dominates(loadB, loadA))
      continue;

    // 3. There is no write between loadA and loadB.
    if (!hasNoInterveningEffect<MemoryEffects::Write>(loadB.getOperation(),
                                                      loadA))
      continue;

    // Check if two values have the same shape. This is needed for affine vector
    // loads.
    if (loadB.getValue().getType() != loadA.getValue().getType())
      continue;

    loadCandidates.push_back(loadB);
  }

  // Of the legal load candidates, use the one that dominates all others
  // to minimize the subsequent need to loadCSE
  Value loadB;
  for (AffineReadOpInterface option : loadCandidates) {
    if (llvm::all_of(loadCandidates, [&](AffineReadOpInterface depStore) {
          return depStore == option ||
                 domInfo.dominates(option.getOperation(),
                                   depStore.getOperation());
        })) {
      loadB = option.getValue();
      break;
    }
  }

  if (loadB) {
    loadA.getValue().replaceAllUsesWith(loadB);
    // Record this to erase later.
    loadOpsToErase.push_back(loadA);
  }
}

// The store to load forwarding and load CSE rely on three conditions:
//
// 1) store/load providing a replacement value and load being replaced need to
// have mathematically equivalent affine access functions (checked after full
// composition of load/store operands); this implies that they access the same
// single memref element for all iterations of the common surrounding loop,
//
// 2) the store/load op should dominate the load op,
//
// 3) no operation that may write to memory read by the load being replaced can
// occur after executing the instruction (load or store) providing the
// replacement value and before the load being replaced (thus potentially
// allowing overwriting the memory read by the load).
//
// The above conditions are simple to check, sufficient, and powerful for most
// cases in practice - they are sufficient, but not necessary --- since they
// don't reason about loops that are guaranteed to execute at least once or
// multiple sources to forward from.
//
// TODO: more forwarding can be done when support for
// loop/conditional live-out SSA values is available.
// TODO: do general dead store elimination for memref's. This pass
// currently only eliminates the stores only if no other loads/uses (other
// than dealloc) remain.
//
void mlir::affineScalarReplace(FuncOp f, DominanceInfo &domInfo,
                               PostDominanceInfo &postDomInfo) {
  // Load op's whose results were replaced by those forwarded from stores.
  SmallVector<Operation *, 8> opsToErase;

  // A list of memref's that are potentially dead / could be eliminated.
  SmallPtrSet<Value, 4> memrefsToErase;

  // Walk all load's and perform store to load forwarding.
  f.walk([&](AffineReadOpInterface loadOp) {
    if (failed(
            forwardStoreToLoad(loadOp, opsToErase, memrefsToErase, domInfo))) {
      loadCSE(loadOp, opsToErase, domInfo);
    }
  });

  // Erase all load op's whose results were replaced with store fwd'ed ones.
  for (auto *op : opsToErase)
    op->erase();
  opsToErase.clear();

  // Walk all store's and perform unused store elimination
  f.walk([&](AffineWriteOpInterface storeOp) {
    findUnusedStore(storeOp, opsToErase, memrefsToErase, postDomInfo);
  });
  // Erase all store op's which don't impact the program
  for (auto *op : opsToErase)
    op->erase();

  // Check if the store fwd'ed memrefs are now left with only stores and can
  // thus be completely deleted. Note: the canonicalize pass should be able
  // to do this as well, but we'll do it here since we collected these anyway.
  for (auto memref : memrefsToErase) {
    // If the memref hasn't been alloc'ed in this function, skip.
    Operation *defOp = memref.getDefiningOp();
    if (!defOp || !isa<memref::AllocOp>(defOp))
      // TODO: if the memref was returned by a 'call' operation, we
      // could still erase it if the call had no side-effects.
      continue;
    if (llvm::any_of(memref.getUsers(), [&](Operation *ownerOp) {
          return !isa<AffineWriteOpInterface, memref::DeallocOp>(ownerOp);
        }))
      continue;

    // Erase all stores, the dealloc, and the alloc on the memref.
    for (auto *user : llvm::make_early_inc_range(memref.getUsers()))
      user->erase();
    defOp->erase();
  }
}
