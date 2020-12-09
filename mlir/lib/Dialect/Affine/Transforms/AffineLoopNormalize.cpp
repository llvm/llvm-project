//===- AffineLoopNormalize.cpp - AffineLoopNormalize Pass -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a normalizer for affine loop-like ops.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/LoopUtils.h"

using namespace mlir;

void mlir::normalizeAffineParallel(AffineParallelOp op) {
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

  AffineValueMap ranges = op.getRangesValueMap();
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
    iv.replaceAllUsesExcept(apply, SmallPtrSet<Operation *, 1>{apply});
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
static void normalizeAffineFor(AffineForOp op) {
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
  op.getInductionVar().replaceAllUsesExcept(newIV->getResult(0),
                                            SmallPtrSet<Operation *, 1>{newIV});
}

namespace {

/// Normalize affine.parallel ops so that lower bounds are 0 and steps are 1.
/// As currently implemented, this pass cannot fail, but it might skip over ops
/// that are already in a normalized form.
struct AffineLoopNormalizePass
    : public AffineLoopNormalizeBase<AffineLoopNormalizePass> {

  void runOnFunction() override {
    getFunction().walk([](Operation *op) {
      if (auto affineParallel = dyn_cast<AffineParallelOp>(op))
        normalizeAffineParallel(affineParallel);
      else if (auto affineFor = dyn_cast<AffineForOp>(op))
        normalizeAffineFor(affineFor);
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::createAffineLoopNormalizePass() {
  return std::make_unique<AffineLoopNormalizePass>();
}
