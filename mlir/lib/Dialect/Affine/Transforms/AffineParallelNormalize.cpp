//===- AffineParallelNormalize.cpp - AffineParallelNormalize Pass ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a normalizer for affine parallel loops.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

void normalizeAffineParallel(AffineParallelOp op) {
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

namespace {

/// Normalize affine.parallel ops so that lower bounds are 0 and steps are 1.
/// As currently implemented, this pass cannot fail, but it might skip over ops
/// that are already in a normalized form.
struct AffineParallelNormalizePass
    : public AffineParallelNormalizeBase<AffineParallelNormalizePass> {

  void runOnFunction() override { getFunction().walk(normalizeAffineParallel); }
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::createAffineParallelNormalizePass() {
  return std::make_unique<AffineParallelNormalizePass>();
}
