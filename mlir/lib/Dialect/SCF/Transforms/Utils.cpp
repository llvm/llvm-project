//===- LoopUtils.cpp ---- Misc utilities for loop transformation ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements miscellaneous loop transformation routines.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/Utils.h"

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/SetVector.h"

using namespace mlir;

scf::ForOp mlir::cloneWithNewYields(OpBuilder &b, scf::ForOp loop,
                                    ValueRange newIterOperands,
                                    ValueRange newYieldedValues,
                                    bool replaceLoopResults) {
  assert(newIterOperands.size() == newYieldedValues.size() &&
         "newIterOperands must be of the same size as newYieldedValues");

  // Create a new loop before the existing one, with the extra operands.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(loop);
  auto operands = llvm::to_vector<4>(loop.getIterOperands());
  operands.append(newIterOperands.begin(), newIterOperands.end());
  scf::ForOp newLoop =
      b.create<scf::ForOp>(loop.getLoc(), loop.lowerBound(), loop.upperBound(),
                           loop.step(), operands);

  auto &loopBody = *loop.getBody();
  auto &newLoopBody = *newLoop.getBody();
  // Clone / erase the yield inside the original loop to both:
  //   1. augment its operands with the newYieldedValues.
  //   2. automatically apply the BlockAndValueMapping on its operand
  auto yield = cast<scf::YieldOp>(loopBody.getTerminator());
  b.setInsertionPoint(yield);
  auto yieldOperands = llvm::to_vector<4>(yield.getOperands());
  yieldOperands.append(newYieldedValues.begin(), newYieldedValues.end());
  auto newYield = b.create<scf::YieldOp>(yield.getLoc(), yieldOperands);

  // Clone the loop body with remaps.
  BlockAndValueMapping bvm;
  // a. remap the induction variable.
  bvm.map(loop.getInductionVar(), newLoop.getInductionVar());
  // b. remap the BB args.
  bvm.map(loopBody.getArguments(),
          newLoopBody.getArguments().take_front(loopBody.getNumArguments()));
  // c. remap the iter args.
  bvm.map(newIterOperands,
          newLoop.getRegionIterArgs().take_back(newIterOperands.size()));
  b.setInsertionPointToStart(&newLoopBody);
  // Skip the original yield terminator which does not have enough operands.
  for (auto &o : loopBody.without_terminator())
    b.clone(o, bvm);

  // Replace `loop`'s results if requested.
  if (replaceLoopResults) {
    for (auto it : llvm::zip(loop.getResults(), newLoop.getResults().take_front(
                                                    loop.getNumResults())))
      std::get<0>(it).replaceAllUsesWith(std::get<1>(it));
  }

  // TODO: this is unsafe in the context of a PatternRewrite.
  newYield.erase();

  return newLoop;
}

void mlir::outlineIfOp(OpBuilder &b, scf::IfOp ifOp, FuncOp *thenFn,
                       StringRef thenFnName, FuncOp *elseFn,
                       StringRef elseFnName) {
  Location loc = ifOp.getLoc();
  MLIRContext *ctx = ifOp.getContext();
  auto outline = [&](Region &ifOrElseRegion, StringRef funcName) {
    assert(!funcName.empty() && "Expected function name for outlining");
    assert(ifOrElseRegion.getBlocks().size() <= 1 &&
           "Expected at most one block");

    // Outline before current function.
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(ifOp->getParentOfType<FuncOp>());

    SetVector<Value> captures;
    getUsedValuesDefinedAbove(ifOrElseRegion, captures);

    ValueRange values(captures.getArrayRef());
    FunctionType type =
        FunctionType::get(ctx, values.getTypes(), ifOp.getResultTypes());
    auto outlinedFunc = b.create<FuncOp>(loc, funcName, type);
    b.setInsertionPointToStart(outlinedFunc.addEntryBlock());
    BlockAndValueMapping bvm;
    for (auto it : llvm::zip(values, outlinedFunc.getArguments()))
      bvm.map(std::get<0>(it), std::get<1>(it));
    for (Operation &op : ifOrElseRegion.front().without_terminator())
      b.clone(op, bvm);

    Operation *term = ifOrElseRegion.front().getTerminator();
    SmallVector<Value, 4> terminatorOperands;
    for (auto op : term->getOperands())
      terminatorOperands.push_back(bvm.lookup(op));
    b.create<ReturnOp>(loc, term->getResultTypes(), terminatorOperands);

    ifOrElseRegion.front().clear();
    b.setInsertionPointToEnd(&ifOrElseRegion.front());
    Operation *call = b.create<CallOp>(loc, outlinedFunc, values);
    b.create<scf::YieldOp>(loc, call->getResults());
    return outlinedFunc;
  };

  if (thenFn && !ifOp.thenRegion().empty())
    *thenFn = outline(ifOp.thenRegion(), thenFnName);
  if (elseFn && !ifOp.elseRegion().empty())
    *elseFn = outline(ifOp.elseRegion(), elseFnName);
}

bool mlir::getInnermostParallelLoops(Operation *rootOp,
                                     SmallVectorImpl<scf::ParallelOp> &result) {
  assert(rootOp != nullptr && "Root operation must not be a nullptr.");
  bool rootEnclosesPloops = false;
  for (Region &region : rootOp->getRegions()) {
    for (Block &block : region.getBlocks()) {
      for (Operation &op : block) {
        bool enclosesPloops = getInnermostParallelLoops(&op, result);
        rootEnclosesPloops |= enclosesPloops;
        if (auto ploop = dyn_cast<scf::ParallelOp>(op)) {
          rootEnclosesPloops = true;

          // Collect parallel loop if it is an innermost one.
          if (!enclosesPloops)
            result.push_back(ploop);
        }
      }
    }
  }
  return rootEnclosesPloops;
}

/// Given the `lbVal`, `ubVal` and `stepVal` of a loop, append `lbVal` and
/// `ubVal` to `dims` and `stepVal` to `symbols`.
/// Create new AffineDimExpr (`%lb` and `%ub`) and AffineSymbolExpr (`%step`)
/// with positions matching the newly appended values. Then create a min
/// expression (i.e. `%lb`) and a max expression
/// (i.e. `%lb + %step * floordiv(%ub -1 - %lb, %step)`.
static std::pair<AffineExpr, AffineExpr>
getMinMaxLoopIndVar(Value lbVal, Value ubVal, Value stepVal,
                    SmallVectorImpl<Value> &dims,
                    SmallVectorImpl<Value> &symbols) {
  MLIRContext *ctx = lbVal.getContext();
  AffineExpr lb = getAffineDimExpr(dims.size(), ctx);
  dims.push_back(lbVal);
  AffineExpr ub = getAffineDimExpr(dims.size(), ctx);
  dims.push_back(ubVal);
  AffineExpr step = getAffineSymbolExpr(symbols.size(), ctx);
  symbols.push_back(stepVal);
  return std::make_pair(lb, lb + step * ((ub - 1) - lb).floorDiv(step));
}

/// Return the min/max expressions for `value` if it is an induction variable
/// from scf.for or scf.parallel loop.
/// if `loopFilter` is passed, the filter determines which loop to consider.
/// Other induction variables are ignored.
Optional<std::pair<AffineExpr, AffineExpr>> mlir::getSCFMinMaxExpr(
    Value value, SmallVectorImpl<Value> &dims, SmallVectorImpl<Value> &symbols,
    llvm::function_ref<bool(Operation *)> substituteOperation) {
  if (auto forOp = scf::getForInductionVarOwner(value))
    return getMinMaxLoopIndVar(forOp.lowerBound(), forOp.upperBound(),
                               forOp.step(), dims, symbols);

  if (auto parallelForOp = scf::getParallelForInductionVarOwner(value))
    for (unsigned idx = 0, e = parallelForOp.getNumLoops(); idx < e; ++idx)
      if (parallelForOp.getInductionVars()[idx] == value)
        return getMinMaxLoopIndVar(parallelForOp.lowerBound()[idx],
                                   parallelForOp.upperBound()[idx],
                                   parallelForOp.step()[idx], dims, symbols);
  return {};
}
