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

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/STLExtras.h"
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
      b.create<scf::ForOp>(loop.getLoc(), loop.getLowerBound(),
                           loop.getUpperBound(), loop.getStep(), operands);

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

/// Outline a region with a single block into a new FuncOp.
/// Assumes the FuncOp result types is the type of the yielded operands of the
/// single block. This constraint makes it easy to determine the result.
/// This method also clones the `arith::ConstantIndexOp` at the start of
/// `outlinedFuncBody` to alloc simple canonicalizations.
// TODO: support more than single-block regions.
// TODO: more flexible constant handling.
FailureOr<FuncOp> mlir::outlineSingleBlockRegion(RewriterBase &rewriter,
                                                 Location loc, Region &region,
                                                 StringRef funcName) {
  assert(!funcName.empty() && "funcName cannot be empty");
  if (!region.hasOneBlock())
    return failure();

  Block *originalBlock = &region.front();
  Operation *originalTerminator = originalBlock->getTerminator();

  // Outline before current function.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(region.getParentOfType<FuncOp>());

  SetVector<Value> captures;
  getUsedValuesDefinedAbove(region, captures);

  ValueRange outlinedValues(captures.getArrayRef());
  SmallVector<Type> outlinedFuncArgTypes;
  SmallVector<Location> outlinedFuncArgLocs;
  // Region's arguments are exactly the first block's arguments as per
  // Region::getArguments().
  // Func's arguments are cat(regions's arguments, captures arguments).
  for (BlockArgument arg : region.getArguments()) {
    outlinedFuncArgTypes.push_back(arg.getType());
    outlinedFuncArgLocs.push_back(arg.getLoc());
  }
  for (Value value : outlinedValues) {
    outlinedFuncArgTypes.push_back(value.getType());
    outlinedFuncArgLocs.push_back(value.getLoc());
  }
  FunctionType outlinedFuncType =
      FunctionType::get(rewriter.getContext(), outlinedFuncArgTypes,
                        originalTerminator->getOperandTypes());
  auto outlinedFunc = rewriter.create<FuncOp>(loc, funcName, outlinedFuncType);
  Block *outlinedFuncBody = outlinedFunc.addEntryBlock();

  // Merge blocks while replacing the original block operands.
  // Warning: `mergeBlocks` erases the original block, reconstruct it later.
  int64_t numOriginalBlockArguments = originalBlock->getNumArguments();
  auto outlinedFuncBlockArgs = outlinedFuncBody->getArguments();
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToEnd(outlinedFuncBody);
    rewriter.mergeBlocks(
        originalBlock, outlinedFuncBody,
        outlinedFuncBlockArgs.take_front(numOriginalBlockArguments));
    // Explicitly set up a new ReturnOp terminator.
    rewriter.setInsertionPointToEnd(outlinedFuncBody);
    rewriter.create<ReturnOp>(loc, originalTerminator->getResultTypes(),
                              originalTerminator->getOperands());
  }

  // Reconstruct the block that was deleted and add a
  // terminator(call_results).
  Block *newBlock = rewriter.createBlock(
      &region, region.begin(),
      TypeRange{outlinedFuncArgTypes}.take_front(numOriginalBlockArguments),
      ArrayRef<Location>(outlinedFuncArgLocs)
          .take_front(numOriginalBlockArguments));
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToEnd(newBlock);
    SmallVector<Value> callValues;
    llvm::append_range(callValues, newBlock->getArguments());
    llvm::append_range(callValues, outlinedValues);
    Operation *call = rewriter.create<CallOp>(loc, outlinedFunc, callValues);

    // `originalTerminator` was moved to `outlinedFuncBody` and is still valid.
    // Clone `originalTerminator` to take the callOp results then erase it from
    // `outlinedFuncBody`.
    BlockAndValueMapping bvm;
    bvm.map(originalTerminator->getOperands(), call->getResults());
    rewriter.clone(*originalTerminator, bvm);
    rewriter.eraseOp(originalTerminator);
  }

  // Lastly, explicit RAUW outlinedValues, only for uses within `outlinedFunc`.
  // Clone the `arith::ConstantIndexOp` at the start of `outlinedFuncBody`.
  for (auto it : llvm::zip(outlinedValues, outlinedFuncBlockArgs.take_back(
                                               outlinedValues.size()))) {
    Value orig = std::get<0>(it);
    Value repl = std::get<1>(it);
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(outlinedFuncBody);
      if (Operation *cst = orig.getDefiningOp<arith::ConstantIndexOp>()) {
        BlockAndValueMapping bvm;
        repl = rewriter.clone(*cst, bvm)->getResult(0);
      }
    }
    orig.replaceUsesWithIf(repl, [&](OpOperand &opOperand) {
      return outlinedFunc->isProperAncestor(opOperand.getOwner());
    });
  }

  return outlinedFunc;
}

LogicalResult mlir::outlineIfOp(RewriterBase &b, scf::IfOp ifOp, FuncOp *thenFn,
                                StringRef thenFnName, FuncOp *elseFn,
                                StringRef elseFnName) {
  IRRewriter rewriter(b);
  Location loc = ifOp.getLoc();
  FailureOr<FuncOp> outlinedFuncOpOrFailure;
  if (thenFn && !ifOp.getThenRegion().empty()) {
    outlinedFuncOpOrFailure = outlineSingleBlockRegion(
        rewriter, loc, ifOp.getThenRegion(), thenFnName);
    if (failed(outlinedFuncOpOrFailure))
      return failure();
    *thenFn = *outlinedFuncOpOrFailure;
  }
  if (elseFn && !ifOp.getElseRegion().empty()) {
    outlinedFuncOpOrFailure = outlineSingleBlockRegion(
        rewriter, loc, ifOp.getElseRegion(), elseFnName);
    if (failed(outlinedFuncOpOrFailure))
      return failure();
    *elseFn = *outlinedFuncOpOrFailure;
  }
  return success();
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
