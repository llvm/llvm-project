//===- Utils.cpp ---- Misc utilities for loop transformation ----------===//
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

#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {
// This structure is to pass and return sets of loop parameters without
// confusing the order.
struct LoopParams {
  Value lowerBound;
  Value upperBound;
  Value step;
};
} // namespace

SmallVector<scf::ForOp> mlir::replaceLoopNestWithNewYields(
    RewriterBase &rewriter, MutableArrayRef<scf::ForOp> loopNest,
    ValueRange newIterOperands, const NewYieldValuesFn &newYieldValuesFn,
    bool replaceIterOperandsUsesInLoop) {
  if (loopNest.empty())
    return {};
  // This method is recursive (to make it more readable). Adding an
  // assertion here to limit the recursion. (See
  // https://discourse.llvm.org/t/rfc-update-to-mlir-developer-policy-on-recursion/62235)
  assert(loopNest.size() <= 10 &&
         "exceeded recursion limit when yielding value from loop nest");

  // To yield a value from a perfectly nested loop nest, the following
  // pattern needs to be created, i.e. starting with
  //
  // ```mlir
  //  scf.for .. {
  //    scf.for .. {
  //      scf.for .. {
  //        %value = ...
  //      }
  //    }
  //  }
  // ```
  //
  // needs to be modified to
  //
  // ```mlir
  // %0 = scf.for .. iter_args(%arg0 = %init) {
  //   %1 = scf.for .. iter_args(%arg1 = %arg0) {
  //     %2 = scf.for .. iter_args(%arg2 = %arg1) {
  //       %value = ...
  //       scf.yield %value
  //     }
  //     scf.yield %2
  //   }
  //   scf.yield %1
  // }
  // ```
  //
  // The inner most loop is handled using the `replaceWithAdditionalYields`
  // that works on a single loop.
  if (loopNest.size() == 1) {
    auto innerMostLoop =
        cast<scf::ForOp>(*loopNest.back().replaceWithAdditionalYields(
            rewriter, newIterOperands, replaceIterOperandsUsesInLoop,
            newYieldValuesFn));
    return {innerMostLoop};
  }
  // The outer loops are modified by calling this method recursively
  // - The return value of the inner loop is the value yielded by this loop.
  // - The region iter args of this loop are the init_args for the inner loop.
  SmallVector<scf::ForOp> newLoopNest;
  NewYieldValuesFn fn =
      [&](OpBuilder &innerBuilder, Location loc,
          ArrayRef<BlockArgument> innerNewBBArgs) -> SmallVector<Value> {
    newLoopNest = replaceLoopNestWithNewYields(rewriter, loopNest.drop_front(),
                                               innerNewBBArgs, newYieldValuesFn,
                                               replaceIterOperandsUsesInLoop);
    return llvm::to_vector(llvm::map_range(
        newLoopNest.front().getResults().take_back(innerNewBBArgs.size()),
        [](OpResult r) -> Value { return r; }));
  };
  scf::ForOp outerMostLoop =
      cast<scf::ForOp>(*loopNest.front().replaceWithAdditionalYields(
          rewriter, newIterOperands, replaceIterOperandsUsesInLoop, fn));
  newLoopNest.insert(newLoopNest.begin(), outerMostLoop);
  return newLoopNest;
}

/// Outline a region with a single block into a new FuncOp.
/// Assumes the FuncOp result types is the type of the yielded operands of the
/// single block. This constraint makes it easy to determine the result.
/// This method also clones the `arith::ConstantIndexOp` at the start of
/// `outlinedFuncBody` to alloc simple canonicalizations. If `callOp` is
/// provided, it will be set to point to the operation that calls the outlined
/// function.
// TODO: support more than single-block regions.
// TODO: more flexible constant handling.
FailureOr<func::FuncOp> mlir::outlineSingleBlockRegion(RewriterBase &rewriter,
                                                       Location loc,
                                                       Region &region,
                                                       StringRef funcName,
                                                       func::CallOp *callOp) {
  assert(!funcName.empty() && "funcName cannot be empty");
  if (!region.hasOneBlock())
    return failure();

  Block *originalBlock = &region.front();
  Operation *originalTerminator = originalBlock->getTerminator();

  // Outline before current function.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(region.getParentOfType<func::FuncOp>());

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
  auto outlinedFunc =
      rewriter.create<func::FuncOp>(loc, funcName, outlinedFuncType);
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
    rewriter.create<func::ReturnOp>(loc, originalTerminator->getResultTypes(),
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
    auto call = rewriter.create<func::CallOp>(loc, outlinedFunc, callValues);
    if (callOp)
      *callOp = call;

    // `originalTerminator` was moved to `outlinedFuncBody` and is still valid.
    // Clone `originalTerminator` to take the callOp results then erase it from
    // `outlinedFuncBody`.
    IRMapping bvm;
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
        IRMapping bvm;
        repl = rewriter.clone(*cst, bvm)->getResult(0);
      }
    }
    orig.replaceUsesWithIf(repl, [&](OpOperand &opOperand) {
      return outlinedFunc->isProperAncestor(opOperand.getOwner());
    });
  }

  return outlinedFunc;
}

LogicalResult mlir::outlineIfOp(RewriterBase &b, scf::IfOp ifOp,
                                func::FuncOp *thenFn, StringRef thenFnName,
                                func::FuncOp *elseFn, StringRef elseFnName) {
  IRRewriter rewriter(b);
  Location loc = ifOp.getLoc();
  FailureOr<func::FuncOp> outlinedFuncOpOrFailure;
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

// Build the IR that performs ceil division of a positive value by a constant:
//    ceildiv(a, B) = divis(a + (B-1), B)
// where divis is rounding-to-zero division.
static Value ceilDivPositive(OpBuilder &builder, Location loc, Value dividend,
                             int64_t divisor) {
  assert(divisor > 0 && "expected positive divisor");
  assert(dividend.getType().isIndex() && "expected index-typed value");

  Value divisorMinusOneCst =
      builder.create<arith::ConstantIndexOp>(loc, divisor - 1);
  Value divisorCst = builder.create<arith::ConstantIndexOp>(loc, divisor);
  Value sum = builder.create<arith::AddIOp>(loc, dividend, divisorMinusOneCst);
  return builder.create<arith::DivUIOp>(loc, sum, divisorCst);
}

// Build the IR that performs ceil division of a positive value by another
// positive value:
//    ceildiv(a, b) = divis(a + (b - 1), b)
// where divis is rounding-to-zero division.
static Value ceilDivPositive(OpBuilder &builder, Location loc, Value dividend,
                             Value divisor) {
  assert(dividend.getType().isIndex() && "expected index-typed value");

  Value cstOne = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value divisorMinusOne = builder.create<arith::SubIOp>(loc, divisor, cstOne);
  Value sum = builder.create<arith::AddIOp>(loc, dividend, divisorMinusOne);
  return builder.create<arith::DivUIOp>(loc, sum, divisor);
}

/// Generates unrolled copies of scf::ForOp 'loopBodyBlock', with
/// associated 'forOpIV' by 'unrollFactor', calling 'ivRemapFn' to remap
/// 'forOpIV' for each unrolled body. If specified, annotates the Ops in each
/// unrolled iteration using annotateFn.
static void generateUnrolledLoop(
    Block *loopBodyBlock, Value forOpIV, uint64_t unrollFactor,
    function_ref<Value(unsigned, Value, OpBuilder)> ivRemapFn,
    function_ref<void(unsigned, Operation *, OpBuilder)> annotateFn,
    ValueRange iterArgs, ValueRange yieldedValues) {
  // Builder to insert unrolled bodies just before the terminator of the body of
  // 'forOp'.
  auto builder = OpBuilder::atBlockTerminator(loopBodyBlock);

  if (!annotateFn)
    annotateFn = [](unsigned, Operation *, OpBuilder) {};

  // Keep a pointer to the last non-terminator operation in the original block
  // so that we know what to clone (since we are doing this in-place).
  Block::iterator srcBlockEnd = std::prev(loopBodyBlock->end(), 2);

  // Unroll the contents of 'forOp' (append unrollFactor - 1 additional copies).
  SmallVector<Value, 4> lastYielded(yieldedValues);

  for (unsigned i = 1; i < unrollFactor; i++) {
    IRMapping operandMap;

    // Prepare operand map.
    operandMap.map(iterArgs, lastYielded);

    // If the induction variable is used, create a remapping to the value for
    // this unrolled instance.
    if (!forOpIV.use_empty()) {
      Value ivUnroll = ivRemapFn(i, forOpIV, builder);
      operandMap.map(forOpIV, ivUnroll);
    }

    // Clone the original body of 'forOp'.
    for (auto it = loopBodyBlock->begin(); it != std::next(srcBlockEnd); it++) {
      Operation *clonedOp = builder.clone(*it, operandMap);
      annotateFn(i, clonedOp, builder);
    }

    // Update yielded values.
    for (unsigned i = 0, e = lastYielded.size(); i < e; i++)
      lastYielded[i] = operandMap.lookup(yieldedValues[i]);
  }

  // Make sure we annotate the Ops in the original body. We do this last so that
  // any annotations are not copied into the cloned Ops above.
  for (auto it = loopBodyBlock->begin(); it != std::next(srcBlockEnd); it++)
    annotateFn(0, &*it, builder);

  // Update operands of the yield statement.
  loopBodyBlock->getTerminator()->setOperands(lastYielded);
}

/// Unrolls 'forOp' by 'unrollFactor', returns success if the loop is unrolled.
LogicalResult mlir::loopUnrollByFactor(
    scf::ForOp forOp, uint64_t unrollFactor,
    function_ref<void(unsigned, Operation *, OpBuilder)> annotateFn) {
  assert(unrollFactor > 0 && "expected positive unroll factor");

  // Return if the loop body is empty.
  if (llvm::hasSingleElement(forOp.getBody()->getOperations()))
    return success();

  // Compute tripCount = ceilDiv((upperBound - lowerBound), step) and populate
  // 'upperBoundUnrolled' and 'stepUnrolled' for static and dynamic cases.
  OpBuilder boundsBuilder(forOp);
  IRRewriter rewriter(forOp.getContext());
  auto loc = forOp.getLoc();
  Value step = forOp.getStep();
  Value upperBoundUnrolled;
  Value stepUnrolled;
  bool generateEpilogueLoop = true;

  std::optional<int64_t> lbCstOp = getConstantIntValue(forOp.getLowerBound());
  std::optional<int64_t> ubCstOp = getConstantIntValue(forOp.getUpperBound());
  std::optional<int64_t> stepCstOp = getConstantIntValue(forOp.getStep());
  if (lbCstOp && ubCstOp && stepCstOp) {
    // Constant loop bounds computation.
    int64_t lbCst = lbCstOp.value();
    int64_t ubCst = ubCstOp.value();
    int64_t stepCst = stepCstOp.value();
    assert(lbCst >= 0 && ubCst >= 0 && stepCst >= 0 &&
           "expected positive loop bounds and step");
    int64_t tripCount = mlir::ceilDiv(ubCst - lbCst, stepCst);

    if (unrollFactor == 1) {
      if (tripCount == 1 && failed(forOp.promoteIfSingleIteration(rewriter)))
        return failure();
      return success();
    }

    int64_t tripCountEvenMultiple = tripCount - (tripCount % unrollFactor);
    int64_t upperBoundUnrolledCst = lbCst + tripCountEvenMultiple * stepCst;
    int64_t stepUnrolledCst = stepCst * unrollFactor;

    // Create constant for 'upperBoundUnrolled' and set epilogue loop flag.
    generateEpilogueLoop = upperBoundUnrolledCst < ubCst;
    if (generateEpilogueLoop)
      upperBoundUnrolled = boundsBuilder.create<arith::ConstantIndexOp>(
          loc, upperBoundUnrolledCst);
    else
      upperBoundUnrolled = forOp.getUpperBound();

    // Create constant for 'stepUnrolled'.
    stepUnrolled = stepCst == stepUnrolledCst
                       ? step
                       : boundsBuilder.create<arith::ConstantIndexOp>(
                             loc, stepUnrolledCst);
  } else {
    // Dynamic loop bounds computation.
    // TODO: Add dynamic asserts for negative lb/ub/step, or
    // consider using ceilDiv from AffineApplyExpander.
    auto lowerBound = forOp.getLowerBound();
    auto upperBound = forOp.getUpperBound();
    Value diff =
        boundsBuilder.create<arith::SubIOp>(loc, upperBound, lowerBound);
    Value tripCount = ceilDivPositive(boundsBuilder, loc, diff, step);
    Value unrollFactorCst =
        boundsBuilder.create<arith::ConstantIndexOp>(loc, unrollFactor);
    Value tripCountRem =
        boundsBuilder.create<arith::RemSIOp>(loc, tripCount, unrollFactorCst);
    // Compute tripCountEvenMultiple = tripCount - (tripCount % unrollFactor)
    Value tripCountEvenMultiple =
        boundsBuilder.create<arith::SubIOp>(loc, tripCount, tripCountRem);
    // Compute upperBoundUnrolled = lowerBound + tripCountEvenMultiple * step
    upperBoundUnrolled = boundsBuilder.create<arith::AddIOp>(
        loc, lowerBound,
        boundsBuilder.create<arith::MulIOp>(loc, tripCountEvenMultiple, step));
    // Scale 'step' by 'unrollFactor'.
    stepUnrolled =
        boundsBuilder.create<arith::MulIOp>(loc, step, unrollFactorCst);
  }

  // Create epilogue clean up loop starting at 'upperBoundUnrolled'.
  if (generateEpilogueLoop) {
    OpBuilder epilogueBuilder(forOp->getContext());
    epilogueBuilder.setInsertionPoint(forOp->getBlock(),
                                      std::next(Block::iterator(forOp)));
    auto epilogueForOp = cast<scf::ForOp>(epilogueBuilder.clone(*forOp));
    epilogueForOp.setLowerBound(upperBoundUnrolled);

    // Update uses of loop results.
    auto results = forOp.getResults();
    auto epilogueResults = epilogueForOp.getResults();

    for (auto e : llvm::zip(results, epilogueResults)) {
      std::get<0>(e).replaceAllUsesWith(std::get<1>(e));
    }
    epilogueForOp->setOperands(epilogueForOp.getNumControlOperands(),
                               epilogueForOp.getInitArgs().size(), results);
    (void)epilogueForOp.promoteIfSingleIteration(rewriter);
  }

  // Create unrolled loop.
  forOp.setUpperBound(upperBoundUnrolled);
  forOp.setStep(stepUnrolled);

  auto iterArgs = ValueRange(forOp.getRegionIterArgs());
  auto yieldedValues = forOp.getBody()->getTerminator()->getOperands();

  generateUnrolledLoop(
      forOp.getBody(), forOp.getInductionVar(), unrollFactor,
      [&](unsigned i, Value iv, OpBuilder b) {
        // iv' = iv + step * i;
        auto stride = b.create<arith::MulIOp>(
            loc, step, b.create<arith::ConstantIndexOp>(loc, i));
        return b.create<arith::AddIOp>(loc, iv, stride);
      },
      annotateFn, iterArgs, yieldedValues);
  // Promote the loop body up if this has turned into a single iteration loop.
  (void)forOp.promoteIfSingleIteration(rewriter);
  return success();
}

/// Return the new lower bound, upper bound, and step in that order. Insert any
/// additional bounds calculations before the given builder and any additional
/// conversion back to the original loop induction value inside the given Block.
static LoopParams normalizeLoop(OpBuilder &boundsBuilder,
                                OpBuilder &insideLoopBuilder, Location loc,
                                Value lowerBound, Value upperBound, Value step,
                                Value inductionVar) {
  // Check if the loop is already known to have a constant zero lower bound or
  // a constant one step.
  bool isZeroBased = false;
  if (auto ubCst = getConstantIntValue(lowerBound))
    isZeroBased = ubCst.value() == 0;

  bool isStepOne = false;
  if (auto stepCst = getConstantIntValue(step))
    isStepOne = stepCst.value() == 1;

  // Compute the number of iterations the loop executes: ceildiv(ub - lb, step)
  // assuming the step is strictly positive.  Update the bounds and the step
  // of the loop to go from 0 to the number of iterations, if necessary.
  if (isZeroBased && isStepOne)
    return {/*lowerBound=*/lowerBound, /*upperBound=*/upperBound,
            /*step=*/step};

  Value diff = boundsBuilder.create<arith::SubIOp>(loc, upperBound, lowerBound);
  Value newUpperBound =
      boundsBuilder.create<arith::CeilDivSIOp>(loc, diff, step);

  Value newLowerBound =
      isZeroBased ? lowerBound
                  : boundsBuilder.create<arith::ConstantOp>(
                        loc, boundsBuilder.getZeroAttr(lowerBound.getType()));
  Value newStep =
      isStepOne ? step
                : boundsBuilder.create<arith::ConstantOp>(
                      loc, boundsBuilder.getIntegerAttr(step.getType(), 1));

  // Insert code computing the value of the original loop induction variable
  // from the "normalized" one.
  Value scaled =
      isStepOne
          ? inductionVar
          : insideLoopBuilder.create<arith::MulIOp>(loc, inductionVar, step);
  Value shifted =
      isZeroBased
          ? scaled
          : insideLoopBuilder.create<arith::AddIOp>(loc, scaled, lowerBound);

  SmallPtrSet<Operation *, 2> preserve{scaled.getDefiningOp(),
                                       shifted.getDefiningOp()};
  inductionVar.replaceAllUsesExcept(shifted, preserve);
  return {/*lowerBound=*/newLowerBound, /*upperBound=*/newUpperBound,
          /*step=*/newStep};
}

/// Transform a loop with a strictly positive step
///   for %i = %lb to %ub step %s
/// into a 0-based loop with step 1
///   for %ii = 0 to ceildiv(%ub - %lb, %s) step 1 {
///     %i = %ii * %s + %lb
/// Insert the induction variable remapping in the body of `inner`, which is
/// expected to be either `loop` or another loop perfectly nested under `loop`.
/// Insert the definition of new bounds immediate before `outer`, which is
/// expected to be either `loop` or its parent in the loop nest.
static void normalizeLoop(scf::ForOp loop, scf::ForOp outer, scf::ForOp inner) {
  OpBuilder builder(outer);
  OpBuilder innerBuilder = OpBuilder::atBlockBegin(inner.getBody());
  auto loopPieces = normalizeLoop(builder, innerBuilder, loop.getLoc(),
                                  loop.getLowerBound(), loop.getUpperBound(),
                                  loop.getStep(), loop.getInductionVar());

  loop.setLowerBound(loopPieces.lowerBound);
  loop.setUpperBound(loopPieces.upperBound);
  loop.setStep(loopPieces.step);
}

LogicalResult mlir::coalesceLoops(MutableArrayRef<scf::ForOp> loops) {
  if (loops.size() < 2)
    return failure();

  scf::ForOp innermost = loops.back();
  scf::ForOp outermost = loops.front();

  // 1. Make sure all loops iterate from 0 to upperBound with step 1.  This
  // allows the following code to assume upperBound is the number of iterations.
  for (auto loop : loops)
    normalizeLoop(loop, outermost, innermost);

  // 2. Emit code computing the upper bound of the coalesced loop as product
  // of the number of iterations of all loops.
  OpBuilder builder(outermost);
  Location loc = outermost.getLoc();
  Value upperBound = outermost.getUpperBound();
  for (auto loop : loops.drop_front())
    upperBound =
        builder.create<arith::MulIOp>(loc, upperBound, loop.getUpperBound());
  outermost.setUpperBound(upperBound);

  builder.setInsertionPointToStart(outermost.getBody());

  // 3. Remap induction variables. For each original loop, the value of the
  // induction variable can be obtained by dividing the induction variable of
  // the linearized loop by the total number of iterations of the loops nested
  // in it modulo the number of iterations in this loop (remove the values
  // related to the outer loops):
  //   iv_i = floordiv(iv_linear, product-of-loop-ranges-until-i) mod range_i.
  // Compute these iteratively from the innermost loop by creating a "running
  // quotient" of division by the range.
  Value previous = outermost.getInductionVar();
  for (unsigned i = 0, e = loops.size(); i < e; ++i) {
    unsigned idx = loops.size() - i - 1;
    if (i != 0)
      previous = builder.create<arith::DivSIOp>(loc, previous,
                                                loops[idx + 1].getUpperBound());

    Value iv = (i == e - 1) ? previous
                            : builder.create<arith::RemSIOp>(
                                  loc, previous, loops[idx].getUpperBound());
    replaceAllUsesInRegionWith(loops[idx].getInductionVar(), iv,
                               loops.back().getRegion());
  }

  // 4. Move the operations from the innermost just above the second-outermost
  // loop, delete the extra terminator and the second-outermost loop.
  scf::ForOp second = loops[1];
  innermost.getBody()->back().erase();
  outermost.getBody()->getOperations().splice(
      Block::iterator(second.getOperation()),
      innermost.getBody()->getOperations());
  second.erase();
  return success();
}

void mlir::collapseParallelLoops(
    scf::ParallelOp loops, ArrayRef<std::vector<unsigned>> combinedDimensions) {
  OpBuilder outsideBuilder(loops);
  Location loc = loops.getLoc();

  // Presort combined dimensions.
  auto sortedDimensions = llvm::to_vector<3>(combinedDimensions);
  for (auto &dims : sortedDimensions)
    llvm::sort(dims);

  // Normalize ParallelOp's iteration pattern.
  SmallVector<Value, 3> normalizedLowerBounds, normalizedSteps,
      normalizedUpperBounds;
  for (unsigned i = 0, e = loops.getNumLoops(); i < e; ++i) {
    OpBuilder insideLoopBuilder = OpBuilder::atBlockBegin(loops.getBody());
    auto resultBounds =
        normalizeLoop(outsideBuilder, insideLoopBuilder, loc,
                      loops.getLowerBound()[i], loops.getUpperBound()[i],
                      loops.getStep()[i], loops.getBody()->getArgument(i));

    normalizedLowerBounds.push_back(resultBounds.lowerBound);
    normalizedUpperBounds.push_back(resultBounds.upperBound);
    normalizedSteps.push_back(resultBounds.step);
  }

  // Combine iteration spaces.
  SmallVector<Value, 3> lowerBounds, upperBounds, steps;
  auto cst0 = outsideBuilder.create<arith::ConstantIndexOp>(loc, 0);
  auto cst1 = outsideBuilder.create<arith::ConstantIndexOp>(loc, 1);
  for (unsigned i = 0, e = sortedDimensions.size(); i < e; ++i) {
    Value newUpperBound = outsideBuilder.create<arith::ConstantIndexOp>(loc, 1);
    for (auto idx : sortedDimensions[i]) {
      newUpperBound = outsideBuilder.create<arith::MulIOp>(
          loc, newUpperBound, normalizedUpperBounds[idx]);
    }
    lowerBounds.push_back(cst0);
    steps.push_back(cst1);
    upperBounds.push_back(newUpperBound);
  }

  // Create new ParallelLoop with conversions to the original induction values.
  // The loop below uses divisions to get the relevant range of values in the
  // new induction value that represent each range of the original induction
  // value. The remainders then determine based on that range, which iteration
  // of the original induction value this represents. This is a normalized value
  // that is un-normalized already by the previous logic.
  auto newPloop = outsideBuilder.create<scf::ParallelOp>(
      loc, lowerBounds, upperBounds, steps,
      [&](OpBuilder &insideBuilder, Location, ValueRange ploopIVs) {
        for (unsigned i = 0, e = combinedDimensions.size(); i < e; ++i) {
          Value previous = ploopIVs[i];
          unsigned numberCombinedDimensions = combinedDimensions[i].size();
          // Iterate over all except the last induction value.
          for (unsigned j = numberCombinedDimensions - 1; j > 0; --j) {
            unsigned idx = combinedDimensions[i][j];

            // Determine the current induction value's current loop iteration
            Value iv = insideBuilder.create<arith::RemSIOp>(
                loc, previous, normalizedUpperBounds[idx]);
            replaceAllUsesInRegionWith(loops.getBody()->getArgument(idx), iv,
                                       loops.getRegion());

            // Remove the effect of the current induction value to prepare for
            // the next value.
            previous = insideBuilder.create<arith::DivSIOp>(
                loc, previous, normalizedUpperBounds[idx]);
          }

          // The final induction value is just the remaining value.
          unsigned idx = combinedDimensions[i][0];
          replaceAllUsesInRegionWith(loops.getBody()->getArgument(idx),
                                     previous, loops.getRegion());
        }
      });

  // Replace the old loop with the new loop.
  loops.getBody()->back().erase();
  newPloop.getBody()->getOperations().splice(
      Block::iterator(newPloop.getBody()->back()),
      loops.getBody()->getOperations());
  loops.erase();
}

// Hoist the ops within `outer` that appear before `inner`.
// Such ops include the ops that have been introduced by parametric tiling.
// Ops that come from triangular loops (i.e. that belong to the program slice
// rooted at `outer`) and ops that have side effects cannot be hoisted.
// Return failure when any op fails to hoist.
static LogicalResult hoistOpsBetween(scf::ForOp outer, scf::ForOp inner) {
  SetVector<Operation *> forwardSlice;
  ForwardSliceOptions options;
  options.filter = [&inner](Operation *op) {
    return op != inner.getOperation();
  };
  getForwardSlice(outer.getInductionVar(), &forwardSlice, options);
  LogicalResult status = success();
  SmallVector<Operation *, 8> toHoist;
  for (auto &op : outer.getBody()->without_terminator()) {
    // Stop when encountering the inner loop.
    if (&op == inner.getOperation())
      break;
    // Skip over non-hoistable ops.
    if (forwardSlice.count(&op) > 0) {
      status = failure();
      continue;
    }
    // Skip intermediate scf::ForOp, these are not considered a failure.
    if (isa<scf::ForOp>(op))
      continue;
    // Skip other ops with regions.
    if (op.getNumRegions() > 0) {
      status = failure();
      continue;
    }
    // Skip if op has side effects.
    // TODO: loads to immutable memory regions are ok.
    if (!isMemoryEffectFree(&op)) {
      status = failure();
      continue;
    }
    toHoist.push_back(&op);
  }
  auto *outerForOp = outer.getOperation();
  for (auto *op : toHoist)
    op->moveBefore(outerForOp);
  return status;
}

// Traverse the interTile and intraTile loops and try to hoist ops such that
// bands of perfectly nested loops are isolated.
// Return failure if either perfect interTile or perfect intraTile bands cannot
// be formed.
static LogicalResult tryIsolateBands(const TileLoops &tileLoops) {
  LogicalResult status = success();
  const Loops &interTile = tileLoops.first;
  const Loops &intraTile = tileLoops.second;
  auto size = interTile.size();
  assert(size == intraTile.size());
  if (size <= 1)
    return success();
  for (unsigned s = 1; s < size; ++s)
    status = succeeded(status) ? hoistOpsBetween(intraTile[0], intraTile[s])
                               : failure();
  for (unsigned s = 1; s < size; ++s)
    status = succeeded(status) ? hoistOpsBetween(interTile[0], interTile[s])
                               : failure();
  return status;
}

/// Collect perfectly nested loops starting from `rootForOps`.  Loops are
/// perfectly nested if each loop is the first and only non-terminator operation
/// in the parent loop.  Collect at most `maxLoops` loops and append them to
/// `forOps`.
template <typename T>
static void getPerfectlyNestedLoopsImpl(
    SmallVectorImpl<T> &forOps, T rootForOp,
    unsigned maxLoops = std::numeric_limits<unsigned>::max()) {
  for (unsigned i = 0; i < maxLoops; ++i) {
    forOps.push_back(rootForOp);
    Block &body = rootForOp.getRegion().front();
    if (body.begin() != std::prev(body.end(), 2))
      return;

    rootForOp = dyn_cast<T>(&body.front());
    if (!rootForOp)
      return;
  }
}

static Loops stripmineSink(scf::ForOp forOp, Value factor,
                           ArrayRef<scf::ForOp> targets) {
  auto originalStep = forOp.getStep();
  auto iv = forOp.getInductionVar();

  OpBuilder b(forOp);
  forOp.setStep(b.create<arith::MulIOp>(forOp.getLoc(), originalStep, factor));

  Loops innerLoops;
  for (auto t : targets) {
    // Save information for splicing ops out of t when done
    auto begin = t.getBody()->begin();
    auto nOps = t.getBody()->getOperations().size();

    // Insert newForOp before the terminator of `t`.
    auto b = OpBuilder::atBlockTerminator((t.getBody()));
    Value stepped = b.create<arith::AddIOp>(t.getLoc(), iv, forOp.getStep());
    Value less = b.create<arith::CmpIOp>(t.getLoc(), arith::CmpIPredicate::slt,
                                         forOp.getUpperBound(), stepped);
    Value ub = b.create<arith::SelectOp>(t.getLoc(), less,
                                         forOp.getUpperBound(), stepped);

    // Splice [begin, begin + nOps - 1) into `newForOp` and replace uses.
    auto newForOp = b.create<scf::ForOp>(t.getLoc(), iv, ub, originalStep);
    newForOp.getBody()->getOperations().splice(
        newForOp.getBody()->getOperations().begin(),
        t.getBody()->getOperations(), begin, std::next(begin, nOps - 1));
    replaceAllUsesInRegionWith(iv, newForOp.getInductionVar(),
                               newForOp.getRegion());

    innerLoops.push_back(newForOp);
  }

  return innerLoops;
}

// Stripmines a `forOp` by `factor` and sinks it under a single `target`.
// Returns the new for operation, nested immediately under `target`.
template <typename SizeType>
static scf::ForOp stripmineSink(scf::ForOp forOp, SizeType factor,
                                scf::ForOp target) {
  // TODO: Use cheap structural assertions that targets are nested under
  // forOp and that targets are not nested under each other when DominanceInfo
  // exposes the capability. It seems overkill to construct a whole function
  // dominance tree at this point.
  auto res = stripmineSink(forOp, factor, ArrayRef<scf::ForOp>(target));
  assert(res.size() == 1 && "Expected 1 inner forOp");
  return res[0];
}

SmallVector<Loops, 8> mlir::tile(ArrayRef<scf::ForOp> forOps,
                                 ArrayRef<Value> sizes,
                                 ArrayRef<scf::ForOp> targets) {
  SmallVector<SmallVector<scf::ForOp, 8>, 8> res;
  SmallVector<scf::ForOp, 8> currentTargets(targets.begin(), targets.end());
  for (auto it : llvm::zip(forOps, sizes)) {
    auto step = stripmineSink(std::get<0>(it), std::get<1>(it), currentTargets);
    res.push_back(step);
    currentTargets = step;
  }
  return res;
}

Loops mlir::tile(ArrayRef<scf::ForOp> forOps, ArrayRef<Value> sizes,
                 scf::ForOp target) {
  SmallVector<scf::ForOp, 8> res;
  for (auto loops : tile(forOps, sizes, ArrayRef<scf::ForOp>(target))) {
    assert(loops.size() == 1);
    res.push_back(loops[0]);
  }
  return res;
}

Loops mlir::tilePerfectlyNested(scf::ForOp rootForOp, ArrayRef<Value> sizes) {
  // Collect perfectly nested loops.  If more size values provided than nested
  // loops available, truncate `sizes`.
  SmallVector<scf::ForOp, 4> forOps;
  forOps.reserve(sizes.size());
  getPerfectlyNestedLoopsImpl(forOps, rootForOp, sizes.size());
  if (forOps.size() < sizes.size())
    sizes = sizes.take_front(forOps.size());

  return ::tile(forOps, sizes, forOps.back());
}

void mlir::getPerfectlyNestedLoops(SmallVectorImpl<scf::ForOp> &nestedLoops,
                                   scf::ForOp root) {
  getPerfectlyNestedLoopsImpl(nestedLoops, root);
}

TileLoops mlir::extractFixedOuterLoops(scf::ForOp rootForOp,
                                       ArrayRef<int64_t> sizes) {
  // Collect perfectly nested loops.  If more size values provided than nested
  // loops available, truncate `sizes`.
  SmallVector<scf::ForOp, 4> forOps;
  forOps.reserve(sizes.size());
  getPerfectlyNestedLoopsImpl(forOps, rootForOp, sizes.size());
  if (forOps.size() < sizes.size())
    sizes = sizes.take_front(forOps.size());

  // Compute the tile sizes such that i-th outer loop executes size[i]
  // iterations.  Given that the loop current executes
  //   numIterations = ceildiv((upperBound - lowerBound), step)
  // iterations, we need to tile with size ceildiv(numIterations, size[i]).
  SmallVector<Value, 4> tileSizes;
  tileSizes.reserve(sizes.size());
  for (unsigned i = 0, e = sizes.size(); i < e; ++i) {
    assert(sizes[i] > 0 && "expected strictly positive size for strip-mining");

    auto forOp = forOps[i];
    OpBuilder builder(forOp);
    auto loc = forOp.getLoc();
    Value diff = builder.create<arith::SubIOp>(loc, forOp.getUpperBound(),
                                               forOp.getLowerBound());
    Value numIterations = ceilDivPositive(builder, loc, diff, forOp.getStep());
    Value iterationsPerBlock =
        ceilDivPositive(builder, loc, numIterations, sizes[i]);
    tileSizes.push_back(iterationsPerBlock);
  }

  // Call parametric tiling with the given sizes.
  auto intraTile = tile(forOps, tileSizes, forOps.back());
  TileLoops tileLoops = std::make_pair(forOps, intraTile);

  // TODO: for now we just ignore the result of band isolation.
  // In the future, mapping decisions may be impacted by the ability to
  // isolate perfectly nested bands.
  (void)tryIsolateBands(tileLoops);

  return tileLoops;
}

scf::ForallOp mlir::fuseIndependentSiblingForallLoops(scf::ForallOp target,
                                                      scf::ForallOp source,
                                                      RewriterBase &rewriter) {
  unsigned numTargetOuts = target.getNumResults();
  unsigned numSourceOuts = source.getNumResults();

  OperandRange targetOuts = target.getOutputs();
  OperandRange sourceOuts = source.getOutputs();

  // Create fused shared_outs.
  SmallVector<Value> fusedOuts;
  fusedOuts.reserve(numTargetOuts + numSourceOuts);
  fusedOuts.append(targetOuts.begin(), targetOuts.end());
  fusedOuts.append(sourceOuts.begin(), sourceOuts.end());

  // Create a new scf::forall op after the source loop.
  rewriter.setInsertionPointAfter(source);
  scf::ForallOp fusedLoop = rewriter.create<scf::ForallOp>(
      source.getLoc(), source.getMixedLowerBound(), source.getMixedUpperBound(),
      source.getMixedStep(), fusedOuts, source.getMapping());

  // Map control operands.
  IRMapping fusedMapping;
  fusedMapping.map(target.getInductionVars(), fusedLoop.getInductionVars());
  fusedMapping.map(source.getInductionVars(), fusedLoop.getInductionVars());

  // Map shared outs.
  fusedMapping.map(target.getOutputBlockArguments(),
                   fusedLoop.getOutputBlockArguments().slice(0, numTargetOuts));
  fusedMapping.map(
      source.getOutputBlockArguments(),
      fusedLoop.getOutputBlockArguments().slice(numTargetOuts, numSourceOuts));

  // Append everything except the terminator into the fused operation.
  rewriter.setInsertionPointToStart(fusedLoop.getBody());
  for (Operation &op : target.getBody()->without_terminator())
    rewriter.clone(op, fusedMapping);
  for (Operation &op : source.getBody()->without_terminator())
    rewriter.clone(op, fusedMapping);

  // Fuse the old terminator in_parallel ops into the new one.
  scf::InParallelOp targetTerm = target.getTerminator();
  scf::InParallelOp sourceTerm = source.getTerminator();
  scf::InParallelOp fusedTerm = fusedLoop.getTerminator();

  rewriter.setInsertionPointToStart(fusedTerm.getBody());
  for (Operation &op : targetTerm.getYieldingOps())
    rewriter.clone(op, fusedMapping);
  for (Operation &op : sourceTerm.getYieldingOps())
    rewriter.clone(op, fusedMapping);

  // Replace all uses of the old loops with the fused loop.
  rewriter.replaceAllUsesWith(target.getResults(),
                              fusedLoop.getResults().slice(0, numTargetOuts));
  rewriter.replaceAllUsesWith(
      source.getResults(),
      fusedLoop.getResults().slice(numTargetOuts, numSourceOuts));

  // Erase the old loops.
  rewriter.eraseOp(target);
  rewriter.eraseOp(source);

  return fusedLoop;
}
