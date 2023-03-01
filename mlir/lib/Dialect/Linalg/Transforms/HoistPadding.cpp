//===- HoistPadding.cpp - Hoisting for tensor::PadOp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements functions concerned with hoisting padding operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/Debug.h"

using llvm::dbgs;

#define DEBUG_TYPE "hoist-padding"

#define DBGS() (dbgs() << '[' << DEBUG_TYPE << "] ")

using namespace mlir;
using namespace mlir::linalg;

#ifndef NDEBUG
static bool debugPrintLoopInShortForm(Operation *op) {
  AsmState state(op->getParentOfType<func::FuncOp>());
  (void)state;
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    forOp.getInductionVar().printAsOperand(dbgs(), state);
    dbgs() << " @ " << forOp.getOperation();
    return true;
  }
  return false;
}
#endif

static void debugPrintBackwardSlice(SetVector<Operation *> &backwardSlice) {
  LLVM_DEBUG(llvm::interleaveComma(backwardSlice, DBGS() << "--backwardSlice:",
                                   [](Operation *op) {
                                     dbgs() << "\n";
                                     DBGS() << "----";
                                     if (debugPrintLoopInShortForm(op)) {
                                       dbgs() << "\n";
                                       return;
                                     }
                                     dbgs() << *op << "\n";
                                   });
             DBGS() << "\n";);
}

/// Analysis class to support tensor::PadOp hoisting across multiple enclosing
/// loops. The failure conditions are:
///   1. Pad op has a use that is not an input of a LinalgOp.
///   2. Pad op does not have a constant padding value.
///   3. There is no immediately enclosing scf::ForOp.
///   4. The backward slice from the pad op to the scf::ForOp to hoist above
///      contains an unknown op with non index type operands, a region, or a
///      memory effect.
///   5. The backward slice from the pad op to the scf::ForOp to hoist above is
///      empty.
///   6. The source tensor of pad op is not defined by an extract slice op.
///   7. The source tensor of the extract slice op is not defined outside of
///      the outermost enclosing scf::ForOp.
///   8. There is no enclosing scf::ForOp that indexes the padded data.
/// Other cases succeed and will trigger hoisting of the pad op.
struct HoistingAnalysis {
  HoistingAnalysis(tensor::PadOp padOp, int numLoops);

  bool isValid() { return valid; }

  /// Footprint of the packedTensor, computed from the packingLoops.
  SmallVector<Value> getPackedTensorSizes(RewriterBase &b, Location loc);

  /// The outermost loop, determined by `nLevels` above which `padOp` will
  /// be hoisted.
  scf::ForOp outermostEnclosingForOp;

  /// Backward slice rooted at `padOp` and nested under
  /// `outermostEnclosingForOp`.
  SetVector<Operation *> backwardSlice;

  /// The scf::ForOp immediately enclosing `padOp` such that:
  ///  1. they are nested under `outermostEnclosingForOp` (inclusive)
  ///  2. whose induction variable is used, directly or indirectly, in the
  ///     computation of `padOp`.
  /// The span of these loops determines the footprint of the packed tensor.
  SmallVector<scf::ForOp> packingLoops;

  /// The ExtractSliceOp that feeds the PadOp we want to hoist.
  tensor::ExtractSliceOp sliceOp;

  /// If non-empty, this is the unique scf::ForOp that consumes the `sliceOp`.
  scf::ForOp padConsumingForOp;

private:
  /// Drop any non-index dependencies of `padOp` and `sliceOp` from
  /// `backwardSlice`. The method follows the use-def chains of the index
  /// operands consumed by `padOp` and `sliceOp` and drops the operations
  /// not part of this index computation. Afterwards, the filtered
  /// `backwardSlice` contains only the loops whose induction variable is
  /// used, directly or indirectly, to index the padded tensor. The method
  /// returns failure if the filtered backward slice contains an unexpected
  /// operation.
  ///
  /// Example:
  /// ```
  /// %source = linalg.fill(%cst, %arg0)
  /// scf.for %i
  ///   %unrelated = linalg.fill(%cst, %arg1)    // not used to index
  ///   %source! scf.for %j (%arg2 = %unrelated)
  ///     scf.for %k                             // not used to index
  ///     %source!
  ///       %ubi = affine.min #map(%i)
  ///       %ubj = affine.min #map(%j)
  ///       %slice = tensor.extract_slice %source [%i, %j] [%ubi, %ubj]
  ///       %padded_slice = tensor.pad %slice
  /// ```
  /// dropNonIndexDependencies(%padded_slice, %slice)
  /// removes [scf.for %k, linalg.fill(%cst, %arg1)] from backwardSlice.
  LogicalResult dropNonIndexDependencies(tensor::PadOp padOp);

  /// Encodes whether the analysis is valid and hoisting can proceed.
  bool valid;
};

/// Return at most nLevels of immediately enclosing scf::ForOp loops.
/// Stops at the first parent that is not an scf::ForOp.
/// Multi-loops such as scf.parallel or linalg.tiled_loop are not modeled atm.
/// Control-flow and other containing ops with regions are not modeled atm.
static void
getAtMostNEnclosingLoops(tensor::PadOp padOp, int nLevels,
                         SmallVector<scf::ForOp> &reverseEnclosingLoops) {
  scf::ForOp outermostEnclosingForOp = nullptr;
  Operation *nextEnclosingOp = padOp->getParentOp();
  while (nLevels-- > 0 &&
         (outermostEnclosingForOp = dyn_cast<scf::ForOp>(nextEnclosingOp))) {
    LLVM_DEBUG(DBGS() << "loops: ";
               debugPrintLoopInShortForm(outermostEnclosingForOp);
               dbgs() << "\n");
    reverseEnclosingLoops.push_back(outermostEnclosingForOp);
    nextEnclosingOp = outermostEnclosingForOp->getParentOp();
  }
}

// Get all the ops in the backwards slice starting from `padOp` and that
// are dominated by the outermost enclosing loop.
// This also requires tracking ops defining values used in the region but
// defined above.
static void computeBackwardSlice(tensor::PadOp padOp,
                                 scf::ForOp outermostEnclosingForOp,
                                 SetVector<Operation *> &backwardSlice) {
  DominanceInfo domInfo(outermostEnclosingForOp);
  auto filter = [&](Operation *op) {
    return domInfo.dominates(outermostEnclosingForOp, op) &&
           !padOp->isProperAncestor(op);
  };
  // First, add the ops required to compute the region to the backwardSlice.
  SetVector<Value> valuesDefinedAbove;
  getUsedValuesDefinedAbove(padOp.getRegion(), padOp.getRegion(),
                            valuesDefinedAbove);
  for (Value v : valuesDefinedAbove) {
    getBackwardSlice(v, &backwardSlice, filter, /*inclusive=*/true);
  }
  // Then, add the backward slice from padOp itself.
  getBackwardSlice(padOp.getOperation(), &backwardSlice, filter,
                   /*inclusive=*/true);
}

HoistingAnalysis::HoistingAnalysis(tensor::PadOp padOp, int numLoops) {
  valid = false;

  // Get at most `numLoops` of immediately enclosing loops.
  SmallVector<scf::ForOp> reverseEnclosingLoops;
  getAtMostNEnclosingLoops(padOp, numLoops, reverseEnclosingLoops);
  if (reverseEnclosingLoops.empty()) {
    LLVM_DEBUG(DBGS() << "--No immediately enclosing loop -> Skip\n");
    return;
  }

  outermostEnclosingForOp = reverseEnclosingLoops.back();

  // Get the `sliceOp` that defines the source tensor of `padOp` and
  // check its source is defined outside of the outermost loop. This check
  // ensures the padded data is available for packing before entering the
  // outermost enclosing loop.
  //
  // Example:
  // ```
  // %source = linalg.fill(%cst, %arg0)
  // // %source is available for packing here!
  // scf.for %i
  //   scf.for %j
  //     scf.for %k
  //       %slice = tensor.extract_slice %source [%i, %j]
  //       %padded_slice = tensor.pad %slice
  // ```
  sliceOp = padOp.getSource().getDefiningOp<tensor::ExtractSliceOp>();
  if (!sliceOp) {
    LLVM_DEBUG(DBGS() << "--Cannot find the extract slice op -> Skip\n");
    return;
  }
  // If the padded data is not yet available before entering the outermost
  // enclosing loop, try to apply hoisting on this outermost loop.
  // TODO: we may want finer-grained hoisting of only that particular `sliceOp`.
  IRRewriter rewriter(outermostEnclosingForOp->getContext());
  if (!outermostEnclosingForOp.isDefinedOutsideOfLoop(sliceOp.getSource())) {
    outermostEnclosingForOp =
        hoistRedundantSubsetExtractInsert(rewriter, outermostEnclosingForOp);
  }
  if (!outermostEnclosingForOp.isDefinedOutsideOfLoop(sliceOp.getSource())) {
    LLVM_DEBUG(DBGS() << "--outermostEnclosingForOp:\n"
                      << outermostEnclosingForOp << "\n"
                      << "--sliceOp: " << sliceOp << "\n"
                      << "--sliceOp.getSource(): " << sliceOp.getSource()
                      << "\n");
    LLVM_DEBUG(DBGS() << "----Source not defined outside of loops -> Skip\n");
    return;
  }
  if (sliceOp->hasOneUse()) {
    padConsumingForOp = dyn_cast<scf::ForOp>(*(sliceOp->getUsers().begin()));
  }

  // Check the region of `padOp` depends on a constant only. Adding hoisting
  // support for arbitrary padding regions would require cloning all
  // dependencies captured by the padding region.
  Value paddingValue = padOp.getConstantPaddingValue();
  if (!paddingValue ||
      !isa_and_nonnull<arith::ConstantOp>(paddingValue.getDefiningOp())) {
    LLVM_DEBUG(DBGS() << "Cannot find constant padding value -> Skip\n");
    return;
  }

  computeBackwardSlice(padOp, outermostEnclosingForOp, backwardSlice);
  if (backwardSlice.size() <= 1)
    return;

  debugPrintBackwardSlice(backwardSlice);
  // Remove all ops in the backward slice that are not used to index
  // the padded tensor. In particular, keep `padOp`, `sliceOp`, and
  // the loop and affine operations used for the index computation.
  if (failed(dropNonIndexDependencies(padOp))) {
    LLVM_DEBUG(DBGS() << "--Cannot dropNonIndexDependencies -> Skip\n");
    return;
  }
  debugPrintBackwardSlice(backwardSlice);

  // Add only the loops part of the filtered `backwardSlice` to the
  // packing loops. All other loops are not used to index the padded
  // data and consequently access the same data in every loop
  // iteration. Adding them to the packing loops would increase the
  // cache footprint of the packed data by storing the same data
  // multiple times.
  for (scf::ForOp forOp : llvm::reverse(reverseEnclosingLoops))
    if (backwardSlice.contains(forOp))
      packingLoops.push_back(forOp);

  // TODO: for multiple loops we need to track the use to the innermost loop.
  if (packingLoops.size() > 1 && padConsumingForOp) {
    LLVM_DEBUG(DBGS() << "--Cannot hoist multiple loops through iter_args -> "
                         "Downgrade to 1 loop\n");
    packingLoops.resize(1);
  }

  // Note: at this point, packing loops may be empty but we would still like
  // to hoist the padding if so specified.

  // The analysis is valid and hoisting can occur.
  valid = true;
}

LogicalResult HoistingAnalysis::dropNonIndexDependencies(tensor::PadOp padOp) {
  // Set of all values used for index computation.
  SetVector<Value> indexEdges;

  // Add all index operands of `operation` to `indexEdges`. An index operand
  // is an operand of type index.
  auto addIndexOperandsToIndexEdges = [&](Operation *operation) {
    for (Value operand : operation->getOperands())
      if (operand.getType().isIndex())
        indexEdges.insert(operand);
  };

  // Check if any operation result is contained in `indexEdges`.
  auto hasIndexResult = [&](Operation *operation) {
    return llvm::any_of(operation->getResults(), [&](Value result) {
      return indexEdges.contains(result);
    });
  };

  // Starting from `padOp` and `sliceOp` walk the use-def edges of index
  // type in `backwardSlice`. Add the index operands of an operation to
  // `indexEdges` and remove all operations from `backwardSlice` that are not
  // part of the index computation.
  //
  // Example:
  // ```
  // %source = linalg.fill(%cst, %arg0)
  // scf.for %i
  //   %unrelated = linalg.fill(%cst, %arg1)    // not used to index %source!
  //   scf.for %j (%arg2 = %unrelated)
  //     scf.for %k                             // not used to index %source!
  //       %ubi = affine.min #map(%i)
  //       %ubj = affine.min #map(%j)
  //       %slice = tensor.extract_slice %source [%i, %j] [%ubi, %ubj]
  //       %padded_slice = tensor.pad %slice
  // ```
  // After iterating `backwardSlice` we obtain:
  // indexEdges = [%i, %j, %ubi, %ubj]
  // backwardSlice = backwardSlice / [linalg.fill(%cst, %arg1), scf.for %k]
  SetVector<Operation *> operationsToRemove;
  for (Operation *op : llvm::reverse(backwardSlice)) {
    // Add the index operands of `padOp` and `sliceOp` to start the
    // exploration of the index computation.
    if (op == padOp || op == sliceOp) {
      addIndexOperandsToIndexEdges(op);
      continue;
    }
    // Add the index operands of the loop if its induction variable is
    // used for index computation.
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      if (!hasIndexResult(op) && indexEdges.contains(forOp.getInductionVar())) {
        addIndexOperandsToIndexEdges(op);
        continue;
      }
    }
    // Add the index operands of all other operations if at least one result
    // is used for index computation.
    if (hasIndexResult(op)) {
      addIndexOperandsToIndexEdges(op);
      // Check the operands of the remaining operations all have index type.
      if (llvm::any_of(op->getOperandTypes(),
                       [](Type type) { return !type.isIndex(); })) {
        LLVM_DEBUG(DBGS() << "Unsupported op with non index type operands: "
                          << op << " -> Skip\n");
        return failure();
      }
      // Check the remaining operations do not have regions or memory effects.
      auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op);
      bool hasMemoryEffect = effectInterface && !effectInterface.hasNoEffect();
      if (hasMemoryEffect || op->getNumRegions() != 0) {
        LLVM_DEBUG(DBGS() << "Unsupported op with region or memory effect: "
                          << op << " -> Skip\n");
        return failure();
      }
      continue;
    }
    // Remove all other operations not used by the index computation. An
    // exception are constant operations that may be used by `padOp`.
    if (!isa<arith::ConstantOp>(op))
      operationsToRemove.insert(op);
  }
  backwardSlice.set_subtract(operationsToRemove);
  return success();
}

SmallVector<Value>
HoistingAnalysis::getPackedTensorSizes(RewriterBase &rewriter, Location loc) {
  SmallVector<Value> dynamicTensorSizes;

  // Upper bound the packing loop lengths to size the packed tensor. Taking
  // upper bounds can make the sizes of the packed tensor independent of the
  // enclosing loops. This independence is a prerequisite for reusing the same
  // buffer for all enclosing loop iterations and hoisting its allocation out
  // of the enclosing loops.
  for (auto forOp : packingLoops) {
    // Compute an upper bound `ubVal` for the upper bound of `forOp`.
    AffineMap boundMap;
    SmallVector<Value> boundOperands;
    getUpperBoundForIndex(forOp.getUpperBound(), boundMap, boundOperands);
    Value ubVal =
        rewriter.createOrFold<AffineMinOp>(loc, boundMap, boundOperands);
    // Compute the maximal packing loop length as (ub - lb).ceilDiv(step) and
    // store the result to `dynamicTensorSizes`.
    // TODO: instead of using the lower bound of `forOp` directly, implement a
    // lower bound computation similar to the upper bound computation.
    AffineExpr lb, ub, step;
    bindDims(rewriter.getContext(), lb, ub);
    bindSymbols(rewriter.getContext(), step);
    Value res = rewriter.createOrFold<AffineApplyOp>(
        loc, (ub - lb).ceilDiv(step),
        ValueRange{forOp.getLowerBound(), ubVal,
                   cast<scf::ForOp>(forOp).getStep()});
    dynamicTensorSizes.push_back(res);
  }

  return dynamicTensorSizes;
}

static bool isDefinedOutsideOrConstant(scf::ForOp outer, Value v) {
  return outer.isDefinedOutsideOfLoop(v) || matchPattern(v, m_Constant());
}

/// Return the current iteration number in the loop (iv - lb).ceilDiv(step).
/// The returned Value is guaranteed not to depend on any loop comprised in
/// [`outer`, `forOp`].
/// Return null if such a loop-independent quantity cannot be computed.
static Value buildLoopIterationCount(RewriterBase &rewriter, scf::ForOp outer,
                                     scf::ForOp forOp) {
  MLIRContext *ctx = forOp->getContext();
  AffineExpr iv, lb, step;
  bindDims(ctx, iv, lb);
  bindSymbols(ctx, step);
  if (!isDefinedOutsideOrConstant(outer, forOp.getLowerBound()) ||
      !isDefinedOutsideOrConstant(outer, forOp.getStep()))
    return Value();
  Value ivVal = forOp.getInductionVar(), lbVal = forOp.getLowerBound(),
        stepVal = forOp.getStep();
  auto loc = forOp->getLoc();
  return rewriter.createOrFold<AffineApplyOp>(
      loc, (iv - lb).ceilDiv(step), ValueRange{ivVal, lbVal, stepVal});
}

struct PackingLoopNestResult {
  SmallVector<OpFoldResult> offsets, sizes, strides;
  SmallVector<Value> clonedLoopIvs, leadingPackedTensorIndexings;
  GenericOp maybeTransposeOp;
};

// Build a packing loop nest by iteratively traversing the backward slice and
// clone the operations, iteratively stepping into the loops that we encounter.
// The implementation proceeds in a stack-like fashion:
//   1. Iteratively clone and step into the loops, pushing the `packedTensor`
//      deeper in the stack.
//   2. At the innermost loop level, create a GenericOp if `transposeVector` is
//      non-empty.
//   3. At the innermost loop level, create a InsertSliceOp.
//   4. Iteratively pop and yield the result of the InsertSliceOp across the
//      cloned loops.
static PackingLoopNestResult buildPackingLoopNest(
    RewriterBase &rewriter, IRMapping &bvm, tensor::PadOp opToHoist,
    ArrayRef<int64_t> transposeVector, RankedTensorType transposedTensorType,
    tensor::EmptyOp emptyOp, const HoistingAnalysis &analysis) {
  SmallVector<OpFoldResult> offsets, sizes, strides;
  SmallVector<Value> clonedLoopIvs, leadingPackedTensorIndexings;

  scf::ForOp outerLoop = analysis.outermostEnclosingForOp;

  Location loc = opToHoist->getLoc();
  RankedTensorType paddedTensorType = opToHoist.getResultType();
  int paddedRank = paddedTensorType.getRank();

  Value packedTensor = emptyOp.getResult();
  // Step 1. iteratively clone loops and push `packedTensor`.
  OpBuilder::InsertionGuard g(rewriter);
  for (Operation *op : analysis.backwardSlice) {
    // Specifically sit out in the extract_slice(packedTensor) case: this is
    // the piece we seek to replace.
    if (auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
      if (bvm.lookupOrDefault(sliceOp.getSource()) == packedTensor) {
        LLVM_DEBUG(DBGS() << "--Skip: " << sliceOp << "\n");
        continue;
      }
    }

    // Clone all operations except loops which require special handling.
    auto forOp = dyn_cast<scf::ForOp>(op);
    if (!forOp) {
      // We are at the right insertion point within the loop nest.
      rewriter.clone(*op, bvm);
      continue;
    }

    // Create a packing loop that takes `packedTensor` as iteration argument.
    auto clonedForOp = rewriter.create<scf::ForOp>(
        loc, bvm.lookupOrDefault(forOp.getLowerBound()),
        bvm.lookupOrDefault(forOp.getUpperBound()),
        bvm.lookupOrDefault(forOp.getStep()), packedTensor);

    // Map the induction var, region args and results to the `clonedForOp`.
    bvm.map(forOp.getInductionVar(), clonedForOp.getInductionVar());
    bvm.map(forOp.getRegionIterArgs(), clonedForOp.getRegionIterArgs());
    bvm.map(forOp.getResults(), clonedForOp.getResults());
    assert(clonedForOp->getNumRegions() == 1);
    clonedLoopIvs.push_back(clonedForOp.getInductionVar());

    // Do not insert guard here, we get deeper into the loop nest.
    rewriter.setInsertionPointToStart(&clonedForOp->getRegion(0).front());
    Value loopIndependentIterationCount =
        buildLoopIterationCount(rewriter, outerLoop, clonedForOp);

    // Assert the loop-independent iteration count can be computed.
    if (!loopIndependentIterationCount)
      llvm_unreachable("loop independence prerequisite not met");
    leadingPackedTensorIndexings.push_back(loopIndependentIterationCount);
    packedTensor = clonedForOp.getRegionIterArgs().front();
  }

  // Step 2. Construct offsets, sizes and strides for the innermost level of the
  // packing loop.
  int64_t nPackedLoops = clonedLoopIvs.size();
  // offsets = [clonedLoopIvs, 0 .. 0].
  offsets = SmallVector<OpFoldResult>{leadingPackedTensorIndexings.begin(),
                                      leadingPackedTensorIndexings.end()};
  offsets.append(paddedRank, rewriter.getIndexAttr(0));
  // sizes = [1 .. 1, transposedShape].
  sizes = SmallVector<OpFoldResult>(nPackedLoops, rewriter.getIndexAttr(1));
  for (int64_t sz : transposedTensorType.getShape()) {
    // TODO: go grab dims when needed, atm tensor::PadOp yields a static tensor.
    assert(!ShapedType::isDynamic(sz) && "padded tensor needs static sizes");
    sizes.push_back(rewriter.getIndexAttr(sz));
  }
  // strides = [1 .. 1].
  strides = SmallVector<OpFoldResult>(nPackedLoops + paddedRank,
                                      rewriter.getIndexAttr(1));

  // Step 3. Optionally transpose the padded tensor.
  GenericOp maybeTransposeOp;
  Value paddedTensor = bvm.lookup(opToHoist.getResult());
  if (!transposeVector.empty()) {
    Value outputTensor = rewriter.create<tensor::ExtractSliceOp>(
        loc, transposedTensorType, packedTensor, offsets, sizes, strides);
    maybeTransposeOp = makeTransposeOp(rewriter, loc, paddedTensor,
                                       outputTensor, transposeVector);
    paddedTensor = maybeTransposeOp.getResult(0);
  }

  // Innermost tensor.insert_slice and yields are optional / need loops.
  if (nPackedLoops > 0) {
    // Step 4. Create InsertSliceOp at the innermost loop level, inserting an
    // optionally transposed padded slice into the packed tensor.
    Value inserted = rewriter.create<tensor::InsertSliceOp>(
        loc, paddedTensor, packedTensor, offsets, sizes, strides);

    // Step 5. Iteratively pop the stack and propagate the yield.
    Value valueToYield = inserted;
    for (Value iv : llvm::reverse(clonedLoopIvs)) {
      auto forOp = scf::getForInductionVarOwner(iv);
      rewriter.setInsertionPointToEnd(&forOp.getRegion().front());
      rewriter.create<scf::YieldOp>(loc, valueToYield);
      valueToYield = forOp.getResult(0);
    }
  }

  return PackingLoopNestResult{offsets,
                               sizes,
                               strides,
                               clonedLoopIvs,
                               leadingPackedTensorIndexings,
                               maybeTransposeOp};
}

// If the original consumer of `sliceOp` was a `forOp` (i.e. through an iter
// arg), propagate the `packedTensor` value through the same iter arg.
// TODO: for multiple loops we need to track the use to the innermost loop.
static Value padThroughLoopIterArg(RewriterBase &rewriter, Value packedTensor,
                                   tensor::ExtractSliceOp sliceOp,
                                   scf::ForOp forOp) {
  OpOperand *pUse = nullptr;
  for (OpOperand &use : sliceOp->getUses()) {
    if (use.getOwner() == forOp) {
      assert(!pUse && "Multiple slice uses in the for loop");
      pUse = &use;
    }
  }
  assert(pUse && "No slice use in the for loop");
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(packedTensor.getDefiningOp());
  Value casted = rewriter.create<tensor::CastOp>(
      packedTensor.getLoc(), pUse->get().getType(), packedTensor);

  std::optional<unsigned> operandNumber =
      forOp.getIterArgNumberForOpOperand(*pUse);
  assert(operandNumber.has_value() && "expected a proper iter arg number");
  SmallVector<Value> initArgs = forOp.getInitArgs();
  initArgs[operandNumber.value()] = casted;
  rewriter.startRootUpdate(forOp);
  forOp.getInitArgsMutable().assign(initArgs);
  rewriter.finalizeRootUpdate(forOp);
  return forOp.getRegionIterArgForOpOperand(*pUse);
}

/// Produce a tensor extracted from the packingResult. This can be used as a
/// replacement for `opToHoist` in callers.
static Value replaceByPackingLoopNestResult(
    RewriterBase &rewriter, const IRMapping &bvm, tensor::PadOp opToHoist,
    RankedTensorType transposedTensorType, const HoistingAnalysis &analysis,
    const PackingLoopNestResult &packingResult) {
  // The replacement occurs under a single insertion point within the original
  // loop, just before opToHoist.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(opToHoist);

  Location loc = opToHoist->getLoc();
  RankedTensorType paddedTensorType = opToHoist.getResultType();
  int paddedRank = paddedTensorType.getRank();

  int64_t nPackedLoops = packingResult.clonedLoopIvs.size();
  LLVM_DEBUG(DBGS() << "nPackedLoops: " << nPackedLoops << " loops\n");

  scf::ForOp outerLoop = analysis.outermostEnclosingForOp;
  ArrayRef<scf::ForOp> packingLoops = analysis.packingLoops;

  Value packedTensor;
  SmallVector<Value> loopIterationCounts;
  SmallVector<OpFoldResult> offsets(nPackedLoops + paddedRank,
                                    rewriter.getIndexAttr(0));
  if (nPackedLoops > 0) {
    loopIterationCounts =
        llvm::to_vector<4>(llvm::map_range(packingLoops, [&](Operation *loop) {
          return buildLoopIterationCount(rewriter, outerLoop,
                                         cast<scf::ForOp>(loop));
        }));
    // Assert all loop iteration counts can be computed.
    if (llvm ::any_of(loopIterationCounts, [](Value v) { return !v; }))
      llvm_unreachable("loop independence prerequisite not met");

    // offsets = [maybe_leading_ivs = originalLoopIvs, 0 .. 0].
    std::copy(loopIterationCounts.begin(), loopIterationCounts.end(),
              offsets.begin());
    packedTensor =
        scf::getForInductionVarOwner(packingResult.clonedLoopIvs.front())
            ->getResult(0);
  } else {
    // If no loops were created, this is just hoisting without packing.
    auto padOp =
        cast<tensor::PadOp>(bvm.lookup(opToHoist.getResult()).getDefiningOp());
    tensor::ExtractSliceOp sliceOp = analysis.sliceOp;
    rewriter.startRootUpdate(padOp);
    padOp.getSourceMutable().assign(sliceOp.getResult());
    rewriter.finalizeRootUpdate(padOp);
    packedTensor = padOp;
  }

  LLVM_DEBUG(DBGS() << "packedTensor: " << packedTensor << "\n");

  // If the consumer of `padOp` was a `forOp`, propagate through iter args.
  scf::ForOp forOp = analysis.padConsumingForOp;
  if (forOp) {
    packedTensor =
        padThroughLoopIterArg(rewriter, packedTensor, analysis.sliceOp, forOp);
  }

  // offsets = [maybe_leading_ivs, 0 .. 0].
  // sizes = [1 .. 1, transposedShape] (defined above).
  // strides = [1 .. 1] (defined above)
  return rewriter.create<tensor::ExtractSliceOp>(
      loc, transposedTensorType, packedTensor, offsets, packingResult.sizes,
      packingResult.strides);
}

FailureOr<Value> mlir::linalg::hoistPaddingOnTensors(
    RewriterBase &rewriter, tensor::PadOp opToHoist, int64_t numLoops,
    ArrayRef<int64_t> transposeVector, tensor::PadOp &hoistedOp,
    SmallVectorImpl<GenericOp> &transposeOps) {
  LLVM_DEBUG(DBGS() << "\n"; DBGS() << " Try to hoist " << *(opToHoist) << "\n";
             DBGS() << " by " << numLoops << " loops\n");
  HoistingAnalysis analysis(opToHoist, numLoops);
  if (!analysis.isValid()) {
    LLVM_DEBUG(DBGS() << "--Analysis failed -> Skip\n");
    return failure();
  }

  // Update actual number of loops, which may be smaller.
  int nPackedLoops = analysis.packingLoops.size();
  LLVM_DEBUG(DBGS() << "\n";
             DBGS() << "Func:\n"
                    << *opToHoist->getParentOfType<func::FuncOp>() << "\n";
             DBGS() << "Start hoisting above " << nPackedLoops << " loops\n");

  Location loc = opToHoist->getLoc();
  RankedTensorType paddedTensorType = opToHoist.getResultType();

  // Compute the type of the transposed padded tensor.
  FailureOr<RankedTensorType> transposedTensorType =
      tensor::computeTransposedType(paddedTensorType, transposeVector);
  if (failed(transposedTensorType)) {
    LLVM_DEBUG(DBGS() << "--Could not compute transposed type -> Skip\n");
    return failure();
  }

  // Create the packed tensor<?x?x..? x transposedShape>.
  SmallVector<int64_t> packedShape(nPackedLoops, ShapedType::kDynamic);
  // TODO: go grab dims when needed, atm tensor::PadOp yields a static tensor.
  llvm::append_range(packedShape, transposedTensorType->getShape());
  auto packedTensorType = RankedTensorType::get(
      packedShape, transposedTensorType->getElementType());

  // Set the insertion point right before the outer loop and start packing.
  scf::ForOp outerLoop = analysis.outermostEnclosingForOp;
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(outerLoop);
  SmallVector<Value> dynamicTensorSizes =
      analysis.getPackedTensorSizes(rewriter, loc);
  auto emptyOp = rewriter.create<tensor::EmptyOp>(
      loc, packedTensorType.getShape(), packedTensorType.getElementType(),
      dynamicTensorSizes);

  /// Construct the packing loop nest.
  IRMapping bvm;
  PackingLoopNestResult packingResult =
      buildPackingLoopNest(rewriter, bvm, opToHoist, transposeVector,
                           *transposedTensorType, emptyOp, analysis);
  if (!transposeVector.empty())
    transposeOps.push_back(packingResult.maybeTransposeOp);

  // Now the packed tensor is ready, replace the original padding op by a
  // 1x..x1 slice [originalLoopIvs, 0 .. 0][1 .. 1, paddedShape][1 .. 1].
  Value newResult = replaceByPackingLoopNestResult(
      rewriter, bvm, opToHoist, *transposedTensorType, analysis, packingResult);
  if (!transposeVector.empty()) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(newResult.getDefiningOp());
    // Transpose the packed tensor back to the original storage order.
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, paddedTensorType.getShape(), paddedTensorType.getElementType());
    GenericOp unTransposeOp =
        makeTransposeOp(rewriter, loc, newResult, emptyTensor, transposeVector);
    newResult = unTransposeOp.getResult(0);
    transposeOps.push_back(unTransposeOp);
  }

  LLVM_DEBUG(DBGS() << "newResult: " << newResult << "\n");
  LLVM_DEBUG(
      DBGS() << "After hoisting: "
             << newResult.getDefiningOp()->getParentOfType<func::FuncOp>()
             << "\n");

  // Make the newly cloned `opToHoist` available to the caller.
  hoistedOp =
      cast<tensor::PadOp>(bvm.lookup(opToHoist.getResult()).getDefiningOp());

  LLVM_DEBUG(DBGS() << "--SUCCESS\n");
  return newResult;
}

FailureOr<Value>
mlir::linalg::hoistPaddingOnTensors(tensor::PadOp opToHoist, int64_t numLoops,
                                    ArrayRef<int64_t> transposeVector,
                                    tensor::PadOp &hoistedOp,
                                    SmallVectorImpl<GenericOp> &transposeOps) {
  IRRewriter rewriter(opToHoist.getContext());
  return hoistPaddingOnTensors(rewriter, opToHoist, numLoops, transposeVector,
                               hoistedOp, transposeOps);
}
