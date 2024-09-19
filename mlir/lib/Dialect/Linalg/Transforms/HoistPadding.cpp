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

#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
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
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/Debug.h"

using llvm::dbgs;

#define DEBUG_TYPE "hoist-padding"

#define DBGS() (dbgs() << '[' << DEBUG_TYPE << "] ")

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::linalg::detail;

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

/// Return at most nLevels of immediately enclosing scf::ForOp loops.
/// Stops at the first parent that is not an scf::ForOp.
/// Multi-loops such as scf.parallel or linalg.tiled_loop are not modeled atm.
/// Control-flow and other containing ops with regions are not modeled atm.
static void
getEnclosingLoopsUntil(tensor::PadOp padOp, scf::ForOp untilLoop,
                       SmallVector<scf::ForOp> &reverseEnclosingLoops) {
  scf::ForOp outermostEnclosingForOp = nullptr;
  Operation *nextEnclosingOp = padOp->getParentOp();
  while (outermostEnclosingForOp != untilLoop &&
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
  BackwardSliceOptions sliceOptions;
  sliceOptions.filter = [&](Operation *op) {
    return domInfo.dominates(outermostEnclosingForOp, op) &&
           !padOp->isProperAncestor(op);
  };
  sliceOptions.inclusive = true;

  // First, add the ops required to compute the region to the backwardSlice.
  SetVector<Value> valuesDefinedAbove;
  getUsedValuesDefinedAbove(padOp.getRegion(), padOp.getRegion(),
                            valuesDefinedAbove);
  for (Value v : valuesDefinedAbove) {
    getBackwardSlice(v, &backwardSlice, sliceOptions);
  }
  // Then, add the backward slice from padOp itself.
  getBackwardSlice(padOp.getOperation(), &backwardSlice, sliceOptions);
}

//===----------------------------------------------------------------------===//
// HoistPaddingAnalysis Implementation.
//===----------------------------------------------------------------------===//

namespace {
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
struct HoistPaddingAnalysis {
  HoistPaddingAnalysis(tensor::PadOp padOp, int numLoops);
  HoistPaddingAnalysis(tensor::PadOp padOp, scf::ForOp outermostEnclosingForOp);

  bool isValid() { return valid.has_value() && valid.value(); }
  bool isInvalid() { return valid.has_value() && !valid.value(); }

  /// Footprint of the hoistedPackedTensor, computed from the packingLoops.
  SmallVector<Value> getHoistedPackedTensorSizes(RewriterBase &rewriter,
                                                 Location loc) const;

  /// Performs optional hoisting to enable hoist padding to occur. This may be
  /// necessary when `sliceOp` is not defined outside of the outermost enclosing
  /// loop we want to hoist above.
  ///
  /// Example:
  /// ```
  /// %source = linalg.fill(%cst, %arg0)
  /// // %source is available for packing here!
  /// scf.for %i
  ///   scf.for %j
  ///     scf.for %k
  ///       %slice = tensor.extract_slice %source [%i, %j]
  ///       %padded_slice = tensor.pad %slice
  /// ```
  void enableHoistPadding(RewriterBase &rewriter);

  /// Common analysis builder to finalize the construction of the analysis once
  /// optional `enableHoistPadding` has run.
  /// `reverseEnclosingLoops.back()` is the loop to hoist above.
  void finalizeHoistPaddingAnalysis();

private:
  /// Encodes whether the analysis is valid and hoisting can proceed.
  std::optional<bool> valid;

  /// The padOp to hoist.
  tensor::PadOp opToHoist;

  /// Immediately enclosing loops considered for hoisting padding.
  SmallVector<scf::ForOp> reverseEnclosingLoops;

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
  LogicalResult dropNonIndexDependencies();

public:
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
};

} // namespace

HoistPaddingAnalysis::HoistPaddingAnalysis(tensor::PadOp padOp, int numLoops)
    : valid(std::nullopt), opToHoist(padOp) {
  // Get at most `numLoops` of immediately enclosing loops.
  getAtMostNEnclosingLoops(opToHoist, numLoops, reverseEnclosingLoops);
  if (reverseEnclosingLoops.empty()) {
    LLVM_DEBUG(DBGS() << "--No immediately enclosing loop -> Skip\n");
    valid = false;
    return;
  }
  outermostEnclosingForOp = reverseEnclosingLoops.back();
  sliceOp = opToHoist.getSource().getDefiningOp<tensor::ExtractSliceOp>();
  if (!sliceOp) {
    LLVM_DEBUG(DBGS() << "--Cannot find the extract slice op -> Skip\n");
    valid = false;
    return;
  }
}

HoistPaddingAnalysis::HoistPaddingAnalysis(tensor::PadOp padOp,
                                           scf::ForOp outermostEnclosingForOp)
    : valid(std::nullopt), opToHoist(padOp) {
  // Get enclosing loops until outermostEnclosingForOp.
  getEnclosingLoopsUntil(opToHoist, outermostEnclosingForOp,
                         reverseEnclosingLoops);
  if (reverseEnclosingLoops.empty()) {
    LLVM_DEBUG(DBGS() << "--No immediately enclosing loop -> Skip\n");
    valid = false;
    return;
  }
  this->outermostEnclosingForOp = reverseEnclosingLoops.back();
  if (this->outermostEnclosingForOp != outermostEnclosingForOp) {
    LLVM_DEBUG(DBGS() << "--Unexpected outermost enclosing loop -> Skip\n");
    valid = false;
    return;
  }
  sliceOp = opToHoist.getSource().getDefiningOp<tensor::ExtractSliceOp>();
  if (!sliceOp) {
    LLVM_DEBUG(DBGS() << "--Cannot find the extract slice op -> Skip\n");
    valid = false;
    return;
  }
}

void HoistPaddingAnalysis::enableHoistPadding(RewriterBase &rewriter) {
  if (isInvalid())
    return;
  // If the padded data is not yet available before entering the outermost
  // enclosing loop, try to apply hoisting on this outermost loop.
  // TODO: we may want finer-grained hoisting of only that particular `sliceOp`.
  if (!outermostEnclosingForOp.isDefinedOutsideOfLoop(sliceOp.getSource())) {
    outermostEnclosingForOp = cast<scf::ForOp>(
        hoistLoopInvariantSubsets(rewriter, outermostEnclosingForOp));
  }
}

void HoistPaddingAnalysis::finalizeHoistPaddingAnalysis() {
  if (isInvalid())
    return;

  if (!outermostEnclosingForOp.isDefinedOutsideOfLoop(sliceOp.getSource())) {
    LLVM_DEBUG(DBGS() << "--outermostEnclosingForOp:\n"
                      << outermostEnclosingForOp << "\n"
                      << "--sliceOp: " << sliceOp << "\n"
                      << "--sliceOp.getSource(): " << sliceOp.getSource()
                      << "\n");
    LLVM_DEBUG(DBGS() << "----Source not defined outside of loops -> Skip\n");
    valid = false;
    return;
  }
  if (sliceOp->hasOneUse()) {
    padConsumingForOp = dyn_cast<scf::ForOp>(*(sliceOp->getUsers().begin()));
  }

  // Check the region of `padOp` depends on a constant only. Adding hoisting
  // support for arbitrary padding regions would require cloning all
  // dependencies captured by the padding region.
  Value paddingValue = opToHoist.getConstantPaddingValue();
  if (!paddingValue ||
      !isa_and_nonnull<arith::ConstantOp>(paddingValue.getDefiningOp())) {
    LLVM_DEBUG(DBGS() << "Cannot find constant padding value -> Skip\n");
    valid = false;
    return;
  }

  computeBackwardSlice(opToHoist, outermostEnclosingForOp, backwardSlice);
  if (backwardSlice.size() <= 1) {
    valid = false;
    return;
  }

  debugPrintBackwardSlice(backwardSlice);
  // Remove all ops in the backward slice that are not used to index
  // the padded tensor. In particular, keep `padOp`, `sliceOp`, and
  // the loop and affine operations used for the index computation.
  if (failed(dropNonIndexDependencies())) {
    LLVM_DEBUG(DBGS() << "--Cannot dropNonIndexDependencies -> Skip\n");
    valid = false;
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

LogicalResult HoistPaddingAnalysis::dropNonIndexDependencies() {
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

  // Starting from `opToHoist` and `sliceOp` walk the use-def edges of index
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
    // Add the index operands of `opToHoist` and `sliceOp` to start the
    // exploration of the index computation.
    if (op == opToHoist || op == sliceOp) {
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
    // exception are constant operations that may be used by `opToHoist`.
    if (!isa<arith::ConstantOp>(op))
      operationsToRemove.insert(op);
  }
  backwardSlice.set_subtract(operationsToRemove);
  return success();
}

SmallVector<Value>
HoistPaddingAnalysis::getHoistedPackedTensorSizes(RewriterBase &rewriter,
                                                  Location loc) const {
  SmallVector<Value> dynamicTensorSizes;

  // Upper bound the packing loop lengths to size the packed tensor. Taking
  // upper bounds can make the sizes of the packed tensor independent of the
  // enclosing loops. This independence is a prerequisite for reusing the same
  // buffer for all enclosing loop iterations and hoisting its allocation out
  // of the enclosing loops.
  for (auto forOp : packingLoops) {
    // Compute an upper bound `ubVal` for the upper bound of `forOp`.
    FailureOr<OpFoldResult> loopUb = affine::reifyIndexValueBound(
        rewriter, loc, presburger::BoundType::UB, forOp.getUpperBound(),
        /*stopCondition=*/
        [&](Value v, std::optional<int64_t> d, ValueBoundsConstraintSet &cstr) {
          if (v == forOp.getUpperBound())
            return false;
          // Compute a bound that is independent of any affine op results.
          Operation *op = v.getDefiningOp();
          if (!op)
            return true;
          return !isa<affine::AffineMinOp, affine::AffineMaxOp,
                      affine::AffineApplyOp>(op);
        },
        /*closedUB=*/true);
    assert(succeeded(loopUb) && "could not get upper bound");
    Value ubVal = getValueOrCreateConstantIndexOp(rewriter, loc, *loopUb);

    // Compute the maximal packing loop length as (ub - lb).ceilDiv(step) and
    // store the result to `dynamicTensorSizes`.
    // TODO: instead of using the lower bound of `forOp` directly, implement a
    // lower bound computation similar to the upper bound computation.
    AffineExpr lb, ub, step;
    bindDims(rewriter.getContext(), lb, ub);
    bindSymbols(rewriter.getContext(), step);
    Value res = rewriter.createOrFold<affine::AffineApplyOp>(
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

//===----------------------------------------------------------------------===//
// buildPackingLoopNest Implementation.
//===----------------------------------------------------------------------===//

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
  return rewriter.createOrFold<affine::AffineApplyOp>(
      loc, (iv - lb).ceilDiv(step), ValueRange{ivVal, lbVal, stepVal});
}

// Build a packing loop nest by iteratively traversing the backward slice and
// clone the operations, iteratively stepping into the loops that we encounter.
// The implementation proceeds in a stack-like fashion:
//   1. Iteratively clone and step into the loops, pushing the
//   `hoistedPackedTensor`
//      deeper in the stack.
//   2. At the innermost loop level, create a GenericOp if `transposeVector` is
//      non-empty.
//   3. At the innermost loop level, create a InsertSliceOp.
//   4. Iteratively pop and yield the result of the InsertSliceOp across the
//      cloned loops.
static FailureOr<PackingResult> buildPackingLoopNestImpl(
    RewriterBase &rewriter, IRMapping &bvm, tensor::PadOp opToHoist,
    ArrayRef<int64_t> transposeVector, RankedTensorType transposedTensorType,
    tensor::EmptyOp emptyOp, const HoistPaddingAnalysis &analysis) {
  SmallVector<OpFoldResult> offsets, sizes, strides;
  SmallVector<Value> clonedLoopIvs, leadingHoistedPackedTensorIndexings;

  scf::ForOp outerLoop = analysis.outermostEnclosingForOp;

  Location loc = opToHoist->getLoc();
  RankedTensorType paddedTensorType = opToHoist.getResultType();
  int paddedRank = paddedTensorType.getRank();

  // Step 0. Populate bvm with opToHoist.getSource if relevant.
  BlockArgument bbArg = dyn_cast<BlockArgument>(opToHoist.getSource());
  while (bbArg) {
    auto forOp = dyn_cast<scf::ForOp>(bbArg.getOwner()->getParentOp());
    if (!forOp)
      break;
    if (forOp != outerLoop && !outerLoop->isAncestor(forOp))
      break;
    OpOperand &operand = *forOp.getTiedLoopInit(bbArg);
    bvm.map(bbArg, operand.get());
    bbArg = dyn_cast<BlockArgument>(operand.get());
  }

  // Step 1. iteratively clone loops and push `hoistedPackedTensor`.
  Value hoistedPackedTensor = emptyOp.getResult();
  OpBuilder::InsertionGuard g(rewriter);
  for (Operation *op : analysis.backwardSlice) {
    // Specifically sit out in the extract_slice(hoistedPackedTensor) case: this
    // is the piece we seek to replace.
    if (auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
      if (bvm.lookupOrDefault(sliceOp.getSource()) == hoistedPackedTensor) {
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

    // Create a packing loop that takes `hoistedPackedTensor` as iteration
    // argument.
    auto clonedForOp = rewriter.create<scf::ForOp>(
        loc, bvm.lookupOrDefault(forOp.getLowerBound()),
        bvm.lookupOrDefault(forOp.getUpperBound()),
        bvm.lookupOrDefault(forOp.getStep()), hoistedPackedTensor);

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
    leadingHoistedPackedTensorIndexings.push_back(
        loopIndependentIterationCount);
    hoistedPackedTensor = clonedForOp.getRegionIterArgs().front();
  }

  // Step 2. Construct offsets, sizes and strides for the innermost level of the
  // packing loop.
  int64_t nPackedLoops = clonedLoopIvs.size();
  // offsets = [clonedLoopIvs, 0 .. 0].
  offsets =
      SmallVector<OpFoldResult>{leadingHoistedPackedTensorIndexings.begin(),
                                leadingHoistedPackedTensorIndexings.end()};
  offsets.append(paddedRank, rewriter.getIndexAttr(0));
  // sizes = [1 .. 1, transposedShape].
  sizes = SmallVector<OpFoldResult>(nPackedLoops, rewriter.getIndexAttr(1));
  for (int64_t sz : transposedTensorType.getShape()) {
    // TODO: go grab dims when needed, atm tensor::PadOp yields a static tensor.
    if (ShapedType::isDynamic(sz))
      return failure();
    sizes.push_back(rewriter.getIndexAttr(sz));
  }
  // strides = [1 .. 1].
  strides = SmallVector<OpFoldResult>(nPackedLoops + paddedRank,
                                      rewriter.getIndexAttr(1));

  // Step 3. Optionally transpose the padded tensor.
  TransposeOp maybeTransposeOp;
  Value paddedTensor = bvm.lookup(opToHoist.getResult());
  if (!transposeVector.empty()) {
    Value outputTensor = rewriter.create<tensor::ExtractSliceOp>(
        loc, transposedTensorType, hoistedPackedTensor, offsets, sizes,
        strides);
    maybeTransposeOp = rewriter.create<linalg::TransposeOp>(
        loc, paddedTensor, outputTensor, transposeVector);
    paddedTensor = maybeTransposeOp.getResult()[0];
  }

  // Innermost tensor.insert_slice and yields are optional / need loops.
  if (nPackedLoops > 0) {
    // Step 4. Create InsertSliceOp at the innermost loop level, inserting an
    // optionally transposed padded slice into the packed tensor.
    Value inserted = rewriter.create<tensor::InsertSliceOp>(
        loc, paddedTensor, hoistedPackedTensor, offsets, sizes, strides);

    // Step 5. Iteratively pop the stack and propagate the yield.
    Value valueToYield = inserted;
    for (Value iv : llvm::reverse(clonedLoopIvs)) {
      auto forOp = scf::getForInductionVarOwner(iv);
      rewriter.setInsertionPointToEnd(&forOp.getRegion().front());
      rewriter.create<scf::YieldOp>(loc, valueToYield);
      valueToYield = forOp.getResult(0);
    }
  }

  return PackingResult{
      offsets,
      sizes,
      strides,
      clonedLoopIvs,
      leadingHoistedPackedTensorIndexings,
      maybeTransposeOp,
      cast<tensor::PadOp>(bvm.lookup(opToHoist.getResult()).getDefiningOp())};
}

/// Build the packing loop nest required to hoist `opToHoist` above
/// `outermostEnclosingForOp`.
/// The loop nest is built just before `outermostEnclosingForOp`.
static FailureOr<PackingResult> buildPackingLoopNestImpl(
    RewriterBase &rewriter, IRMapping &bvm, tensor::PadOp opToHoist,
    ArrayRef<int64_t> transposeVector, const HoistPaddingAnalysis &analysis) {
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
  auto hoistedPackedTensorType = RankedTensorType::get(
      packedShape, transposedTensorType->getElementType());

  // Set the insertion point right before the outer loop and start packing.
  scf::ForOp outerLoop = analysis.outermostEnclosingForOp;
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(outerLoop);
  SmallVector<Value> dynamicTensorSizes =
      analysis.getHoistedPackedTensorSizes(rewriter, loc);
  auto emptyOp = rewriter.create<tensor::EmptyOp>(
      loc, hoistedPackedTensorType.getShape(),
      hoistedPackedTensorType.getElementType(), dynamicTensorSizes);

  return buildPackingLoopNestImpl(rewriter, bvm, opToHoist, transposeVector,
                                  *transposedTensorType, emptyOp, analysis);
}

/// Build the packing loop nest required to hoist `opToHoist` above
/// `outermostEnclosingForOp`.
/// The loop nest is built just before `outermostEnclosingForOp`.
FailureOr<PackingResult> mlir::linalg::detail::buildPackingLoopNest(
    RewriterBase &rewriter, tensor::PadOp opToHoist,
    scf::ForOp outermostEnclosingForOp, ArrayRef<int64_t> transposeVector) {
  HoistPaddingAnalysis analysis(opToHoist, outermostEnclosingForOp);
  analysis.enableHoistPadding(rewriter);
  analysis.finalizeHoistPaddingAnalysis();
  if (!analysis.isValid()) {
    LLVM_DEBUG(DBGS() << "--Analysis failed -> Skip\n");
    return failure();
  }
  IRMapping bvm;
  return buildPackingLoopNestImpl(rewriter, bvm, opToHoist, transposeVector,
                                  analysis);
}

//===----------------------------------------------------------------------===//
// hoistPaddingOnTensors Implementation.
//===----------------------------------------------------------------------===//

/// Return true if we can walk back the use-def chain from `extractSliceOp` to
/// expectedSource going through DestinationStyleOpInterface inits only.
/// This is a poor man's analysis that is sufficient to check the extractSliceOp
/// the matches tensor.pad we want to hoist.
/// In the future, it will be easier to ensure this with a matching symmetric
/// tensor.unpad op.
static bool tracesBackToExpectedValue(tensor::ExtractSliceOp extractSliceOp,
                                      Value expectedSource) {
  LLVM_DEBUG(DBGS() << "Start tracesBackToExpectedValue on: " << extractSliceOp
                    << "\n");
  LLVM_DEBUG(DBGS() << "--with extractSlice: " << extractSliceOp << "\n");
  Value source = extractSliceOp.getSource();
  LLVM_DEBUG(DBGS() << "--with starting source: " << source << "\n");
  while (source && source != expectedSource) {
    auto destOp =
        dyn_cast_or_null<DestinationStyleOpInterface>(source.getDefiningOp());
    if (!destOp)
      break;
    LLVM_DEBUG(DBGS() << "--step dest op: " << destOp << "\n");
    source = destOp.getDpsInitOperand(cast<OpResult>(source).getResultNumber())
                 ->get();
  }
  LLVM_DEBUG(DBGS() << "--final source: " << source << "\n");
  LLVM_DEBUG(DBGS() << "--expected source: " << expectedSource << "\n");
  return source == expectedSource;
}

/// If the original consumer of `outerSliceOp` was a `forOp` (i.e. through an
/// iter arg), propagate the `hoistedPackedTensor` value through the same iter
/// arg.
/// TODO: for multiple loops we need to track the use to the innermost loop.
///
/// Match:
/// ```
///   %outerSliceOp = tensor.extract_slice ..
///   %f = scf.for ... iter_args(%arg0 = %outerSliceOp) {
///     %hoistedPackedTensor = tensor.pad %arg0
///     %1 = compute %hoistedPackedTensor
///     %2 = tensor.extract_slice %1
///     scf.yield %2
///   }
/// ```
///
/// and rewrite as:
/// ```
///   %outerSliceOp = tensor.extract_slice ..
///   %hoistedPackedTensor = tensor.pad %outerSliceOp
///   %f = scf.for ... iter_args(%arg0 = %hoistedPackedTensor) {
///     %1 = compute %arg0
///     scf.yield %1
///   }
///   %2 = tensor.extract_slice %forOp
/// ```
///
/// Return null when no rewrite happened.
static tensor::ExtractSliceOp
padThroughLoopIterArg(RewriterBase &rewriter, Value paddedValueBeforeHoisting,
                      Value hoistedPackedTensor,
                      tensor::ExtractSliceOp outerSliceOp, scf::ForOp forOp) {
  LLVM_DEBUG(DBGS() << "Start padThroughLoopIterArg on: " << forOp << "\n");
  LLVM_DEBUG(DBGS() << "--paddedValueBeforeHoisting: "
                    << paddedValueBeforeHoisting << "\n");
  OpOperand *pUse = nullptr;
  for (OpOperand &use : outerSliceOp->getUses()) {
    if (use.getOwner() == forOp) {
      assert(!pUse && "Multiple slice uses in the for loop");
      pUse = &use;
    }
  }
  assert(pUse && "No slice use in the for loop");
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(hoistedPackedTensor.getDefiningOp());

  unsigned iterArgNumber = forOp.getTiedLoopResult(pUse).getResultNumber();
  auto yieldingExtractSliceOp = forOp.getYieldedValues()[iterArgNumber]
                                    .getDefiningOp<tensor::ExtractSliceOp>();
  if (!yieldingExtractSliceOp)
    return tensor::ExtractSliceOp();

  // Poor man's analysis sufficient to ensure extractSlice matches tensor.pad.
  // In the future, it will be easier to ensure this with a matching symmetric
  // tensor.unpad op.
  if (!tracesBackToExpectedValue(yieldingExtractSliceOp,
                                 paddedValueBeforeHoisting))
    return tensor::ExtractSliceOp();

  SmallVector<Value> initArgs = forOp.getInitArgs();
  initArgs[iterArgNumber] = hoistedPackedTensor;
  SmallVector<Value> yieldOperands = llvm::to_vector(forOp.getYieldedValues());
  yieldOperands[iterArgNumber] = yieldingExtractSliceOp.getSource();

  int64_t numOriginalForOpResults = initArgs.size();
  LLVM_DEBUG(DBGS() << "numOriginalForOpResults: " << numOriginalForOpResults
                    << "\n");
  tensor::ExtractSliceOp extracted;
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(forOp);
    extracted = rewriter.create<tensor::ExtractSliceOp>(
        hoistedPackedTensor.getLoc(), hoistedPackedTensor,
        outerSliceOp.getMixedOffsets(), outerSliceOp.getMixedSizes(),
        outerSliceOp.getMixedStrides());
    rewriter.replaceAllUsesWith(forOp.getResult(iterArgNumber), extracted);
  }
  scf::ForOp newForOp = cast<scf::ForOp>(*forOp.replaceWithAdditionalYields(
      rewriter, initArgs, /*replaceInitOperandUsesInLoop=*/true,
      [&](OpBuilder &b, Location loc, ArrayRef<BlockArgument> newBBArgs) {
        return yieldOperands;
      }));

  LLVM_DEBUG(DBGS() << "newForOp results: " << newForOp.getNumResults()
                    << "\n");
  LLVM_DEBUG(DBGS() << "replace source of: " << extracted << "\n");
  LLVM_DEBUG(DBGS() << "with result #"
                    << numOriginalForOpResults + iterArgNumber
                    << " of forOp, giving us: " << extracted << "\n");
  rewriter.startOpModification(extracted);
  extracted.getSourceMutable().assign(
      newForOp.getResult(numOriginalForOpResults + iterArgNumber));
  rewriter.finalizeOpModification(extracted);

  LLVM_DEBUG(DBGS() << "replace uses of: " << paddedValueBeforeHoisting
                    << "\n");
  LLVM_DEBUG(DBGS() << "with region iter arg #"
                    << numOriginalForOpResults + iterArgNumber << "\n");
  rewriter.replaceAllUsesWith(
      paddedValueBeforeHoisting,
      newForOp.getRegionIterArg(numOriginalForOpResults + iterArgNumber));

  return extracted;
}

/// Produce a tensor extracted from the packingResult. This can be used as a
/// replacement for `opToHoist` in callers.
static Value replaceByPackingResult(RewriterBase &rewriter,
                                    const IRMapping &bvm,
                                    tensor::PadOp opToHoist,
                                    RankedTensorType transposedTensorType,
                                    const HoistPaddingAnalysis &analysis,
                                    const PackingResult &packingResult) {
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

  Value hoistedPackedTensor;
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
    hoistedPackedTensor =
        scf::getForInductionVarOwner(packingResult.clonedLoopIvs.front())
            ->getResult(0);
  } else {
    // If no loops were created, this is just hoisting without packing.
    hoistedPackedTensor = bvm.lookup(opToHoist.getResult());
  }

  LLVM_DEBUG(DBGS() << "hoistedPackedTensor: " << hoistedPackedTensor << "\n");

  // If the consumer of `padOp` was a `forOp`, propagate through iter args.
  scf::ForOp forOp = analysis.padConsumingForOp;
  if (forOp) {
    return padThroughLoopIterArg(rewriter, opToHoist, hoistedPackedTensor,
                                 analysis.sliceOp, forOp);
  }

  // offsets = [maybe_leading_ivs, 0 .. 0].
  // sizes = [1 .. 1, transposedShape] (defined above).
  // strides = [1 .. 1] (defined above)
  return rewriter.create<tensor::ExtractSliceOp>(
      loc, transposedTensorType, hoistedPackedTensor, offsets,
      packingResult.sizes, packingResult.strides);
}

FailureOr<Value> mlir::linalg::hoistPaddingOnTensors(
    RewriterBase &rewriter, tensor::PadOp opToHoist, int64_t numLoops,
    ArrayRef<int64_t> transposeVector, tensor::PadOp &hoistedOp,
    SmallVectorImpl<TransposeOp> &transposeOps) {
  LLVM_DEBUG(DBGS() << "\n"; DBGS() << " Try to hoist " << *(opToHoist) << "\n";
             DBGS() << " by " << numLoops << " loops\n");

  HoistPaddingAnalysis analysis(opToHoist, numLoops);
  analysis.enableHoistPadding(rewriter);
  analysis.finalizeHoistPaddingAnalysis();
  if (!analysis.isValid()) {
    LLVM_DEBUG(DBGS() << "--Analysis failed -> Skip\n");
    return failure();
  }

  /// Construct the packing loop nest.
  IRMapping bvm;
  FailureOr<PackingResult> packingResult = buildPackingLoopNestImpl(
      rewriter, bvm, opToHoist, transposeVector, analysis);
  if (failed(packingResult)) {
    LLVM_DEBUG(DBGS() << "--buildPackingLoopNestImpl failed -> Skip\n");
    return failure();
  }

  if (!transposeVector.empty())
    transposeOps.push_back(packingResult->maybeTransposeOp);

  FailureOr<RankedTensorType> transposedTensorType =
      tensor::computeTransposedType(opToHoist.getResultType(), transposeVector);
  assert(succeeded(transposedTensorType) && "unexpected failure in type");

  // Now the packed tensor is ready, replace the original padding op by a
  // 1x..x1 slice [originalLoopIvs, 0 .. 0][1 .. 1, paddedShape][1 .. 1].
  Value newResult =
      replaceByPackingResult(rewriter, bvm, opToHoist, *transposedTensorType,
                             analysis, *packingResult);

  Location loc = opToHoist->getLoc();
  RankedTensorType paddedTensorType = opToHoist.getResultType();
  if (!transposeVector.empty()) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(newResult.getDefiningOp());
    // Transpose the packed tensor back to the original storage order.
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, paddedTensorType.getShape(), paddedTensorType.getElementType());
    TransposeOp unTransposeOp = rewriter.create<linalg::TransposeOp>(
        loc, newResult, emptyTensor, transposeVector);
    newResult = unTransposeOp.getResult()[0];
    transposeOps.push_back(unTransposeOp);
  }

  LLVM_DEBUG(DBGS() << "newResult: " << newResult << "\n");
  LLVM_DEBUG(
      DBGS() << "After hoisting: "
             << newResult.getDefiningOp()->getParentOfType<func::FuncOp>()
             << "\n");

  // Make the newly cloned `opToHoist` available to the caller.
  hoistedOp = packingResult->hoistedPadOp;

  LLVM_DEBUG(DBGS() << "--SUCCESS\n");
  return newResult;
}

FailureOr<Value> mlir::linalg::hoistPaddingOnTensors(
    tensor::PadOp opToHoist, int64_t numLoops,
    ArrayRef<int64_t> transposeVector, tensor::PadOp &hoistedOp,
    SmallVectorImpl<TransposeOp> &transposeOps) {
  IRRewriter rewriter(opToHoist.getContext());
  return hoistPaddingOnTensors(rewriter, opToHoist, numLoops, transposeVector,
                               hoistedOp, transposeOps);
}
