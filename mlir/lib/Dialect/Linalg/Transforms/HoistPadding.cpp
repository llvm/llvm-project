//===- HoistPadding.cpp - Hoisting transformation for PadTensorOp ---------===//
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

#include "mlir/Dialect/Linalg/Transforms/HoistPadding.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Vector/VectorUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/LoopUtils.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

using llvm::dbgs;

#define DEBUG_TYPE "hoist-padding"

#define DBGS() (dbgs() << '[' << DEBUG_TYPE << "] ")

using namespace mlir;
using namespace mlir::linalg;

/// Analysis class to support PadTensorOp hoisting across multiple enclosing
/// loops. The failure conditions are:
///   1. Pad op has a use that is not an input of a LinalgOp.
///   2. There is no immediately enclosing scf::ForOp.
///   3. The backward slice from the pad op to the scf::ForOp to hoist above
///      contains an unknown op with a region.
///   4. The backward slice from the pad op to the scf::ForOp to hoist above is
///      empty.
///   5. The source tensor of pad op is not defined by an extract slice op.
///   6. The source tensor of the extract slice op is not defined outside of
///      the outermost enclosing scf::ForOp.
///   7. There is no enclosing scf::ForOp that indexes the padded data.
/// Other cases succeed and will trigger hoisting of the pad op.
struct HoistingAnalysis {
  HoistingAnalysis(PadTensorOp padTensorOp, int numLoops);

  bool isValid() { return valid; }

  /// Footprint of the packedTensor, computed from the packingLoops.
  SmallVector<Value> getPackedTensorSizes(ImplicitLocOpBuilder &b);

  /// The outermost loop, determined by `nLevels` above which `padTensorOp` will
  /// be hoisted.
  scf::ForOp outermostEnclosingForOp;

  /// Backward slice rooted at `padTensorOp` and nested under
  /// `outermostEnclosingForOp`.
  SetVector<Operation *> backwardSlice;

  /// The scf::ForOp immediately enclosing `padTensorOp` such that:
  ///  1. they are nested under `outermostEnclosingForOp` (inclusive)
  ///  2. whose induction variable is used, directly or indirectly, in the
  ///     computation of `padTensorOp`.
  /// The span of these loops determines the footprint of the packed tensor.
  SmallVector<scf::ForOp> packingLoops;

private:
  /// Returns the loops in `backwardSlice` used to index the padded data. The
  /// method starts from `padTensorOp` and `sliceOp`, follows the use-def
  /// chains of their index operands, and stores any enclosing loop whose
  /// induction variable is part of the walked index computation.
  ///
  /// Example:
  /// ```
  /// %source = linalg.fill(%cst, %arg0)
  /// scf.for %i
  ///   scf.for %j
  ///     scf.for %k // not used to index %source!
  ///       %ubi = affine.min #map(%i)
  ///       %ubj = affine.min #map(%j)
  ///       %slice = tensor.extract_slice %source [%i, %j] [%ubi, %ubj]
  ///       %padded_slice = linalg.pad_tensor %slice
  /// ```
  /// getIndexingLoops(%padded_slice, %slice) returns [scf.for %i, scf.for %j]
  SmallVector<scf::ForOp> getIndexingLoops(PadTensorOp padTensorOp,
                                           tensor::ExtractSliceOp sliceOp);

  /// Encodes whether the analysis is valid and hoisting can proceed.
  bool valid;
};

/// Return true if all uses of `padTensorOp` are an input tensor of some
/// LinalgOp.
static bool isOnlyUsedAsInputOfLinalgOp(PadTensorOp padTensorOp) {
  for (OpOperand &use : padTensorOp.result().getUses()) {
    auto linalgUser = dyn_cast<linalg::LinalgOp>(use.getOwner());
    if (!linalgUser || !linalgUser.isInputTensor(&use)) {
      LLVM_DEBUG(DBGS() << "Found a use of " << *(padTensorOp)
                        << "\nthat is not an input tensor of a LinalgOp, "
                        << "cannot hoist\n"
                        << *(use.getOwner()) << "\n");
      return false;
    }
  }
  return true;
}

/// Return at most nLevels of immediately enclosing scf::ForOp loops.
/// Stops at the first parent that is not an scf::ForOp.
/// Multi-loops such as scf.parallel or linalg.tiled_loop are not modeled atm.
/// Control-flow and other containing ops with regions are not modeled atm.
static void
getAtMostNEnclosingLoops(PadTensorOp padTensorOp, int nLevels,
                         SmallVector<scf::ForOp> &reverseEnclosingLoops) {
  AsmState state(padTensorOp->getParentOfType<mlir::FuncOp>());
  (void)state;
  scf::ForOp outermostEnclosingForOp = nullptr;
  Operation *nextEnclosingOp = padTensorOp->getParentOp();
  while (nLevels-- > 0 &&
         (outermostEnclosingForOp = dyn_cast<scf::ForOp>(nextEnclosingOp))) {
    LLVM_DEBUG(
        DBGS() << "loops: ";
        outermostEnclosingForOp.getInductionVar().printAsOperand(dbgs(), state);
        dbgs() << "\n");
    reverseEnclosingLoops.push_back(outermostEnclosingForOp);
    nextEnclosingOp = outermostEnclosingForOp->getParentOp();
  }
}

HoistingAnalysis::HoistingAnalysis(PadTensorOp padTensorOp, int numLoops) {
  valid = false;

  // Bail on any use that isn't an input of a Linalg op.
  // Hoisting of inplace updates happens after vectorization.
  if (!isOnlyUsedAsInputOfLinalgOp(padTensorOp))
    return;

  // Get at most nLevels of immediately enclosing loops.
  SmallVector<scf::ForOp> reverseEnclosingLoops;
  getAtMostNEnclosingLoops(padTensorOp, numLoops, reverseEnclosingLoops);
  if (reverseEnclosingLoops.empty()) {
    LLVM_DEBUG(DBGS() << "No immediately enclosing loop -> skip\n");
    return;
  }

  outermostEnclosingForOp = reverseEnclosingLoops.back();

  // Get all the ops in the backwards slice starting from `padTensorOp` and that
  // are dominated by the outermost enclosing loop.
  // Bail on any op with a region that is not either a scf::ForOp or a LinalgOp.
  bool analysisFailure = false;
  DominanceInfo domInfo(outermostEnclosingForOp);
  getBackwardSlice(
      padTensorOp.getOperation(), &backwardSlice, [&](Operation *op) {
        if (!domInfo.dominates(outermostEnclosingForOp, op))
          return false;
        if (op != padTensorOp && op->getNumRegions() > 0 &&
            !isa<scf::ForOp, LinalgOp>(op)) {
          analysisFailure = true;
          LLVM_DEBUG(DBGS()
                     << "Unsupported op with region: " << *op << " -> skip\n");
          return false;
        }
        return true;
      });

  if (analysisFailure || backwardSlice.empty())
    return;

  // Get the `sliceOp` that defines the source tensor of `padTensorOp` and
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
  //       %padded_slice = linalg.pad_tensor %slice
  // ```
  auto sliceOp = padTensorOp.source().getDefiningOp<tensor::ExtractSliceOp>();
  if (!sliceOp) {
    LLVM_DEBUG(DBGS() << "Cannot find the extract slice op -> skip\n");
    return;
  }
  if (!outermostEnclosingForOp.isDefinedOutsideOfLoop(sliceOp.source())) {
    LLVM_DEBUG(DBGS() << "Source not defined outside of loops -> skip\n");
    return;
  }

  // Search the loops found in `backwardSlice` used to index the padded data.
  SmallVector<scf::ForOp> indexingLoops =
      getIndexingLoops(padTensorOp, sliceOp);

  // Add only the loops part of `indexingLoops` to the packing loops. All other
  // loops are not used to index the padded data and consequently access the
  // same data in every loop iteration. Adding them to the packing loops would
  // increase the cache footprint of the packed data by storing the same data
  // multiple times.
  for (scf::ForOp forOp : llvm::reverse(reverseEnclosingLoops))
    if (!indexingLoops.empty() && indexingLoops.back() == forOp)
      packingLoops.push_back(indexingLoops.pop_back_val());
  assert(indexingLoops.empty() &&
         "expect the all indexing loops are enclosing loops");

  if (packingLoops.empty()) {
    LLVM_DEBUG(DBGS() << "Cannot find a packing loop -> skip\n");
    return;
  }

  // The analysis is valid and hoisting can occur.
  valid = true;
}

SmallVector<scf::ForOp>
HoistingAnalysis::getIndexingLoops(PadTensorOp padTensorOp,
                                   tensor::ExtractSliceOp sliceOp) {
  // Set of all values used for index computation.
  SetVector<Value> indexEdges;

  // Add all index operands of `operation` to `indexEdges`. An index operand is
  // an operand of type index.
  auto addIndexOperandsToIndexEdges = [&](Operation *operation) {
    for (Value operand : operation->getOperands())
      if (operand.getType().isIndex())
        indexEdges.insert(operand);
  };

  // Starting from `padTensorOp` and `sliceOp` walk the use-def edges of index
  // type in `backwardSlice`. Add the index operands of an operation to
  // `indexEdges` if one of its results is an index edge found so far and store
  // all loops part of the index computation to `indexingLoops`.
  //
  // Example:
  // ```
  // %source = linalg.fill(%cst, %arg0)
  // scf.for %i
  //   scf.for %j
  //     scf.for %k // not used to index %source!
  //       %ubi = affine.min #map(%i)
  //       %ubj = affine.min #map(%j)
  //       %slice = tensor.extract_slice %source [%i, %j] [%ubi, %ubj]
  //       %padded_slice = linalg.pad_tensor %slice
  // ```
  // After iterating `backwardSlice` we obtain:
  // indexEdges = [%i, %j, %ubi, %ubj]
  // indexingLoops = [scf.for %i, scf.for %j]
  SmallVector<scf::ForOp> indexingLoops;
  for (Operation *op : llvm::reverse(backwardSlice)) {
    // Add the index operands of `padTensorOp` and `sliceOp` to start the
    // exploration of the index computation.
    if (op == padTensorOp || op == sliceOp) {
      addIndexOperandsToIndexEdges(op);
      continue;
    }
    // Add the index operands of the loop if its induction variable is
    // used for index computation. Additionally, insert the loop into
    // `indexingLoops`
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      if (indexEdges.contains(forOp.getInductionVar())) {
        addIndexOperandsToIndexEdges(op);
        indexingLoops.push_back(forOp);
        continue;
      }
    }
    // Add the index operands of all other operations if at least one result is
    // used for index computation.
    if (llvm::any_of(op->getResults(),
                     [&](Value result) { return indexEdges.contains(result); }))
      addIndexOperandsToIndexEdges(op);
  }
  return indexingLoops;
}

SmallVector<Value>
HoistingAnalysis::getPackedTensorSizes(ImplicitLocOpBuilder &b) {
  SmallVector<Value> dynamicTensorSizes;

  // Upper bound the packing loop lengths to size the packed tensor. Taking
  // upper bounds can make the sizes of the packed tensor independent of the
  // enclosing loops. This independence is a prerequisite for reusing the same
  // buffer for all enclosing loop iterations and hoisting its allocation out of
  // the enclosing loops.
  for (auto forOp : packingLoops) {
    // Compute an upper bound `ubVal` for the upper bound of `forOp`.
    AffineMap boundMap;
    SmallVector<Value> boundOperands;
    getUpperBoundForIndex(forOp.upperBound(), boundMap, boundOperands);
    Value ubVal = b.createOrFold<AffineMinOp>(boundMap, boundOperands);
    // Compute the maximal packing loop length as (ub - lb).ceilDiv(step) and
    // store the result to `dynamicTensorSizes`.
    // TODO: instead of using the lower bound of `forOp` directly, implement a
    // lower bound computation similar to the upper bound computation.
    AffineExpr lb, ub, step;
    bindDims(b.getContext(), lb, ub);
    bindSymbols(b.getContext(), step);
    Value res = b.createOrFold<AffineApplyOp>(
        (ub - lb).ceilDiv(step),
        ValueRange{forOp.lowerBound(), ubVal, cast<scf::ForOp>(forOp).step()});
    dynamicTensorSizes.push_back(res);
  }

  return dynamicTensorSizes;
}

static bool isDefinedOutsideOrConstant(scf::ForOp outer, Value v) {
  return outer.isDefinedOutsideOfLoop(v) || v.getDefiningOp<ConstantOp>();
}

/// Return the current iteration number in the loop (iv - lb).ceilDiv(step).
/// The returned Value is guaranteed not to depend on any loop comprised in
/// [`outer`, `forOp`].
/// Return null if such a loop-independent quantity cannot be computed.
static Value buildLoopIterationCount(OpBuilder &b, scf::ForOp outer,
                                     scf::ForOp forOp) {
  MLIRContext *ctx = forOp->getContext();
  AffineExpr iv, lb, step;
  bindDims(ctx, iv, lb);
  bindSymbols(ctx, step);
  if (!isDefinedOutsideOrConstant(outer, forOp.lowerBound()) ||
      !isDefinedOutsideOrConstant(outer, forOp.step()))
    return Value();
  Value ivVal = forOp.getInductionVar(), lbVal = forOp.lowerBound(),
        stepVal = forOp.step();
  auto loc = forOp->getLoc();
  return b.createOrFold<AffineApplyOp>(loc, (iv - lb).ceilDiv(step),
                                       ValueRange{ivVal, lbVal, stepVal});
}

FailureOr<Value> mlir::linalg::hoistPaddingOnTensors(PadTensorOp opToHoist,
                                                     int numLoops,
                                                     PadTensorOp &hoistedOp) {
  LLVM_DEBUG(DBGS() << "Try to hoist " << *(opToHoist) << " by " << numLoops
                    << " loops\n");
  HoistingAnalysis analysis(opToHoist, numLoops);
  if (!analysis.isValid()) {
    LLVM_DEBUG(DBGS() << "Analysis failed -> Skip\n");
    return failure();
  }

  scf::ForOp outer = analysis.outermostEnclosingForOp;
  ImplicitLocOpBuilder b(outer->getLoc(), outer);

  SmallVector<Value> dynamicTensorSizes = analysis.getPackedTensorSizes(b);

  // Update actual number of loops, which may be smaller.
  int nPackedLoops = analysis.packingLoops.size();

  Location loc = opToHoist->getLoc();
  RankedTensorType paddedTensorType = opToHoist.getResultType();
  int paddedRank = paddedTensorType.getRank();

  // Create the packed tensor<?x?x..?xpadded_shape> into which we amortize
  // padding.
  SmallVector<int64_t> packedShape(nPackedLoops, ShapedType::kDynamicSize);
  // TODO: go grab dims when necessary, for now PadTensorOp returns a static
  // tensor.
  llvm::append_range(packedShape, paddedTensorType.getShape());
  auto packedTensorType =
      RankedTensorType::get(packedShape, paddedTensorType.getElementType());
  Value packedTensor = b.create<linalg::InitTensorOp>(
      loc, dynamicTensorSizes, packedTensorType.getShape(),
      packedTensorType.getElementType());

  // Clone the operations involved in the backward slice, iteratively stepping
  // into the loops that we encounter.
  // The implementation proceeds in a stack-like fashion:
  //   1. Iteratively clone and step into the loops, pushing the `packedTensor`
  //      deeper in the stack.
  //   2. Create a InsertSliceOp at the top of the stack.
  //   3. Iteratively pop and yield the result of the InsertSliceOp across
  //     the cloned loops.
  SmallVector<Value> clonedLoopIvs, leadingPackedTensorIndexings;
  clonedLoopIvs.reserve(nPackedLoops);
  leadingPackedTensorIndexings.reserve(nPackedLoops);
  BlockAndValueMapping bvm;
  // Insert `opToHoist` into the backwardSlice so we clone it too.
  analysis.backwardSlice.insert(opToHoist);
  // Stack step 1. iteratively clone loops and push `packedTensor`.
  for (Operation *op : analysis.backwardSlice) {
    // Specifically sit out in the extract_slice(packedTensor) case: this is the
    // piece we seek to replace.
    if (auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(op))
      if (bvm.lookupOrDefault(sliceOp.source()) == packedTensor)
        continue;
    auto effects = dyn_cast<MemoryEffectOpInterface>(op);
    bool hasNoEffects = !effects || effects.hasNoEffect();
    if (hasNoEffects &&
        (op->getNumRegions() == 0 || isa<linalg::PadTensorOp>(op))) {
      b.clone(*op, bvm);
      continue;
    }
    // TODO: support more cases as they appear.
    auto forOp = dyn_cast<scf::ForOp>(op);
    assert(forOp && "Expected scf::ForOp when hoisting pad ops");
    // Unused loop, just skip it.
    if (!llvm::is_contained(analysis.packingLoops, forOp))
      continue;

    auto clonedForOp =
        b.create<scf::ForOp>(loc, bvm.lookupOrDefault(forOp.lowerBound()),
                             bvm.lookupOrDefault(forOp.upperBound()),
                             bvm.lookupOrDefault(forOp.step()), packedTensor);
    // Map the induction var, region args and results to the `clonedForOp`.
    bvm.map(forOp.getInductionVar(), clonedForOp.getInductionVar());
    bvm.map(forOp.getRegionIterArgs(), clonedForOp.getRegionIterArgs());
    bvm.map(forOp.getResults(), clonedForOp.getResults());
    assert(clonedForOp->getNumRegions() == 1);
    clonedLoopIvs.push_back(clonedForOp.getInductionVar());

    b.setInsertionPointToStart(&clonedForOp->getRegion(0).front());
    Value loopIndependentIterationCount =
        buildLoopIterationCount(b, outer, clonedForOp);
    // Assert the loop-independent iteration count can be computed.
    if (!loopIndependentIterationCount)
      llvm_unreachable("loop independence prerequisite not met");
    leadingPackedTensorIndexings.push_back(loopIndependentIterationCount);
    packedTensor = clonedForOp.getRegionIterArgs().front();
  }

  // Stack step 2. create InsertSliceOp at the top of the stack.
  // offsets = [clonedLoopIvs, 0 .. 0].
  SmallVector<OpFoldResult> offsets(leadingPackedTensorIndexings.begin(),
                                    leadingPackedTensorIndexings.end());
  offsets.append(paddedRank, b.getIndexAttr(0));
  // sizes = [1 .. 1, paddedShape].
  SmallVector<OpFoldResult> sizes(nPackedLoops, b.getIndexAttr(1));
  for (int64_t sz : paddedTensorType.getShape()) {
    // TODO: go grab dims when necessary, for now PadTensorOp returns a static
    // tensor.
    assert(!ShapedType::isDynamic(sz) && "padded tensor needs static sizes");
    sizes.push_back(b.getIndexAttr(sz));
  }
  // strides = [1 .. 1].
  SmallVector<OpFoldResult> strides(nPackedLoops + paddedRank,
                                    b.getIndexAttr(1));

  Value inserted =
      b.create<tensor::InsertSliceOp>(loc, bvm.lookup(opToHoist.result()),
                                      packedTensor, offsets, sizes, strides);

  // Stack step 3. iteratively pop the stack and propagate the yield.
  Value valueToYield = inserted;
  for (Value iv : llvm::reverse(clonedLoopIvs)) {
    auto forOp = scf::getForInductionVarOwner(iv);
    b.setInsertionPointToEnd(&forOp.getRegion().front());
    b.create<scf::YieldOp>(loc, valueToYield);
    valueToYield = forOp.getResult(0);
  }

  // Now the packed tensor is ready, replace the original padding op by a
  // 1x..x1 slice [originalLoopIvs, 0 .. 0][1 .. 1, paddedShape][1 .. 1].
  b.setInsertionPoint(opToHoist);
  SmallVector<Value> loopIterationCounts = llvm::to_vector<4>(
      llvm::map_range(analysis.packingLoops, [&](Operation *loop) {
        return buildLoopIterationCount(b, outer, cast<scf::ForOp>(loop));
      }));
  // Assert all loop iteration counts can be computed.
  if (llvm::any_of(loopIterationCounts, [](Value v) { return !v; }))
    llvm_unreachable("loop independence prerequisite not met");
  // offsets = [originalLoopIvs, 0 .. 0].
  offsets.assign(loopIterationCounts.begin(), loopIterationCounts.end());
  offsets.append(paddedRank, b.getIndexAttr(0));
  // sizes = [1 .. 1, paddedShape] (definedabove).
  // strides = [1 .. 1] (defined above)
  packedTensor =
      scf::getForInductionVarOwner(clonedLoopIvs.front())->getResult(0);
  Value newResult = b.create<tensor::ExtractSliceOp>(
      loc, opToHoist.getResultType(), packedTensor, offsets, sizes, strides);

  // Make the newly cloned `opToHoist` available to the caller.
  hoistedOp = cast<PadTensorOp>(bvm.lookup(opToHoist.result()).getDefiningOp());
  return newResult;
}
