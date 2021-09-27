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
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
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
///   contains
///      an unknown op with a region.
///   4. The backward slice from the pad op to the scf::ForOp to hoist above is
///   empty.
/// Other cases succeed and will trigger hoisting of the pad op.
struct HoistingAnalysis {
  HoistingAnalysis(PadTensorOp padTensorOp, int nLevels);

  bool isValid() { return valid; }

  /// Footprint of the packedTensor, computed from the packingLoops and
  /// `backwardSlice`.
  FailureOr<SmallVector<Value>> getPackedTensorSizes(ImplicitLocOpBuilder &b);

  /// The padTensorOp that needs to be hoisted.
  PadTensorOp padTensorOp;

  /// The maximum number of immediately enclosing scf::ForOp to hoist over.
  int nLevels;

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
  /// SmallSetVector<scf::ForOp> packingLoops;
  SetVector<scf::ForOp, SmallVector<scf::ForOp>, DenseSet<Operation *>>
      packingLoops;

private:
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

HoistingAnalysis::HoistingAnalysis(PadTensorOp padTensorOp, int nLevels)
    : padTensorOp(padTensorOp), nLevels(nLevels), valid(false) {
  AsmState state(padTensorOp->getParentOfType<mlir::FuncOp>());
  (void)state;

  // Bail on any use that isn't an input of a Linalg op.
  // Hoisting of inplace updates happens after vectorization.
  if (!isOnlyUsedAsInputOfLinalgOp(padTensorOp))
    return;

  // Get at most nLevels of immediately enclosing loops.
  SmallVector<scf::ForOp> reverseEnclosingLoops;
  getAtMostNEnclosingLoops(padTensorOp, nLevels, reverseEnclosingLoops);
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

  // Backward slice is a topologically sorted list of ops starting at
  // `outermostEnclosingForOp`.
  assert(outermostEnclosingForOp == backwardSlice.front());

  // Filter out the loops whose induction variable is not used to compute the
  // padded result. As a first approximation, just look for IVs that have no use
  // in the backwardSlice.
  // These are the dimensions of reuse that we can exploit to reduce the amount
  // of copy / memory.
  for (scf::ForOp forOp : llvm::reverse(reverseEnclosingLoops)) {
    for (Operation *user : forOp.getInductionVar().getUsers()) {
      if (backwardSlice.contains(user)) {
        packingLoops.insert(forOp);
        break;
      }
    }
  }

  // The analysis is valid and hoisting can occur.
  valid = true;
}

static bool isDefinedOutsideOrConstant(scf::ForOp outer, Value v) {
  return outer.isDefinedOutsideOfLoop(v) || v.getDefiningOp<ConstantOp>();
}

/// For each loop in `loops`, determine the ops involved in the construction of
/// its upper bound---up to the outerLimit loop--- and fold them as new
/// inequalities in the constraint set.
/// This is achieved by computing the backwardSlice of the loop's upper bound
/// and iteratively folding each op in reverse topological order to guarantee
/// use-def ordering.
/// As operations are folded in, their result is projected out of the
/// constraints set.
/// The following operations are supported:
///   - scf::ForOp are simply skipped.
///   - AffineApplyOp are composed to replace the result by an equality.
///   - AffineMinOp are composed by adding each entry as an upper bound.
/// If any other operation is met, return failure.
// TODO: extend on a per-need basis.
static LogicalResult
foldUpperBoundsIntoConstraintsSet(FlatAffineValueConstraints &constraints,
                                  scf::ForOp outerLimit,
                                  ArrayRef<scf::ForOp> loops) {
  SetVector<Value> toProjectOut;
  for (scf::ForOp loop : loops) {
    auto ub = loop.upperBound();
    if (isDefinedOutsideOrConstant(outerLimit, ub))
      continue;

    // Compute a backward slice up to, but not including, `outerLimit`.
    SetVector<Operation *> backwardSlice;
    getBackwardSlice(ub, &backwardSlice, [&](Operation *op) {
      return outerLimit->isProperAncestor(op);
    });
    backwardSlice.insert(ub.getDefiningOp());

    // Iterate over all ops in the slice and compose them in the constraints.
    for (Operation *op : llvm::reverse(backwardSlice)) {
      if (!isa<scf::ForOp, AffineApplyOp, AffineMinOp>(op))
        return failure();
      if (isa<scf::ForOp>(op))
        continue;
      // Ensure there is a
      auto ensureIdFailed = [&](Value v) {
        if (constraints.containsId(v)) {
          unsigned pos;
          constraints.findId(v, &pos);
          return pos >= constraints.getNumDimIds();
        }
        constraints.appendDimId(v);
        return false;
      };

      // Ensure all ids exist and add results for later projection.
      if (llvm::any_of(op->getResults(), ensureIdFailed) ||
          llvm::any_of(op->getOperands(), ensureIdFailed))
        return failure();

      // All supported ops have 1 result.
      // TODO: extend when needed.
      toProjectOut.insert(op->getResult(0));

      // Compose supported ops.
      if (auto affineApplyOp = dyn_cast<AffineApplyOp>(op)) {
        AffineValueMap avm(affineApplyOp.getAffineMap(),
                           affineApplyOp.getOperands(),
                           affineApplyOp.getResult());
        if (failed(constraints.composeMap(&avm)))
          return failure();
        continue;
      }
      auto affineMinOp = cast<AffineMinOp>(op);
      unsigned pos;
      bool foundMinOp = constraints.findId(affineMinOp.getResult(), &pos);
      (void)foundMinOp;
      assert(foundMinOp);
      AffineMap alignedMap = constraints.computeAlignedMap(
          affineMinOp.getAffineMap(), affineMinOp.getOperands());
      if (failed(
              constraints.addBound(FlatAffineConstraints::UB, pos, alignedMap)))
        return failure();
    }
  }
  for (Value v : toProjectOut)
    constraints.projectOut(v);
  return success();
}

// Footprint of the packedTensor, computed from the packingLoops and
// `backwardSlice`.
FailureOr<SmallVector<Value>>
HoistingAnalysis::getPackedTensorSizes(ImplicitLocOpBuilder &b) {
  // Create the base affine constaints for the packedLoops.
  auto constraints = FlatAffineValueConstraints::getHyperrectangular(
      llvm::to_vector<8>(llvm::map_range(
          packingLoops, [](scf::ForOp op) { return op.getInductionVar(); })),
      llvm::to_vector<8>(llvm::map_range(
          packingLoops, [](scf::ForOp op) { return op.lowerBound(); })),
      llvm::to_vector<8>(llvm::map_range(
          packingLoops, [](scf::ForOp op) { return op.upperBound(); })));

  // Iteratively try to fold the upper bounds into the constraints set.
  if (failed(foldUpperBoundsIntoConstraintsSet(
          constraints, outermostEnclosingForOp, packingLoops.getArrayRef())))
    return failure();

  int nPackedLoops = packingLoops.size();
  SmallVector<AffineMap> lbs(nPackedLoops), ubs(nPackedLoops);
  // Compute the bounds of the first positions, assuming the others are fixed.
  constraints.getSliceBounds(/*pos=*/0, /*num=*/nPackedLoops,
                             outermostEnclosingForOp->getContext(), &lbs, &ubs);

  SmallVector<Value> allValues;
  constraints.getAllValues(&allValues);
  SmallVector<Value> allNonLoopValues(allValues.begin() + nPackedLoops,
                                      allValues.end());

  // For each packingLoop, create the extent by (ub - lb).ceilDiv(step).
  // IP just before the outermost loop considered that we hoist above.
  assert(nPackedLoops == static_cast<int64_t>(lbs.size()) &&
         "expected matching lb sizes");
  assert(nPackedLoops == static_cast<int64_t>(ubs.size()) &&
         "expected matching ub sizes");
  SmallVector<Value> dynamicTensorSizes;
  for (auto it : llvm::zip(packingLoops, lbs, ubs)) {
    scf::ForOp loop = std::get<0>(it);
    AffineMap lbMap = std::get<1>(it);
    AffineMap ubMap = std::get<2>(it);
    SmallVector<Value> lbOperands(allNonLoopValues);
    canonicalizeMapAndOperands(&lbMap, &lbOperands);
    Value lbVal = b.createOrFold<AffineMaxOp>(lbMap, lbOperands);

    SmallVector<Value> ubOperands(allNonLoopValues);
    canonicalizeMapAndOperands(&ubMap, &ubOperands);
    Value ubVal = b.createOrFold<AffineMinOp>(ubMap, ubOperands);

    AffineExpr lb, ub, step;
    bindDims(b.getContext(), lb, ub);
    bindSymbols(b.getContext(), step);
    Value res = b.createOrFold<AffineApplyOp>(
        (ub - lb).ceilDiv(step),
        ValueRange{lbVal, ubVal, cast<scf::ForOp>(loop).step()});

    dynamicTensorSizes.push_back(res);
  }
  return dynamicTensorSizes;
}

/// Return success if `v` is a value that is only transitively defined by ops of
/// type in `OpTypeList`.
template <typename... OpTypeList>
static bool backwardsSliceOnlyHasOpsOfType(scf::ForOp outerLimit, Value v) {
  // Compute a backward slice up to, but not including, `outerLimit`.
  SetVector<Operation *> backwardSlice;
  getBackwardSlice(v, &backwardSlice, [&](Operation *op) {
    return outerLimit->isProperAncestor(op);
  });
  // Traverse the backward slice and ensure we can perform the computation to
  // hoist.
  for (Operation *op : backwardSlice) {
    if (isa<OpTypeList...>(op))
      continue;
    LLVM_DEBUG(DBGS() << "Abort: unadmissible op in slice " << *op << "\n");
    return false;
  }
  return true;
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

LogicalResult mlir::linalg::hoistPaddingOnTensors(PadTensorOp &padTensorOp,
                                                  int nLoops) {
  LLVM_DEBUG(DBGS() << "Try to hoist " << *(padTensorOp) << " by " << nLoops
                    << " loops\n");
  HoistingAnalysis analysis(padTensorOp, nLoops);
  if (!analysis.isValid()) {
    LLVM_DEBUG(DBGS() << "Analysis failed -> Skip\n");
    return failure();
  }

  scf::ForOp outer = analysis.outermostEnclosingForOp;
  ImplicitLocOpBuilder b(outer->getLoc(), outer);

  auto maybeDynamicTensorSizes = analysis.getPackedTensorSizes(b);
  if (failed(maybeDynamicTensorSizes))
    return failure();
  SmallVector<Value> dynamicTensorSizes = *maybeDynamicTensorSizes;

  // Update actual number of loops, which may be smaller.
  int nPackedLoops = analysis.packingLoops.size();

  Location loc = padTensorOp->getLoc();
  RankedTensorType paddedTensorType = padTensorOp.getResultType();
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
  // Insert `padTensorOp` into the backwardSlice so we clone it too.
  analysis.backwardSlice.insert(padTensorOp);
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
    if (!analysis.packingLoops.contains(forOp))
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
      b.create<tensor::InsertSliceOp>(loc, bvm.lookup(padTensorOp.result()),
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
  b.setInsertionPoint(padTensorOp);
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
  padTensorOp.replaceAllUsesWith(
      b.create<tensor::ExtractSliceOp>(loc, padTensorOp.getResultType(),
                                       packedTensor, offsets, sizes, strides)
          ->getResult(0));

  Operation *toErase = padTensorOp;

  // Make the newly cloned `padTensorOp` available to the caller.
  padTensorOp =
      cast<PadTensorOp>(bvm.lookup(padTensorOp.result()).getDefiningOp());

  toErase->erase();

  return success();
}
