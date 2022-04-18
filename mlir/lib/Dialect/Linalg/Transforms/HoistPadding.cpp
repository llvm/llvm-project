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

#include "mlir/Dialect/Linalg/Transforms/HoistPadding.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

using llvm::dbgs;

#define DEBUG_TYPE "hoist-padding"

#define DBGS() (dbgs() << '[' << DEBUG_TYPE << "] ")

using namespace mlir;
using namespace mlir::linalg;

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
  SmallVector<Value> getPackedTensorSizes(ImplicitLocOpBuilder &b);

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

private:
  /// Drop any non-index dependencies of `padOp` and `sliceOp` from
  /// `backwardSlice`. The method follows the use-def chains of the index
  /// operands consumed by `padOp` and `sliceOp` and drops the operations
  /// not part of this index computation. Afterwards, the filtered
  /// `backwardSlice` contains only the loops whose induction variable is used,
  /// directly or indirectly, to index the padded tensor. The method returns
  /// failure if the filtered backward slice contains an unexpected operation.
  ///
  /// Example:
  /// ```
  /// %source = linalg.fill(%cst, %arg0)
  /// scf.for %i
  ///   %unrelated = linalg.fill(%cst, %arg1)    // not used to index %source!
  ///   scf.for %j (%arg2 = %unrelated)
  ///     scf.for %k                             // not used to index %source!
  ///       %ubi = affine.min #map(%i)
  ///       %ubj = affine.min #map(%j)
  ///       %slice = tensor.extract_slice %source [%i, %j] [%ubi, %ubj]
  ///       %padded_slice = tensor.pad %slice
  /// ```
  /// dropNonIndexDependencies(%padded_slice, %slice)
  /// removes [scf.for %k, linalg.fill(%cst, %arg1)] from backwardSlice.
  LogicalResult dropNonIndexDependencies(tensor::PadOp padOp,
                                         tensor::ExtractSliceOp sliceOp);

  /// Encodes whether the analysis is valid and hoisting can proceed.
  bool valid;
};

/// Return true if all uses of `padOp` are an input tensor of some
/// LinalgOp.
static bool isOnlyUsedAsInputOfLinalgOp(tensor::PadOp padOp) {
  for (OpOperand &use : padOp.result().getUses()) {
    auto linalgUser = dyn_cast<linalg::LinalgOp>(use.getOwner());
    if (!linalgUser || !linalgUser.isInputTensor(&use)) {
      LLVM_DEBUG(DBGS() << "Found a use of " << *(padOp)
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
getAtMostNEnclosingLoops(tensor::PadOp padOp, int nLevels,
                         SmallVector<scf::ForOp> &reverseEnclosingLoops) {
  AsmState state(padOp->getParentOfType<func::FuncOp>());
  (void)state;
  scf::ForOp outermostEnclosingForOp = nullptr;
  Operation *nextEnclosingOp = padOp->getParentOp();
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

/// Returns the transposed `rankedTensorType` if `transposeVector` is non-empty.
/// Fail if `transposeVector` is no permutation matching the tensor rank.
static FailureOr<RankedTensorType>
computeTransposedType(RankedTensorType rankedTensorType,
                      ArrayRef<int64_t> transposeVector) {
  if (transposeVector.empty())
    return rankedTensorType;
  if (!isPermutation(transposeVector) ||
      transposeVector.size() != static_cast<size_t>(rankedTensorType.getRank()))
    return failure();

  SmallVector<int64_t> transposedShape(rankedTensorType.getShape().begin(),
                                       rankedTensorType.getShape().end());
  applyPermutationToVector(transposedShape, transposeVector);

  using RTTBuilder = RankedTensorType::Builder;
  RankedTensorType transposedTensorType =
      RTTBuilder(rankedTensorType).setShape(transposedShape);
  return transposedTensorType;
}

HoistingAnalysis::HoistingAnalysis(tensor::PadOp padOp, int numLoops) {
  valid = false;

  // Bail on any use that isn't an input of a LinalgOp.
  // Hoisting of inplace updates happens after vectorization.
  if (!isOnlyUsedAsInputOfLinalgOp(padOp))
    return;

  // Get at most `numLoops` of immediately enclosing loops.
  SmallVector<scf::ForOp> reverseEnclosingLoops;
  getAtMostNEnclosingLoops(padOp, numLoops, reverseEnclosingLoops);
  if (reverseEnclosingLoops.empty()) {
    LLVM_DEBUG(DBGS() << "No immediately enclosing loop -> skip\n");
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
  auto sliceOp = padOp.source().getDefiningOp<tensor::ExtractSliceOp>();
  if (!sliceOp) {
    LLVM_DEBUG(DBGS() << "Cannot find the extract slice op -> skip\n");
    return;
  }
  if (!outermostEnclosingForOp.isDefinedOutsideOfLoop(sliceOp.source())) {
    LLVM_DEBUG(DBGS() << "Source not defined outside of loops -> skip\n");
    return;
  }

  // Check the region of `padOp` depends on a constant only. Adding
  // hoisting support for arbitrary padding regions would require cloning all
  // dependencies captured by the padding region.
  Value paddingValue = padOp.getConstantPaddingValue();
  if (!paddingValue ||
      !isa_and_nonnull<arith::ConstantOp>(paddingValue.getDefiningOp())) {
    LLVM_DEBUG(DBGS() << "Cannot find constant padding value -> skip\n");
    return;
  }

  // Get all the ops in the backwards slice starting from `padOp` and that
  // are dominated by the outermost enclosing loop.
  DominanceInfo domInfo(outermostEnclosingForOp);
  getBackwardSlice(padOp.getOperation(), &backwardSlice, [&](Operation *op) {
    return domInfo.dominates(outermostEnclosingForOp, op);
  });
  if (backwardSlice.empty())
    return;
  // Add `padOp` itself to the backward slice.
  backwardSlice.insert(padOp.getOperation());

  // Remove all ops in the backward slice that are not used to index the padded
  // tensor. In particular, keep `padOp`, `sliceOp`, and the loop and
  // affine operations used for the index computation.
  if (failed(dropNonIndexDependencies(padOp, sliceOp)))
    return;

  // Add only the loops part of the filtered `backwardSlice` to the packing
  // loops. All other loops are not used to index the padded data and
  // consequently access the same data in every loop iteration. Adding them to
  // the packing loops would increase the cache footprint of the packed data
  // by storing the same data multiple times.
  for (scf::ForOp forOp : llvm::reverse(reverseEnclosingLoops))
    if (backwardSlice.contains(forOp))
      packingLoops.push_back(forOp);
  if (packingLoops.empty()) {
    LLVM_DEBUG(DBGS() << "Cannot find a packing loop -> skip\n");
    return;
  }

  // The analysis is valid and hoisting can occur.
  valid = true;
}

LogicalResult
HoistingAnalysis::dropNonIndexDependencies(tensor::PadOp padOp,
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
    // Add the index operands of all other operations if at least one result is
    // used for index computation.
    if (hasIndexResult(op)) {
      addIndexOperandsToIndexEdges(op);
      // Check the operands of the remaining operations all have index type.
      if (llvm::any_of(op->getOperandTypes(),
                       [](Type type) { return !type.isIndex(); })) {
        LLVM_DEBUG(DBGS() << "Unsupported op with non index type operands: "
                          << op << " -> skip\n");
        return failure();
      }
      // Check the remaining operations do not have regions or memory effects.
      auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op);
      bool hasMemoryEffect = effectInterface && !effectInterface.hasNoEffect();
      if (hasMemoryEffect || op->getNumRegions() != 0) {
        LLVM_DEBUG(DBGS() << "Unsupported op with region or memory effect: "
                          << op << " -> skip\n");
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
    getUpperBoundForIndex(forOp.getUpperBound(), boundMap, boundOperands);
    Value ubVal = b.createOrFold<AffineMinOp>(boundMap, boundOperands);
    // Compute the maximal packing loop length as (ub - lb).ceilDiv(step) and
    // store the result to `dynamicTensorSizes`.
    // TODO: instead of using the lower bound of `forOp` directly, implement a
    // lower bound computation similar to the upper bound computation.
    AffineExpr lb, ub, step;
    bindDims(b.getContext(), lb, ub);
    bindSymbols(b.getContext(), step);
    Value res = b.createOrFold<AffineApplyOp>(
        (ub - lb).ceilDiv(step), ValueRange{forOp.getLowerBound(), ubVal,
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
static Value buildLoopIterationCount(OpBuilder &b, scf::ForOp outer,
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
  return b.createOrFold<AffineApplyOp>(loc, (iv - lb).ceilDiv(step),
                                       ValueRange{ivVal, lbVal, stepVal});
}

FailureOr<Value> mlir::linalg::hoistPaddingOnTensors(
    tensor::PadOp opToHoist, int numLoops, ArrayRef<int64_t> transposeVector,
    tensor::PadOp &hoistedOp, SmallVectorImpl<GenericOp> &transposeOps) {
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

  // Compute the type of the transposed padded tensor.
  FailureOr<RankedTensorType> transposedTensorType =
      computeTransposedType(paddedTensorType, transposeVector);
  if (failed(transposedTensorType))
    return failure();

  // Create the packed tensor<?x?x..?xtransposedShape> into which we amortize
  // padding.
  SmallVector<int64_t> packedShape(nPackedLoops, ShapedType::kDynamicSize);
  // TODO: go grab dims when necessary, for now tensor::PadOp returns a static
  // tensor.
  llvm::append_range(packedShape, transposedTensorType->getShape());
  auto packedTensorType = RankedTensorType::get(
      packedShape, transposedTensorType->getElementType());
  Value packedTensor = b.create<linalg::InitTensorOp>(
      loc, dynamicTensorSizes, packedTensorType.getShape(),
      packedTensorType.getElementType());

  // Clone the operations involved in the backward slice, iteratively stepping
  // into the loops that we encounter.
  // The implementation proceeds in a stack-like fashion:
  //   1. Iteratively clone and step into the loops, pushing the `packedTensor`
  //      deeper in the stack.
  //   2. Create a GenericOp if `transposeVector` is non-empty.
  //   3. Create a InsertSliceOp at the top of the stack.
  //   4. Iteratively pop and yield the result of the InsertSliceOp across
  //      the cloned loops.
  SmallVector<Value> clonedLoopIvs, leadingPackedTensorIndexings;
  clonedLoopIvs.reserve(nPackedLoops);
  leadingPackedTensorIndexings.reserve(nPackedLoops);
  BlockAndValueMapping bvm;
  // Stack step 1. iteratively clone loops and push `packedTensor`.
  for (Operation *op : analysis.backwardSlice) {
    // Specifically sit out in the extract_slice(packedTensor) case: this is the
    // piece we seek to replace.
    if (auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(op))
      if (bvm.lookupOrDefault(sliceOp.source()) == packedTensor)
        continue;
    // Clone all operations except it is a loop.
    auto forOp = dyn_cast<scf::ForOp>(op);
    if (!forOp) {
      b.clone(*op, bvm);
      continue;
    }
    // Create a packing loop that takes `packedTensor` as iteration argument.
    auto clonedForOp = b.create<scf::ForOp>(
        loc, bvm.lookupOrDefault(forOp.getLowerBound()),
        bvm.lookupOrDefault(forOp.getUpperBound()),
        bvm.lookupOrDefault(forOp.getStep()), packedTensor);
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

  // offsets = [clonedLoopIvs, 0 .. 0].
  SmallVector<OpFoldResult> offsets(leadingPackedTensorIndexings.begin(),
                                    leadingPackedTensorIndexings.end());
  offsets.append(paddedRank, b.getIndexAttr(0));
  // sizes = [1 .. 1, transposedShape].
  SmallVector<OpFoldResult> sizes(nPackedLoops, b.getIndexAttr(1));
  for (int64_t sz : transposedTensorType->getShape()) {
    // TODO: go grab dims when necessary, for now tensor::PadOp returns a static
    assert(!ShapedType::isDynamic(sz) && "padded tensor needs static sizes");
    sizes.push_back(b.getIndexAttr(sz));
  }
  // strides = [1 .. 1].
  SmallVector<OpFoldResult> strides(nPackedLoops + paddedRank,
                                    b.getIndexAttr(1));

  // Stack step 2. create GenericOp if `transposeVector` is non-empty.
  Value paddedTensor = bvm.lookup(opToHoist.result());
  if (!transposeVector.empty()) {
    Value outputTensor = b.create<tensor::ExtractSliceOp>(
        loc, *transposedTensorType, packedTensor, offsets, sizes, strides);
    transposeOps.push_back(
        makeTransposeOp(b, loc, paddedTensor, outputTensor, transposeVector));
    paddedTensor = transposeOps.back()->getResult(0);
  }

  // Stack step 3. create InsertSliceOp at the top of the stack.
  Value inserted = b.create<tensor::InsertSliceOp>(
      loc, paddedTensor, packedTensor, offsets, sizes, strides);

  // Stack step 4. iteratively pop the stack and propagate the yield.
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
  // sizes = [1 .. 1, transposedShape] (definedabove).
  // strides = [1 .. 1] (defined above)
  packedTensor =
      scf::getForInductionVarOwner(clonedLoopIvs.front())->getResult(0);
  Value newResult = b.create<tensor::ExtractSliceOp>(
      loc, *transposedTensorType, packedTensor, offsets, sizes, strides);

  // Transpose the packed tensor back to the original storage order.
  if (!transposeVector.empty()) {
    Value initTensor =
        b.create<InitTensorOp>(loc, ValueRange{}, paddedTensorType.getShape(),
                               paddedTensorType.getElementType());
    transposeOps.push_back(
        makeTransposeOp(b, loc, newResult, initTensor, transposeVector));
    newResult = transposeOps.back()->getResult(0);
  }

  // Make the newly cloned `opToHoist` available to the caller.
  hoistedOp =
      cast<tensor::PadOp>(bvm.lookup(opToHoist.result()).getDefiningOp());
  return newResult;
}
