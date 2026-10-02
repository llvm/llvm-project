//===- IntegerDivisibilityAnalysis.cpp - Integer divisibility ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the dataflow analysis class for integer divisibility
// inference. Operations participate in the analysis by implementing
// `InferIntDivisibilityOpInterface`.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/IntegerDivisibilityAnalysis.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "int-divisibility-analysis"

using llvm::dbgs;

namespace mlir::dataflow {

void IntegerDivisibilityAnalysis::setToEntryState(
    IntegerDivisibilityLattice *lattice) {
  propagateIfChanged(lattice,
                     lattice->join(IntegerDivisibility::getMinDivisibility()));
}

LogicalResult IntegerDivisibilityAnalysis::visitOperation(
    Operation *op, ArrayRef<const IntegerDivisibilityLattice *> operands,
    ArrayRef<IntegerDivisibilityLattice *> results) {
  auto inferrable = dyn_cast<InferIntDivisibilityOpInterface>(op);
  if (!inferrable) {
    setAllToEntryStates(results);
    return success();
  }

  LLVM_DEBUG(dbgs() << "Inferring divisibility for " << *op << "\n");
  auto argDivs = llvm::map_to_vector(
      operands, [](const IntegerDivisibilityLattice *lattice) {
        return lattice->getValue();
      });
  auto joinCallback = [&](Value v, const IntegerDivisibility &newDiv) {
    auto result = dyn_cast<OpResult>(v);
    if (!result) {
      return;
    }
    assert(llvm::is_contained(op->getResults(), result));

    LLVM_DEBUG(dbgs() << "Inferred divisibility " << newDiv << "\n");
    IntegerDivisibilityLattice *lattice = results[result.getResultNumber()];
    IntegerDivisibility oldDiv = lattice->getValue();

    ChangeResult changed = lattice->join(newDiv);

    // Catch loop results with loop-variant divisibility and conservatively
    // set them to divisibility 1 (no information) so we don't ratchet
    // indefinitely (the dataflow analysis in MLIR doesn't attempt to work
    // out trip counts and often can't).
    bool isYieldedResult = llvm::any_of(v.getUsers(), [](Operation *op) {
      return op->hasTrait<OpTrait::IsTerminator>();
    });
    if (isYieldedResult && !oldDiv.isUninitialized() &&
        !(lattice->getValue() == oldDiv)) {
      LLVM_DEBUG(llvm::dbgs() << "Loop variant loop result detected\n");
      changed |= lattice->join(IntegerDivisibility::getMinDivisibility());
    }
    propagateIfChanged(lattice, changed);
  };

  inferrable.inferResultDivisibility(argDivs, joinCallback);
  return success();
}

void IntegerDivisibilityAnalysis::visitNonControlFlowArguments(
    Operation *op, const RegionSuccessor &successor, ValueRange successorInputs,
    ArrayRef<IntegerDivisibilityLattice *> argLattices) {
  // Get the constant divisibility, or query the lattice for Values.
  auto getDivFromOfr = [&](std::optional<OpFoldResult> ofr, Block *block,
                           bool isUnsigned) -> uint64_t {
    if (ofr.has_value()) {
      if (auto constBound = getConstantIntValue(*ofr)) {
        return constBound.value();
      }
      auto value = cast<Value>(ofr.value());
      const IntegerDivisibilityLattice *lattice =
          getLatticeElementFor(getProgramPointBefore(block), value);
      if (lattice != nullptr && !lattice->getValue().isUninitialized()) {
        return isUnsigned ? lattice->getValue().getValue().udiv()
                          : lattice->getValue().getValue().sdiv();
      }
    }
    return isUnsigned
               ? IntegerDivisibility::getMinDivisibility().getValue().udiv()
               : IntegerDivisibility::getMinDivisibility().getValue().sdiv();
  };

  // Infer bounds for loop arguments that have static bounds
  if (auto loop = dyn_cast<LoopLikeOpInterface>(op)) {
    std::optional<SmallVector<Value>> ivs = loop.getLoopInductionVars();
    std::optional<SmallVector<OpFoldResult>> lbs = loop.getLoopLowerBounds();
    std::optional<SmallVector<OpFoldResult>> steps = loop.getLoopSteps();
    if (!ivs || !lbs || !steps) {
      return SparseForwardDataFlowAnalysis::visitNonControlFlowArguments(
          op, successor, successorInputs, argLattices);
    }
    for (auto [iv, lb, step] : llvm::zip_equal(*ivs, *lbs, *steps)) {
      IntegerDivisibilityLattice *ivEntry = getLatticeElement(iv);
      Block *block = iv.getParentBlock();
      uint64_t stepUDiv = getDivFromOfr(step, block, /*unsigned=*/true);
      uint64_t stepSDiv = getDivFromOfr(step, block, /*unsigned=*/false);
      uint64_t lbUDiv = getDivFromOfr(lb, block, /*unsigned=*/true);
      uint64_t lbSDiv = getDivFromOfr(lb, block, /*unsigned=*/false);
      ConstantIntDivisibility lbDiv(lbUDiv, lbSDiv);
      ConstantIntDivisibility stepDiv(stepUDiv, stepSDiv);

      // Loop induction variables are computed as `lb + i * step`. The
      // divisibility for `i * step` is just the divisibility of `step`, so
      // the total divisibility is obtained by unioning the step divisibility
      // with the lower bound divisibility, which takes the GCD of the two.
      ConstantIntDivisibility ivDiv = stepDiv.getUnion(lbDiv);
      propagateIfChanged(ivEntry, ivEntry->join(ivDiv));
    }
    return;
  }

  return SparseForwardDataFlowAnalysis::visitNonControlFlowArguments(
      op, successor, successorInputs, argLattices);
}

} // namespace mlir::dataflow
