//===- IntegerRangeAnalysis.cpp - Integer range analysis --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the dataflow analysis class for integer range inference
// which is used in transformations over the `arith` dialect such as
// branch elimination or signed->unsigned rewriting
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include <cassert>
#include <optional>
#include <utility>

#define DEBUG_TYPE "int-range-analysis"

using namespace mlir;
using namespace mlir::dataflow;

namespace mlir::dataflow {
LogicalResult staticallyNonNegative(DataFlowSolver &solver, Value v) {
  auto *result = solver.lookupState<IntegerValueRangeLattice>(v);
  if (!result || result->getValue().isUninitialized())
    return failure();
  const ConstantIntRanges &range = result->getValue().getValue();
  return success(range.smin().isNonNegative());
}

LogicalResult staticallyNonNegative(DataFlowSolver &solver, Operation *op) {
  auto nonNegativePred = [&solver](Value v) -> bool {
    return succeeded(staticallyNonNegative(solver, v));
  };
  return success(llvm::all_of(op->getOperands(), nonNegativePred) &&
                 llvm::all_of(op->getResults(), nonNegativePred));
}
} // namespace mlir::dataflow

/// Number of merge-site joins a single integer-range lattice element is
/// allowed to absorb before `IntegerValueRangeLattice::join` forces it to
/// its max as a sound over-approximation.
///
/// Trade-off: high enough that realistic loops with dynamic bounds (which
/// typically converge to a tight range in a small number of merge
/// iterations) are not widened prematurely; low enough that the +1
/// ratchet pathology this widening exists to cut off (loop-carried ranges
/// growing by one per worklist visit) terminates after at most this many
/// extra solver iterations rather than ~2^31.
static constexpr unsigned kIntegerRangeWideningBudget = 128;

IntegerValueRange IntegerValueRangeLattice::getMaxRange() const {
  return IntegerValueRange::getMaxRange(
      cast<Value>(getAnchor()),
      indexBitwidth.value_or(IndexType::kInternalStorageBitWidth));
}

ChangeResult IntegerValueRangeLattice::join(const IntegerValueRange &rhs) {
  if (rhs.isUninitialized())
    return ChangeResult::NoChange;
  Type type = getElementTypeOrSelf(cast<Value>(getAnchor()).getType());
  if (type.isIndex() &&
      (!indexBitwidth ||
       rhs.getValue().umin().getBitWidth() != *indexBitwidth)) {
    if (auto constant = rhs.getValue().getConstantValue();
        constant && constant->isZero())
      return Lattice::join(IntegerValueRange(ConstantIntRanges::constant(APInt(
          indexBitwidth.value_or(IndexType::kInternalStorageBitWidth), 0))));
    return Lattice::join(getMaxRange());
  }
  return Lattice::join(rhs);
}

ChangeResult IntegerValueRangeLattice::join(const AbstractSparseLattice &rhs) {
  ChangeResult changed =
      join(static_cast<const IntegerValueRangeLattice &>(rhs).getValue());
  if (mergeChangeCount >= kIntegerRangeWideningBudget)
    return changed | join(getMaxRange());
  if (changed == ChangeResult::Change)
    ++mergeChangeCount;
  return changed;
}

IntegerValueRangeLattice *IntegerRangeAnalysis::getLatticeElement(Value value) {
  auto *lattice = SparseForwardDataFlowAnalysis::getLatticeElement(value);
  if (getElementTypeOrSelf(value.getType()).isIndex()) {
    Operation *scope = value.getDefiningOp();
    if (!scope)
      scope = cast<BlockArgument>(value).getOwner()->getParentOp();
    lattice->setIndexBitwidth(getWidth(scope));
  }
  return lattice;
}

LogicalResult IntegerRangeAnalysis::visitOperation(
    Operation *op, ArrayRef<const IntegerValueRangeLattice *> operands,
    ArrayRef<IntegerValueRangeLattice *> results) {
  auto inferrable = dyn_cast<InferIntRangeInterface>(op);
  std::optional<unsigned> indexBitwidth = getWidth(op);
  if (!inferrable) {
    setAllToEntryStates(results);
    return success();
  }

  LDBG() << "Inferring ranges for "
         << OpWithFlags(op, OpPrintingFlags().skipRegions());
  auto argRanges = llvm::map_to_vector(
      operands, [](const IntegerValueRangeLattice *lattice) {
        return lattice->getValue();
      });

  for (auto [operand, range] : llvm::zip_equal(op->getOperands(), argRanges)) {
    if (indexBitwidth && getElementTypeOrSelf(operand.getType()).isIndex() &&
        !range.isUninitialized() &&
        range.getValue().umin().getBitWidth() != *indexBitwidth)
      range = IntegerValueRange::getMaxRange(operand, *indexBitwidth);
  }

  auto joinCallback = [&](Value v, const IntegerValueRange &attrs) {
    auto result = dyn_cast<OpResult>(v);
    if (!result)
      return;
    assert(llvm::is_contained(op->getResults(), result));

    LDBG() << "Inferred range " << attrs;
    IntegerValueRangeLattice *lattice = results[result.getResultNumber()];
    propagateIfChanged(lattice, lattice->join(attrs));
  };

  inferrable.inferResultRangesFromOptional(argRanges, indexBitwidth,
                                           joinCallback);
  return success();
}

void IntegerRangeAnalysis::visitNonControlFlowArguments(
    Operation *op, const RegionSuccessor &successor,
    ValueRange nonSuccessorInputs,
    ArrayRef<IntegerValueRangeLattice *> nonSuccessorInputLattices) {
  assert(nonSuccessorInputs.size() == nonSuccessorInputLattices.size() &&
         "size mismatch");
  std::optional<unsigned> indexBitwidth = getWidth(op);
  if (auto inferrable = dyn_cast<InferIntRangeInterface>(op)) {
    LDBG() << "Inferring ranges for "
           << OpWithFlags(op, OpPrintingFlags().skipRegions());

    auto argRanges = llvm::map_to_vector(op->getOperands(), [&](Value value) {
      IntegerValueRange range =
          getLatticeElementFor(getProgramPointAfter(op), value)->getValue();
      if (indexBitwidth && getElementTypeOrSelf(value.getType()).isIndex() &&
          !range.isUninitialized() &&
          range.getValue().umin().getBitWidth() != *indexBitwidth)
        return IntegerValueRange::getMaxRange(value, *indexBitwidth);
      return range;
    });

    auto joinCallback = [&](Value v, const IntegerValueRange &attrs) {
      auto arg = dyn_cast<BlockArgument>(v);
      if (!arg)
        return;
      if (!llvm::is_contained(successor.getSuccessor()->getArguments(), arg))
        return;

      LDBG() << "Inferred range " << attrs;
      auto it = llvm::find(successor.getSuccessor()->getArguments(), arg);
      unsigned nonSuccessorInputIdx =
          std::distance(successor.getSuccessor()->getArguments().begin(), it);
      IntegerValueRangeLattice *lattice =
          nonSuccessorInputLattices[nonSuccessorInputIdx];
      propagateIfChanged(lattice, lattice->join(attrs));
    };

    inferrable.inferResultRangesFromOptional(argRanges, indexBitwidth,
                                             joinCallback);
    return;
  }

  /// Given a lower bound, upper bound, or step from a LoopLikeInterface return
  /// the lower/upper bound for that result if possible.
  auto getLoopBoundFromFold = [&](OpFoldResult loopBound, Type boundType,
                                  Block *block, bool getUpper) {
    unsigned int width =
        ConstantIntRanges::getBitwidth(boundType, indexBitwidth.value_or(0));
    if (auto attr = dyn_cast<Attribute>(loopBound)) {
      if (auto bound = dyn_cast<IntegerAttr>(attr))
        return bound.getValue().sextOrTrunc(width);
    } else if (auto value = llvm::dyn_cast<Value>(loopBound)) {
      const IntegerValueRangeLattice *lattice =
          getLatticeElementFor(getProgramPointBefore(block), value);
      if (lattice != nullptr && !lattice->getValue().isUninitialized() &&
          lattice->getValue().getValue().umin().getBitWidth() == width)
        return getUpper ? lattice->getValue().getValue().smax()
                        : lattice->getValue().getValue().smin();
    }
    // Given the results of getConstant{Lower,Upper}Bound()
    // or getConstantStep() on a LoopLikeInterface return the lower/upper
    // bound
    return getUpper ? APInt::getSignedMaxValue(width)
                    : APInt::getSignedMinValue(width);
  };

  // Infer bounds for loop arguments that have static bounds
  if (auto loop = dyn_cast<LoopLikeOpInterface>(op)) {
    std::optional<llvm::SmallVector<Value>> maybeIvs =
        loop.getLoopInductionVars();
    if (!maybeIvs) {
      return SparseForwardDataFlowAnalysis ::visitNonControlFlowArguments(
          op, successor, nonSuccessorInputs, nonSuccessorInputLattices);
    }
    // Some loop implementations may return nullopt for non-constant bounds
    // (e.g. affine.for with a dynamic upper bound), even when induction
    // variables exist. Fall back to the generic analysis in that case.
    std::optional<SmallVector<OpFoldResult>> maybeLowerBounds =
        loop.getLoopLowerBounds();
    std::optional<SmallVector<OpFoldResult>> maybeUpperBounds =
        loop.getLoopUpperBounds();
    std::optional<SmallVector<OpFoldResult>> maybeSteps = loop.getLoopSteps();
    if (!maybeLowerBounds || !maybeUpperBounds || !maybeSteps) {
      return SparseForwardDataFlowAnalysis::visitNonControlFlowArguments(
          op, successor, nonSuccessorInputs, nonSuccessorInputLattices);
    }
    SmallVector<OpFoldResult> lowerBounds = *maybeLowerBounds;
    SmallVector<OpFoldResult> upperBounds = *maybeUpperBounds;
    SmallVector<OpFoldResult> steps = *maybeSteps;
    for (auto [iv, lowerBound, upperBound, step] :
         llvm::zip_equal(*maybeIvs, lowerBounds, upperBounds, steps)) {
      if (!ConstantIntRanges::getBitwidth(iv.getType(), indexBitwidth)) {
        setToEntryState(getLatticeElement(iv));
        continue;
      }
      Block *block = iv.getParentBlock();
      APInt min = getLoopBoundFromFold(lowerBound, iv.getType(), block,
                                       /*getUpper=*/false);
      APInt max = getLoopBoundFromFold(upperBound, iv.getType(), block,
                                       /*getUpper=*/true);
      // Assume positivity for uniscoverable steps by way of getUpper = true.
      APInt stepVal =
          getLoopBoundFromFold(step, iv.getType(), block, /*getUpper=*/true);

      if (stepVal.isNegative()) {
        std::swap(min, max);
      } else {
        // Correct the upper bound by subtracting 1 so that it becomes a <=
        // bound, because loops do not generally include their upper bound.
        max -= 1;
      }

      // If we infer the lower bound to be larger than the upper bound, the
      // resulting range is meaningless and should not be used in further
      // inferences.
      if (max.sge(min)) {
        IntegerValueRangeLattice *ivEntry = getLatticeElement(iv);
        auto ivRange = ConstantIntRanges::fromSigned(min, max);
        propagateIfChanged(ivEntry, ivEntry->join(IntegerValueRange{ivRange}));
      }
    }
    return;
  }

  return SparseForwardDataFlowAnalysis::visitNonControlFlowArguments(
      op, successor, nonSuccessorInputs, nonSuccessorInputLattices);
}
