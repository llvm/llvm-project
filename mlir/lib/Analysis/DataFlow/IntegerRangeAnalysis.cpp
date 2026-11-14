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
#include "mlir/Dialect/Arith/IR/ArithAttributes.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
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

ChangeResult IntegerValueRangeLattice::join(const AbstractSparseLattice &rhs) {
  ChangeResult changed = Lattice::join(rhs);
  if (mergeChangeCount >= kIntegerRangeWideningBudget) {
    return changed | Lattice::join(IntegerValueRange::getMaxRange(
                         cast<Value>(getAnchor())));
  }
  if (changed == ChangeResult::Change)
    ++mergeChangeCount;
  return changed;
}

/// Return a range bounding the values in `lhs` that satisfy `pred` wrt. to the
/// constant `rhs`. Return std::nullopt if a contradiction is detected. The
/// result may include values excluded by `ne` because a range cannot represent
/// arbitrary holes.
/// The operands must have the same bitwidth.
static std::optional<ConstantIntRanges>
refineWithComparison(arith::CmpIPredicate pred, const ConstantIntRanges &lhs,
                     const APInt &rhs) {
  assert(rhs.getBitWidth() != 0 &&
         lhs.umin().getBitWidth() == rhs.getBitWidth() &&
         "expected matching, non-zero bitwidths");
  APInt umin = lhs.umin(), umax = lhs.umax();
  APInt smin = lhs.smin(), smax = lhs.smax();
  if (umin.ugt(umax) || smin.sgt(smax))
    return std::nullopt;

  auto maxU = [](const APInt &a, const APInt &b) { return a.ugt(b) ? a : b; };
  auto minU = [](const APInt &a, const APInt &b) { return a.ult(b) ? a : b; };
  auto maxS = [](const APInt &a, const APInt &b) { return a.sgt(b) ? a : b; };
  auto minS = [](const APInt &a, const APInt &b) { return a.slt(b) ? a : b; };

  switch (pred) {
  case arith::CmpIPredicate::eq:
    if (rhs.ult(umin) || rhs.ugt(umax) || rhs.slt(smin) || rhs.sgt(smax))
      return std::nullopt;
    return ConstantIntRanges::constant(rhs);
  case arith::CmpIPredicate::ne:
    if ((umin == rhs && umax == rhs) || (smin == rhs && smax == rhs))
      return std::nullopt;
    // Excluding an endpoint can tighten either interpretation.
    if (umin == rhs)
      ++umin;
    if (umax == rhs)
      --umax;
    if (smin == rhs)
      ++smin;
    if (smax == rhs)
      --smax;
    break;
  case arith::CmpIPredicate::slt:
    if (rhs.isMinSignedValue())
      return std::nullopt;
    smax = minS(smax, rhs - 1);
    break;
  case arith::CmpIPredicate::sle:
    smax = minS(smax, rhs);
    break;
  case arith::CmpIPredicate::sgt:
    if (rhs.isMaxSignedValue())
      return std::nullopt;
    smin = maxS(smin, rhs + 1);
    break;
  case arith::CmpIPredicate::sge:
    smin = maxS(smin, rhs);
    break;
  case arith::CmpIPredicate::ult:
    if (rhs.isZero())
      return std::nullopt;
    umax = minU(umax, rhs - 1);
    break;
  case arith::CmpIPredicate::ule:
    umax = minU(umax, rhs);
    break;
  case arith::CmpIPredicate::ugt:
    if (rhs.isAllOnes())
      return std::nullopt;
    umin = maxU(umin, rhs + 1);
    break;
  case arith::CmpIPredicate::uge:
    umin = maxU(umin, rhs);
    break;
  }

  if (umin.ugt(umax) || smin.sgt(smax))
    return std::nullopt;

  // Derive bounds for the other interpretation and intersect the two ranges.
  ConstantIntRanges unsignedRange = ConstantIntRanges::fromUnsigned(umin, umax);
  ConstantIntRanges signedRange = ConstantIntRanges::fromSigned(smin, smax);
  ConstantIntRanges result = unsignedRange.intersection(signedRange);
  if (result.umin().ugt(result.umax()) || result.smin().sgt(result.smax()))
    return std::nullopt;
  return result;
}

LogicalResult IntegerRangeAnalysis::visitOperation(
    Operation *op, ArrayRef<const IntegerValueRangeLattice *> operands,
    ArrayRef<IntegerValueRangeLattice *> results) {
  if (auto assume = dyn_cast<ub::AssumeOp>(op)) {
    const IntegerValueRange &inputRange = operands.front()->getValue();
    if (inputRange.isUninitialized())
      return success();

    auto predicate = dyn_cast<arith::AssumeCmpIAttr>(assume.getPredicate());
    if (!predicate) {
      // Even without understanding the predicate, a non-poison result equals
      // the predicated value and therefore preserves its range.
      propagateIfChanged(results.front(), results.front()->join(inputRange));
      return success();
    }

    std::optional<ConstantIntRanges> range = refineWithComparison(
        predicate.getPredicate(), inputRange.getValue(),
        predicate.getRhs().getValue());
    // Contradictory assumptions produce poison, not unreachable control flow.
    // The lattice cannot represent a poison-only value. Use a conservative
    // range instead of publishing inverted bounds or an uninitialized
    // state. Only the result is refined, never the predicated value or other
    // operands.
    IntegerValueRange inferred =
        range ? IntegerValueRange(*range)
              : IntegerValueRange::getMaxRange(assume.getResult());
    propagateIfChanged(results.front(), results.front()->join(inferred));
    return success();
  }

  auto inferrable = dyn_cast<InferIntRangeInterface>(op);
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

  auto joinCallback = [&](Value v, const IntegerValueRange &attrs) {
    auto result = dyn_cast<OpResult>(v);
    if (!result)
      return;
    assert(llvm::is_contained(op->getResults(), result));

    LDBG() << "Inferred range " << attrs;
    IntegerValueRangeLattice *lattice = results[result.getResultNumber()];
    propagateIfChanged(lattice, lattice->join(attrs));
  };

  inferrable.inferResultRangesFromOptional(argRanges, joinCallback);
  return success();
}

void IntegerRangeAnalysis::visitNonControlFlowArguments(
    Operation *op, const RegionSuccessor &successor,
    ValueRange nonSuccessorInputs,
    ArrayRef<IntegerValueRangeLattice *> nonSuccessorInputLattices,
    GetLatticeFn getLattice) {
  assert(nonSuccessorInputs.size() == nonSuccessorInputLattices.size() &&
         "size mismatch");
  if (auto inferrable = dyn_cast<InferIntRangeInterface>(op)) {
    LDBG() << "Inferring ranges for "
           << OpWithFlags(op, OpPrintingFlags().skipRegions());

    auto argRanges = llvm::map_to_vector(op->getOperands(), [&](Value value) {
      return getLattice(value)->getValue();
    });

    auto joinCallback = [&](Value v, const IntegerValueRange &attrs) {
      auto it = llvm::find(nonSuccessorInputs, v);
      if (it == nonSuccessorInputs.end())
        return;

      LDBG() << "Inferred range " << attrs;
      unsigned nonSuccessorInputIdx =
          std::distance(nonSuccessorInputs.begin(), it);
      IntegerValueRangeLattice *lattice =
          nonSuccessorInputLattices[nonSuccessorInputIdx];
      propagateIfChanged(lattice, lattice->join(attrs));
    };

    inferrable.inferResultRangesFromOptional(argRanges, joinCallback);
    return;
  }

  /// Given a lower bound, upper bound, or step from a LoopLikeInterface return
  /// the lower/upper bound for that result if possible.
  auto getLoopBoundFromFold = [&](OpFoldResult loopBound, Type boundType,
                                  bool getUpper) {
    unsigned int width = ConstantIntRanges::getStorageBitwidth(boundType);
    if (auto attr = dyn_cast<Attribute>(loopBound)) {
      if (auto bound = dyn_cast<IntegerAttr>(attr))
        return bound.getValue();
    } else if (auto value = llvm::dyn_cast<Value>(loopBound)) {
      const IntegerValueRangeLattice *lattice = getLattice(value);
      if (lattice != nullptr && !lattice->getValue().isUninitialized())
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
      return SparseForwardDataFlowAnalysis::visitNonControlFlowArguments(
          op, successor, nonSuccessorInputs, nonSuccessorInputLattices,
          getLattice);
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
          op, successor, nonSuccessorInputs, nonSuccessorInputLattices,
          getLattice);
    }
    SmallVector<OpFoldResult> lowerBounds = *maybeLowerBounds;
    SmallVector<OpFoldResult> upperBounds = *maybeUpperBounds;
    SmallVector<OpFoldResult> steps = *maybeSteps;
    for (auto [iv, lowerBound, upperBound, step] :
         llvm::zip_equal(*maybeIvs, lowerBounds, upperBounds, steps)) {
      APInt min = getLoopBoundFromFold(lowerBound, iv.getType(),
                                       /*getUpper=*/false);
      APInt max = getLoopBoundFromFold(upperBound, iv.getType(),
                                       /*getUpper=*/true);
      // Assume positivity for uniscoverable steps by way of getUpper = true.
      APInt stepVal =
          getLoopBoundFromFold(step, iv.getType(), /*getUpper=*/true);

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
      op, successor, nonSuccessorInputs, nonSuccessorInputLattices, getLattice);
}
