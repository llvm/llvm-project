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
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "int-range-analysis"

using namespace mlir;
using namespace mlir::dataflow;

IntegerValueRange IntegerValueRange::getMaxRange(Value value) {
  unsigned width = ConstantIntRanges::getStorageBitwidth(value.getType());
  if (width == 0)
    return {};
  APInt umin = APInt::getMinValue(width);
  APInt umax = APInt::getMaxValue(width);
  APInt smin = width != 0 ? APInt::getSignedMinValue(width) : umin;
  APInt smax = width != 0 ? APInt::getSignedMaxValue(width) : umax;
  return IntegerValueRange{ConstantIntRanges{umin, umax, smin, smax}};
}

void IntegerValueRangeLattice::onUpdate(DataFlowSolver *solver) const {
  Lattice::onUpdate(solver);

  // If the integer range can be narrowed to a constant, update the constant
  // value of the SSA value.
  std::optional<APInt> constant = getValue().getValue().getConstantValue();
  auto value = point.get<Value>();
  auto *cv = solver->getOrCreateState<Lattice<ConstantValue>>(value);
  if (!constant)
    return solver->propagateIfChanged(
        cv, cv->join(ConstantValue::getUnknownConstant()));

  Dialect *dialect;
  if (auto *parent = value.getDefiningOp())
    dialect = parent->getDialect();
  else
    dialect = value.getParentBlock()->getParentOp()->getDialect();
  solver->propagateIfChanged(
      cv, cv->join(ConstantValue(IntegerAttr::get(value.getType(), *constant),
                                 dialect)));
}

void IntegerRangeAnalysis::visitOperation(
    Operation *op, ArrayRef<const IntegerValueRangeLattice *> operands,
    ArrayRef<IntegerValueRangeLattice *> results) {
  // If the lattice on any operand is unitialized, bail out.
  if (llvm::any_of(operands, [](const IntegerValueRangeLattice *lattice) {
        return lattice->getValue().isUninitialized();
      })) {
    return;
  }

  // Ignore non-integer outputs - return early if the op has no scalar
  // integer results
  bool hasIntegerResult = false;
  for (auto it : llvm::zip(results, op->getResults())) {
    Value value = std::get<1>(it);
    if (value.getType().isIntOrIndex()) {
      hasIntegerResult = true;
    } else {
      IntegerValueRangeLattice *lattice = std::get<0>(it);
      propagateIfChanged(lattice,
                         lattice->join(IntegerValueRange::getMaxRange(value)));
    }
  }
  if (!hasIntegerResult)
    return;

  auto inferrable = dyn_cast<InferIntRangeInterface>(op);
  if (!inferrable)
    return setAllToEntryStates(results);

  LLVM_DEBUG(llvm::dbgs() << "Inferring ranges for " << *op << "\n");
  SmallVector<ConstantIntRanges> argRanges(
      llvm::map_range(operands, [](const IntegerValueRangeLattice *val) {
        return val->getValue().getValue();
      }));

  auto joinCallback = [&](Value v, const ConstantIntRanges &attrs) {
    auto result = v.dyn_cast<OpResult>();
    if (!result)
      return;
    assert(llvm::is_contained(op->getResults(), result));

    LLVM_DEBUG(llvm::dbgs() << "Inferred range " << attrs << "\n");
    IntegerValueRangeLattice *lattice = results[result.getResultNumber()];
    IntegerValueRange oldRange = lattice->getValue();

    ChangeResult changed = lattice->join(IntegerValueRange{attrs});

    // Catch loop results with loop variant bounds and conservatively make
    // them [-inf, inf] so we don't circle around infinitely often (because
    // the dataflow analysis in MLIR doesn't attempt to work out trip counts
    // and often can't).
    bool isYieldedResult = llvm::any_of(v.getUsers(), [](Operation *op) {
      return op->hasTrait<OpTrait::IsTerminator>();
    });
    if (isYieldedResult && !oldRange.isUninitialized() &&
        !(lattice->getValue() == oldRange)) {
      LLVM_DEBUG(llvm::dbgs() << "Loop variant loop result detected\n");
      changed |= lattice->join(IntegerValueRange::getMaxRange(v));
    }
    propagateIfChanged(lattice, changed);
  };

  inferrable.inferResultRanges(argRanges, joinCallback);
}

void IntegerRangeAnalysis::visitNonControlFlowArguments(
    Operation *op, const RegionSuccessor &successor,
    ArrayRef<IntegerValueRangeLattice *> argLattices, unsigned firstIndex) {
  if (auto inferrable = dyn_cast<InferIntRangeInterface>(op)) {
    LLVM_DEBUG(llvm::dbgs() << "Inferring ranges for " << *op << "\n");
    SmallVector<ConstantIntRanges> argRanges(
        llvm::map_range(op->getOperands(), [&](Value value) {
          return getLatticeElementFor(op, value)->getValue().getValue();
        }));

    auto joinCallback = [&](Value v, const ConstantIntRanges &attrs) {
      auto arg = v.dyn_cast<BlockArgument>();
      if (!arg)
        return;
      if (!llvm::is_contained(successor.getSuccessor()->getArguments(), arg))
        return;

      LLVM_DEBUG(llvm::dbgs() << "Inferred range " << attrs << "\n");
      IntegerValueRangeLattice *lattice = argLattices[arg.getArgNumber()];
      IntegerValueRange oldRange = lattice->getValue();

      ChangeResult changed = lattice->join(IntegerValueRange{attrs});

      // Catch loop results with loop variant bounds and conservatively make
      // them [-inf, inf] so we don't circle around infinitely often (because
      // the dataflow analysis in MLIR doesn't attempt to work out trip counts
      // and often can't).
      bool isYieldedValue = llvm::any_of(v.getUsers(), [](Operation *op) {
        return op->hasTrait<OpTrait::IsTerminator>();
      });
      if (isYieldedValue && !oldRange.isUninitialized() &&
          !(lattice->getValue() == oldRange)) {
        LLVM_DEBUG(llvm::dbgs() << "Loop variant loop result detected\n");
        changed |= lattice->join(IntegerValueRange::getMaxRange(v));
      }
      propagateIfChanged(lattice, changed);
    };

    inferrable.inferResultRanges(argRanges, joinCallback);
    return;
  }

  /// Given the results of getConstant{Lower,Upper}Bound() or getConstantStep()
  /// on a LoopLikeInterface return the lower/upper bound for that result if
  /// possible.
  auto getLoopBoundFromFold = [&](std::optional<OpFoldResult> loopBound,
                                  Type boundType, bool getUpper) {
    unsigned int width = ConstantIntRanges::getStorageBitwidth(boundType);
    if (loopBound.has_value()) {
      if (loopBound->is<Attribute>()) {
        if (auto bound =
                loopBound->get<Attribute>().dyn_cast_or_null<IntegerAttr>())
          return bound.getValue();
      } else if (auto value = loopBound->dyn_cast<Value>()) {
        const IntegerValueRangeLattice *lattice =
            getLatticeElementFor(op, value);
        if (lattice != nullptr)
          return getUpper ? lattice->getValue().getValue().smax()
                          : lattice->getValue().getValue().smin();
      }
    }
    // Given the results of getConstant{Lower,Upper}Bound()
    // or getConstantStep() on a LoopLikeInterface return the lower/upper
    // bound
    return getUpper ? APInt::getSignedMaxValue(width)
                    : APInt::getSignedMinValue(width);
  };

  // Infer bounds for loop arguments that have static bounds
  if (auto loop = dyn_cast<LoopLikeOpInterface>(op)) {
    std::optional<Value> iv = loop.getSingleInductionVar();
    if (!iv) {
      return SparseDataFlowAnalysis ::visitNonControlFlowArguments(
          op, successor, argLattices, firstIndex);
    }
    std::optional<OpFoldResult> lowerBound = loop.getSingleLowerBound();
    std::optional<OpFoldResult> upperBound = loop.getSingleUpperBound();
    std::optional<OpFoldResult> step = loop.getSingleStep();
    APInt min = getLoopBoundFromFold(lowerBound, iv->getType(),
                                     /*getUpper=*/false);
    APInt max = getLoopBoundFromFold(upperBound, iv->getType(),
                                     /*getUpper=*/true);
    // Assume positivity for uniscoverable steps by way of getUpper = true.
    APInt stepVal =
        getLoopBoundFromFold(step, iv->getType(), /*getUpper=*/true);

    if (stepVal.isNegative()) {
      std::swap(min, max);
    } else {
      // Correct the upper bound by subtracting 1 so that it becomes a <=
      // bound, because loops do not generally include their upper bound.
      max -= 1;
    }

    IntegerValueRangeLattice *ivEntry = getLatticeElement(*iv);
    auto ivRange = ConstantIntRanges::fromSigned(min, max);
    propagateIfChanged(ivEntry, ivEntry->join(IntegerValueRange{ivRange}));
    return;
  }

  return SparseDataFlowAnalysis::visitNonControlFlowArguments(
      op, successor, argLattices, firstIndex);
}
