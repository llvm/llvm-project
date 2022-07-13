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

#define DEBUG_TYPE "int-range-analysis"

using namespace mlir;
using namespace mlir::dataflow;

IntegerValueRange IntegerValueRange::getPessimisticValueState(Value value) {
  unsigned width = ConstantIntRanges::getStorageBitwidth(value.getType());
  APInt umin = APInt::getMinValue(width);
  APInt umax = APInt::getMaxValue(width);
  APInt smin = width != 0 ? APInt::getSignedMinValue(width) : umin;
  APInt smax = width != 0 ? APInt::getSignedMaxValue(width) : umax;
  return {{umin, umax, smin, smax}};
}

void IntegerValueRangeLattice::onUpdate(DataFlowSolver *solver) const {
  Lattice::onUpdate(solver);

  // If the integer range can be narrowed to a constant, update the constant
  // value of the SSA value.
  Optional<APInt> constant = getValue().getValue().getConstantValue();
  auto value = point.get<Value>();
  auto *cv = solver->getOrCreateState<Lattice<ConstantValue>>(value);
  if (!constant)
    return solver->propagateIfChanged(cv, cv->markPessimisticFixpoint());

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
  // Ignore non-integer outputs - return early if the op has no scalar
  // integer results
  bool hasIntegerResult = false;
  for (auto it : llvm::zip(results, op->getResults())) {
    if (std::get<1>(it).getType().isIntOrIndex()) {
      hasIntegerResult = true;
    } else {
      propagateIfChanged(std::get<0>(it),
                         std::get<0>(it)->markPessimisticFixpoint());
    }
  }
  if (!hasIntegerResult)
    return;

  auto inferrable = dyn_cast<InferIntRangeInterface>(op);
  if (!inferrable)
    return markAllPessimisticFixpoint(results);

  LLVM_DEBUG(llvm::dbgs() << "Inferring ranges for " << *op << "\n");
  SmallVector<ConstantIntRanges> argRanges(
      llvm::map_range(operands, [](const IntegerValueRangeLattice *val) {
        return val->getValue().getValue();
      }));

  auto joinCallback = [&](Value v, const ConstantIntRanges &attrs) {
    auto result = v.dyn_cast<OpResult>();
    if (!result)
      return;
    assert(llvm::find(op->getResults(), result) != op->result_end());

    LLVM_DEBUG(llvm::dbgs() << "Inferred range " << attrs << "\n");
    IntegerValueRangeLattice *lattice = results[result.getResultNumber()];
    Optional<IntegerValueRange> oldRange;
    if (!lattice->isUninitialized())
      oldRange = lattice->getValue();

    ChangeResult changed = lattice->join(attrs);

    // Catch loop results with loop variant bounds and conservatively make
    // them [-inf, inf] so we don't circle around infinitely often (because
    // the dataflow analysis in MLIR doesn't attempt to work out trip counts
    // and often can't).
    bool isYieldedResult = llvm::any_of(v.getUsers(), [](Operation *op) {
      return op->hasTrait<OpTrait::IsTerminator>();
    });
    if (isYieldedResult && oldRange.has_value() &&
        !(lattice->getValue() == *oldRange)) {
      LLVM_DEBUG(llvm::dbgs() << "Loop variant loop result detected\n");
      changed |= lattice->markPessimisticFixpoint();
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
      if (llvm::find(successor.getSuccessor()->getArguments(), arg) ==
          successor.getSuccessor()->args_end())
        return;

      LLVM_DEBUG(llvm::dbgs() << "Inferred range " << attrs << "\n");
      IntegerValueRangeLattice *lattice = argLattices[arg.getArgNumber()];
      Optional<IntegerValueRange> oldRange;
      if (!lattice->isUninitialized())
        oldRange = lattice->getValue();

      ChangeResult changed = lattice->join(attrs);

      // Catch loop results with loop variant bounds and conservatively make
      // them [-inf, inf] so we don't circle around infinitely often (because
      // the dataflow analysis in MLIR doesn't attempt to work out trip counts
      // and often can't).
      bool isYieldedValue = llvm::any_of(v.getUsers(), [](Operation *op) {
        return op->hasTrait<OpTrait::IsTerminator>();
      });
      if (isYieldedValue && oldRange && !(lattice->getValue() == *oldRange)) {
        LLVM_DEBUG(llvm::dbgs() << "Loop variant loop result detected\n");
        changed |= lattice->markPessimisticFixpoint();
      }
      propagateIfChanged(lattice, changed);
    };

    inferrable.inferResultRanges(argRanges, joinCallback);
    return;
  }

  /// Given the results of getConstant{Lower,Upper}Bound() or getConstantStep()
  /// on a LoopLikeInterface return the lower/upper bound for that result if
  /// possible.
  auto getLoopBoundFromFold = [&](Optional<OpFoldResult> loopBound,
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
    Optional<Value> iv = loop.getSingleInductionVar();
    if (!iv) {
      return SparseDataFlowAnalysis ::visitNonControlFlowArguments(
          op, successor, argLattices, firstIndex);
    }
    Optional<OpFoldResult> lowerBound = loop.getSingleLowerBound();
    Optional<OpFoldResult> upperBound = loop.getSingleUpperBound();
    Optional<OpFoldResult> step = loop.getSingleStep();
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
    propagateIfChanged(ivEntry, ivEntry->join(ivRange));
    return;
  }

  return SparseDataFlowAnalysis::visitNonControlFlowArguments(
      op, successor, argLattices, firstIndex);
}
