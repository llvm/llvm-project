//===- IntRangeAnalysis.cpp - Infer Ranges Interfaces --*- C++ -*-===//
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

#include "mlir/Analysis/IntRangeAnalysis.h"
#include "mlir/Analysis/DataFlowAnalysis.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "int-range-analysis"

using namespace mlir;

namespace {
/// A wrapper around ConstantIntRanges that provides the lattice functions
/// expected by dataflow analysis.
struct IntRangeLattice {
  IntRangeLattice(const ConstantIntRanges &value) : value(value){};
  IntRangeLattice(ConstantIntRanges &&value) : value(value){};

  bool operator==(const IntRangeLattice &other) const {
    return value == other.value;
  }

  /// wrapper around rangeUnion()
  static IntRangeLattice join(const IntRangeLattice &a,
                              const IntRangeLattice &b) {
    return a.value.rangeUnion(b.value);
  }

  /// Creates a range with bitwidth 0 to represent that we don't know if the
  /// value being marked overdefined is even an integer.
  static IntRangeLattice getPessimisticValueState(MLIRContext *context) {
    APInt noIntValue = APInt::getZeroWidth();
    return ConstantIntRanges(noIntValue, noIntValue, noIntValue, noIntValue);
  }

  /// Create a maximal range ([0, uint_max(t)] / [int_min(t), int_max(t)])
  /// range that is used to mark the value v as unable to be analyzed further,
  /// where t is the type of v.
  static IntRangeLattice getPessimisticValueState(Value v) {
    unsigned int width = ConstantIntRanges::getStorageBitwidth(v.getType());
    APInt umin = APInt::getMinValue(width);
    APInt umax = APInt::getMaxValue(width);
    APInt smin = width != 0 ? APInt::getSignedMinValue(width) : umin;
    APInt smax = width != 0 ? APInt::getSignedMaxValue(width) : umax;
    return ConstantIntRanges{umin, umax, smin, smax};
  }

  ConstantIntRanges value;
};
} // end anonymous namespace

namespace mlir {
namespace detail {
class IntRangeAnalysisImpl : public ForwardDataFlowAnalysis<IntRangeLattice> {
  using ForwardDataFlowAnalysis<IntRangeLattice>::ForwardDataFlowAnalysis;

public:
  /// Define bounds on the results or block arguments of the operation
  /// based on the bounds on the arguments given in `operands`
  ChangeResult
  visitOperation(Operation *op,
                 ArrayRef<LatticeElement<IntRangeLattice> *> operands) final;

  /// Skip regions of branch ops when we can statically infer constant
  /// values for operands to the branch op and said op tells us it's safe to do
  /// so.
  LogicalResult
  getSuccessorsForOperands(BranchOpInterface branch,
                           ArrayRef<LatticeElement<IntRangeLattice> *> operands,
                           SmallVectorImpl<Block *> &successors) final;

  /// Skip regions of branch or loop ops when we can statically infer constant
  /// values for operands to the branch op and said op tells us it's safe to do
  /// so.
  void
  getSuccessorsForOperands(RegionBranchOpInterface branch,
                           Optional<unsigned> sourceIndex,
                           ArrayRef<LatticeElement<IntRangeLattice> *> operands,
                           SmallVectorImpl<RegionSuccessor> &successors) final;

  /// Call the InferIntRangeInterface implementation for region-using ops
  /// that implement it, and infer the bounds of loop induction variables
  /// for ops that implement LoopLikeOPInterface.
  ChangeResult visitNonControlFlowArguments(
      Operation *op, const RegionSuccessor &region,
      ArrayRef<LatticeElement<IntRangeLattice> *> operands) final;
};
} // end namespace detail
} // end namespace mlir

/// Given the results of getConstant{Lower,Upper}Bound()
/// or getConstantStep() on a LoopLikeInterface return the lower/upper bound for
/// that result if possible.
static APInt getLoopBoundFromFold(Optional<OpFoldResult> loopBound,
                                  Type boundType,
                                  detail::IntRangeAnalysisImpl &analysis,
                                  bool getUpper) {
  unsigned int width = ConstantIntRanges::getStorageBitwidth(boundType);
  if (loopBound) {
    if (loopBound->is<Attribute>()) {
      if (auto bound =
              loopBound->get<Attribute>().dyn_cast_or_null<IntegerAttr>())
        return bound.getValue();
    } else if (loopBound->is<Value>()) {
      LatticeElement<IntRangeLattice> *lattice =
          analysis.lookupLatticeElement(loopBound->get<Value>());
      if (lattice != nullptr)
        return getUpper ? lattice->getValue().value.smax()
                        : lattice->getValue().value.smin();
    }
  }
  return getUpper ? APInt::getSignedMaxValue(width)
                  : APInt::getSignedMinValue(width);
}

ChangeResult detail::IntRangeAnalysisImpl::visitOperation(
    Operation *op, ArrayRef<LatticeElement<IntRangeLattice> *> operands) {
  ChangeResult result = ChangeResult::NoChange;
  // Ignore non-integer outputs - return early if the op has no scalar
  // integer results
  bool hasIntegerResult = false;
  for (Value v : op->getResults()) {
    if (v.getType().isIntOrIndex())
      hasIntegerResult = true;
    else
      result |= markAllPessimisticFixpoint(v);
  }
  if (!hasIntegerResult)
    return result;

  if (auto inferrable = dyn_cast<InferIntRangeInterface>(op)) {
    LLVM_DEBUG(llvm::dbgs() << "Inferring ranges for ");
    LLVM_DEBUG(inferrable->print(llvm::dbgs()));
    LLVM_DEBUG(llvm::dbgs() << "\n");
    SmallVector<ConstantIntRanges> argRanges(
        llvm::map_range(operands, [](LatticeElement<IntRangeLattice> *val) {
          return val->getValue().value;
        }));

    auto joinCallback = [&](Value v, const ConstantIntRanges &attrs) {
      LLVM_DEBUG(llvm::dbgs() << "Inferred range " << attrs << "\n");
      LatticeElement<IntRangeLattice> &lattice = getLatticeElement(v);
      Optional<IntRangeLattice> oldRange;
      if (!lattice.isUninitialized())
        oldRange = lattice.getValue();
      result |= lattice.join(IntRangeLattice(attrs));

      // Catch loop results with loop variant bounds and conservatively make
      // them [-inf, inf] so we don't circle around infinitely often (because
      // the dataflow analysis in MLIR doesn't attempt to work out trip counts
      // and often can't).
      bool isYieldedResult = llvm::any_of(v.getUsers(), [](Operation *op) {
        return op->hasTrait<OpTrait::IsTerminator>();
      });
      if (isYieldedResult && oldRange && !(lattice.getValue() == *oldRange)) {
        LLVM_DEBUG(llvm::dbgs() << "Loop variant loop result detected\n");
        result |= lattice.markPessimisticFixpoint();
      }
    };

    inferrable.inferResultRanges(argRanges, joinCallback);
    for (Value opResult : op->getResults()) {
      LatticeElement<IntRangeLattice> &lattice = getLatticeElement(opResult);
      // setResultRange() not called, make pessimistic.
      if (lattice.isUninitialized())
        result |= lattice.markPessimisticFixpoint();
    }
  } else if (op->getNumRegions() == 0) {
    // No regions + no result inference method -> unbounded results (ex. memory
    // ops)
    result |= markAllPessimisticFixpoint(op->getResults());
  }
  return result;
}

LogicalResult detail::IntRangeAnalysisImpl::getSuccessorsForOperands(
    BranchOpInterface branch,
    ArrayRef<LatticeElement<IntRangeLattice> *> operands,
    SmallVectorImpl<Block *> &successors) {
  auto toConstantAttr = [&branch](auto enumPair) -> Attribute {
    Optional<APInt> maybeConstValue =
        enumPair.value()->getValue().value.getConstantValue();

    if (maybeConstValue) {
      return IntegerAttr::get(branch->getOperand(enumPair.index()).getType(),
                              *maybeConstValue);
    }
    return {};
  };
  SmallVector<Attribute> inferredConsts(
      llvm::map_range(llvm::enumerate(operands), toConstantAttr));
  if (Block *singleSucc = branch.getSuccessorForOperands(inferredConsts)) {
    successors.push_back(singleSucc);
    return success();
  }
  return failure();
}

void detail::IntRangeAnalysisImpl::getSuccessorsForOperands(
    RegionBranchOpInterface branch, Optional<unsigned> sourceIndex,
    ArrayRef<LatticeElement<IntRangeLattice> *> operands,
    SmallVectorImpl<RegionSuccessor> &successors) {
  // Get a type with which to construct a constant.
  auto getOperandType = [branch, sourceIndex](unsigned index) {
    // The types of all return-like operations are the same.
    if (!sourceIndex)
      return branch->getOperand(index).getType();

    for (Block &block : branch->getRegion(*sourceIndex)) {
      Operation *terminator = block.getTerminator();
      if (getRegionBranchSuccessorOperands(terminator, *sourceIndex))
        return terminator->getOperand(index).getType();
    }
    return Type();
  };

  auto toConstantAttr = [&getOperandType](auto enumPair) -> Attribute {
    if (Optional<APInt> maybeConstValue =
            enumPair.value()->getValue().value.getConstantValue()) {
      return IntegerAttr::get(getOperandType(enumPair.index()),
                              *maybeConstValue);
    }
    return {};
  };
  SmallVector<Attribute> inferredConsts(
      llvm::map_range(llvm::enumerate(operands), toConstantAttr));
  branch.getSuccessorRegions(sourceIndex, inferredConsts, successors);
}

ChangeResult detail::IntRangeAnalysisImpl::visitNonControlFlowArguments(
    Operation *op, const RegionSuccessor &region,
    ArrayRef<LatticeElement<IntRangeLattice> *> operands) {
  if (auto inferrable = dyn_cast<InferIntRangeInterface>(op)) {
    LLVM_DEBUG(llvm::dbgs() << "Inferring ranges for ");
    LLVM_DEBUG(inferrable->print(llvm::dbgs()));
    LLVM_DEBUG(llvm::dbgs() << "\n");
    SmallVector<ConstantIntRanges> argRanges(
        llvm::map_range(operands, [](LatticeElement<IntRangeLattice> *val) {
          return val->getValue().value;
        }));

    ChangeResult result = ChangeResult::NoChange;
    auto joinCallback = [&](Value v, const ConstantIntRanges &attrs) {
      LLVM_DEBUG(llvm::dbgs() << "Inferred range " << attrs << "\n");
      LatticeElement<IntRangeLattice> &lattice = getLatticeElement(v);
      Optional<IntRangeLattice> oldRange;
      if (!lattice.isUninitialized())
        oldRange = lattice.getValue();
      result |= lattice.join(IntRangeLattice(attrs));

      // Catch loop results with loop variant bounds and conservatively make
      // them [-inf, inf] so we don't circle around infinitely often (because
      // the dataflow analysis in MLIR doesn't attempt to work out trip counts
      // and often can't).
      bool isYieldedValue = llvm::any_of(v.getUsers(), [](Operation *op) {
        return op->hasTrait<OpTrait::IsTerminator>();
      });
      if (isYieldedValue && oldRange && !(lattice.getValue() == *oldRange)) {
        LLVM_DEBUG(llvm::dbgs() << "Loop variant loop result detected\n");
        result |= lattice.markPessimisticFixpoint();
      }
    };

    inferrable.inferResultRanges(argRanges, joinCallback);
    for (Value regionArg : region.getSuccessor()->getArguments()) {
      LatticeElement<IntRangeLattice> &lattice = getLatticeElement(regionArg);
      // setResultRange() not called, make pessimistic.
      if (lattice.isUninitialized())
        result |= lattice.markPessimisticFixpoint();
    }

    return result;
  }

  // Infer bounds for loop arguments that have static bounds
  if (auto loop = dyn_cast<LoopLikeOpInterface>(op)) {
    Optional<Value> iv = loop.getSingleInductionVar();
    if (!iv) {
      return ForwardDataFlowAnalysis<
          IntRangeLattice>::visitNonControlFlowArguments(op, region, operands);
    }
    Optional<OpFoldResult> lowerBound = loop.getSingleLowerBound();
    Optional<OpFoldResult> upperBound = loop.getSingleUpperBound();
    Optional<OpFoldResult> step = loop.getSingleStep();
    APInt min = getLoopBoundFromFold(lowerBound, iv->getType(), *this,
                                     /*getUpper=*/false);
    APInt max = getLoopBoundFromFold(upperBound, iv->getType(), *this,
                                     /*getUpper=*/true);
    // Assume positivity for uniscoverable steps by way of getUpper = true.
    APInt stepVal =
        getLoopBoundFromFold(step, iv->getType(), *this, /*getUpper=*/true);

    if (stepVal.isNegative()) {
      std::swap(min, max);
    } else {
      // Correct the upper bound by subtracting 1 so that it becomes a <= bound,
      // because loops do not generally include their upper bound.
      max -= 1;
    }

    LatticeElement<IntRangeLattice> &ivEntry = getLatticeElement(*iv);
    return ivEntry.join(ConstantIntRanges::fromSigned(min, max));
  }
  return ForwardDataFlowAnalysis<IntRangeLattice>::visitNonControlFlowArguments(
      op, region, operands);
}

IntRangeAnalysis::IntRangeAnalysis(Operation *topLevelOperation) {
  impl = std::make_unique<mlir::detail::IntRangeAnalysisImpl>(
      topLevelOperation->getContext());
  impl->run(topLevelOperation);
}

IntRangeAnalysis::~IntRangeAnalysis() = default;
IntRangeAnalysis::IntRangeAnalysis(IntRangeAnalysis &&other) = default;

Optional<ConstantIntRanges> IntRangeAnalysis::getResult(Value v) {
  LatticeElement<IntRangeLattice> *result = impl->lookupLatticeElement(v);
  if (result == nullptr || result->isUninitialized())
    return llvm::None;
  return result->getValue().value;
}
