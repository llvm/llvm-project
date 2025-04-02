//===- SubsetOpInterface.cpp - Tensor Subsets -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/SubsetOpInterface.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/IR/Matchers.h"

#include "llvm/ADT/APSInt.h"

#include "mlir/Interfaces/SubsetOpInterface.cpp.inc"

using namespace mlir;

OpOperand &detail::defaultGetDestinationOperand(Operation *op) {
  auto dstOp = dyn_cast<DestinationStyleOpInterface>(op);
  assert(dstOp && "getDestination must be implemented for non-DPS ops");
  assert(
      dstOp.getNumDpsInits() == 1 &&
      "getDestination must be implemented for ops with 0 or more than 1 init");
  return *dstOp.getDpsInitOperand(0);
}

OpResult detail::defaultGetUpdatedDestination(Operation *op) {
  auto dstOp = dyn_cast<DestinationStyleOpInterface>(op);
  assert(dstOp && "getUpdatedDestination must be implemented for non-DPS ops");
  auto insertionOp = cast<SubsetInsertionOpInterface>(op);
  return dstOp.getTiedOpResult(&insertionOp.getDestinationOperand());
}

// === Copied from DialectUtils ===
/// If ofr is a constant integer or an IntegerAttr, return the integer.
static std::optional<int64_t> getConstantIntValue(OpFoldResult ofr) {
  // Case 1: Check for Constant integer.
  if (auto val = llvm::dyn_cast_if_present<Value>(ofr)) {
    APSInt intVal;
    if (matchPattern(val, m_ConstantInt(&intVal)))
      return intVal.getSExtValue();
    return std::nullopt;
  }
  // Case 2: Check for IntegerAttr.
  Attribute attr = llvm::dyn_cast_if_present<Attribute>(ofr);
  if (auto intAttr = dyn_cast_or_null<IntegerAttr>(attr))
    return intAttr.getValue().getSExtValue();
  return std::nullopt;
}

static bool isConstantIntValue(OpFoldResult ofr, int64_t value) {
  auto val = getConstantIntValue(ofr);
  return val && *val == value;
}

static bool areAllConstantIntValue(ArrayRef<OpFoldResult> ofrs, int64_t value) {
  return llvm::all_of(
      ofrs, [&](OpFoldResult ofr) { return isConstantIntValue(ofr, value); });
}
// === End Copied from DialectUtils ===

bool detail::defaultIsEquivalentSubset(
    Operation *op, Value candidate,
    function_ref<bool(Value, Value)> equivalenceFn) {
  assert(isa<SubsetInsertionOpInterface>(op) &&
         "expected SubsetInsertionOpInterface");
  auto subsetOp = cast<SubsetOpInterface>(op);

  // Check if the insertion subset matches the candidate directly.
  FailureOr<HyperrectangularSlice> slice = subsetOp.getAccessedHyperrectangularSlice();
  if (succeeded(slice)) {
    bool allStridesOne =
      areAllConstantIntValue(slice->getMixedStrides(), 1);
    bool allOffsetsZero =
      areAllConstantIntValue(slice->getMixedOffsets(), 0);
    if (equivalenceFn(subsetOp.getTensorContainer(), candidate) && allOffsetsZero && allStridesOne) {
      bool isEquivalentSlice = true;
      auto candidateTensorType = dyn_cast<RankedTensorType>(candidate.getType());
      assert(slice->getMixedSizes().size() == candidateTensorType.getRank() && "rank mismatch");
      for (int64_t i = 0, e = candidateTensorType.getRank(); i < e; ++i) {
        ValueBoundsConstraintSet::Variable var1(candidate, i);
        ValueBoundsConstraintSet::Variable var2(slice->getMixedSizes()[i]);
        if (!ValueBoundsConstraintSet::compare(var1, ValueBoundsConstraintSet::ComparisonOperator::EQ, var2)) {
          isEquivalentSlice = false;
          break;
        }
      }
      if (isEquivalentSlice)
        return true;
    }
  }

  if (!candidate.getDefiningOp<SubsetExtractionOpInterface>())
    return false;
  return subsetOp.operatesOnEquivalentSubset(
      candidate.getDefiningOp<SubsetOpInterface>(), equivalenceFn);
}

bool detail::defaultOperatesOnEquivalentSubset(
    Operation *op, SubsetOpInterface candidate,
    function_ref<bool(Value, Value)> equivalenceFn) {
  auto subsetOp = cast<SubsetOpInterface>(op);
  FailureOr<HyperrectangularSlice> slice =
      subsetOp.getAccessedHyperrectangularSlice();
  assert(succeeded(slice) &&
         "operatesOnEquivalentSubset must be implemented if "
         "getAccessedHyperrectangularSlice is not implemented");
  FailureOr<HyperrectangularSlice> otherSlice =
      candidate.getAccessedHyperrectangularSlice();
  if (failed(otherSlice))
    return false;
  if (!equivalenceFn(subsetOp.getTensorContainer(),
                     candidate.getTensorContainer()))
    return false;
  FailureOr<bool> equivalent = ValueBoundsConstraintSet::areEquivalentSlices(
      op->getContext(), *slice, *otherSlice);
  return succeeded(equivalent) && *equivalent;
}

bool detail::defaultOperatesOnDisjointSubset(
    Operation *op, SubsetOpInterface candidate,
    function_ref<bool(Value, Value)> equivalenceFn) {
  auto subsetOp = cast<SubsetOpInterface>(op);
  FailureOr<HyperrectangularSlice> slice =
      subsetOp.getAccessedHyperrectangularSlice();
  assert(succeeded(slice) &&
         "defaultOperatesOnDisjointSubset must be implemented if "
         "getAccessedHyperrectangularSlice is not implemented");
  FailureOr<HyperrectangularSlice> otherSlice =
      candidate.getAccessedHyperrectangularSlice();
  if (failed(otherSlice))
    return false;
  if (!equivalenceFn(subsetOp.getTensorContainer(),
                     candidate.getTensorContainer()))
    return false;
  FailureOr<bool> overlapping = ValueBoundsConstraintSet::areOverlappingSlices(
      op->getContext(), *slice, *otherSlice);
  return succeeded(overlapping) && !*overlapping;
}

Value detail::getTensorContainer(Operation *op) {
  if (auto insertionOp = dyn_cast<::mlir::SubsetInsertionOpInterface>(op))
    return insertionOp.getDestinationOperand().get();
  return cast<::mlir::SubsetExtractionOpInterface>(op).getSourceOperand().get();
}

LogicalResult detail::verifySubsetOpInterface(SubsetOpInterface op) {
  if (!(isa<SubsetExtractionOpInterface>(op.getOperation()) ^
        isa<SubsetInsertionOpInterface>(op.getOperation())))
    return op->emitOpError(
        "SubsetOpInterface ops must implement either "
        "SubsetExtractionOpInterface or SubsetInsertionOpInterface");
  return success();
}

LogicalResult
detail::verifySubsetExtractionOpInterface(SubsetExtractionOpInterface op) {
  if (op->getNumResults() != 1)
    return op->emitOpError(
        "SubsetExtractionOpInterface ops must have one result");
  return success();
}
