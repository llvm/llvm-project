//===- OpFoldResult.cpp - Results of an operation fold --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/OpFoldResult.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/RegionKindInterface.h"

using namespace mlir;

/// Return true if no slot holds a replacement.
static bool replacesNone(ArrayRef<OpFoldResult> slots) {
  return llvm::none_of(
      slots, [](OpFoldResult slot) { return static_cast<bool>(slot); });
}

//===----------------------------------------------------------------------===//
// OpFoldResults
//===----------------------------------------------------------------------===//

OpFoldResults::OpFoldResults(std::nullptr_t) {}

OpFoldResults::OpFoldResults(LogicalResult status)
    : inPlace(status.succeeded()) {}

OpFoldResults::OpFoldResults(OpFoldResult replacement) {
  if (replacement)
    slots.push_back(replacement);
}

OpFoldResults::OpFoldResults(Value replacement)
    : OpFoldResults(OpFoldResult(replacement)) {}

OpFoldResults::OpFoldResults(Attribute replacement)
    : OpFoldResults(OpFoldResult(replacement)) {}

OpFoldResults::OpFoldResults(std::initializer_list<OpFoldResult> replacements)
    : slots(replacements) {}

OpFoldResults::OpFoldResults(Operation *op) {
  assert(op && "expected a non-null operation");
  slots.resize(op->getNumResults());
}

void OpFoldResults::replace(Value result, OpFoldResult replacement) {
  auto opResult = dyn_cast_if_present<OpResult>(result);
  assert(opResult && "expected an OpResult");
  unsigned numResults = opResult.getOwner()->getNumResults();
  if (slots.empty())
    slots.resize(numResults);
  assert(slots.size() == numResults && "expected one slot per result");
  if (dyn_cast_if_present<Value>(replacement) == result)
    replacement = OpFoldResult();
  replace(opResult.getResultNumber(), replacement);
}

void OpFoldResults::replace(unsigned resultIndex, OpFoldResult replacement) {
  assert(resultIndex < slots.size() && "result index out of range");
  slots[resultIndex] = replacement;
}

void OpFoldResults::setModifiedInPlace(bool modified) { inPlace = modified; }

bool OpFoldResults::succeeded() const { return inPlace || replacesAny(); }

bool OpFoldResults::failed() const { return !succeeded(); }

bool OpFoldResults::modifiedInPlace() const { return inPlace; }

bool OpFoldResults::replacesAny() const { return !replacesNone(slots); }

bool OpFoldResults::replacesAll() const {
  return !slots.empty() && llvm::all_of(slots, [](OpFoldResult slot) {
    return static_cast<bool>(slot);
  });
}

unsigned OpFoldResults::size() const { return slots.size(); }

OpFoldResult OpFoldResults::operator[](unsigned i) const {
  if (slots.empty())
    return OpFoldResult();
  assert(i < slots.size() && "result index out of range");
  return slots[i];
}

ArrayRef<OpFoldResult> OpFoldResults::getSlots() const { return slots; }

void OpFoldResults::normalize(Operation *op) {
  assert(op && "expected a non-null operation");
  if (!slots.empty()) {
    assert(slots.size() == op->getNumResults() &&
           "expected one fold result slot per operation result");
    for (auto [slot, result] : llvm::zip_equal(slots, op->getResults()))
      if (dyn_cast_if_present<Value>(slot) == result)
        slot = OpFoldResult();
    if (replacesNone(slots))
      slots.clear();
  }

#ifndef NDEBUG
  for (auto [index, slot] : llvm::enumerate(slots)) {
    auto value = dyn_cast_if_present<Value>(slot);
    if (!value)
      continue;
    Type expectedType = op->getResult(index).getType();
    if (value.getType() != expectedType) {
      op->emitOpError() << "folder produced a value of incorrect type: "
                        << value.getType() << ", expected: " << expectedType;
      assert(false && "incorrect fold result type");
    }
  }
#endif // NDEBUG
}

//===----------------------------------------------------------------------===//
// Legacy fold results
//===----------------------------------------------------------------------===//

OpFoldResults detail::convertLegacyFoldResults(LogicalResult status,
                                               ArrayRef<OpFoldResult> results) {
  if (failed(status))
    return failure();
  if (results.empty())
    return success();
  assert(llvm::all_of(
             results,
             [](OpFoldResult result) { return static_cast<bool>(result); }) &&
         "legacy fold returned a null result");
  return results;
}

void detail::dropReplacementsOfReplacedResults(Operation *op,
                                               OpFoldResults &result) {
  bool namesReplacedResult =
      llvm::any_of(result.getSlots(), [&](OpFoldResult slot) {
        auto opResult =
            dyn_cast_if_present<OpResult>(dyn_cast_if_present<Value>(slot));
        return opResult && opResult.getOwner() == op &&
               result[opResult.getResultNumber()];
      });
  if (!namesReplacedResult)
    return;
  assert(op->getParentRegion() && mayBeGraphRegion(*op->getParentRegion()) &&
         "fold result names a result of the folded operation that the same "
         "fold replaces");
  result = success(result.modifiedInPlace());
}
