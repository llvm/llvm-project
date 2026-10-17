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
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// OpFoldResult
//===----------------------------------------------------------------------===//

void OpFoldResult::dump() const { llvm::errs() << *this << "\n"; }

raw_ostream &mlir::operator<<(raw_ostream &os, OpFoldResult ofr) {
  if (Value value = llvm::dyn_cast_if_present<Value>(ofr))
    value.print(os);
  else
    llvm::dyn_cast_if_present<Attribute>(ofr).print(os);
  return os;
}

/// Return true if every replacement is null.
static bool replacesNone(ArrayRef<OpFoldResult> replacements) {
  return llvm::none_of(replacements, llvm::identity{});
}

//===----------------------------------------------------------------------===//
// OpFoldResults
//===----------------------------------------------------------------------===//

OpFoldResults::OpFoldResults(LogicalResult status) {
  inPlace = status.succeeded();
}

OpFoldResults::OpFoldResults(OpFoldResult replacement) {
  if (replacement)
    replacements.push_back(replacement);
}

OpFoldResults::OpFoldResults(std::initializer_list<OpFoldResult> list)
    : OpFoldResults(ArrayRef<OpFoldResult>(list)) {}

OpFoldResults OpFoldResults::fromLegacy(LogicalResult status,
                                        ArrayRef<OpFoldResult> results) {
  if (failed(status))
    return failure();
  if (results.empty())
    return success();
  assert(llvm::all_of(results, llvm::identity{}) &&
         "legacy fold returned a null result");
  return results;
}

void OpFoldResults::setModifiedInPlace(bool modified) { inPlace = modified; }

//===----------------------------------------------------------------------===//
// NormalizedOpFoldResults
//===----------------------------------------------------------------------===//

/// Assert that each value replacement has the type of its result.
static void verifyReplacementTypes(Operation *op,
                                   ArrayRef<OpFoldResult> replacements) {
#ifndef NDEBUG
  for (auto [index, replacement] : llvm::enumerate(replacements)) {
    auto value = dyn_cast_if_present<Value>(replacement);
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

NormalizedOpFoldResults::NormalizedOpFoldResults(Operation *op,
                                                 OpFoldResults &&results)
    : OpFoldResultsBase(std::move(results)) {
  assert(op && "expected a non-null operation");
  if (!replacements.empty()) {
    assert(replacements.size() == op->getNumResults() &&
           "expected one replacement per operation result");
    for (auto [replacement, result] :
         llvm::zip_equal(replacements, op->getResults()))
      if (dyn_cast_if_present<Value>(replacement) == result)
        replacement = OpFoldResult();
    if (replacesNone(replacements))
      replacements.clear();
  }
  verifyReplacementTypes(op, replacements);
}

void NormalizedOpFoldResults::setModifiedInPlace(bool modified) {
  inPlace = modified;
}

bool NormalizedOpFoldResults::modifiedInPlace() const { return inPlace; }

bool NormalizedOpFoldResults::replacesAny() const {
  return !replacements.empty();
}

bool NormalizedOpFoldResults::replacesAll() const {
  return replacesAny() && llvm::all_of(replacements, llvm::identity{});
}

OpFoldResult NormalizedOpFoldResults::operator[](unsigned i) const {
  if (replacements.empty())
    return OpFoldResult();
  assert(i < replacements.size() && "result index out of range");
  return replacements[i];
}

ArrayRef<OpFoldResult> NormalizedOpFoldResults::getReplacements() const {
  return replacements;
}

OpFoldResults detail::convertSingleResultFold(Operation *op,
                                              OpFoldResult result) {
  if (dyn_cast_if_present<Value>(result) == op->getResult(0))
    return success();
  return result;
}

NormalizedOpFoldResults
detail::dropReplacementsOfReplacedResults(Operation *op,
                                          NormalizedOpFoldResults result) {
  bool namesReplacedResult =
      llvm::any_of(result.getReplacements(), [&](OpFoldResult replacement) {
        auto opResult = dyn_cast_if_present<OpResult>(
            dyn_cast_if_present<Value>(replacement));
        return opResult && opResult.getOwner() == op &&
               result[opResult.getResultNumber()];
      });
  if (!namesReplacedResult)
    return result;
  return NormalizedOpFoldResults(
      op, OpFoldResults(success(result.modifiedInPlace())));
}
