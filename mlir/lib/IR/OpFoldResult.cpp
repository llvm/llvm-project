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
  return llvm::none_of(replacements, [](OpFoldResult replacement) {
    return static_cast<bool>(replacement);
  });
}

//===----------------------------------------------------------------------===//
// OpFoldResults
//===----------------------------------------------------------------------===//

OpFoldResults::OpFoldResults(std::nullptr_t) {}

OpFoldResults::OpFoldResults(LogicalResult status)
    : inPlace(status.succeeded()) {}

OpFoldResults::OpFoldResults(OpFoldResult replacement) {
  if (replacement)
    replacements.push_back(replacement);
}

OpFoldResults::OpFoldResults(Value replacement)
    : OpFoldResults(OpFoldResult(replacement)) {}

OpFoldResults::OpFoldResults(Attribute replacement)
    : OpFoldResults(OpFoldResult(replacement)) {}

OpFoldResults::OpFoldResults(std::initializer_list<OpFoldResult> replacements)
    : replacements(replacements) {}

OpFoldResults::OpFoldResults(Operation *op) {
  assert(op && "expected a non-null operation");
  replacements.resize(op->getNumResults());
}

void OpFoldResults::replace(Value result, OpFoldResult replacement) {
  auto opResult = dyn_cast_if_present<OpResult>(result);
  assert(opResult && "expected an OpResult");
  if (dyn_cast_if_present<Value>(replacement) == result)
    replacement = OpFoldResult();
  replace(opResult.getResultNumber(), replacement);
}

void OpFoldResults::replace(unsigned resultIndex, OpFoldResult replacement) {
  assert(resultIndex < replacements.size() && "result index out of range");
  replacements[resultIndex] = replacement;
}

void OpFoldResults::setModifiedInPlace(bool modified) { inPlace = modified; }

bool OpFoldResults::succeeded() const { return inPlace || replacesAny(); }

bool OpFoldResults::failed() const { return !succeeded(); }

bool OpFoldResults::modifiedInPlace() const { return inPlace; }

bool OpFoldResults::replacesAny() const { return !replacesNone(replacements); }

bool OpFoldResults::replacesAll() const {
  return !replacements.empty() &&
         llvm::all_of(replacements, [](OpFoldResult replacement) {
           return static_cast<bool>(replacement);
         });
}

unsigned OpFoldResults::size() const { return replacements.size(); }

OpFoldResult OpFoldResults::operator[](unsigned i) const {
  if (replacements.empty())
    return OpFoldResult();
  assert(i < replacements.size() && "result index out of range");
  return replacements[i];
}

ArrayRef<OpFoldResult> OpFoldResults::getReplacements() const {
  return replacements;
}

void OpFoldResults::normalize(Operation *op) {
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
      llvm::any_of(result.getReplacements(), [&](OpFoldResult replacement) {
        auto opResult = dyn_cast_if_present<OpResult>(
            dyn_cast_if_present<Value>(replacement));
        return opResult && opResult.getOwner() == op &&
               result[opResult.getResultNumber()];
      });
  if (!namesReplacedResult)
    return;
  result = success(result.modifiedInPlace());
}
