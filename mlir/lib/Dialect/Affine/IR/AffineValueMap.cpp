//===- AffineValueMap.cpp - MLIR Affine Value Map Class -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

using namespace mlir;
using namespace mlir::affine;

AffineValueMap::AffineValueMap(AffineMap map, ValueRange operands,
                               ValueRange results)
    : map(map), operands(operands.begin(), operands.end()),
      results(results.begin(), results.end()) {}

void AffineValueMap::reset(AffineMap map, ValueRange operands,
                           ValueRange results) {
  this->map.reset(map);
  this->operands.assign(operands.begin(), operands.end());
  this->results.assign(results.begin(), results.end());
}

void AffineValueMap::composeSimplifyAndCanonicalize() {
  AffineMap sMap = getAffineMap();
  fullyComposeAffineMapAndOperands(&sMap, &operands);
  // Full composition also canonicalizes and simplifies before returning. We
  // need to canonicalize once more to drop unused operands.
  canonicalizeMapAndOperands(&sMap, &operands);
  this->map.reset(sMap);
}

void AffineValueMap::difference(const AffineValueMap &a,
                                const AffineValueMap &b, AffineValueMap *res) {
  assert(a.getNumResults() == b.getNumResults() && "invalid inputs");

  SmallVector<Value, 4> allOperands;
  allOperands.reserve(a.getNumOperands() + b.getNumOperands());
  auto aDims = a.getOperands().take_front(a.getNumDims());
  auto bDims = b.getOperands().take_front(b.getNumDims());
  auto aSyms = a.getOperands().take_back(a.getNumSymbols());
  auto bSyms = b.getOperands().take_back(b.getNumSymbols());
  allOperands.append(aDims.begin(), aDims.end());
  allOperands.append(bDims.begin(), bDims.end());
  allOperands.append(aSyms.begin(), aSyms.end());
  allOperands.append(bSyms.begin(), bSyms.end());

  // Shift dims and symbols of b's map.
  auto bMap = b.getAffineMap()
                  .shiftDims(a.getNumDims())
                  .shiftSymbols(a.getNumSymbols());

  // Construct the difference expressions.
  auto aMap = a.getAffineMap();
  SmallVector<AffineExpr, 4> diffExprs;
  diffExprs.reserve(a.getNumResults());
  for (unsigned i = 0, e = bMap.getNumResults(); i < e; ++i)
    diffExprs.push_back(aMap.getResult(i) - bMap.getResult(i));

  auto diffMap = AffineMap::get(bMap.getNumDims(), bMap.getNumSymbols(),
                                diffExprs, bMap.getContext());
  fullyComposeAffineMapAndOperands(&diffMap, &allOperands);
  canonicalizeMapAndOperands(&diffMap, &allOperands);
  diffMap = simplifyAffineMap(diffMap);
  res->reset(diffMap, allOperands);
}

// Returns true and sets 'indexOfMatch' if 'valueToMatch' is found in
// 'valuesToSearch' beginning at 'indexStart'. Returns false otherwise.
static bool findIndex(Value valueToMatch, ArrayRef<Value> valuesToSearch,
                      unsigned indexStart, unsigned *indexOfMatch) {
  unsigned size = valuesToSearch.size();
  for (unsigned i = indexStart; i < size; ++i) {
    if (valueToMatch == valuesToSearch[i]) {
      *indexOfMatch = i;
      return true;
    }
  }
  return false;
}

bool AffineValueMap::isMultipleOf(unsigned idx, int64_t factor) const {
  return map.isMultipleOf(idx, factor);
}

/// This method uses the invariant that operands are always positionally aligned
/// with the AffineDimExpr in the underlying AffineMap.
bool AffineValueMap::isFunctionOf(unsigned idx, Value value) const {
  unsigned index;
  if (!findIndex(value, operands, /*indexStart=*/0, &index)) {
    return false;
  }
  auto expr = const_cast<AffineValueMap *>(this)->getAffineMap().getResult(idx);
  // TODO: this is better implemented on a flattened representation.
  // At least for now it is conservative.
  return expr.isFunctionOfDim(index);
}

Value AffineValueMap::getOperand(unsigned i) const {
  return static_cast<Value>(operands[i]);
}

ArrayRef<Value> AffineValueMap::getOperands() const {
  return ArrayRef<Value>(operands);
}

AffineMap AffineValueMap::getAffineMap() const { return map.getAffineMap(); }

AffineValueMap::~AffineValueMap() = default;
