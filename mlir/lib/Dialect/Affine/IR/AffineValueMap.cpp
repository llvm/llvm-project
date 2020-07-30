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

void AffineValueMap::difference(const AffineValueMap &a,
                                const AffineValueMap &b, AffineValueMap *res) {
  assert(a.getNumResults() == b.getNumResults() && "invalid inputs");

  // Fully compose A's map + operands.
  auto aMap = a.getAffineMap();
  SmallVector<Value, 4> aOperands(a.getOperands().begin(),
                                  a.getOperands().end());
  fullyComposeAffineMapAndOperands(&aMap, &aOperands);

  // Use the affine apply normalizer to get B's map into A's coordinate space.
  AffineApplyNormalizer normalizer(aMap, aOperands);
  SmallVector<Value, 4> bOperands(b.getOperands().begin(),
                                  b.getOperands().end());
  auto bMap = b.getAffineMap();
  normalizer.normalize(&bMap, &bOperands);

  assert(std::equal(bOperands.begin(), bOperands.end(),
                    normalizer.getOperands().begin()) &&
         "operands are expected to be the same after normalization");

  // Construct the difference expressions.
  SmallVector<AffineExpr, 4> diffExprs;
  diffExprs.reserve(a.getNumResults());
  for (unsigned i = 0, e = bMap.getNumResults(); i < e; ++i)
    diffExprs.push_back(normalizer.getAffineMap().getResult(i) -
                        bMap.getResult(i));

  auto diffMap =
      AffineMap::get(normalizer.getNumDims(), normalizer.getNumSymbols(),
                     diffExprs, aMap.getContext());
  canonicalizeMapAndOperands(&diffMap, &bOperands);
  diffMap = simplifyAffineMap(diffMap);
  res->reset(diffMap, bOperands);
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

AffineValueMap::~AffineValueMap() {}
