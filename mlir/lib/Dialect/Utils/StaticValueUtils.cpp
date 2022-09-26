//===- StaticValueUtils.cpp - Utilities for dealing with static values ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Arithmetic/Utils/Utils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APSInt.h"

namespace mlir {

std::tuple<SmallVector<OpFoldResult>, SmallVector<OpFoldResult>,
           SmallVector<OpFoldResult>>
getOffsetsSizesAndStrides(ArrayRef<Range> ranges) {
  SmallVector<OpFoldResult> offsets, sizes, strides;
  offsets.reserve(ranges.size());
  sizes.reserve(ranges.size());
  strides.reserve(ranges.size());
  for (const auto &[offset, size, stride] : ranges) {
    offsets.push_back(offset);
    sizes.push_back(size);
    strides.push_back(stride);
  }
  return std::make_tuple(offsets, sizes, strides);
}

/// Helper function to dispatch an OpFoldResult into `staticVec` if:
///   a) it is an IntegerAttr
/// In other cases, the OpFoldResult is dispached to the `dynamicVec`.
/// In such dynamic cases, a copy of the `sentinel` value is also pushed to
/// `staticVec`. This is useful to extract mixed static and dynamic entries that
/// come from an AttrSizedOperandSegments trait.
void dispatchIndexOpFoldResult(OpFoldResult ofr,
                               SmallVectorImpl<Value> &dynamicVec,
                               SmallVectorImpl<int64_t> &staticVec,
                               int64_t sentinel) {
  auto v = ofr.dyn_cast<Value>();
  if (!v) {
    APInt apInt = ofr.get<Attribute>().cast<IntegerAttr>().getValue();
    staticVec.push_back(apInt.getSExtValue());
    return;
  }
  dynamicVec.push_back(v);
  staticVec.push_back(sentinel);
}

void dispatchIndexOpFoldResults(ArrayRef<OpFoldResult> ofrs,
                                SmallVectorImpl<Value> &dynamicVec,
                                SmallVectorImpl<int64_t> &staticVec,
                                int64_t sentinel) {
  for (OpFoldResult ofr : ofrs)
    dispatchIndexOpFoldResult(ofr, dynamicVec, staticVec, sentinel);
}

/// Extract int64_t values from the assumed ArrayAttr of IntegerAttr.
SmallVector<int64_t, 4> extractFromI64ArrayAttr(Attribute attr) {
  return llvm::to_vector<4>(
      llvm::map_range(attr.cast<ArrayAttr>(), [](Attribute a) -> int64_t {
        return a.cast<IntegerAttr>().getInt();
      }));
}

/// Given a value, try to extract a constant Attribute. If this fails, return
/// the original value.
OpFoldResult getAsOpFoldResult(Value val) {
  if (!val)
    return OpFoldResult();
  Attribute attr;
  if (matchPattern(val, m_Constant(&attr)))
    return attr;
  return val;
}

/// Given an array of values, try to extract a constant Attribute from each
/// value. If this fails, return the original value.
SmallVector<OpFoldResult> getAsOpFoldResult(ValueRange values) {
  return llvm::to_vector<4>(
      llvm::map_range(values, [](Value v) { return getAsOpFoldResult(v); }));
}

/// Convert `arrayAttr` to a vector of OpFoldResult.
SmallVector<OpFoldResult> getAsOpFoldResult(ArrayAttr arrayAttr) {
  SmallVector<OpFoldResult> res;
  res.reserve(arrayAttr.size());
  for (Attribute a : arrayAttr)
    res.push_back(a);
  return res;
}

/// If ofr is a constant integer or an IntegerAttr, return the integer.
Optional<int64_t> getConstantIntValue(OpFoldResult ofr) {
  // Case 1: Check for Constant integer.
  if (auto val = ofr.dyn_cast<Value>()) {
    APSInt intVal;
    if (matchPattern(val, m_ConstantInt(&intVal)))
      return intVal.getSExtValue();
    return llvm::None;
  }
  // Case 2: Check for IntegerAttr.
  Attribute attr = ofr.dyn_cast<Attribute>();
  if (auto intAttr = attr.dyn_cast_or_null<IntegerAttr>())
    return intAttr.getValue().getSExtValue();
  return llvm::None;
}

/// Return true if `ofr` is constant integer equal to `value`.
bool isConstantIntValue(OpFoldResult ofr, int64_t value) {
  auto val = getConstantIntValue(ofr);
  return val && *val == value;
}

/// Return true if ofr1 and ofr2 are the same integer constant attribute values
/// or the same SSA value.
/// Ignore integer bitwidth and type mismatch that come from the fact there is
/// no IndexAttr and that IndexType has no bitwidth.
bool isEqualConstantIntOrValue(OpFoldResult ofr1, OpFoldResult ofr2) {
  auto cst1 = getConstantIntValue(ofr1), cst2 = getConstantIntValue(ofr2);
  if (cst1 && cst2 && *cst1 == *cst2)
    return true;
  auto v1 = ofr1.dyn_cast<Value>(), v2 = ofr2.dyn_cast<Value>();
  return v1 && v1 == v2;
}

/// Helper function to convert a vector of `OpFoldResult`s into a vector of
/// `Value`s. For each `OpFoldResult` in `valueOrAttrVec` return the fold result
/// if it casts to  a `Value` or create an index-type constant if it casts to
/// `IntegerAttr`. No other attribute types are supported.
SmallVector<Value> getAsValues(OpBuilder &b, Location loc,
                               ArrayRef<OpFoldResult> valueOrAttrVec) {
  return llvm::to_vector<4>(
      llvm::map_range(valueOrAttrVec, [&](OpFoldResult value) -> Value {
        return getValueOrCreateConstantIndexOp(b, loc, value);
      }));
}
} // namespace mlir
