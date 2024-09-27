//===- StaticValueUtils.cpp - Utilities for dealing with static values ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {

bool isZeroIndex(OpFoldResult v) {
  if (!v)
    return false;
  std::optional<int64_t> constint = getConstantIntValue(v);
  if (!constint)
    return false;
  return *constint == 0;
}

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
                               SmallVectorImpl<int64_t> &staticVec) {
  auto v = llvm::dyn_cast_if_present<Value>(ofr);
  if (!v) {
    APInt apInt = cast<IntegerAttr>(ofr.get<Attribute>()).getValue();
    staticVec.push_back(apInt.getSExtValue());
    return;
  }
  dynamicVec.push_back(v);
  staticVec.push_back(ShapedType::kDynamic);
}

void dispatchIndexOpFoldResults(ArrayRef<OpFoldResult> ofrs,
                                SmallVectorImpl<Value> &dynamicVec,
                                SmallVectorImpl<int64_t> &staticVec) {
  for (OpFoldResult ofr : ofrs)
    dispatchIndexOpFoldResult(ofr, dynamicVec, staticVec);
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
  return llvm::to_vector(
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

OpFoldResult getAsIndexOpFoldResult(MLIRContext *ctx, int64_t val) {
  return IntegerAttr::get(IndexType::get(ctx), val);
}

SmallVector<OpFoldResult> getAsIndexOpFoldResult(MLIRContext *ctx,
                                                 ArrayRef<int64_t> values) {
  return llvm::to_vector(llvm::map_range(
      values, [ctx](int64_t v) { return getAsIndexOpFoldResult(ctx, v); }));
}

/// If ofr is a constant integer or an IntegerAttr, return the integer.
std::optional<int64_t> getConstantIntValue(OpFoldResult ofr) {
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

std::optional<SmallVector<int64_t>>
getConstantIntValues(ArrayRef<OpFoldResult> ofrs) {
  bool failed = false;
  SmallVector<int64_t> res = llvm::map_to_vector(ofrs, [&](OpFoldResult ofr) {
    auto cv = getConstantIntValue(ofr);
    if (!cv.has_value())
      failed = true;
    return cv.value_or(0);
  });
  if (failed)
    return std::nullopt;
  return res;
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
  auto v1 = llvm::dyn_cast_if_present<Value>(ofr1),
       v2 = llvm::dyn_cast_if_present<Value>(ofr2);
  return v1 && v1 == v2;
}

bool isEqualConstantIntOrValueArray(ArrayRef<OpFoldResult> ofrs1,
                                    ArrayRef<OpFoldResult> ofrs2) {
  if (ofrs1.size() != ofrs2.size())
    return false;
  for (auto [ofr1, ofr2] : llvm::zip_equal(ofrs1, ofrs2))
    if (!isEqualConstantIntOrValue(ofr1, ofr2))
      return false;
  return true;
}

/// Return a vector of OpFoldResults with the same size a staticValues, but all
/// elements for which ShapedType::isDynamic is true, will be replaced by
/// dynamicValues.
SmallVector<OpFoldResult> getMixedValues(ArrayRef<int64_t> staticValues,
                                         ValueRange dynamicValues, Builder &b) {
  SmallVector<OpFoldResult> res;
  res.reserve(staticValues.size());
  unsigned numDynamic = 0;
  unsigned count = static_cast<unsigned>(staticValues.size());
  for (unsigned idx = 0; idx < count; ++idx) {
    int64_t value = staticValues[idx];
    res.push_back(ShapedType::isDynamic(value)
                      ? OpFoldResult{dynamicValues[numDynamic++]}
                      : OpFoldResult{b.getI64IntegerAttr(staticValues[idx])});
  }
  return res;
}

/// Decompose a vector of mixed static or dynamic values into the corresponding
/// pair of arrays. This is the inverse function of `getMixedValues`.
std::pair<SmallVector<int64_t>, SmallVector<Value>>
decomposeMixedValues(const SmallVectorImpl<OpFoldResult> &mixedValues) {
  SmallVector<int64_t> staticValues;
  SmallVector<Value> dynamicValues;
  for (const auto &it : mixedValues) {
    if (it.is<Attribute>()) {
      staticValues.push_back(cast<IntegerAttr>(it.get<Attribute>()).getInt());
    } else {
      staticValues.push_back(ShapedType::kDynamic);
      dynamicValues.push_back(it.get<Value>());
    }
  }
  return {staticValues, dynamicValues};
}

/// Helper to sort `values` according to matching `keys`.
template <typename K, typename V>
static SmallVector<V>
getValuesSortedByKeyImpl(ArrayRef<K> keys, ArrayRef<V> values,
                         llvm::function_ref<bool(K, K)> compare) {
  if (keys.empty())
    return SmallVector<V>{values};
  assert(keys.size() == values.size() && "unexpected mismatching sizes");
  auto indices = llvm::to_vector(llvm::seq<int64_t>(0, values.size()));
  std::sort(indices.begin(), indices.end(),
            [&](int64_t i, int64_t j) { return compare(keys[i], keys[j]); });
  SmallVector<V> res;
  res.reserve(values.size());
  for (int64_t i = 0, e = indices.size(); i < e; ++i)
    res.push_back(values[indices[i]]);
  return res;
}

SmallVector<Value>
getValuesSortedByKey(ArrayRef<Attribute> keys, ArrayRef<Value> values,
                     llvm::function_ref<bool(Attribute, Attribute)> compare) {
  return getValuesSortedByKeyImpl(keys, values, compare);
}

SmallVector<OpFoldResult>
getValuesSortedByKey(ArrayRef<Attribute> keys, ArrayRef<OpFoldResult> values,
                     llvm::function_ref<bool(Attribute, Attribute)> compare) {
  return getValuesSortedByKeyImpl(keys, values, compare);
}

SmallVector<int64_t>
getValuesSortedByKey(ArrayRef<Attribute> keys, ArrayRef<int64_t> values,
                     llvm::function_ref<bool(Attribute, Attribute)> compare) {
  return getValuesSortedByKeyImpl(keys, values, compare);
}

/// Return the number of iterations for a loop with a lower bound `lb`, upper
/// bound `ub` and step `step`.
std::optional<int64_t> constantTripCount(OpFoldResult lb, OpFoldResult ub,
                                         OpFoldResult step) {
  if (lb == ub)
    return 0;

  std::optional<int64_t> lbConstant = getConstantIntValue(lb);
  if (!lbConstant)
    return std::nullopt;
  std::optional<int64_t> ubConstant = getConstantIntValue(ub);
  if (!ubConstant)
    return std::nullopt;
  std::optional<int64_t> stepConstant = getConstantIntValue(step);
  if (!stepConstant)
    return std::nullopt;

  return llvm::divideCeilSigned(*ubConstant - *lbConstant, *stepConstant);
}

bool hasValidSizesOffsets(SmallVector<int64_t> sizesOrOffsets) {
  return llvm::none_of(sizesOrOffsets, [](int64_t value) {
    return !ShapedType::isDynamic(value) && value < 0;
  });
}

bool hasValidStrides(SmallVector<int64_t> strides) {
  return llvm::none_of(strides, [](int64_t value) {
    return !ShapedType::isDynamic(value) && value == 0;
  });
}

LogicalResult foldDynamicIndexList(SmallVectorImpl<OpFoldResult> &ofrs,
                                   bool onlyNonNegative, bool onlyNonZero) {
  bool valuesChanged = false;
  for (OpFoldResult &ofr : ofrs) {
    if (ofr.is<Attribute>())
      continue;
    Attribute attr;
    if (matchPattern(ofr.get<Value>(), m_Constant(&attr))) {
      // Note: All ofrs have index type.
      if (onlyNonNegative && *getConstantIntValue(attr) < 0)
        continue;
      if (onlyNonZero && *getConstantIntValue(attr) == 0)
        continue;
      ofr = attr;
      valuesChanged = true;
    }
  }
  return success(valuesChanged);
}

LogicalResult
foldDynamicOffsetSizeList(SmallVectorImpl<OpFoldResult> &offsetsOrSizes) {
  return foldDynamicIndexList(offsetsOrSizes, /*onlyNonNegative=*/true,
                              /*onlyNonZero=*/false);
}

LogicalResult foldDynamicStrideList(SmallVectorImpl<OpFoldResult> &strides) {
  return foldDynamicIndexList(strides, /*onlyNonNegative=*/false,
                              /*onlyNonZero=*/true);
}

} // namespace mlir
