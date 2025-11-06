//===- StaticValueUtils.cpp - Utilities for dealing with static values ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {

bool isZeroInteger(OpFoldResult v) { return isConstantIntValue(v, 0); }

bool isOneInteger(OpFoldResult v) { return isConstantIntValue(v, 1); }

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
    APInt apInt = cast<IntegerAttr>(cast<Attribute>(ofr)).getValue();
    staticVec.push_back(apInt.getSExtValue());
    return;
  }
  dynamicVec.push_back(v);
  staticVec.push_back(ShapedType::kDynamic);
}

std::pair<int64_t, OpFoldResult>
getSimplifiedOfrAndStaticSizePair(OpFoldResult tileSizeOfr, Builder &b) {
  int64_t tileSizeForShape =
      getConstantIntValue(tileSizeOfr).value_or(ShapedType::kDynamic);

  OpFoldResult tileSizeOfrSimplified =
      (tileSizeForShape != ShapedType::kDynamic)
          ? b.getIndexAttr(tileSizeForShape)
          : tileSizeOfr;

  return std::pair<int64_t, OpFoldResult>(tileSizeForShape,
                                          tileSizeOfrSimplified);
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
/// The boolean indicates whether the value is an index type.
std::optional<std::pair<APInt, bool>> getConstantAPIntValue(OpFoldResult ofr) {
  // Case 1: Check for Constant integer.
  if (auto val = llvm::dyn_cast_if_present<Value>(ofr)) {
    APInt intVal;
    if (matchPattern(val, m_ConstantInt(&intVal)))
      return std::make_pair(intVal, val.getType().isIndex());
    return std::nullopt;
  }
  // Case 2: Check for IntegerAttr.
  Attribute attr = llvm::dyn_cast_if_present<Attribute>(ofr);
  if (auto intAttr = dyn_cast_or_null<IntegerAttr>(attr))
    return std::make_pair(intAttr.getValue(), intAttr.getType().isIndex());
  return std::nullopt;
}

/// If ofr is a constant integer or an IntegerAttr, return the integer.
std::optional<int64_t> getConstantIntValue(OpFoldResult ofr) {
  std::optional<std::pair<APInt, bool>> apInt = getConstantAPIntValue(ofr);
  if (!apInt)
    return std::nullopt;
  return apInt->first.getSExtValue();
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

bool isConstantIntValue(OpFoldResult ofr, int64_t value) {
  return getConstantIntValue(ofr) == value;
}

bool areAllConstantIntValue(ArrayRef<OpFoldResult> ofrs, int64_t value) {
  return llvm::all_of(
      ofrs, [&](OpFoldResult ofr) { return isConstantIntValue(ofr, value); });
}

bool areConstantIntValues(ArrayRef<OpFoldResult> ofrs,
                          ArrayRef<int64_t> values) {
  if (ofrs.size() != values.size())
    return false;
  std::optional<SmallVector<int64_t>> constOfrs = getConstantIntValues(ofrs);
  return constOfrs && llvm::equal(constOfrs.value(), values);
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

/// Return a vector of OpFoldResults with the same size as staticValues, but all
/// elements for which ShapedType::isDynamic is true, will be replaced by
/// dynamicValues.
SmallVector<OpFoldResult> getMixedValues(ArrayRef<int64_t> staticValues,
                                         ValueRange dynamicValues,
                                         MLIRContext *context) {
  assert(dynamicValues.size() == static_cast<size_t>(llvm::count_if(
                                     staticValues, ShapedType::isDynamic)) &&
         "expected the rank of dynamic values to match the number of "
         "values known to be dynamic");
  SmallVector<OpFoldResult> res;
  res.reserve(staticValues.size());
  unsigned numDynamic = 0;
  unsigned count = static_cast<unsigned>(staticValues.size());
  for (unsigned idx = 0; idx < count; ++idx) {
    int64_t value = staticValues[idx];
    res.push_back(ShapedType::isDynamic(value)
                      ? OpFoldResult{dynamicValues[numDynamic++]}
                      : OpFoldResult{IntegerAttr::get(
                            IntegerType::get(context, 64), staticValues[idx])});
  }
  return res;
}
SmallVector<OpFoldResult> getMixedValues(ArrayRef<int64_t> staticValues,
                                         ValueRange dynamicValues, Builder &b) {
  return getMixedValues(staticValues, dynamicValues, b.getContext());
}

/// Decompose a vector of mixed static or dynamic values into the corresponding
/// pair of arrays. This is the inverse function of `getMixedValues`.
std::pair<SmallVector<int64_t>, SmallVector<Value>>
decomposeMixedValues(ArrayRef<OpFoldResult> mixedValues) {
  SmallVector<int64_t> staticValues;
  SmallVector<Value> dynamicValues;
  for (const auto &it : mixedValues) {
    if (auto attr = dyn_cast<Attribute>(it)) {
      staticValues.push_back(cast<IntegerAttr>(attr).getInt());
    } else {
      staticValues.push_back(ShapedType::kDynamic);
      dynamicValues.push_back(cast<Value>(it));
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
  llvm::sort(indices,
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
std::optional<APInt> constantTripCount(
    OpFoldResult lb, OpFoldResult ub, OpFoldResult step, bool isSigned,
    llvm::function_ref<std::optional<llvm::APSInt>(Value, Value, bool)>
        computeUbMinusLb) {
  // This is the bitwidth used to return 0 when loop does not execute.
  // We infer it from the type of the bound if it isn't an index type.
  auto getBitwidth = [&](OpFoldResult ofr) -> std::tuple<int, bool> {
    if (auto intAttr =
            dyn_cast_or_null<IntegerAttr>(dyn_cast<Attribute>(ofr))) {
      if (auto intType = dyn_cast<IntegerType>(intAttr.getType()))
        return std::make_tuple(intType.getWidth(), intType.isIndex());
    } else {
      auto val = cast<Value>(ofr);
      if (auto intType = dyn_cast<IntegerType>(val.getType()))
        return std::make_tuple(intType.getWidth(), intType.isIndex());
    }
    return std::make_tuple(IndexType::kInternalStorageBitWidth, true);
  };
  auto [bitwidth, isIndex] = getBitwidth(lb);
  // This would better be an assert, but unfortunately it breaks scf.for_all
  // which is missing attributes and SSA value optionally for its bounds, and
  // uses Index type for the dynamic bounds but i64 for the static bounds. This
  // is broken...
  if (std::tie(bitwidth, isIndex) != getBitwidth(ub)) {
    LDBG() << "mismatch between lb and ub bitwidth/type: " << ub << " vs "
           << lb;
    return std::nullopt;
  }
  if (lb == ub)
    return APInt(bitwidth, 0);

  std::optional<std::pair<APInt, bool>> maybeStepCst =
      getConstantAPIntValue(step);

  if (maybeStepCst) {
    auto &stepCst = maybeStepCst->first;
    assert(static_cast<int>(stepCst.getBitWidth()) == bitwidth &&
           "step must have the same bitwidth as lb and ub");
    if (stepCst.isZero())
      return stepCst;
    if (stepCst.isNegative())
      return APInt(bitwidth, 0);
  }

  if (isIndex) {
    LDBG()
        << "Computing loop trip count for index type may break with overflow";
    // TODO: we can't compute the trip count for index type. We should fix this
    // but too many tests are failing right now.
    //   return {};
  }

  /// Compute the difference between the upper and lower bound: either from the
  /// constant value or using the computeUbMinusLb callback.
  llvm::APSInt diff;
  std::optional<std::pair<APInt, bool>> maybeLbCst = getConstantAPIntValue(lb);
  std::optional<std::pair<APInt, bool>> maybeUbCst = getConstantAPIntValue(ub);
  if (maybeLbCst) {
    // If one of the bounds is not a constant, we can't compute the trip count.
    if (!maybeUbCst)
      return std::nullopt;
    APSInt lbCst(maybeLbCst->first, /*isUnsigned=*/!isSigned);
    APSInt ubCst(maybeUbCst->first, /*isUnsigned=*/!isSigned);
    if (!maybeUbCst)
      return std::nullopt;
    if (ubCst <= lbCst) {
      LDBG() << "constantTripCount is 0 because ub <= lb (" << lbCst << "("
             << lbCst.getBitWidth() << ") <= " << ubCst << "("
             << ubCst.getBitWidth() << "), "
             << (isSigned ? "isSigned" : "isUnsigned") << ")";
      return APInt(bitwidth, 0);
    }
    diff = ubCst - lbCst;
  } else {
    if (maybeUbCst)
      return std::nullopt;

    /// Non-constant bound, let's try to compute the difference between the
    /// upper and lower bound
    std::optional<llvm::APSInt> maybeDiff =
        computeUbMinusLb(cast<Value>(lb), cast<Value>(ub), isSigned);
    if (!maybeDiff)
      return std::nullopt;
    diff = *maybeDiff;
  }
  LDBG() << "constantTripCount: " << (isSigned ? "isSigned" : "isUnsigned")
         << ", ub-lb: " << diff << "(" << diff.getBitWidth() << "b)";
  if (diff.isNegative()) {
    LDBG() << "constantTripCount is 0 because ub-lb diff is negative";
    return APInt(bitwidth, 0);
  }
  if (!maybeStepCst) {
    LDBG()
        << "constantTripCount can't be computed because step is not a constant";
    return std::nullopt;
  }
  auto &stepCst = maybeStepCst->first;
  llvm::APInt tripCount = diff.sdiv(stepCst);
  llvm::APInt r = diff.srem(stepCst);
  if (!r.isZero())
    tripCount = tripCount + 1;
  LDBG() << "constantTripCount found: " << tripCount;
  return tripCount;
}

bool hasValidSizesOffsets(SmallVector<int64_t> sizesOrOffsets) {
  return llvm::none_of(sizesOrOffsets, [](int64_t value) {
    return ShapedType::isStatic(value) && value < 0;
  });
}

bool hasValidStrides(SmallVector<int64_t> strides) {
  return llvm::none_of(strides, [](int64_t value) {
    return ShapedType::isStatic(value) && value == 0;
  });
}

LogicalResult foldDynamicIndexList(SmallVectorImpl<OpFoldResult> &ofrs,
                                   bool onlyNonNegative, bool onlyNonZero) {
  bool valuesChanged = false;
  for (OpFoldResult &ofr : ofrs) {
    if (isa<Attribute>(ofr))
      continue;
    Attribute attr;
    if (matchPattern(cast<Value>(ofr), m_Constant(&attr))) {
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
