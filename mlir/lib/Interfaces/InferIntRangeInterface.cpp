//===- InferIntRangeInterface.cpp -  Integer range inference interface ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/InferIntRangeInterface.cpp.inc"
#include <optional>

using namespace mlir;

bool ConstantIntRanges::operator==(const ConstantIntRanges &other) const {
  return umin().getBitWidth() == other.umin().getBitWidth() &&
         umin() == other.umin() && umax() == other.umax() &&
         smin() == other.smin() && smax() == other.smax();
}

const APInt &ConstantIntRanges::umin() const { return uminVal; }

const APInt &ConstantIntRanges::umax() const { return umaxVal; }

const APInt &ConstantIntRanges::smin() const { return sminVal; }

const APInt &ConstantIntRanges::smax() const { return smaxVal; }

unsigned ConstantIntRanges::getStorageBitwidth(Type type) {
  type = getElementTypeOrSelf(type);
  if (type.isIndex())
    return IndexType::kInternalStorageBitWidth;
  if (auto integerType = dyn_cast<IntegerType>(type))
    return integerType.getWidth();
  // Non-integer types have their bounds stored in width 0 `APInt`s.
  return 0;
}

ConstantIntRanges ConstantIntRanges::maxRange(unsigned bitwidth) {
  return fromUnsigned(APInt::getZero(bitwidth), APInt::getMaxValue(bitwidth));
}

ConstantIntRanges ConstantIntRanges::constant(const APInt &value) {
  return {value, value, value, value};
}

ConstantIntRanges ConstantIntRanges::range(const APInt &min, const APInt &max,
                                           bool isSigned) {
  if (isSigned)
    return fromSigned(min, max);
  return fromUnsigned(min, max);
}

ConstantIntRanges ConstantIntRanges::fromSigned(const APInt &smin,
                                                const APInt &smax) {
  unsigned int width = smin.getBitWidth();
  APInt umin, umax;
  if (smin.isNonNegative() == smax.isNonNegative()) {
    umin = smin.ult(smax) ? smin : smax;
    umax = smin.ugt(smax) ? smin : smax;
  } else {
    umin = APInt::getMinValue(width);
    umax = APInt::getMaxValue(width);
  }
  return {umin, umax, smin, smax};
}

ConstantIntRanges ConstantIntRanges::fromUnsigned(const APInt &umin,
                                                  const APInt &umax) {
  unsigned int width = umin.getBitWidth();
  APInt smin, smax;
  if (umin.isNonNegative() == umax.isNonNegative()) {
    smin = umin.slt(umax) ? umin : umax;
    smax = umin.sgt(umax) ? umin : umax;
  } else {
    smin = APInt::getSignedMinValue(width);
    smax = APInt::getSignedMaxValue(width);
  }
  return {umin, umax, smin, smax};
}

ConstantIntRanges
ConstantIntRanges::rangeUnion(const ConstantIntRanges &other) const {
  // "Not an integer" poisons everything and also cannot be fed to comparison
  // operators.
  if (umin().getBitWidth() == 0)
    return *this;
  if (other.umin().getBitWidth() == 0)
    return other;

  const APInt &uminUnion = umin().ult(other.umin()) ? umin() : other.umin();
  const APInt &umaxUnion = umax().ugt(other.umax()) ? umax() : other.umax();
  const APInt &sminUnion = smin().slt(other.smin()) ? smin() : other.smin();
  const APInt &smaxUnion = smax().sgt(other.smax()) ? smax() : other.smax();

  return {uminUnion, umaxUnion, sminUnion, smaxUnion};
}

ConstantIntRanges
ConstantIntRanges::intersection(const ConstantIntRanges &other) const {
  // "Not an integer" poisons everything and also cannot be fed to comparison
  // operators.
  if (umin().getBitWidth() == 0)
    return *this;
  if (other.umin().getBitWidth() == 0)
    return other;

  const APInt &uminIntersect = umin().ugt(other.umin()) ? umin() : other.umin();
  const APInt &umaxIntersect = umax().ult(other.umax()) ? umax() : other.umax();
  const APInt &sminIntersect = smin().sgt(other.smin()) ? smin() : other.smin();
  const APInt &smaxIntersect = smax().slt(other.smax()) ? smax() : other.smax();

  return {uminIntersect, umaxIntersect, sminIntersect, smaxIntersect};
}

std::optional<APInt> ConstantIntRanges::getConstantValue() const {
  // Note: we need to exclude the trivially-equal width 0 values here.
  if (umin() == umax() && umin().getBitWidth() != 0)
    return umin();
  if (smin() == smax() && smin().getBitWidth() != 0)
    return smin();
  return std::nullopt;
}

raw_ostream &mlir::operator<<(raw_ostream &os, const ConstantIntRanges &range) {
  os << "unsigned : [";
  range.umin().print(os, /*isSigned*/ false);
  os << ", ";
  range.umax().print(os, /*isSigned*/ false);
  return os << "] signed : [" << range.smin() << ", " << range.smax() << "]";
}

IntegerValueRange IntegerValueRange::getMaxRange(Value value) {
  unsigned width = ConstantIntRanges::getStorageBitwidth(value.getType());
  APInt umin = APInt::getMinValue(width);
  APInt umax = APInt::getMaxValue(width);
  APInt smin = width != 0 ? APInt::getSignedMinValue(width) : umin;
  APInt smax = width != 0 ? APInt::getSignedMaxValue(width) : umax;
  return IntegerValueRange{ConstantIntRanges{umin, umax, smin, smax}};
}

raw_ostream &mlir::operator<<(raw_ostream &os, const IntegerValueRange &range) {
  range.print(os);
  return os;
}

SmallVector<IntegerValueRange>
mlir::getIntValueRanges(ArrayRef<OpFoldResult> values,
                        GetIntRangeFn getIntRange, int32_t indexBitwidth) {
  SmallVector<IntegerValueRange> ranges;
  ranges.reserve(values.size());
  for (OpFoldResult ofr : values) {
    if (auto value = dyn_cast<Value>(ofr)) {
      ranges.push_back(getIntRange(value));
      continue;
    }

    // Create a constant range.
    auto attr = cast<IntegerAttr>(cast<Attribute>(ofr));
    ranges.emplace_back(ConstantIntRanges::constant(
        attr.getValue().sextOrTrunc(indexBitwidth)));
  }
  return ranges;
}

void mlir::intrange::detail::defaultInferResultRanges(
    InferIntRangeInterface interface, ArrayRef<IntegerValueRange> argRanges,
    SetIntLatticeFn setResultRanges) {
  llvm::SmallVector<ConstantIntRanges> unpacked;
  unpacked.reserve(argRanges.size());

  for (const IntegerValueRange &range : argRanges) {
    if (range.isUninitialized())
      return;
    unpacked.push_back(range.getValue());
  }

  interface.inferResultRanges(
      unpacked,
      [&setResultRanges](Value value, const ConstantIntRanges &argRanges) {
        setResultRanges(value, IntegerValueRange{argRanges});
      });
}

void mlir::intrange::detail::defaultInferResultRangesFromOptional(
    InferIntRangeInterface interface, ArrayRef<ConstantIntRanges> argRanges,
    SetIntRangeFn setResultRanges) {
  auto ranges = llvm::to_vector_of<IntegerValueRange>(argRanges);
  interface.inferResultRangesFromOptional(
      ranges,
      [&setResultRanges](Value value, const IntegerValueRange &argRanges) {
        if (!argRanges.isUninitialized())
          setResultRanges(value, argRanges.getValue());
      });
}
