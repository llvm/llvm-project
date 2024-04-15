//===- InferIntRangeInterface.cpp -  Integer range inference interface ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/IR/BuiltinTypes.h"
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
  return os << "unsigned : [" << range.umin() << ", " << range.umax()
            << "] signed : [" << range.smin() << ", " << range.smax() << "]";
}
