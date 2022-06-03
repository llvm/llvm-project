//===- InferIntRangeInterface.h - Integer Range Inference --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions of the integer range inference interface
// defined in `InferIntRange.td`
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_INFERINTRANGEINTERFACE_H
#define MLIR_INTERFACES_INFERINTRANGEINTERFACE_H

#include "mlir/IR/OpDefinition.h"

namespace mlir {
/// A set of arbitrary-precision integers representing bounds on a given integer
/// value. These bounds are inclusive on both ends, so
/// bounds of [4, 5] mean 4 <= x <= 5. Separate bounds are tracked for
/// the unsigned and signed interpretations of values in order to enable more
/// precice inference of the interplay between operations with signed and
/// unsigned semantics.
class ConstantIntRanges {
public:
  /// Bound umin <= (unsigned)x <= umax and smin <= signed(x) <= smax.
  /// Non-integer values should be bounded by APInts of bitwidth 0.
  ConstantIntRanges(const APInt &umin, const APInt &umax, const APInt &smin,
                    const APInt &smax)
      : uminVal(umin), umaxVal(umax), sminVal(smin), smaxVal(smax) {
    assert(uminVal.getBitWidth() == umaxVal.getBitWidth() &&
           umaxVal.getBitWidth() == sminVal.getBitWidth() &&
           sminVal.getBitWidth() == smaxVal.getBitWidth() &&
           "All bounds in the ranges must have the same bitwidth");
  }

  bool operator==(const ConstantIntRanges &other) const;

  /// The minimum value of an integer when it is interpreted as unsigned.
  const APInt &umin() const;

  /// The maximum value of an integer when it is interpreted as unsigned.
  const APInt &umax() const;

  /// The minimum value of an integer when it is interpreted as signed.
  const APInt &smin() const;

  /// The maximum value of an integer when it is interpreted as signed.
  const APInt &smax() const;

  /// Return the bitwidth that should be used for integer ranges describing
  /// `type`. For concrete integer types, this is their bitwidth, for `index`,
  /// this is the internal storage bitwidth of `index` attributes, and for
  /// non-integer types this is 0.
  static unsigned getStorageBitwidth(Type type);

  /// Create an `IntRangeAttrs` where `min` is both the signed and unsigned
  /// minimum and `max` is both the signed and unsigned maximum.
  static ConstantIntRanges range(const APInt &min, const APInt &max);

  /// Create an `IntRangeAttrs` with the signed minimum and maximum equal
  /// to `smin` and `smax`, where the unsigned bounds are constructed from the
  /// signed ones if they correspond to a contigious range of bit patterns when
  /// viewed as unsigned values and are left at [0, int_max()] otherwise.
  static ConstantIntRanges fromSigned(const APInt &smin, const APInt &smax);

  /// Create an `IntRangeAttrs` with the unsigned minimum and maximum equal
  /// to `umin` and `umax` and the signed part equal to `umin` and `umax`
  /// unless the sign bit changes between the minimum and maximum.
  static ConstantIntRanges fromUnsigned(const APInt &umin, const APInt &umax);

  /// Returns the union (computed separately for signed and unsigned bounds)
  /// of `a` and `b`.
  ConstantIntRanges rangeUnion(const ConstantIntRanges &other) const;

  /// If either the signed or unsigned interpretations of the range
  /// indicate that the value it bounds is a constant, return that constant
  /// value.
  Optional<APInt> getConstantValue() const;

  friend raw_ostream &operator<<(raw_ostream &os,
                                 const ConstantIntRanges &range);

private:
  APInt uminVal, umaxVal, sminVal, smaxVal;
};

/// The type of the `setResultRanges` callback provided to ops implementing
/// InferIntRangeInterface. It should be called once for each integer result
/// value and be passed the ConstantIntRanges corresponding to that value.
using SetIntRangeFn = function_ref<void(Value, const ConstantIntRanges &)>;
} // end namespace mlir

#include "mlir/Interfaces/InferIntRangeInterface.h.inc"

#endif // MLIR_INTERFACES_INFERINTRANGEINTERFACE_H
