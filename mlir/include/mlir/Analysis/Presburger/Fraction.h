//===- Fraction.h - MLIR Fraction Class -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a simple class to represent fractions. It supports multiplication,
// comparison, floor, and ceiling operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_FRACTION_H
#define MLIR_ANALYSIS_PRESBURGER_FRACTION_H

#include "mlir/Support/MathExtras.h"

namespace mlir {

/// A class to represent fractions. The sign of the fraction is represented
/// in the sign of the numerator; the denominator is always positive.
///
/// Note that overflows may occur if the numerator or denominator are not
/// representable by 64-bit integers.
struct Fraction {
  /// Default constructor initializes the represented rational number to zero.
  Fraction() = default;

  /// Construct a Fraction from a numerator and denominator.
  Fraction(int64_t oNum, int64_t oDen) : num(oNum), den(oDen) {
    if (den < 0) {
      num = -num;
      den = -den;
    }
  }

  // Return the value of the fraction as an integer. This should only be called
  // when the fraction's value is really an integer.
  int64_t getAsInteger() const {
    assert(num % den == 0 && "Get as integer called on non-integral fraction!");
    return num / den;
  }

  /// The numerator and denominator, respectively. The denominator is always
  /// positive.
  int64_t num{0}, den{1};
};

/// Three-way comparison between two fractions.
/// Returns +1, 0, and -1 if the first fraction is greater than, equal to, or
/// less than the second fraction, respectively.
inline int compare(Fraction x, Fraction y) {
  int64_t diff = x.num * y.den - y.num * x.den;
  if (diff > 0)
    return +1;
  if (diff < 0)
    return -1;
  return 0;
}

inline int64_t floor(Fraction f) { return floorDiv(f.num, f.den); }

inline int64_t ceil(Fraction f) { return ceilDiv(f.num, f.den); }

inline Fraction operator-(Fraction x) { return Fraction(-x.num, x.den); }

inline bool operator<(Fraction x, Fraction y) { return compare(x, y) < 0; }

inline bool operator<=(Fraction x, Fraction y) { return compare(x, y) <= 0; }

inline bool operator==(Fraction x, Fraction y) { return compare(x, y) == 0; }

inline bool operator!=(Fraction x, Fraction y) { return compare(x, y) != 0; }

inline bool operator>(Fraction x, Fraction y) { return compare(x, y) > 0; }

inline bool operator>=(Fraction x, Fraction y) { return compare(x, y) >= 0; }

inline Fraction operator*(Fraction x, Fraction y) {
  return Fraction(x.num * y.num, x.den * y.den);
}

} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_FRACTION_H
