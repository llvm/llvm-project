//===- Polynomial.h - A data class for polynomials --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_POLYNOMIAL_IR_POLYNOMIAL_H_
#define MLIR_DIALECT_POLYNOMIAL_IR_POLYNOMIAL_H_

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {

class MLIRContext;

namespace polynomial {

/// This restricts statically defined polynomials to have at most 64-bit
/// coefficients. This may be relaxed in the future, but it seems unlikely one
/// would want to specify 128-bit polynomials statically in the source code.
constexpr unsigned apintBitWidth = 64;

/// A class representing a monomial of a single-variable polynomial with integer
/// coefficients.
class Monomial {
public:
  Monomial(int64_t coeff, uint64_t expo)
      : coefficient(apintBitWidth, coeff), exponent(apintBitWidth, expo) {}

  Monomial(const APInt &coeff, const APInt &expo)
      : coefficient(coeff), exponent(expo) {}

  Monomial() : coefficient(apintBitWidth, 0), exponent(apintBitWidth, 0) {}

  bool operator==(const Monomial &other) const {
    return other.coefficient == coefficient && other.exponent == exponent;
  }
  bool operator!=(const Monomial &other) const {
    return other.coefficient != coefficient || other.exponent != exponent;
  }

  /// Monomials are ordered by exponent.
  bool operator<(const Monomial &other) const {
    return (exponent.ult(other.exponent));
  }

  friend ::llvm::hash_code hash_value(const Monomial &arg);

public:
  APInt coefficient;

  // Always unsigned
  APInt exponent;
};

/// A single-variable polynomial with integer coefficients.
///
/// Eg: x^1024 + x + 1
///
/// The symbols used as the polynomial's indeterminate don't matter, so long as
/// it is used consistently throughout the polynomial.
class Polynomial {
public:
  Polynomial() = delete;

  explicit Polynomial(ArrayRef<Monomial> terms) : terms(terms){};

  // Returns a Polynomial from a list of monomials.
  // Fails if two monomials have the same exponent.
  static FailureOr<Polynomial> fromMonomials(ArrayRef<Monomial> monomials);

  /// Returns a polynomial with coefficients given by `coeffs`. The value
  /// coeffs[i] is converted to a monomial with exponent i.
  static Polynomial fromCoefficients(ArrayRef<int64_t> coeffs);

  explicit operator bool() const { return !terms.empty(); }
  bool operator==(const Polynomial &other) const {
    return other.terms == terms;
  }
  bool operator!=(const Polynomial &other) const {
    return !(other.terms == terms);
  }

  // Prints polynomial to 'os'.
  void print(raw_ostream &os) const;
  void print(raw_ostream &os, ::llvm::StringRef separator,
             ::llvm::StringRef exponentiation) const;
  void dump() const;

  // Prints polynomial so that it can be used as a valid identifier
  std::string toIdentifier() const;

  unsigned getDegree() const;

  ArrayRef<Monomial> getTerms() const { return terms; }

  friend ::llvm::hash_code hash_value(const Polynomial &arg);

private:
  // The monomial terms for this polynomial.
  SmallVector<Monomial> terms;
};

// Make Polynomial hashable.
inline ::llvm::hash_code hash_value(const Polynomial &arg) {
  return ::llvm::hash_combine_range(arg.terms.begin(), arg.terms.end());
}

inline ::llvm::hash_code hash_value(const Monomial &arg) {
  return llvm::hash_combine(::llvm::hash_value(arg.coefficient),
                            ::llvm::hash_value(arg.exponent));
}

inline raw_ostream &operator<<(raw_ostream &os, const Polynomial &polynomial) {
  polynomial.print(os);
  return os;
}

} // namespace polynomial
} // namespace mlir

#endif // MLIR_DIALECT_POLYNOMIAL_IR_POLYNOMIAL_H_
