//===- Polynomial.h - A storage class for polynomial types --------*- C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef INCLUDE_MLIR_DIALECT_POLYNOMIAL_IR_POLYNOMIAL_H_
#define INCLUDE_MLIR_DIALECT_POLYNOMIAL_IR_POLYNOMIAL_H_

#include <utility>

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"

namespace mlir {

class MLIRContext;

namespace polynomial {

// This restricts statically defined polynomials to have at most 64-bit
// coefficients. This may be relaxed in the future, but it seems unlikely one
// would want to specify 128-bit polynomials statically in the source code.
constexpr unsigned apintBitWidth = 64;

namespace detail {
struct PolynomialStorage;
} // namespace detail

class Monomial {
public:
  Monomial(int64_t coeff, uint64_t expo)
      : coefficient(apintBitWidth, coeff), exponent(apintBitWidth, expo) {}

  Monomial(APInt coeff, APInt expo)
      : coefficient(std::move(coeff)), exponent(std::move(expo)) {}

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

  // Prints polynomial to 'os'.
  void print(raw_ostream &os) const;

  friend ::llvm::hash_code hash_value(Monomial arg);

public:
  APInt coefficient;

  // Always unsigned
  APInt exponent;
};

/// A single-variable polynomial with integer coefficients. Polynomials are
/// immutable and uniqued.
///
/// Eg: x^1024 + x + 1
///
/// The symbols used as the polynomial's indeterminate don't matter, so long as
/// it is used consistently throughout the polynomial.
class Polynomial {
public:
  using ImplType = detail::PolynomialStorage;

  constexpr Polynomial() = default;
  explicit Polynomial(ImplType *terms) : terms(terms) {}

  static Polynomial fromMonomials(ArrayRef<Monomial> monomials,
                                  MLIRContext *context);
  /// Returns a polynomial with coefficients given by `coeffs`
  static Polynomial fromCoefficients(ArrayRef<int64_t> coeffs,
                                     MLIRContext *context);

  MLIRContext *getContext() const;

  explicit operator bool() const { return terms != nullptr; }
  bool operator==(Polynomial other) const { return other.terms == terms; }
  bool operator!=(Polynomial other) const { return !(other.terms == terms); }

  // Prints polynomial to 'os'.
  void print(raw_ostream &os) const;
  void print(raw_ostream &os, const std::string &separator,
             const std::string &exponentiation) const;
  void dump() const;

  // Prints polynomial so that it can be used as a valid identifier
  std::string toIdentifier() const;

  // A polynomial's terms are canonically stored in order of increasing degree.
  ArrayRef<Monomial> getTerms() const;

  unsigned getDegree() const;

  friend ::llvm::hash_code hash_value(Polynomial arg);

private:
  ImplType *terms{nullptr};
};

// Make Polynomial hashable.
inline ::llvm::hash_code hash_value(Polynomial arg) {
  return ::llvm::hash_value(arg.terms);
}

inline ::llvm::hash_code hash_value(Monomial arg) {
  return ::llvm::hash_value(arg.coefficient) ^ ::llvm::hash_value(arg.exponent);
}

inline raw_ostream &operator<<(raw_ostream &os, Polynomial polynomial) {
  polynomial.print(os);
  return os;
}

} // namespace polynomial
} // namespace mlir

namespace llvm {

// Polynomials hash just like pointers
template <>
struct DenseMapInfo<mlir::polynomial::Polynomial> {
  static mlir::polynomial::Polynomial getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::polynomial::Polynomial(
        static_cast<mlir::polynomial::Polynomial::ImplType *>(pointer));
  }
  static mlir::polynomial::Polynomial getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::polynomial::Polynomial(
        static_cast<mlir::polynomial::Polynomial::ImplType *>(pointer));
  }
  static unsigned getHashValue(mlir::polynomial::Polynomial val) {
    return mlir::polynomial::hash_value(val);
  }
  static bool isEqual(mlir::polynomial::Polynomial lhs,
                      mlir::polynomial::Polynomial rhs) {
    return lhs == rhs;
  }
};

} // namespace llvm

#endif // INCLUDE_MLIR_DIALECT_POLYNOMIAL_IR_POLYNOMIAL_H_
