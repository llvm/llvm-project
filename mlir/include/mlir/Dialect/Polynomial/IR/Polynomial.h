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
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

class MLIRContext;

namespace polynomial {

/// This restricts statically defined polynomials to have at most 64-bit
/// coefficients. This may be relaxed in the future, but it seems unlikely one
/// would want to specify 128-bit polynomials statically in the source code.
constexpr unsigned apintBitWidth = 64;

template <class Derived, typename CoefficientType>
class MonomialBase {
public:
  MonomialBase(const CoefficientType &coeff, const APInt &expo)
      : coefficient(coeff), exponent(expo) {}
  virtual ~MonomialBase() = default;

  const CoefficientType &getCoefficient() const { return coefficient; }
  CoefficientType &getMutableCoefficient() { return coefficient; }
  const APInt &getExponent() const { return exponent; }
  void setCoefficient(const CoefficientType &coeff) { coefficient = coeff; }
  void setExponent(const APInt &exp) { exponent = exp; }

  bool operator==(const MonomialBase &other) const {
    return other.coefficient == coefficient && other.exponent == exponent;
  }
  bool operator!=(const MonomialBase &other) const {
    return other.coefficient != coefficient || other.exponent != exponent;
  }

  /// Monomials are ordered by exponent.
  bool operator<(const MonomialBase &other) const {
    return (exponent.ult(other.exponent));
  }

  Derived add(const Derived &other) {
    assert(exponent == other.exponent);
    CoefficientType newCoeff = coefficient + other.coefficient;
    Derived result;
    result.setCoefficient(newCoeff);
    result.setExponent(exponent);
    return result;
  }

  virtual bool isMonic() const = 0;
  virtual void
  coefficientToString(llvm::SmallString<16> &coeffString) const = 0;

  template <class D, typename T>
  friend ::llvm::hash_code hash_value(const MonomialBase<D, T> &arg);

protected:
  CoefficientType coefficient;
  APInt exponent;
};

/// A class representing a monomial of a single-variable polynomial with integer
/// coefficients.
class IntMonomial : public MonomialBase<IntMonomial, APInt> {
public:
  IntMonomial(int64_t coeff, uint64_t expo)
      : MonomialBase(APInt(apintBitWidth, coeff), APInt(apintBitWidth, expo)) {}

  IntMonomial()
      : MonomialBase(APInt(apintBitWidth, 0), APInt(apintBitWidth, 0)) {}

  ~IntMonomial() override = default;

  bool isMonic() const override { return coefficient == 1; }

  void coefficientToString(llvm::SmallString<16> &coeffString) const override {
    coefficient.toStringSigned(coeffString);
  }
};

/// A class representing a monomial of a single-variable polynomial with integer
/// coefficients.
class FloatMonomial : public MonomialBase<FloatMonomial, APFloat> {
public:
  FloatMonomial(double coeff, uint64_t expo)
      : MonomialBase(APFloat(coeff), APInt(apintBitWidth, expo)) {}

  FloatMonomial() : MonomialBase(APFloat((double)0), APInt(apintBitWidth, 0)) {}

  ~FloatMonomial() override = default;

  bool isMonic() const override { return coefficient == APFloat(1.0); }

  void coefficientToString(llvm::SmallString<16> &coeffString) const override {
    coefficient.toString(coeffString);
  }
};

template <class Derived, typename Monomial>
class PolynomialBase {
public:
  PolynomialBase() = delete;

  explicit PolynomialBase(ArrayRef<Monomial> terms) : terms(terms) {};

  explicit operator bool() const { return !terms.empty(); }
  bool operator==(const PolynomialBase &other) const {
    return other.terms == terms;
  }
  bool operator!=(const PolynomialBase &other) const {
    return !(other.terms == terms);
  }

  void print(raw_ostream &os, ::llvm::StringRef separator,
             ::llvm::StringRef exponentiation) const {
    bool first = true;
    for (const Monomial &term : getTerms()) {
      if (first) {
        first = false;
      } else {
        os << separator;
      }
      std::string coeffToPrint;
      if (term.isMonic() && term.getExponent().uge(1)) {
        coeffToPrint = "";
      } else {
        llvm::SmallString<16> coeffString;
        term.coefficientToString(coeffString);
        coeffToPrint = coeffString.str();
      }

      if (term.getExponent() == 0) {
        os << coeffToPrint;
      } else if (term.getExponent() == 1) {
        os << coeffToPrint << "x";
      } else {
        llvm::SmallString<16> expString;
        term.getExponent().toStringSigned(expString);
        os << coeffToPrint << "x" << exponentiation << expString;
      }
    }
  }

  Derived add(const Derived &other) {
    SmallVector<Monomial> newTerms;
    auto it1 = terms.begin();
    auto it2 = other.terms.begin();
    while (it1 != terms.end() || it2 != other.terms.end()) {
      if (it1 == terms.end()) {
        newTerms.emplace_back(*it2);
        it2++;
        continue;
      }

      if (it2 == other.terms.end()) {
        newTerms.emplace_back(*it1);
        it1++;
        continue;
      }

      while (it1->getExponent().ult(it2->getExponent())) {
        newTerms.emplace_back(*it1);
        it1++;
        if (it1 == terms.end())
          break;
      }

      while (it2->getExponent().ult(it1->getExponent())) {
        newTerms.emplace_back(*it2);
        it2++;
        if (it2 == terms.end())
          break;
      }

      newTerms.emplace_back(it1->add(*it2));
      it1++;
      it2++;
    }
    return Derived(newTerms);
  }

  // Prints polynomial to 'os'.
  void print(raw_ostream &os) const { print(os, " + ", "**"); }

  void dump() const;

  // Prints polynomial so that it can be used as a valid identifier
  std::string toIdentifier() const {
    std::string result;
    llvm::raw_string_ostream os(result);
    print(os, "_", "");
    return os.str();
  }

  unsigned getDegree() const {
    return terms.back().getExponent().getZExtValue();
  }

  ArrayRef<Monomial> getTerms() const { return terms; }

  template <class D, typename T>
  friend ::llvm::hash_code hash_value(const PolynomialBase<D, T> &arg);

private:
  // The monomial terms for this polynomial.
  SmallVector<Monomial> terms;
};

/// A single-variable polynomial with integer coefficients.
///
/// Eg: x^1024 + x + 1
class IntPolynomial : public PolynomialBase<IntPolynomial, IntMonomial> {
public:
  explicit IntPolynomial(ArrayRef<IntMonomial> terms) : PolynomialBase(terms) {}

  // Returns a Polynomial from a list of monomials.
  // Fails if two monomials have the same exponent.
  static FailureOr<IntPolynomial>
  fromMonomials(ArrayRef<IntMonomial> monomials);

  /// Returns a polynomial with coefficients given by `coeffs`. The value
  /// coeffs[i] is converted to a monomial with exponent i.
  static IntPolynomial fromCoefficients(ArrayRef<int64_t> coeffs);
};

/// A single-variable polynomial with double coefficients.
///
/// Eg: 1.0 x^1024 + 3.5 x + 1e-05
class FloatPolynomial : public PolynomialBase<FloatPolynomial, FloatMonomial> {
public:
  explicit FloatPolynomial(ArrayRef<FloatMonomial> terms)
      : PolynomialBase(terms) {}

  // Returns a Polynomial from a list of monomials.
  // Fails if two monomials have the same exponent.
  static FailureOr<FloatPolynomial>
  fromMonomials(ArrayRef<FloatMonomial> monomials);

  /// Returns a polynomial with coefficients given by `coeffs`. The value
  /// coeffs[i] is converted to a monomial with exponent i.
  static FloatPolynomial fromCoefficients(ArrayRef<double> coeffs);
};

// Make Polynomials hashable.
template <class D, typename T>
inline ::llvm::hash_code hash_value(const PolynomialBase<D, T> &arg) {
  return ::llvm::hash_combine_range(arg.terms.begin(), arg.terms.end());
}

template <class D, typename T>
inline ::llvm::hash_code hash_value(const MonomialBase<D, T> &arg) {
  return llvm::hash_combine(::llvm::hash_value(arg.coefficient),
                            ::llvm::hash_value(arg.exponent));
}

template <class D, typename T>
inline raw_ostream &operator<<(raw_ostream &os,
                               const PolynomialBase<D, T> &polynomial) {
  polynomial.print(os);
  return os;
}

} // namespace polynomial
} // namespace mlir

#endif // MLIR_DIALECT_POLYNOMIAL_IR_POLYNOMIAL_H_
