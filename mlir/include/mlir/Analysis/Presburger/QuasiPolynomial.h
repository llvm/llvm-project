//===- QuasiPolynomial.h - QuasiPolynomial Class ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definition of the QuasiPolynomial class for Barvinok's algorithm,
// which represents a single-valued function on a set of parameters.
// It is an expression of the form
// f(x) = \sum_i c_i * \prod_j ⌊g_{ij}(x)⌋
// where c_i \in Q and
// g_{ij} : Q^d -> Q are affine functionals over d parameters.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_QUASIPOLYNOMIAL_H
#define MLIR_ANALYSIS_PRESBURGER_QUASIPOLYNOMIAL_H

#include "mlir/Analysis/Presburger/Fraction.h"
#include "mlir/Analysis/Presburger/Matrix.h"

namespace mlir {
namespace presburger {

// A class to describe quasi-polynomials.
// A quasipolynomial consists of a set of terms.
// The ith term is a constant `coefficients[i]`, multiplied
// by the product of a set of affine functions on n parameters.
// Represents functions f : Q^n -> Q of the form
//
// f(x) = \sum_i c_i * \prod_j ⌊g_{ij}(x)⌋
//
// where c_i \in Q and
// g_{ij} : Q^n -> Q are affine functionals.
class QuasiPolynomial {
public:
  QuasiPolynomial(unsigned numParam, SmallVector<Fraction> coeffs = {},
                  std::vector<std::vector<SmallVector<Fraction>>> aff = {});

  unsigned getNumParams() const;
  SmallVector<Fraction> getCoefficients() const;
  std::vector<std::vector<SmallVector<Fraction>>> getAffine() const;

  QuasiPolynomial operator+(const QuasiPolynomial &x) const {
    assert(numParam == x.getNumParams() &&
           "two quasi-polynomials with different numbers of parameters cannot "
           "be added!");
    SmallVector<Fraction> sumCoeffs = coefficients;
    sumCoeffs.append(x.coefficients);
    std::vector<std::vector<SmallVector<Fraction>>> sumAff = affine;
    sumAff.insert(sumAff.end(), x.affine.begin(), x.affine.end());
    return QuasiPolynomial(numParam, sumCoeffs, sumAff);
  }

  QuasiPolynomial operator-(const QuasiPolynomial &x) const {
    assert(numParam == x.getNumParams() &&
           "two quasi-polynomials with different numbers of parameters cannot "
           "be subtracted!");
    QuasiPolynomial qp(numParam, x.coefficients, x.affine);
    for (Fraction &coeff : qp.coefficients)
      coeff = -coeff;
    return *this + qp;
  }

  QuasiPolynomial operator*(const QuasiPolynomial &x) const {
    assert(numParam == x.getNumParams() &&
           "two quasi-polynomials with different numbers of "
           "parameters cannot be multiplied!");

    SmallVector<Fraction> coeffs;
    coeffs.reserve(coefficients.size() * x.coefficients.size());
    for (const Fraction &coeff : coefficients) {
      for (const Fraction &xcoeff : x.coefficients) {
        coeffs.push_back(coeff * xcoeff);
      }
    }

    std::vector<SmallVector<Fraction>> product;
    std::vector<std::vector<SmallVector<Fraction>>> aff;
    aff.reserve(affine.size() * x.affine.size());
    for (const std::vector<SmallVector<Fraction>> &term : affine) {
      for (const std::vector<SmallVector<Fraction>> &xterm : x.affine) {
        product.clear();
        product.insert(product.end(), term.begin(), term.end());
        product.insert(product.end(), xterm.begin(), xterm.end());
        aff.push_back(product);
      }
    }

    return QuasiPolynomial(numParam, coeffs, aff);
  }

  QuasiPolynomial operator/(const Fraction x) const {
    assert(x != 0 && "division by zero!");
    QuasiPolynomial qp(*this);
    for (Fraction &coeff : qp.coefficients)
      coeff /= x;
    return qp;
  };

  // Removes terms which evaluate to zero from the expression.
  QuasiPolynomial simplify();

private:
  unsigned numParam;
  SmallVector<Fraction> coefficients;
  std::vector<std::vector<SmallVector<Fraction>>> affine;
};

} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_QUASIPOLYNOMIAL_H