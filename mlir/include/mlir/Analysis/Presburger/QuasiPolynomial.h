//===- QuasiPolynomial.h - Quasipolynomial Class ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definition of the QuasiPolynomial class for Barvinok's algorithm,
// which represents a single-valued function on a set of parameters.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_QUASIPOLYNOMIAL_H
#define MLIR_ANALYSIS_PRESBURGER_QUASIPOLYNOMIAL_H

#include "mlir/Analysis/Presburger/Fraction.h"
#include "mlir/Analysis/Presburger/Matrix.h"

namespace mlir {
namespace presburger {

// A class to describe the quasi-polynomials obtained by
// substituting the unit vector in the type of generating
// function described above.
// Consists of a set of terms.
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

  unsigned getNumParams();
  SmallVector<Fraction> getCoefficients();
  std::vector<std::vector<SmallVector<Fraction>>> getAffine();

  QuasiPolynomial operator+(QuasiPolynomial &x) {
    assert(numParam == x.getNumParams() &&
           "two quasi-polynomials with different numbers of parameters cannot "
           "be added!");
    coefficients.append(x.coefficients);
    affine.insert(affine.end(), x.affine.begin(), x.affine.end());
    return *this;
  }

  QuasiPolynomial operator-(QuasiPolynomial &x) {
    assert(numParam == x.getNumParams() &&
           "two quasi-polynomials with different numbers of parameters cannot "
           "be subtracted!");
    QuasiPolynomial qp(numParam, x.coefficients, x.affine);
    for (Fraction &coeff : qp.coefficients)
      coeff = -coeff;
    return *this + qp;
  }

  QuasiPolynomial operator*(QuasiPolynomial &x) {
    assert(numParam = x.getNumParams() &&
                      "two quasi-polynomials with different numbers of "
                      "parameters cannot be multiplied!");
    QuasiPolynomial qp(numParam);
    std::vector<SmallVector<Fraction>> product;
    for (unsigned i = 0, e = coefficients.size(); i < e; i++) {
      for (unsigned j = 0, e = x.coefficients.size(); j < e; j++) {
        qp.coefficients.append({coefficients[i] * x.coefficients[j]});

        product.clear();
        product.insert(product.end(), affine[i].begin(), affine[i].end());
        product.insert(product.end(), x.affine[j].begin(), x.affine[j].end());

        qp.affine.push_back(product);
      }
    }

    return qp;
  }

  QuasiPolynomial operator/(Fraction x) {
    assert(x != 0 && "division by zero!");
    for (Fraction &coeff : coefficients)
      coeff /= x;
    return *this;
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