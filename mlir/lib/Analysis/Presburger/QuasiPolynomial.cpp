//===- QuasiPolynomial.cpp - Quasipolynomial Class --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/QuasiPolynomial.h"
#include "mlir/Analysis/Presburger/Fraction.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include "mlir/Analysis/Presburger/Utils.h"
#include "mlir/Support/LogicalResult.h"
#include <optional>

using namespace mlir;
using namespace presburger;

QuasiPolynomial::QuasiPolynomial(
    unsigned numParam, SmallVector<Fraction> coeffs,
    std::vector<std::vector<SmallVector<Fraction>>> aff)
    : numParam(numParam), coefficients(coeffs), affine(aff) {
  // For each term which involves at least one affine function,
  for (const std::vector<SmallVector<Fraction>> &term : affine) {
    if (term.size() == 0)
      continue;
    // the number of elements in each affine function is
    // one more than the number of parameters.
    for (const SmallVector<Fraction> &aff : term) {
      assert(aff.size() == numParam + 1 &&
             "dimensionality of affine functions does not match number of "
             "parameters!");
    }
  }
}

QuasiPolynomial QuasiPolynomial::operator+(const QuasiPolynomial &x) const {
  assert(numParam == x.getNumParams() &&
         "two quasi-polynomials with different numbers of parameters cannot "
         "be added!");
  SmallVector<Fraction> sumCoeffs = coefficients;
  sumCoeffs.append(x.coefficients);
  std::vector<std::vector<SmallVector<Fraction>>> sumAff = affine;
  sumAff.insert(sumAff.end(), x.affine.begin(), x.affine.end());
  return QuasiPolynomial(numParam, sumCoeffs, sumAff);
}

QuasiPolynomial QuasiPolynomial::operator-(const QuasiPolynomial &x) const {
  assert(numParam == x.getNumParams() &&
         "two quasi-polynomials with different numbers of parameters cannot "
         "be subtracted!");
  QuasiPolynomial qp(numParam, x.coefficients, x.affine);
  for (Fraction &coeff : qp.coefficients)
    coeff = -coeff;
  return *this + qp;
}

QuasiPolynomial QuasiPolynomial::operator*(const QuasiPolynomial &x) const {
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

QuasiPolynomial QuasiPolynomial::operator/(const Fraction x) const {
  assert(x != 0 && "division by zero!");
  QuasiPolynomial qp(*this);
  for (Fraction &coeff : qp.coefficients)
    coeff /= x;
  return qp;
}

// Removes terms which evaluate to zero from the expression.
QuasiPolynomial QuasiPolynomial::simplify() {
  SmallVector<Fraction> newCoeffs({});
  std::vector<std::vector<SmallVector<Fraction>>> newAffine({});
  for (unsigned i = 0, e = coefficients.size(); i < e; i++) {
    // A term is zero if its coefficient is zero, or
    if (coefficients[i] == Fraction(0, 1))
      continue;
    bool product_is_zero =
        // if any of the affine functions in the product
        llvm::any_of(affine[i], [](const SmallVector<Fraction> &affine_ij) {
          // has all its coefficients as zero.
          return llvm::all_of(affine_ij,
                              [](const Fraction &f) { return f == 0; });
        });
    if (product_is_zero)
      continue;
    newCoeffs.push_back(coefficients[i]);
    newAffine.push_back(affine[i]);
  }
  return QuasiPolynomial(numParam, newCoeffs, newAffine);
}