//===- QuasiPolynomial.cpp - Quasipolynomial Class --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/QuasiPolynomial.h"
#include "mlir/Analysis/Presburger/Fraction.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include "mlir/Analysis/Presburger/Utils.h"

using namespace mlir;
using namespace presburger;

QuasiPolynomial::QuasiPolynomial(
    unsigned numVars, SmallVector<Fraction> coeffs,
    std::vector<std::vector<SmallVector<Fraction>>> aff)
    : PresburgerSpace(/*numDomain=*/numVars, /*numRange=*/1, /*numSymbols=*/0,
                      /*numLocals=*/0),
      coefficients(coeffs), affine(aff) {
#ifndef NDEBUG
  // For each term which involves at least one affine function,
  for (const std::vector<SmallVector<Fraction>> &term : affine) {
    if (term.empty())
      continue;
    // the number of elements in each affine function is
    // one more than the number of symbols.
    for (const SmallVector<Fraction> &aff : term) {
      assert(aff.size() == getNumInputs() + 1 &&
             "dimensionality of affine functions does not match number of "
             "symbols!");
    }
  }
#endif // NDEBUG
}

QuasiPolynomial QuasiPolynomial::operator+(const QuasiPolynomial &x) const {
  assert(getNumInputs() == x.getNumInputs() &&
         "two quasi-polynomials with different numbers of symbols cannot "
         "be added!");
  SmallVector<Fraction> sumCoeffs = coefficients;
  sumCoeffs.append(x.coefficients);
  std::vector<std::vector<SmallVector<Fraction>>> sumAff = affine;
  sumAff.insert(sumAff.end(), x.affine.begin(), x.affine.end());
  return QuasiPolynomial(getNumInputs(), sumCoeffs, sumAff);
}

QuasiPolynomial QuasiPolynomial::operator-(const QuasiPolynomial &x) const {
  assert(getNumInputs() == x.getNumInputs() &&
         "two quasi-polynomials with different numbers of symbols cannot "
         "be subtracted!");
  QuasiPolynomial qp(getNumInputs(), x.coefficients, x.affine);
  for (Fraction &coeff : qp.coefficients)
    coeff = -coeff;
  return *this + qp;
}

QuasiPolynomial QuasiPolynomial::operator*(const QuasiPolynomial &x) const {
  assert(getNumInputs() == x.getNumInputs() &&
         "two quasi-polynomials with different numbers of "
         "symbols cannot be multiplied!");

  SmallVector<Fraction> coeffs;
  coeffs.reserve(coefficients.size() * x.coefficients.size());
  for (const Fraction &coeff : coefficients)
    for (const Fraction &xcoeff : x.coefficients)
      coeffs.push_back(coeff * xcoeff);

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

  return QuasiPolynomial(getNumInputs(), coeffs, aff);
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
  return QuasiPolynomial(getNumInputs(), newCoeffs, newAffine);
}
