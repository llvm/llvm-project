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
  // Find the first term which involves some affine function.
  for (const std::vector<SmallVector<Fraction>> &term : affine) {
    if (term.size() == 0)
      continue;
    // The number of elements in the affine function is
    // one more than the number of parameters.
    assert(term[0].size() - 1 == numParam &&
           "dimensionality of affine functions does not match number of "
           "parameters!");
  }
}

// Find the number of parameters involved in the polynomial.
unsigned QuasiPolynomial::getNumParams() { return numParam; }

SmallVector<Fraction> QuasiPolynomial::getCoefficients() {
  return coefficients;
}

std::vector<std::vector<SmallVector<Fraction>>> QuasiPolynomial::getAffine() {
  return affine;
}

// Removes terms which evaluate to zero from the expression.
QuasiPolynomial QuasiPolynomial::simplify() {
  SmallVector<Fraction> newCoeffs({});
  std::vector<std::vector<SmallVector<Fraction>>> newAffine({});
  for (unsigned i = 0, e = coefficients.size(); i < e; i++) {
    // A term is zero if its coefficient is zero, or
    if (coefficients[i] == Fraction(0, 1) ||
        // if any of the affine functions in the product
        llvm::any_of(affine[i], [](SmallVector<Fraction> affine_ij) {
          // has all its coefficients as zero.
          return llvm::all_of(affine_ij, [](Fraction f) { return f == 0; });
        }))
      continue;
    newCoeffs.append({coefficients[i]});
    newAffine.push_back({affine[i]});
  }
  return QuasiPolynomial(numParam, newCoeffs, newAffine);
}