//===- Barvinok.h - Barvinok's Algorithm -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functions and classes for Barvinok's algorithm in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_BARVINOK_H
#define MLIR_ANALYSIS_PRESBURGER_BARVINOK_H

#include "mlir/Analysis/Presburger/Fraction.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include "mlir/Analysis/Presburger/Utils.h"
#include "mlir/Support/LogicalResult.h"
#include <optional>

namespace mlir {
namespace presburger {

// The H (inequality) representation of both general
// polyhedra and cones specifically is an integer relation.
using PolyhedronH = IntegerRelation;
using ConeH = PolyhedronH;

// The V (generator) representation of both general
// polyhedra and cones specifically is simply a matrix
// whose rows are the generators.
using PolyhedronV = Matrix<MPInt>;
using ConeV = PolyhedronV;

// A parametric point is a vector, each of whose elements
// is an affine function of n parameters. Each row
// in the matrix represents the affine function and
// has n+1 elements.
using ParamPoint = Matrix<Fraction>;

// A point is simply a vector.
using Point = SmallVector<Fraction>;

// A class to describe the type of generating function
// used to enumerate the integer points in a polytope.
// Consists of a set of terms, where the ith term has
// * a sign, ±1, stored in `signs[i]`
// * a numerator, of the form x^{n},
//      where n, stored in `numerators[i]`,
//      is a parametric point (a vertex).
// * a denominator, of the form (1 - x^{d1})...(1 - x^{dn}),
//      where each dj, stored in `denominators[i][j]`,
//      is a vector (a generator).
class GeneratingFunction {
public:
  GeneratingFunction(SmallVector<int, 16> signs, std::vector<ParamPoint> nums,
                     std::vector<std::vector<Point>> dens)
      : signs(signs), numerators(nums), denominators(dens) {}

  // Find the number of parameters involved in the function
  // from the dimensionality of the affine functions.
  unsigned getNumParams() {
    for (auto term : numerators)
      // The number of elements in the affine function is
      // one more than the number of parameters.
      return (term.getNumColumns() - 1);
    // The polynomial can be treated as having any number
    // of parameters.
    return -1;
  }

  GeneratingFunction operator+(const GeneratingFunction &gf) {
    bool sameNumParams = (getNumParams() == -1) || (gf.getNumParams() == -1) ||
                         (getNumParams() == gf.getNumParams());
    assert(
        sameNumParams &&
        "two generators with different numbers of parameters cannot be added!");
    signs.append(gf.signs);
    numerators.insert(numerators.end(), gf.numerators.begin(),
                      gf.numerators.end());
    denominators.insert(denominators.end(), gf.denominators.begin(),
                        gf.denominators.end());
    return *this;
  }

  llvm::raw_ostream &print(llvm::raw_ostream &os) const {
    for (unsigned i = 0; i < signs.size(); i++) {
      if (signs[i] == 1)
        os << " + ";
      else
        os << " - ";

      os << "(x^[";
      for (unsigned j = 0; j < numerators[i].size() - 1; j++)
        os << numerators[i][j] << ",";
      os << numerators[i].back() << "])/";

      for (Point den : denominators[i]) {
        os << "(x^[";
        for (unsigned j = 0; j < den.size() - 1; j++)
          os << den[j] << ",";
        os << den[den.size() - 1] << "])";
      }
    }
    return os;
  }

  SmallVector<int, 8> signs;
  std::vector<ParamPoint> numerators;
  std::vector<std::vector<Point>> denominators;
};

// A class to describe the quasi-polynomials obtained by
// substituting the unit vector in the type of generating
// function described above.
// Consists of a set of terms.
// The ith term is a constant `coefficients[i]`, multiplied
// by the product of a set of affine functions on n parameters.
class QuasiPolynomial {
public:
  QuasiPolynomial(SmallVector<Fraction> coeffs = {},
                  std::vector<std::vector<SmallVector<Fraction>>> aff = {})
      : coefficients(coeffs), affine(aff) {}

  QuasiPolynomial(Fraction cons) : coefficients({cons}), affine({{}}) {}

  QuasiPolynomial(QuasiPolynomial const &) = default;

  SmallVector<Fraction> coefficients;
  std::vector<std::vector<SmallVector<Fraction>>> affine;

  // Find the number of parameters involved in the polynomial
  // from the dimensionality of the affine functions.
  unsigned getNumParams() {
    // Find the first term which involves some affine function.
    for (auto term : affine) {
      if (term.size() == 0)
        continue;
      // The number of elements in the affine function is
      // one more than the number of parameters.
      return (term[0].size() - 1);
    }
    // The polynomial can be treated as having any number
    // of parameters.
    return -1;
  }

  QuasiPolynomial operator+(const QuasiPolynomial &x) {
    bool sameNumParams = (getNumParams() == -1) || (x.getNumParams() == -1) ||
                         (getNumParams() == x.getNumParams());
    assert(sameNumParams && "two quasi-polynomials with different numbers of "
                            "parameters cannot be added!");
    coefficients.append(x.coefficients);
    affine.insert(affine.end(), x.affine.begin(), x.affine.end());
    return *this;
  }

  QuasiPolynomial operator-(const QuasiPolynomial &x) {
    bool sameNumParams = (getNumParams() == -1) || (x.getNumParams() == -1) ||
                         (getNumParams() == x.getNumParams());
    assert(sameNumParams && "two quasi-polynomials with different numbers of "
                            "parameters cannot be subtracted!");
    QuasiPolynomial qp(x.coefficients, x.affine);
    for (unsigned i = 0; i < x.coefficients.size(); i++)
      qp.coefficients[i] = -qp.coefficients[i];
    return (*this + qp);
  }

  QuasiPolynomial operator*(const QuasiPolynomial &x) {
    bool sameNumParams = (getNumParams() == -1) || (x.getNumParams() == -1) ||
                         (getNumParams() == x.getNumParams());
    assert(sameNumParams && "two quasi-polynomials with different numbers of "
                            "parameters cannot be multiplied!");
    QuasiPolynomial qp();
    std::vector<SmallVector<Fraction>> product;
    for (unsigned i = 0; i < coefficients.size(); i++) {
      for (unsigned j = 0; j < x.coefficients.size(); j++) {
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
    for (unsigned i = 0; i < coefficients.size(); i++)
      coefficients[i] = coefficients[i] / x;
    return *this;
  };

  // Removes terms which evaluate to zero from the expression.
  QuasiPolynomial reduce() {
    SmallVector<Fraction> newCoeffs({});
    std::vector<std::vector<SmallVector<Fraction>>> newAffine({});
    bool prodIsNonz, sumIsNonz;
    for (unsigned i = 0; i < coefficients.size(); i++) {
      prodIsNonz = true;
      for (unsigned j = 0; j < affine[i].size(); j++) {
        sumIsNonz = false;
        for (unsigned k = 0; k < affine[i][j].size(); k++)
          if (affine[i][j][k] != Fraction(0, 1))
            sumIsNonz = true;
        if (sumIsNonz == false)
          prodIsNonz = false;
      }
      if (prodIsNonz == true && coefficients[i] != Fraction(0, 1)) {
        newCoeffs.append({coefficients[i]});
        newAffine.push_back({affine[i]});
      }
    }
    return QuasiPolynomial(newCoeffs, newAffine);
  }
};

} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_BARVINOK_H