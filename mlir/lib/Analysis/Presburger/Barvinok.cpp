//===- Barvinok.cpp - Barvinok's Algorithm ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Barvinok.h"
#include "llvm/ADT/Sequence.h"
#include <algorithm>

using namespace mlir;
using namespace presburger;
using namespace mlir::presburger::detail;

/// Assuming that the input cone is pointed at the origin,
/// converts it to its dual in V-representation.
/// Essentially we just remove the all-zeroes constant column.
ConeV mlir::presburger::detail::getDual(ConeH cone) {
  unsigned numIneq = cone.getNumInequalities();
  unsigned numVar = cone.getNumCols() - 1;
  ConeV dual(numIneq, numVar, 0, 0);
  // Assuming that an inequality of the form
  // a1*x1 + ... + an*xn + b ≥ 0
  // is represented as a row [a1, ..., an, b]
  // and that b = 0.

  for (auto i : llvm::seq<int>(0, numIneq)) {
    assert(cone.atIneq(i, numVar) == 0 &&
           "H-representation of cone is not centred at the origin!");
    for (unsigned j = 0; j < numVar; ++j) {
      dual.at(i, j) = cone.atIneq(i, j);
    }
  }

  // Now dual is of the form [ [a1, ..., an] , ... ]
  // which is the V-representation of the dual.
  return dual;
}

/// Converts a cone in V-representation to the H-representation
/// of its dual, pointed at the origin (not at the original vertex).
/// Essentially adds a column consisting only of zeroes to the end.
ConeH mlir::presburger::detail::getDual(ConeV cone) {
  unsigned rows = cone.getNumRows();
  unsigned columns = cone.getNumColumns();
  ConeH dual = defineHRep(columns);
  // Add a new column (for constants) at the end.
  // This will be initialized to zero.
  cone.insertColumn(columns);

  for (unsigned i = 0; i < rows; ++i)
    dual.addInequality(cone.getRow(i));

  // Now dual is of the form [ [a1, ..., an, 0] , ... ]
  // which is the H-representation of the dual.
  return dual;
}

/// Find the index of a cone in V-representation.
MPInt mlir::presburger::detail::getIndex(ConeV cone) {
  if (cone.getNumRows() > cone.getNumColumns())
    return MPInt(0);

  return cone.determinant();
}

/// Compute the generating function for a unimodular cone.
/// This consists of a single term of the form
/// sign * x^num / prod_j (1 - x^den_j)
///
/// sign is either +1 or -1.
/// den_j is defined as the set of generators of the cone.
/// num is computed by expressing the vertex as a weighted
/// sum of the generators, and then taking the floor of the
/// coefficients.
GeneratingFunction mlir::presburger::detail::unimodularConeGeneratingFunction(
    ParamPoint vertex, int sign, ConeH cone) {
  // Consider a cone with H-representation [0  -1].
  //                                       [-1 -2]
  // Let the vertex be given by the matrix [ 2  2   0], with 2 params.
  //                                       [-1 -1/2 1]

  // `cone` must be unimodular.
  assert(getIndex(getDual(cone)) == 1 && "input cone is not unimodular!");

  unsigned numVar = cone.getNumVars();
  unsigned numIneq = cone.getNumInequalities();

  // Thus its ray matrix, U, is the inverse of the
  // transpose of its inequality matrix, `cone`.
  // The last column of the inequality matrix is null,
  // so we remove it to obtain a square matrix.
  FracMatrix transp = FracMatrix(cone.getInequalities()).transpose();
  transp.removeRow(numVar);

  FracMatrix generators(numVar, numIneq);
  transp.determinant(/*inverse=*/&generators); // This is the U-matrix.
  // Thus the generators are given by U = [2  -1].
  //                                      [-1  0]

  // The powers in the denominator of the generating
  // function are given by the generators of the cone,
  // i.e., the rows of the matrix U.
  std::vector<Point> denominator(numIneq);
  ArrayRef<Fraction> row;
  for (auto i : llvm::seq<int>(0, numVar)) {
    row = generators.getRow(i);
    denominator[i] = Point(row);
  }

  // The vertex is v \in Z^{d x (n+1)}
  // We need to find affine functions of parameters λ_i(p)
  // such that v = Σ λ_i(p)*u_i,
  // where u_i are the rows of U (generators)
  // The λ_i are given by the columns of Λ = v^T U^{-1}, and
  // we have transp = U^{-1}.
  // Then the exponent in the numerator will be
  // Σ -floor(-λ_i(p))*u_i.
  // Thus we store the (exponent of the) numerator as the affine function -Λ,
  // since the generators u_i are already stored as the exponent of the
  // denominator. Note that the outer -1 will have to be accounted for, as it is
  // not stored. See end for an example.

  unsigned numColumns = vertex.getNumColumns();
  unsigned numRows = vertex.getNumRows();
  ParamPoint numerator(numColumns, numRows);
  SmallVector<Fraction> ithCol(numRows);
  for (auto i : llvm::seq<int>(0, numColumns)) {
    for (auto j : llvm::seq<int>(0, numRows))
      ithCol[j] = vertex(j, i);
    numerator.setRow(i, transp.preMultiplyWithRow(ithCol));
    numerator.negateRow(i);
  }
  // Therefore Λ will be given by [ 1    0 ] and the negation of this will be
  //                              [ 1/2 -1 ]
  //                              [ -1  -2 ]
  // stored as the numerator.
  // Algebraically, the numerator exponent is
  // [ -2 ⌊ - N - M/2 + 1 ⌋ + 1 ⌊ 0 + M + 2 ⌋ ] -> first  COLUMN of U is [2, -1]
  // [  1 ⌊ - N - M/2 + 1 ⌋ + 0 ⌊ 0 + M + 2 ⌋ ] -> second COLUMN of U is [-1, 0]

  return GeneratingFunction(numColumns - 1, SmallVector<int>(1, sign),
                            std::vector({numerator}),
                            std::vector({denominator}));
}

/// We use an iterative procedure to find a vector not orthogonal
/// to a given set, ignoring the null vectors.
/// Let the inputs be {x_1, ..., x_k}, all vectors of length n.
///
/// In the following,
/// vs[:i] means the elements of vs up to and including the i'th one,
/// <vs, us> means the dot product of vs and us,
/// vs ++ [v] means the vector vs with the new element v appended to it.
///
/// We proceed iteratively; for steps d = 0, ... n-1, we construct a vector
/// which is not orthogonal to any of {x_1[:d], ..., x_n[:d]}, ignoring
/// the null vectors.
/// At step d = 0, we let vs = [1]. Clearly this is not orthogonal to
/// any vector in the set {x_1[0], ..., x_n[0]}, except the null ones,
/// which we ignore.
/// At step d > 0 , we need a number v
/// s.t. <x_i[:d], vs++[v]> != 0 for all i.
/// => <x_i[:d-1], vs> + x_i[d]*v != 0
/// => v != - <x_i[:d-1], vs> / x_i[d]
/// We compute this value for all x_i, and then
/// set v to be the maximum element of this set plus one. Thus
/// v is outside the set as desired, and we append it to vs
/// to obtain the result of the d'th step.
Point mlir::presburger::detail::getNonOrthogonalVector(
    ArrayRef<Point> vectors) {
  unsigned dim = vectors[0].size();
  assert(
      llvm::all_of(vectors,
                   [&](const Point &vector) { return vector.size() == dim; }) &&
      "all vectors need to be the same size!");

  SmallVector<Fraction> newPoint = {Fraction(1, 1)};
  Fraction maxDisallowedValue = -Fraction(1, 0),
           disallowedValue = Fraction(0, 1);

  for (unsigned d = 1; d < dim; ++d) {
    // Compute the disallowed values  - <x_i[:d-1], vs> / x_i[d] for each i.
    maxDisallowedValue = -Fraction(1, 0);
    for (const Point &vector : vectors) {
      if (vector[d] == 0)
        continue;
      disallowedValue =
          -dotProduct(ArrayRef(vector).slice(0, d), newPoint) / vector[d];

      // Find the biggest such value
      maxDisallowedValue = std::max(maxDisallowedValue, disallowedValue);
    }
    newPoint.push_back(maxDisallowedValue + 1);
  }
  return newPoint;
}

/// We use the following recursive formula to find the coefficient of
/// s^power in the rational function given by P(s)/Q(s).
///
/// Let P[i] denote the coefficient of s^i in the polynomial P(s).
/// (P/Q)[r] =
/// if (r == 0) then
///   P[0]/Q[0]
/// else
///   (P[r] - {Σ_{i=1}^r (P/Q)[r-i] * Q[i])}/(Q[0])
/// We therefore recursively call `getCoefficientInRationalFunction` on
/// all i \in [0, power).
///
/// https://math.ucdavis.edu/~deloera/researchsummary/
/// barvinokalgorithm-latte1.pdf, p. 1285
QuasiPolynomial mlir::presburger::detail::getCoefficientInRationalFunction(
    unsigned power, ArrayRef<QuasiPolynomial> num, ArrayRef<Fraction> den) {
  assert(den.size() != 0 &&
         "division by empty denominator in rational function!");

  unsigned numParam = num[0].getNumInputs();
  // We use the `isEqual` method of PresburgerSpace, which QuasiPolynomial
  // inherits from.
  assert(
      llvm::all_of(
          num, [&](const QuasiPolynomial &qp) { return num[0].isEqual(qp); }) &&
      "the quasipolynomials should all belong to the same space!");

  std::vector<QuasiPolynomial> coefficients;
  coefficients.reserve(power + 1);

  coefficients.push_back(num[0] / den[0]);
  for (unsigned i = 1; i <= power; ++i) {
    // If the power is not there in the numerator, the coefficient is zero.
    coefficients.push_back(i < num.size() ? num[i]
                                          : QuasiPolynomial(numParam, 0));

    // After den.size(), the coefficients are zero, so we stop
    // subtracting at that point (if it is less than i).
    unsigned limit = std::min<unsigned long>(i, den.size() - 1);
    for (unsigned j = 1; j <= limit; ++j)
      coefficients[i] = coefficients[i] -
                        coefficients[i - j] * QuasiPolynomial(numParam, den[j]);

    coefficients[i] = coefficients[i] / den[0];
  }
  return coefficients[power].simplify();
}
