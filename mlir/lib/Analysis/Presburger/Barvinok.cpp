//===- Barvinok.cpp - Barvinok's Algorithm ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Barvinok.h"

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

  for (unsigned i = 0; i < numIneq; ++i) {
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
/// x^num / prod_j (1 - x^den_j)
///
/// den_j is defined as the set of generators of the cone.
/// num is computed by expressing the vertex as a weighted
/// sum of the generators, and then taking the floor of the
/// coefficients.
GeneratingFunction mlir::presburger::detail::unimodularConeGeneratingFunction(
    ParamPoint vertex, int sign, ConeH cone) {
  // `cone` is assumed to be unimodular.
  assert(getIndex(getDual(cone)) == 1 && "input cone is not unimodular!");

  unsigned numVar = cone.getNumVars();
  unsigned numIneq = cone.getNumInequalities();

  // Thus its ray matrix, U, is the inverse of the
  // transpose of its inequality matrix, `cone`.
  FracMatrix transp(numVar, numIneq);
  for (unsigned i = 0; i < numVar; ++i)
    for (unsigned j = 0; j < numIneq; ++j)
      transp(j, i) = Fraction(cone.atIneq(i, j), 1);

  FracMatrix generators(numVar, numIneq);
  transp.determinant(&generators); // This is the U-matrix.

  // The denominators of the generating function
  // are given by the generators of the cone, i.e.,
  // the rows of the matrix U.
  std::vector<Point> denominator(numIneq);
  ArrayRef<Fraction> row;
  for (unsigned i = 0; i < numVar; ++i) {
    row = generators.getRow(i);
    denominator[i] = Point(row);
  }

  // The vertex is v : [d, n+1].
  // We need to find affine functions of parameters λi(p)
  // such that v = Σ λi(p)*ui.
  // The λi are given by the columns of Λ = v^T @ U^{-1} = v^T @ transp.
  // Then the numerator will be Σ -floor(-λi(p))*u_i.
  // Thus we store the numerator as the affine function -Λ,
  // since the generators are already stored in the denominator.
  // Note that the outer -1 will have to be accounted for, as it is not stored.
  // See end for an example.

  unsigned numColumns = vertex.getNumColumns();
  unsigned numRows = vertex.getNumRows();
  ParamPoint numerator(numColumns, numRows);
  SmallVector<Fraction> ithCol(numRows);
  for (unsigned i = 0; i < numColumns; ++i) {
    for (unsigned j = 0; j < numRows; ++j)
      ithCol[j] = vertex(j, i);
    numerator.setRow(i, transp.preMultiplyWithRow(ithCol));
    numerator.negateRow(i);
  }

  return GeneratingFunction(numColumns - 1, SmallVector<int>(1, sign),
                            std::vector({numerator}),
                            std::vector({denominator}));

  // Suppose the vertex is given by the matrix [ 2  2   0], with 2 params
  //                                           [-1 -1/2 1]
  // and the cone has H-representation [0  -1] => U-matrix [ 2 -1]
  //                                   [-1 -2]             [-1  0]
  // Therefore Λ will be given by [ 1    0 ] and the negation of this will be
  // stored as the numerator.
  //                              [ 1/2 -1 ]
  //                              [ -1  -2 ]

  // Algebraically, the numerator exponent is
  // [ -2 ⌊ -N - M/2 +1 ⌋ + 1 ⌊ 0 +M +2 ⌋ ] -> first  COLUMN of U is [2, -1]
  // [  1 ⌊ -N - M/2 +1 ⌋ + 0 ⌊ 0 +M +2 ⌋ ] -> second COLUMN of U is [-1, 0]
}
