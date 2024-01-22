//===- Barvinok.cpp - Barvinok's Algorithm ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Barvinok.h"
#include "mlir/Analysis/Presburger/Utils.h"
#include "llvm/ADT/Sequence.h"
#include <algorithm>
#include <bitset>

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

/// We use Gaussian elimination to find the solution to a set of d equations
/// of the form
/// a_1 x_1 + ... + a_d x_d + b_1 m_1 + ... + b_p m_p + c = 0
/// where x_i are variables,
/// m_i are parameters and
/// a_i, b_i, c are rational coefficients.
/// The solution expresses each x_i as an affine function of the m_i, and is
/// therefore represented as a matrix of size d x (p+1).
/// If there is no solution, we return null.
std::optional<ParamPoint>
mlir::presburger::detail::findVertex(IntMatrix equations) {
  // equations is a d x (d + p + 1) matrix.
  // Each row represents an equation.

  unsigned numEqs = equations.getNumRows();
  unsigned numCols = equations.getNumColumns();

  // First, we check that the system has a solution, and return
  // null if not.
  IntMatrix coeffs(numEqs, numEqs);
  for (unsigned i = 0; i < numEqs; i++)
    for (unsigned j = 0; j < numEqs; j++)
      coeffs(i, j) = equations(i, j);

  if (coeffs.determinant() == 0)
    return std::nullopt;

  // We work with rational numbers.
  FracMatrix equationsF(equations);

  for (unsigned i = 0; i < numEqs; ++i) {
    // First ensure that the diagonal element is nonzero, by swapping
    // it with a nonzero row.
    if (equationsF(i, i) == 0) {
      for (unsigned j = i + 1; j < numEqs; ++j) {
        if (equationsF(j, i) != 0) {
          equationsF.swapRows(j, i);
          break;
        }
      }
    }

    Fraction b = equationsF(i, i);

    // Set all elements except the diagonal to zero.
    for (unsigned j = 0; j < numEqs; ++j) {
      if (equationsF(j, i) == 0 || j == i)
        continue;
      // Set element (j, i) to zero
      // by subtracting the ith row,
      // appropriately scaled.
      Fraction a = equationsF(j, i);
      equationsF.addToRow(j, equationsF.getRow(i), -a / b);
    }
  }

  // Rescale diagonal elements to 1.
  for (unsigned i = 0; i < numEqs; ++i) {
    Fraction a = equationsF(i, i);
    for (unsigned j = 0; j < numCols; ++j)
      equationsF(i, j) = equationsF(i, j) / a;
  }

  // We copy the last p+1 columns of the matrix as the values of x_i.
  // We shift the parameter terms to the RHS, and so flip their sign.
  ParamPoint vertex(numEqs, numCols - numEqs);
  for (unsigned i = 0; i < numEqs; ++i)
    for (unsigned j = 0; j < numCols - numEqs; ++j)
      vertex(i, j) = -equationsF(i, numEqs + j);

  return vertex;
}

/// For a polytope expressed as a set of inequalities, compute the generating
/// function corresponding to the number of lattice points present. This
/// algorithm has three main steps:
/// 1. Enumerate the vertices, by iterating over subsets of inequalities and
///    checking for solubility.
/// 2. For each vertex, identify the tangent cone and compute the generating
///    function corresponding to it. The sum of these GFs is the GF of the
///    polytope.
/// 3. [Clauss-Loechner decomposition] Identify the regions in parameter space
///    (chambers) where each vertex is active, and accordingly compute the
///    GF of the polytope in each chamber.
///
/// Verdoolaege, Sven, et al. "Counting integer points in parametric
/// polytopes using Barvinok's rational functions." Algorithmica 48 (2007):
/// 37-66.
std::vector<std::pair<PresburgerRelation, GeneratingFunction>>
mlir::presburger::detail::polytopeGeneratingFunction(PolyhedronH poly) {
  unsigned numVars = poly.getNumRangeVars();
  unsigned numParams = poly.getNumSymbolVars();
  unsigned numIneqs = poly.getNumInequalities();

  // The generating function of the polytope is computed as a set of generating
  // functions, each one associated with a region in parameter space (chamber).
  std::vector<std::pair<PresburgerRelation, GeneratingFunction>> gf({});

  // The active region will be defined as activeRegionCoeffs @ p +
  // activeRegionConstant ≥ 0. The active region is a polyhedron in parameter
  // space.
  FracMatrix activeRegion(numIneqs - numVars, numParams + 1);

  // These vectors store lists of
  // subsets of inequalities,
  // the vertices corresponding to them, and
  // the active regions of the vertices, in order.
  std::vector<IntMatrix> subsets;
  std::vector<ParamPoint> vertices;
  std::vector<PresburgerRelation> activeRegions;

  FracMatrix a2(numIneqs - numVars, numVars);
  FracMatrix b2c2(numIneqs - numVars, numParams + 1);

  // We iterate over all subsets of inequalities with cardinality numVars,
  // using bitsets up to 2^numIneqs to enumerate.
  for (std::bitset<16> indicator(((1ul << numVars) - 1ul)
                                 << (numIneqs - numVars));
       indicator.to_ulong() <=
       ((1ul << numVars) - 1ul)
           << (numIneqs - numVars); // d 1's followed by n-numVars 0's
       indicator = std::bitset<16>(indicator.to_ulong() - 1)) {

    if (indicator.count() != numVars)
      continue;

    // Collect the inequalities corresponding to the bits which are set.
    IntMatrix subset(numVars, numVars + numParams + 1);
    unsigned j1 = 0, j2 = 0;
    for (unsigned i = 0; i < numIneqs; i++)
      if (indicator.test(i))
        subset.setRow(j1++, poly.getInequality(i));

      else {
        // All other inequalities are stored in a2 and b2c2.
        // These are column-wise splits of the inequalities;
        // a2 stores the coefficients of the variables, and
        // b2c2 stores the coefficients of the parameters and the constant term.
        for (unsigned k = 0; k < numVars; k++)
          a2(j2, k) = poly.atIneq(i, k);
        for (unsigned k = numVars; k < numVars + numParams + 1; k++)
          b2c2(j2, k - numVars) = poly.atIneq(i, k);
        j2++;
      }

    // Find the vertex, if any, corresponding to the current subset of
    // inequalities.
    std::optional<ParamPoint> vertex = findVertex(subset); // d x (p+1)

    if (vertex == std::nullopt)
      continue;
    // If this subset corresponds to a vertex, store it.
    vertices.push_back(*vertex);
    subsets.push_back(subset);

    // Let the current vertex be [X | y], where
    // X represents the coefficients of the parameters and
    // y represents the constant term.

    // The region (in parameter space) where this vertex is active is given
    // by substituting the vertex into the *remaining* inequalities of the
    // polytope (those which were not collected into `subset`), i.e.,
    // [A2 | B2 | c2].
    // Thus, the coefficients of the parameters after substitution become
    // (A2 • X + B2)
    // and the constant terms become
    // (A2 • y + c2).
    // The region is therefore given by
    // (A2 • X + B2) p + (A2 • y + c2) ≥ 0
    // This is equivalent to A2 • [X | y] + [B2 | c2]
    // Thus we premultiply [X | y] with each row of A2
    // and add each row of [B2 | c2].
    for (unsigned i = 0; i < numIneqs - numVars; i++) {
      activeRegion.setRow(i, (*vertex).preMultiplyWithRow(a2.getRow(i)));
      activeRegion.addToRow(i, b2c2.getRow(i), 1);
    }

    // We convert the representation of the active region to an integers-only
    // form so as to store it as an PresburgerRelation.
    // We do this by taking the LCM of the denominators of all the coefficients
    // and multiplying by it throughout.
    IntMatrix activeRegionNorm = IntMatrix(numIneqs - numVars, numParams + 1);
    IntegerRelation activeRegionRel =
        IntegerRelation(PresburgerSpace::getRelationSpace(0, numParams, 0, 0));
    MPInt lcmDenoms = MPInt(1);
    for (unsigned i = 0; i < numIneqs - numVars; i++) {
      for (unsigned j = 0; j < numParams + 1; j++)
        lcmDenoms = lcm(lcmDenoms, activeRegion(i, j).den);
      for (unsigned j = 0; j < numParams + 1; j++)
        activeRegionNorm(i, j) =
            (activeRegion(i, j) * lcmDenoms).getAsInteger();

      activeRegionRel.addInequality(activeRegionNorm.getRow(i));
    }

    activeRegions.push_back(PresburgerRelation(activeRegionRel));
  }

  // Now, we use Clauss-Loechner decomposition to identify regions in parameter
  // space where each vertex is active. These regions (chambers) have the
  // property that no two of them have a full-dimensional intersection, i.e.,
  // they may share "faces" or "edges", but their intersection can only have
  // up to numVars-1 dimensions.

  // We maintain a list of regions and their associated vertex sets,
  // initialized with the first vertex and its corresponding activity region.
  std::vector<std::pair<PresburgerRelation, std::vector<unsigned>>> chambers = {
      std::make_pair(activeRegions[0], std::vector({0u}))};
  // Note that instead of storing lists of actual vertices, we store lists
  // of indices. Thus the set {2, 3, 4} represents the vertex set
  // {vertices[2], vertices[3], vertices[4]}.

  std::vector<std::pair<PresburgerRelation, std::vector<unsigned>>> newChambers;

  // We iterate over the vertex set.
  // For each vertex v_j and its activity region R_j,
  // we examine all the current chambers R_i.
  // If R_j has a full-dimensional intersection with an existing chamber R_i,
  // then that chamber is replaced by two new ones:
  // 1. the intersection R_i \cap R_j, where v_j is active;
  // 2. the difference R_i - R_j, where v_j is inactive.
  // Once we have examined all R_i, we add a final chamber
  // R_j - (union of all existing chambers),
  // in which only v_j is active.
  for (unsigned j = 1; j < vertices.size(); j++) {
    newChambers.clear();

    PresburgerRelation r_j = activeRegions[j];
    ParamPoint v_j = vertices[j];

    for (unsigned i = 0; i < chambers.size(); i++) {
      auto [r_i, v_i] = chambers[i];

      // First, we check if the intersection of R_j and R_i.
      // It is a disjoint union of convex regions in the parameter space,
      // and so we know that it is full-dimensional if any of the disjuncts
      // is full-dimensional.
      PresburgerRelation intersection = r_i.intersect(r_j);
      bool isFullDim = false;
      for (auto disjunct : intersection.getAllDisjuncts())
        if (disjunct.isFullDim()) {
          isFullDim = true;
          break;
        }
      isFullDim = (numParams == 0) || isFullDim;

      // If the intersection is not full-dimensional, we do not modify
      // the chamber list.
      if (!isFullDim)
        newChambers.push_back(chambers[i]);
      else {
        // If it is, we add the intersection and the difference as new chambers.
        PresburgerRelation subtraction = r_i.subtract(r_j);
        newChambers.push_back(std::make_pair(subtraction, v_i));

        v_i.push_back(j);
        newChambers.push_back(std::make_pair(intersection, v_i));
      }
    }

    // Finally we compute the chamber where only v_j is active by subtracting
    // all existing chambers from R_j.
    for (auto chamber : newChambers)
      r_j = r_j.subtract(chamber.first);
    newChambers.push_back(std::make_pair(r_j, std::vector({j})));

    // We filter `chambers` to remove empty regions.
    chambers.clear();
    for (auto chamber : newChambers) {
      bool empty = true;
      for (auto disjunct : chamber.first.getAllDisjuncts())
        if (!disjunct.isEmpty()) {
          empty = false;
          break;
        }
      if (!empty)
        chambers.push_back(chamber);
    }
  }

  // Now, we compute the generating function. For each chamber, we iterate over
  // the vertices active in it, and compute the generating function for each
  // of them. The sum of these generating functions is the GF corresponding to
  // the entire polytope.
  SmallVector<MPInt> ineq(numVars + 1);
  for (auto chamber : chambers) {
    GeneratingFunction chamberGf(numParams, {}, {}, {});
    for (unsigned i : chamber.second) {
      // We collect the inequalities corresponding to each vertex.
      // We only need the coefficients of the variables (NOT the parameters)
      // as the generating function only depends on these.
      ConeH tgtCone = defineHRep(numVars);
      for (unsigned j = 0; j < numVars; j++) {
        for (unsigned k = 0; k < numVars; k++)
          ineq[k] = subsets[i](j, k);
        ineq[numVars] = subsets[i](j, numVars + numParams);
        tgtCone.addInequality(ineq);
      }
      // We assume that the tangent cone is unimodular.
      SmallVector<std::pair<int, ConeH>, 4> unimodCones = {
          std::make_pair(1, tgtCone)};
      for (auto signedCone : unimodCones)
        chamberGf =
            chamberGf + unimodularConeGeneratingFunction(
                            vertices[i], signedCone.first, signedCone.second);
    }
    gf.push_back(std::make_pair(chamber.first, chamberGf));
  }
  return gf;
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

/// Substitute x_i = t^μ_i in one term of a generating function, returning
/// a quasipolynomial which represents the exponent of the numerator
/// of the result, and a vector which represents the exponents of the
/// denominator of the result.
/// If the returned value is {num, dens}, it represents the function
/// t^num / \prod_j (1 - t^dens[j]).
/// v represents the affine functions whose floors are multiplied by the
/// generators, and ds represents the list of generators.
std::pair<QuasiPolynomial, std::vector<Fraction>>
substituteMuInTerm(unsigned numParams, ParamPoint v, std::vector<Point> ds,
                   Point mu) {
  unsigned numDims = mu.size();
  for (const Point &d : ds)
    assert(d.size() == numDims &&
           "μ has to have the same number of dimensions as the generators!");

  // First, the exponent in the numerator becomes
  // - (μ • u_1) * (floor(first col of v))
  // - (μ • u_2) * (floor(second col of v)) - ...
  // - (μ • u_d) * (floor(d'th col of v))
  // So we store the negation of the dot products.

  // We have d terms, each of whose coefficient is the negative dot product.
  SmallVector<Fraction> coefficients;
  coefficients.reserve(numDims);
  for (const Point &d : ds)
    coefficients.push_back(-dotProduct(mu, d));

  // Then, the affine function is a single floor expression, given by the
  // corresponding column of v.
  ParamPoint vTranspose = v.transpose();
  std::vector<std::vector<SmallVector<Fraction>>> affine;
  affine.reserve(numDims);
  for (unsigned j = 0; j < numDims; ++j)
    affine.push_back({SmallVector<Fraction>(vTranspose.getRow(j))});

  QuasiPolynomial num(numParams, coefficients, affine);
  num = num.simplify();

  std::vector<Fraction> dens;
  dens.reserve(ds.size());
  // Similarly, each term in the denominator has exponent
  // given by the dot product of μ with u_i.
  for (const Point &d : ds) {
    // This term in the denominator is
    // (1 - t^dens.back())
    dens.push_back(dotProduct(d, mu));
  }

  return {num, dens};
}

/// Normalize all denominator exponents `dens` to their absolute values
/// by multiplying and dividing by the inverses, in a function of the form
/// sign * t^num / prod_j (1 - t^dens[j]).
/// Here, sign = ± 1,
/// num is a QuasiPolynomial, and
/// each dens[j] is a Fraction.
void normalizeDenominatorExponents(int &sign, QuasiPolynomial &num,
                                   std::vector<Fraction> &dens) {
  // We track the number of exponents that are negative in the
  // denominator, and convert them to their absolute values.
  unsigned numNegExps = 0;
  Fraction sumNegExps(0, 1);
  for (unsigned j = 0, e = dens.size(); j < e; ++j) {
    if (dens[j] < 0) {
      numNegExps += 1;
      sumNegExps += dens[j];
    }
  }

  // If we have (1 - t^-c) in the denominator, for positive c,
  // multiply and divide by t^c.
  // We convert all negative-exponent terms at once; therefore
  // we multiply and divide by t^sumNegExps.
  // Then we get
  // -(1 - t^c) in the denominator,
  // increase the numerator by c, and
  // flip the sign of the function.
  if (numNegExps % 2 == 1)
    sign = -sign;
  num = num - QuasiPolynomial(num.getNumInputs(), sumNegExps);
}

/// Compute the binomial coefficients nCi for 0 ≤ i ≤ r,
/// where n is a QuasiPolynomial.
std::vector<QuasiPolynomial> getBinomialCoefficients(QuasiPolynomial n,
                                                     unsigned r) {
  unsigned numParams = n.getNumInputs();
  std::vector<QuasiPolynomial> coefficients;
  coefficients.reserve(r + 1);
  coefficients.push_back(QuasiPolynomial(numParams, 1));
  for (unsigned j = 1; j <= r; ++j)
    // We use the recursive formula for binomial coefficients here and below.
    coefficients.push_back(
        (coefficients[j - 1] * (n - QuasiPolynomial(numParams, j - 1)) /
         Fraction(j, 1))
            .simplify());
  return coefficients;
}

/// Compute the binomial coefficients nCi for 0 ≤ i ≤ r,
/// where n is a QuasiPolynomial.
std::vector<Fraction> getBinomialCoefficients(Fraction n, Fraction r) {
  std::vector<Fraction> coefficients;
  coefficients.reserve((int64_t)floor(r));
  coefficients.push_back(1);
  for (unsigned j = 1; j <= r; ++j)
    coefficients.push_back(coefficients[j - 1] * (n - (j - 1)) / (j));
  return coefficients;
}

/// We have a generating function of the form
/// f_p(x) = \sum_i sign_i * (x^n_i(p)) / (\prod_j (1 - x^d_{ij})
///
/// where sign_i is ±1,
/// n_i \in Q^p -> Q^d is the sum of the vectors d_{ij}, weighted by the
/// floors of d affine functions on p parameters.
/// d_{ij} \in Q^d are vectors.
///
/// We need to find the number of terms of the form x^t in the expansion of
/// this function.
/// However, direct substitution (x = (1, ..., 1)) causes the denominator
/// to become zero.
///
/// We therefore use the following procedure instead:
/// 1. Substitute x_i = (s+1)^μ_i for some vector μ. This makes the generating
/// function a function of a scalar s.
/// 2. Write each term in this function as P(s)/Q(s), where P and Q are
/// polynomials. P has coefficients as quasipolynomials in d parameters, while
/// Q has coefficients as scalars.
/// 3. Find the constant term in the expansion of each term P(s)/Q(s). This is
/// equivalent to substituting s = 0.
///
/// Verdoolaege, Sven, et al. "Counting integer points in parametric
/// polytopes using Barvinok's rational functions." Algorithmica 48 (2007):
/// 37-66.
QuasiPolynomial
mlir::presburger::detail::computeNumTerms(const GeneratingFunction &gf) {
  // Step (1) We need to find a μ such that we can substitute x_i =
  // (s+1)^μ_i. After this substitution, the exponent of (s+1) in the
  // denominator is (μ_i • d_{ij}) in each term. Clearly, this cannot become
  // zero. Hence we find a vector μ that is not orthogonal to any of the
  // d_{ij} and substitute x accordingly.
  std::vector<Point> allDenominators;
  for (ArrayRef<Point> den : gf.getDenominators())
    allDenominators.insert(allDenominators.end(), den.begin(), den.end());
  Point mu = getNonOrthogonalVector(allDenominators);

  unsigned numParams = gf.getNumParams();
  const std::vector<std::vector<Point>> &ds = gf.getDenominators();
  QuasiPolynomial totalTerm(numParams, 0);
  for (unsigned i = 0, e = ds.size(); i < e; ++i) {
    int sign = gf.getSigns()[i];

    // Compute the new exponents of (s+1) for the numerator and the
    // denominator after substituting μ.
    auto [numExp, dens] =
        substituteMuInTerm(numParams, gf.getNumerators()[i], ds[i], mu);
    // Now the numerator is (s+1)^numExp
    // and the denominator is \prod_j (1 - (s+1)^dens[j]).

    // Step (2) We need to express the terms in the function as quotients of
    // polynomials. Each term is now of the form
    // sign_i * (s+1)^numExp / (\prod_j (1 - (s+1)^dens[j]))
    // For the i'th term, we first normalize the denominator to have only
    // positive exponents. We convert all the dens[j] to their
    // absolute values and change the sign and exponent in the numerator.
    normalizeDenominatorExponents(sign, numExp, dens);

    // Then, using the formula for geometric series, we replace each (1 -
    // (s+1)^(dens[j])) with
    // (-s)(\sum_{0 ≤ k < dens[j]} (s+1)^k).
    for (unsigned j = 0, e = dens.size(); j < e; ++j)
      dens[j] = abs(dens[j]) - 1;
    // Note that at this point, the semantics of `dens[j]` changes to mean
    // a term (\sum_{0 ≤ k ≤ dens[j]} (s+1)^k). The denominator is, as before,
    // a product of these terms.

    // Since the -s are taken out, the sign changes if there is an odd number
    // of such terms.
    unsigned r = dens.size();
    if (dens.size() % 2 == 1)
      sign = -sign;

    // Thus the term overall now has the form
    // sign'_i * (s+1)^numExp /
    // (s^r * \prod_j (\sum_{0 ≤ k < dens[j]} (s+1)^k)).
    // This means that
    // the numerator is a polynomial in s, with coefficients as
    // quasipolynomials (given by binomial coefficients), and the denominator
    // is a polynomial in s, with integral coefficients (given by taking the
    // convolution over all j).

    // Step (3) We need to find the constant term in the expansion of each
    // term. Since each term has s^r as a factor in the denominator, we avoid
    // substituting s = 0 directly; instead, we find the coefficient of s^r in
    // sign'_i * (s+1)^numExp / (\prod_j (\sum_k (s+1)^k)),
    // Letting P(s) = (s+1)^numExp and Q(s) = \prod_j (...),
    // we need to find the coefficient of s^r in P(s)/Q(s),
    // for which we use the `getCoefficientInRationalFunction()` function.

    // First, we compute the coefficients of P(s), which are binomial
    // coefficients.
    // We only need the first r+1 of these, as higher-order terms do not
    // contribute to the coefficient of s^r.
    std::vector<QuasiPolynomial> numeratorCoefficients =
        getBinomialCoefficients(numExp, r);

    // Then we compute the coefficients of each individual term in Q(s),
    // which are (dens[i]+1) C (k+1) for 0 ≤ k ≤ dens[i].
    std::vector<std::vector<Fraction>> eachTermDenCoefficients;
    std::vector<Fraction> singleTermDenCoefficients;
    eachTermDenCoefficients.reserve(r);
    for (const Fraction &den : dens) {
      singleTermDenCoefficients = getBinomialCoefficients(den + 1, den + 1);
      eachTermDenCoefficients.push_back(
          ArrayRef<Fraction>(singleTermDenCoefficients).slice(1));
    }

    // Now we find the coefficients in Q(s) itself
    // by taking the convolution of the coefficients
    // of all the terms.
    std::vector<Fraction> denominatorCoefficients;
    denominatorCoefficients = eachTermDenCoefficients[0];
    for (unsigned j = 1, e = eachTermDenCoefficients.size(); j < e; ++j)
      denominatorCoefficients = multiplyPolynomials(denominatorCoefficients,
                                                    eachTermDenCoefficients[j]);

    totalTerm =
        totalTerm + getCoefficientInRationalFunction(r, numeratorCoefficients,
                                                     denominatorCoefficients) *
                        QuasiPolynomial(numParams, sign);
  }

  return totalTerm.simplify();
}