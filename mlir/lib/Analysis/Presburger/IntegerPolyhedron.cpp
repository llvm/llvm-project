//===- IntegerPolyhedron.cpp - MLIR IntegerPolyhedron Class ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A class to represent an integer polyhedron.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/IntegerPolyhedron.h"
#include "mlir/Analysis/Presburger/LinearTransform.h"
#include "mlir/Analysis/Presburger/PresburgerSet.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Analysis/Presburger/Utils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "presburger"

using namespace mlir;
using namespace presburger;

using llvm::SmallDenseMap;
using llvm::SmallDenseSet;

std::unique_ptr<IntegerPolyhedron> IntegerPolyhedron::clone() const {
  return std::make_unique<IntegerPolyhedron>(*this);
}

void IntegerPolyhedron::append(const IntegerPolyhedron &other) {
  assert(PresburgerLocalSpace::isEqual(other) && "Spaces must be equal.");

  inequalities.reserveRows(inequalities.getNumRows() +
                           other.getNumInequalities());
  equalities.reserveRows(equalities.getNumRows() + other.getNumEqualities());

  for (unsigned r = 0, e = other.getNumInequalities(); r < e; r++) {
    addInequality(other.getInequality(r));
  }
  for (unsigned r = 0, e = other.getNumEqualities(); r < e; r++) {
    addEquality(other.getEquality(r));
  }
}

bool IntegerPolyhedron::isEqual(const IntegerPolyhedron &other) const {
  return PresburgerSet(*this).isEqual(PresburgerSet(other));
}

bool IntegerPolyhedron::isSubsetOf(const IntegerPolyhedron &other) const {
  return PresburgerSet(*this).isSubsetOf(PresburgerSet(other));
}

MaybeOptimum<SmallVector<Fraction, 8>>
IntegerPolyhedron::findRationalLexMin() const {
  assert(getNumSymbolIds() == 0 && "Symbols are not supported!");
  MaybeOptimum<SmallVector<Fraction, 8>> maybeLexMin =
      LexSimplex(*this).findRationalLexMin();

  if (!maybeLexMin.isBounded())
    return maybeLexMin;

  // The Simplex returns the lexmin over all the variables including locals. But
  // locals are not actually part of the space and should not be returned in the
  // result. Since the locals are placed last in the list of identifiers, they
  // will be minimized last in the lexmin. So simply truncating out the locals
  // from the end of the answer gives the desired lexmin over the dimensions.
  assert(maybeLexMin->size() == getNumIds() &&
         "Incorrect number of vars in lexMin!");
  maybeLexMin->resize(getNumDimAndSymbolIds());
  return maybeLexMin;
}

MaybeOptimum<SmallVector<int64_t, 8>>
IntegerPolyhedron::findIntegerLexMin() const {
  assert(getNumSymbolIds() == 0 && "Symbols are not supported!");
  MaybeOptimum<SmallVector<int64_t, 8>> maybeLexMin =
      LexSimplex(*this).findIntegerLexMin();

  if (!maybeLexMin.isBounded())
    return maybeLexMin.getKind();

  // The Simplex returns the lexmin over all the variables including locals. But
  // locals are not actually part of the space and should not be returned in the
  // result. Since the locals are placed last in the list of identifiers, they
  // will be minimized last in the lexmin. So simply truncating out the locals
  // from the end of the answer gives the desired lexmin over the dimensions.
  assert(maybeLexMin->size() == getNumIds() &&
         "Incorrect number of vars in lexMin!");
  maybeLexMin->resize(getNumDimAndSymbolIds());
  return maybeLexMin;
}

unsigned IntegerPolyhedron::insertId(IdKind kind, unsigned pos, unsigned num) {
  assert(pos <= getNumIdKind(kind));

  unsigned insertPos = PresburgerLocalSpace::insertId(kind, pos, num);
  inequalities.insertColumns(insertPos, num);
  equalities.insertColumns(insertPos, num);
  return insertPos;
}

unsigned IntegerPolyhedron::appendId(IdKind kind, unsigned num) {
  unsigned pos = getNumIdKind(kind);
  return insertId(kind, pos, num);
}

void IntegerPolyhedron::addEquality(ArrayRef<int64_t> eq) {
  assert(eq.size() == getNumCols());
  unsigned row = equalities.appendExtraRow();
  for (unsigned i = 0, e = eq.size(); i < e; ++i)
    equalities(row, i) = eq[i];
}

void IntegerPolyhedron::addInequality(ArrayRef<int64_t> inEq) {
  assert(inEq.size() == getNumCols());
  unsigned row = inequalities.appendExtraRow();
  for (unsigned i = 0, e = inEq.size(); i < e; ++i)
    inequalities(row, i) = inEq[i];
}

void IntegerPolyhedron::removeId(IdKind kind, unsigned pos) {
  removeIdRange(kind, pos, pos + 1);
}

void IntegerPolyhedron::removeId(unsigned pos) { removeIdRange(pos, pos + 1); }

void IntegerPolyhedron::removeIdRange(IdKind kind, unsigned idStart,
                                      unsigned idLimit) {
  assert(idLimit <= getNumIdKind(kind));
  removeIdRange(getIdKindOffset(kind) + idStart,
                getIdKindOffset(kind) + idLimit);
}

void IntegerPolyhedron::removeIdRange(unsigned idStart, unsigned idLimit) {
  // Update space paramaters.
  PresburgerLocalSpace::removeIdRange(idStart, idLimit);

  // Remove eliminated identifiers from the constraints..
  equalities.removeColumns(idStart, idLimit - idStart);
  inequalities.removeColumns(idStart, idLimit - idStart);
}

void IntegerPolyhedron::removeEquality(unsigned pos) {
  equalities.removeRow(pos);
}

void IntegerPolyhedron::removeInequality(unsigned pos) {
  inequalities.removeRow(pos);
}

void IntegerPolyhedron::removeEqualityRange(unsigned start, unsigned end) {
  if (start >= end)
    return;
  equalities.removeRows(start, end - start);
}

void IntegerPolyhedron::removeInequalityRange(unsigned start, unsigned end) {
  if (start >= end)
    return;
  inequalities.removeRows(start, end - start);
}

void IntegerPolyhedron::swapId(unsigned posA, unsigned posB) {
  assert(posA < getNumIds() && "invalid position A");
  assert(posB < getNumIds() && "invalid position B");

  if (posA == posB)
    return;

  inequalities.swapColumns(posA, posB);
  equalities.swapColumns(posA, posB);
}

void IntegerPolyhedron::clearConstraints() {
  equalities.resizeVertically(0);
  inequalities.resizeVertically(0);
}

/// Gather all lower and upper bounds of the identifier at `pos`, and
/// optionally any equalities on it. In addition, the bounds are to be
/// independent of identifiers in position range [`offset`, `offset` + `num`).
void IntegerPolyhedron::getLowerAndUpperBoundIndices(
    unsigned pos, SmallVectorImpl<unsigned> *lbIndices,
    SmallVectorImpl<unsigned> *ubIndices, SmallVectorImpl<unsigned> *eqIndices,
    unsigned offset, unsigned num) const {
  assert(pos < getNumIds() && "invalid position");
  assert(offset + num < getNumCols() && "invalid range");

  // Checks for a constraint that has a non-zero coeff for the identifiers in
  // the position range [offset, offset + num) while ignoring `pos`.
  auto containsConstraintDependentOnRange = [&](unsigned r, bool isEq) {
    unsigned c, f;
    auto cst = isEq ? getEquality(r) : getInequality(r);
    for (c = offset, f = offset + num; c < f; ++c) {
      if (c == pos)
        continue;
      if (cst[c] != 0)
        break;
    }
    return c < f;
  };

  // Gather all lower bounds and upper bounds of the variable. Since the
  // canonical form c_1*x_1 + c_2*x_2 + ... + c_0 >= 0, a constraint is a lower
  // bound for x_i if c_i >= 1, and an upper bound if c_i <= -1.
  for (unsigned r = 0, e = getNumInequalities(); r < e; r++) {
    // The bounds are to be independent of [offset, offset + num) columns.
    if (containsConstraintDependentOnRange(r, /*isEq=*/false))
      continue;
    if (atIneq(r, pos) >= 1) {
      // Lower bound.
      lbIndices->push_back(r);
    } else if (atIneq(r, pos) <= -1) {
      // Upper bound.
      ubIndices->push_back(r);
    }
  }

  // An equality is both a lower and upper bound. Record any equalities
  // involving the pos^th identifier.
  if (!eqIndices)
    return;

  for (unsigned r = 0, e = getNumEqualities(); r < e; r++) {
    if (atEq(r, pos) == 0)
      continue;
    if (containsConstraintDependentOnRange(r, /*isEq=*/true))
      continue;
    eqIndices->push_back(r);
  }
}

bool IntegerPolyhedron::hasConsistentState() const {
  if (!inequalities.hasConsistentState())
    return false;
  if (!equalities.hasConsistentState())
    return false;
  return true;
}

void IntegerPolyhedron::setAndEliminate(unsigned pos,
                                        ArrayRef<int64_t> values) {
  if (values.empty())
    return;
  assert(pos + values.size() <= getNumIds() &&
         "invalid position or too many values");
  // Setting x_j = p in sum_i a_i x_i + c is equivalent to adding p*a_j to the
  // constant term and removing the id x_j. We do this for all the ids
  // pos, pos + 1, ... pos + values.size() - 1.
  unsigned constantColPos = getNumCols() - 1;
  for (unsigned i = 0, numVals = values.size(); i < numVals; ++i)
    inequalities.addToColumn(i + pos, constantColPos, values[i]);
  for (unsigned i = 0, numVals = values.size(); i < numVals; ++i)
    equalities.addToColumn(i + pos, constantColPos, values[i]);
  removeIdRange(pos, pos + values.size());
}

void IntegerPolyhedron::clearAndCopyFrom(const IntegerPolyhedron &other) {
  *this = other;
}

// Searches for a constraint with a non-zero coefficient at `colIdx` in
// equality (isEq=true) or inequality (isEq=false) constraints.
// Returns true and sets row found in search in `rowIdx`, false otherwise.
bool IntegerPolyhedron::findConstraintWithNonZeroAt(unsigned colIdx, bool isEq,
                                                    unsigned *rowIdx) const {
  assert(colIdx < getNumCols() && "position out of bounds");
  auto at = [&](unsigned rowIdx) -> int64_t {
    return isEq ? atEq(rowIdx, colIdx) : atIneq(rowIdx, colIdx);
  };
  unsigned e = isEq ? getNumEqualities() : getNumInequalities();
  for (*rowIdx = 0; *rowIdx < e; ++(*rowIdx)) {
    if (at(*rowIdx) != 0) {
      return true;
    }
  }
  return false;
}

void IntegerPolyhedron::normalizeConstraintsByGCD() {
  for (unsigned i = 0, e = getNumEqualities(); i < e; ++i)
    equalities.normalizeRow(i);
  for (unsigned i = 0, e = getNumInequalities(); i < e; ++i)
    inequalities.normalizeRow(i);
}

bool IntegerPolyhedron::hasInvalidConstraint() const {
  assert(hasConsistentState());
  auto check = [&](bool isEq) -> bool {
    unsigned numCols = getNumCols();
    unsigned numRows = isEq ? getNumEqualities() : getNumInequalities();
    for (unsigned i = 0, e = numRows; i < e; ++i) {
      unsigned j;
      for (j = 0; j < numCols - 1; ++j) {
        int64_t v = isEq ? atEq(i, j) : atIneq(i, j);
        // Skip rows with non-zero variable coefficients.
        if (v != 0)
          break;
      }
      if (j < numCols - 1) {
        continue;
      }
      // Check validity of constant term at 'numCols - 1' w.r.t 'isEq'.
      // Example invalid constraints include: '1 == 0' or '-1 >= 0'
      int64_t v = isEq ? atEq(i, numCols - 1) : atIneq(i, numCols - 1);
      if ((isEq && v != 0) || (!isEq && v < 0)) {
        return true;
      }
    }
    return false;
  };
  if (check(/*isEq=*/true))
    return true;
  return check(/*isEq=*/false);
}

/// Eliminate identifier from constraint at `rowIdx` based on coefficient at
/// pivotRow, pivotCol. Columns in range [elimColStart, pivotCol) will not be
/// updated as they have already been eliminated.
static void eliminateFromConstraint(IntegerPolyhedron *constraints,
                                    unsigned rowIdx, unsigned pivotRow,
                                    unsigned pivotCol, unsigned elimColStart,
                                    bool isEq) {
  // Skip if equality 'rowIdx' if same as 'pivotRow'.
  if (isEq && rowIdx == pivotRow)
    return;
  auto at = [&](unsigned i, unsigned j) -> int64_t {
    return isEq ? constraints->atEq(i, j) : constraints->atIneq(i, j);
  };
  int64_t leadCoeff = at(rowIdx, pivotCol);
  // Skip if leading coefficient at 'rowIdx' is already zero.
  if (leadCoeff == 0)
    return;
  int64_t pivotCoeff = constraints->atEq(pivotRow, pivotCol);
  int64_t sign = (leadCoeff * pivotCoeff > 0) ? -1 : 1;
  int64_t lcm = mlir::lcm(pivotCoeff, leadCoeff);
  int64_t pivotMultiplier = sign * (lcm / std::abs(pivotCoeff));
  int64_t rowMultiplier = lcm / std::abs(leadCoeff);

  unsigned numCols = constraints->getNumCols();
  for (unsigned j = 0; j < numCols; ++j) {
    // Skip updating column 'j' if it was just eliminated.
    if (j >= elimColStart && j < pivotCol)
      continue;
    int64_t v = pivotMultiplier * constraints->atEq(pivotRow, j) +
                rowMultiplier * at(rowIdx, j);
    isEq ? constraints->atEq(rowIdx, j) = v
         : constraints->atIneq(rowIdx, j) = v;
  }
}

/// Returns the position of the identifier that has the minimum <number of lower
/// bounds> times <number of upper bounds> from the specified range of
/// identifiers [start, end). It is often best to eliminate in the increasing
/// order of these counts when doing Fourier-Motzkin elimination since FM adds
/// that many new constraints.
static unsigned getBestIdToEliminate(const IntegerPolyhedron &cst,
                                     unsigned start, unsigned end) {
  assert(start < cst.getNumIds() && end < cst.getNumIds() + 1);

  auto getProductOfNumLowerUpperBounds = [&](unsigned pos) {
    unsigned numLb = 0;
    unsigned numUb = 0;
    for (unsigned r = 0, e = cst.getNumInequalities(); r < e; r++) {
      if (cst.atIneq(r, pos) > 0) {
        ++numLb;
      } else if (cst.atIneq(r, pos) < 0) {
        ++numUb;
      }
    }
    return numLb * numUb;
  };

  unsigned minLoc = start;
  unsigned min = getProductOfNumLowerUpperBounds(start);
  for (unsigned c = start + 1; c < end; c++) {
    unsigned numLbUbProduct = getProductOfNumLowerUpperBounds(c);
    if (numLbUbProduct < min) {
      min = numLbUbProduct;
      minLoc = c;
    }
  }
  return minLoc;
}

// Checks for emptiness of the set by eliminating identifiers successively and
// using the GCD test (on all equality constraints) and checking for trivially
// invalid constraints. Returns 'true' if the constraint system is found to be
// empty; false otherwise.
bool IntegerPolyhedron::isEmpty() const {
  if (isEmptyByGCDTest() || hasInvalidConstraint())
    return true;

  IntegerPolyhedron tmpCst(*this);

  // First, eliminate as many local variables as possible using equalities.
  tmpCst.removeRedundantLocalVars();
  if (tmpCst.isEmptyByGCDTest() || tmpCst.hasInvalidConstraint())
    return true;

  // Eliminate as many identifiers as possible using Gaussian elimination.
  unsigned currentPos = 0;
  while (currentPos < tmpCst.getNumIds()) {
    tmpCst.gaussianEliminateIds(currentPos, tmpCst.getNumIds());
    ++currentPos;
    // We check emptiness through trivial checks after eliminating each ID to
    // detect emptiness early. Since the checks isEmptyByGCDTest() and
    // hasInvalidConstraint() are linear time and single sweep on the constraint
    // buffer, this appears reasonable - but can optimize in the future.
    if (tmpCst.hasInvalidConstraint() || tmpCst.isEmptyByGCDTest())
      return true;
  }

  // Eliminate the remaining using FM.
  for (unsigned i = 0, e = tmpCst.getNumIds(); i < e; i++) {
    tmpCst.fourierMotzkinEliminate(
        getBestIdToEliminate(tmpCst, 0, tmpCst.getNumIds()));
    // Check for a constraint explosion. This rarely happens in practice, but
    // this check exists as a safeguard against improperly constructed
    // constraint systems or artificially created arbitrarily complex systems
    // that aren't the intended use case for IntegerPolyhedron. This is
    // needed since FM has a worst case exponential complexity in theory.
    if (tmpCst.getNumConstraints() >= kExplosionFactor * getNumIds()) {
      LLVM_DEBUG(llvm::dbgs() << "FM constraint explosion detected\n");
      return false;
    }

    // FM wouldn't have modified the equalities in any way. So no need to again
    // run GCD test. Check for trivial invalid constraints.
    if (tmpCst.hasInvalidConstraint())
      return true;
  }
  return false;
}

// Runs the GCD test on all equality constraints. Returns 'true' if this test
// fails on any equality. Returns 'false' otherwise.
// This test can be used to disprove the existence of a solution. If it returns
// true, no integer solution to the equality constraints can exist.
//
// GCD test definition:
//
// The equality constraint:
//
//  c_1*x_1 + c_2*x_2 + ... + c_n*x_n = c_0
//
// has an integer solution iff:
//
//  GCD of c_1, c_2, ..., c_n divides c_0.
//
bool IntegerPolyhedron::isEmptyByGCDTest() const {
  assert(hasConsistentState());
  unsigned numCols = getNumCols();
  for (unsigned i = 0, e = getNumEqualities(); i < e; ++i) {
    uint64_t gcd = std::abs(atEq(i, 0));
    for (unsigned j = 1; j < numCols - 1; ++j) {
      gcd = llvm::GreatestCommonDivisor64(gcd, std::abs(atEq(i, j)));
    }
    int64_t v = std::abs(atEq(i, numCols - 1));
    if (gcd > 0 && (v % gcd != 0)) {
      return true;
    }
  }
  return false;
}

// Returns a matrix where each row is a vector along which the polytope is
// bounded. The span of the returned vectors is guaranteed to contain all
// such vectors. The returned vectors are NOT guaranteed to be linearly
// independent. This function should not be called on empty sets.
//
// It is sufficient to check the perpendiculars of the constraints, as the set
// of perpendiculars which are bounded must span all bounded directions.
Matrix IntegerPolyhedron::getBoundedDirections() const {
  // Note that it is necessary to add the equalities too (which the constructor
  // does) even though we don't need to check if they are bounded; whether an
  // inequality is bounded or not depends on what other constraints, including
  // equalities, are present.
  Simplex simplex(*this);

  assert(!simplex.isEmpty() && "It is not meaningful to ask whether a "
                               "direction is bounded in an empty set.");

  SmallVector<unsigned, 8> boundedIneqs;
  // The constructor adds the inequalities to the simplex first, so this
  // processes all the inequalities.
  for (unsigned i = 0, e = getNumInequalities(); i < e; ++i) {
    if (simplex.isBoundedAlongConstraint(i))
      boundedIneqs.push_back(i);
  }

  // The direction vector is given by the coefficients and does not include the
  // constant term, so the matrix has one fewer column.
  unsigned dirsNumCols = getNumCols() - 1;
  Matrix dirs(boundedIneqs.size() + getNumEqualities(), dirsNumCols);

  // Copy the bounded inequalities.
  unsigned row = 0;
  for (unsigned i : boundedIneqs) {
    for (unsigned col = 0; col < dirsNumCols; ++col)
      dirs(row, col) = atIneq(i, col);
    ++row;
  }

  // Copy the equalities. All the equalities' perpendiculars are bounded.
  for (unsigned i = 0, e = getNumEqualities(); i < e; ++i) {
    for (unsigned col = 0; col < dirsNumCols; ++col)
      dirs(row, col) = atEq(i, col);
    ++row;
  }

  return dirs;
}

bool eqInvolvesSuffixDims(const IntegerPolyhedron &poly, unsigned eqIndex,
                          unsigned numDims) {
  for (unsigned e = poly.getNumIds(), j = e - numDims; j < e; ++j)
    if (poly.atEq(eqIndex, j) != 0)
      return true;
  return false;
}
bool ineqInvolvesSuffixDims(const IntegerPolyhedron &poly, unsigned ineqIndex,
                            unsigned numDims) {
  for (unsigned e = poly.getNumIds(), j = e - numDims; j < e; ++j)
    if (poly.atIneq(ineqIndex, j) != 0)
      return true;
  return false;
}

void removeConstraintsInvolvingSuffixDims(IntegerPolyhedron &poly,
                                          unsigned unboundedDims) {
  // We iterate backwards so that whether we remove constraint i - 1 or not, the
  // next constraint to be tested is always i - 2.
  for (unsigned i = poly.getNumEqualities(); i > 0; i--)
    if (eqInvolvesSuffixDims(poly, i - 1, unboundedDims))
      poly.removeEquality(i - 1);
  for (unsigned i = poly.getNumInequalities(); i > 0; i--)
    if (ineqInvolvesSuffixDims(poly, i - 1, unboundedDims))
      poly.removeInequality(i - 1);
}

bool IntegerPolyhedron::isIntegerEmpty() const {
  return !findIntegerSample().hasValue();
}

/// Let this set be S. If S is bounded then we directly call into the GBR
/// sampling algorithm. Otherwise, there are some unbounded directions, i.e.,
/// vectors v such that S extends to infinity along v or -v. In this case we
/// use an algorithm described in the integer set library (isl) manual and used
/// by the isl_set_sample function in that library. The algorithm is:
///
/// 1) Apply a unimodular transform T to S to obtain S*T, such that all
/// dimensions in which S*T is bounded lie in the linear span of a prefix of the
/// dimensions.
///
/// 2) Construct a set B by removing all constraints that involve
/// the unbounded dimensions and then deleting the unbounded dimensions. Note
/// that B is a Bounded set.
///
/// 3) Try to obtain a sample from B using the GBR sampling
/// algorithm. If no sample is found, return that S is empty.
///
/// 4) Otherwise, substitute the obtained sample into S*T to obtain a set
/// C. C is a full-dimensional Cone and always contains a sample.
///
/// 5) Obtain an integer sample from C.
///
/// 6) Return T*v, where v is the concatenation of the samples from B and C.
///
/// The following is a sketch of a proof that
/// a) If the algorithm returns empty, then S is empty.
/// b) If the algorithm returns a sample, it is a valid sample in S.
///
/// The algorithm returns empty only if B is empty, in which case S*T is
/// certainly empty since B was obtained by removing constraints and then
/// deleting unconstrained dimensions from S*T. Since T is unimodular, a vector
/// v is in S*T iff T*v is in S. So in this case, since
/// S*T is empty, S is empty too.
///
/// Otherwise, the algorithm substitutes the sample from B into S*T. All the
/// constraints of S*T that did not involve unbounded dimensions are satisfied
/// by this substitution. All dimensions in the linear span of the dimensions
/// outside the prefix are unbounded in S*T (step 1). Substituting values for
/// the bounded dimensions cannot make these dimensions bounded, and these are
/// the only remaining dimensions in C, so C is unbounded along every vector (in
/// the positive or negative direction, or both). C is hence a full-dimensional
/// cone and therefore always contains an integer point.
///
/// Concatenating the samples from B and C gives a sample v in S*T, so the
/// returned sample T*v is a sample in S.
Optional<SmallVector<int64_t, 8>> IntegerPolyhedron::findIntegerSample() const {
  // First, try the GCD test heuristic.
  if (isEmptyByGCDTest())
    return {};

  Simplex simplex(*this);
  if (simplex.isEmpty())
    return {};

  // For a bounded set, we directly call into the GBR sampling algorithm.
  if (!simplex.isUnbounded())
    return simplex.findIntegerSample();

  // The set is unbounded. We cannot directly use the GBR algorithm.
  //
  // m is a matrix containing, in each row, a vector in which S is
  // bounded, such that the linear span of all these dimensions contains all
  // bounded dimensions in S.
  Matrix m = getBoundedDirections();
  // In column echelon form, each row of m occupies only the first rank(m)
  // columns and has zeros on the other columns. The transform T that brings S
  // to column echelon form is unimodular as well, so this is a suitable
  // transform to use in step 1 of the algorithm.
  std::pair<unsigned, LinearTransform> result =
      LinearTransform::makeTransformToColumnEchelon(std::move(m));
  const LinearTransform &transform = result.second;
  // 1) Apply T to S to obtain S*T.
  IntegerPolyhedron transformedSet = transform.applyTo(*this);

  // 2) Remove the unbounded dimensions and constraints involving them to
  // obtain a bounded set.
  IntegerPolyhedron boundedSet(transformedSet);
  unsigned numBoundedDims = result.first;
  unsigned numUnboundedDims = getNumIds() - numBoundedDims;
  removeConstraintsInvolvingSuffixDims(boundedSet, numUnboundedDims);
  boundedSet.removeIdRange(numBoundedDims, boundedSet.getNumIds());

  // 3) Try to obtain a sample from the bounded set.
  Optional<SmallVector<int64_t, 8>> boundedSample =
      Simplex(boundedSet).findIntegerSample();
  if (!boundedSample)
    return {};
  assert(boundedSet.containsPoint(*boundedSample) &&
         "Simplex returned an invalid sample!");

  // 4) Substitute the values of the bounded dimensions into S*T to obtain a
  // full-dimensional cone, which necessarily contains an integer sample.
  transformedSet.setAndEliminate(0, *boundedSample);
  IntegerPolyhedron &cone = transformedSet;

  // 5) Obtain an integer sample from the cone.
  //
  // We shrink the cone such that for any rational point in the shrunken cone,
  // rounding up each of the point's coordinates produces a point that still
  // lies in the original cone.
  //
  // Rounding up a point x adds a number e_i in [0, 1) to each coordinate x_i.
  // For each inequality sum_i a_i x_i + c >= 0 in the original cone, the
  // shrunken cone will have the inequality tightened by some amount s, such
  // that if x satisfies the shrunken cone's tightened inequality, then x + e
  // satisfies the original inequality, i.e.,
  //
  // sum_i a_i x_i + c + s >= 0 implies sum_i a_i (x_i + e_i) + c >= 0
  //
  // for any e_i values in [0, 1). In fact, we will handle the slightly more
  // general case where e_i can be in [0, 1]. For example, consider the
  // inequality 2x_1 - 3x_2 - 7x_3 - 6 >= 0, and let x = (3, 0, 0). How low
  // could the LHS go if we added a number in [0, 1] to each coordinate? The LHS
  // is minimized when we add 1 to the x_i with negative coefficient a_i and
  // keep the other x_i the same. In the example, we would get x = (3, 1, 1),
  // changing the value of the LHS by -3 + -7 = -10.
  //
  // In general, the value of the LHS can change by at most the sum of the
  // negative a_i, so we accomodate this by shifting the inequality by this
  // amount for the shrunken cone.
  for (unsigned i = 0, e = cone.getNumInequalities(); i < e; ++i) {
    for (unsigned j = 0; j < cone.getNumIds(); ++j) {
      int64_t coeff = cone.atIneq(i, j);
      if (coeff < 0)
        cone.atIneq(i, cone.getNumIds()) += coeff;
    }
  }

  // Obtain an integer sample in the cone by rounding up a rational point from
  // the shrunken cone. Shrinking the cone amounts to shifting its apex
  // "inwards" without changing its "shape"; the shrunken cone is still a
  // full-dimensional cone and is hence non-empty.
  Simplex shrunkenConeSimplex(cone);
  assert(!shrunkenConeSimplex.isEmpty() && "Shrunken cone cannot be empty!");

  // The sample will always exist since the shrunken cone is non-empty.
  SmallVector<Fraction, 8> shrunkenConeSample =
      *shrunkenConeSimplex.getRationalSample();

  SmallVector<int64_t, 8> coneSample(llvm::map_range(shrunkenConeSample, ceil));

  // 6) Return transform * concat(boundedSample, coneSample).
  SmallVector<int64_t, 8> &sample = boundedSample.getValue();
  sample.append(coneSample.begin(), coneSample.end());
  return transform.postMultiplyWithColumn(sample);
}

/// Helper to evaluate an affine expression at a point.
/// The expression is a list of coefficients for the dimensions followed by the
/// constant term.
static int64_t valueAt(ArrayRef<int64_t> expr, ArrayRef<int64_t> point) {
  assert(expr.size() == 1 + point.size() &&
         "Dimensionalities of point and expression don't match!");
  int64_t value = expr.back();
  for (unsigned i = 0; i < point.size(); ++i)
    value += expr[i] * point[i];
  return value;
}

/// A point satisfies an equality iff the value of the equality at the
/// expression is zero, and it satisfies an inequality iff the value of the
/// inequality at that point is non-negative.
bool IntegerPolyhedron::containsPoint(ArrayRef<int64_t> point) const {
  for (unsigned i = 0, e = getNumEqualities(); i < e; ++i) {
    if (valueAt(getEquality(i), point) != 0)
      return false;
  }
  for (unsigned i = 0, e = getNumInequalities(); i < e; ++i) {
    if (valueAt(getInequality(i), point) < 0)
      return false;
  }
  return true;
}

void IntegerPolyhedron::getLocalReprs(std::vector<MaybeLocalRepr> &repr) const {
  std::vector<SmallVector<int64_t, 8>> dividends(getNumLocalIds());
  SmallVector<unsigned, 4> denominators(getNumLocalIds());
  getLocalReprs(dividends, denominators, repr);
}

void IntegerPolyhedron::getLocalReprs(
    std::vector<SmallVector<int64_t, 8>> &dividends,
    SmallVector<unsigned, 4> &denominators) const {
  std::vector<MaybeLocalRepr> repr(getNumLocalIds());
  getLocalReprs(dividends, denominators, repr);
}

void IntegerPolyhedron::getLocalReprs(
    std::vector<SmallVector<int64_t, 8>> &dividends,
    SmallVector<unsigned, 4> &denominators,
    std::vector<MaybeLocalRepr> &repr) const {

  repr.resize(getNumLocalIds());
  dividends.resize(getNumLocalIds());
  denominators.resize(getNumLocalIds());

  SmallVector<bool, 8> foundRepr(getNumIds(), false);
  for (unsigned i = 0, e = getNumDimAndSymbolIds(); i < e; ++i)
    foundRepr[i] = true;

  unsigned divOffset = getNumDimAndSymbolIds();
  bool changed;
  do {
    // Each time changed is true, at end of this iteration, one or more local
    // vars have been detected as floor divs.
    changed = false;
    for (unsigned i = 0, e = getNumLocalIds(); i < e; ++i) {
      if (!foundRepr[i + divOffset]) {
        MaybeLocalRepr res = computeSingleVarRepr(
            *this, foundRepr, divOffset + i, dividends[i], denominators[i]);
        if (!res)
          continue;
        foundRepr[i + divOffset] = true;
        repr[i] = res;
        changed = true;
      }
    }
  } while (changed);

  // Set 0 denominator for identifiers for which no division representation
  // could be found.
  for (unsigned i = 0, e = repr.size(); i < e; ++i)
    if (!repr[i])
      denominators[i] = 0;
}

/// Tightens inequalities given that we are dealing with integer spaces. This is
/// analogous to the GCD test but applied to inequalities. The constant term can
/// be reduced to the preceding multiple of the GCD of the coefficients, i.e.,
///  64*i - 100 >= 0  =>  64*i - 128 >= 0 (since 'i' is an integer). This is a
/// fast method - linear in the number of coefficients.
// Example on how this affects practical cases: consider the scenario:
// 64*i >= 100, j = 64*i; without a tightening, elimination of i would yield
// j >= 100 instead of the tighter (exact) j >= 128.
void IntegerPolyhedron::gcdTightenInequalities() {
  unsigned numCols = getNumCols();
  for (unsigned i = 0, e = getNumInequalities(); i < e; ++i) {
    // Normalize the constraint and tighten the constant term by the GCD.
    uint64_t gcd = inequalities.normalizeRow(i, getNumCols() - 1);
    if (gcd > 1)
      atIneq(i, numCols - 1) = mlir::floorDiv(atIneq(i, numCols - 1), gcd);
  }
}

// Eliminates all identifier variables in column range [posStart, posLimit).
// Returns the number of variables eliminated.
unsigned IntegerPolyhedron::gaussianEliminateIds(unsigned posStart,
                                                 unsigned posLimit) {
  // Return if identifier positions to eliminate are out of range.
  assert(posLimit <= getNumIds());
  assert(hasConsistentState());

  if (posStart >= posLimit)
    return 0;

  gcdTightenInequalities();

  unsigned pivotCol = 0;
  for (pivotCol = posStart; pivotCol < posLimit; ++pivotCol) {
    // Find a row which has a non-zero coefficient in column 'j'.
    unsigned pivotRow;
    if (!findConstraintWithNonZeroAt(pivotCol, /*isEq=*/true, &pivotRow)) {
      // No pivot row in equalities with non-zero at 'pivotCol'.
      if (!findConstraintWithNonZeroAt(pivotCol, /*isEq=*/false, &pivotRow)) {
        // If inequalities are also non-zero in 'pivotCol', it can be
        // eliminated.
        continue;
      }
      break;
    }

    // Eliminate identifier at 'pivotCol' from each equality row.
    for (unsigned i = 0, e = getNumEqualities(); i < e; ++i) {
      eliminateFromConstraint(this, i, pivotRow, pivotCol, posStart,
                              /*isEq=*/true);
      equalities.normalizeRow(i);
    }

    // Eliminate identifier at 'pivotCol' from each inequality row.
    for (unsigned i = 0, e = getNumInequalities(); i < e; ++i) {
      eliminateFromConstraint(this, i, pivotRow, pivotCol, posStart,
                              /*isEq=*/false);
      inequalities.normalizeRow(i);
    }
    removeEquality(pivotRow);
    gcdTightenInequalities();
  }
  // Update position limit based on number eliminated.
  posLimit = pivotCol;
  // Remove eliminated columns from all constraints.
  removeIdRange(posStart, posLimit);
  return posLimit - posStart;
}

// A more complex check to eliminate redundant inequalities. Uses FourierMotzkin
// to check if a constraint is redundant.
void IntegerPolyhedron::removeRedundantInequalities() {
  SmallVector<bool, 32> redun(getNumInequalities(), false);
  // To check if an inequality is redundant, we replace the inequality by its
  // complement (for eg., i - 1 >= 0 by i <= 0), and check if the resulting
  // system is empty. If it is, the inequality is redundant.
  IntegerPolyhedron tmpCst(*this);
  for (unsigned r = 0, e = getNumInequalities(); r < e; r++) {
    // Change the inequality to its complement.
    tmpCst.inequalities.negateRow(r);
    tmpCst.atIneq(r, tmpCst.getNumCols() - 1)--;
    if (tmpCst.isEmpty()) {
      redun[r] = true;
      // Zero fill the redundant inequality.
      inequalities.fillRow(r, /*value=*/0);
      tmpCst.inequalities.fillRow(r, /*value=*/0);
    } else {
      // Reverse the change (to avoid recreating tmpCst each time).
      tmpCst.atIneq(r, tmpCst.getNumCols() - 1)++;
      tmpCst.inequalities.negateRow(r);
    }
  }

  unsigned pos = 0;
  for (unsigned r = 0, e = getNumInequalities(); r < e; ++r) {
    if (!redun[r])
      inequalities.copyRow(r, pos++);
  }
  inequalities.resizeVertically(pos);
}

// A more complex check to eliminate redundant inequalities and equalities. Uses
// Simplex to check if a constraint is redundant.
void IntegerPolyhedron::removeRedundantConstraints() {
  // First, we run gcdTightenInequalities. This allows us to catch some
  // constraints which are not redundant when considering rational solutions
  // but are redundant in terms of integer solutions.
  gcdTightenInequalities();
  Simplex simplex(*this);
  simplex.detectRedundant();

  unsigned pos = 0;
  unsigned numIneqs = getNumInequalities();
  // Scan to get rid of all inequalities marked redundant, in-place. In Simplex,
  // the first constraints added are the inequalities.
  for (unsigned r = 0; r < numIneqs; r++) {
    if (!simplex.isMarkedRedundant(r))
      inequalities.copyRow(r, pos++);
  }
  inequalities.resizeVertically(pos);

  // Scan to get rid of all equalities marked redundant, in-place. In Simplex,
  // after the inequalities, a pair of constraints for each equality is added.
  // An equality is redundant if both the inequalities in its pair are
  // redundant.
  pos = 0;
  for (unsigned r = 0, e = getNumEqualities(); r < e; r++) {
    if (!(simplex.isMarkedRedundant(numIneqs + 2 * r) &&
          simplex.isMarkedRedundant(numIneqs + 2 * r + 1)))
      equalities.copyRow(r, pos++);
  }
  equalities.resizeVertically(pos);
}

Optional<uint64_t> IntegerPolyhedron::computeVolume() const {
  assert(getNumSymbolIds() == 0 && "Symbols are not yet supported!");

  Simplex simplex(*this);
  // If the polytope is rationally empty, there are certainly no integer
  // points.
  if (simplex.isEmpty())
    return 0;

  // Just find the maximum and minimum integer value of each non-local id
  // separately, thus finding the number of integer values each such id can
  // take. Multiplying these together gives a valid overapproximation of the
  // number of integer points in the polyhedron. The result this gives is
  // equivalent to projecting (rationally) the polyhedron onto its non-local ids
  // and returning the number of integer points in a minimal axis-parallel
  // hyperrectangular overapproximation of that.
  //
  // We also handle the special case where one dimension is unbounded and
  // another dimension can take no integer values. In this case, the volume is
  // zero.
  //
  // If there is no such empty dimension, if any dimension is unbounded we
  // just return the result as unbounded.
  uint64_t count = 1;
  SmallVector<int64_t, 8> dim(getNumIds() + 1);
  bool hasUnboundedId = false;
  for (unsigned i = 0, e = getNumDimAndSymbolIds(); i < e; ++i) {
    dim[i] = 1;
    MaybeOptimum<int64_t> min, max;
    std::tie(min, max) = simplex.computeIntegerBounds(dim);
    dim[i] = 0;

    assert((!min.isEmpty() && !max.isEmpty()) &&
           "Polytope should be rationally non-empty!");

    // One of the dimensions is unbounded. Note this fact. We will return
    // unbounded if none of the other dimensions makes the volume zero.
    if (min.isUnbounded() || max.isUnbounded()) {
      hasUnboundedId = true;
      continue;
    }

    // In this case there are no valid integer points and the volume is
    // definitely zero.
    if (min.getBoundedOptimum() > max.getBoundedOptimum())
      return 0;

    count *= (*max - *min + 1);
  }

  if (count == 0)
    return 0;
  if (hasUnboundedId)
    return {};
  return count;
}

void IntegerPolyhedron::eliminateRedundantLocalId(unsigned posA,
                                                  unsigned posB) {
  assert(posA < getNumLocalIds() && "Invalid local id position");
  assert(posB < getNumLocalIds() && "Invalid local id position");

  unsigned localOffset = getIdKindOffset(IdKind::Local);
  posA += localOffset;
  posB += localOffset;
  inequalities.addToColumn(posB, posA, 1);
  equalities.addToColumn(posB, posA, 1);
  removeId(posB);
}

/// Adds additional local ids to the sets such that they both have the union
/// of the local ids in each set, without changing the set of points that
/// lie in `this` and `other`.
///
/// To detect local ids that always take the same in both sets, each local id is
/// represented as a floordiv with constant denominator in terms of other ids.
/// After extracting these divisions, local ids with the same division
/// representation are considered duplicate and are merged. It is possible that
/// division representation for some local id cannot be obtained, and thus these
/// local ids are not considered for detecting duplicates.
void IntegerPolyhedron::mergeLocalIds(IntegerPolyhedron &other) {
  assert(PresburgerSpace::isEqual(other) && "Spaces should match.");

  IntegerPolyhedron &polyA = *this;
  IntegerPolyhedron &polyB = other;

  // Merge local ids of polyA and polyB without using division information,
  // i.e. append local ids of `polyB` to `polyA` and insert local ids of `polyA`
  // to `polyB` at start of its local ids.
  unsigned initLocals = polyA.getNumLocalIds();
  insertId(IdKind::Local, polyA.getNumLocalIds(), polyB.getNumLocalIds());
  polyB.insertId(IdKind::Local, 0, initLocals);

  // Get division representations from each poly.
  std::vector<SmallVector<int64_t, 8>> divsA, divsB;
  SmallVector<unsigned, 4> denomsA, denomsB;
  polyA.getLocalReprs(divsA, denomsA);
  polyB.getLocalReprs(divsB, denomsB);

  // Copy division information for polyB into `divsA` and `denomsA`, so that
  // these have the combined division information of both polys. Since newly
  // added local variables in polyA and polyB have no constraints, they will not
  // have any division representation.
  std::copy(divsB.begin() + initLocals, divsB.end(),
            divsA.begin() + initLocals);
  std::copy(denomsB.begin() + initLocals, denomsB.end(),
            denomsA.begin() + initLocals);

  // Merge function that merges the local variables in both sets by treating
  // them as the same identifier.
  auto merge = [&polyA, &polyB](unsigned i, unsigned j) -> bool {
    polyA.eliminateRedundantLocalId(i, j);
    polyB.eliminateRedundantLocalId(i, j);
    return true;
  };

  // Merge all divisions by removing duplicate divisions.
  unsigned localOffset = getIdKindOffset(IdKind::Local);
  removeDuplicateDivs(divsA, denomsA, localOffset, merge);
}

/// Removes local variables using equalities. Each equality is checked if it
/// can be reduced to the form: `e = affine-expr`, where `e` is a local
/// variable and `affine-expr` is an affine expression not containing `e`.
/// If an equality satisfies this form, the local variable is replaced in
/// each constraint and then removed. The equality used to replace this local
/// variable is also removed.
void IntegerPolyhedron::removeRedundantLocalVars() {
  // Normalize the equality constraints to reduce coefficients of local
  // variables to 1 wherever possible.
  for (unsigned i = 0, e = getNumEqualities(); i < e; ++i)
    equalities.normalizeRow(i);

  while (true) {
    unsigned i, e, j, f;
    for (i = 0, e = getNumEqualities(); i < e; ++i) {
      // Find a local variable to eliminate using ith equality.
      for (j = getNumDimAndSymbolIds(), f = getNumIds(); j < f; ++j)
        if (std::abs(atEq(i, j)) == 1)
          break;

      // Local variable can be eliminated using ith equality.
      if (j < f)
        break;
    }

    // No equality can be used to eliminate a local variable.
    if (i == e)
      break;

    // Use the ith equality to simplify other equalities. If any changes
    // are made to an equality constraint, it is normalized by GCD.
    for (unsigned k = 0, t = getNumEqualities(); k < t; ++k) {
      if (atEq(k, j) != 0) {
        eliminateFromConstraint(this, k, i, j, j, /*isEq=*/true);
        equalities.normalizeRow(k);
      }
    }

    // Use the ith equality to simplify inequalities.
    for (unsigned k = 0, t = getNumInequalities(); k < t; ++k)
      eliminateFromConstraint(this, k, i, j, j, /*isEq=*/false);

    // Remove the ith equality and the found local variable.
    removeId(j);
    removeEquality(i);
  }
}

void IntegerPolyhedron::convertDimToLocal(unsigned dimStart,
                                          unsigned dimLimit) {
  assert(dimLimit <= getNumDimIds() && "Invalid dim pos range");

  if (dimStart >= dimLimit)
    return;

  // Append new local variables corresponding to the dimensions to be converted.
  unsigned convertCount = dimLimit - dimStart;
  unsigned newLocalIdStart = getNumIds();
  appendId(IdKind::Local, convertCount);

  // Swap the new local variables with dimensions.
  for (unsigned i = 0; i < convertCount; ++i)
    swapId(i + dimStart, i + newLocalIdStart);

  // Remove dimensions converted to local variables.
  removeIdRange(dimStart, dimLimit);
}

void IntegerPolyhedron::addBound(BoundType type, unsigned pos, int64_t value) {
  assert(pos < getNumCols());
  if (type == BoundType::EQ) {
    unsigned row = equalities.appendExtraRow();
    equalities(row, pos) = 1;
    equalities(row, getNumCols() - 1) = -value;
  } else {
    unsigned row = inequalities.appendExtraRow();
    inequalities(row, pos) = type == BoundType::LB ? 1 : -1;
    inequalities(row, getNumCols() - 1) =
        type == BoundType::LB ? -value : value;
  }
}

void IntegerPolyhedron::addBound(BoundType type, ArrayRef<int64_t> expr,
                                 int64_t value) {
  assert(type != BoundType::EQ && "EQ not implemented");
  assert(expr.size() == getNumCols());
  unsigned row = inequalities.appendExtraRow();
  for (unsigned i = 0, e = expr.size(); i < e; ++i)
    inequalities(row, i) = type == BoundType::LB ? expr[i] : -expr[i];
  inequalities(inequalities.getNumRows() - 1, getNumCols() - 1) +=
      type == BoundType::LB ? -value : value;
}

/// Adds a new local identifier as the floordiv of an affine function of other
/// identifiers, the coefficients of which are provided in 'dividend' and with
/// respect to a positive constant 'divisor'. Two constraints are added to the
/// system to capture equivalence with the floordiv.
///      q = expr floordiv c    <=>   c*q <= expr <= c*q + c - 1.
void IntegerPolyhedron::addLocalFloorDiv(ArrayRef<int64_t> dividend,
                                         int64_t divisor) {
  assert(dividend.size() == getNumCols() && "incorrect dividend size");
  assert(divisor > 0 && "positive divisor expected");

  appendId(IdKind::Local);

  // Add two constraints for this new identifier 'q'.
  SmallVector<int64_t, 8> bound(dividend.size() + 1);

  // dividend - q * divisor >= 0
  std::copy(dividend.begin(), dividend.begin() + dividend.size() - 1,
            bound.begin());
  bound.back() = dividend.back();
  bound[getNumIds() - 1] = -divisor;
  addInequality(bound);

  // -dividend +qdivisor * q + divisor - 1 >= 0
  std::transform(bound.begin(), bound.end(), bound.begin(),
                 std::negate<int64_t>());
  bound[bound.size() - 1] += divisor - 1;
  addInequality(bound);
}

/// Finds an equality that equates the specified identifier to a constant.
/// Returns the position of the equality row. If 'symbolic' is set to true,
/// symbols are also treated like a constant, i.e., an affine function of the
/// symbols is also treated like a constant. Returns -1 if such an equality
/// could not be found.
static int findEqualityToConstant(const IntegerPolyhedron &cst, unsigned pos,
                                  bool symbolic = false) {
  assert(pos < cst.getNumIds() && "invalid position");
  for (unsigned r = 0, e = cst.getNumEqualities(); r < e; r++) {
    int64_t v = cst.atEq(r, pos);
    if (v * v != 1)
      continue;
    unsigned c;
    unsigned f = symbolic ? cst.getNumDimIds() : cst.getNumIds();
    // This checks for zeros in all positions other than 'pos' in [0, f)
    for (c = 0; c < f; c++) {
      if (c == pos)
        continue;
      if (cst.atEq(r, c) != 0) {
        // Dependent on another identifier.
        break;
      }
    }
    if (c == f)
      // Equality is free of other identifiers.
      return r;
  }
  return -1;
}

LogicalResult IntegerPolyhedron::constantFoldId(unsigned pos) {
  assert(pos < getNumIds() && "invalid position");
  int rowIdx;
  if ((rowIdx = findEqualityToConstant(*this, pos)) == -1)
    return failure();

  // atEq(rowIdx, pos) is either -1 or 1.
  assert(atEq(rowIdx, pos) * atEq(rowIdx, pos) == 1);
  int64_t constVal = -atEq(rowIdx, getNumCols() - 1) / atEq(rowIdx, pos);
  setAndEliminate(pos, constVal);
  return success();
}

void IntegerPolyhedron::constantFoldIdRange(unsigned pos, unsigned num) {
  for (unsigned s = pos, t = pos, e = pos + num; s < e; s++) {
    if (failed(constantFoldId(t)))
      t++;
  }
}

/// Returns a non-negative constant bound on the extent (upper bound - lower
/// bound) of the specified identifier if it is found to be a constant; returns
/// None if it's not a constant. This methods treats symbolic identifiers
/// specially, i.e., it looks for constant differences between affine
/// expressions involving only the symbolic identifiers. See comments at
/// function definition for example. 'lb', if provided, is set to the lower
/// bound associated with the constant difference. Note that 'lb' is purely
/// symbolic and thus will contain the coefficients of the symbolic identifiers
/// and the constant coefficient.
//  Egs: 0 <= i <= 15, return 16.
//       s0 + 2 <= i <= s0 + 17, returns 16. (s0 has to be a symbol)
//       s0 + s1 + 16 <= d0 <= s0 + s1 + 31, returns 16.
//       s0 - 7 <= 8*j <= s0 returns 1 with lb = s0, lbDivisor = 8 (since lb =
//       ceil(s0 - 7 / 8) = floor(s0 / 8)).
Optional<int64_t> IntegerPolyhedron::getConstantBoundOnDimSize(
    unsigned pos, SmallVectorImpl<int64_t> *lb, int64_t *boundFloorDivisor,
    SmallVectorImpl<int64_t> *ub, unsigned *minLbPos,
    unsigned *minUbPos) const {
  assert(pos < getNumDimIds() && "Invalid identifier position");

  // Find an equality for 'pos'^th identifier that equates it to some function
  // of the symbolic identifiers (+ constant).
  int eqPos = findEqualityToConstant(*this, pos, /*symbolic=*/true);
  if (eqPos != -1) {
    auto eq = getEquality(eqPos);
    // If the equality involves a local var, punt for now.
    // TODO: this can be handled in the future by using the explicit
    // representation of the local vars.
    if (!std::all_of(eq.begin() + getNumDimAndSymbolIds(), eq.end() - 1,
                     [](int64_t coeff) { return coeff == 0; }))
      return None;

    // This identifier can only take a single value.
    if (lb) {
      // Set lb to that symbolic value.
      lb->resize(getNumSymbolIds() + 1);
      if (ub)
        ub->resize(getNumSymbolIds() + 1);
      for (unsigned c = 0, f = getNumSymbolIds() + 1; c < f; c++) {
        int64_t v = atEq(eqPos, pos);
        // atEq(eqRow, pos) is either -1 or 1.
        assert(v * v == 1);
        (*lb)[c] = v < 0 ? atEq(eqPos, getNumDimIds() + c) / -v
                         : -atEq(eqPos, getNumDimIds() + c) / v;
        // Since this is an equality, ub = lb.
        if (ub)
          (*ub)[c] = (*lb)[c];
      }
      assert(boundFloorDivisor &&
             "both lb and divisor or none should be provided");
      *boundFloorDivisor = 1;
    }
    if (minLbPos)
      *minLbPos = eqPos;
    if (minUbPos)
      *minUbPos = eqPos;
    return 1;
  }

  // Check if the identifier appears at all in any of the inequalities.
  unsigned r, e;
  for (r = 0, e = getNumInequalities(); r < e; r++) {
    if (atIneq(r, pos) != 0)
      break;
  }
  if (r == e)
    // If it doesn't, there isn't a bound on it.
    return None;

  // Positions of constraints that are lower/upper bounds on the variable.
  SmallVector<unsigned, 4> lbIndices, ubIndices;

  // Gather all symbolic lower bounds and upper bounds of the variable, i.e.,
  // the bounds can only involve symbolic (and local) identifiers. Since the
  // canonical form c_1*x_1 + c_2*x_2 + ... + c_0 >= 0, a constraint is a lower
  // bound for x_i if c_i >= 1, and an upper bound if c_i <= -1.
  getLowerAndUpperBoundIndices(pos, &lbIndices, &ubIndices,
                               /*eqIndices=*/nullptr, /*offset=*/0,
                               /*num=*/getNumDimIds());

  Optional<int64_t> minDiff = None;
  unsigned minLbPosition = 0, minUbPosition = 0;
  for (auto ubPos : ubIndices) {
    for (auto lbPos : lbIndices) {
      // Look for a lower bound and an upper bound that only differ by a
      // constant, i.e., pairs of the form  0 <= c_pos - f(c_i's) <= diffConst.
      // For example, if ii is the pos^th variable, we are looking for
      // constraints like ii >= i, ii <= ii + 50, 50 being the difference. The
      // minimum among all such constant differences is kept since that's the
      // constant bounding the extent of the pos^th variable.
      unsigned j, e;
      for (j = 0, e = getNumCols() - 1; j < e; j++)
        if (atIneq(ubPos, j) != -atIneq(lbPos, j)) {
          break;
        }
      if (j < getNumCols() - 1)
        continue;
      int64_t diff = ceilDiv(atIneq(ubPos, getNumCols() - 1) +
                                 atIneq(lbPos, getNumCols() - 1) + 1,
                             atIneq(lbPos, pos));
      // This bound is non-negative by definition.
      diff = std::max<int64_t>(diff, 0);
      if (minDiff == None || diff < minDiff) {
        minDiff = diff;
        minLbPosition = lbPos;
        minUbPosition = ubPos;
      }
    }
  }
  if (lb && minDiff.hasValue()) {
    // Set lb to the symbolic lower bound.
    lb->resize(getNumSymbolIds() + 1);
    if (ub)
      ub->resize(getNumSymbolIds() + 1);
    // The lower bound is the ceildiv of the lb constraint over the coefficient
    // of the variable at 'pos'. We express the ceildiv equivalently as a floor
    // for uniformity. For eg., if the lower bound constraint was: 32*d0 - N +
    // 31 >= 0, the lower bound for d0 is ceil(N - 31, 32), i.e., floor(N, 32).
    *boundFloorDivisor = atIneq(minLbPosition, pos);
    assert(*boundFloorDivisor == -atIneq(minUbPosition, pos));
    for (unsigned c = 0, e = getNumSymbolIds() + 1; c < e; c++) {
      (*lb)[c] = -atIneq(minLbPosition, getNumDimIds() + c);
    }
    if (ub) {
      for (unsigned c = 0, e = getNumSymbolIds() + 1; c < e; c++)
        (*ub)[c] = atIneq(minUbPosition, getNumDimIds() + c);
    }
    // The lower bound leads to a ceildiv while the upper bound is a floordiv
    // whenever the coefficient at pos != 1. ceildiv (val / d) = floordiv (val +
    // d - 1 / d); hence, the addition of 'atIneq(minLbPosition, pos) - 1' to
    // the constant term for the lower bound.
    (*lb)[getNumSymbolIds()] += atIneq(minLbPosition, pos) - 1;
  }
  if (minLbPos)
    *minLbPos = minLbPosition;
  if (minUbPos)
    *minUbPos = minUbPosition;
  return minDiff;
}

template <bool isLower>
Optional<int64_t>
IntegerPolyhedron::computeConstantLowerOrUpperBound(unsigned pos) {
  assert(pos < getNumIds() && "invalid position");
  // Project to 'pos'.
  projectOut(0, pos);
  projectOut(1, getNumIds() - 1);
  // Check if there's an equality equating the '0'^th identifier to a constant.
  int eqRowIdx = findEqualityToConstant(*this, 0, /*symbolic=*/false);
  if (eqRowIdx != -1)
    // atEq(rowIdx, 0) is either -1 or 1.
    return -atEq(eqRowIdx, getNumCols() - 1) / atEq(eqRowIdx, 0);

  // Check if the identifier appears at all in any of the inequalities.
  unsigned r, e;
  for (r = 0, e = getNumInequalities(); r < e; r++) {
    if (atIneq(r, 0) != 0)
      break;
  }
  if (r == e)
    // If it doesn't, there isn't a bound on it.
    return None;

  Optional<int64_t> minOrMaxConst = None;

  // Take the max across all const lower bounds (or min across all constant
  // upper bounds).
  for (unsigned r = 0, e = getNumInequalities(); r < e; r++) {
    if (isLower) {
      if (atIneq(r, 0) <= 0)
        // Not a lower bound.
        continue;
    } else if (atIneq(r, 0) >= 0) {
      // Not an upper bound.
      continue;
    }
    unsigned c, f;
    for (c = 0, f = getNumCols() - 1; c < f; c++)
      if (c != 0 && atIneq(r, c) != 0)
        break;
    if (c < getNumCols() - 1)
      // Not a constant bound.
      continue;

    int64_t boundConst =
        isLower ? mlir::ceilDiv(-atIneq(r, getNumCols() - 1), atIneq(r, 0))
                : mlir::floorDiv(atIneq(r, getNumCols() - 1), -atIneq(r, 0));
    if (isLower) {
      if (minOrMaxConst == None || boundConst > minOrMaxConst)
        minOrMaxConst = boundConst;
    } else {
      if (minOrMaxConst == None || boundConst < minOrMaxConst)
        minOrMaxConst = boundConst;
    }
  }
  return minOrMaxConst;
}

Optional<int64_t> IntegerPolyhedron::getConstantBound(BoundType type,
                                                      unsigned pos) const {
  if (type == BoundType::LB)
    return IntegerPolyhedron(*this)
        .computeConstantLowerOrUpperBound</*isLower=*/true>(pos);
  if (type == BoundType::UB)
    return IntegerPolyhedron(*this)
        .computeConstantLowerOrUpperBound</*isLower=*/false>(pos);

  assert(type == BoundType::EQ && "expected EQ");
  Optional<int64_t> lb =
      IntegerPolyhedron(*this)
          .computeConstantLowerOrUpperBound</*isLower=*/true>(pos);
  Optional<int64_t> ub =
      IntegerPolyhedron(*this)
          .computeConstantLowerOrUpperBound</*isLower=*/false>(pos);
  return (lb && ub && *lb == *ub) ? Optional<int64_t>(*ub) : None;
}

// A simple (naive and conservative) check for hyper-rectangularity.
bool IntegerPolyhedron::isHyperRectangular(unsigned pos, unsigned num) const {
  assert(pos < getNumCols() - 1);
  // Check for two non-zero coefficients in the range [pos, pos + sum).
  for (unsigned r = 0, e = getNumInequalities(); r < e; r++) {
    unsigned sum = 0;
    for (unsigned c = pos; c < pos + num; c++) {
      if (atIneq(r, c) != 0)
        sum++;
    }
    if (sum > 1)
      return false;
  }
  for (unsigned r = 0, e = getNumEqualities(); r < e; r++) {
    unsigned sum = 0;
    for (unsigned c = pos; c < pos + num; c++) {
      if (atEq(r, c) != 0)
        sum++;
    }
    if (sum > 1)
      return false;
  }
  return true;
}

/// Removes duplicate constraints, trivially true constraints, and constraints
/// that can be detected as redundant as a result of differing only in their
/// constant term part. A constraint of the form <non-negative constant> >= 0 is
/// considered trivially true.
//  Uses a DenseSet to hash and detect duplicates followed by a linear scan to
//  remove duplicates in place.
void IntegerPolyhedron::removeTrivialRedundancy() {
  gcdTightenInequalities();
  normalizeConstraintsByGCD();

  // A map used to detect redundancy stemming from constraints that only differ
  // in their constant term. The value stored is <row position, const term>
  // for a given row.
  SmallDenseMap<ArrayRef<int64_t>, std::pair<unsigned, int64_t>>
      rowsWithoutConstTerm;
  // To unique rows.
  SmallDenseSet<ArrayRef<int64_t>, 8> rowSet;

  // Check if constraint is of the form <non-negative-constant> >= 0.
  auto isTriviallyValid = [&](unsigned r) -> bool {
    for (unsigned c = 0, e = getNumCols() - 1; c < e; c++) {
      if (atIneq(r, c) != 0)
        return false;
    }
    return atIneq(r, getNumCols() - 1) >= 0;
  };

  // Detect and mark redundant constraints.
  SmallVector<bool, 256> redunIneq(getNumInequalities(), false);
  for (unsigned r = 0, e = getNumInequalities(); r < e; r++) {
    int64_t *rowStart = &inequalities(r, 0);
    auto row = ArrayRef<int64_t>(rowStart, getNumCols());
    if (isTriviallyValid(r) || !rowSet.insert(row).second) {
      redunIneq[r] = true;
      continue;
    }

    // Among constraints that only differ in the constant term part, mark
    // everything other than the one with the smallest constant term redundant.
    // (eg: among i - 16j - 5 >= 0, i - 16j - 1 >=0, i - 16j - 7 >= 0, the
    // former two are redundant).
    int64_t constTerm = atIneq(r, getNumCols() - 1);
    auto rowWithoutConstTerm = ArrayRef<int64_t>(rowStart, getNumCols() - 1);
    const auto &ret =
        rowsWithoutConstTerm.insert({rowWithoutConstTerm, {r, constTerm}});
    if (!ret.second) {
      // Check if the other constraint has a higher constant term.
      auto &val = ret.first->second;
      if (val.second > constTerm) {
        // The stored row is redundant. Mark it so, and update with this one.
        redunIneq[val.first] = true;
        val = {r, constTerm};
      } else {
        // The one stored makes this one redundant.
        redunIneq[r] = true;
      }
    }
  }

  // Scan to get rid of all rows marked redundant, in-place.
  unsigned pos = 0;
  for (unsigned r = 0, e = getNumInequalities(); r < e; r++)
    if (!redunIneq[r])
      inequalities.copyRow(r, pos++);

  inequalities.resizeVertically(pos);

  // TODO: consider doing this for equalities as well, but probably not worth
  // the savings.
}

#undef DEBUG_TYPE
#define DEBUG_TYPE "fm"

/// Eliminates identifier at the specified position using Fourier-Motzkin
/// variable elimination. This technique is exact for rational spaces but
/// conservative (in "rare" cases) for integer spaces. The operation corresponds
/// to a projection operation yielding the (convex) set of integer points
/// contained in the rational shadow of the set. An emptiness test that relies
/// on this method will guarantee emptiness, i.e., it disproves the existence of
/// a solution if it says it's empty.
/// If a non-null isResultIntegerExact is passed, it is set to true if the
/// result is also integer exact. If it's set to false, the obtained solution
/// *may* not be exact, i.e., it may contain integer points that do not have an
/// integer pre-image in the original set.
///
/// Eg:
/// j >= 0, j <= i + 1
/// i >= 0, i <= N + 1
/// Eliminating i yields,
///   j >= 0, 0 <= N + 1, j - 1 <= N + 1
///
/// If darkShadow = true, this method computes the dark shadow on elimination;
/// the dark shadow is a convex integer subset of the exact integer shadow. A
/// non-empty dark shadow proves the existence of an integer solution. The
/// elimination in such a case could however be an under-approximation, and thus
/// should not be used for scanning sets or used by itself for dependence
/// checking.
///
/// Eg: 2-d set, * represents grid points, 'o' represents a point in the set.
///            ^
///            |
///            | * * * * o o
///         i  | * * o o o o
///            | o * * * * *
///            --------------->
///                 j ->
///
/// Eliminating i from this system (projecting on the j dimension):
/// rational shadow / integer light shadow:  1 <= j <= 6
/// dark shadow:                             3 <= j <= 6
/// exact integer shadow:                    j = 1 \union  3 <= j <= 6
/// holes/splinters:                         j = 2
///
/// darkShadow = false, isResultIntegerExact = nullptr are default values.
// TODO: a slight modification to yield dark shadow version of FM (tightened),
// which can prove the existence of a solution if there is one.
void IntegerPolyhedron::fourierMotzkinEliminate(unsigned pos, bool darkShadow,
                                                bool *isResultIntegerExact) {
  LLVM_DEBUG(llvm::dbgs() << "FM input (eliminate pos " << pos << "):\n");
  LLVM_DEBUG(dump());
  assert(pos < getNumIds() && "invalid position");
  assert(hasConsistentState());

  // Check if this identifier can be eliminated through a substitution.
  for (unsigned r = 0, e = getNumEqualities(); r < e; r++) {
    if (atEq(r, pos) != 0) {
      // Use Gaussian elimination here (since we have an equality).
      LogicalResult ret = gaussianEliminateId(pos);
      (void)ret;
      assert(succeeded(ret) && "Gaussian elimination guaranteed to succeed");
      LLVM_DEBUG(llvm::dbgs() << "FM output (through Gaussian elimination):\n");
      LLVM_DEBUG(dump());
      return;
    }
  }

  // A fast linear time tightening.
  gcdTightenInequalities();

  // Check if the identifier appears at all in any of the inequalities.
  if (isColZero(pos)) {
    // If it doesn't appear, just remove the column and return.
    // TODO: refactor removeColumns to use it from here.
    removeId(pos);
    LLVM_DEBUG(llvm::dbgs() << "FM output:\n");
    LLVM_DEBUG(dump());
    return;
  }

  // Positions of constraints that are lower bounds on the variable.
  SmallVector<unsigned, 4> lbIndices;
  // Positions of constraints that are lower bounds on the variable.
  SmallVector<unsigned, 4> ubIndices;
  // Positions of constraints that do not involve the variable.
  std::vector<unsigned> nbIndices;
  nbIndices.reserve(getNumInequalities());

  // Gather all lower bounds and upper bounds of the variable. Since the
  // canonical form c_1*x_1 + c_2*x_2 + ... + c_0 >= 0, a constraint is a lower
  // bound for x_i if c_i >= 1, and an upper bound if c_i <= -1.
  for (unsigned r = 0, e = getNumInequalities(); r < e; r++) {
    if (atIneq(r, pos) == 0) {
      // Id does not appear in bound.
      nbIndices.push_back(r);
    } else if (atIneq(r, pos) >= 1) {
      // Lower bound.
      lbIndices.push_back(r);
    } else {
      // Upper bound.
      ubIndices.push_back(r);
    }
  }

  // Set the number of dimensions, symbols, locals in the resulting system.
  unsigned newNumDims =
      getNumDimIds() - getIdKindOverlap(IdKind::SetDim, pos, pos + 1);
  unsigned newNumSymbols =
      getNumSymbolIds() - getIdKindOverlap(IdKind::Symbol, pos, pos + 1);
  unsigned newNumLocals =
      getNumLocalIds() - getIdKindOverlap(IdKind::Local, pos, pos + 1);

  /// Create the new system which has one identifier less.
  IntegerPolyhedron newPoly(lbIndices.size() * ubIndices.size() +
                                nbIndices.size(),
                            getNumEqualities(), getNumCols() - 1, newNumDims,
                            newNumSymbols, newNumLocals);

  // This will be used to check if the elimination was integer exact.
  unsigned lcmProducts = 1;

  // Let x be the variable we are eliminating.
  // For each lower bound, lb <= c_l*x, and each upper bound c_u*x <= ub, (note
  // that c_l, c_u >= 1) we have:
  // lb*lcm(c_l, c_u)/c_l <= lcm(c_l, c_u)*x <= ub*lcm(c_l, c_u)/c_u
  // We thus generate a constraint:
  // lcm(c_l, c_u)/c_l*lb <= lcm(c_l, c_u)/c_u*ub.
  // Note if c_l = c_u = 1, all integer points captured by the resulting
  // constraint correspond to integer points in the original system (i.e., they
  // have integer pre-images). Hence, if the lcm's are all 1, the elimination is
  // integer exact.
  for (auto ubPos : ubIndices) {
    for (auto lbPos : lbIndices) {
      SmallVector<int64_t, 4> ineq;
      ineq.reserve(newPoly.getNumCols());
      int64_t lbCoeff = atIneq(lbPos, pos);
      // Note that in the comments above, ubCoeff is the negation of the
      // coefficient in the canonical form as the view taken here is that of the
      // term being moved to the other size of '>='.
      int64_t ubCoeff = -atIneq(ubPos, pos);
      // TODO: refactor this loop to avoid all branches inside.
      for (unsigned l = 0, e = getNumCols(); l < e; l++) {
        if (l == pos)
          continue;
        assert(lbCoeff >= 1 && ubCoeff >= 1 && "bounds wrongly identified");
        int64_t lcm = mlir::lcm(lbCoeff, ubCoeff);
        ineq.push_back(atIneq(ubPos, l) * (lcm / ubCoeff) +
                       atIneq(lbPos, l) * (lcm / lbCoeff));
        lcmProducts *= lcm;
      }
      if (darkShadow) {
        // The dark shadow is a convex subset of the exact integer shadow. If
        // there is a point here, it proves the existence of a solution.
        ineq[ineq.size() - 1] += lbCoeff * ubCoeff - lbCoeff - ubCoeff + 1;
      }
      // TODO: we need to have a way to add inequalities in-place in
      // IntegerPolyhedron instead of creating and copying over.
      newPoly.addInequality(ineq);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "FM isResultIntegerExact: " << (lcmProducts == 1)
                          << "\n");
  if (lcmProducts == 1 && isResultIntegerExact)
    *isResultIntegerExact = true;

  // Copy over the constraints not involving this variable.
  for (auto nbPos : nbIndices) {
    SmallVector<int64_t, 4> ineq;
    ineq.reserve(getNumCols() - 1);
    for (unsigned l = 0, e = getNumCols(); l < e; l++) {
      if (l == pos)
        continue;
      ineq.push_back(atIneq(nbPos, l));
    }
    newPoly.addInequality(ineq);
  }

  assert(newPoly.getNumConstraints() ==
         lbIndices.size() * ubIndices.size() + nbIndices.size());

  // Copy over the equalities.
  for (unsigned r = 0, e = getNumEqualities(); r < e; r++) {
    SmallVector<int64_t, 4> eq;
    eq.reserve(newPoly.getNumCols());
    for (unsigned l = 0, e = getNumCols(); l < e; l++) {
      if (l == pos)
        continue;
      eq.push_back(atEq(r, l));
    }
    newPoly.addEquality(eq);
  }

  // GCD tightening and normalization allows detection of more trivially
  // redundant constraints.
  newPoly.gcdTightenInequalities();
  newPoly.normalizeConstraintsByGCD();
  newPoly.removeTrivialRedundancy();
  clearAndCopyFrom(newPoly);
  LLVM_DEBUG(llvm::dbgs() << "FM output:\n");
  LLVM_DEBUG(dump());
}

#undef DEBUG_TYPE
#define DEBUG_TYPE "presburger"

void IntegerPolyhedron::projectOut(unsigned pos, unsigned num) {
  if (num == 0)
    return;

  // 'pos' can be at most getNumCols() - 2 if num > 0.
  assert((getNumCols() < 2 || pos <= getNumCols() - 2) && "invalid position");
  assert(pos + num < getNumCols() && "invalid range");

  // Eliminate as many identifiers as possible using Gaussian elimination.
  unsigned currentPos = pos;
  unsigned numToEliminate = num;
  unsigned numGaussianEliminated = 0;

  while (currentPos < getNumIds()) {
    unsigned curNumEliminated =
        gaussianEliminateIds(currentPos, currentPos + numToEliminate);
    ++currentPos;
    numToEliminate -= curNumEliminated + 1;
    numGaussianEliminated += curNumEliminated;
  }

  // Eliminate the remaining using Fourier-Motzkin.
  for (unsigned i = 0; i < num - numGaussianEliminated; i++) {
    unsigned numToEliminate = num - numGaussianEliminated - i;
    fourierMotzkinEliminate(
        getBestIdToEliminate(*this, pos, pos + numToEliminate));
  }

  // Fast/trivial simplifications.
  gcdTightenInequalities();
  // Normalize constraints after tightening since the latter impacts this, but
  // not the other way round.
  normalizeConstraintsByGCD();
}

namespace {

enum BoundCmpResult { Greater, Less, Equal, Unknown };

/// Compares two affine bounds whose coefficients are provided in 'first' and
/// 'second'. The last coefficient is the constant term.
static BoundCmpResult compareBounds(ArrayRef<int64_t> a, ArrayRef<int64_t> b) {
  assert(a.size() == b.size());

  // For the bounds to be comparable, their corresponding identifier
  // coefficients should be equal; the constant terms are then compared to
  // determine less/greater/equal.

  if (!std::equal(a.begin(), a.end() - 1, b.begin()))
    return Unknown;

  if (a.back() == b.back())
    return Equal;

  return a.back() < b.back() ? Less : Greater;
}
} // namespace

// Returns constraints that are common to both A & B.
static void getCommonConstraints(const IntegerPolyhedron &a,
                                 const IntegerPolyhedron &b,
                                 IntegerPolyhedron &c) {
  c = IntegerPolyhedron(a.getNumDimIds(), a.getNumSymbolIds(),
                        a.getNumLocalIds());
  // a naive O(n^2) check should be enough here given the input sizes.
  for (unsigned r = 0, e = a.getNumInequalities(); r < e; ++r) {
    for (unsigned s = 0, f = b.getNumInequalities(); s < f; ++s) {
      if (a.getInequality(r) == b.getInequality(s)) {
        c.addInequality(a.getInequality(r));
        break;
      }
    }
  }
  for (unsigned r = 0, e = a.getNumEqualities(); r < e; ++r) {
    for (unsigned s = 0, f = b.getNumEqualities(); s < f; ++s) {
      if (a.getEquality(r) == b.getEquality(s)) {
        c.addEquality(a.getEquality(r));
        break;
      }
    }
  }
}

// Computes the bounding box with respect to 'other' by finding the min of the
// lower bounds and the max of the upper bounds along each of the dimensions.
LogicalResult
IntegerPolyhedron::unionBoundingBox(const IntegerPolyhedron &otherCst) {
  assert(PresburgerLocalSpace::isEqual(otherCst) && "Spaces should match.");
  assert(getNumLocalIds() == 0 && "local ids not supported yet here");

  // Get the constraints common to both systems; these will be added as is to
  // the union.
  IntegerPolyhedron commonCst;
  getCommonConstraints(*this, otherCst, commonCst);

  std::vector<SmallVector<int64_t, 8>> boundingLbs;
  std::vector<SmallVector<int64_t, 8>> boundingUbs;
  boundingLbs.reserve(2 * getNumDimIds());
  boundingUbs.reserve(2 * getNumDimIds());

  // To hold lower and upper bounds for each dimension.
  SmallVector<int64_t, 4> lb, otherLb, ub, otherUb;
  // To compute min of lower bounds and max of upper bounds for each dimension.
  SmallVector<int64_t, 4> minLb(getNumSymbolIds() + 1);
  SmallVector<int64_t, 4> maxUb(getNumSymbolIds() + 1);
  // To compute final new lower and upper bounds for the union.
  SmallVector<int64_t, 8> newLb(getNumCols()), newUb(getNumCols());

  int64_t lbFloorDivisor, otherLbFloorDivisor;
  for (unsigned d = 0, e = getNumDimIds(); d < e; ++d) {
    auto extent = getConstantBoundOnDimSize(d, &lb, &lbFloorDivisor, &ub);
    if (!extent.hasValue())
      // TODO: symbolic extents when necessary.
      // TODO: handle union if a dimension is unbounded.
      return failure();

    auto otherExtent = otherCst.getConstantBoundOnDimSize(
        d, &otherLb, &otherLbFloorDivisor, &otherUb);
    if (!otherExtent.hasValue() || lbFloorDivisor != otherLbFloorDivisor)
      // TODO: symbolic extents when necessary.
      return failure();

    assert(lbFloorDivisor > 0 && "divisor always expected to be positive");

    auto res = compareBounds(lb, otherLb);
    // Identify min.
    if (res == BoundCmpResult::Less || res == BoundCmpResult::Equal) {
      minLb = lb;
      // Since the divisor is for a floordiv, we need to convert to ceildiv,
      // i.e., i >= expr floordiv div <=> i >= (expr - div + 1) ceildiv div <=>
      // div * i >= expr - div + 1.
      minLb.back() -= lbFloorDivisor - 1;
    } else if (res == BoundCmpResult::Greater) {
      minLb = otherLb;
      minLb.back() -= otherLbFloorDivisor - 1;
    } else {
      // Uncomparable - check for constant lower/upper bounds.
      auto constLb = getConstantBound(BoundType::LB, d);
      auto constOtherLb = otherCst.getConstantBound(BoundType::LB, d);
      if (!constLb.hasValue() || !constOtherLb.hasValue())
        return failure();
      std::fill(minLb.begin(), minLb.end(), 0);
      minLb.back() = std::min(constLb.getValue(), constOtherLb.getValue());
    }

    // Do the same for ub's but max of upper bounds. Identify max.
    auto uRes = compareBounds(ub, otherUb);
    if (uRes == BoundCmpResult::Greater || uRes == BoundCmpResult::Equal) {
      maxUb = ub;
    } else if (uRes == BoundCmpResult::Less) {
      maxUb = otherUb;
    } else {
      // Uncomparable - check for constant lower/upper bounds.
      auto constUb = getConstantBound(BoundType::UB, d);
      auto constOtherUb = otherCst.getConstantBound(BoundType::UB, d);
      if (!constUb.hasValue() || !constOtherUb.hasValue())
        return failure();
      std::fill(maxUb.begin(), maxUb.end(), 0);
      maxUb.back() = std::max(constUb.getValue(), constOtherUb.getValue());
    }

    std::fill(newLb.begin(), newLb.end(), 0);
    std::fill(newUb.begin(), newUb.end(), 0);

    // The divisor for lb, ub, otherLb, otherUb at this point is lbDivisor,
    // and so it's the divisor for newLb and newUb as well.
    newLb[d] = lbFloorDivisor;
    newUb[d] = -lbFloorDivisor;
    // Copy over the symbolic part + constant term.
    std::copy(minLb.begin(), minLb.end(), newLb.begin() + getNumDimIds());
    std::transform(newLb.begin() + getNumDimIds(), newLb.end(),
                   newLb.begin() + getNumDimIds(), std::negate<int64_t>());
    std::copy(maxUb.begin(), maxUb.end(), newUb.begin() + getNumDimIds());

    boundingLbs.push_back(newLb);
    boundingUbs.push_back(newUb);
  }

  // Clear all constraints and add the lower/upper bounds for the bounding box.
  clearConstraints();
  for (unsigned d = 0, e = getNumDimIds(); d < e; ++d) {
    addInequality(boundingLbs[d]);
    addInequality(boundingUbs[d]);
  }

  // Add the constraints that were common to both systems.
  append(commonCst);
  removeTrivialRedundancy();

  // TODO: copy over pure symbolic constraints from this and 'other' over to the
  // union (since the above are just the union along dimensions); we shouldn't
  // be discarding any other constraints on the symbols.

  return success();
}

bool IntegerPolyhedron::isColZero(unsigned pos) const {
  unsigned rowPos;
  return !findConstraintWithNonZeroAt(pos, /*isEq=*/false, &rowPos) &&
         !findConstraintWithNonZeroAt(pos, /*isEq=*/true, &rowPos);
}

/// Find positions of inequalities and equalities that do not have a coefficient
/// for [pos, pos + num) identifiers.
static void getIndependentConstraints(const IntegerPolyhedron &cst,
                                      unsigned pos, unsigned num,
                                      SmallVectorImpl<unsigned> &nbIneqIndices,
                                      SmallVectorImpl<unsigned> &nbEqIndices) {
  assert(pos < cst.getNumIds() && "invalid start position");
  assert(pos + num <= cst.getNumIds() && "invalid limit");

  for (unsigned r = 0, e = cst.getNumInequalities(); r < e; r++) {
    // The bounds are to be independent of [offset, offset + num) columns.
    unsigned c;
    for (c = pos; c < pos + num; ++c) {
      if (cst.atIneq(r, c) != 0)
        break;
    }
    if (c == pos + num)
      nbIneqIndices.push_back(r);
  }

  for (unsigned r = 0, e = cst.getNumEqualities(); r < e; r++) {
    // The bounds are to be independent of [offset, offset + num) columns.
    unsigned c;
    for (c = pos; c < pos + num; ++c) {
      if (cst.atEq(r, c) != 0)
        break;
    }
    if (c == pos + num)
      nbEqIndices.push_back(r);
  }
}

void IntegerPolyhedron::removeIndependentConstraints(unsigned pos,
                                                     unsigned num) {
  assert(pos + num <= getNumIds() && "invalid range");

  // Remove constraints that are independent of these identifiers.
  SmallVector<unsigned, 4> nbIneqIndices, nbEqIndices;
  getIndependentConstraints(*this, /*pos=*/0, num, nbIneqIndices, nbEqIndices);

  // Iterate in reverse so that indices don't have to be updated.
  // TODO: This method can be made more efficient (because removal of each
  // inequality leads to much shifting/copying in the underlying buffer).
  for (auto nbIndex : llvm::reverse(nbIneqIndices))
    removeInequality(nbIndex);
  for (auto nbIndex : llvm::reverse(nbEqIndices))
    removeEquality(nbIndex);
}

void IntegerPolyhedron::printSpace(raw_ostream &os) const {
  os << "\nConstraints (" << getNumDimIds() << " dims, " << getNumSymbolIds()
     << " symbols, " << getNumLocalIds() << " locals), (" << getNumConstraints()
     << " constraints)\n";
}

void IntegerPolyhedron::print(raw_ostream &os) const {
  assert(hasConsistentState());
  printSpace(os);
  for (unsigned i = 0, e = getNumEqualities(); i < e; ++i) {
    for (unsigned j = 0, f = getNumCols(); j < f; ++j) {
      os << atEq(i, j) << " ";
    }
    os << "= 0\n";
  }
  for (unsigned i = 0, e = getNumInequalities(); i < e; ++i) {
    for (unsigned j = 0, f = getNumCols(); j < f; ++j) {
      os << atIneq(i, j) << " ";
    }
    os << ">= 0\n";
  }
  os << '\n';
}

void IntegerPolyhedron::dump() const { print(llvm::errs()); }
