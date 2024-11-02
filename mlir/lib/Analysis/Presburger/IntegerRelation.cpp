//===- IntegerRelation.cpp - MLIR IntegerRelation Class ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A class to represent an relation over integer tuples. A relation is
// represented as a constraint system over a space of tuples of integer valued
// variables supporting symbolic variables and existential quantification.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/Fraction.h"
#include "mlir/Analysis/Presburger/LinearTransform.h"
#include "mlir/Analysis/Presburger/MPInt.h"
#include "mlir/Analysis/Presburger/PWMAFunction.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Analysis/Presburger/Utils.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#define DEBUG_TYPE "presburger"

using namespace mlir;
using namespace presburger;

using llvm::SmallDenseMap;
using llvm::SmallDenseSet;

std::unique_ptr<IntegerRelation> IntegerRelation::clone() const {
  return std::make_unique<IntegerRelation>(*this);
}

std::unique_ptr<IntegerPolyhedron> IntegerPolyhedron::clone() const {
  return std::make_unique<IntegerPolyhedron>(*this);
}

void IntegerRelation::setSpace(const PresburgerSpace &oSpace) {
  assert(space.getNumVars() == oSpace.getNumVars() && "invalid space!");
  space = oSpace;
}

void IntegerRelation::setSpaceExceptLocals(const PresburgerSpace &oSpace) {
  assert(oSpace.getNumLocalVars() == 0 && "no locals should be present!");
  assert(oSpace.getNumVars() <= getNumVars() && "invalid space!");
  unsigned newNumLocals = getNumVars() - oSpace.getNumVars();
  space = oSpace;
  space.insertVar(VarKind::Local, 0, newNumLocals);
}

void IntegerRelation::append(const IntegerRelation &other) {
  assert(space.isEqual(other.getSpace()) && "Spaces must be equal.");

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

IntegerRelation IntegerRelation::intersect(IntegerRelation other) const {
  IntegerRelation result = *this;
  result.mergeLocalVars(other);
  result.append(other);
  return result;
}

bool IntegerRelation::isEqual(const IntegerRelation &other) const {
  assert(space.isCompatible(other.getSpace()) && "Spaces must be compatible.");
  return PresburgerRelation(*this).isEqual(PresburgerRelation(other));
}

bool IntegerRelation::isObviouslyEqual(const IntegerRelation &other) const {
  if (!space.isEqual(other.getSpace()))
    return false;
  if (getNumEqualities() != other.getNumEqualities())
    return false;
  if (getNumInequalities() != other.getNumInequalities())
    return false;

  unsigned cols = getNumCols();
  for (unsigned i = 0, eqs = getNumEqualities(); i < eqs; ++i) {
    for (unsigned j = 0; j < cols; ++j) {
      if (atEq(i, j) != other.atEq(i, j))
        return false;
    }
  }
  for (unsigned i = 0, ineqs = getNumInequalities(); i < ineqs; ++i) {
    for (unsigned j = 0; j < cols; ++j) {
      if (atIneq(i, j) != other.atIneq(i, j))
        return false;
    }
  }
  return true;
}

bool IntegerRelation::isSubsetOf(const IntegerRelation &other) const {
  assert(space.isCompatible(other.getSpace()) && "Spaces must be compatible.");
  return PresburgerRelation(*this).isSubsetOf(PresburgerRelation(other));
}

MaybeOptimum<SmallVector<Fraction, 8>>
IntegerRelation::findRationalLexMin() const {
  assert(getNumSymbolVars() == 0 && "Symbols are not supported!");
  MaybeOptimum<SmallVector<Fraction, 8>> maybeLexMin =
      LexSimplex(*this).findRationalLexMin();

  if (!maybeLexMin.isBounded())
    return maybeLexMin;

  // The Simplex returns the lexmin over all the variables including locals. But
  // locals are not actually part of the space and should not be returned in the
  // result. Since the locals are placed last in the list of variables, they
  // will be minimized last in the lexmin. So simply truncating out the locals
  // from the end of the answer gives the desired lexmin over the dimensions.
  assert(maybeLexMin->size() == getNumVars() &&
         "Incorrect number of vars in lexMin!");
  maybeLexMin->resize(getNumDimAndSymbolVars());
  return maybeLexMin;
}

MaybeOptimum<SmallVector<MPInt, 8>> IntegerRelation::findIntegerLexMin() const {
  assert(getNumSymbolVars() == 0 && "Symbols are not supported!");
  MaybeOptimum<SmallVector<MPInt, 8>> maybeLexMin =
      LexSimplex(*this).findIntegerLexMin();

  if (!maybeLexMin.isBounded())
    return maybeLexMin.getKind();

  // The Simplex returns the lexmin over all the variables including locals. But
  // locals are not actually part of the space and should not be returned in the
  // result. Since the locals are placed last in the list of variables, they
  // will be minimized last in the lexmin. So simply truncating out the locals
  // from the end of the answer gives the desired lexmin over the dimensions.
  assert(maybeLexMin->size() == getNumVars() &&
         "Incorrect number of vars in lexMin!");
  maybeLexMin->resize(getNumDimAndSymbolVars());
  return maybeLexMin;
}

static bool rangeIsZero(ArrayRef<MPInt> range) {
  return llvm::all_of(range, [](const MPInt &x) { return x == 0; });
}

static void removeConstraintsInvolvingVarRange(IntegerRelation &poly,
                                               unsigned begin, unsigned count) {
  // We loop until i > 0 and index into i - 1 to avoid sign issues.
  //
  // We iterate backwards so that whether we remove constraint i - 1 or not, the
  // next constraint to be tested is always i - 2.
  for (unsigned i = poly.getNumEqualities(); i > 0; i--)
    if (!rangeIsZero(poly.getEquality(i - 1).slice(begin, count)))
      poly.removeEquality(i - 1);
  for (unsigned i = poly.getNumInequalities(); i > 0; i--)
    if (!rangeIsZero(poly.getInequality(i - 1).slice(begin, count)))
      poly.removeInequality(i - 1);
}

IntegerRelation::CountsSnapshot IntegerRelation::getCounts() const {
  return {getSpace(), getNumInequalities(), getNumEqualities()};
}

void IntegerRelation::truncateVarKind(VarKind kind, unsigned num) {
  unsigned curNum = getNumVarKind(kind);
  assert(num <= curNum && "Can't truncate to more vars!");
  removeVarRange(kind, num, curNum);
}

void IntegerRelation::truncateVarKind(VarKind kind,
                                      const CountsSnapshot &counts) {
  truncateVarKind(kind, counts.getSpace().getNumVarKind(kind));
}

void IntegerRelation::truncate(const CountsSnapshot &counts) {
  truncateVarKind(VarKind::Domain, counts);
  truncateVarKind(VarKind::Range, counts);
  truncateVarKind(VarKind::Symbol, counts);
  truncateVarKind(VarKind::Local, counts);
  removeInequalityRange(counts.getNumIneqs(), getNumInequalities());
  removeEqualityRange(counts.getNumEqs(), getNumEqualities());
}

PresburgerRelation IntegerRelation::computeReprWithOnlyDivLocals() const {
  // If there are no locals, we're done.
  if (getNumLocalVars() == 0)
    return PresburgerRelation(*this);

  // Move all the non-div locals to the end, as the current API to
  // SymbolicLexOpt requires these to form a contiguous range.
  //
  // Take a copy so we can perform mutations.
  IntegerRelation copy = *this;
  std::vector<MaybeLocalRepr> reprs(getNumLocalVars());
  copy.getLocalReprs(&reprs);

  // Iterate through all the locals. The last `numNonDivLocals` are the locals
  // that have been scanned already and do not have division representations.
  unsigned numNonDivLocals = 0;
  unsigned offset = copy.getVarKindOffset(VarKind::Local);
  for (unsigned i = 0, e = copy.getNumLocalVars(); i < e - numNonDivLocals;) {
    if (!reprs[i]) {
      // Whenever we come across a local that does not have a division
      // representation, we swap it to the `numNonDivLocals`-th last position
      // and increment `numNonDivLocal`s. `reprs` also needs to be swapped.
      copy.swapVar(offset + i, offset + e - numNonDivLocals - 1);
      std::swap(reprs[i], reprs[e - numNonDivLocals - 1]);
      ++numNonDivLocals;
      continue;
    }
    ++i;
  }

  // If there are no non-div locals, we're done.
  if (numNonDivLocals == 0)
    return PresburgerRelation(*this);

  // We computeSymbolicIntegerLexMin by considering the non-div locals as
  // "non-symbols" and considering everything else as "symbols". This will
  // compute a function mapping assignments to "symbols" to the
  // lexicographically minimal valid assignment of "non-symbols", when a
  // satisfying assignment exists. It separately returns the set of assignments
  // to the "symbols" such that a satisfying assignment to the "non-symbols"
  // exists but the lexmin is unbounded. We basically want to find the set of
  // values of the "symbols" such that an assignment to the "non-symbols"
  // exists, which is the union of the domain of the returned lexmin function
  // and the returned set of assignments to the "symbols" that makes the lexmin
  // unbounded.
  SymbolicLexOpt lexminResult =
      SymbolicLexSimplex(copy, /*symbolOffset*/ 0,
                         IntegerPolyhedron(PresburgerSpace::getSetSpace(
                             /*numDims=*/copy.getNumVars() - numNonDivLocals)))
          .computeSymbolicIntegerLexMin();
  PresburgerRelation result =
      lexminResult.lexopt.getDomain().unionSet(lexminResult.unboundedDomain);

  // The result set might lie in the wrong space -- all its ids are dims.
  // Set it to the desired space and return.
  PresburgerSpace space = getSpace();
  space.removeVarRange(VarKind::Local, 0, getNumLocalVars());
  result.setSpace(space);
  return result;
}

SymbolicLexOpt IntegerRelation::findSymbolicIntegerLexMin() const {
  // Symbol and Domain vars will be used as symbols for symbolic lexmin.
  // In other words, for every value of the symbols and domain, return the
  // lexmin value of the (range, locals).
  llvm::SmallBitVector isSymbol(getNumVars(), false);
  isSymbol.set(getVarKindOffset(VarKind::Symbol),
               getVarKindEnd(VarKind::Symbol));
  isSymbol.set(getVarKindOffset(VarKind::Domain),
               getVarKindEnd(VarKind::Domain));
  // Compute the symbolic lexmin of the dims and locals, with the symbols being
  // the actual symbols of this set.
  // The resultant space of lexmin is the space of the relation itself.
  SymbolicLexOpt result =
      SymbolicLexSimplex(*this,
                         IntegerPolyhedron(PresburgerSpace::getSetSpace(
                             /*numDims=*/getNumDomainVars(),
                             /*numSymbols=*/getNumSymbolVars())),
                         isSymbol)
          .computeSymbolicIntegerLexMin();

  // We want to return only the lexmin over the dims, so strip the locals from
  // the computed lexmin.
  result.lexopt.removeOutputs(result.lexopt.getNumOutputs() - getNumLocalVars(),
                              result.lexopt.getNumOutputs());
  return result;
}

/// findSymbolicIntegerLexMax is implemented using findSymbolicIntegerLexMin as
/// follows:
/// 1. A new relation is created which is `this` relation with the sign of
/// each dimension variable in range flipped;
/// 2. findSymbolicIntegerLexMin is called on the range negated relation to
/// compute the negated lexmax of `this` relation;
/// 3. The sign of the negated lexmax is flipped and returned.
SymbolicLexOpt IntegerRelation::findSymbolicIntegerLexMax() const {
  IntegerRelation flippedRel = *this;
  // Flip range sign by flipping the sign of range variables in all constraints.
  for (unsigned j = getNumDomainVars(),
                b = getNumDomainVars() + getNumRangeVars();
       j < b; j++) {
    for (unsigned i = 0, a = getNumEqualities(); i < a; i++)
      flippedRel.atEq(i, j) = -1 * atEq(i, j);
    for (unsigned i = 0, a = getNumInequalities(); i < a; i++)
      flippedRel.atIneq(i, j) = -1 * atIneq(i, j);
  }
  // Compute negated lexmax by computing lexmin.
  SymbolicLexOpt flippedSymbolicIntegerLexMax =
                     flippedRel.findSymbolicIntegerLexMin(),
                 symbolicIntegerLexMax(
                     flippedSymbolicIntegerLexMax.lexopt.getSpace());
  // Get lexmax by flipping range sign in the PWMA constraints.
  for (auto &flippedPiece :
       flippedSymbolicIntegerLexMax.lexopt.getAllPieces()) {
    IntMatrix mat = flippedPiece.output.getOutputMatrix();
    for (unsigned i = 0, e = mat.getNumRows(); i < e; i++)
      mat.negateRow(i);
    MultiAffineFunction maf(flippedPiece.output.getSpace(), mat);
    PWMAFunction::Piece piece = {flippedPiece.domain, maf};
    symbolicIntegerLexMax.lexopt.addPiece(piece);
  }
  symbolicIntegerLexMax.unboundedDomain =
      flippedSymbolicIntegerLexMax.unboundedDomain;
  return symbolicIntegerLexMax;
}

PresburgerRelation
IntegerRelation::subtract(const PresburgerRelation &set) const {
  return PresburgerRelation(*this).subtract(set);
}

unsigned IntegerRelation::insertVar(VarKind kind, unsigned pos, unsigned num) {
  assert(pos <= getNumVarKind(kind));

  unsigned insertPos = space.insertVar(kind, pos, num);
  inequalities.insertColumns(insertPos, num);
  equalities.insertColumns(insertPos, num);
  return insertPos;
}

unsigned IntegerRelation::appendVar(VarKind kind, unsigned num) {
  unsigned pos = getNumVarKind(kind);
  return insertVar(kind, pos, num);
}

void IntegerRelation::addEquality(ArrayRef<MPInt> eq) {
  assert(eq.size() == getNumCols());
  unsigned row = equalities.appendExtraRow();
  for (unsigned i = 0, e = eq.size(); i < e; ++i)
    equalities(row, i) = eq[i];
}

void IntegerRelation::addInequality(ArrayRef<MPInt> inEq) {
  assert(inEq.size() == getNumCols());
  unsigned row = inequalities.appendExtraRow();
  for (unsigned i = 0, e = inEq.size(); i < e; ++i)
    inequalities(row, i) = inEq[i];
}

void IntegerRelation::removeVar(VarKind kind, unsigned pos) {
  removeVarRange(kind, pos, pos + 1);
}

void IntegerRelation::removeVar(unsigned pos) { removeVarRange(pos, pos + 1); }

void IntegerRelation::removeVarRange(VarKind kind, unsigned varStart,
                                     unsigned varLimit) {
  assert(varLimit <= getNumVarKind(kind));

  if (varStart >= varLimit)
    return;

  // Remove eliminated variables from the constraints.
  unsigned offset = getVarKindOffset(kind);
  equalities.removeColumns(offset + varStart, varLimit - varStart);
  inequalities.removeColumns(offset + varStart, varLimit - varStart);

  // Remove eliminated variables from the space.
  space.removeVarRange(kind, varStart, varLimit);
}

void IntegerRelation::removeVarRange(unsigned varStart, unsigned varLimit) {
  assert(varLimit <= getNumVars());

  if (varStart >= varLimit)
    return;

  // Helper function to remove vars of the specified kind in the given range
  // [start, limit), The range is absolute (i.e. it is not relative to the kind
  // of variable). Also updates `limit` to reflect the deleted variables.
  auto removeVarKindInRange = [this](VarKind kind, unsigned &start,
                                     unsigned &limit) {
    if (start >= limit)
      return;

    unsigned offset = getVarKindOffset(kind);
    unsigned num = getNumVarKind(kind);

    // Get `start`, `limit` relative to the specified kind.
    unsigned relativeStart =
        start <= offset ? 0 : std::min(num, start - offset);
    unsigned relativeLimit =
        limit <= offset ? 0 : std::min(num, limit - offset);

    // Remove vars of the specified kind in the relative range.
    removeVarRange(kind, relativeStart, relativeLimit);

    // Update `limit` to reflect deleted variables.
    // `start` does not need to be updated because any variables that are
    // deleted are after position `start`.
    limit -= relativeLimit - relativeStart;
  };

  removeVarKindInRange(VarKind::Domain, varStart, varLimit);
  removeVarKindInRange(VarKind::Range, varStart, varLimit);
  removeVarKindInRange(VarKind::Symbol, varStart, varLimit);
  removeVarKindInRange(VarKind::Local, varStart, varLimit);
}

void IntegerRelation::removeEquality(unsigned pos) {
  equalities.removeRow(pos);
}

void IntegerRelation::removeInequality(unsigned pos) {
  inequalities.removeRow(pos);
}

void IntegerRelation::removeEqualityRange(unsigned start, unsigned end) {
  if (start >= end)
    return;
  equalities.removeRows(start, end - start);
}

void IntegerRelation::removeInequalityRange(unsigned start, unsigned end) {
  if (start >= end)
    return;
  inequalities.removeRows(start, end - start);
}

void IntegerRelation::swapVar(unsigned posA, unsigned posB) {
  assert(posA < getNumVars() && "invalid position A");
  assert(posB < getNumVars() && "invalid position B");

  if (posA == posB)
    return;

  VarKind kindA = space.getVarKindAt(posA);
  VarKind kindB = space.getVarKindAt(posB);
  unsigned relativePosA = posA - getVarKindOffset(kindA);
  unsigned relativePosB = posB - getVarKindOffset(kindB);
  space.swapVar(kindA, kindB, relativePosA, relativePosB);

  inequalities.swapColumns(posA, posB);
  equalities.swapColumns(posA, posB);
}

void IntegerRelation::clearConstraints() {
  equalities.resizeVertically(0);
  inequalities.resizeVertically(0);
}

/// Gather all lower and upper bounds of the variable at `pos`, and
/// optionally any equalities on it. In addition, the bounds are to be
/// independent of variables in position range [`offset`, `offset` + `num`).
void IntegerRelation::getLowerAndUpperBoundIndices(
    unsigned pos, SmallVectorImpl<unsigned> *lbIndices,
    SmallVectorImpl<unsigned> *ubIndices, SmallVectorImpl<unsigned> *eqIndices,
    unsigned offset, unsigned num) const {
  assert(pos < getNumVars() && "invalid position");
  assert(offset + num < getNumCols() && "invalid range");

  // Checks for a constraint that has a non-zero coeff for the variables in
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
  // involving the pos^th variable.
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

bool IntegerRelation::hasConsistentState() const {
  if (!inequalities.hasConsistentState())
    return false;
  if (!equalities.hasConsistentState())
    return false;
  return true;
}

void IntegerRelation::setAndEliminate(unsigned pos, ArrayRef<MPInt> values) {
  if (values.empty())
    return;
  assert(pos + values.size() <= getNumVars() &&
         "invalid position or too many values");
  // Setting x_j = p in sum_i a_i x_i + c is equivalent to adding p*a_j to the
  // constant term and removing the var x_j. We do this for all the vars
  // pos, pos + 1, ... pos + values.size() - 1.
  unsigned constantColPos = getNumCols() - 1;
  for (unsigned i = 0, numVals = values.size(); i < numVals; ++i)
    inequalities.addToColumn(i + pos, constantColPos, values[i]);
  for (unsigned i = 0, numVals = values.size(); i < numVals; ++i)
    equalities.addToColumn(i + pos, constantColPos, values[i]);
  removeVarRange(pos, pos + values.size());
}

void IntegerRelation::clearAndCopyFrom(const IntegerRelation &other) {
  *this = other;
}

// Searches for a constraint with a non-zero coefficient at `colIdx` in
// equality (isEq=true) or inequality (isEq=false) constraints.
// Returns true and sets row found in search in `rowIdx`, false otherwise.
bool IntegerRelation::findConstraintWithNonZeroAt(unsigned colIdx, bool isEq,
                                                  unsigned *rowIdx) const {
  assert(colIdx < getNumCols() && "position out of bounds");
  auto at = [&](unsigned rowIdx) -> MPInt {
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

void IntegerRelation::normalizeConstraintsByGCD() {
  for (unsigned i = 0, e = getNumEqualities(); i < e; ++i)
    equalities.normalizeRow(i);
  for (unsigned i = 0, e = getNumInequalities(); i < e; ++i)
    inequalities.normalizeRow(i);
}

bool IntegerRelation::hasInvalidConstraint() const {
  assert(hasConsistentState());
  auto check = [&](bool isEq) -> bool {
    unsigned numCols = getNumCols();
    unsigned numRows = isEq ? getNumEqualities() : getNumInequalities();
    for (unsigned i = 0, e = numRows; i < e; ++i) {
      unsigned j;
      for (j = 0; j < numCols - 1; ++j) {
        MPInt v = isEq ? atEq(i, j) : atIneq(i, j);
        // Skip rows with non-zero variable coefficients.
        if (v != 0)
          break;
      }
      if (j < numCols - 1) {
        continue;
      }
      // Check validity of constant term at 'numCols - 1' w.r.t 'isEq'.
      // Example invalid constraints include: '1 == 0' or '-1 >= 0'
      MPInt v = isEq ? atEq(i, numCols - 1) : atIneq(i, numCols - 1);
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

/// Eliminate variable from constraint at `rowIdx` based on coefficient at
/// pivotRow, pivotCol. Columns in range [elimColStart, pivotCol) will not be
/// updated as they have already been eliminated.
static void eliminateFromConstraint(IntegerRelation *constraints,
                                    unsigned rowIdx, unsigned pivotRow,
                                    unsigned pivotCol, unsigned elimColStart,
                                    bool isEq) {
  // Skip if equality 'rowIdx' if same as 'pivotRow'.
  if (isEq && rowIdx == pivotRow)
    return;
  auto at = [&](unsigned i, unsigned j) -> MPInt {
    return isEq ? constraints->atEq(i, j) : constraints->atIneq(i, j);
  };
  MPInt leadCoeff = at(rowIdx, pivotCol);
  // Skip if leading coefficient at 'rowIdx' is already zero.
  if (leadCoeff == 0)
    return;
  MPInt pivotCoeff = constraints->atEq(pivotRow, pivotCol);
  int sign = (leadCoeff * pivotCoeff > 0) ? -1 : 1;
  MPInt lcm = presburger::lcm(pivotCoeff, leadCoeff);
  MPInt pivotMultiplier = sign * (lcm / abs(pivotCoeff));
  MPInt rowMultiplier = lcm / abs(leadCoeff);

  unsigned numCols = constraints->getNumCols();
  for (unsigned j = 0; j < numCols; ++j) {
    // Skip updating column 'j' if it was just eliminated.
    if (j >= elimColStart && j < pivotCol)
      continue;
    MPInt v = pivotMultiplier * constraints->atEq(pivotRow, j) +
              rowMultiplier * at(rowIdx, j);
    isEq ? constraints->atEq(rowIdx, j) = v
         : constraints->atIneq(rowIdx, j) = v;
  }
}

/// Returns the position of the variable that has the minimum <number of lower
/// bounds> times <number of upper bounds> from the specified range of
/// variables [start, end). It is often best to eliminate in the increasing
/// order of these counts when doing Fourier-Motzkin elimination since FM adds
/// that many new constraints.
static unsigned getBestVarToEliminate(const IntegerRelation &cst,
                                      unsigned start, unsigned end) {
  assert(start < cst.getNumVars() && end < cst.getNumVars() + 1);

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

// Checks for emptiness of the set by eliminating variables successively and
// using the GCD test (on all equality constraints) and checking for trivially
// invalid constraints. Returns 'true' if the constraint system is found to be
// empty; false otherwise.
bool IntegerRelation::isEmpty() const {
  if (isEmptyByGCDTest() || hasInvalidConstraint())
    return true;

  IntegerRelation tmpCst(*this);

  // First, eliminate as many local variables as possible using equalities.
  tmpCst.removeRedundantLocalVars();
  if (tmpCst.isEmptyByGCDTest() || tmpCst.hasInvalidConstraint())
    return true;

  // Eliminate as many variables as possible using Gaussian elimination.
  unsigned currentPos = 0;
  while (currentPos < tmpCst.getNumVars()) {
    tmpCst.gaussianEliminateVars(currentPos, tmpCst.getNumVars());
    ++currentPos;
    // We check emptiness through trivial checks after eliminating each ID to
    // detect emptiness early. Since the checks isEmptyByGCDTest() and
    // hasInvalidConstraint() are linear time and single sweep on the constraint
    // buffer, this appears reasonable - but can optimize in the future.
    if (tmpCst.hasInvalidConstraint() || tmpCst.isEmptyByGCDTest())
      return true;
  }

  // Eliminate the remaining using FM.
  for (unsigned i = 0, e = tmpCst.getNumVars(); i < e; i++) {
    tmpCst.fourierMotzkinEliminate(
        getBestVarToEliminate(tmpCst, 0, tmpCst.getNumVars()));
    // Check for a constraint explosion. This rarely happens in practice, but
    // this check exists as a safeguard against improperly constructed
    // constraint systems or artificially created arbitrarily complex systems
    // that aren't the intended use case for IntegerRelation. This is
    // needed since FM has a worst case exponential complexity in theory.
    if (tmpCst.getNumConstraints() >= kExplosionFactor * getNumVars()) {
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

bool IntegerRelation::isObviouslyEmpty() const {
  if (isEmptyByGCDTest() || hasInvalidConstraint())
    return true;
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
bool IntegerRelation::isEmptyByGCDTest() const {
  assert(hasConsistentState());
  unsigned numCols = getNumCols();
  for (unsigned i = 0, e = getNumEqualities(); i < e; ++i) {
    MPInt gcd = abs(atEq(i, 0));
    for (unsigned j = 1; j < numCols - 1; ++j) {
      gcd = presburger::gcd(gcd, abs(atEq(i, j)));
    }
    MPInt v = abs(atEq(i, numCols - 1));
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
IntMatrix IntegerRelation::getBoundedDirections() const {
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
  IntMatrix dirs(boundedIneqs.size() + getNumEqualities(), dirsNumCols);

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

bool IntegerRelation::isIntegerEmpty() const { return !findIntegerSample(); }

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
std::optional<SmallVector<MPInt, 8>>
IntegerRelation::findIntegerSample() const {
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
  IntMatrix m = getBoundedDirections();
  // In column echelon form, each row of m occupies only the first rank(m)
  // columns and has zeros on the other columns. The transform T that brings S
  // to column echelon form is unimodular as well, so this is a suitable
  // transform to use in step 1 of the algorithm.
  std::pair<unsigned, LinearTransform> result =
      LinearTransform::makeTransformToColumnEchelon(m);
  const LinearTransform &transform = result.second;
  // 1) Apply T to S to obtain S*T.
  IntegerRelation transformedSet = transform.applyTo(*this);

  // 2) Remove the unbounded dimensions and constraints involving them to
  // obtain a bounded set.
  IntegerRelation boundedSet(transformedSet);
  unsigned numBoundedDims = result.first;
  unsigned numUnboundedDims = getNumVars() - numBoundedDims;
  removeConstraintsInvolvingVarRange(boundedSet, numBoundedDims,
                                     numUnboundedDims);
  boundedSet.removeVarRange(numBoundedDims, boundedSet.getNumVars());

  // 3) Try to obtain a sample from the bounded set.
  std::optional<SmallVector<MPInt, 8>> boundedSample =
      Simplex(boundedSet).findIntegerSample();
  if (!boundedSample)
    return {};
  assert(boundedSet.containsPoint(*boundedSample) &&
         "Simplex returned an invalid sample!");

  // 4) Substitute the values of the bounded dimensions into S*T to obtain a
  // full-dimensional cone, which necessarily contains an integer sample.
  transformedSet.setAndEliminate(0, *boundedSample);
  IntegerRelation &cone = transformedSet;

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
    for (unsigned j = 0; j < cone.getNumVars(); ++j) {
      MPInt coeff = cone.atIneq(i, j);
      if (coeff < 0)
        cone.atIneq(i, cone.getNumVars()) += coeff;
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

  SmallVector<MPInt, 8> coneSample(llvm::map_range(shrunkenConeSample, ceil));

  // 6) Return transform * concat(boundedSample, coneSample).
  SmallVector<MPInt, 8> &sample = *boundedSample;
  sample.append(coneSample.begin(), coneSample.end());
  return transform.postMultiplyWithColumn(sample);
}

/// Helper to evaluate an affine expression at a point.
/// The expression is a list of coefficients for the dimensions followed by the
/// constant term.
static MPInt valueAt(ArrayRef<MPInt> expr, ArrayRef<MPInt> point) {
  assert(expr.size() == 1 + point.size() &&
         "Dimensionalities of point and expression don't match!");
  MPInt value = expr.back();
  for (unsigned i = 0; i < point.size(); ++i)
    value += expr[i] * point[i];
  return value;
}

/// A point satisfies an equality iff the value of the equality at the
/// expression is zero, and it satisfies an inequality iff the value of the
/// inequality at that point is non-negative.
bool IntegerRelation::containsPoint(ArrayRef<MPInt> point) const {
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

/// Just substitute the values given and check if an integer sample exists for
/// the local vars.
///
/// TODO: this could be made more efficient by handling divisions separately.
/// Instead of finding an integer sample over all the locals, we can first
/// compute the values of the locals that have division representations and
/// only use the integer emptiness check for the locals that don't have this.
/// Handling this correctly requires ordering the divs, though.
std::optional<SmallVector<MPInt, 8>>
IntegerRelation::containsPointNoLocal(ArrayRef<MPInt> point) const {
  assert(point.size() == getNumVars() - getNumLocalVars() &&
         "Point should contain all vars except locals!");
  assert(getVarKindOffset(VarKind::Local) == getNumVars() - getNumLocalVars() &&
         "This function depends on locals being stored last!");
  IntegerRelation copy = *this;
  copy.setAndEliminate(0, point);
  return copy.findIntegerSample();
}

DivisionRepr
IntegerRelation::getLocalReprs(std::vector<MaybeLocalRepr> *repr) const {
  SmallVector<bool, 8> foundRepr(getNumVars(), false);
  for (unsigned i = 0, e = getNumDimAndSymbolVars(); i < e; ++i)
    foundRepr[i] = true;

  unsigned localOffset = getVarKindOffset(VarKind::Local);
  DivisionRepr divs(getNumVars(), getNumLocalVars());
  bool changed;
  do {
    // Each time changed is true, at end of this iteration, one or more local
    // vars have been detected as floor divs.
    changed = false;
    for (unsigned i = 0, e = getNumLocalVars(); i < e; ++i) {
      if (!foundRepr[i + localOffset]) {
        MaybeLocalRepr res =
            computeSingleVarRepr(*this, foundRepr, localOffset + i,
                                 divs.getDividend(i), divs.getDenom(i));
        if (!res) {
          // No representation was found, so clear the representation and
          // continue.
          divs.clearRepr(i);
          continue;
        }
        foundRepr[localOffset + i] = true;
        if (repr)
          (*repr)[i] = res;
        changed = true;
      }
    }
  } while (changed);

  return divs;
}

/// Tightens inequalities given that we are dealing with integer spaces. This is
/// analogous to the GCD test but applied to inequalities. The constant term can
/// be reduced to the preceding multiple of the GCD of the coefficients, i.e.,
///  64*i - 100 >= 0  =>  64*i - 128 >= 0 (since 'i' is an integer). This is a
/// fast method - linear in the number of coefficients.
// Example on how this affects practical cases: consider the scenario:
// 64*i >= 100, j = 64*i; without a tightening, elimination of i would yield
// j >= 100 instead of the tighter (exact) j >= 128.
void IntegerRelation::gcdTightenInequalities() {
  unsigned numCols = getNumCols();
  for (unsigned i = 0, e = getNumInequalities(); i < e; ++i) {
    // Normalize the constraint and tighten the constant term by the GCD.
    MPInt gcd = inequalities.normalizeRow(i, getNumCols() - 1);
    if (gcd > 1)
      atIneq(i, numCols - 1) = floorDiv(atIneq(i, numCols - 1), gcd);
  }
}

// Eliminates all variable variables in column range [posStart, posLimit).
// Returns the number of variables eliminated.
unsigned IntegerRelation::gaussianEliminateVars(unsigned posStart,
                                                unsigned posLimit) {
  // Return if variable positions to eliminate are out of range.
  assert(posLimit <= getNumVars());
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

    // Eliminate variable at 'pivotCol' from each equality row.
    for (unsigned i = 0, e = getNumEqualities(); i < e; ++i) {
      eliminateFromConstraint(this, i, pivotRow, pivotCol, posStart,
                              /*isEq=*/true);
      equalities.normalizeRow(i);
    }

    // Eliminate variable at 'pivotCol' from each inequality row.
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
  removeVarRange(posStart, posLimit);
  return posLimit - posStart;
}

bool IntegerRelation::gaussianEliminate() {
  gcdTightenInequalities();
  unsigned firstVar = 0, vars = getNumVars();
  unsigned nowDone, eqs, pivotRow;
  for (nowDone = 0, eqs = getNumEqualities(); nowDone < eqs; ++nowDone) {
    // Finds the first non-empty column.
    for (; firstVar < vars; ++firstVar) {
      if (!findConstraintWithNonZeroAt(firstVar, true, &pivotRow))
        continue;
      break;
    }
    // The matrix has been normalized to row echelon form.
    if (firstVar >= vars)
      break;

    // The first pivot row found is below where it should currently be placed.
    if (pivotRow > nowDone) {
      equalities.swapRows(pivotRow, nowDone);
      pivotRow = nowDone;
    }

    // Normalize all lower equations and all inequalities.
    for (unsigned i = nowDone + 1; i < eqs; ++i) {
      eliminateFromConstraint(this, i, pivotRow, firstVar, 0, true);
      equalities.normalizeRow(i);
    }
    for (unsigned i = 0, ineqs = getNumInequalities(); i < ineqs; ++i) {
      eliminateFromConstraint(this, i, pivotRow, firstVar, 0, false);
      inequalities.normalizeRow(i);
    }
    gcdTightenInequalities();
  }

  // No redundant rows.
  if (nowDone == eqs)
    return false;

  // Check to see if the redundant rows constant is zero, a non-zero value means
  // the set is empty.
  for (unsigned i = nowDone; i < eqs; ++i) {
    if (atEq(i, vars) == 0)
      continue;

    *this = getEmpty(getSpace());
    return true;
  }
  // Eliminate rows that are confined to be all zeros.
  removeEqualityRange(nowDone, eqs);
  return true;
}

// A more complex check to eliminate redundant inequalities. Uses FourierMotzkin
// to check if a constraint is redundant.
void IntegerRelation::removeRedundantInequalities() {
  SmallVector<bool, 32> redun(getNumInequalities(), false);
  // To check if an inequality is redundant, we replace the inequality by its
  // complement (for eg., i - 1 >= 0 by i <= 0), and check if the resulting
  // system is empty. If it is, the inequality is redundant.
  IntegerRelation tmpCst(*this);
  for (unsigned r = 0, e = getNumInequalities(); r < e; r++) {
    // Change the inequality to its complement.
    tmpCst.inequalities.negateRow(r);
    --tmpCst.atIneq(r, tmpCst.getNumCols() - 1);
    if (tmpCst.isEmpty()) {
      redun[r] = true;
      // Zero fill the redundant inequality.
      inequalities.fillRow(r, /*value=*/0);
      tmpCst.inequalities.fillRow(r, /*value=*/0);
    } else {
      // Reverse the change (to avoid recreating tmpCst each time).
      ++tmpCst.atIneq(r, tmpCst.getNumCols() - 1);
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
void IntegerRelation::removeRedundantConstraints() {
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

std::optional<MPInt> IntegerRelation::computeVolume() const {
  assert(getNumSymbolVars() == 0 && "Symbols are not yet supported!");

  Simplex simplex(*this);
  // If the polytope is rationally empty, there are certainly no integer
  // points.
  if (simplex.isEmpty())
    return MPInt(0);

  // Just find the maximum and minimum integer value of each non-local var
  // separately, thus finding the number of integer values each such var can
  // take. Multiplying these together gives a valid overapproximation of the
  // number of integer points in the relation. The result this gives is
  // equivalent to projecting (rationally) the relation onto its non-local vars
  // and returning the number of integer points in a minimal axis-parallel
  // hyperrectangular overapproximation of that.
  //
  // We also handle the special case where one dimension is unbounded and
  // another dimension can take no integer values. In this case, the volume is
  // zero.
  //
  // If there is no such empty dimension, if any dimension is unbounded we
  // just return the result as unbounded.
  MPInt count(1);
  SmallVector<MPInt, 8> dim(getNumVars() + 1);
  bool hasUnboundedVar = false;
  for (unsigned i = 0, e = getNumDimAndSymbolVars(); i < e; ++i) {
    dim[i] = 1;
    auto [min, max] = simplex.computeIntegerBounds(dim);
    dim[i] = 0;

    assert((!min.isEmpty() && !max.isEmpty()) &&
           "Polytope should be rationally non-empty!");

    // One of the dimensions is unbounded. Note this fact. We will return
    // unbounded if none of the other dimensions makes the volume zero.
    if (min.isUnbounded() || max.isUnbounded()) {
      hasUnboundedVar = true;
      continue;
    }

    // In this case there are no valid integer points and the volume is
    // definitely zero.
    if (min.getBoundedOptimum() > max.getBoundedOptimum())
      return MPInt(0);

    count *= (*max - *min + 1);
  }

  if (count == 0)
    return MPInt(0);
  if (hasUnboundedVar)
    return {};
  return count;
}

void IntegerRelation::eliminateRedundantLocalVar(unsigned posA, unsigned posB) {
  assert(posA < getNumLocalVars() && "Invalid local var position");
  assert(posB < getNumLocalVars() && "Invalid local var position");

  unsigned localOffset = getVarKindOffset(VarKind::Local);
  posA += localOffset;
  posB += localOffset;
  inequalities.addToColumn(posB, posA, 1);
  equalities.addToColumn(posB, posA, 1);
  removeVar(posB);
}

/// mergeAndAlignSymbols's implementation can be broken down into two steps:
/// 1. Merge and align identifiers into `other` from `this. If an identifier
/// from `this` exists in `other` then we align it. Otherwise, we assume it is a
/// new identifier and insert it into `other` in the same position as `this`.
/// 2. Add identifiers that are in `other` but not `this to `this`.
void IntegerRelation::mergeAndAlignSymbols(IntegerRelation &other) {
  assert(space.isUsingIds() && other.space.isUsingIds() &&
         "both relations need to have identifers to merge and align");

  unsigned i = 0;
  for (const Identifier identifier : space.getIds(VarKind::Symbol)) {
    // Search in `other` starting at position `i` since the left of `i` is
    // aligned.
    const Identifier *findBegin =
        other.space.getIds(VarKind::Symbol).begin() + i;
    const Identifier *findEnd = other.space.getIds(VarKind::Symbol).end();
    const Identifier *itr = std::find(findBegin, findEnd, identifier);
    if (itr != findEnd) {
      other.swapVar(other.getVarKindOffset(VarKind::Symbol) + i,
                    other.getVarKindOffset(VarKind::Symbol) + i +
                        std::distance(findBegin, itr));
    } else {
      other.insertVar(VarKind::Symbol, i);
      other.space.getId(VarKind::Symbol, i) = identifier;
    }
    ++i;
  }

  for (unsigned e = other.getNumVarKind(VarKind::Symbol); i < e; ++i) {
    insertVar(VarKind::Symbol, i);
    space.getId(VarKind::Symbol, i) = other.space.getId(VarKind::Symbol, i);
  }
}

/// Adds additional local ids to the sets such that they both have the union
/// of the local ids in each set, without changing the set of points that
/// lie in `this` and `other`.
///
/// To detect local ids that always take the same value, each local id is
/// represented as a floordiv with constant denominator in terms of other ids.
/// After extracting these divisions, local ids in `other` with the same
/// division representation as some other local id in any set are considered
/// duplicate and are merged.
///
/// It is possible that division representation for some local id cannot be
/// obtained, and thus these local ids are not considered for detecting
/// duplicates.
unsigned IntegerRelation::mergeLocalVars(IntegerRelation &other) {
  IntegerRelation &relA = *this;
  IntegerRelation &relB = other;

  unsigned oldALocals = relA.getNumLocalVars();

  // Merge function that merges the local variables in both sets by treating
  // them as the same variable.
  auto merge = [&relA, &relB, oldALocals](unsigned i, unsigned j) -> bool {
    // We only merge from local at pos j to local at pos i, where j > i.
    if (i >= j)
      return false;

    // If i < oldALocals, we are trying to merge duplicate divs. Since we do not
    // want to merge duplicates in A, we ignore this call.
    if (j < oldALocals)
      return false;

    // Merge local at pos j into local at position i.
    relA.eliminateRedundantLocalVar(i, j);
    relB.eliminateRedundantLocalVar(i, j);
    return true;
  };

  presburger::mergeLocalVars(*this, other, merge);

  // Since we do not remove duplicate divisions in relA, this is guranteed to be
  // non-negative.
  return relA.getNumLocalVars() - oldALocals;
}

bool IntegerRelation::hasOnlyDivLocals() const {
  return getLocalReprs().hasAllReprs();
}

void IntegerRelation::removeDuplicateDivs() {
  DivisionRepr divs = getLocalReprs();
  auto merge = [this](unsigned i, unsigned j) -> bool {
    eliminateRedundantLocalVar(i, j);
    return true;
  };
  divs.removeDuplicateDivs(merge);
}

void IntegerRelation::simplify() {
  bool changed = true;
  // Repeat until we reach a fixed point.
  while (changed) {
    if (isObviouslyEmpty())
      return;
    changed = false;
    normalizeConstraintsByGCD();
    changed |= gaussianEliminate();
    changed |= removeDuplicateConstraints();
  }
  // Current set is not empty.
}

/// Removes local variables using equalities. Each equality is checked if it
/// can be reduced to the form: `e = affine-expr`, where `e` is a local
/// variable and `affine-expr` is an affine expression not containing `e`.
/// If an equality satisfies this form, the local variable is replaced in
/// each constraint and then removed. The equality used to replace this local
/// variable is also removed.
void IntegerRelation::removeRedundantLocalVars() {
  // Normalize the equality constraints to reduce coefficients of local
  // variables to 1 wherever possible.
  for (unsigned i = 0, e = getNumEqualities(); i < e; ++i)
    equalities.normalizeRow(i);

  while (true) {
    unsigned i, e, j, f;
    for (i = 0, e = getNumEqualities(); i < e; ++i) {
      // Find a local variable to eliminate using ith equality.
      for (j = getNumDimAndSymbolVars(), f = getNumVars(); j < f; ++j)
        if (abs(atEq(i, j)) == 1)
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
    removeVar(j);
    removeEquality(i);
  }
}

void IntegerRelation::convertVarKind(VarKind srcKind, unsigned varStart,
                                     unsigned varLimit, VarKind dstKind,
                                     unsigned pos) {
  assert(varLimit <= getNumVarKind(srcKind) && "Invalid id range");

  if (varStart >= varLimit)
    return;

  // Append new local variables corresponding to the dimensions to be converted.
  unsigned convertCount = varLimit - varStart;
  unsigned newVarsBegin = insertVar(dstKind, pos, convertCount);

  // Swap the new local variables with dimensions.
  //
  // Essentially, this moves the information corresponding to the specified ids
  // of kind `srcKind` to the `convertCount` newly created ids of kind
  // `dstKind`. In particular, this moves the columns in the constraint
  // matrices, and zeros out the initially occupied columns (because the newly
  // created ids we're swapping with were zero-initialized).
  unsigned offset = getVarKindOffset(srcKind);
  for (unsigned i = 0; i < convertCount; ++i)
    swapVar(offset + varStart + i, newVarsBegin + i);

  // Complete the move by deleting the initially occupied columns.
  removeVarRange(srcKind, varStart, varLimit);
}

void IntegerRelation::addBound(BoundType type, unsigned pos,
                               const MPInt &value) {
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

void IntegerRelation::addBound(BoundType type, ArrayRef<MPInt> expr,
                               const MPInt &value) {
  assert(type != BoundType::EQ && "EQ not implemented");
  assert(expr.size() == getNumCols());
  unsigned row = inequalities.appendExtraRow();
  for (unsigned i = 0, e = expr.size(); i < e; ++i)
    inequalities(row, i) = type == BoundType::LB ? expr[i] : -expr[i];
  inequalities(inequalities.getNumRows() - 1, getNumCols() - 1) +=
      type == BoundType::LB ? -value : value;
}

/// Adds a new local variable as the floordiv of an affine function of other
/// variables, the coefficients of which are provided in 'dividend' and with
/// respect to a positive constant 'divisor'. Two constraints are added to the
/// system to capture equivalence with the floordiv.
///      q = expr floordiv c    <=>   c*q <= expr <= c*q + c - 1.
void IntegerRelation::addLocalFloorDiv(ArrayRef<MPInt> dividend,
                                       const MPInt &divisor) {
  assert(dividend.size() == getNumCols() && "incorrect dividend size");
  assert(divisor > 0 && "positive divisor expected");

  appendVar(VarKind::Local);

  SmallVector<MPInt, 8> dividendCopy(dividend.begin(), dividend.end());
  dividendCopy.insert(dividendCopy.end() - 1, MPInt(0));
  addInequality(
      getDivLowerBound(dividendCopy, divisor, dividendCopy.size() - 2));
  addInequality(
      getDivUpperBound(dividendCopy, divisor, dividendCopy.size() - 2));
}

/// Finds an equality that equates the specified variable to a constant.
/// Returns the position of the equality row. If 'symbolic' is set to true,
/// symbols are also treated like a constant, i.e., an affine function of the
/// symbols is also treated like a constant. Returns -1 if such an equality
/// could not be found.
static int findEqualityToConstant(const IntegerRelation &cst, unsigned pos,
                                  bool symbolic = false) {
  assert(pos < cst.getNumVars() && "invalid position");
  for (unsigned r = 0, e = cst.getNumEqualities(); r < e; r++) {
    MPInt v = cst.atEq(r, pos);
    if (v * v != 1)
      continue;
    unsigned c;
    unsigned f = symbolic ? cst.getNumDimVars() : cst.getNumVars();
    // This checks for zeros in all positions other than 'pos' in [0, f)
    for (c = 0; c < f; c++) {
      if (c == pos)
        continue;
      if (cst.atEq(r, c) != 0) {
        // Dependent on another variable.
        break;
      }
    }
    if (c == f)
      // Equality is free of other variables.
      return r;
  }
  return -1;
}

LogicalResult IntegerRelation::constantFoldVar(unsigned pos) {
  assert(pos < getNumVars() && "invalid position");
  int rowIdx;
  if ((rowIdx = findEqualityToConstant(*this, pos)) == -1)
    return failure();

  // atEq(rowIdx, pos) is either -1 or 1.
  assert(atEq(rowIdx, pos) * atEq(rowIdx, pos) == 1);
  MPInt constVal = -atEq(rowIdx, getNumCols() - 1) / atEq(rowIdx, pos);
  setAndEliminate(pos, constVal);
  return success();
}

void IntegerRelation::constantFoldVarRange(unsigned pos, unsigned num) {
  for (unsigned s = pos, t = pos, e = pos + num; s < e; s++) {
    if (failed(constantFoldVar(t)))
      t++;
  }
}

/// Returns a non-negative constant bound on the extent (upper bound - lower
/// bound) of the specified variable if it is found to be a constant; returns
/// std::nullopt if it's not a constant. This methods treats symbolic variables
/// specially, i.e., it looks for constant differences between affine
/// expressions involving only the symbolic variables. See comments at function
/// definition for example. 'lb', if provided, is set to the lower bound
/// associated with the constant difference. Note that 'lb' is purely symbolic
/// and thus will contain the coefficients of the symbolic variables and the
/// constant coefficient.
//  Egs: 0 <= i <= 15, return 16.
//       s0 + 2 <= i <= s0 + 17, returns 16. (s0 has to be a symbol)
//       s0 + s1 + 16 <= d0 <= s0 + s1 + 31, returns 16.
//       s0 - 7 <= 8*j <= s0 returns 1 with lb = s0, lbDivisor = 8 (since lb =
//       ceil(s0 - 7 / 8) = floor(s0 / 8)).
std::optional<MPInt> IntegerRelation::getConstantBoundOnDimSize(
    unsigned pos, SmallVectorImpl<MPInt> *lb, MPInt *boundFloorDivisor,
    SmallVectorImpl<MPInt> *ub, unsigned *minLbPos, unsigned *minUbPos) const {
  assert(pos < getNumDimVars() && "Invalid variable position");

  // Find an equality for 'pos'^th variable that equates it to some function
  // of the symbolic variables (+ constant).
  int eqPos = findEqualityToConstant(*this, pos, /*symbolic=*/true);
  if (eqPos != -1) {
    auto eq = getEquality(eqPos);
    // If the equality involves a local var, punt for now.
    // TODO: this can be handled in the future by using the explicit
    // representation of the local vars.
    if (!std::all_of(eq.begin() + getNumDimAndSymbolVars(), eq.end() - 1,
                     [](const MPInt &coeff) { return coeff == 0; }))
      return std::nullopt;

    // This variable can only take a single value.
    if (lb) {
      // Set lb to that symbolic value.
      lb->resize(getNumSymbolVars() + 1);
      if (ub)
        ub->resize(getNumSymbolVars() + 1);
      for (unsigned c = 0, f = getNumSymbolVars() + 1; c < f; c++) {
        MPInt v = atEq(eqPos, pos);
        // atEq(eqRow, pos) is either -1 or 1.
        assert(v * v == 1);
        (*lb)[c] = v < 0 ? atEq(eqPos, getNumDimVars() + c) / -v
                         : -atEq(eqPos, getNumDimVars() + c) / v;
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
    return MPInt(1);
  }

  // Check if the variable appears at all in any of the inequalities.
  unsigned r, e;
  for (r = 0, e = getNumInequalities(); r < e; r++) {
    if (atIneq(r, pos) != 0)
      break;
  }
  if (r == e)
    // If it doesn't, there isn't a bound on it.
    return std::nullopt;

  // Positions of constraints that are lower/upper bounds on the variable.
  SmallVector<unsigned, 4> lbIndices, ubIndices;

  // Gather all symbolic lower bounds and upper bounds of the variable, i.e.,
  // the bounds can only involve symbolic (and local) variables. Since the
  // canonical form c_1*x_1 + c_2*x_2 + ... + c_0 >= 0, a constraint is a lower
  // bound for x_i if c_i >= 1, and an upper bound if c_i <= -1.
  getLowerAndUpperBoundIndices(pos, &lbIndices, &ubIndices,
                               /*eqIndices=*/nullptr, /*offset=*/0,
                               /*num=*/getNumDimVars());

  std::optional<MPInt> minDiff;
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
      MPInt diff = ceilDiv(atIneq(ubPos, getNumCols() - 1) +
                               atIneq(lbPos, getNumCols() - 1) + 1,
                           atIneq(lbPos, pos));
      // This bound is non-negative by definition.
      diff = std::max<MPInt>(diff, MPInt(0));
      if (minDiff == std::nullopt || diff < minDiff) {
        minDiff = diff;
        minLbPosition = lbPos;
        minUbPosition = ubPos;
      }
    }
  }
  if (lb && minDiff) {
    // Set lb to the symbolic lower bound.
    lb->resize(getNumSymbolVars() + 1);
    if (ub)
      ub->resize(getNumSymbolVars() + 1);
    // The lower bound is the ceildiv of the lb constraint over the coefficient
    // of the variable at 'pos'. We express the ceildiv equivalently as a floor
    // for uniformity. For eg., if the lower bound constraint was: 32*d0 - N +
    // 31 >= 0, the lower bound for d0 is ceil(N - 31, 32), i.e., floor(N, 32).
    *boundFloorDivisor = atIneq(minLbPosition, pos);
    assert(*boundFloorDivisor == -atIneq(minUbPosition, pos));
    for (unsigned c = 0, e = getNumSymbolVars() + 1; c < e; c++) {
      (*lb)[c] = -atIneq(minLbPosition, getNumDimVars() + c);
    }
    if (ub) {
      for (unsigned c = 0, e = getNumSymbolVars() + 1; c < e; c++)
        (*ub)[c] = atIneq(minUbPosition, getNumDimVars() + c);
    }
    // The lower bound leads to a ceildiv while the upper bound is a floordiv
    // whenever the coefficient at pos != 1. ceildiv (val / d) = floordiv (val +
    // d - 1 / d); hence, the addition of 'atIneq(minLbPosition, pos) - 1' to
    // the constant term for the lower bound.
    (*lb)[getNumSymbolVars()] += atIneq(minLbPosition, pos) - 1;
  }
  if (minLbPos)
    *minLbPos = minLbPosition;
  if (minUbPos)
    *minUbPos = minUbPosition;
  return minDiff;
}

template <bool isLower>
std::optional<MPInt>
IntegerRelation::computeConstantLowerOrUpperBound(unsigned pos) {
  assert(pos < getNumVars() && "invalid position");
  // Project to 'pos'.
  projectOut(0, pos);
  projectOut(1, getNumVars() - 1);
  // Check if there's an equality equating the '0'^th variable to a constant.
  int eqRowIdx = findEqualityToConstant(*this, 0, /*symbolic=*/false);
  if (eqRowIdx != -1)
    // atEq(rowIdx, 0) is either -1 or 1.
    return -atEq(eqRowIdx, getNumCols() - 1) / atEq(eqRowIdx, 0);

  // Check if the variable appears at all in any of the inequalities.
  unsigned r, e;
  for (r = 0, e = getNumInequalities(); r < e; r++) {
    if (atIneq(r, 0) != 0)
      break;
  }
  if (r == e)
    // If it doesn't, there isn't a bound on it.
    return std::nullopt;

  std::optional<MPInt> minOrMaxConst;

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

    MPInt boundConst =
        isLower ? ceilDiv(-atIneq(r, getNumCols() - 1), atIneq(r, 0))
                : floorDiv(atIneq(r, getNumCols() - 1), -atIneq(r, 0));
    if (isLower) {
      if (minOrMaxConst == std::nullopt || boundConst > minOrMaxConst)
        minOrMaxConst = boundConst;
    } else {
      if (minOrMaxConst == std::nullopt || boundConst < minOrMaxConst)
        minOrMaxConst = boundConst;
    }
  }
  return minOrMaxConst;
}

std::optional<MPInt> IntegerRelation::getConstantBound(BoundType type,
                                                       unsigned pos) const {
  if (type == BoundType::LB)
    return IntegerRelation(*this)
        .computeConstantLowerOrUpperBound</*isLower=*/true>(pos);
  if (type == BoundType::UB)
    return IntegerRelation(*this)
        .computeConstantLowerOrUpperBound</*isLower=*/false>(pos);

  assert(type == BoundType::EQ && "expected EQ");
  std::optional<MPInt> lb =
      IntegerRelation(*this).computeConstantLowerOrUpperBound</*isLower=*/true>(
          pos);
  std::optional<MPInt> ub =
      IntegerRelation(*this)
          .computeConstantLowerOrUpperBound</*isLower=*/false>(pos);
  return (lb && ub && *lb == *ub) ? std::optional<MPInt>(*ub) : std::nullopt;
}

// A simple (naive and conservative) check for hyper-rectangularity.
bool IntegerRelation::isHyperRectangular(unsigned pos, unsigned num) const {
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
void IntegerRelation::removeTrivialRedundancy() {
  gcdTightenInequalities();
  normalizeConstraintsByGCD();

  // A map used to detect redundancy stemming from constraints that only differ
  // in their constant term. The value stored is <row position, const term>
  // for a given row.
  SmallDenseMap<ArrayRef<MPInt>, std::pair<unsigned, MPInt>>
      rowsWithoutConstTerm;
  // To unique rows.
  SmallDenseSet<ArrayRef<MPInt>, 8> rowSet;

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
    MPInt *rowStart = &inequalities(r, 0);
    auto row = ArrayRef<MPInt>(rowStart, getNumCols());
    if (isTriviallyValid(r) || !rowSet.insert(row).second) {
      redunIneq[r] = true;
      continue;
    }

    // Among constraints that only differ in the constant term part, mark
    // everything other than the one with the smallest constant term redundant.
    // (eg: among i - 16j - 5 >= 0, i - 16j - 1 >=0, i - 16j - 7 >= 0, the
    // former two are redundant).
    MPInt constTerm = atIneq(r, getNumCols() - 1);
    auto rowWithoutConstTerm = ArrayRef<MPInt>(rowStart, getNumCols() - 1);
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

/// Eliminates variable at the specified position using Fourier-Motzkin
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
void IntegerRelation::fourierMotzkinEliminate(unsigned pos, bool darkShadow,
                                              bool *isResultIntegerExact) {
  LLVM_DEBUG(llvm::dbgs() << "FM input (eliminate pos " << pos << "):\n");
  LLVM_DEBUG(dump());
  assert(pos < getNumVars() && "invalid position");
  assert(hasConsistentState());

  // Check if this variable can be eliminated through a substitution.
  for (unsigned r = 0, e = getNumEqualities(); r < e; r++) {
    if (atEq(r, pos) != 0) {
      // Use Gaussian elimination here (since we have an equality).
      LogicalResult ret = gaussianEliminateVar(pos);
      (void)ret;
      assert(succeeded(ret) && "Gaussian elimination guaranteed to succeed");
      LLVM_DEBUG(llvm::dbgs() << "FM output (through Gaussian elimination):\n");
      LLVM_DEBUG(dump());
      return;
    }
  }

  // A fast linear time tightening.
  gcdTightenInequalities();

  // Check if the variable appears at all in any of the inequalities.
  if (isColZero(pos)) {
    // If it doesn't appear, just remove the column and return.
    // TODO: refactor removeColumns to use it from here.
    removeVar(pos);
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
      // Var does not appear in bound.
      nbIndices.push_back(r);
    } else if (atIneq(r, pos) >= 1) {
      // Lower bound.
      lbIndices.push_back(r);
    } else {
      // Upper bound.
      ubIndices.push_back(r);
    }
  }

  PresburgerSpace newSpace = getSpace();
  VarKind idKindRemove = newSpace.getVarKindAt(pos);
  unsigned relativePos = pos - newSpace.getVarKindOffset(idKindRemove);
  newSpace.removeVarRange(idKindRemove, relativePos, relativePos + 1);

  /// Create the new system which has one variable less.
  IntegerRelation newRel(lbIndices.size() * ubIndices.size() + nbIndices.size(),
                         getNumEqualities(), getNumCols() - 1, newSpace);

  // This will be used to check if the elimination was integer exact.
  bool allLCMsAreOne = true;

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
      SmallVector<MPInt, 4> ineq;
      ineq.reserve(newRel.getNumCols());
      MPInt lbCoeff = atIneq(lbPos, pos);
      // Note that in the comments above, ubCoeff is the negation of the
      // coefficient in the canonical form as the view taken here is that of the
      // term being moved to the other size of '>='.
      MPInt ubCoeff = -atIneq(ubPos, pos);
      // TODO: refactor this loop to avoid all branches inside.
      for (unsigned l = 0, e = getNumCols(); l < e; l++) {
        if (l == pos)
          continue;
        assert(lbCoeff >= 1 && ubCoeff >= 1 && "bounds wrongly identified");
        MPInt lcm = presburger::lcm(lbCoeff, ubCoeff);
        ineq.push_back(atIneq(ubPos, l) * (lcm / ubCoeff) +
                       atIneq(lbPos, l) * (lcm / lbCoeff));
        assert(lcm > 0 && "lcm should be positive!");
        if (lcm != 1)
          allLCMsAreOne = false;
      }
      if (darkShadow) {
        // The dark shadow is a convex subset of the exact integer shadow. If
        // there is a point here, it proves the existence of a solution.
        ineq[ineq.size() - 1] += lbCoeff * ubCoeff - lbCoeff - ubCoeff + 1;
      }
      // TODO: we need to have a way to add inequalities in-place in
      // IntegerRelation instead of creating and copying over.
      newRel.addInequality(ineq);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "FM isResultIntegerExact: " << allLCMsAreOne
                          << "\n");
  if (allLCMsAreOne && isResultIntegerExact)
    *isResultIntegerExact = true;

  // Copy over the constraints not involving this variable.
  for (auto nbPos : nbIndices) {
    SmallVector<MPInt, 4> ineq;
    ineq.reserve(getNumCols() - 1);
    for (unsigned l = 0, e = getNumCols(); l < e; l++) {
      if (l == pos)
        continue;
      ineq.push_back(atIneq(nbPos, l));
    }
    newRel.addInequality(ineq);
  }

  assert(newRel.getNumConstraints() ==
         lbIndices.size() * ubIndices.size() + nbIndices.size());

  // Copy over the equalities.
  for (unsigned r = 0, e = getNumEqualities(); r < e; r++) {
    SmallVector<MPInt, 4> eq;
    eq.reserve(newRel.getNumCols());
    for (unsigned l = 0, e = getNumCols(); l < e; l++) {
      if (l == pos)
        continue;
      eq.push_back(atEq(r, l));
    }
    newRel.addEquality(eq);
  }

  // GCD tightening and normalization allows detection of more trivially
  // redundant constraints.
  newRel.gcdTightenInequalities();
  newRel.normalizeConstraintsByGCD();
  newRel.removeTrivialRedundancy();
  clearAndCopyFrom(newRel);
  LLVM_DEBUG(llvm::dbgs() << "FM output:\n");
  LLVM_DEBUG(dump());
}

#undef DEBUG_TYPE
#define DEBUG_TYPE "presburger"

void IntegerRelation::projectOut(unsigned pos, unsigned num) {
  if (num == 0)
    return;

  // 'pos' can be at most getNumCols() - 2 if num > 0.
  assert((getNumCols() < 2 || pos <= getNumCols() - 2) && "invalid position");
  assert(pos + num < getNumCols() && "invalid range");

  // Eliminate as many variables as possible using Gaussian elimination.
  unsigned currentPos = pos;
  unsigned numToEliminate = num;
  unsigned numGaussianEliminated = 0;

  while (currentPos < getNumVars()) {
    unsigned curNumEliminated =
        gaussianEliminateVars(currentPos, currentPos + numToEliminate);
    ++currentPos;
    numToEliminate -= curNumEliminated + 1;
    numGaussianEliminated += curNumEliminated;
  }

  // Eliminate the remaining using Fourier-Motzkin.
  for (unsigned i = 0; i < num - numGaussianEliminated; i++) {
    unsigned numToEliminate = num - numGaussianEliminated - i;
    fourierMotzkinEliminate(
        getBestVarToEliminate(*this, pos, pos + numToEliminate));
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
static BoundCmpResult compareBounds(ArrayRef<MPInt> a, ArrayRef<MPInt> b) {
  assert(a.size() == b.size());

  // For the bounds to be comparable, their corresponding variable
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
static void getCommonConstraints(const IntegerRelation &a,
                                 const IntegerRelation &b, IntegerRelation &c) {
  c = IntegerRelation(a.getSpace());
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
IntegerRelation::unionBoundingBox(const IntegerRelation &otherCst) {
  assert(space.isEqual(otherCst.getSpace()) && "Spaces should match.");
  assert(getNumLocalVars() == 0 && "local ids not supported yet here");

  // Get the constraints common to both systems; these will be added as is to
  // the union.
  IntegerRelation commonCst(PresburgerSpace::getRelationSpace());
  getCommonConstraints(*this, otherCst, commonCst);

  std::vector<SmallVector<MPInt, 8>> boundingLbs;
  std::vector<SmallVector<MPInt, 8>> boundingUbs;
  boundingLbs.reserve(2 * getNumDimVars());
  boundingUbs.reserve(2 * getNumDimVars());

  // To hold lower and upper bounds for each dimension.
  SmallVector<MPInt, 4> lb, otherLb, ub, otherUb;
  // To compute min of lower bounds and max of upper bounds for each dimension.
  SmallVector<MPInt, 4> minLb(getNumSymbolVars() + 1);
  SmallVector<MPInt, 4> maxUb(getNumSymbolVars() + 1);
  // To compute final new lower and upper bounds for the union.
  SmallVector<MPInt, 8> newLb(getNumCols()), newUb(getNumCols());

  MPInt lbFloorDivisor, otherLbFloorDivisor;
  for (unsigned d = 0, e = getNumDimVars(); d < e; ++d) {
    auto extent = getConstantBoundOnDimSize(d, &lb, &lbFloorDivisor, &ub);
    if (!extent.has_value())
      // TODO: symbolic extents when necessary.
      // TODO: handle union if a dimension is unbounded.
      return failure();

    auto otherExtent = otherCst.getConstantBoundOnDimSize(
        d, &otherLb, &otherLbFloorDivisor, &otherUb);
    if (!otherExtent.has_value() || lbFloorDivisor != otherLbFloorDivisor)
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
      if (!constLb.has_value() || !constOtherLb.has_value())
        return failure();
      std::fill(minLb.begin(), minLb.end(), 0);
      minLb.back() = std::min(*constLb, *constOtherLb);
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
      if (!constUb.has_value() || !constOtherUb.has_value())
        return failure();
      std::fill(maxUb.begin(), maxUb.end(), 0);
      maxUb.back() = std::max(*constUb, *constOtherUb);
    }

    std::fill(newLb.begin(), newLb.end(), 0);
    std::fill(newUb.begin(), newUb.end(), 0);

    // The divisor for lb, ub, otherLb, otherUb at this point is lbDivisor,
    // and so it's the divisor for newLb and newUb as well.
    newLb[d] = lbFloorDivisor;
    newUb[d] = -lbFloorDivisor;
    // Copy over the symbolic part + constant term.
    std::copy(minLb.begin(), minLb.end(), newLb.begin() + getNumDimVars());
    std::transform(newLb.begin() + getNumDimVars(), newLb.end(),
                   newLb.begin() + getNumDimVars(), std::negate<MPInt>());
    std::copy(maxUb.begin(), maxUb.end(), newUb.begin() + getNumDimVars());

    boundingLbs.push_back(newLb);
    boundingUbs.push_back(newUb);
  }

  // Clear all constraints and add the lower/upper bounds for the bounding box.
  clearConstraints();
  for (unsigned d = 0, e = getNumDimVars(); d < e; ++d) {
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

bool IntegerRelation::isColZero(unsigned pos) const {
  unsigned rowPos;
  return !findConstraintWithNonZeroAt(pos, /*isEq=*/false, &rowPos) &&
         !findConstraintWithNonZeroAt(pos, /*isEq=*/true, &rowPos);
}

/// Find positions of inequalities and equalities that do not have a coefficient
/// for [pos, pos + num) variables.
static void getIndependentConstraints(const IntegerRelation &cst, unsigned pos,
                                      unsigned num,
                                      SmallVectorImpl<unsigned> &nbIneqIndices,
                                      SmallVectorImpl<unsigned> &nbEqIndices) {
  assert(pos < cst.getNumVars() && "invalid start position");
  assert(pos + num <= cst.getNumVars() && "invalid limit");

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

void IntegerRelation::removeIndependentConstraints(unsigned pos, unsigned num) {
  assert(pos + num <= getNumVars() && "invalid range");

  // Remove constraints that are independent of these variables.
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

IntegerPolyhedron IntegerRelation::getDomainSet() const {
  IntegerRelation copyRel = *this;

  // Convert Range variables to Local variables.
  copyRel.convertVarKind(VarKind::Range, 0, getNumVarKind(VarKind::Range),
                         VarKind::Local);

  // Convert Domain variables to SetDim(Range) variables.
  copyRel.convertVarKind(VarKind::Domain, 0, getNumVarKind(VarKind::Domain),
                         VarKind::SetDim);

  return IntegerPolyhedron(std::move(copyRel));
}

bool IntegerRelation::removeDuplicateConstraints() {
  bool changed = false;
  SmallDenseMap<ArrayRef<MPInt>, unsigned> hashTable;
  unsigned ineqs = getNumInequalities(), cols = getNumCols();

  if (ineqs <= 1)
    return changed;

  // Check if the non-constant part of the constraint is the same.
  ArrayRef<MPInt> row = getInequality(0).drop_back();
  hashTable.insert({row, 0});
  for (unsigned k = 1; k < ineqs; ++k) {
    row = getInequality(k).drop_back();
    if (!hashTable.contains(row)) {
      hashTable.insert({row, k});
      continue;
    }

    // For identical cases, keep only the smaller part of the constant term.
    unsigned l = hashTable[row];
    changed = true;
    if (atIneq(k, cols - 1) <= atIneq(l, cols - 1))
      inequalities.swapRows(k, l);
    removeInequality(k);
    --k;
    --ineqs;
  }

  // Check the neg form of each inequality, need an extra vector to store it.
  SmallVector<MPInt> negIneq(cols - 1);
  for (unsigned k = 0; k < ineqs; ++k) {
    row = getInequality(k).drop_back();
    negIneq.assign(row.begin(), row.end());
    for (MPInt &ele : negIneq)
      ele = -ele;
    if (!hashTable.contains(negIneq))
      continue;

    // For cases where the neg is the same as other inequalities, check that the
    // sum of their constant terms is positive.
    unsigned l = hashTable[row];
    auto sum = atIneq(l, cols - 1) + atIneq(k, cols - 1);
    if (sum > 0 || l == k)
      continue;

    // A sum of constant terms equal to zero combines two inequalities into one
    // equation, less than zero means the set is empty.
    changed = true;
    if (k < l)
      std::swap(l, k);
    if (sum == 0) {
      addEquality(getInequality(k));
      removeInequality(k);
      removeInequality(l);
    } else
      *this = getEmpty(getSpace());
    break;
  }

  return changed;
}

IntegerPolyhedron IntegerRelation::getRangeSet() const {
  IntegerRelation copyRel = *this;

  // Convert Domain variables to Local variables.
  copyRel.convertVarKind(VarKind::Domain, 0, getNumVarKind(VarKind::Domain),
                         VarKind::Local);

  // We do not need to do anything to Range variables since they are already in
  // SetDim position.

  return IntegerPolyhedron(std::move(copyRel));
}

void IntegerRelation::intersectDomain(const IntegerPolyhedron &poly) {
  assert(getDomainSet().getSpace().isCompatible(poly.getSpace()) &&
         "Domain set is not compatible with poly");

  // Treating the poly as a relation, convert it from `0 -> R` to `R -> 0`.
  IntegerRelation rel = poly;
  rel.inverse();

  // Append dummy range variables to make the spaces compatible.
  rel.appendVar(VarKind::Range, getNumRangeVars());

  // Intersect in place.
  mergeLocalVars(rel);
  append(rel);
}

void IntegerRelation::intersectRange(const IntegerPolyhedron &poly) {
  assert(getRangeSet().getSpace().isCompatible(poly.getSpace()) &&
         "Range set is not compatible with poly");

  IntegerRelation rel = poly;

  // Append dummy domain variables to make the spaces compatible.
  rel.appendVar(VarKind::Domain, getNumDomainVars());

  mergeLocalVars(rel);
  append(rel);
}

void IntegerRelation::inverse() {
  unsigned numRangeVars = getNumVarKind(VarKind::Range);
  convertVarKind(VarKind::Domain, 0, getVarKindEnd(VarKind::Domain),
                 VarKind::Range);
  convertVarKind(VarKind::Range, 0, numRangeVars, VarKind::Domain);
}

void IntegerRelation::compose(const IntegerRelation &rel) {
  assert(getRangeSet().getSpace().isCompatible(rel.getDomainSet().getSpace()) &&
         "Range of `this` should be compatible with Domain of `rel`");

  IntegerRelation copyRel = rel;

  // Let relation `this` be R1: A -> B, and `rel` be R2: B -> C.
  // We convert R1 to A -> (B X C), and R2 to B X C then intersect the range of
  // R1 with R2. After this, we get R1: A -> C, by projecting out B.
  // TODO: Using nested spaces here would help, since we could directly
  // intersect the range with another relation.
  unsigned numBVars = getNumRangeVars();

  // Convert R1 from A -> B to A -> (B X C).
  appendVar(VarKind::Range, copyRel.getNumRangeVars());

  // Convert R2 to B X C.
  copyRel.convertVarKind(VarKind::Domain, 0, numBVars, VarKind::Range, 0);

  // Intersect R2 to range of R1.
  intersectRange(IntegerPolyhedron(copyRel));

  // Project out B in R1.
  convertVarKind(VarKind::Range, 0, numBVars, VarKind::Local);
}

void IntegerRelation::applyDomain(const IntegerRelation &rel) {
  inverse();
  compose(rel);
  inverse();
}

void IntegerRelation::applyRange(const IntegerRelation &rel) { compose(rel); }

void IntegerRelation::printSpace(raw_ostream &os) const {
  space.print(os);
  os << getNumConstraints() << " constraints\n";
}

void IntegerRelation::print(raw_ostream &os) const {
  assert(hasConsistentState());
  printSpace(os);
  for (unsigned i = 0, e = getNumEqualities(); i < e; ++i) {
    os << " ";
    for (unsigned j = 0, f = getNumCols(); j < f; ++j) {
      os << atEq(i, j) << "\t";
    }
    os << "= 0\n";
  }
  for (unsigned i = 0, e = getNumInequalities(); i < e; ++i) {
    os << " ";
    for (unsigned j = 0, f = getNumCols(); j < f; ++j) {
      os << atIneq(i, j) << "\t";
    }
    os << ">= 0\n";
  }
  os << '\n';
}

void IntegerRelation::dump() const { print(llvm::errs()); }

unsigned IntegerPolyhedron::insertVar(VarKind kind, unsigned pos,
                                      unsigned num) {
  assert((kind != VarKind::Domain || num == 0) &&
         "Domain has to be zero in a set");
  return IntegerRelation::insertVar(kind, pos, num);
}
IntegerPolyhedron
IntegerPolyhedron::intersect(const IntegerPolyhedron &other) const {
  return IntegerPolyhedron(IntegerRelation::intersect(other));
}

PresburgerSet IntegerPolyhedron::subtract(const PresburgerSet &other) const {
  return PresburgerSet(IntegerRelation::subtract(other));
}
