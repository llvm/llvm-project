//===- Simplex.cpp - MLIR Simplex Class -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Analysis/Presburger/Fraction.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include "mlir/Analysis/Presburger/Utils.h"
#include "llvm/ADT/DynamicAPInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <functional>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>

using namespace mlir;
using namespace presburger;

using Direction = Simplex::Direction;

const int nullIndex = std::numeric_limits<int>::max();

// Return a + scale*b;
LLVM_ATTRIBUTE_UNUSED
static SmallVector<DynamicAPInt, 8>
scaleAndAddForAssert(ArrayRef<DynamicAPInt> a, const DynamicAPInt &scale,
                     ArrayRef<DynamicAPInt> b) {
  assert(a.size() == b.size());
  SmallVector<DynamicAPInt, 8> res;
  res.reserve(a.size());
  for (unsigned i = 0, e = a.size(); i < e; ++i)
    res.emplace_back(a[i] + scale * b[i]);
  return res;
}

SimplexBase::SimplexBase(unsigned nVar, bool mustUseBigM)
    : usingBigM(mustUseBigM), nRedundant(0), nSymbol(0),
      tableau(0, getNumFixedCols() + nVar), empty(false) {
  var.reserve(nVar);
  colUnknown.reserve(nVar + 1);
  colUnknown.insert(colUnknown.begin(), getNumFixedCols(), nullIndex);
  for (unsigned i = 0; i < nVar; ++i) {
    var.emplace_back(Orientation::Column, /*restricted=*/false,
                     /*pos=*/getNumFixedCols() + i);
    colUnknown.emplace_back(i);
  }
}

SimplexBase::SimplexBase(unsigned nVar, bool mustUseBigM,
                         const llvm::SmallBitVector &isSymbol)
    : SimplexBase(nVar, mustUseBigM) {
  assert(isSymbol.size() == nVar && "invalid bitmask!");
  // Invariant: nSymbol is the number of symbols that have been marked
  // already and these occupy the columns
  // [getNumFixedCols(), getNumFixedCols() + nSymbol).
  for (unsigned symbolIdx : isSymbol.set_bits()) {
    var[symbolIdx].isSymbol = true;
    swapColumns(var[symbolIdx].pos, getNumFixedCols() + nSymbol);
    ++nSymbol;
  }
}

const Simplex::Unknown &SimplexBase::unknownFromIndex(int index) const {
  assert(index != nullIndex && "nullIndex passed to unknownFromIndex");
  return index >= 0 ? var[index] : con[~index];
}

const Simplex::Unknown &SimplexBase::unknownFromColumn(unsigned col) const {
  assert(col < getNumColumns() && "Invalid column");
  return unknownFromIndex(colUnknown[col]);
}

const Simplex::Unknown &SimplexBase::unknownFromRow(unsigned row) const {
  assert(row < getNumRows() && "Invalid row");
  return unknownFromIndex(rowUnknown[row]);
}

Simplex::Unknown &SimplexBase::unknownFromIndex(int index) {
  assert(index != nullIndex && "nullIndex passed to unknownFromIndex");
  return index >= 0 ? var[index] : con[~index];
}

Simplex::Unknown &SimplexBase::unknownFromColumn(unsigned col) {
  assert(col < getNumColumns() && "Invalid column");
  return unknownFromIndex(colUnknown[col]);
}

Simplex::Unknown &SimplexBase::unknownFromRow(unsigned row) {
  assert(row < getNumRows() && "Invalid row");
  return unknownFromIndex(rowUnknown[row]);
}

unsigned SimplexBase::addZeroRow(bool makeRestricted) {
  // Resize the tableau to accommodate the extra row.
  unsigned newRow = tableau.appendExtraRow();
  assert(getNumRows() == getNumRows() && "Inconsistent tableau size");
  rowUnknown.emplace_back(~con.size());
  con.emplace_back(Orientation::Row, makeRestricted, newRow);
  undoLog.emplace_back(UndoLogEntry::RemoveLastConstraint);
  tableau(newRow, 0) = 1;
  return newRow;
}

/// Add a new row to the tableau corresponding to the given constant term and
/// list of coefficients. The coefficients are specified as a vector of
/// (variable index, coefficient) pairs.
unsigned SimplexBase::addRow(ArrayRef<DynamicAPInt> coeffs,
                             bool makeRestricted) {
  assert(coeffs.size() == var.size() + 1 &&
         "Incorrect number of coefficients!");
  assert(var.size() + getNumFixedCols() == getNumColumns() &&
         "inconsistent column count!");

  unsigned newRow = addZeroRow(makeRestricted);
  tableau(newRow, 1) = coeffs.back();
  if (usingBigM) {
    // When the lexicographic pivot rule is used, instead of the variables
    //
    // x, y, z ...
    //
    // we internally use the variables
    //
    // M, M + x, M + y, M + z, ...
    //
    // where M is the big M parameter. As such, when the user tries to add
    // a row ax + by + cz + d, we express it in terms of our internal variables
    // as -(a + b + c)M + a(M + x) + b(M + y) + c(M + z) + d.
    //
    // Symbols don't use the big M parameter since they do not get lex
    // optimized.
    DynamicAPInt bigMCoeff(0);
    for (unsigned i = 0; i < coeffs.size() - 1; ++i)
      if (!var[i].isSymbol)
        bigMCoeff -= coeffs[i];
    // The coefficient to the big M parameter is stored in column 2.
    tableau(newRow, 2) = bigMCoeff;
  }

  // Process each given variable coefficient.
  for (unsigned i = 0; i < var.size(); ++i) {
    unsigned pos = var[i].pos;
    if (coeffs[i] == 0)
      continue;

    if (var[i].orientation == Orientation::Column) {
      // If a variable is in column position at column col, then we just add the
      // coefficient for that variable (scaled by the common row denominator) to
      // the corresponding entry in the new row.
      tableau(newRow, pos) += coeffs[i] * tableau(newRow, 0);
      continue;
    }

    // If the variable is in row position, we need to add that row to the new
    // row, scaled by the coefficient for the variable, accounting for the two
    // rows potentially having different denominators. The new denominator is
    // the lcm of the two.
    DynamicAPInt lcm = llvm::lcm(tableau(newRow, 0), tableau(pos, 0));
    DynamicAPInt nRowCoeff = lcm / tableau(newRow, 0);
    DynamicAPInt idxRowCoeff = coeffs[i] * (lcm / tableau(pos, 0));
    tableau(newRow, 0) = lcm;
    for (unsigned col = 1, e = getNumColumns(); col < e; ++col)
      tableau(newRow, col) =
          nRowCoeff * tableau(newRow, col) + idxRowCoeff * tableau(pos, col);
  }

  tableau.normalizeRow(newRow);
  // Push to undo log along with the index of the new constraint.
  return con.size() - 1;
}

namespace {
bool signMatchesDirection(const DynamicAPInt &elem, Direction direction) {
  assert(elem != 0 && "elem should not be 0");
  return direction == Direction::Up ? elem > 0 : elem < 0;
}

Direction flippedDirection(Direction direction) {
  return direction == Direction::Up ? Direction::Down : Simplex::Direction::Up;
}
} // namespace

/// We simply make the tableau consistent while maintaining a lexicopositive
/// basis transform, and then return the sample value. If the tableau becomes
/// empty, we return empty.
///
/// Let the variables be x = (x_1, ... x_n).
/// Let the basis unknowns be y = (y_1, ... y_n).
/// We have that x = A*y + b for some n x n matrix A and n x 1 column vector b.
///
/// As we will show below, A*y is either zero or lexicopositive.
/// Adding a lexicopositive vector to b will make it lexicographically
/// greater, so A*y + b is always equal to or lexicographically greater than b.
/// Thus, since we can attain x = b, that is the lexicographic minimum.
///
/// We have that every column in A is lexicopositive, i.e., has at least
/// one non-zero element, with the first such element being positive. Since for
/// the tableau to be consistent we must have non-negative sample values not
/// only for the constraints but also for the variables, we also have x >= 0 and
/// y >= 0, by which we mean every element in these vectors is non-negative.
///
/// Proof that if every column in A is lexicopositive, and y >= 0, then
/// A*y is zero or lexicopositive. Begin by considering A_1, the first row of A.
/// If this row is all zeros, then (A*y)_1 = (A_1)*y = 0; proceed to the next
/// row. If we run out of rows, A*y is zero and we are done; otherwise, we
/// encounter some row A_i that has a non-zero element. Every column is
/// lexicopositive and so has some positive element before any negative elements
/// occur, so the element in this row for any column, if non-zero, must be
/// positive. Consider (A*y)_i = (A_i)*y. All the elements in both vectors are
/// non-negative, so if this is non-zero then it must be positive. Then the
/// first non-zero element of A*y is positive so A*y is lexicopositive.
///
/// Otherwise, if (A_i)*y is zero, then for every column j that had a non-zero
/// element in A_i, y_j is zero. Thus these columns have no contribution to A*y
/// and we can completely ignore these columns of A. We now continue downwards,
/// looking for rows of A that have a non-zero element other than in the ignored
/// columns. If we find one, say A_k, once again these elements must be positive
/// since they are the first non-zero element in each of these columns, so if
/// (A_k)*y is not zero then we have that A*y is lexicopositive and if not we
/// add these to the set of ignored columns and continue to the next row. If we
/// run out of rows, then A*y is zero and we are done.
MaybeOptimum<SmallVector<Fraction, 8>> LexSimplex::findRationalLexMin() {
  if (restoreRationalConsistency().failed()) {
    markEmpty();
    return OptimumKind::Empty;
  }
  return getRationalSample();
}

/// Given a row that has a non-integer sample value, add an inequality such
/// that this fractional sample value is cut away from the polytope. The added
/// inequality will be such that no integer points are removed. i.e., the
/// integer lexmin, if it exists, is the same with and without this constraint.
///
/// Let the row be
/// (c + coeffM*M + a_1*s_1 + ... + a_m*s_m + b_1*y_1 + ... + b_n*y_n)/d,
/// where s_1, ... s_m are the symbols and
///       y_1, ... y_n are the other basis unknowns.
///
/// For this to be an integer, we want
/// coeffM*M + a_1*s_1 + ... + a_m*s_m + b_1*y_1 + ... + b_n*y_n = -c (mod d)
/// Note that this constraint must always hold, independent of the basis,
/// becuse the row unknown's value always equals this expression, even if *we*
/// later compute the sample value from a different expression based on a
/// different basis.
///
/// Let us assume that M has a factor of d in it. Imposing this constraint on M
/// does not in any way hinder us from finding a value of M that is big enough.
/// Moreover, this function is only called when the symbolic part of the sample,
/// a_1*s_1 + ... + a_m*s_m, is known to be an integer.
///
/// Also, we can safely reduce the coefficients modulo d, so we have:
///
/// (b_1%d)y_1 + ... + (b_n%d)y_n = (-c%d) + k*d for some integer `k`
///
/// Note that all coefficient modulos here are non-negative. Also, all the
/// unknowns are non-negative here as both constraints and variables are
/// non-negative in LexSimplexBase. (We used the big M trick to make the
/// variables non-negative). Therefore, the LHS here is non-negative.
/// Since 0 <= (-c%d) < d, k is the quotient of dividing the LHS by d and
/// is therefore non-negative as well.
///
/// So we have
/// ((b_1%d)y_1 + ... + (b_n%d)y_n - (-c%d))/d >= 0.
///
/// The constraint is violated when added (it would be useless otherwise)
/// so we immediately try to move it to a column.
LogicalResult LexSimplexBase::addCut(unsigned row) {
  DynamicAPInt d = tableau(row, 0);
  unsigned cutRow = addZeroRow(/*makeRestricted=*/true);
  tableau(cutRow, 0) = d;
  tableau(cutRow, 1) = -mod(-tableau(row, 1), d); // -c%d.
  tableau(cutRow, 2) = 0;
  for (unsigned col = 3 + nSymbol, e = getNumColumns(); col < e; ++col)
    tableau(cutRow, col) = mod(tableau(row, col), d); // b_i%d.
  return moveRowUnknownToColumn(cutRow);
}

std::optional<unsigned> LexSimplex::maybeGetNonIntegralVarRow() const {
  for (const Unknown &u : var) {
    if (u.orientation == Orientation::Column)
      continue;
    // If the sample value is of the form (a/d)M + b/d, we need b to be
    // divisible by d. We assume M contains all possible
    // factors and is divisible by everything.
    unsigned row = u.pos;
    if (tableau(row, 1) % tableau(row, 0) != 0)
      return row;
  }
  return {};
}

MaybeOptimum<SmallVector<DynamicAPInt, 8>> LexSimplex::findIntegerLexMin() {
  // We first try to make the tableau consistent.
  if (restoreRationalConsistency().failed())
    return OptimumKind::Empty;

  // Then, if the sample value is integral, we are done.
  while (std::optional<unsigned> maybeRow = maybeGetNonIntegralVarRow()) {
    // Otherwise, for the variable whose row has a non-integral sample value,
    // we add a cut, a constraint that remove this rational point
    // while preserving all integer points, thus keeping the lexmin the same.
    // We then again try to make the tableau with the new constraint
    // consistent. This continues until the tableau becomes empty, in which
    // case there is no integer point, or until there are no variables with
    // non-integral sample values.
    //
    // Failure indicates that the tableau became empty, which occurs when the
    // polytope is integer empty.
    if (addCut(*maybeRow).failed())
      return OptimumKind::Empty;
    if (restoreRationalConsistency().failed())
      return OptimumKind::Empty;
  }

  MaybeOptimum<SmallVector<Fraction, 8>> sample = getRationalSample();
  assert(!sample.isEmpty() && "If we reached here the sample should exist!");
  if (sample.isUnbounded())
    return OptimumKind::Unbounded;
  return llvm::to_vector<8>(
      llvm::map_range(*sample, std::mem_fn(&Fraction::getAsInteger)));
}

bool LexSimplex::isSeparateInequality(ArrayRef<DynamicAPInt> coeffs) {
  SimplexRollbackScopeExit scopeExit(*this);
  addInequality(coeffs);
  return findIntegerLexMin().isEmpty();
}

bool LexSimplex::isRedundantInequality(ArrayRef<DynamicAPInt> coeffs) {
  return isSeparateInequality(getComplementIneq(coeffs));
}

SmallVector<DynamicAPInt, 8>
SymbolicLexSimplex::getSymbolicSampleNumerator(unsigned row) const {
  SmallVector<DynamicAPInt, 8> sample;
  sample.reserve(nSymbol + 1);
  for (unsigned col = 3; col < 3 + nSymbol; ++col)
    sample.emplace_back(tableau(row, col));
  sample.emplace_back(tableau(row, 1));
  return sample;
}

SmallVector<DynamicAPInt, 8>
SymbolicLexSimplex::getSymbolicSampleIneq(unsigned row) const {
  SmallVector<DynamicAPInt, 8> sample = getSymbolicSampleNumerator(row);
  // The inequality is equivalent to the GCD-normalized one.
  normalizeRange(sample);
  return sample;
}

void LexSimplexBase::appendSymbol() {
  appendVariable();
  swapColumns(3 + nSymbol, getNumColumns() - 1);
  var.back().isSymbol = true;
  nSymbol++;
}

static bool isRangeDivisibleBy(ArrayRef<DynamicAPInt> range,
                               const DynamicAPInt &divisor) {
  assert(divisor > 0 && "divisor must be positive!");
  return llvm::all_of(
      range, [divisor](const DynamicAPInt &x) { return x % divisor == 0; });
}

bool SymbolicLexSimplex::isSymbolicSampleIntegral(unsigned row) const {
  DynamicAPInt denom = tableau(row, 0);
  return tableau(row, 1) % denom == 0 &&
         isRangeDivisibleBy(tableau.getRow(row).slice(3, nSymbol), denom);
}

/// This proceeds similarly to LexSimplexBase::addCut(). We are given a row that
/// has a symbolic sample value with fractional coefficients.
///
/// Let the row be
/// (c + coeffM*M + sum_i a_i*s_i + sum_j b_j*y_j)/d,
/// where s_1, ... s_m are the symbols and
///       y_1, ... y_n are the other basis unknowns.
///
/// As in LexSimplex::addCut, for this to be an integer, we want
///
/// coeffM*M + sum_j b_j*y_j = -c + sum_i (-a_i*s_i) (mod d)
///
/// This time, a_1*s_1 + ... + a_m*s_m may not be an integer. We find that
///
/// sum_i (b_i%d)y_i = ((-c%d) + sum_i (-a_i%d)s_i)%d + k*d for some integer k
///
/// where we take a modulo of the whole symbolic expression on the right to
/// bring it into the range [0, d - 1]. Therefore, as in addCut(),
/// k is the quotient on dividing the LHS by d, and since LHS >= 0, we have
/// k >= 0 as well. If all the a_i are divisible by d, then we can add the
/// constraint directly.  Otherwise, we realize the modulo of the symbolic
/// expression by adding a division variable
///
/// q = ((-c%d) + sum_i (-a_i%d)s_i)/d
///
/// to the symbol domain, so the equality becomes
///
/// sum_i (b_i%d)y_i = (-c%d) + sum_i (-a_i%d)s_i - q*d + k*d for some integer k
///
/// So the cut is
/// (sum_i (b_i%d)y_i - (-c%d) - sum_i (-a_i%d)s_i + q*d)/d >= 0
/// This constraint is violated when added so we immediately try to move it to a
/// column.
LogicalResult SymbolicLexSimplex::addSymbolicCut(unsigned row) {
  DynamicAPInt d = tableau(row, 0);
  if (isRangeDivisibleBy(tableau.getRow(row).slice(3, nSymbol), d)) {
    // The coefficients of symbols in the symbol numerator are divisible
    // by the denominator, so we can add the constraint directly,
    // i.e., ignore the symbols and add a regular cut as in addCut().
    return addCut(row);
  }

  // Construct the division variable `q = ((-c%d) + sum_i (-a_i%d)s_i)/d`.
  SmallVector<DynamicAPInt, 8> divCoeffs;
  divCoeffs.reserve(nSymbol + 1);
  DynamicAPInt divDenom = d;
  for (unsigned col = 3; col < 3 + nSymbol; ++col)
    divCoeffs.emplace_back(mod(-tableau(row, col), divDenom)); // (-a_i%d)s_i
  divCoeffs.emplace_back(mod(-tableau(row, 1), divDenom));     // -c%d.
  normalizeDiv(divCoeffs, divDenom);

  domainSimplex.addDivisionVariable(divCoeffs, divDenom);
  domainPoly.addLocalFloorDiv(divCoeffs, divDenom);

  // Update `this` to account for the additional symbol we just added.
  appendSymbol();

  // Add the cut (sum_i (b_i%d)y_i - (-c%d) + sum_i -(-a_i%d)s_i + q*d)/d >= 0.
  unsigned cutRow = addZeroRow(/*makeRestricted=*/true);
  tableau(cutRow, 0) = d;
  tableau(cutRow, 2) = 0;

  tableau(cutRow, 1) = -mod(-tableau(row, 1), d); // -(-c%d).
  for (unsigned col = 3; col < 3 + nSymbol - 1; ++col)
    tableau(cutRow, col) = -mod(-tableau(row, col), d); // -(-a_i%d)s_i.
  tableau(cutRow, 3 + nSymbol - 1) = d;                 // q*d.

  for (unsigned col = 3 + nSymbol, e = getNumColumns(); col < e; ++col)
    tableau(cutRow, col) = mod(tableau(row, col), d); // (b_i%d)y_i.
  return moveRowUnknownToColumn(cutRow);
}

void SymbolicLexSimplex::recordOutput(SymbolicLexOpt &result) const {
  IntMatrix output(0, domainPoly.getNumVars() + 1);
  output.reserveRows(result.lexopt.getNumOutputs());
  for (const Unknown &u : var) {
    if (u.isSymbol)
      continue;

    if (u.orientation == Orientation::Column) {
      // M + u has a sample value of zero so u has a sample value of -M, i.e,
      // unbounded.
      result.unboundedDomain.unionInPlace(domainPoly);
      return;
    }

    DynamicAPInt denom = tableau(u.pos, 0);
    if (tableau(u.pos, 2) < denom) {
      // M + u has a sample value of fM + something, where f < 1, so
      // u = (f - 1)M + something, which has a negative coefficient for M,
      // and so is unbounded.
      result.unboundedDomain.unionInPlace(domainPoly);
      return;
    }
    assert(tableau(u.pos, 2) == denom &&
           "Coefficient of M should not be greater than 1!");

    SmallVector<DynamicAPInt, 8> sample = getSymbolicSampleNumerator(u.pos);
    for (DynamicAPInt &elem : sample) {
      assert(elem % denom == 0 && "coefficients must be integral!");
      elem /= denom;
    }
    output.appendExtraRow(sample);
  }

  // Store the output in a MultiAffineFunction and add it the result.
  PresburgerSpace funcSpace = result.lexopt.getSpace();
  funcSpace.insertVar(VarKind::Local, 0, domainPoly.getNumLocalVars());

  result.lexopt.addPiece(
      {PresburgerSet(domainPoly),
       MultiAffineFunction(funcSpace, output, domainPoly.getLocalReprs())});
}

std::optional<unsigned> SymbolicLexSimplex::maybeGetAlwaysViolatedRow() {
  // First look for rows that are clearly violated just from the big M
  // coefficient, without needing to perform any simplex queries on the domain.
  for (unsigned row = 0, e = getNumRows(); row < e; ++row)
    if (tableau(row, 2) < 0)
      return row;

  for (unsigned row = 0, e = getNumRows(); row < e; ++row) {
    if (tableau(row, 2) > 0)
      continue;
    if (domainSimplex.isSeparateInequality(getSymbolicSampleIneq(row))) {
      // Sample numerator always takes negative values in the symbol domain.
      return row;
    }
  }
  return {};
}

std::optional<unsigned> SymbolicLexSimplex::maybeGetNonIntegralVarRow() {
  for (const Unknown &u : var) {
    if (u.orientation == Orientation::Column)
      continue;
    assert(!u.isSymbol && "Symbol should not be in row orientation!");
    if (!isSymbolicSampleIntegral(u.pos))
      return u.pos;
  }
  return {};
}

/// The non-branching pivots are just the ones moving the rows
/// that are always violated in the symbol domain.
LogicalResult SymbolicLexSimplex::doNonBranchingPivots() {
  while (std::optional<unsigned> row = maybeGetAlwaysViolatedRow())
    if (moveRowUnknownToColumn(*row).failed())
      return failure();
  return success();
}

SymbolicLexOpt SymbolicLexSimplex::computeSymbolicIntegerLexMin() {
  SymbolicLexOpt result(PresburgerSpace::getRelationSpace(
      /*numDomain=*/domainPoly.getNumDimVars(),
      /*numRange=*/var.size() - nSymbol,
      /*numSymbols=*/domainPoly.getNumSymbolVars()));

  /// The algorithm is more naturally expressed recursively, but we implement
  /// it iteratively here to avoid potential issues with stack overflows in the
  /// compiler. We explicitly maintain the stack frames in a vector.
  ///
  /// To "recurse", we store the current "stack frame", i.e., state variables
  /// that we will need when we "return", into `stack`, increment `level`, and
  /// `continue`. To "tail recurse", we just `continue`.
  /// To "return", we decrement `level` and `continue`.
  ///
  /// When there is no stack frame for the current `level`, this indicates that
  /// we have just "recursed" or "tail recursed". When there does exist one,
  /// this indicates that we have just "returned" from recursing. There is only
  /// one point at which non-tail calls occur so we always "return" there.
  unsigned level = 1;
  struct StackFrame {
    int splitIndex;
    unsigned snapshot;
    unsigned domainSnapshot;
    IntegerRelation::CountsSnapshot domainPolyCounts;
  };
  SmallVector<StackFrame, 8> stack;

  while (level > 0) {
    assert(level >= stack.size());
    if (level > stack.size()) {
      if (empty || domainSimplex.findIntegerLexMin().isEmpty()) {
        // No integer points; return.
        --level;
        continue;
      }

      if (doNonBranchingPivots().failed()) {
        // Could not find pivots for violated constraints; return.
        --level;
        continue;
      }

      SmallVector<DynamicAPInt, 8> symbolicSample;
      unsigned splitRow = 0;
      for (unsigned e = getNumRows(); splitRow < e; ++splitRow) {
        if (tableau(splitRow, 2) > 0)
          continue;
        assert(tableau(splitRow, 2) == 0 &&
               "Non-branching pivots should have been handled already!");

        symbolicSample = getSymbolicSampleIneq(splitRow);
        if (domainSimplex.isRedundantInequality(symbolicSample))
          continue;

        // It's neither redundant nor separate, so it takes both positive and
        // negative values, and hence constitutes a row for which we need to
        // split the domain and separately run each case.
        assert(!domainSimplex.isSeparateInequality(symbolicSample) &&
               "Non-branching pivots should have been handled already!");
        break;
      }

      if (splitRow < getNumRows()) {
        unsigned domainSnapshot = domainSimplex.getSnapshot();
        IntegerRelation::CountsSnapshot domainPolyCounts =
            domainPoly.getCounts();

        // First, we consider the part of the domain where the row is not
        // violated. We don't have to do any pivots for the row in this case,
        // but we record the additional constraint that defines this part of
        // the domain.
        domainSimplex.addInequality(symbolicSample);
        domainPoly.addInequality(symbolicSample);

        // Recurse.
        //
        // On return, the basis as a set is preserved but not the internal
        // ordering within rows or columns. Thus, we take note of the index of
        // the Unknown that caused the split, which may be in a different
        // row when we come back from recursing. We will need this to recurse
        // on the other part of the split domain, where the row is violated.
        //
        // Note that we have to capture the index above and not a reference to
        // the Unknown itself, since the array it lives in might get
        // reallocated.
        int splitIndex = rowUnknown[splitRow];
        unsigned snapshot = getSnapshot();
        stack.emplace_back(
            StackFrame{splitIndex, snapshot, domainSnapshot, domainPolyCounts});
        ++level;
        continue;
      }

      // The tableau is rationally consistent for the current domain.
      // Now we look for non-integral sample values and add cuts for them.
      if (std::optional<unsigned> row = maybeGetNonIntegralVarRow()) {
        if (addSymbolicCut(*row).failed()) {
          // No integral points; return.
          --level;
          continue;
        }

        // Rerun this level with the added cut constraint (tail recurse).
        continue;
      }

      // Record output and return.
      recordOutput(result);
      --level;
      continue;
    }

    if (level == stack.size()) {
      // We have "returned" from "recursing".
      const StackFrame &frame = stack.back();
      domainPoly.truncate(frame.domainPolyCounts);
      domainSimplex.rollback(frame.domainSnapshot);
      rollback(frame.snapshot);
      const Unknown &u = unknownFromIndex(frame.splitIndex);

      // Drop the frame. We don't need it anymore.
      stack.pop_back();

      // Now we consider the part of the domain where the unknown `splitIndex`
      // was negative.
      assert(u.orientation == Orientation::Row &&
             "The split row should have been returned to row orientation!");
      SmallVector<DynamicAPInt, 8> splitIneq =
          getComplementIneq(getSymbolicSampleIneq(u.pos));
      normalizeRange(splitIneq);
      if (moveRowUnknownToColumn(u.pos).failed()) {
        // The unknown can't be made non-negative; return.
        --level;
        continue;
      }

      // The unknown can be made negative; recurse with the corresponding domain
      // constraints.
      domainSimplex.addInequality(splitIneq);
      domainPoly.addInequality(splitIneq);

      // We are now taking care of the second half of the domain and we don't
      // need to do anything else here after returning, so it's a tail recurse.
      continue;
    }
  }

  return result;
}

bool LexSimplex::rowIsViolated(unsigned row) const {
  if (tableau(row, 2) < 0)
    return true;
  if (tableau(row, 2) == 0 && tableau(row, 1) < 0)
    return true;
  return false;
}

std::optional<unsigned> LexSimplex::maybeGetViolatedRow() const {
  for (unsigned row = 0, e = getNumRows(); row < e; ++row)
    if (rowIsViolated(row))
      return row;
  return {};
}

/// We simply look for violated rows and keep trying to move them to column
/// orientation, which always succeeds unless the constraints have no solution
/// in which case we just give up and return.
LogicalResult LexSimplex::restoreRationalConsistency() {
  if (empty)
    return failure();
  while (std::optional<unsigned> maybeViolatedRow = maybeGetViolatedRow())
    if (moveRowUnknownToColumn(*maybeViolatedRow).failed())
      return failure();
  return success();
}

// Move the row unknown to column orientation while preserving lexicopositivity
// of the basis transform. The sample value of the row must be non-positive.
//
// We only consider pivots where the pivot element is positive. Suppose no such
// pivot exists, i.e., some violated row has no positive coefficient for any
// basis unknown. The row can be represented as (s + c_1*u_1 + ... + c_n*u_n)/d,
// where d is the denominator, s is the sample value and the c_i are the basis
// coefficients. If s != 0, then since any feasible assignment of the basis
// satisfies u_i >= 0 for all i, and we have s < 0 as well as c_i < 0 for all i,
// any feasible assignment would violate this row and therefore the constraints
// have no solution.
//
// We can preserve lexicopositivity by picking the pivot column with positive
// pivot element that makes the lexicographically smallest change to the sample
// point.
//
// Proof. Let
// x = (x_1, ... x_n) be the variables,
// z = (z_1, ... z_m) be the constraints,
// y = (y_1, ... y_n) be the current basis, and
// define w = (x_1, ... x_n, z_1, ... z_m) = B*y + s.
// B is basically the simplex tableau of our implementation except that instead
// of only describing the transform to get back the non-basis unknowns, it
// defines the values of all the unknowns in terms of the basis unknowns.
// Similarly, s is the column for the sample value.
//
// Our goal is to show that each column in B, restricted to the first n
// rows, is lexicopositive after the pivot if it is so before. This is
// equivalent to saying the columns in the whole matrix are lexicopositive;
// there must be some non-zero element in every column in the first n rows since
// the n variables cannot be spanned without using all the n basis unknowns.
//
// Consider a pivot where z_i replaces y_j in the basis. Recall the pivot
// transform for the tableau derived for SimplexBase::pivot:
//
//            pivot col    other col                   pivot col    other col
// pivot row     a             b       ->   pivot row     1/a         -b/a
// other row     c             d            other row     c/a        d - bc/a
//
// Similarly, a pivot results in B changing to B' and c to c'; the difference
// between the tableau and these matrices B and B' is that there is no special
// case for the pivot row, since it continues to represent the same unknown. The
// same formula applies for all rows:
//
// B'.col(j) = B.col(j) / B(i,j)
// B'.col(k) = B.col(k) - B(i,k) * B.col(j) / B(i,j) for k != j
// and similarly, s' = s - s_i * B.col(j) / B(i,j).
//
// If s_i == 0, then the sample value remains unchanged. Otherwise, if s_i < 0,
// the change in sample value when pivoting with column a is lexicographically
// smaller than that when pivoting with column b iff B.col(a) / B(i, a) is
// lexicographically smaller than B.col(b) / B(i, b).
//
// Since B(i, j) > 0, column j remains lexicopositive.
//
// For the other columns, suppose C.col(k) is not lexicopositive.
// This means that for some p, for all t < p,
// C(t,k) = 0 => B(t,k) = B(t,j) * B(i,k) / B(i,j) and
// C(t,k) < 0 => B(p,k) < B(t,j) * B(i,k) / B(i,j),
// which is in contradiction to the fact that B.col(j) / B(i,j) must be
// lexicographically smaller than B.col(k) / B(i,k), since it lexicographically
// minimizes the change in sample value.
LogicalResult LexSimplexBase::moveRowUnknownToColumn(unsigned row) {
  std::optional<unsigned> maybeColumn;
  for (unsigned col = 3 + nSymbol, e = getNumColumns(); col < e; ++col) {
    if (tableau(row, col) <= 0)
      continue;
    maybeColumn =
        !maybeColumn ? col : getLexMinPivotColumn(row, *maybeColumn, col);
  }

  if (!maybeColumn)
    return failure();

  pivot(row, *maybeColumn);
  return success();
}

unsigned LexSimplexBase::getLexMinPivotColumn(unsigned row, unsigned colA,
                                              unsigned colB) const {
  // First, let's consider the non-symbolic case.
  // A pivot causes the following change. (in the diagram the matrix elements
  // are shown as rationals and there is no common denominator used)
  //
  //            pivot col    big M col      const col
  // pivot row     a            p               b
  // other row     c            q               d
  //                        |
  //                        v
  //
  //            pivot col    big M col      const col
  // pivot row     1/a         -p/a           -b/a
  // other row     c/a        q - pc/a       d - bc/a
  //
  // Let the sample value of the pivot row be s = pM + b before the pivot. Since
  // the pivot row represents a violated constraint we know that s < 0.
  //
  // If the variable is a non-pivot column, its sample value is zero before and
  // after the pivot.
  //
  // If the variable is the pivot column, then its sample value goes from 0 to
  // (-p/a)M + (-b/a), i.e. 0 to -(pM + b)/a. Thus the change in the sample
  // value is -s/a.
  //
  // If the variable is the pivot row, its sample value goes from s to 0, for a
  // change of -s.
  //
  // If the variable is a non-pivot row, its sample value changes from
  // qM + d to qM + d + (-pc/a)M + (-bc/a). Thus the change in sample value
  // is -(pM + b)(c/a) = -sc/a.
  //
  // Thus the change in sample value is either 0, -s/a, -s, or -sc/a. Here -s is
  // fixed for all calls to this function since the row and tableau are fixed.
  // The callee just wants to compare the return values with the return value of
  // other invocations of the same function. So the -s is common for all
  // comparisons involved and can be ignored, since -s is strictly positive.
  //
  // Thus we take away this common factor and just return 0, 1/a, 1, or c/a as
  // appropriate. This allows us to run the entire algorithm treating M
  // symbolically, as the pivot to be performed does not depend on the value
  // of M, so long as the sample value s is negative. Note that this is not
  // because of any special feature of M; by the same argument, we ignore the
  // symbols too. The caller ensure that the sample value s is negative for
  // all possible values of the symbols.
  auto getSampleChangeCoeffForVar = [this, row](unsigned col,
                                                const Unknown &u) -> Fraction {
    DynamicAPInt a = tableau(row, col);
    if (u.orientation == Orientation::Column) {
      // Pivot column case.
      if (u.pos == col)
        return {1, a};

      // Non-pivot column case.
      return {0, 1};
    }

    // Pivot row case.
    if (u.pos == row)
      return {1, 1};

    // Non-pivot row case.
    DynamicAPInt c = tableau(u.pos, col);
    return {c, a};
  };

  for (const Unknown &u : var) {
    Fraction changeA = getSampleChangeCoeffForVar(colA, u);
    Fraction changeB = getSampleChangeCoeffForVar(colB, u);
    if (changeA < changeB)
      return colA;
    if (changeA > changeB)
      return colB;
  }

  // If we reached here, both result in exactly the same changes, so it
  // doesn't matter which we return.
  return colA;
}

/// Find a pivot to change the sample value of the row in the specified
/// direction. The returned pivot row will involve `row` if and only if the
/// unknown is unbounded in the specified direction.
///
/// To increase (resp. decrease) the value of a row, we need to find a live
/// column with a non-zero coefficient. If the coefficient is positive, we need
/// to increase (decrease) the value of the column, and if the coefficient is
/// negative, we need to decrease (increase) the value of the column. Also,
/// we cannot decrease the sample value of restricted columns.
///
/// If multiple columns are valid, we break ties by considering a lexicographic
/// ordering where we prefer unknowns with lower index.
std::optional<SimplexBase::Pivot>
Simplex::findPivot(int row, Direction direction) const {
  std::optional<unsigned> col;
  for (unsigned j = 2, e = getNumColumns(); j < e; ++j) {
    DynamicAPInt elem = tableau(row, j);
    if (elem == 0)
      continue;

    if (unknownFromColumn(j).restricted &&
        !signMatchesDirection(elem, direction))
      continue;
    if (!col || colUnknown[j] < colUnknown[*col])
      col = j;
  }

  if (!col)
    return {};

  Direction newDirection =
      tableau(row, *col) < 0 ? flippedDirection(direction) : direction;
  std::optional<unsigned> maybePivotRow = findPivotRow(row, newDirection, *col);
  return Pivot{maybePivotRow.value_or(row), *col};
}

/// Swap the associated unknowns for the row and the column.
///
/// First we swap the index associated with the row and column. Then we update
/// the unknowns to reflect their new position and orientation.
void SimplexBase::swapRowWithCol(unsigned row, unsigned col) {
  std::swap(rowUnknown[row], colUnknown[col]);
  Unknown &uCol = unknownFromColumn(col);
  Unknown &uRow = unknownFromRow(row);
  uCol.orientation = Orientation::Column;
  uRow.orientation = Orientation::Row;
  uCol.pos = col;
  uRow.pos = row;
}

void SimplexBase::pivot(Pivot pair) { pivot(pair.row, pair.column); }

/// Pivot pivotRow and pivotCol.
///
/// Let R be the pivot row unknown and let C be the pivot col unknown.
/// Since initially R = a*C + sum b_i * X_i
/// (where the sum is over the other column's unknowns, x_i)
/// C = (R - (sum b_i * X_i))/a
///
/// Let u be some other row unknown.
/// u = c*C + sum d_i * X_i
/// So u = c*(R - sum b_i * X_i)/a + sum d_i * X_i
///
/// This results in the following transform:
///            pivot col    other col                   pivot col    other col
/// pivot row     a             b       ->   pivot row     1/a         -b/a
/// other row     c             d            other row     c/a        d - bc/a
///
/// Taking into account the common denominators p and q:
///
///            pivot col    other col                    pivot col   other col
/// pivot row     a/p          b/p     ->   pivot row      p/a         -b/a
/// other row     c/q          d/q          other row     cp/aq    (da - bc)/aq
///
/// The pivot row transform is accomplished be swapping a with the pivot row's
/// common denominator and negating the pivot row except for the pivot column
/// element.
void SimplexBase::pivot(unsigned pivotRow, unsigned pivotCol) {
  assert(pivotCol >= getNumFixedCols() && "Refusing to pivot invalid column");
  assert(!unknownFromColumn(pivotCol).isSymbol);

  swapRowWithCol(pivotRow, pivotCol);
  std::swap(tableau(pivotRow, 0), tableau(pivotRow, pivotCol));
  // We need to negate the whole pivot row except for the pivot column.
  if (tableau(pivotRow, 0) < 0) {
    // If the denominator is negative, we negate the row by simply negating the
    // denominator.
    tableau(pivotRow, 0) = -tableau(pivotRow, 0);
    tableau(pivotRow, pivotCol) = -tableau(pivotRow, pivotCol);
  } else {
    for (unsigned col = 1, e = getNumColumns(); col < e; ++col) {
      if (col == pivotCol)
        continue;
      tableau(pivotRow, col) = -tableau(pivotRow, col);
    }
  }
  tableau.normalizeRow(pivotRow);

  for (unsigned row = 0, numRows = getNumRows(); row < numRows; ++row) {
    if (row == pivotRow)
      continue;
    if (tableau(row, pivotCol) == 0) // Nothing to do.
      continue;
    tableau(row, 0) *= tableau(pivotRow, 0);
    for (unsigned col = 1, numCols = getNumColumns(); col < numCols; ++col) {
      if (col == pivotCol)
        continue;
      // Add rather than subtract because the pivot row has been negated.
      tableau(row, col) = tableau(row, col) * tableau(pivotRow, 0) +
                          tableau(row, pivotCol) * tableau(pivotRow, col);
    }
    tableau(row, pivotCol) *= tableau(pivotRow, pivotCol);
    tableau.normalizeRow(row);
  }
}

/// Perform pivots until the unknown has a non-negative sample value or until
/// no more upward pivots can be performed. Return success if we were able to
/// bring the row to a non-negative sample value, and failure otherwise.
LogicalResult Simplex::restoreRow(Unknown &u) {
  assert(u.orientation == Orientation::Row &&
         "unknown should be in row position");

  while (tableau(u.pos, 1) < 0) {
    std::optional<Pivot> maybePivot = findPivot(u.pos, Direction::Up);
    if (!maybePivot)
      break;

    pivot(*maybePivot);
    if (u.orientation == Orientation::Column)
      return success(); // the unknown is unbounded above.
  }
  return success(tableau(u.pos, 1) >= 0);
}

/// Find a row that can be used to pivot the column in the specified direction.
/// This returns an empty optional if and only if the column is unbounded in the
/// specified direction (ignoring skipRow, if skipRow is set).
///
/// If skipRow is set, this row is not considered, and (if it is restricted) its
/// restriction may be violated by the returned pivot. Usually, skipRow is set
/// because we don't want to move it to column position unless it is unbounded,
/// and we are either trying to increase the value of skipRow or explicitly
/// trying to make skipRow negative, so we are not concerned about this.
///
/// If the direction is up (resp. down) and a restricted row has a negative
/// (positive) coefficient for the column, then this row imposes a bound on how
/// much the sample value of the column can change. Such a row with constant
/// term c and coefficient f for the column imposes a bound of c/|f| on the
/// change in sample value (in the specified direction). (note that c is
/// non-negative here since the row is restricted and the tableau is consistent)
///
/// We iterate through the rows and pick the row which imposes the most
/// stringent bound, since pivoting with a row changes the row's sample value to
/// 0 and hence saturates the bound it imposes. We break ties between rows that
/// impose the same bound by considering a lexicographic ordering where we
/// prefer unknowns with lower index value.
std::optional<unsigned> Simplex::findPivotRow(std::optional<unsigned> skipRow,
                                              Direction direction,
                                              unsigned col) const {
  std::optional<unsigned> retRow;
  // Initialize these to zero in order to silence a warning about retElem and
  // retConst being used uninitialized in the initialization of `diff` below. In
  // reality, these are always initialized when that line is reached since these
  // are set whenever retRow is set.
  DynamicAPInt retElem, retConst;
  for (unsigned row = nRedundant, e = getNumRows(); row < e; ++row) {
    if (skipRow && row == *skipRow)
      continue;
    DynamicAPInt elem = tableau(row, col);
    if (elem == 0)
      continue;
    if (!unknownFromRow(row).restricted)
      continue;
    if (signMatchesDirection(elem, direction))
      continue;
    DynamicAPInt constTerm = tableau(row, 1);

    if (!retRow) {
      retRow = row;
      retElem = elem;
      retConst = constTerm;
      continue;
    }

    DynamicAPInt diff = retConst * elem - constTerm * retElem;
    if ((diff == 0 && rowUnknown[row] < rowUnknown[*retRow]) ||
        (diff != 0 && !signMatchesDirection(diff, direction))) {
      retRow = row;
      retElem = elem;
      retConst = constTerm;
    }
  }
  return retRow;
}

bool SimplexBase::isEmpty() const { return empty; }

void SimplexBase::swapRows(unsigned i, unsigned j) {
  if (i == j)
    return;
  tableau.swapRows(i, j);
  std::swap(rowUnknown[i], rowUnknown[j]);
  unknownFromRow(i).pos = i;
  unknownFromRow(j).pos = j;
}

void SimplexBase::swapColumns(unsigned i, unsigned j) {
  assert(i < getNumColumns() && j < getNumColumns() &&
         "Invalid columns provided!");
  if (i == j)
    return;
  tableau.swapColumns(i, j);
  std::swap(colUnknown[i], colUnknown[j]);
  unknownFromColumn(i).pos = i;
  unknownFromColumn(j).pos = j;
}

/// Mark this tableau empty and push an entry to the undo stack.
void SimplexBase::markEmpty() {
  // If the set is already empty, then we shouldn't add another UnmarkEmpty log
  // entry, since in that case the Simplex will be erroneously marked as
  // non-empty when rolling back past this point.
  if (empty)
    return;
  undoLog.emplace_back(UndoLogEntry::UnmarkEmpty);
  empty = true;
}

/// Add an inequality to the tableau. If coeffs is c_0, c_1, ... c_n, where n
/// is the current number of variables, then the corresponding inequality is
/// c_n + c_0*x_0 + c_1*x_1 + ... + c_{n-1}*x_{n-1} >= 0.
///
/// We add the inequality and mark it as restricted. We then try to make its
/// sample value non-negative. If this is not possible, the tableau has become
/// empty and we mark it as such.
void Simplex::addInequality(ArrayRef<DynamicAPInt> coeffs) {
  unsigned conIndex = addRow(coeffs, /*makeRestricted=*/true);
  LogicalResult result = restoreRow(con[conIndex]);
  if (result.failed())
    markEmpty();
}

/// Add an equality to the tableau. If coeffs is c_0, c_1, ... c_n, where n
/// is the current number of variables, then the corresponding equality is
/// c_n + c_0*x_0 + c_1*x_1 + ... + c_{n-1}*x_{n-1} == 0.
///
/// We simply add two opposing inequalities, which force the expression to
/// be zero.
void SimplexBase::addEquality(ArrayRef<DynamicAPInt> coeffs) {
  addInequality(coeffs);
  SmallVector<DynamicAPInt, 8> negatedCoeffs;
  negatedCoeffs.reserve(coeffs.size());
  for (const DynamicAPInt &coeff : coeffs)
    negatedCoeffs.emplace_back(-coeff);
  addInequality(negatedCoeffs);
}

unsigned SimplexBase::getNumVariables() const { return var.size(); }
unsigned SimplexBase::getNumConstraints() const { return con.size(); }

/// Return a snapshot of the current state. This is just the current size of the
/// undo log.
unsigned SimplexBase::getSnapshot() const { return undoLog.size(); }

unsigned SimplexBase::getSnapshotBasis() {
  SmallVector<int, 8> basis;
  basis.reserve(colUnknown.size());
  for (int index : colUnknown) {
    if (index != nullIndex)
      basis.emplace_back(index);
  }
  savedBases.emplace_back(std::move(basis));

  undoLog.emplace_back(UndoLogEntry::RestoreBasis);
  return undoLog.size() - 1;
}

void SimplexBase::removeLastConstraintRowOrientation() {
  assert(con.back().orientation == Orientation::Row);

  // Move this unknown to the last row and remove the last row from the
  // tableau.
  swapRows(con.back().pos, getNumRows() - 1);
  // It is not strictly necessary to shrink the tableau, but for now we
  // maintain the invariant that the tableau has exactly getNumRows()
  // rows.
  tableau.resizeVertically(getNumRows() - 1);
  rowUnknown.pop_back();
  con.pop_back();
}

// This doesn't find a pivot row only if the column has zero
// coefficients for every row.
//
// If the unknown is a constraint, this can't happen, since it was added
// initially as a row. Such a row could never have been pivoted to a column. So
// a pivot row will always be found if we have a constraint.
//
// If we have a variable, then the column has zero coefficients for every row
// iff no constraints have been added with a non-zero coefficient for this row.
std::optional<unsigned> SimplexBase::findAnyPivotRow(unsigned col) {
  for (unsigned row = nRedundant, e = getNumRows(); row < e; ++row)
    if (tableau(row, col) != 0)
      return row;
  return {};
}

// It's not valid to remove the constraint by deleting the column since this
// would result in an invalid basis.
void Simplex::undoLastConstraint() {
  if (con.back().orientation == Orientation::Column) {
    // We try to find any pivot row for this column that preserves tableau
    // consistency (except possibly the column itself, which is going to be
    // deallocated anyway).
    //
    // If no pivot row is found in either direction, then the unknown is
    // unbounded in both directions and we are free to perform any pivot at
    // all. To do this, we just need to find any row with a non-zero
    // coefficient for the column. findAnyPivotRow will always be able to
    // find such a row for a constraint.
    unsigned column = con.back().pos;
    if (std::optional<unsigned> maybeRow =
            findPivotRow({}, Direction::Up, column)) {
      pivot(*maybeRow, column);
    } else if (std::optional<unsigned> maybeRow =
                   findPivotRow({}, Direction::Down, column)) {
      pivot(*maybeRow, column);
    } else {
      std::optional<unsigned> row = findAnyPivotRow(column);
      assert(row && "Pivot should always exist for a constraint!");
      pivot(*row, column);
    }
  }
  removeLastConstraintRowOrientation();
}

// It's not valid to remove the constraint by deleting the column since this
// would result in an invalid basis.
void LexSimplexBase::undoLastConstraint() {
  if (con.back().orientation == Orientation::Column) {
    // When removing the last constraint during a rollback, we just need to find
    // any pivot at all, i.e., any row with non-zero coefficient for the
    // column, because when rolling back a lexicographic simplex, we always
    // end by restoring the exact basis that was present at the time of the
    // snapshot, so what pivots we perform while undoing doesn't matter as
    // long as we get the unknown to row orientation and remove it.
    unsigned column = con.back().pos;
    std::optional<unsigned> row = findAnyPivotRow(column);
    assert(row && "Pivot should always exist for a constraint!");
    pivot(*row, column);
  }
  removeLastConstraintRowOrientation();
}

void SimplexBase::undo(UndoLogEntry entry) {
  if (entry == UndoLogEntry::RemoveLastConstraint) {
    // Simplex and LexSimplex handle this differently, so we call out to a
    // virtual function to handle this.
    undoLastConstraint();
  } else if (entry == UndoLogEntry::RemoveLastVariable) {
    // Whenever we are rolling back the addition of a variable, it is guaranteed
    // that the variable will be in column position.
    //
    // We can see this as follows: any constraint that depends on this variable
    // was added after this variable was added, so the addition of such
    // constraints should already have been rolled back by the time we get to
    // rolling back the addition of the variable. Therefore, no constraint
    // currently has a component along the variable, so the variable itself must
    // be part of the basis.
    assert(var.back().orientation == Orientation::Column &&
           "Variable to be removed must be in column orientation!");

    if (var.back().isSymbol)
      nSymbol--;

    // Move this variable to the last column and remove the column from the
    // tableau.
    swapColumns(var.back().pos, getNumColumns() - 1);
    tableau.resizeHorizontally(getNumColumns() - 1);
    var.pop_back();
    colUnknown.pop_back();
  } else if (entry == UndoLogEntry::UnmarkEmpty) {
    empty = false;
  } else if (entry == UndoLogEntry::UnmarkLastRedundant) {
    nRedundant--;
  } else if (entry == UndoLogEntry::RestoreBasis) {
    assert(!savedBases.empty() && "No bases saved!");

    SmallVector<int, 8> basis = std::move(savedBases.back());
    savedBases.pop_back();

    for (int index : basis) {
      Unknown &u = unknownFromIndex(index);
      if (u.orientation == Orientation::Column)
        continue;
      for (unsigned col = getNumFixedCols(), e = getNumColumns(); col < e;
           col++) {
        assert(colUnknown[col] != nullIndex &&
               "Column should not be a fixed column!");
        if (llvm::is_contained(basis, colUnknown[col]))
          continue;
        if (tableau(u.pos, col) == 0)
          continue;
        pivot(u.pos, col);
        break;
      }

      assert(u.orientation == Orientation::Column && "No pivot found!");
    }
  }
}

/// Rollback to the specified snapshot.
///
/// We undo all the log entries until the log size when the snapshot was taken
/// is reached.
void SimplexBase::rollback(unsigned snapshot) {
  while (undoLog.size() > snapshot) {
    undo(undoLog.back());
    undoLog.pop_back();
  }
}

/// We add the usual floor division constraints:
/// `0 <= coeffs - denom*q <= denom - 1`, where `q` is the new division
/// variable.
///
/// This constrains the remainder `coeffs - denom*q` to be in the
/// range `[0, denom - 1]`, which fixes the integer value of the quotient `q`.
void SimplexBase::addDivisionVariable(ArrayRef<DynamicAPInt> coeffs,
                                      const DynamicAPInt &denom) {
  assert(denom > 0 && "Denominator must be positive!");
  appendVariable();

  SmallVector<DynamicAPInt, 8> ineq(coeffs);
  DynamicAPInt constTerm = ineq.back();
  ineq.back() = -denom;
  ineq.emplace_back(constTerm);
  addInequality(ineq);

  for (DynamicAPInt &coeff : ineq)
    coeff = -coeff;
  ineq.back() += denom - 1;
  addInequality(ineq);
}

void SimplexBase::appendVariable(unsigned count) {
  if (count == 0)
    return;
  var.reserve(var.size() + count);
  colUnknown.reserve(colUnknown.size() + count);
  for (unsigned i = 0; i < count; ++i) {
    var.emplace_back(Orientation::Column, /*restricted=*/false,
                     /*pos=*/getNumColumns() + i);
    colUnknown.emplace_back(var.size() - 1);
  }
  tableau.resizeHorizontally(getNumColumns() + count);
  undoLog.insert(undoLog.end(), count, UndoLogEntry::RemoveLastVariable);
}

/// Add all the constraints from the given IntegerRelation.
void SimplexBase::intersectIntegerRelation(const IntegerRelation &rel) {
  assert(rel.getNumVars() == getNumVariables() &&
         "IntegerRelation must have same dimensionality as simplex");
  for (unsigned i = 0, e = rel.getNumInequalities(); i < e; ++i)
    addInequality(rel.getInequality(i));
  for (unsigned i = 0, e = rel.getNumEqualities(); i < e; ++i)
    addEquality(rel.getEquality(i));
}

MaybeOptimum<Fraction> Simplex::computeRowOptimum(Direction direction,
                                                  unsigned row) {
  // Keep trying to find a pivot for the row in the specified direction.
  while (std::optional<Pivot> maybePivot = findPivot(row, direction)) {
    // If findPivot returns a pivot involving the row itself, then the optimum
    // is unbounded, so we return std::nullopt.
    if (maybePivot->row == row)
      return OptimumKind::Unbounded;
    pivot(*maybePivot);
  }

  // The row has reached its optimal sample value, which we return.
  // The sample value is the entry in the constant column divided by the common
  // denominator for this row.
  return Fraction(tableau(row, 1), tableau(row, 0));
}

/// Compute the optimum of the specified expression in the specified direction,
/// or std::nullopt if it is unbounded.
MaybeOptimum<Fraction> Simplex::computeOptimum(Direction direction,
                                               ArrayRef<DynamicAPInt> coeffs) {
  if (empty)
    return OptimumKind::Empty;

  SimplexRollbackScopeExit scopeExit(*this);
  unsigned conIndex = addRow(coeffs);
  unsigned row = con[conIndex].pos;
  return computeRowOptimum(direction, row);
}

MaybeOptimum<Fraction> Simplex::computeOptimum(Direction direction,
                                               Unknown &u) {
  if (empty)
    return OptimumKind::Empty;
  if (u.orientation == Orientation::Column) {
    unsigned column = u.pos;
    std::optional<unsigned> pivotRow = findPivotRow({}, direction, column);
    // If no pivot is returned, the constraint is unbounded in the specified
    // direction.
    if (!pivotRow)
      return OptimumKind::Unbounded;
    pivot(*pivotRow, column);
  }

  unsigned row = u.pos;
  MaybeOptimum<Fraction> optimum = computeRowOptimum(direction, row);
  if (u.restricted && direction == Direction::Down &&
      (optimum.isUnbounded() || *optimum < Fraction(0, 1))) {
    if (restoreRow(u).failed())
      llvm_unreachable("Could not restore row!");
  }
  return optimum;
}

bool Simplex::isBoundedAlongConstraint(unsigned constraintIndex) {
  assert(!empty && "It is not meaningful to ask whether a direction is bounded "
                   "in an empty set.");
  // The constraint's perpendicular is already bounded below, since it is a
  // constraint. If it is also bounded above, we can return true.
  return computeOptimum(Direction::Up, con[constraintIndex]).isBounded();
}

/// Redundant constraints are those that are in row orientation and lie in
/// rows 0 to nRedundant - 1.
bool Simplex::isMarkedRedundant(unsigned constraintIndex) const {
  const Unknown &u = con[constraintIndex];
  return u.orientation == Orientation::Row && u.pos < nRedundant;
}

/// Mark the specified row redundant.
///
/// This is done by moving the unknown to the end of the block of redundant
/// rows (namely, to row nRedundant) and incrementing nRedundant to
/// accomodate the new redundant row.
void Simplex::markRowRedundant(Unknown &u) {
  assert(u.orientation == Orientation::Row &&
         "Unknown should be in row position!");
  assert(u.pos >= nRedundant && "Unknown is already marked redundant!");
  swapRows(u.pos, nRedundant);
  ++nRedundant;
  undoLog.emplace_back(UndoLogEntry::UnmarkLastRedundant);
}

/// Find a subset of constraints that is redundant and mark them redundant.
void Simplex::detectRedundant(unsigned offset, unsigned count) {
  assert(offset + count <= con.size() && "invalid range!");
  // It is not meaningful to talk about redundancy for empty sets.
  if (empty)
    return;

  // Iterate through the constraints and check for each one if it can attain
  // negative sample values. If it can, it's not redundant. Otherwise, it is.
  // We mark redundant constraints redundant.
  //
  // Constraints that get marked redundant in one iteration are not respected
  // when checking constraints in later iterations. This prevents, for example,
  // two identical constraints both being marked redundant since each is
  // redundant given the other one. In this example, only the first of the
  // constraints that is processed will get marked redundant, as it should be.
  for (unsigned i = 0; i < count; ++i) {
    Unknown &u = con[offset + i];
    if (u.orientation == Orientation::Column) {
      unsigned column = u.pos;
      std::optional<unsigned> pivotRow =
          findPivotRow({}, Direction::Down, column);
      // If no downward pivot is returned, the constraint is unbounded below
      // and hence not redundant.
      if (!pivotRow)
        continue;
      pivot(*pivotRow, column);
    }

    unsigned row = u.pos;
    MaybeOptimum<Fraction> minimum = computeRowOptimum(Direction::Down, row);
    if (minimum.isUnbounded() || *minimum < Fraction(0, 1)) {
      // Constraint is unbounded below or can attain negative sample values and
      // hence is not redundant.
      if (restoreRow(u).failed())
        llvm_unreachable("Could not restore non-redundant row!");
      continue;
    }

    markRowRedundant(u);
  }
}

bool Simplex::isUnbounded() {
  if (empty)
    return false;

  SmallVector<DynamicAPInt, 8> dir(var.size() + 1);
  for (unsigned i = 0; i < var.size(); ++i) {
    dir[i] = 1;

    if (computeOptimum(Direction::Up, dir).isUnbounded())
      return true;

    if (computeOptimum(Direction::Down, dir).isUnbounded())
      return true;

    dir[i] = 0;
  }
  return false;
}

/// Make a tableau to represent a pair of points in the original tableau.
///
/// The product constraints and variables are stored as: first A's, then B's.
///
/// The product tableau has row layout:
///   A's redundant rows, B's redundant rows, A's other rows, B's other rows.
///
/// It has column layout:
///   denominator, constant, A's columns, B's columns.
Simplex Simplex::makeProduct(const Simplex &a, const Simplex &b) {
  unsigned numVar = a.getNumVariables() + b.getNumVariables();
  unsigned numCon = a.getNumConstraints() + b.getNumConstraints();
  Simplex result(numVar);

  result.tableau.reserveRows(numCon);
  result.empty = a.empty || b.empty;

  auto concat = [](ArrayRef<Unknown> v, ArrayRef<Unknown> w) {
    SmallVector<Unknown, 8> result;
    result.reserve(v.size() + w.size());
    result.insert(result.end(), v.begin(), v.end());
    result.insert(result.end(), w.begin(), w.end());
    return result;
  };
  result.con = concat(a.con, b.con);
  result.var = concat(a.var, b.var);

  auto indexFromBIndex = [&](int index) {
    return index >= 0 ? a.getNumVariables() + index
                      : ~(a.getNumConstraints() + ~index);
  };

  result.colUnknown.assign(2, nullIndex);
  for (unsigned i = 2, e = a.getNumColumns(); i < e; ++i) {
    result.colUnknown.emplace_back(a.colUnknown[i]);
    result.unknownFromIndex(result.colUnknown.back()).pos =
        result.colUnknown.size() - 1;
  }
  for (unsigned i = 2, e = b.getNumColumns(); i < e; ++i) {
    result.colUnknown.emplace_back(indexFromBIndex(b.colUnknown[i]));
    result.unknownFromIndex(result.colUnknown.back()).pos =
        result.colUnknown.size() - 1;
  }

  auto appendRowFromA = [&](unsigned row) {
    unsigned resultRow = result.tableau.appendExtraRow();
    for (unsigned col = 0, e = a.getNumColumns(); col < e; ++col)
      result.tableau(resultRow, col) = a.tableau(row, col);
    result.rowUnknown.emplace_back(a.rowUnknown[row]);
    result.unknownFromIndex(result.rowUnknown.back()).pos =
        result.rowUnknown.size() - 1;
  };

  // Also fixes the corresponding entry in rowUnknown and var/con (as the case
  // may be).
  auto appendRowFromB = [&](unsigned row) {
    unsigned resultRow = result.tableau.appendExtraRow();
    result.tableau(resultRow, 0) = b.tableau(row, 0);
    result.tableau(resultRow, 1) = b.tableau(row, 1);

    unsigned offset = a.getNumColumns() - 2;
    for (unsigned col = 2, e = b.getNumColumns(); col < e; ++col)
      result.tableau(resultRow, offset + col) = b.tableau(row, col);
    result.rowUnknown.emplace_back(indexFromBIndex(b.rowUnknown[row]));
    result.unknownFromIndex(result.rowUnknown.back()).pos =
        result.rowUnknown.size() - 1;
  };

  result.nRedundant = a.nRedundant + b.nRedundant;
  for (unsigned row = 0; row < a.nRedundant; ++row)
    appendRowFromA(row);
  for (unsigned row = 0; row < b.nRedundant; ++row)
    appendRowFromB(row);
  for (unsigned row = a.nRedundant, e = a.getNumRows(); row < e; ++row)
    appendRowFromA(row);
  for (unsigned row = b.nRedundant, e = b.getNumRows(); row < e; ++row)
    appendRowFromB(row);

  return result;
}

std::optional<SmallVector<Fraction, 8>> Simplex::getRationalSample() const {
  if (empty)
    return {};

  SmallVector<Fraction, 8> sample;
  sample.reserve(var.size());
  // Push the sample value for each variable into the vector.
  for (const Unknown &u : var) {
    if (u.orientation == Orientation::Column) {
      // If the variable is in column position, its sample value is zero.
      sample.emplace_back(0, 1);
    } else {
      // If the variable is in row position, its sample value is the
      // entry in the constant column divided by the denominator.
      DynamicAPInt denom = tableau(u.pos, 0);
      sample.emplace_back(tableau(u.pos, 1), denom);
    }
  }
  return sample;
}

void LexSimplexBase::addInequality(ArrayRef<DynamicAPInt> coeffs) {
  addRow(coeffs, /*makeRestricted=*/true);
}

MaybeOptimum<SmallVector<Fraction, 8>> LexSimplex::getRationalSample() const {
  if (empty)
    return OptimumKind::Empty;

  SmallVector<Fraction, 8> sample;
  sample.reserve(var.size());
  // Push the sample value for each variable into the vector.
  for (const Unknown &u : var) {
    // When the big M parameter is being used, each variable x is represented
    // as M + x, so its sample value is finite if and only if it is of the
    // form 1*M + c. If the coefficient of M is not one then the sample value
    // is infinite, and we return an empty optional.

    if (u.orientation == Orientation::Column) {
      // If the variable is in column position, the sample value of M + x is
      // zero, so x = -M which is unbounded.
      return OptimumKind::Unbounded;
    }

    // If the variable is in row position, its sample value is the
    // entry in the constant column divided by the denominator.
    DynamicAPInt denom = tableau(u.pos, 0);
    if (usingBigM)
      if (tableau(u.pos, 2) != denom)
        return OptimumKind::Unbounded;
    sample.emplace_back(tableau(u.pos, 1), denom);
  }
  return sample;
}

std::optional<SmallVector<DynamicAPInt, 8>>
Simplex::getSamplePointIfIntegral() const {
  // If the tableau is empty, no sample point exists.
  if (empty)
    return {};

  // The value will always exist since the Simplex is non-empty.
  SmallVector<Fraction, 8> rationalSample = *getRationalSample();
  SmallVector<DynamicAPInt, 8> integerSample;
  integerSample.reserve(var.size());
  for (const Fraction &coord : rationalSample) {
    // If the sample is non-integral, return std::nullopt.
    if (coord.num % coord.den != 0)
      return {};
    integerSample.emplace_back(coord.num / coord.den);
  }
  return integerSample;
}

/// Given a simplex for a polytope, construct a new simplex whose variables are
/// identified with a pair of points (x, y) in the original polytope. Supports
/// some operations needed for generalized basis reduction. In what follows,
/// dotProduct(x, y) = x_1 * y_1 + x_2 * y_2 + ... x_n * y_n where n is the
/// dimension of the original polytope.
///
/// This supports adding equality constraints dotProduct(dir, x - y) == 0. It
/// also supports rolling back this addition, by maintaining a snapshot stack
/// that contains a snapshot of the Simplex's state for each equality, just
/// before that equality was added.
class presburger::GBRSimplex {
  using Orientation = Simplex::Orientation;

public:
  GBRSimplex(const Simplex &originalSimplex)
      : simplex(Simplex::makeProduct(originalSimplex, originalSimplex)),
        simplexConstraintOffset(simplex.getNumConstraints()) {}

  /// Add an equality dotProduct(dir, x - y) == 0.
  /// First pushes a snapshot for the current simplex state to the stack so
  /// that this can be rolled back later.
  void addEqualityForDirection(ArrayRef<DynamicAPInt> dir) {
    assert(llvm::any_of(dir, [](const DynamicAPInt &x) { return x != 0; }) &&
           "Direction passed is the zero vector!");
    snapshotStack.emplace_back(simplex.getSnapshot());
    simplex.addEquality(getCoeffsForDirection(dir));
  }
  /// Compute max(dotProduct(dir, x - y)).
  Fraction computeWidth(ArrayRef<DynamicAPInt> dir) {
    MaybeOptimum<Fraction> maybeWidth =
        simplex.computeOptimum(Direction::Up, getCoeffsForDirection(dir));
    assert(maybeWidth.isBounded() && "Width should be bounded!");
    return *maybeWidth;
  }

  /// Compute max(dotProduct(dir, x - y)) and save the dual variables for only
  /// the direction equalities to `dual`.
  Fraction computeWidthAndDuals(ArrayRef<DynamicAPInt> dir,
                                SmallVectorImpl<DynamicAPInt> &dual,
                                DynamicAPInt &dualDenom) {
    // We can't just call into computeWidth or computeOptimum since we need to
    // access the state of the tableau after computing the optimum, and these
    // functions rollback the insertion of the objective function into the
    // tableau before returning. We instead add a row for the objective function
    // ourselves, call into computeOptimum, compute the duals from the tableau
    // state, and finally rollback the addition of the row before returning.
    SimplexRollbackScopeExit scopeExit(simplex);
    unsigned conIndex = simplex.addRow(getCoeffsForDirection(dir));
    unsigned row = simplex.con[conIndex].pos;
    MaybeOptimum<Fraction> maybeWidth =
        simplex.computeRowOptimum(Simplex::Direction::Up, row);
    assert(maybeWidth.isBounded() && "Width should be bounded!");
    dualDenom = simplex.tableau(row, 0);
    dual.clear();
    dual.reserve((conIndex - simplexConstraintOffset) / 2);

    // The increment is i += 2 because equalities are added as two inequalities,
    // one positive and one negative. Each iteration processes one equality.
    for (unsigned i = simplexConstraintOffset; i < conIndex; i += 2) {
      // The dual variable for an inequality in column orientation is the
      // negative of its coefficient at the objective row. If the inequality is
      // in row orientation, the corresponding dual variable is zero.
      //
      // We want the dual for the original equality, which corresponds to two
      // inequalities: a positive inequality, which has the same coefficients as
      // the equality, and a negative equality, which has negated coefficients.
      //
      // Note that at most one of these inequalities can be in column
      // orientation because the column unknowns should form a basis and hence
      // must be linearly independent. If the positive inequality is in column
      // position, its dual is the dual corresponding to the equality. If the
      // negative inequality is in column position, the negation of its dual is
      // the dual corresponding to the equality. If neither is in column
      // position, then that means that this equality is redundant, and its dual
      // is zero.
      //
      // Note that it is NOT valid to perform pivots during the computation of
      // the duals. This entire dual computation must be performed on the same
      // tableau configuration.
      assert((simplex.con[i].orientation != Orientation::Column ||
              simplex.con[i + 1].orientation != Orientation::Column) &&
             "Both inequalities for the equality cannot be in column "
             "orientation!");
      if (simplex.con[i].orientation == Orientation::Column)
        dual.emplace_back(-simplex.tableau(row, simplex.con[i].pos));
      else if (simplex.con[i + 1].orientation == Orientation::Column)
        dual.emplace_back(simplex.tableau(row, simplex.con[i + 1].pos));
      else
        dual.emplace_back(0);
    }
    return *maybeWidth;
  }

  /// Remove the last equality that was added through addEqualityForDirection.
  ///
  /// We do this by rolling back to the snapshot at the top of the stack, which
  /// should be a snapshot taken just before the last equality was added.
  void removeLastEquality() {
    assert(!snapshotStack.empty() && "Snapshot stack is empty!");
    simplex.rollback(snapshotStack.back());
    snapshotStack.pop_back();
  }

private:
  /// Returns coefficients of the expression 'dot_product(dir, x - y)',
  /// i.e.,   dir_1 * x_1 + dir_2 * x_2 + ... + dir_n * x_n
  ///       - dir_1 * y_1 - dir_2 * y_2 - ... - dir_n * y_n,
  /// where n is the dimension of the original polytope.
  SmallVector<DynamicAPInt, 8>
  getCoeffsForDirection(ArrayRef<DynamicAPInt> dir) {
    assert(2 * dir.size() == simplex.getNumVariables() &&
           "Direction vector has wrong dimensionality");
    SmallVector<DynamicAPInt, 8> coeffs(dir);
    coeffs.reserve(dir.size() + 1);
    for (const DynamicAPInt &coeff : dir)
      coeffs.emplace_back(-coeff);
    coeffs.emplace_back(0); // constant term
    return coeffs;
  }

  Simplex simplex;
  /// The first index of the equality constraints, the index immediately after
  /// the last constraint in the initial product simplex.
  unsigned simplexConstraintOffset;
  /// A stack of snapshots, used for rolling back.
  SmallVector<unsigned, 8> snapshotStack;
};

/// Reduce the basis to try and find a direction in which the polytope is
/// "thin". This only works for bounded polytopes.
///
/// This is an implementation of the algorithm described in the paper
/// "An Implementation of Generalized Basis Reduction for Integer Programming"
/// by W. Cook, T. Rutherford, H. E. Scarf, D. Shallcross.
///
/// Let b_{level}, b_{level + 1}, ... b_n be the current basis.
/// Let width_i(v) = max <v, x - y> where x and y are points in the original
/// polytope such that <b_j, x - y> = 0 is satisfied for all level <= j < i.
///
/// In every iteration, we first replace b_{i+1} with b_{i+1} + u*b_i, where u
/// is the integer such that width_i(b_{i+1} + u*b_i) is minimized. Let dual_i
/// be the dual variable associated with the constraint <b_i, x - y> = 0 when
/// computing width_{i+1}(b_{i+1}). It can be shown that dual_i is the
/// minimizing value of u, if it were allowed to be fractional. Due to
/// convexity, the minimizing integer value is either floor(dual_i) or
/// ceil(dual_i), so we just need to check which of these gives a lower
/// width_{i+1} value. If dual_i turned out to be an integer, then u = dual_i.
///
/// Now if width_i(b_{i+1}) < 0.75 * width_i(b_i), we swap b_i and (the new)
/// b_{i + 1} and decrement i (unless i = level, in which case we stay at the
/// same i). Otherwise, we increment i.
///
/// We keep f values and duals cached and invalidate them when necessary.
/// Whenever possible, we use them instead of recomputing them. We implement the
/// algorithm as follows.
///
/// In an iteration at i we need to compute:
///   a) width_i(b_{i + 1})
///   b) width_i(b_i)
///   c) the integer u that minimizes width_i(b_{i + 1} + u*b_i)
///
/// If width_i(b_i) is not already cached, we compute it.
///
/// If the duals are not already cached, we compute width_{i+1}(b_{i+1}) and
/// store the duals from this computation.
///
/// We call updateBasisWithUAndGetFCandidate, which finds the minimizing value
/// of u as explained before, caches the duals from this computation, sets
/// b_{i+1} to b_{i+1} + u*b_i, and returns the new value of width_i(b_{i+1}).
///
/// Now if width_i(b_{i+1}) < 0.75 * width_i(b_i), we swap b_i and b_{i+1} and
/// decrement i, resulting in the basis
/// ... b_{i - 1}, b_{i + 1} + u*b_i, b_i, b_{i+2}, ...
/// with corresponding f values
/// ... width_{i-1}(b_{i-1}), width_i(b_{i+1} + u*b_i), width_{i+1}(b_i), ...
/// The values up to i - 1 remain unchanged. We have just gotten the middle
/// value from updateBasisWithUAndGetFCandidate, so we can update that in the
/// cache. The value at width_{i+1}(b_i) is unknown, so we evict this value from
/// the cache. The iteration after decrementing needs exactly the duals from the
/// computation of width_i(b_{i + 1} + u*b_i), so we keep these in the cache.
///
/// When incrementing i, no cached f values get invalidated. However, the cached
/// duals do get invalidated as the duals for the higher levels are different.
void Simplex::reduceBasis(IntMatrix &basis, unsigned level) {
  const Fraction epsilon(3, 4);

  if (level == basis.getNumRows() - 1)
    return;

  GBRSimplex gbrSimplex(*this);
  SmallVector<Fraction, 8> width;
  SmallVector<DynamicAPInt, 8> dual;
  DynamicAPInt dualDenom;

  // Finds the value of u that minimizes width_i(b_{i+1} + u*b_i), caches the
  // duals from this computation, sets b_{i+1} to b_{i+1} + u*b_i, and returns
  // the new value of width_i(b_{i+1}).
  //
  // If dual_i is not an integer, the minimizing value must be either
  // floor(dual_i) or ceil(dual_i). We compute the expression for both and
  // choose the minimizing value.
  //
  // If dual_i is an integer, we don't need to perform these computations. We
  // know that in this case,
  //   a) u = dual_i.
  //   b) one can show that dual_j for j < i are the same duals we would have
  //      gotten from computing width_i(b_{i + 1} + u*b_i), so the correct duals
  //      are the ones already in the cache.
  //   c) width_i(b_{i+1} + u*b_i) = min_{alpha} width_i(b_{i+1} + alpha * b_i),
  //   which
  //      one can show is equal to width_{i+1}(b_{i+1}). The latter value must
  //      be in the cache, so we get it from there and return it.
  auto updateBasisWithUAndGetFCandidate = [&](unsigned i) -> Fraction {
    assert(i < level + dual.size() && "dual_i is not known!");

    DynamicAPInt u = floorDiv(dual[i - level], dualDenom);
    basis.addToRow(i, i + 1, u);
    if (dual[i - level] % dualDenom != 0) {
      SmallVector<DynamicAPInt, 8> candidateDual[2];
      DynamicAPInt candidateDualDenom[2];
      Fraction widthI[2];

      // Initially u is floor(dual) and basis reflects this.
      widthI[0] = gbrSimplex.computeWidthAndDuals(
          basis.getRow(i + 1), candidateDual[0], candidateDualDenom[0]);

      // Now try ceil(dual), i.e. floor(dual) + 1.
      ++u;
      basis.addToRow(i, i + 1, 1);
      widthI[1] = gbrSimplex.computeWidthAndDuals(
          basis.getRow(i + 1), candidateDual[1], candidateDualDenom[1]);

      unsigned j = widthI[0] < widthI[1] ? 0 : 1;
      if (j == 0)
        // Subtract 1 to go from u = ceil(dual) back to floor(dual).
        basis.addToRow(i, i + 1, -1);

      // width_i(b{i+1} + u*b_i) should be minimized at our value of u.
      // We assert that this holds by checking that the values of width_i at
      // u - 1 and u + 1 are greater than or equal to the value at u. If the
      // width is lesser at either of the adjacent values, then our computed
      // value of u is clearly not the minimizer. Otherwise by convexity the
      // computed value of u is really the minimizer.

      // Check the value at u - 1.
      assert(gbrSimplex.computeWidth(scaleAndAddForAssert(
                 basis.getRow(i + 1), DynamicAPInt(-1), basis.getRow(i))) >=
                 widthI[j] &&
             "Computed u value does not minimize the width!");
      // Check the value at u + 1.
      assert(gbrSimplex.computeWidth(scaleAndAddForAssert(
                 basis.getRow(i + 1), DynamicAPInt(+1), basis.getRow(i))) >=
                 widthI[j] &&
             "Computed u value does not minimize the width!");

      dual = std::move(candidateDual[j]);
      dualDenom = candidateDualDenom[j];
      return widthI[j];
    }

    assert(i + 1 - level < width.size() && "width_{i+1} wasn't saved");
    // f_i(b_{i+1} + dual*b_i) == width_{i+1}(b_{i+1}) when `dual` minimizes the
    // LHS. (note: the basis has already been updated, so b_{i+1} + dual*b_i in
    // the above expression is equal to basis.getRow(i+1) below.)
    assert(gbrSimplex.computeWidth(basis.getRow(i + 1)) ==
           width[i + 1 - level]);
    return width[i + 1 - level];
  };

  // In the ith iteration of the loop, gbrSimplex has constraints for directions
  // from `level` to i - 1.
  unsigned i = level;
  while (i < basis.getNumRows() - 1) {
    if (i >= level + width.size()) {
      // We don't even know the value of f_i(b_i), so let's find that first.
      // We have to do this first since later we assume that width already
      // contains values up to and including i.

      assert((i == 0 || i - 1 < level + width.size()) &&
             "We are at level i but we don't know the value of width_{i-1}");

      // We don't actually use these duals at all, but it doesn't matter
      // because this case should only occur when i is level, and there are no
      // duals in that case anyway.
      assert(i == level && "This case should only occur when i == level");
      width.emplace_back(
          gbrSimplex.computeWidthAndDuals(basis.getRow(i), dual, dualDenom));
    }

    if (i >= level + dual.size()) {
      assert(i + 1 >= level + width.size() &&
             "We don't know dual_i but we know width_{i+1}");
      // We don't know dual for our level, so let's find it.
      gbrSimplex.addEqualityForDirection(basis.getRow(i));
      width.emplace_back(gbrSimplex.computeWidthAndDuals(basis.getRow(i + 1),
                                                         dual, dualDenom));
      gbrSimplex.removeLastEquality();
    }

    // This variable stores width_i(b_{i+1} + u*b_i).
    Fraction widthICandidate = updateBasisWithUAndGetFCandidate(i);
    if (widthICandidate < epsilon * width[i - level]) {
      basis.swapRows(i, i + 1);
      width[i - level] = widthICandidate;
      // The values of width_{i+1}(b_{i+1}) and higher may change after the
      // swap, so we remove the cached values here.
      width.resize(i - level + 1);
      if (i == level) {
        dual.clear();
        continue;
      }

      gbrSimplex.removeLastEquality();
      i--;
      continue;
    }

    // Invalidate duals since the higher level needs to recompute its own duals.
    dual.clear();
    gbrSimplex.addEqualityForDirection(basis.getRow(i));
    i++;
  }
}

/// Search for an integer sample point using a branch and bound algorithm.
///
/// Each row in the basis matrix is a vector, and the set of basis vectors
/// should span the space. Initially this is the identity matrix,
/// i.e., the basis vectors are just the variables.
///
/// In every level, a value is assigned to the level-th basis vector, as
/// follows. Compute the minimum and maximum rational values of this direction.
/// If only one integer point lies in this range, constrain the variable to
/// have this value and recurse to the next variable.
///
/// If the range has multiple values, perform generalized basis reduction via
/// reduceBasis and then compute the bounds again. Now we try constraining
/// this direction in the first value in this range and "recurse" to the next
/// level. If we fail to find a sample, we try assigning the direction the next
/// value in this range, and so on.
///
/// If no integer sample is found from any of the assignments, or if the range
/// contains no integer value, then of course the polytope is empty for the
/// current assignment of the values in previous levels, so we return to
/// the previous level.
///
/// If we reach the last level where all the variables have been assigned values
/// already, then we simply return the current sample point if it is integral,
/// and go back to the previous level otherwise.
///
/// To avoid potentially arbitrarily large recursion depths leading to stack
/// overflows, this algorithm is implemented iteratively.
std::optional<SmallVector<DynamicAPInt, 8>> Simplex::findIntegerSample() {
  if (empty)
    return {};

  unsigned nDims = var.size();
  IntMatrix basis = IntMatrix::identity(nDims);

  unsigned level = 0;
  // The snapshot just before constraining a direction to a value at each level.
  SmallVector<unsigned, 8> snapshotStack;
  // The maximum value in the range of the direction for each level.
  SmallVector<DynamicAPInt, 8> upperBoundStack;
  // The next value to try constraining the basis vector to at each level.
  SmallVector<DynamicAPInt, 8> nextValueStack;

  snapshotStack.reserve(basis.getNumRows());
  upperBoundStack.reserve(basis.getNumRows());
  nextValueStack.reserve(basis.getNumRows());
  while (level != -1u) {
    if (level == basis.getNumRows()) {
      // We've assigned values to all variables. Return if we have a sample,
      // or go back up to the previous level otherwise.
      if (auto maybeSample = getSamplePointIfIntegral())
        return maybeSample;
      level--;
      continue;
    }

    if (level >= upperBoundStack.size()) {
      // We haven't populated the stack values for this level yet, so we have
      // just come down a level ("recursed"). Find the lower and upper bounds.
      // If there is more than one integer point in the range, perform
      // generalized basis reduction.
      SmallVector<DynamicAPInt, 8> basisCoeffs =
          llvm::to_vector<8>(basis.getRow(level));
      basisCoeffs.emplace_back(0);

      auto [minRoundedUp, maxRoundedDown] = computeIntegerBounds(basisCoeffs);

      // We don't have any integer values in the range.
      // Pop the stack and return up a level.
      if (minRoundedUp.isEmpty() || maxRoundedDown.isEmpty()) {
        assert((minRoundedUp.isEmpty() && maxRoundedDown.isEmpty()) &&
               "If one bound is empty, both should be.");
        snapshotStack.pop_back();
        nextValueStack.pop_back();
        upperBoundStack.pop_back();
        level--;
        continue;
      }

      // We already checked the empty case above.
      assert((minRoundedUp.isBounded() && maxRoundedDown.isBounded()) &&
             "Polyhedron should be bounded!");

      // Heuristic: if the sample point is integral at this point, just return
      // it.
      if (auto maybeSample = getSamplePointIfIntegral())
        return *maybeSample;

      if (*minRoundedUp < *maxRoundedDown) {
        reduceBasis(basis, level);
        basisCoeffs = llvm::to_vector<8>(basis.getRow(level));
        basisCoeffs.emplace_back(0);
        std::tie(minRoundedUp, maxRoundedDown) =
            computeIntegerBounds(basisCoeffs);
      }

      snapshotStack.emplace_back(getSnapshot());
      // The smallest value in the range is the next value to try.
      // The values in the optionals are guaranteed to exist since we know the
      // polytope is bounded.
      nextValueStack.emplace_back(*minRoundedUp);
      upperBoundStack.emplace_back(*maxRoundedDown);
    }

    assert((snapshotStack.size() - 1 == level &&
            nextValueStack.size() - 1 == level &&
            upperBoundStack.size() - 1 == level) &&
           "Mismatched variable stack sizes!");

    // Whether we "recursed" or "returned" from a lower level, we rollback
    // to the snapshot of the starting state at this level. (in the "recursed"
    // case this has no effect)
    rollback(snapshotStack.back());
    DynamicAPInt nextValue = nextValueStack.back();
    ++nextValueStack.back();
    if (nextValue > upperBoundStack.back()) {
      // We have exhausted the range and found no solution. Pop the stack and
      // return up a level.
      snapshotStack.pop_back();
      nextValueStack.pop_back();
      upperBoundStack.pop_back();
      level--;
      continue;
    }

    // Try the next value in the range and "recurse" into the next level.
    SmallVector<DynamicAPInt, 8> basisCoeffs(basis.getRow(level).begin(),
                                             basis.getRow(level).end());
    basisCoeffs.emplace_back(-nextValue);
    addEquality(basisCoeffs);
    level++;
  }

  return {};
}

/// Compute the minimum and maximum integer values the expression can take. We
/// compute each separately.
std::pair<MaybeOptimum<DynamicAPInt>, MaybeOptimum<DynamicAPInt>>
Simplex::computeIntegerBounds(ArrayRef<DynamicAPInt> coeffs) {
  MaybeOptimum<DynamicAPInt> minRoundedUp(
      computeOptimum(Simplex::Direction::Down, coeffs).map(ceil));
  MaybeOptimum<DynamicAPInt> maxRoundedDown(
      computeOptimum(Simplex::Direction::Up, coeffs).map(floor));
  return {minRoundedUp, maxRoundedDown};
}

bool Simplex::isFlatAlong(ArrayRef<DynamicAPInt> coeffs) {
  assert(!isEmpty() && "cannot check for flatness of empty simplex!");
  auto upOpt = computeOptimum(Simplex::Direction::Up, coeffs);
  auto downOpt = computeOptimum(Simplex::Direction::Down, coeffs);

  if (!upOpt.isBounded())
    return false;
  if (!downOpt.isBounded())
    return false;

  return *upOpt == *downOpt;
}

void SimplexBase::print(raw_ostream &os) const {
  os << "rows = " << getNumRows() << ", columns = " << getNumColumns() << "\n";
  if (empty)
    os << "Simplex marked empty!\n";
  os << "var: ";
  for (unsigned i = 0; i < var.size(); ++i) {
    if (i > 0)
      os << ", ";
    var[i].print(os);
  }
  os << "\ncon: ";
  for (unsigned i = 0; i < con.size(); ++i) {
    if (i > 0)
      os << ", ";
    con[i].print(os);
  }
  os << '\n';
  for (unsigned row = 0, e = getNumRows(); row < e; ++row) {
    if (row > 0)
      os << ", ";
    os << "r" << row << ": " << rowUnknown[row];
  }
  os << '\n';
  os << "c0: denom, c1: const";
  for (unsigned col = 2, e = getNumColumns(); col < e; ++col)
    os << ", c" << col << ": " << colUnknown[col];
  os << '\n';
  PrintTableMetrics ptm = {0, 0, "-"};
  for (unsigned row = 0, numRows = getNumRows(); row < numRows; ++row)
    for (unsigned col = 0, numCols = getNumColumns(); col < numCols; ++col)
      updatePrintMetrics<DynamicAPInt>(tableau(row, col), ptm);
  unsigned MIN_SPACING = 1;
  for (unsigned row = 0, numRows = getNumRows(); row < numRows; ++row) {
    for (unsigned col = 0, numCols = getNumColumns(); col < numCols; ++col) {
      printWithPrintMetrics<DynamicAPInt>(os, tableau(row, col), MIN_SPACING,
                                          ptm);
    }
    os << '\n';
  }
  os << '\n';
}

void SimplexBase::dump() const { print(llvm::errs()); }

bool Simplex::isRationalSubsetOf(const IntegerRelation &rel) {
  if (isEmpty())
    return true;

  for (unsigned i = 0, e = rel.getNumInequalities(); i < e; ++i)
    if (findIneqType(rel.getInequality(i)) != IneqType::Redundant)
      return false;

  for (unsigned i = 0, e = rel.getNumEqualities(); i < e; ++i)
    if (!isRedundantEquality(rel.getEquality(i)))
      return false;

  return true;
}

/// Returns the type of the inequality with coefficients `coeffs`.
/// Possible types are:
/// Redundant   The inequality is satisfied by all points in the polytope
/// Cut         The inequality is satisfied by some points, but not by others
/// Separate    The inequality is not satisfied by any point
///
/// Internally, this computes the minimum and the maximum the inequality with
/// coefficients `coeffs` can take. If the minimum is >= 0, the inequality holds
/// for all points in the polytope, so it is redundant.  If the minimum is <= 0
/// and the maximum is >= 0, the points in between the minimum and the
/// inequality do not satisfy it, the points in between the inequality and the
/// maximum satisfy it. Hence, it is a cut inequality. If both are < 0, no
/// points of the polytope satisfy the inequality, which means it is a separate
/// inequality.
Simplex::IneqType Simplex::findIneqType(ArrayRef<DynamicAPInt> coeffs) {
  MaybeOptimum<Fraction> minimum = computeOptimum(Direction::Down, coeffs);
  if (minimum.isBounded() && *minimum >= Fraction(0, 1)) {
    return IneqType::Redundant;
  }
  MaybeOptimum<Fraction> maximum = computeOptimum(Direction::Up, coeffs);
  if ((!minimum.isBounded() || *minimum <= Fraction(0, 1)) &&
      (!maximum.isBounded() || *maximum >= Fraction(0, 1))) {
    return IneqType::Cut;
  }
  return IneqType::Separate;
}

/// Checks whether the type of the inequality with coefficients `coeffs`
/// is Redundant.
bool Simplex::isRedundantInequality(ArrayRef<DynamicAPInt> coeffs) {
  assert(!empty &&
         "It is not meaningful to ask about redundancy in an empty set!");
  return findIneqType(coeffs) == IneqType::Redundant;
}

/// Check whether the equality given by `coeffs == 0` is redundant given
/// the existing constraints. This is redundant when `coeffs` is already
/// always zero under the existing constraints. `coeffs` is always zero
/// when the minimum and maximum value that `coeffs` can take are both zero.
bool Simplex::isRedundantEquality(ArrayRef<DynamicAPInt> coeffs) {
  assert(!empty &&
         "It is not meaningful to ask about redundancy in an empty set!");
  MaybeOptimum<Fraction> minimum = computeOptimum(Direction::Down, coeffs);
  MaybeOptimum<Fraction> maximum = computeOptimum(Direction::Up, coeffs);
  assert((!minimum.isEmpty() && !maximum.isEmpty()) &&
         "Optima should be non-empty for a non-empty set");
  return minimum.isBounded() && maximum.isBounded() &&
         *maximum == Fraction(0, 1) && *minimum == Fraction(0, 1);
}
