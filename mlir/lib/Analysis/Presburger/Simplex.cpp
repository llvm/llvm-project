//===- Simplex.cpp - MLIR Simplex Class -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/Support/MathExtras.h"

namespace mlir {
using Direction = Simplex::Direction;

const int nullIndex = std::numeric_limits<int>::max();

/// Construct a Simplex object with `nVar` variables.
Simplex::Simplex(unsigned nVar)
    : nRow(0), nCol(2), tableau(0, 2 + nVar), empty(false) {
  colUnknown.push_back(nullIndex);
  colUnknown.push_back(nullIndex);
  for (unsigned i = 0; i < nVar; ++i) {
    var.emplace_back(Orientation::Column, /*restricted=*/false, /*pos=*/nCol);
    colUnknown.push_back(i);
    nCol++;
  }
}

Simplex::Simplex(const FlatAffineConstraints &constraints)
    : Simplex(constraints.getNumIds()) {
  for (unsigned i = 0, numIneqs = constraints.getNumInequalities();
       i < numIneqs; ++i)
    addInequality(constraints.getInequality(i));
  for (unsigned i = 0, numEqs = constraints.getNumEqualities(); i < numEqs; ++i)
    addEquality(constraints.getEquality(i));
}

const Simplex::Unknown &Simplex::unknownFromIndex(int index) const {
  assert(index != nullIndex && "nullIndex passed to unknownFromIndex");
  return index >= 0 ? var[index] : con[~index];
}

const Simplex::Unknown &Simplex::unknownFromColumn(unsigned col) const {
  assert(col < nCol && "Invalid column");
  return unknownFromIndex(colUnknown[col]);
}

const Simplex::Unknown &Simplex::unknownFromRow(unsigned row) const {
  assert(row < nRow && "Invalid row");
  return unknownFromIndex(rowUnknown[row]);
}

Simplex::Unknown &Simplex::unknownFromIndex(int index) {
  assert(index != nullIndex && "nullIndex passed to unknownFromIndex");
  return index >= 0 ? var[index] : con[~index];
}

Simplex::Unknown &Simplex::unknownFromColumn(unsigned col) {
  assert(col < nCol && "Invalid column");
  return unknownFromIndex(colUnknown[col]);
}

Simplex::Unknown &Simplex::unknownFromRow(unsigned row) {
  assert(row < nRow && "Invalid row");
  return unknownFromIndex(rowUnknown[row]);
}

/// Add a new row to the tableau corresponding to the given constant term and
/// list of coefficients. The coefficients are specified as a vector of
/// (variable index, coefficient) pairs.
unsigned Simplex::addRow(ArrayRef<int64_t> coeffs) {
  assert(coeffs.size() == 1 + var.size() &&
         "Incorrect number of coefficients!");

  ++nRow;
  // If the tableau is not big enough to accomodate the extra row, we extend it.
  if (nRow >= tableau.getNumRows())
    tableau.resizeVertically(nRow);
  rowUnknown.push_back(~con.size());
  con.emplace_back(Orientation::Row, false, nRow - 1);

  tableau(nRow - 1, 0) = 1;
  tableau(nRow - 1, 1) = coeffs.back();
  for (unsigned col = 2; col < nCol; ++col)
    tableau(nRow - 1, col) = 0;

  // Process each given variable coefficient.
  for (unsigned i = 0; i < var.size(); ++i) {
    unsigned pos = var[i].pos;
    if (coeffs[i] == 0)
      continue;

    if (var[i].orientation == Orientation::Column) {
      // If a variable is in column position at column col, then we just add the
      // coefficient for that variable (scaled by the common row denominator) to
      // the corresponding entry in the new row.
      tableau(nRow - 1, pos) += coeffs[i] * tableau(nRow - 1, 0);
      continue;
    }

    // If the variable is in row position, we need to add that row to the new
    // row, scaled by the coefficient for the variable, accounting for the two
    // rows potentially having different denominators. The new denominator is
    // the lcm of the two.
    int64_t lcm = mlir::lcm(tableau(nRow - 1, 0), tableau(pos, 0));
    int64_t nRowCoeff = lcm / tableau(nRow - 1, 0);
    int64_t idxRowCoeff = coeffs[i] * (lcm / tableau(pos, 0));
    tableau(nRow - 1, 0) = lcm;
    for (unsigned col = 1; col < nCol; ++col)
      tableau(nRow - 1, col) =
          nRowCoeff * tableau(nRow - 1, col) + idxRowCoeff * tableau(pos, col);
  }

  normalizeRow(nRow - 1);
  // Push to undo log along with the index of the new constraint.
  undoLog.push_back(UndoLogEntry::RemoveLastConstraint);
  return con.size() - 1;
}

/// Normalize the row by removing factors that are common between the
/// denominator and all the numerator coefficients.
void Simplex::normalizeRow(unsigned row) {
  int64_t gcd = 0;
  for (unsigned col = 0; col < nCol; ++col) {
    if (gcd == 1)
      break;
    gcd = llvm::greatestCommonDivisor(gcd, std::abs(tableau(row, col)));
  }
  for (unsigned col = 0; col < nCol; ++col)
    tableau(row, col) /= gcd;
}

namespace {
bool signMatchesDirection(int64_t elem, Direction direction) {
  assert(elem != 0 && "elem should not be 0");
  return direction == Direction::Up ? elem > 0 : elem < 0;
}

Direction flippedDirection(Direction direction) {
  return direction == Direction::Up ? Direction::Down : Simplex::Direction::Up;
}
} // anonymous namespace

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
Optional<Simplex::Pivot> Simplex::findPivot(int row,
                                            Direction direction) const {
  Optional<unsigned> col;
  for (unsigned j = 2; j < nCol; ++j) {
    int64_t elem = tableau(row, j);
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
  Optional<unsigned> maybePivotRow = findPivotRow(row, newDirection, *col);
  return Pivot{maybePivotRow.getValueOr(row), *col};
}

/// Swap the associated unknowns for the row and the column.
///
/// First we swap the index associated with the row and column. Then we update
/// the unknowns to reflect their new position and orientation.
void Simplex::swapRowWithCol(unsigned row, unsigned col) {
  std::swap(rowUnknown[row], colUnknown[col]);
  Unknown &uCol = unknownFromColumn(col);
  Unknown &uRow = unknownFromRow(row);
  uCol.orientation = Orientation::Column;
  uRow.orientation = Orientation::Row;
  uCol.pos = col;
  uRow.pos = row;
}

void Simplex::pivot(Pivot pair) { pivot(pair.row, pair.column); }

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
void Simplex::pivot(unsigned pivotRow, unsigned pivotCol) {
  assert(pivotCol >= 2 && "Refusing to pivot invalid column");

  swapRowWithCol(pivotRow, pivotCol);
  std::swap(tableau(pivotRow, 0), tableau(pivotRow, pivotCol));
  // We need to negate the whole pivot row except for the pivot column.
  if (tableau(pivotRow, 0) < 0) {
    // If the denominator is negative, we negate the row by simply negating the
    // denominator.
    tableau(pivotRow, 0) = -tableau(pivotRow, 0);
    tableau(pivotRow, pivotCol) = -tableau(pivotRow, pivotCol);
  } else {
    for (unsigned col = 1; col < nCol; ++col) {
      if (col == pivotCol)
        continue;
      tableau(pivotRow, col) = -tableau(pivotRow, col);
    }
  }
  normalizeRow(pivotRow);

  for (unsigned row = 0; row < nRow; ++row) {
    if (row == pivotRow)
      continue;
    if (tableau(row, pivotCol) == 0) // Nothing to do.
      continue;
    tableau(row, 0) *= tableau(pivotRow, 0);
    for (unsigned j = 1; j < nCol; ++j) {
      if (j == pivotCol)
        continue;
      // Add rather than subtract because the pivot row has been negated.
      tableau(row, j) = tableau(row, j) * tableau(pivotRow, 0) +
                        tableau(row, pivotCol) * tableau(pivotRow, j);
    }
    tableau(row, pivotCol) *= tableau(pivotRow, pivotCol);
    normalizeRow(row);
  }
}

/// Perform pivots until the unknown has a non-negative sample value or until
/// no more upward pivots can be performed. Return the sign of the final sample
/// value.
LogicalResult Simplex::restoreRow(Unknown &u) {
  assert(u.orientation == Orientation::Row &&
         "unknown should be in row position");

  while (tableau(u.pos, 1) < 0) {
    Optional<Pivot> maybePivot = findPivot(u.pos, Direction::Up);
    if (!maybePivot)
      break;

    pivot(*maybePivot);
    if (u.orientation == Orientation::Column)
      return LogicalResult::Success; // the unknown is unbounded above.
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
Optional<unsigned> Simplex::findPivotRow(Optional<unsigned> skipRow,
                                         Direction direction,
                                         unsigned col) const {
  Optional<unsigned> retRow;
  int64_t retElem, retConst;
  for (unsigned row = 0; row < nRow; ++row) {
    if (skipRow && row == *skipRow)
      continue;
    int64_t elem = tableau(row, col);
    if (elem == 0)
      continue;
    if (!unknownFromRow(row).restricted)
      continue;
    if (signMatchesDirection(elem, direction))
      continue;
    int64_t constTerm = tableau(row, 1);

    if (!retRow) {
      retRow = row;
      retElem = elem;
      retConst = constTerm;
      continue;
    }

    int64_t diff = retConst * elem - constTerm * retElem;
    if ((diff == 0 && rowUnknown[row] < rowUnknown[*retRow]) ||
        (diff != 0 && !signMatchesDirection(diff, direction))) {
      retRow = row;
      retElem = elem;
      retConst = constTerm;
    }
  }
  return retRow;
}

bool Simplex::isEmpty() const { return empty; }

void Simplex::swapRows(unsigned i, unsigned j) {
  if (i == j)
    return;
  tableau.swapRows(i, j);
  std::swap(rowUnknown[i], rowUnknown[j]);
  unknownFromRow(i).pos = i;
  unknownFromRow(j).pos = j;
}

/// Mark this tableau empty and push an entry to the undo stack.
void Simplex::markEmpty() {
  undoLog.push_back(UndoLogEntry::UnmarkEmpty);
  empty = true;
}

/// Add an inequality to the tableau. If coeffs is c_0, c_1, ... c_n, where n
/// is the curent number of variables, then the corresponding inequality is
/// c_n + c_0*x_0 + c_1*x_1 + ... + c_{n-1}*x_{n-1} >= 0.
///
/// We add the inequality and mark it as restricted. We then try to make its
/// sample value non-negative. If this is not possible, the tableau has become
/// empty and we mark it as such.
void Simplex::addInequality(ArrayRef<int64_t> coeffs) {
  unsigned conIndex = addRow(coeffs);
  Unknown &u = con[conIndex];
  u.restricted = true;
  LogicalResult result = restoreRow(u);
  if (failed(result))
    markEmpty();
}

/// Add an equality to the tableau. If coeffs is c_0, c_1, ... c_n, where n
/// is the curent number of variables, then the corresponding equality is
/// c_n + c_0*x_0 + c_1*x_1 + ... + c_{n-1}*x_{n-1} == 0.
///
/// We simply add two opposing inequalities, which force the expression to
/// be zero.
void Simplex::addEquality(ArrayRef<int64_t> coeffs) {
  addInequality(coeffs);
  SmallVector<int64_t, 8> negatedCoeffs;
  for (int64_t coeff : coeffs)
    negatedCoeffs.emplace_back(-coeff);
  addInequality(negatedCoeffs);
}

unsigned Simplex::numVariables() const { return var.size(); }
unsigned Simplex::numConstraints() const { return con.size(); }

/// Return a snapshot of the curent state. This is just the current size of the
/// undo log.
unsigned Simplex::getSnapshot() const { return undoLog.size(); }

void Simplex::undo(UndoLogEntry entry) {
  if (entry == UndoLogEntry::RemoveLastConstraint) {
    Unknown &constraint = con.back();
    if (constraint.orientation == Orientation::Column) {
      unsigned column = constraint.pos;
      Optional<unsigned> row;

      // Try to find any pivot row for this column that preserves tableau
      // consistency (except possibly the column itself, which is going to be
      // deallocated anyway).
      //
      // If no pivot row is found in either direction, then the unknown is
      // unbounded in both directions and we are free to
      // perform any pivot at all. To do this, we just need to find any row with
      // a non-zero coefficient for the column.
      if (Optional<unsigned> maybeRow =
              findPivotRow({}, Direction::Up, column)) {
        row = *maybeRow;
      } else if (Optional<unsigned> maybeRow =
                     findPivotRow({}, Direction::Down, column)) {
        row = *maybeRow;
      } else {
        // The loop doesn't find a pivot row only if the column has zero
        // coefficients for every row. But the unknown is a constraint,
        // so it was added initially as a row. Such a row could never have been
        // pivoted to a column. So a pivot row will always be found.
        for (unsigned i = 0; i < nRow; ++i) {
          if (tableau(i, column) != 0) {
            row = i;
            break;
          }
        }
      }
      assert(row.hasValue() && "No pivot row found!");
      pivot(*row, column);
    }

    // Move this unknown to the last row and remove the last row from the
    // tableau.
    swapRows(constraint.pos, nRow - 1);
    // It is not strictly necessary to shrink the tableau, but for now we
    // maintain the invariant that the tableau has exactly nRow rows.
    tableau.resizeVertically(nRow - 1);
    nRow--;
    rowUnknown.pop_back();
    con.pop_back();
  } else if (entry == UndoLogEntry::UnmarkEmpty) {
    empty = false;
  }
}

/// Rollback to the specified snapshot.
///
/// We undo all the log entries until the log size when the snapshot was taken
/// is reached.
void Simplex::rollback(unsigned snapshot) {
  while (undoLog.size() > snapshot) {
    undo(undoLog.back());
    undoLog.pop_back();
  }
}

Optional<Fraction> Simplex::computeRowOptimum(Direction direction,
                                              unsigned row) {
  // Keep trying to find a pivot for the row in the specified direction.
  while (Optional<Pivot> maybePivot = findPivot(row, direction)) {
    // If findPivot returns a pivot involving the row itself, then the optimum
    // is unbounded, so we return None.
    if (maybePivot->row == row)
      return {};
    pivot(*maybePivot);
  }

  // The row has reached its optimal sample value, which we return.
  // The sample value is the entry in the constant column divided by the common
  // denominator for this row.
  return Fraction(tableau(row, 1), tableau(row, 0));
}

/// Compute the optimum of the specified expression in the specified direction,
/// or None if it is unbounded.
Optional<Fraction> Simplex::computeOptimum(Direction direction,
                                           ArrayRef<int64_t> coeffs) {
  assert(!empty && "Tableau should not be empty");

  unsigned snapshot = getSnapshot();
  unsigned conIndex = addRow(coeffs);
  unsigned row = con[conIndex].pos;
  Optional<Fraction> optimum = computeRowOptimum(direction, row);
  rollback(snapshot);
  return optimum;
}

bool Simplex::isUnbounded() {
  if (empty)
    return false;

  SmallVector<int64_t, 8> dir(var.size() + 1);
  for (unsigned i = 0; i < var.size(); ++i) {
    dir[i] = 1;

    Optional<Fraction> maybeMax = computeOptimum(Direction::Up, dir);
    if (!maybeMax)
      return true;

    Optional<Fraction> maybeMin = computeOptimum(Direction::Down, dir);
    if (!maybeMin)
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
///   A's rows, B's rows.
///
/// It has column layout:
///   denominator, constant, A's columns, B's columns.
Simplex Simplex::makeProduct(const Simplex &a, const Simplex &b) {
  unsigned numVar = a.numVariables() + b.numVariables();
  unsigned numCon = a.numConstraints() + b.numConstraints();
  Simplex result(numVar);

  result.tableau.resizeVertically(numCon);
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
    return index >= 0 ? a.numVariables() + index
                      : ~(a.numConstraints() + ~index);
  };

  result.colUnknown.assign(2, nullIndex);
  for (unsigned i = 2; i < a.nCol; ++i) {
    result.colUnknown.push_back(a.colUnknown[i]);
    result.unknownFromIndex(result.colUnknown.back()).pos =
        result.colUnknown.size() - 1;
  }
  for (unsigned i = 2; i < b.nCol; ++i) {
    result.colUnknown.push_back(indexFromBIndex(b.colUnknown[i]));
    result.unknownFromIndex(result.colUnknown.back()).pos =
        result.colUnknown.size() - 1;
  }

  auto appendRowFromA = [&](unsigned row) {
    for (unsigned col = 0; col < a.nCol; ++col)
      result.tableau(result.nRow, col) = a.tableau(row, col);
    result.rowUnknown.push_back(a.rowUnknown[row]);
    result.unknownFromIndex(result.rowUnknown.back()).pos =
        result.rowUnknown.size() - 1;
    result.nRow++;
  };

  // Also fixes the corresponding entry in rowUnknown and var/con (as the case
  // may be).
  auto appendRowFromB = [&](unsigned row) {
    result.tableau(result.nRow, 0) = b.tableau(row, 0);
    result.tableau(result.nRow, 1) = b.tableau(row, 1);

    unsigned offset = a.nCol - 2;
    for (unsigned col = 2; col < b.nCol; ++col)
      result.tableau(result.nRow, offset + col) = b.tableau(row, col);
    result.rowUnknown.push_back(indexFromBIndex(b.rowUnknown[row]));
    result.unknownFromIndex(result.rowUnknown.back()).pos =
        result.rowUnknown.size() - 1;
    result.nRow++;
  };

  for (unsigned row = 0; row < a.nRow; ++row)
    appendRowFromA(row);
  for (unsigned row = 0; row < b.nRow; ++row)
    appendRowFromB(row);

  return result;
}

Optional<SmallVector<int64_t, 8>> Simplex::getSamplePointIfIntegral() const {
  // The tableau is empty, so no sample point exists.
  if (empty)
    return {};

  SmallVector<int64_t, 8> sample;
  // Push the sample value for each variable into the vector.
  for (const Unknown &u : var) {
    if (u.orientation == Orientation::Column) {
      // If the variable is in column position, its sample value is zero.
      sample.push_back(0);
    } else {
      // If the variable is in row position, its sample value is the entry in
      // the constant column divided by the entry in the common denominator
      // column. If this is not an integer, then the sample point is not
      // integral so we return None.
      if (tableau(u.pos, 1) % tableau(u.pos, 0) != 0)
        return {};
      sample.push_back(tableau(u.pos, 1) / tableau(u.pos, 0));
    }
  }
  return sample;
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
class GBRSimplex {
  using Orientation = Simplex::Orientation;

public:
  GBRSimplex(const Simplex &originalSimplex)
      : simplex(Simplex::makeProduct(originalSimplex, originalSimplex)),
        simplexConstraintOffset(simplex.numConstraints()) {}

  /// Add an equality dotProduct(dir, x - y) == 0.
  /// First pushes a snapshot for the current simplex state to the stack so
  /// that this can be rolled back later.
  void addEqualityForDirection(ArrayRef<int64_t> dir) {
    assert(
        std::any_of(dir.begin(), dir.end(), [](int64_t x) { return x != 0; }) &&
        "Direction passed is the zero vector!");
    snapshotStack.push_back(simplex.getSnapshot());
    simplex.addEquality(getCoeffsForDirection(dir));
  }

  /// Compute max(dotProduct(dir, x - y)) and save the dual variables for only
  /// the direction equalities to `dual`.
  Fraction computeWidthAndDuals(ArrayRef<int64_t> dir,
                                SmallVectorImpl<int64_t> &dual,
                                int64_t &dualDenom) {
    unsigned snap = simplex.getSnapshot();
    unsigned conIndex = simplex.addRow(getCoeffsForDirection(dir));
    unsigned row = simplex.con[conIndex].pos;
    Optional<Fraction> maybeWidth =
        simplex.computeRowOptimum(Simplex::Direction::Up, row);
    assert(maybeWidth.hasValue() && "Width should not be unbounded!");
    dualDenom = simplex.tableau(row, 0);
    dual.clear();
    // The increment is i += 2 because equalities are added as two inequalities,
    // one positive and one negative. Each iteration processes one equality.
    for (unsigned i = simplexConstraintOffset; i < conIndex; i += 2) {
      // The dual variable is the negative of the coefficient of the new row
      // in the column of the constraint, if the constraint is in a column.
      // Note that the second inequality for the equality is negated.
      //
      // We want the dual for the original equality. If the positive inequality
      // is in column position, the negative of its row coefficient is the
      // desired dual. If the negative inequality is in column position, its row
      // coefficient is the desired dual. (its coefficients are already the
      // negated coefficients of the original equality, so we don't need to
      // negate it now.)
      //
      // If neither are in column position, we move the negated inequality to
      // column position. Since the inequality must have sample value zero
      // (since it corresponds to an equality), we are free to pivot with
      // any column. Since both the unknowns have sample value before and after
      // pivoting, no other sample values will change and the tableau will
      // remain consistent. To pivot, we just need to find a column that has a
      // non-zero coefficient in this row. There must be one since otherwise the
      // equality would be 0 == 0, which should never be passed to
      // addEqualityForDirection.
      //
      // After finding a column, we pivot with the column, after which we can
      // get the dual from the inequality in column position as explained above.
      if (simplex.con[i].orientation == Orientation::Column) {
        dual.push_back(-simplex.tableau(row, simplex.con[i].pos));
      } else {
        if (simplex.con[i + 1].orientation == Orientation::Row) {
          unsigned ineqRow = simplex.con[i + 1].pos;
          // Since it is an equality, the sample value must be zero.
          assert(simplex.tableau(ineqRow, 1) == 0 &&
                 "Equality's sample value must be zero.");
          for (unsigned col = 2; col < simplex.nCol; ++col) {
            if (simplex.tableau(ineqRow, col) != 0) {
              simplex.pivot(ineqRow, col);
              break;
            }
          }
          assert(simplex.con[i + 1].orientation == Orientation::Column &&
                 "No pivot found. Equality has all-zeros row in tableau!");
        }
        dual.push_back(simplex.tableau(row, simplex.con[i + 1].pos));
      }
    }
    simplex.rollback(snap);
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
  SmallVector<int64_t, 8> getCoeffsForDirection(ArrayRef<int64_t> dir) {
    assert(2 * dir.size() == simplex.numVariables() &&
           "Direction vector has wrong dimensionality");
    SmallVector<int64_t, 8> coeffs(dir.begin(), dir.end());
    coeffs.reserve(2 * dir.size());
    for (int64_t coeff : dir)
      coeffs.push_back(-coeff);
    coeffs.push_back(0); // constant term
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
void Simplex::reduceBasis(Matrix &basis, unsigned level) {
  const Fraction epsilon(3, 4);

  if (level == basis.getNumRows() - 1)
    return;

  GBRSimplex gbrSimplex(*this);
  SmallVector<Fraction, 8> width;
  SmallVector<int64_t, 8> dual;
  int64_t dualDenom;

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

    int64_t u = floorDiv(dual[i - level], dualDenom);
    basis.addToRow(i, i + 1, u);
    if (dual[i - level] % dualDenom != 0) {
      SmallVector<int64_t, 8> candidateDual[2];
      int64_t candidateDualDenom[2];
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
      dual = std::move(candidateDual[j]);
      dualDenom = candidateDualDenom[j];
      return widthI[j];
    }
    assert(i + 1 - level < width.size() && "width_{i+1} wasn't saved");
    // When dual minimizes f_i(b_{i+1} + dual*b_i), this is equal to
    // width_{i+1}(b_{i+1}).
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
      width.push_back(
          gbrSimplex.computeWidthAndDuals(basis.getRow(i), dual, dualDenom));
    }

    if (i >= level + dual.size()) {
      assert(i + 1 >= level + width.size() &&
             "We don't know dual_i but we know width_{i+1}");
      // We don't know dual for our level, so let's find it.
      gbrSimplex.addEqualityForDirection(basis.getRow(i));
      width.push_back(gbrSimplex.computeWidthAndDuals(basis.getRow(i + 1), dual,
                                                      dualDenom));
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
Optional<SmallVector<int64_t, 8>> Simplex::findIntegerSample() {
  if (empty)
    return {};

  unsigned nDims = var.size();
  Matrix basis = Matrix::identity(nDims);

  unsigned level = 0;
  // The snapshot just before constraining a direction to a value at each level.
  SmallVector<unsigned, 8> snapshotStack;
  // The maximum value in the range of the direction for each level.
  SmallVector<int64_t, 8> upperBoundStack;
  // The next value to try constraining the basis vector to at each level.
  SmallVector<int64_t, 8> nextValueStack;

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
      SmallVector<int64_t, 8> basisCoeffs =
          llvm::to_vector<8>(basis.getRow(level));
      basisCoeffs.push_back(0);

      int64_t minRoundedUp, maxRoundedDown;
      std::tie(minRoundedUp, maxRoundedDown) =
          computeIntegerBounds(basisCoeffs);

      // Heuristic: if the sample point is integral at this point, just return
      // it.
      if (auto maybeSample = getSamplePointIfIntegral())
        return *maybeSample;

      if (minRoundedUp < maxRoundedDown) {
        reduceBasis(basis, level);
        basisCoeffs = llvm::to_vector<8>(basis.getRow(level));
        basisCoeffs.push_back(0);
        std::tie(minRoundedUp, maxRoundedDown) =
            computeIntegerBounds(basisCoeffs);
      }

      snapshotStack.push_back(getSnapshot());
      // The smallest value in the range is the next value to try.
      nextValueStack.push_back(minRoundedUp);
      upperBoundStack.push_back(maxRoundedDown);
    }

    assert((snapshotStack.size() - 1 == level &&
            nextValueStack.size() - 1 == level &&
            upperBoundStack.size() - 1 == level) &&
           "Mismatched variable stack sizes!");

    // Whether we "recursed" or "returned" from a lower level, we rollback
    // to the snapshot of the starting state at this level. (in the "recursed"
    // case this has no effect)
    rollback(snapshotStack.back());
    int64_t nextValue = nextValueStack.back();
    nextValueStack.back()++;
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
    SmallVector<int64_t, 8> basisCoeffs(basis.getRow(level).begin(),
                                        basis.getRow(level).end());
    basisCoeffs.push_back(-nextValue);
    addEquality(basisCoeffs);
    level++;
  }

  return {};
}

/// Compute the minimum and maximum integer values the expression can take. We
/// compute each separately.
std::pair<int64_t, int64_t>
Simplex::computeIntegerBounds(ArrayRef<int64_t> coeffs) {
  int64_t minRoundedUp;
  if (Optional<Fraction> maybeMin =
          computeOptimum(Simplex::Direction::Down, coeffs))
    minRoundedUp = ceil(*maybeMin);
  else
    llvm_unreachable("Tableau should not be unbounded");

  int64_t maxRoundedDown;
  if (Optional<Fraction> maybeMax =
          computeOptimum(Simplex::Direction::Up, coeffs))
    maxRoundedDown = floor(*maybeMax);
  else
    llvm_unreachable("Tableau should not be unbounded");

  return {minRoundedUp, maxRoundedDown};
}

void Simplex::print(raw_ostream &os) const {
  os << "rows = " << nRow << ", columns = " << nCol << "\n";
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
  for (unsigned row = 0; row < nRow; ++row) {
    if (row > 0)
      os << ", ";
    os << "r" << row << ": " << rowUnknown[row];
  }
  os << '\n';
  os << "c0: denom, c1: const";
  for (unsigned col = 2; col < nCol; ++col)
    os << ", c" << col << ": " << colUnknown[col];
  os << '\n';
  for (unsigned row = 0; row < nRow; ++row) {
    for (unsigned col = 0; col < nCol; ++col)
      os << tableau(row, col) << '\t';
    os << '\n';
  }
  os << '\n';
}

void Simplex::dump() const { print(llvm::errs()); }

} // namespace mlir
