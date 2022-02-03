//===- Simplex.h - MLIR Simplex Class ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functionality to perform analysis on an IntegerPolyhedron. In particular,
// support for performing emptiness checks, redundancy checks and obtaining the
// lexicographically minimum rational element in a set.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_SIMPLEX_H
#define MLIR_ANALYSIS_PRESBURGER_SIMPLEX_H

#include "mlir/Analysis/Presburger/Fraction.h"
#include "mlir/Analysis/Presburger/IntegerPolyhedron.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

class GBRSimplex;

/// The Simplex class implements a version of the Simplex and Generalized Basis
/// Reduction algorithms, which can perform analysis of integer sets with affine
/// inequalities and equalities. A Simplex can be constructed
/// by specifying the dimensionality of the set. It supports adding affine
/// inequalities and equalities, and can perform emptiness checks, i.e., it can
/// find a solution to the set of constraints if one exists, or say that the
/// set is empty if no solution exists. Furthermore, it can find a subset of
/// these constraints that are redundant, i.e. a subset of constraints that
/// doesn't constrain the affine set further after adding the non-redundant
/// constraints. The LexSimplex class provides support for computing the
/// lexicographical minimum of an IntegerPolyhedron. Both these classes can be
/// constructed from an IntegerPolyhedron, and both inherit common
/// functionality from SimplexBase.
///
/// The implementations of the Simplex and SimplexBase classes, other than the
/// functionality for obtaining an integer sample, are based on the paper
/// "Simplify: A Theorem Prover for Program Checking"
/// by D. Detlefs, G. Nelson, J. B. Saxe.
///
/// We define variables, constraints, and unknowns. Consider the example of a
/// two-dimensional set defined by 1 + 2x + 3y >= 0 and 2x - 3y >= 0. Here,
/// x, y, are variables while 1 + 2x + 3y >= 0, 2x - 3y >= 0 are constraints.
/// Unknowns are either variables or constraints, i.e., x, y, 1 + 2x + 3y >= 0,
/// 2x - 3y >= 0 are all unknowns.
///
/// The implementation involves a matrix called a tableau, which can be thought
/// of as a 2D matrix of rational numbers having number of rows equal to the
/// number of constraints and number of columns equal to one plus the number of
/// variables. In our implementation, instead of storing rational numbers, we
/// store a common denominator for each row, so it is in fact a matrix of
/// integers with number of rows equal to number of constraints and number of
/// columns equal to _two_ plus the number of variables. For example, instead of
/// storing a row of three rationals [1/2, 2/3, 3], we would store [6, 3, 4, 18]
/// since 3/6 = 1/2, 4/6 = 2/3, and 18/6 = 3.
///
/// Every row and column except the first and second columns is associated with
/// an unknown and every unknown is associated with a row or column. An unknown
/// associated with a row or column is said to be in row or column orientation
/// respectively. As described above, the first column is the common
/// denominator. The second column represents the constant term, explained in
/// more detail below. These two are _fixed columns_; they always retain their
/// position as the first and second columns. Additionally, LexSimplex stores
/// a so-call big M parameter (explained below) in the third column, so
/// LexSimplex has three fixed columns.
///
/// LexSimplex does not directly support variables which can be negative, so we
/// introduce the so-called big M parameter, an artificial variable that is
/// considered to have an arbitrarily large value. We then transform the
/// variables, say x, y, z, ... to M, M + x, M + y, M + z. Since M has been
/// added to these variables, they are now known to have non-negative values.
/// For more details, see the documentation for LexSimplex. The big M parameter
/// is not considered a real unknown and is not stored in the `var` data
/// structure; rather the tableau just has an extra fixed column for it just
/// like the constant term.
///
/// The vectors var and con store information about the variables and
/// constraints respectively, namely, whether they are in row or column
/// position, which row or column they are associated with, and whether they
/// correspond to a variable or a constraint.
///
/// An unknown is addressed by its index. If the index i is non-negative, then
/// the variable var[i] is being addressed. If the index i is negative, then
/// the constraint con[~i] is being addressed. Effectively this maps
/// 0 -> var[0], 1 -> var[1], -1 -> con[0], -2 -> con[1], etc. rowUnknown[r] and
/// colUnknown[c] are the indexes of the unknowns associated with row r and
/// column c, respectively.
///
/// The unknowns in column position are together called the basis. Initially the
/// basis is the set of variables -- in our example above, the initial basis is
/// x, y.
///
/// The unknowns in row position are represented in terms of the basis unknowns.
/// If the basis unknowns are u_1, u_2, ... u_m, and a row in the tableau is
/// d, c, a_1, a_2, ... a_m, this represents the unknown for that row as
/// (c + a_1*u_1 + a_2*u_2 + ... + a_m*u_m)/d. In our running example, if the
/// basis is the initial basis of x, y, then the constraint 1 + 2x + 3y >= 0
/// would be represented by the row [1, 1, 2, 3].
///
/// The association of unknowns to rows and columns can be changed by a process
/// called pivoting, where a row unknown and a column unknown exchange places
/// and the remaining row variables' representation is changed accordingly
/// by eliminating the old column unknown in favour of the new column unknown.
/// If we had pivoted the column for x with the row for 2x - 3y >= 0,
/// the new row for x would be [2, 1, 3] since x = (1*(2x - 3y) + 3*y)/2.
/// See the documentation for the pivot member function for details.
///
/// The association of unknowns to rows and columns is called the _tableau
/// configuration_. The _sample value_ of an unknown in a particular tableau
/// configuration is its value if all the column unknowns were set to zero.
/// Concretely, for unknowns in column position the sample value is zero; when
/// the big M parameter is not used, for unknowns in row position the sample
/// value is the constant term divided by the common denominator. When the big M
/// parameter is used, if d is the denominator, p is the big M coefficient, and
/// c is the constant term, then the sample value is (p*M + c)/d. Since M is
/// considered to be positive infinity, this is positive (negative) infinity
/// when p is positive or negative, and c/d when p is zero.
///
/// The tableau configuration is called _consistent_ if the sample value of all
/// restricted unknowns is non-negative. Initially there are no constraints, and
/// the tableau is consistent. When a new constraint is added, its sample value
/// in the current tableau configuration may be negative. In that case, we try
/// to find a series of pivots to bring us to a consistent tableau
/// configuration, i.e. we try to make the new constraint's sample value
/// non-negative without making that of any other constraints negative. (See
/// findPivot and findPivotRow for details.) If this is not possible, then the
/// set of constraints is mutually contradictory and the tableau is marked
/// _empty_, which means the set of constraints has no solution.
///
/// This SimplexBase class also supports taking snapshots of the current state
/// and rolling back to prior snapshots. This works by maintaining an undo log
/// of operations. Snapshots are just pointers to a particular location in the
/// log, and rolling back to a snapshot is done by reverting each log entry's
/// operation from the end until we reach the snapshot's location. SimplexBase
/// also supports taking a snapshot including the exact set of basis unknowns;
/// if this functionality is used, then on rolling back the exact basis will
/// also be restored. This is used by LexSimplex because its algorithm, unlike
/// Simplex, is sensitive to the exact basis used at a point.
class SimplexBase {
public:
  SimplexBase() = delete;
  virtual ~SimplexBase() = default;

  /// Construct a SimplexBase with the specified number of variables and fixed
  /// columns.
  ///
  /// For example, Simplex uses two fixed columns: the denominator and the
  /// constant term, whereas LexSimplex has an extra fixed column for the
  /// so-called big M parameter. For more information see the documentation for
  /// LexSimplex.
  SimplexBase(unsigned nVar, bool mustUseBigM);

  /// Returns true if the tableau is empty (has conflicting constraints),
  /// false otherwise.
  bool isEmpty() const;

  /// Add an inequality to the tableau. If coeffs is c_0, c_1, ... c_n, where n
  /// is the current number of variables, then the corresponding inequality is
  /// c_n + c_0*x_0 + c_1*x_1 + ... + c_{n-1}*x_{n-1} >= 0.
  virtual void addInequality(ArrayRef<int64_t> coeffs) = 0;

  /// Returns the number of variables in the tableau.
  unsigned getNumVariables() const;

  /// Returns the number of constraints in the tableau.
  unsigned getNumConstraints() const;

  /// Add an equality to the tableau. If coeffs is c_0, c_1, ... c_n, where n
  /// is the current number of variables, then the corresponding equality is
  /// c_n + c_0*x_0 + c_1*x_1 + ... + c_{n-1}*x_{n-1} == 0.
  void addEquality(ArrayRef<int64_t> coeffs);

  /// Add new variables to the end of the list of variables.
  void appendVariable(unsigned count = 1);

  /// Mark the tableau as being empty.
  void markEmpty();

  /// Get a snapshot of the current state. This is used for rolling back.
  /// The same basis will not necessarily be restored on rolling back.
  /// The snapshot only captures the set of variables and constraints present
  /// in the Simplex.
  unsigned getSnapshot() const;

  /// Get a snapshot of the current state including the basis. When rolling
  /// back, the exact basis will be restored.
  unsigned getSnapshotBasis();

  /// Rollback to a snapshot. This invalidates all later snapshots.
  void rollback(unsigned snapshot);

  /// Add all the constraints from the given IntegerPolyhedron.
  void intersectIntegerPolyhedron(const IntegerPolyhedron &poly);

  /// Returns the current sample point, which may contain non-integer (rational)
  /// coordinates. Returns an empty optional when the tableau is empty.
  ///
  /// Also returns empty when the big M parameter is used and a variable
  /// has a non-zero big M coefficient, meaning its value is infinite or
  /// unbounded.
  Optional<SmallVector<Fraction, 8>> getRationalSample() const;

  /// Print the tableau's internal state.
  void print(raw_ostream &os) const;
  void dump() const;

protected:
  enum class Orientation { Row, Column };

  /// An Unknown is either a variable or a constraint. It is always associated
  /// with either a row or column. Whether it's a row or a column is specified
  /// by the orientation and pos identifies the specific row or column it is
  /// associated with. If the unknown is restricted, then it has a
  /// non-negativity constraint associated with it, i.e., its sample value must
  /// always be non-negative and if it cannot be made non-negative without
  /// violating other constraints, the tableau is empty.
  struct Unknown {
    Unknown(Orientation oOrientation, bool oRestricted, unsigned oPos)
        : pos(oPos), orientation(oOrientation), restricted(oRestricted) {}
    unsigned pos;
    Orientation orientation;
    bool restricted : 1;

    void print(raw_ostream &os) const {
      os << (orientation == Orientation::Row ? "r" : "c");
      os << pos;
      if (restricted)
        os << " [>=0]";
    }
  };

  struct Pivot {
    unsigned row, column;
  };

  /// Return any row that this column can be pivoted with, ignoring tableau
  /// consistency.
  ///
  /// Returns an empty optional if no pivot is possible, which happens only when
  /// the column unknown is a variable and no constraint has a non-zero
  /// coefficient for it.
  Optional<unsigned> findAnyPivotRow(unsigned col);

  /// Swap the row with the column in the tableau's data structures but not the
  /// tableau itself. This is used by pivot.
  void swapRowWithCol(unsigned row, unsigned col);

  /// Pivot the row with the column.
  void pivot(unsigned row, unsigned col);
  void pivot(Pivot pair);

  /// Returns the unknown associated with index.
  const Unknown &unknownFromIndex(int index) const;
  /// Returns the unknown associated with col.
  const Unknown &unknownFromColumn(unsigned col) const;
  /// Returns the unknown associated with row.
  const Unknown &unknownFromRow(unsigned row) const;
  /// Returns the unknown associated with index.
  Unknown &unknownFromIndex(int index);
  /// Returns the unknown associated with col.
  Unknown &unknownFromColumn(unsigned col);
  /// Returns the unknown associated with row.
  Unknown &unknownFromRow(unsigned row);

  /// Add a new row to the tableau and the associated data structures.
  /// The new row is considered to be a constraint; the new Unknown lives in
  /// con.
  ///
  /// Returns the index of the new Unknown in con.
  unsigned addRow(ArrayRef<int64_t> coeffs, bool makeRestricted = false);

  /// Normalize the given row by removing common factors between the numerator
  /// and the denominator.
  void normalizeRow(unsigned row);

  /// Swap the two rows/columns in the tableau and associated data structures.
  void swapRows(unsigned i, unsigned j);
  void swapColumns(unsigned i, unsigned j);

  /// Enum to denote operations that need to be undone during rollback.
  enum class UndoLogEntry {
    RemoveLastConstraint,
    RemoveLastVariable,
    UnmarkEmpty,
    UnmarkLastRedundant,
    RestoreBasis
  };

  /// Undo the addition of the last constraint. This will only be called from
  /// undo, when rolling back.
  virtual void undoLastConstraint() = 0;

  /// Remove the last constraint, which must be in row orientation.
  void removeLastConstraintRowOrientation();

  /// Undo the operation represented by the log entry.
  void undo(UndoLogEntry entry);

  /// Return the number of fixed columns, as described in the constructor above,
  /// this is the number of columns beyond those for the variables in var.
  unsigned getNumFixedCols() const { return usingBigM ? 3u : 2u; }

  /// Stores whether or not a big M column is present in the tableau.
  const bool usingBigM;

  /// The number of rows in the tableau.
  unsigned nRow;

  /// The number of columns in the tableau, including the common denominator
  /// and the constant column.
  unsigned nCol;

  /// The number of redundant rows in the tableau. These are the first
  /// nRedundant rows.
  unsigned nRedundant;

  /// The matrix representing the tableau.
  Matrix tableau;

  /// This is true if the tableau has been detected to be empty, false
  /// otherwise.
  bool empty;

  /// Holds a log of operations, used for rolling back to a previous state.
  SmallVector<UndoLogEntry, 8> undoLog;

  /// Holds a vector of bases. The ith saved basis is the basis that should be
  /// restored when processing the ith occurrance of UndoLogEntry::RestoreBasis
  /// in undoLog. This is used by getSnapshotBasis.
  SmallVector<SmallVector<int, 8>, 8> savedBases;

  /// These hold the indexes of the unknown at a given row or column position.
  /// We keep these as signed integers since that makes it convenient to check
  /// if an index corresponds to a variable or a constraint by checking the
  /// sign.
  ///
  /// colUnknown is padded with two null indexes at the front since the first
  /// two columns don't correspond to any unknowns.
  SmallVector<int, 8> rowUnknown, colUnknown;

  /// These hold information about each unknown.
  SmallVector<Unknown, 8> con, var;
};

/// Simplex class using the lexicographic pivot rule. Used for lexicographic
/// optimization. The implementation of this class is based on the paper
/// "Parametric Integer Programming" by Paul Feautrier.
///
/// This does not directly support negative-valued variables, so it uses the big
/// M parameter trick to make all the variables non-negative. Basically we
/// introduce an artifical variable M that is considered to have a value of
/// +infinity and instead of the variables x, y, z, we internally use variables
/// M + x, M + y, M + z, which are now guaranteed to be non-negative. See the
/// documentation for Simplex for more details. The whole algorithm is performed
/// without having to fix a "big enough" value of the big M parameter; it is
/// just considered to be infinite throughout and it never appears in the final
/// outputs. We will deal with sample values throughout that may in general be
/// some linear expression involving M like pM + q or aM + b. We can compare
/// these with each other. They have a total order:
/// aM + b < pM + q iff a < p or (a == p and b < q).
/// In particular, aM + b < 0 iff a < 0 or (a == 0 and b < 0).
///
/// Initially all the constraints to be added are added as rows, with no attempt
/// to keep the tableau consistent. Pivots are only performed when some query
/// is made, such as a call to getRationalLexMin. Care is taken to always
/// maintain a lexicopositive basis transform, explained below.
///
/// Let the variables be x = (x_1, ... x_n). Let the basis unknowns at a
/// particular point be  y = (y_1, ... y_n). We know that x = A*y + b for some
/// n x n matrix A and n x 1 column vector b. We want every column in A to be
/// lexicopositive, i.e., have at least one non-zero element, with the first
/// such element being positive. This property is preserved throughout the
/// operation of LexSimplex. Note that on construction, the basis transform A is
/// the indentity matrix and so every column is lexicopositive. Note that for
/// LexSimplex, for the tableau to be consistent we must have non-negative
/// sample values not only for the constraints but also for the variables.
/// So if the tableau is consistent then x >= 0 and y >= 0, by which we mean
/// every element in these vectors is non-negative. (note that this is a
/// different concept from lexicopositivity!)
///
/// When we arrive at a basis such the basis transform is lexicopositive and the
/// tableau is consistent, the sample point is the lexiographically minimum
/// point in the polytope. We will show that A*y is zero or lexicopositive when
/// y >= 0. Adding a lexicopositive vector to b will make it lexicographically
/// bigger, so A*y + b is lexicographically bigger than b for any y >= 0 except
/// y = 0. This shows that no point lexicographically smaller than x = b can be
/// obtained. Since we already know that x = b is valid point in the space, this
/// shows that x = b is the lexicographic minimum.
///
/// Proof that A*y is lexicopositive or zero when y > 0. Recall that every
/// column of A is lexicopositive. Begin by considering A_1, the first row of A.
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
/// ignore more columns; eventually if all these dot products become zero then
/// A*y is zero and we are done.
class LexSimplex : public SimplexBase {
public:
  explicit LexSimplex(unsigned nVar)
      : SimplexBase(nVar, /*mustUseBigM=*/true) {}
  explicit LexSimplex(const IntegerPolyhedron &constraints)
      : LexSimplex(constraints.getNumIds()) {
    intersectIntegerPolyhedron(constraints);
  }
  ~LexSimplex() override = default;

  /// Add an inequality to the tableau. If coeffs is c_0, c_1, ... c_n, where n
  /// is the current number of variables, then the corresponding inequality is
  /// c_n + c_0*x_0 + c_1*x_1 + ... + c_{n-1}*x_{n-1} >= 0.
  ///
  /// This just adds the inequality to the tableau and does not try to create a
  /// consistent tableau configuration.
  void addInequality(ArrayRef<int64_t> coeffs) final {
    addRow(coeffs, /*makeRestricted=*/true);
  }

  /// Get a snapshot of the current state. This is used for rolling back.
  unsigned getSnapshot() { return SimplexBase::getSnapshotBasis(); }

  /// Return the lexicographically minimum rational solution to the constraints.
  Optional<SmallVector<Fraction, 8>> getRationalLexMin();

protected:
  /// Undo the addition of the last constraint. This is only called while
  /// rolling back.
  void undoLastConstraint() final;

  /// Make the tableau configuration consistent.
  void restoreRationalConsistency();

  /// Return whether the specified row is violated;
  bool rowIsViolated(unsigned row) const;

  /// Get a constraint row that is violated, if one exists.
  /// Otherwise, return an empty optional.
  Optional<unsigned> maybeGetViolatedRow() const;

  /// Given two potential pivot columns for a row, return the one that results
  /// in the lexicographically smallest sample vector.
  unsigned getLexMinPivotColumn(unsigned row, unsigned colA,
                                unsigned colB) const;

  /// Try to move the specified row to column orientation while preserving the
  /// lexicopositivity of the basis transform. If this is not possible, return
  /// failure. This only occurs when the constraints have no solution; the
  /// tableau will be marked empty in such a case.
  LogicalResult moveRowUnknownToColumn(unsigned row);
};

/// The Simplex class uses the Normal pivot rule and supports integer emptiness
/// checks as well as detecting redundancies.
///
/// The Simplex class supports redundancy checking via detectRedundant and
/// isMarkedRedundant. A redundant constraint is one which is never violated as
/// long as the other constraints are not violated, i.e., removing a redundant
/// constraint does not change the set of solutions to the constraints. As a
/// heuristic, constraints that have been marked redundant can be ignored for
/// most operations. Therefore, these constraints are kept in rows 0 to
/// nRedundant - 1, where nRedundant is a member variable that tracks the number
/// of constraints that have been marked redundant.
///
/// Finding an integer sample is done with the Generalized Basis Reduction
/// algorithm. See the documentation for findIntegerSample and reduceBasis.
class Simplex : public SimplexBase {
public:
  enum class Direction { Up, Down };

  Simplex() = delete;
  explicit Simplex(unsigned nVar) : SimplexBase(nVar, /*mustUseBigM=*/false) {}
  explicit Simplex(const IntegerPolyhedron &constraints)
      : Simplex(constraints.getNumIds()) {
    intersectIntegerPolyhedron(constraints);
  }
  ~Simplex() override = default;

  /// Add an inequality to the tableau. If coeffs is c_0, c_1, ... c_n, where n
  /// is the current number of variables, then the corresponding inequality is
  /// c_n + c_0*x_0 + c_1*x_1 + ... + c_{n-1}*x_{n-1} >= 0.
  ///
  /// This also tries to restore the tableau configuration to a consistent
  /// state and marks the Simplex empty if this is not possible.
  void addInequality(ArrayRef<int64_t> coeffs) final;

  /// Compute the maximum or minimum value of the given row, depending on
  /// direction. The specified row is never pivoted. On return, the row may
  /// have a negative sample value if the direction is down.
  ///
  /// Returns a Fraction denoting the optimum, or a null value if no optimum
  /// exists, i.e., if the expression is unbounded in this direction.
  Optional<Fraction> computeRowOptimum(Direction direction, unsigned row);

  /// Compute the maximum or minimum value of the given expression, depending on
  /// direction. Should not be called when the Simplex is empty.
  ///
  /// Returns a Fraction denoting the optimum, or a null value if no optimum
  /// exists, i.e., if the expression is unbounded in this direction.
  Optional<Fraction> computeOptimum(Direction direction,
                                    ArrayRef<int64_t> coeffs);

  /// Returns whether the perpendicular of the specified constraint is a
  /// is a direction along which the polytope is bounded.
  bool isBoundedAlongConstraint(unsigned constraintIndex);

  /// Returns whether the specified constraint has been marked as redundant.
  /// Constraints are numbered from 0 starting at the first added inequality.
  /// Equalities are added as a pair of inequalities and so correspond to two
  /// inequalities with successive indices.
  bool isMarkedRedundant(unsigned constraintIndex) const;

  /// Finds a subset of constraints that is redundant, i.e., such that
  /// the set of solutions does not change if these constraints are removed.
  /// Marks these constraints as redundant. Whether a specific constraint has
  /// been marked redundant can be queried using isMarkedRedundant.
  void detectRedundant();

  /// Returns a (min, max) pair denoting the minimum and maximum integer values
  /// of the given expression.
  std::pair<int64_t, int64_t> computeIntegerBounds(ArrayRef<int64_t> coeffs);

  /// Returns true if the polytope is unbounded, i.e., extends to infinity in
  /// some direction. Otherwise, returns false.
  bool isUnbounded();

  /// Make a tableau to represent a pair of points in the given tableaus, one in
  /// tableau A and one in B.
  static Simplex makeProduct(const Simplex &a, const Simplex &b);

  /// Returns an integer sample point if one exists, or None
  /// otherwise. This should only be called for bounded sets.
  Optional<SmallVector<int64_t, 8>> findIntegerSample();

  /// Check if the specified inequality already holds in the polytope.
  bool isRedundantInequality(ArrayRef<int64_t> coeffs);

  /// Check if the specified equality already holds in the polytope.
  bool isRedundantEquality(ArrayRef<int64_t> coeffs);

  /// Returns true if this Simplex's polytope is a rational subset of `poly`.
  /// Otherwise, returns false.
  bool isRationalSubsetOf(const IntegerPolyhedron &poly);

  /// Returns the current sample point if it is integral. Otherwise, returns
  /// None.
  Optional<SmallVector<int64_t, 8>> getSamplePointIfIntegral() const;

private:
  friend class GBRSimplex;

  /// Restore the unknown to a non-negative sample value.
  ///
  /// Returns success if the unknown was successfully restored to a non-negative
  /// sample value, failure otherwise.
  LogicalResult restoreRow(Unknown &u);

  /// Find a pivot to change the sample value of row in the specified
  /// direction while preserving tableau consistency, except that if the
  /// direction is down then the pivot may make the specified row take a
  /// negative value. The returned pivot row will be row if and only if the
  /// unknown is unbounded in the specified direction.
  ///
  /// Returns a (row, col) pair denoting a pivot, or an empty Optional if
  /// no valid pivot exists.
  Optional<Pivot> findPivot(int row, Direction direction) const;

  /// Find a row that can be used to pivot the column in the specified
  /// direction. If skipRow is not null, then this row is excluded
  /// from consideration. The returned pivot will maintain all constraints
  /// except the column itself and skipRow, if it is set. (if these unknowns
  /// are restricted).
  ///
  /// Returns the row to pivot to, or an empty Optional if the column
  /// is unbounded in the specified direction.
  Optional<unsigned> findPivotRow(Optional<unsigned> skipRow,
                                  Direction direction, unsigned col) const;

  /// Undo the addition of the last constraint while preserving tableau
  /// consistency.
  void undoLastConstraint() final;

  /// Compute the maximum or minimum of the specified Unknown, depending on
  /// direction. The specified unknown may be pivoted. If the unknown is
  /// restricted, it will have a non-negative sample value on return.
  /// Should not be called if the Simplex is empty.
  ///
  /// Returns a Fraction denoting the optimum, or a null value if no optimum
  /// exists, i.e., if the expression is unbounded in this direction.
  Optional<Fraction> computeOptimum(Direction direction, Unknown &u);

  /// Mark the specified unknown redundant. This operation is added to the undo
  /// log and will be undone by rollbacks. The specified unknown must be in row
  /// orientation.
  void markRowRedundant(Unknown &u);

  /// Reduce the given basis, starting at the specified level, using general
  /// basis reduction.
  void reduceBasis(Matrix &basis, unsigned level);
};

} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_SIMPLEX_H
