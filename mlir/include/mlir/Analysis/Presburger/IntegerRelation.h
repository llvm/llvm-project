//===- IntegerRelation.h - MLIR IntegerRelation Class ---------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A class to represent a relation over integer tuples. A relation is
// represented as a constraint system over a space of tuples of integer valued
// variables supporting symbolic identifiers and existential quantification.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_INTEGERRELATION_H
#define MLIR_ANALYSIS_PRESBURGER_INTEGERRELATION_H

#include "mlir/Analysis/Presburger/Fraction.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include "mlir/Analysis/Presburger/Utils.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace presburger {

class IntegerRelation;
class IntegerPolyhedron;
class PresburgerSet;

/// An IntegerRelation represents the set of points from a PresburgerSpace that
/// satisfy a list of affine constraints. Affine constraints can be inequalities
/// or equalities in the form:
///
/// Inequality: c_0*x_0 + c_1*x_1 + .... + c_{n-1}*x_{n-1} + c_n >= 0
/// Equality  : c_0*x_0 + c_1*x_1 + .... + c_{n-1}*x_{n-1} + c_n == 0
///
/// where c_0, c_1, ..., c_n are integers and n is the total number of
/// identifiers in the space.
///
/// Such a relation corresponds to the set of integer points lying in a convex
/// polyhedron. For example, consider the relation:
///         (x) -> (y) : (1 <= x <= 7, x = 2y)
/// These can be thought of as points in the polyhedron:
///         (x, y) : (1 <= x <= 7, x = 2y)
/// This relation contains the pairs (2, 1), (4, 2), and (6, 3).
///
/// Since IntegerRelation makes a distinction between dimensions, IdKind::Range
/// and IdKind::Domain should be used to refer to dimension identifiers.
class IntegerRelation {
public:
  /// All derived classes of IntegerRelation.
  enum class Kind {
    FlatAffineConstraints,
    FlatAffineValueConstraints,
    IntegerRelation,
    IntegerPolyhedron,
  };

  /// Constructs a relation reserving memory for the specified number
  /// of constraints and identifiers.
  IntegerRelation(unsigned numReservedInequalities,
                  unsigned numReservedEqualities, unsigned numReservedCols,
                  const PresburgerSpace &space)
      : space(space), equalities(0, space.getNumIds() + 1,
                                 numReservedEqualities, numReservedCols),
        inequalities(0, space.getNumIds() + 1, numReservedInequalities,
                     numReservedCols) {
    assert(numReservedCols >= space.getNumIds() + 1);
  }

  /// Constructs a relation with the specified number of dimensions and symbols.
  explicit IntegerRelation(const PresburgerSpace &space)
      : IntegerRelation(/*numReservedInequalities=*/0,
                        /*numReservedEqualities=*/0,
                        /*numReservedCols=*/space.getNumIds() + 1, space) {}

  virtual ~IntegerRelation() = default;

  /// Return a system with no constraints, i.e., one which is satisfied by all
  /// points.
  static IntegerRelation getUniverse(const PresburgerSpace &space) {
    return IntegerRelation(space);
  }

  /// Return the kind of this IntegerRelation.
  virtual Kind getKind() const { return Kind::IntegerRelation; }

  static bool classof(const IntegerRelation *cst) { return true; }

  // Clones this object.
  std::unique_ptr<IntegerRelation> clone() const;

  /// Returns a reference to the underlying space.
  const PresburgerSpace &getSpace() const { return space; }

  /// Set the space to `oSpace`, which should have the same number of ids as
  /// the current space.
  void setSpace(const PresburgerSpace &oSpace);

  /// Set the space to `oSpace`, which should not have any local ids.
  /// `oSpace` can have fewer ids than the current space; in that case, the
  /// the extra ids in `this` that are not accounted for by `oSpace` will be
  /// considered as local ids. `oSpace` should not have more ids than the
  /// current space; this will result in an assert failure.
  void setSpaceExceptLocals(const PresburgerSpace &oSpace);

  /// Returns a copy of the space without locals.
  PresburgerSpace getSpaceWithoutLocals() const {
    return PresburgerSpace::getRelationSpace(space.getNumDomainIds(),
                                             space.getNumRangeIds(),
                                             space.getNumSymbolIds());
  }

  /// Appends constraints from `other` into `this`. This is equivalent to an
  /// intersection with no simplification of any sort attempted.
  void append(const IntegerRelation &other);

  /// Return the intersection of the two sets.
  /// If there are locals, they will be merged.
  IntegerRelation intersect(IntegerRelation other) const;

  /// Return whether `this` and `other` are equal. This is integer-exact
  /// and somewhat expensive, since it uses the integer emptiness check
  /// (see IntegerRelation::findIntegerSample()).
  bool isEqual(const IntegerRelation &other) const;

  /// Return whether this is a subset of the given IntegerRelation. This is
  /// integer-exact and somewhat expensive, since it uses the integer emptiness
  /// check (see IntegerRelation::findIntegerSample()).
  bool isSubsetOf(const IntegerRelation &other) const;

  /// Returns the value at the specified equality row and column.
  inline int64_t atEq(unsigned i, unsigned j) const { return equalities(i, j); }
  inline int64_t &atEq(unsigned i, unsigned j) { return equalities(i, j); }

  /// Returns the value at the specified inequality row and column.
  inline int64_t atIneq(unsigned i, unsigned j) const {
    return inequalities(i, j);
  }
  inline int64_t &atIneq(unsigned i, unsigned j) { return inequalities(i, j); }

  unsigned getNumConstraints() const {
    return getNumInequalities() + getNumEqualities();
  }

  unsigned getNumDomainIds() const { return space.getNumDomainIds(); }
  unsigned getNumRangeIds() const { return space.getNumRangeIds(); }
  unsigned getNumSymbolIds() const { return space.getNumSymbolIds(); }
  unsigned getNumLocalIds() const { return space.getNumLocalIds(); }

  unsigned getNumDimIds() const { return space.getNumDimIds(); }
  unsigned getNumDimAndSymbolIds() const {
    return space.getNumDimAndSymbolIds();
  }
  unsigned getNumIds() const { return space.getNumIds(); }

  /// Returns the number of columns in the constraint system.
  inline unsigned getNumCols() const { return space.getNumIds() + 1; }

  inline unsigned getNumEqualities() const { return equalities.getNumRows(); }

  inline unsigned getNumInequalities() const {
    return inequalities.getNumRows();
  }

  inline unsigned getNumReservedEqualities() const {
    return equalities.getNumReservedRows();
  }

  inline unsigned getNumReservedInequalities() const {
    return inequalities.getNumReservedRows();
  }

  inline ArrayRef<int64_t> getEquality(unsigned idx) const {
    return equalities.getRow(idx);
  }

  inline ArrayRef<int64_t> getInequality(unsigned idx) const {
    return inequalities.getRow(idx);
  }

  /// Get the number of ids of the specified kind.
  unsigned getNumIdKind(IdKind kind) const { return space.getNumIdKind(kind); };

  /// Return the index at which the specified kind of id starts.
  unsigned getIdKindOffset(IdKind kind) const {
    return space.getIdKindOffset(kind);
  };

  /// Return the index at Which the specified kind of id ends.
  unsigned getIdKindEnd(IdKind kind) const { return space.getIdKindEnd(kind); };

  /// Get the number of elements of the specified kind in the range
  /// [idStart, idLimit).
  unsigned getIdKindOverlap(IdKind kind, unsigned idStart,
                            unsigned idLimit) const {
    return space.getIdKindOverlap(kind, idStart, idLimit);
  };

  /// Return the IdKind of the id at the specified position.
  IdKind getIdKindAt(unsigned pos) const { return space.getIdKindAt(pos); };

  /// The struct CountsSnapshot stores the count of each IdKind, and also of
  /// each constraint type. getCounts() returns a CountsSnapshot object
  /// describing the current state of the IntegerRelation. truncate() truncates
  /// all ids of each IdKind and all constraints of both kinds beyond the counts
  /// in the specified CountsSnapshot object. This can be used to achieve
  /// rudimentary rollback support. As long as none of the existing constraints
  /// or ids are disturbed, and only additional ids or constraints are added,
  /// this addition can be rolled back using truncate.
  struct CountsSnapshot {
  public:
    CountsSnapshot(const PresburgerSpace &space, unsigned numIneqs,
                   unsigned numEqs)
        : space(space), numIneqs(numIneqs), numEqs(numEqs) {}
    const PresburgerSpace &getSpace() const { return space; };
    unsigned getNumIneqs() const { return numIneqs; }
    unsigned getNumEqs() const { return numEqs; }

  private:
    PresburgerSpace space;
    unsigned numIneqs, numEqs;
  };
  CountsSnapshot getCounts() const;
  void truncate(const CountsSnapshot &counts);

  /// Insert `num` identifiers of the specified kind at position `pos`.
  /// Positions are relative to the kind of identifier. The coefficient columns
  /// corresponding to the added identifiers are initialized to zero. Return the
  /// absolute column position (i.e., not relative to the kind of identifier)
  /// of the first added identifier.
  virtual unsigned insertId(IdKind kind, unsigned pos, unsigned num = 1);

  /// Append `num` identifiers of the specified kind after the last identifier.
  /// of that kind. Return the position of the first appended column relative to
  /// the kind of identifier. The coefficient columns corresponding to the added
  /// identifiers are initialized to zero.
  unsigned appendId(IdKind kind, unsigned num = 1);

  /// Adds an inequality (>= 0) from the coefficients specified in `inEq`.
  void addInequality(ArrayRef<int64_t> inEq);
  /// Adds an equality from the coefficients specified in `eq`.
  void addEquality(ArrayRef<int64_t> eq);

  /// Eliminate the `posB^th` local identifier, replacing every instance of it
  /// with the `posA^th` local identifier. This should be used when the two
  /// local variables are known to always take the same values.
  virtual void eliminateRedundantLocalId(unsigned posA, unsigned posB);

  /// Removes identifiers of the specified kind with the specified pos (or
  /// within the specified range) from the system. The specified location is
  /// relative to the first identifier of the specified kind.
  void removeId(IdKind kind, unsigned pos);
  virtual void removeIdRange(IdKind kind, unsigned idStart, unsigned idLimit);

  /// Removes the specified identifier from the system.
  void removeId(unsigned pos);

  void removeEquality(unsigned pos);
  void removeInequality(unsigned pos);

  /// Remove the (in)equalities at positions [start, end).
  void removeEqualityRange(unsigned start, unsigned end);
  void removeInequalityRange(unsigned start, unsigned end);

  /// Get the lexicographically minimum rational point satisfying the
  /// constraints. Returns an empty optional if the relation is empty or if
  /// the lexmin is unbounded. Symbols are not supported and will result in
  /// assert-failure. Note that Domain is minimized first, then range.
  MaybeOptimum<SmallVector<Fraction, 8>> findRationalLexMin() const;

  /// Same as above, but returns lexicographically minimal integer point.
  /// Note: this should be used only when the lexmin is really required.
  /// For a generic integer sampling operation, findIntegerSample is more
  /// robust and should be preferred. Note that Domain is minimized first, then
  /// range.
  MaybeOptimum<SmallVector<int64_t, 8>> findIntegerLexMin() const;

  /// Swap the posA^th identifier with the posB^th identifier.
  virtual void swapId(unsigned posA, unsigned posB);

  /// Removes all equalities and inequalities.
  void clearConstraints();

  /// Sets the `values.size()` identifiers starting at `po`s to the specified
  /// values and removes them.
  void setAndEliminate(unsigned pos, ArrayRef<int64_t> values);

  /// Replaces the contents of this IntegerRelation with `other`.
  virtual void clearAndCopyFrom(const IntegerRelation &other);

  /// Gather positions of all lower and upper bounds of the identifier at `pos`,
  /// and optionally any equalities on it. In addition, the bounds are to be
  /// independent of identifiers in position range [`offset`, `offset` + `num`).
  void
  getLowerAndUpperBoundIndices(unsigned pos,
                               SmallVectorImpl<unsigned> *lbIndices,
                               SmallVectorImpl<unsigned> *ubIndices,
                               SmallVectorImpl<unsigned> *eqIndices = nullptr,
                               unsigned offset = 0, unsigned num = 0) const;

  /// Checks for emptiness by performing variable elimination on all
  /// identifiers, running the GCD test on each equality constraint, and
  /// checking for invalid constraints. Returns true if the GCD test fails for
  /// any equality, or if any invalid constraints are discovered on any row.
  /// Returns false otherwise.
  bool isEmpty() const;

  /// Runs the GCD test on all equality constraints. Returns true if this test
  /// fails on any equality. Returns false otherwise.
  /// This test can be used to disprove the existence of a solution. If it
  /// returns true, no integer solution to the equality constraints can exist.
  bool isEmptyByGCDTest() const;

  /// Returns true if the set of constraints is found to have no solution,
  /// false if a solution exists. Uses the same algorithm as
  /// `findIntegerSample`.
  bool isIntegerEmpty() const;

  /// Returns a matrix where each row is a vector along which the polytope is
  /// bounded. The span of the returned vectors is guaranteed to contain all
  /// such vectors. The returned vectors are NOT guaranteed to be linearly
  /// independent. This function should not be called on empty sets.
  Matrix getBoundedDirections() const;

  /// Find an integer sample point satisfying the constraints using a
  /// branch and bound algorithm with generalized basis reduction, with some
  /// additional processing using Simplex for unbounded sets.
  ///
  /// Returns an integer sample point if one exists, or an empty Optional
  /// otherwise.
  Optional<SmallVector<int64_t, 8>> findIntegerSample() const;

  /// Compute an overapproximation of the number of integer points in the
  /// relation. Symbol ids are currently not supported. If the computed
  /// overapproximation is infinite, an empty optional is returned.
  Optional<uint64_t> computeVolume() const;

  /// Returns true if the given point satisfies the constraints, or false
  /// otherwise. Takes the values of all ids including locals.
  bool containsPoint(ArrayRef<int64_t> point) const;
  /// Given the values of non-local ids, return a satisfying assignment to the
  /// local if one exists, or an empty optional otherwise.
  Optional<SmallVector<int64_t, 8>>
  containsPointNoLocal(ArrayRef<int64_t> point) const;

  /// Find equality and pairs of inequality contraints identified by their
  /// position indices, using which an explicit representation for each local
  /// variable can be computed. The indices of the constraints are stored in
  /// `MaybeLocalRepr` struct. If no such pair can be found, the kind attribute
  /// in `MaybeLocalRepr` is set to None.
  ///
  /// The dividends of the explicit representations are stored in `dividends`
  /// and the denominators in `denominators`. If no explicit representation
  /// could be found for the `i^th` local identifier, `denominators[i]` is set
  /// to 0.
  void getLocalReprs(std::vector<SmallVector<int64_t, 8>> &dividends,
                     SmallVector<unsigned, 4> &denominators,
                     std::vector<MaybeLocalRepr> &repr) const;
  void getLocalReprs(std::vector<MaybeLocalRepr> &repr) const;
  void getLocalReprs(std::vector<SmallVector<int64_t, 8>> &dividends,
                     SmallVector<unsigned, 4> &denominators) const;

  /// The type of bound: equal, lower bound or upper bound.
  enum BoundType { EQ, LB, UB };

  /// Adds a constant bound for the specified identifier.
  void addBound(BoundType type, unsigned pos, int64_t value);

  /// Adds a constant bound for the specified expression.
  void addBound(BoundType type, ArrayRef<int64_t> expr, int64_t value);

  /// Adds a new local identifier as the floordiv of an affine function of other
  /// identifiers, the coefficients of which are provided in `dividend` and with
  /// respect to a positive constant `divisor`. Two constraints are added to the
  /// system to capture equivalence with the floordiv:
  /// q = dividend floordiv c    <=>   c*q <= dividend <= c*q + c - 1.
  void addLocalFloorDiv(ArrayRef<int64_t> dividend, int64_t divisor);

  /// Projects out (aka eliminates) `num` identifiers starting at position
  /// `pos`. The resulting constraint system is the shadow along the dimensions
  /// that still exist. This method may not always be integer exact.
  // TODO: deal with integer exactness when necessary - can return a value to
  // mark exactness for example.
  void projectOut(unsigned pos, unsigned num);
  inline void projectOut(unsigned pos) { return projectOut(pos, 1); }

  /// Tries to fold the specified identifier to a constant using a trivial
  /// equality detection; if successful, the constant is substituted for the
  /// identifier everywhere in the constraint system and then removed from the
  /// system.
  LogicalResult constantFoldId(unsigned pos);

  /// This method calls `constantFoldId` for the specified range of identifiers,
  /// `num` identifiers starting at position `pos`.
  void constantFoldIdRange(unsigned pos, unsigned num);

  /// Updates the constraints to be the smallest bounding (enclosing) box that
  /// contains the points of `this` set and that of `other`, with the symbols
  /// being treated specially. For each of the dimensions, the min of the lower
  /// bounds (symbolic) and the max of the upper bounds (symbolic) is computed
  /// to determine such a bounding box. `other` is expected to have the same
  /// dimensional identifiers as this constraint system (in the same order).
  ///
  /// E.g.:
  /// 1) this   = {0 <= d0 <= 127},
  ///    other  = {16 <= d0 <= 192},
  ///    output = {0 <= d0 <= 192}
  /// 2) this   = {s0 + 5 <= d0 <= s0 + 20},
  ///    other  = {s0 + 1 <= d0 <= s0 + 9},
  ///    output = {s0 + 1 <= d0 <= s0 + 20}
  /// 3) this   = {0 <= d0 <= 5, 1 <= d1 <= 9}
  ///    other  = {2 <= d0 <= 6, 5 <= d1 <= 15},
  ///    output = {0 <= d0 <= 6, 1 <= d1 <= 15}
  LogicalResult unionBoundingBox(const IntegerRelation &other);

  /// Returns the smallest known constant bound for the extent of the specified
  /// identifier (pos^th), i.e., the smallest known constant that is greater
  /// than or equal to 'exclusive upper bound' - 'lower bound' of the
  /// identifier. This constant bound is guaranteed to be non-negative. Returns
  /// None if it's not a constant. This method employs trivial (low complexity /
  /// cost) checks and detection. Symbolic identifiers are treated specially,
  /// i.e., it looks for constant differences between affine expressions
  /// involving only the symbolic identifiers. `lb` and `ub` (along with the
  /// `boundFloorDivisor`) are set to represent the lower and upper bound
  /// associated with the constant difference: `lb`, `ub` have the coefficients,
  /// and `boundFloorDivisor`, their divisor. `minLbPos` and `minUbPos` if
  /// non-null are set to the position of the constant lower bound and upper
  /// bound respectively (to the same if they are from an equality). Ex: if the
  /// lower bound is [(s0 + s2 - 1) floordiv 32] for a system with three
  /// symbolic identifiers, *lb = [1, 0, 1], lbDivisor = 32. See comments at
  /// function definition for examples.
  Optional<int64_t> getConstantBoundOnDimSize(
      unsigned pos, SmallVectorImpl<int64_t> *lb = nullptr,
      int64_t *boundFloorDivisor = nullptr,
      SmallVectorImpl<int64_t> *ub = nullptr, unsigned *minLbPos = nullptr,
      unsigned *minUbPos = nullptr) const;

  /// Returns the constant bound for the pos^th identifier if there is one;
  /// None otherwise.
  Optional<int64_t> getConstantBound(BoundType type, unsigned pos) const;

  /// Removes constraints that are independent of (i.e., do not have a
  /// coefficient) identifiers in the range [pos, pos + num).
  void removeIndependentConstraints(unsigned pos, unsigned num);

  /// Returns true if the set can be trivially detected as being
  /// hyper-rectangular on the specified contiguous set of identifiers.
  bool isHyperRectangular(unsigned pos, unsigned num) const;

  /// Removes duplicate constraints, trivially true constraints, and constraints
  /// that can be detected as redundant as a result of differing only in their
  /// constant term part. A constraint of the form <non-negative constant> >= 0
  /// is considered trivially true. This method is a linear time method on the
  /// constraints, does a single scan, and updates in place. It also normalizes
  /// constraints by their GCD and performs GCD tightening on inequalities.
  void removeTrivialRedundancy();

  /// A more expensive check than `removeTrivialRedundancy` to detect redundant
  /// inequalities.
  void removeRedundantInequalities();

  /// Removes redundant constraints using Simplex. Although the algorithm can
  /// theoretically take exponential time in the worst case (rare), it is known
  /// to perform much better in the average case. If V is the number of vertices
  /// in the polytope and C is the number of constraints, the algorithm takes
  /// O(VC) time.
  void removeRedundantConstraints();

  void removeDuplicateDivs();

  /// Converts identifiers of kind srcKind in the range [idStart, idLimit) to
  /// variables of kind dstKind. If `pos` is given, the variables are placed at
  /// position `pos` of dstKind, otherwise they are placed after all the other
  /// variables of kind dstKind. The internal ordering among the moved variables
  /// is preserved.
  void convertIdKind(IdKind srcKind, unsigned idStart, unsigned idLimit,
                     IdKind dstKind, unsigned pos);
  void convertIdKind(IdKind srcKind, unsigned idStart, unsigned idLimit,
                     IdKind dstKind) {
    convertIdKind(srcKind, idStart, idLimit, dstKind, getNumIdKind(dstKind));
  }
  void convertToLocal(IdKind kind, unsigned idStart, unsigned idLimit) {
    convertIdKind(kind, idStart, idLimit, IdKind::Local);
  }

  /// Adds additional local ids to the sets such that they both have the union
  /// of the local ids in each set, without changing the set of points that
  /// lie in `this` and `other`.
  ///
  /// While taking union, if a local id in `other` has a division representation
  /// which is a duplicate of division representation, of another local id, it
  /// is not added to the final union of local ids and is instead merged. The
  /// new ordering of local ids is:
  ///
  /// [Local ids of `this`] [Non-merged local ids of `other`]
  ///
  /// The relative ordering of local ids is same as before.
  ///
  /// After merging, if the `i^th` local variable in one set has a known
  /// division representation, then the `i^th` local variable in the other set
  /// either has the same division representation or no known division
  /// representation.
  ///
  /// The spaces of both relations should be compatible.
  ///
  /// Returns the number of non-merged local ids of `other`, i.e. the number of
  /// locals that have been added to `this`.
  unsigned mergeLocalIds(IntegerRelation &other);

  /// Check whether all local ids have a division representation.
  bool hasOnlyDivLocals() const;

  /// Changes the partition between dimensions and symbols. Depending on the new
  /// symbol count, either a chunk of dimensional identifiers immediately before
  /// the split become symbols, or some of the symbols immediately after the
  /// split become dimensions.
  void setDimSymbolSeparation(unsigned newSymbolCount) {
    space.setDimSymbolSeparation(newSymbolCount);
  }

  /// Return a set corresponding to all points in the domain of the relation.
  IntegerPolyhedron getDomainSet() const;

  /// Return a set corresponding to all points in the range of the relation.
  IntegerPolyhedron getRangeSet() const;

  /// Intersect the given `poly` with the domain in-place.
  ///
  /// Formally, let the relation `this` be R: A -> B and poly is C, then this
  /// operation modifies R to be (A intersection C) -> B.
  void intersectDomain(const IntegerPolyhedron &poly);

  /// Intersect the given `poly` with the range in-place.
  ///
  /// Formally, let the relation `this` be R: A -> B and poly is C, then this
  /// operation modifies R to be A -> (B intersection C).
  void intersectRange(const IntegerPolyhedron &poly);

  /// Invert the relation i.e., swap its domain and range.
  ///
  /// Formally, let the relation `this` be R: A -> B, then this operation
  /// modifies R to be B -> A.
  void inverse();

  /// Let the relation `this` be R1, and the relation `rel` be R2. Modifies R1
  /// to be the composition of R1 and R2: R1;R2.
  ///
  /// Formally, if R1: A -> B, and R2: B -> C, then this function returns a
  /// relation R3: A -> C such that a point (a, c) belongs to R3 iff there
  /// exists b such that (a, b) is in R1 and, (b, c) is in R2.
  void compose(const IntegerRelation &rel);

  /// Given a relation `rel`, apply the relation to the domain of this relation.
  ///
  /// R1: i -> j : (0 <= i < 2, j = i)
  /// R2: i -> k : (k = i floordiv 2)
  /// R3: k -> j : (0 <= k < 1, 2k <=  j <= 2k + 1)
  ///
  /// R1 = {(0, 0), (1, 1)}. R2 maps both 0 and 1 to 0.
  /// So R3 = {(0, 0), (0, 1)}.
  ///
  /// Formally, R1.applyDomain(R2) = R2.inverse().compose(R1).
  void applyDomain(const IntegerRelation &rel);

  /// Given a relation `rel`, apply the relation to the range of this relation.
  ///
  /// Formally, R1.applyRange(R2) is the same as R1.compose(R2) but we provide
  /// this for uniformity with `applyDomain`.
  void applyRange(const IntegerRelation &rel);

  void print(raw_ostream &os) const;
  void dump() const;

protected:
  /// Checks all rows of equality/inequality constraints for trivial
  /// contradictions (for example: 1 == 0, 0 >= 1), which may have surfaced
  /// after elimination. Returns true if an invalid constraint is found;
  /// false otherwise.
  bool hasInvalidConstraint() const;

  /// Returns the constant lower bound bound if isLower is true, and the upper
  /// bound if isLower is false.
  template <bool isLower>
  Optional<int64_t> computeConstantLowerOrUpperBound(unsigned pos);

  /// Eliminates a single identifier at `position` from equality and inequality
  /// constraints. Returns `success` if the identifier was eliminated, and
  /// `failure` otherwise.
  inline LogicalResult gaussianEliminateId(unsigned position) {
    return success(gaussianEliminateIds(position, position + 1) == 1);
  }

  /// Removes local variables using equalities. Each equality is checked if it
  /// can be reduced to the form: `e = affine-expr`, where `e` is a local
  /// variable and `affine-expr` is an affine expression not containing `e`.
  /// If an equality satisfies this form, the local variable is replaced in
  /// each constraint and then removed. The equality used to replace this local
  /// variable is also removed.
  void removeRedundantLocalVars();

  /// Eliminates identifiers from equality and inequality constraints
  /// in column range [posStart, posLimit).
  /// Returns the number of variables eliminated.
  unsigned gaussianEliminateIds(unsigned posStart, unsigned posLimit);

  /// Eliminates the identifier at the specified position using Fourier-Motzkin
  /// variable elimination, but uses Gaussian elimination if there is an
  /// equality involving that identifier. If the result of the elimination is
  /// integer exact, `*isResultIntegerExact` is set to true. If `darkShadow` is
  /// set to true, a potential under approximation (subset) of the rational
  /// shadow / exact integer shadow is computed.
  // See implementation comments for more details.
  virtual void fourierMotzkinEliminate(unsigned pos, bool darkShadow = false,
                                       bool *isResultIntegerExact = nullptr);

  /// Tightens inequalities given that we are dealing with integer spaces. This
  /// is similar to the GCD test but applied to inequalities. The constant term
  /// can be reduced to the preceding multiple of the GCD of the coefficients,
  /// i.e.,
  ///  64*i - 100 >= 0  =>  64*i - 128 >= 0 (since 'i' is an integer). This is a
  /// fast method (linear in the number of coefficients).
  void gcdTightenInequalities();

  /// Normalized each constraints by the GCD of its coefficients.
  void normalizeConstraintsByGCD();

  /// Searches for a constraint with a non-zero coefficient at `colIdx` in
  /// equality (isEq=true) or inequality (isEq=false) constraints.
  /// Returns true and sets row found in search in `rowIdx`, false otherwise.
  bool findConstraintWithNonZeroAt(unsigned colIdx, bool isEq,
                                   unsigned *rowIdx) const;

  /// Returns true if the pos^th column is all zero for both inequalities and
  /// equalities.
  bool isColZero(unsigned pos) const;

  /// Returns false if the fields corresponding to various identifier counts, or
  /// equality/inequality buffer sizes aren't consistent; true otherwise. This
  /// is meant to be used within an assert internally.
  virtual bool hasConsistentState() const;

  /// Prints the number of constraints, dimensions, symbols and locals in the
  /// IntegerRelation.
  virtual void printSpace(raw_ostream &os) const;

  /// Removes identifiers in the column range [idStart, idLimit), and copies any
  /// remaining valid data into place, updates member variables, and resizes
  /// arrays as needed.
  void removeIdRange(unsigned idStart, unsigned idLimit);

  /// Truncate the ids of the specified kind to the specified number by dropping
  /// some ids at the end. `num` must be less than the current number.
  void truncateIdKind(IdKind kind, unsigned num);

  /// Truncate the ids to the number in the space of the specified
  /// CountsSnapshot.
  void truncateIdKind(IdKind kind, const CountsSnapshot &counts);

  /// A parameter that controls detection of an unrealistic number of
  /// constraints. If the number of constraints is this many times the number of
  /// variables, we consider such a system out of line with the intended use
  /// case of IntegerRelation.
  // The rationale for 32 is that in the typical simplest of cases, an
  // identifier is expected to have one lower bound and one upper bound
  // constraint. With a level of tiling or a connection to another identifier
  // through a div or mod, an extra pair of bounds gets added. As a limit, we
  // don't expect an identifier to have more than 32 lower/upper/equality
  // constraints. This is conservatively set low and can be raised if needed.
  constexpr static unsigned kExplosionFactor = 32;

  PresburgerSpace space;

  /// Coefficients of affine equalities (in == 0 form).
  Matrix equalities;

  /// Coefficients of affine inequalities (in >= 0 form).
  Matrix inequalities;
};

struct SymbolicLexMin;

/// An IntegerPolyhedron represents the set of points from a PresburgerSpace
/// that satisfy a list of affine constraints. Affine constraints can be
/// inequalities or equalities in the form:
///
/// Inequality: c_0*x_0 + c_1*x_1 + .... + c_{n-1}*x_{n-1} + c_n >= 0
/// Equality  : c_0*x_0 + c_1*x_1 + .... + c_{n-1}*x_{n-1} + c_n == 0
///
/// where c_0, c_1, ..., c_n are integers and n is the total number of
/// identifiers in the space.
///
/// An IntegerPolyhedron is similar to an IntegerRelation but it does not make a
/// distinction between Domain and Range identifiers. Internally,
/// IntegerPolyhedron is implemented as a IntegerRelation with zero domain ids.
///
/// Since IntegerPolyhedron does not make a distinction between kinds of
/// dimensions, IdKind::SetDim should be used to refer to dimension identifiers.
class IntegerPolyhedron : public IntegerRelation {
public:
  /// Constructs a set reserving memory for the specified number
  /// of constraints and identifiers.
  IntegerPolyhedron(unsigned numReservedInequalities,
                    unsigned numReservedEqualities, unsigned numReservedCols,
                    const PresburgerSpace &space)
      : IntegerRelation(numReservedInequalities, numReservedEqualities,
                        numReservedCols, space) {
    assert(space.getNumDomainIds() == 0 &&
           "Number of domain id's should be zero in Set kind space.");
  }

  /// Constructs a relation with the specified number of dimensions and
  /// symbols.
  explicit IntegerPolyhedron(const PresburgerSpace &space)
      : IntegerPolyhedron(/*numReservedInequalities=*/0,
                          /*numReservedEqualities=*/0,
                          /*numReservedCols=*/space.getNumIds() + 1, space) {}

  /// Construct a set from an IntegerRelation. The relation should have
  /// no domain ids.
  explicit IntegerPolyhedron(const IntegerRelation &rel)
      : IntegerRelation(rel) {
    assert(space.getNumDomainIds() == 0 &&
           "Number of domain id's should be zero in Set kind space.");
  }

  /// Construct a set from an IntegerRelation, but instead of creating a copy,
  /// use move constructor. The relation should have no domain ids.
  explicit IntegerPolyhedron(IntegerRelation &&rel) : IntegerRelation(rel) {
    assert(space.getNumDomainIds() == 0 &&
           "Number of domain id's should be zero in Set kind space.");
  }

  /// Return a system with no constraints, i.e., one which is satisfied by all
  /// points.
  static IntegerPolyhedron getUniverse(const PresburgerSpace &space) {
    return IntegerPolyhedron(space);
  }

  /// Return the kind of this IntegerRelation.
  Kind getKind() const override { return Kind::IntegerPolyhedron; }

  static bool classof(const IntegerRelation *cst) {
    return cst->getKind() == Kind::IntegerPolyhedron;
  }

  // Clones this object.
  std::unique_ptr<IntegerPolyhedron> clone() const;

  /// Insert `num` identifiers of the specified kind at position `pos`.
  /// Positions are relative to the kind of identifier. Return the absolute
  /// column position (i.e., not relative to the kind of identifier) of the
  /// first added identifier.
  unsigned insertId(IdKind kind, unsigned pos, unsigned num = 1) override;

  /// Compute an equivalent representation of the same set, such that all local
  /// ids have division representations. This representation may involve
  /// local ids that correspond to divisions, and may also be a union of convex
  /// disjuncts.
  PresburgerSet computeReprWithOnlyDivLocals() const;

  /// Compute the symbolic integer lexmin of the polyhedron.
  /// This finds, for every assignment to the symbols, the lexicographically
  /// minimum value attained by the dimensions. For example, the symbolic lexmin
  /// of the set
  ///
  /// (x, y)[a, b, c] : (a <= x, b <= x, x <= c)
  ///
  /// can be written as
  ///
  /// x = a if b <= a, a <= c
  /// x = b if a <  b, b <= c
  ///
  /// This function is stored in the `lexmin` function in the result.
  /// Some assignments to the symbols might make the set empty.
  /// Such points are not part of the function's domain.
  /// In the above example, this happens when max(a, b) > c.
  ///
  /// For some values of the symbols, the lexmin may be unbounded.
  /// `SymbolicLexMin` stores these parts of the symbolic domain in a separate
  /// `PresburgerSet`, `unboundedDomain`.
  SymbolicLexMin findSymbolicIntegerLexMin() const;
};

} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_INTEGERRELATION_H
