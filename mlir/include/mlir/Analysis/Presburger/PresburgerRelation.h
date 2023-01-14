//===- PresburgerRelation.h - MLIR PresburgerRelation Class -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A class to represent unions of IntegerRelations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_PRESBURGERRELATION_H
#define MLIR_ANALYSIS_PRESBURGER_PRESBURGERRELATION_H

#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include <optional>

namespace mlir {
namespace presburger {

/// The SetCoalescer class contains all functionality concerning the coalesce
/// heuristic. It is built from a `PresburgerRelation` and has the `coalesce()`
/// function as its main API.
class SetCoalescer;

/// A PresburgerRelation represents a union of IntegerRelations that live in
/// the same PresburgerSpace with support for union, intersection, subtraction,
/// and complement operations, as well as sampling.
///
/// The IntegerRelations (disjuncts) are stored in a vector, and the set
/// represents the union of these relations. An empty list corresponds to
/// the empty set.
///
/// Note that there are no invariants guaranteed on the list of disjuncts
/// other than that they are all in the same PresburgerSpace. For example, the
/// relations may overlap with each other.
class PresburgerRelation {
public:
  /// Return a universe set of the specified type that contains all points.
  static PresburgerRelation getUniverse(const PresburgerSpace &space);

  /// Return an empty set of the specified type that contains no points.
  static PresburgerRelation getEmpty(const PresburgerSpace &space);

  explicit PresburgerRelation(const IntegerRelation &disjunct);

  unsigned getNumDomainVars() const { return space.getNumDomainVars(); }
  unsigned getNumRangeVars() const { return space.getNumRangeVars(); }
  unsigned getNumSymbolVars() const { return space.getNumSymbolVars(); }
  unsigned getNumLocalVars() const { return space.getNumLocalVars(); }
  unsigned getNumVars() const { return space.getNumVars(); }

  /// Return the number of disjuncts in the union.
  unsigned getNumDisjuncts() const;

  const PresburgerSpace &getSpace() const { return space; }

  /// Set the space to `oSpace`. `oSpace` should not contain any local ids.
  /// `oSpace` need not have the same number of ids as the current space;
  /// it could have more or less. If it has less, the extra ids become
  /// locals of the disjuncts. It can also have more, in which case the
  /// disjuncts will have fewer locals. If its total number of ids
  /// exceeds that of some disjunct, an assert failure will occur.
  void setSpace(const PresburgerSpace &oSpace);

  /// Return a reference to the list of disjuncts.
  ArrayRef<IntegerRelation> getAllDisjuncts() const;

  /// Return the disjunct at the specified index.
  const IntegerRelation &getDisjunct(unsigned index) const;

  /// Mutate this set, turning it into the union of this set and the given
  /// disjunct.
  void unionInPlace(const IntegerRelation &disjunct);

  /// Mutate this set, turning it into the union of this set and the given set.
  void unionInPlace(const PresburgerRelation &set);

  /// Return the union of this set and the given set.
  PresburgerRelation unionSet(const PresburgerRelation &set) const;

  /// Return the intersection of this set and the given set.
  PresburgerRelation intersect(const PresburgerRelation &set) const;

  /// Return true if the set contains the given point, and false otherwise.
  bool containsPoint(ArrayRef<MPInt> point) const;
  bool containsPoint(ArrayRef<int64_t> point) const {
    return containsPoint(getMPIntVec(point));
  }

  /// Return the complement of this set. All local variables in the set must
  /// correspond to floor divisions.
  PresburgerRelation complement() const;

  /// Return the set difference of this set and the given set, i.e.,
  /// return `this \ set`. All local variables in `set` must correspond
  /// to floor divisions, but local variables in `this` need not correspond to
  /// divisions.
  PresburgerRelation subtract(const PresburgerRelation &set) const;

  /// Return true if this set is a subset of the given set, and false otherwise.
  bool isSubsetOf(const PresburgerRelation &set) const;

  /// Return true if this set is equal to the given set, and false otherwise.
  /// All local variables in both sets must correspond to floor divisions.
  bool isEqual(const PresburgerRelation &set) const;

  /// Return true if all the sets in the union are known to be integer empty
  /// false otherwise.
  bool isIntegerEmpty() const;

  /// Find an integer sample from the given set. This should not be called if
  /// any of the disjuncts in the union are unbounded.
  bool findIntegerSample(SmallVectorImpl<MPInt> &sample);

  /// Compute an overapproximation of the number of integer points in the
  /// disjunct. Symbol vars are currently not supported. If the computed
  /// overapproximation is infinite, an empty optional is returned.
  ///
  /// This currently just sums up the overapproximations of the volumes of the
  /// disjuncts, so the approximation might be far from the true volume in the
  /// case when there is a lot of overlap between disjuncts.
  Optional<MPInt> computeVolume() const;

  /// Simplifies the representation of a PresburgerRelation.
  ///
  /// In particular, removes all disjuncts which are subsets of other
  /// disjuncts in the union.
  PresburgerRelation coalesce() const;

  /// Check whether all local ids in all disjuncts have a div representation.
  bool hasOnlyDivLocals() const;

  /// Compute an equivalent representation of the same relation, such that all
  /// local ids in all disjuncts have division representations. This
  /// representation may involve local ids that correspond to divisions, and may
  /// also be a union of convex disjuncts.
  PresburgerRelation computeReprWithOnlyDivLocals() const;

  /// Print the set's internal state.
  void print(raw_ostream &os) const;
  void dump() const;

protected:
  /// Construct an empty PresburgerRelation with the specified number of
  /// dimension and symbols.
  explicit PresburgerRelation(const PresburgerSpace &space) : space(space) {
    assert(space.getNumLocalVars() == 0 &&
           "PresburgerRelation cannot have local vars.");
  }

  PresburgerSpace space;

  /// The list of disjuncts that this set is the union of.
  SmallVector<IntegerRelation, 2> disjuncts;

  friend class SetCoalescer;
};

class PresburgerSet : public PresburgerRelation {
public:
  /// Return a universe set of the specified type that contains all points.
  static PresburgerSet getUniverse(const PresburgerSpace &space);

  /// Return an empty set of the specified type that contains no points.
  static PresburgerSet getEmpty(const PresburgerSpace &space);

  /// Create a set from a relation.
  explicit PresburgerSet(const IntegerPolyhedron &disjunct);
  explicit PresburgerSet(const PresburgerRelation &set);

  /// These operations are the same as the ones in PresburgeRelation, they just
  /// forward the arguement and return the result as a set instead of a
  /// relation.
  PresburgerSet unionSet(const PresburgerRelation &set) const;
  PresburgerSet intersect(const PresburgerRelation &set) const;
  PresburgerSet complement() const;
  PresburgerSet subtract(const PresburgerRelation &set) const;
  PresburgerSet coalesce() const;

protected:
  /// Construct an empty PresburgerRelation with the specified number of
  /// dimension and symbols.
  explicit PresburgerSet(const PresburgerSpace &space)
      : PresburgerRelation(space) {
    assert(space.getNumDomainVars() == 0 &&
           "Set type cannot have domain vars.");
    assert(space.getNumLocalVars() == 0 &&
           "PresburgerRelation cannot have local vars.");
  }
};

} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_PRESBURGERRELATION_H
