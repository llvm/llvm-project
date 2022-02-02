//===- Set.h - MLIR PresburgerSet Class -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A class to represent unions of IntegerPolyhedrons.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_PRESBURGERSET_H
#define MLIR_ANALYSIS_PRESBURGER_PRESBURGERSET_H

#include "mlir/Analysis/Presburger/IntegerPolyhedron.h"

namespace mlir {

/// This class can represent a union of IntegerPolyhedrons, with support for
/// union, intersection, subtraction and complement operations, as well as
/// sampling.
///
/// The IntegerPolyhedrons (Polys) are stored in a vector, and the set
/// represents the union of these Polys. An empty list corresponds to the empty
/// set.
///
/// Note that there are no invariants guaranteed on the list of Poly other than
/// that they are all in the same space, i.e., they all have the same number of
/// dimensions and symbols. For example, the Polys may overlap each other.
class PresburgerSet {
public:
  explicit PresburgerSet(const IntegerPolyhedron &poly);

  /// Return the number of Polys in the union.
  unsigned getNumPolys() const;

  /// Return the number of real dimensions.
  unsigned getNumDimIds() const;

  /// Return the number of symbolic dimensions.
  unsigned getNumSymbolIds() const;

  /// Return a reference to the list of IntegerPolyhedrons.
  ArrayRef<IntegerPolyhedron> getAllIntegerPolyhedron() const;

  /// Return the IntegerPolyhedron at the specified index.
  const IntegerPolyhedron &getIntegerPolyhedron(unsigned index) const;

  /// Mutate this set, turning it into the union of this set and the given
  /// IntegerPolyhedron.
  void unionPolyInPlace(const IntegerPolyhedron &poly);

  /// Mutate this set, turning it into the union of this set and the given set.
  void unionSetInPlace(const PresburgerSet &set);

  /// Return the union of this set and the given set.
  PresburgerSet unionSet(const PresburgerSet &set) const;

  /// Return the intersection of this set and the given set.
  PresburgerSet intersect(const PresburgerSet &set) const;

  /// Return true if the set contains the given point, and false otherwise.
  bool containsPoint(ArrayRef<int64_t> point) const;

  /// Print the set's internal state.
  void print(raw_ostream &os) const;
  void dump() const;

  /// Return the complement of this set. All local variables in the set must
  /// correspond to floor divisions.
  PresburgerSet complement() const;

  /// Return the set difference of this set and the given set, i.e.,
  /// return `this \ set`. All local variables in `set` must correspond
  /// to floor divisions, but local variables in `this` need not correspond to
  /// divisions.
  PresburgerSet subtract(const PresburgerSet &set) const;

  /// Return true if this set is a subset of the given set, and false otherwise.
  bool isSubsetOf(const PresburgerSet &set) const;

  /// Return true if this set is equal to the given set, and false otherwise.
  /// All local variables in both sets must correspond to floor divisions.
  bool isEqual(const PresburgerSet &set) const;

  /// Return a universe set of the specified type that contains all points.
  static PresburgerSet getUniverse(unsigned numDims = 0,
                                   unsigned numSymbols = 0);
  /// Return an empty set of the specified type that contains no points.
  static PresburgerSet getEmptySet(unsigned numDims = 0,
                                   unsigned numSymbols = 0);

  /// Return true if all the sets in the union are known to be integer empty
  /// false otherwise.
  bool isIntegerEmpty() const;

  /// Find an integer sample from the given set. This should not be called if
  /// any of the Polys in the union are unbounded.
  bool findIntegerSample(SmallVectorImpl<int64_t> &sample);

  /// Simplifies the representation of a PresburgerSet.
  ///
  /// In particular, removes all Polys which are subsets of other Polys in the
  /// union.
  PresburgerSet coalesce() const;

private:
  /// Construct an empty PresburgerSet.
  PresburgerSet(unsigned numDims = 0, unsigned numSymbols = 0)
      : numDims(numDims), numSymbols(numSymbols) {}

  /// Return the set difference poly \ set.
  static PresburgerSet getSetDifference(IntegerPolyhedron poly,
                                        const PresburgerSet &set);

  /// Number of identifiers corresponding to real dimensions.
  unsigned numDims;

  /// Number of symbolic dimensions, unknown but constant for analysis, as in
  /// IntegerPolyhedron.
  unsigned numSymbols;

  /// The list of integerPolyhedrons that this set is the union of.
  SmallVector<IntegerPolyhedron, 2> integerPolyhedrons;
};

} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_PRESBURGERSET_H
