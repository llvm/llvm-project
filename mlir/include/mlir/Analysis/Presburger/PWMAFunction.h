//===- PWMAFunction.h - MLIR PWMAFunction Class------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Support for piece-wise multi-affine functions. These are functions that are
// defined on a domain that is a union of IntegerPolyhedrons, and on each domain
// the value of the function is a tuple of integers, with each value in the
// tuple being an affine expression in the ids of the IntegerPolyhedron.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_PWMAFUNCTION_H
#define MLIR_ANALYSIS_PRESBURGER_PWMAFUNCTION_H

#include "mlir/Analysis/Presburger/IntegerPolyhedron.h"
#include "mlir/Analysis/Presburger/PresburgerSet.h"

namespace mlir {

/// This class represents a multi-affine function whose domain is given by an
/// IntegerPolyhedron. This can be thought of as an IntegerPolyhedron with a
/// tuple of integer values attached to every point in the polyhedron, with the
/// value of each element of the tuple given by an affine expression in the ids
/// of the polyhedron. For example we could have the domain
///
/// (x, y) : (x >= 5, y >= x)
///
/// and a tuple of three integers defined at every point in the polyhedron:
///
/// (x, y) -> (x + 2, 2*x - 3y + 5, 2*x + y).
///
/// In this way every point in the polyhedron has a tuple of integers associated
/// with it. If the integer polyhedron has local ids, then the output
/// expressions can use them as well. The output expressions are represented as
/// a matrix with one row for every element in the output vector one column for
/// each id, and an extra column at the end for the constant term.
///
/// Checking equality of two such functions is supported, as well as finding the
/// value of the function at a specified point. Note that local ids in the
/// domain are not yet supported for finding the value at a point.
class MultiAffineFunction : protected IntegerPolyhedron {
public:
  /// We use protected inheritance to avoid inheriting the whole public
  /// interface of IntegerPolyhedron. These using declarations explicitly make
  /// only the relevant functions part of the public interface.
  using IntegerPolyhedron::getNumDimAndSymbolIds;
  using IntegerPolyhedron::getNumDimIds;
  using IntegerPolyhedron::getNumIds;
  using IntegerPolyhedron::getNumLocalIds;
  using IntegerPolyhedron::getNumSymbolIds;

  MultiAffineFunction(const IntegerPolyhedron &domain, const Matrix &output)
      : IntegerPolyhedron(domain), output(output) {}
  MultiAffineFunction(const Matrix &output, unsigned numDims,
                      unsigned numSymbols = 0, unsigned numLocals = 0)
      : IntegerPolyhedron(numDims, numSymbols, numLocals), output(output) {}

  ~MultiAffineFunction() override = default;
  Kind getKind() const override { return Kind::MultiAffineFunction; }
  bool classof(const IntegerPolyhedron *poly) const {
    return poly->getKind() == Kind::MultiAffineFunction;
  }

  unsigned getNumInputs() const { return getNumDimAndSymbolIds(); }
  unsigned getNumOutputs() const { return output.getNumRows(); }
  bool isConsistent() const { return output.getNumColumns() == numIds + 1; }
  const IntegerPolyhedron &getDomain() const { return *this; }

  bool hasCompatibleDimensions(const MultiAffineFunction &f) const;

  /// Insert `num` identifiers of the specified kind at position `pos`.
  /// Positions are relative to the kind of identifier. The coefficient columns
  /// corresponding to the added identifiers are initialized to zero. Return the
  /// absolute column position (i.e., not relative to the kind of identifier)
  /// of the first added identifier.
  unsigned insertId(IdKind kind, unsigned pos, unsigned num = 1) override;

  /// Swap the posA^th identifier with the posB^th identifier.
  void swapId(unsigned posA, unsigned posB) override;

  /// Remove the specified range of ids.
  void removeIdRange(unsigned idStart, unsigned idLimit) override;

  /// Eliminate the `posB^th` local identifier, replacing every instance of it
  /// with the `posA^th` local identifier. This should be used when the two
  /// local variables are known to always take the same values.
  void eliminateRedundantLocalId(unsigned posA, unsigned posB) override;

  /// Return whether the outputs of `this` and `other` agree wherever both
  /// functions are defined, i.e., the outputs should be equal for all points in
  /// the intersection of the domains.
  bool isEqualWhereDomainsOverlap(MultiAffineFunction other) const;

  /// Return whether the `this` and `other` are equal. This is the case if
  /// they lie in the same space, i.e. have the same dimensions, and their
  /// domains are identical and their outputs are equal on their domain.
  bool isEqual(const MultiAffineFunction &other) const;

  /// Get the value of the function at the specified point. If the point lies
  /// outside the domain, an empty optional is returned.
  ///
  /// Note: domains with local ids are not yet supported, and will assert-fail.
  Optional<SmallVector<int64_t, 8>> valueAt(ArrayRef<int64_t> point) const;

  void print(raw_ostream &os) const;

  void dump() const;

private:
  /// The function's output is a tuple of integers, with the ith element of the
  /// tuple defined by the affine expression given by the ith row of this output
  /// matrix.
  Matrix output;
};

/// This class represents a piece-wise MultiAffineFunction. This can be thought
/// of as a list of MultiAffineFunction with disjoint domains, with each having
/// their own affine expressions for their output tuples. For example, we could
/// have a function with two input variables (x, y), defined as
///
/// f(x, y) = (2*x + y, y - 4)  if x >= 0, y >= 0
///         = (-2*x + y, y + 4) if x < 0,  y < 0
///         = (4, 1)            if x < 0,  y >= 0
///
/// Note that the domains all have to be *disjoint*. Otherwise, the behaviour of
/// this class is undefined. The domains need not cover all possible points;
/// this represents a partial function and so could be undefined at some points.
///
/// As in PresburgerSets, the input ids are partitioned into dimension ids and
/// symbolic ids.
///
/// Support is provided to compare equality of two such functions as well as
/// finding the value of the function at a point. Note that local ids in the
/// piece are not supported for the latter.
class PWMAFunction {
public:
  PWMAFunction(unsigned numDims, unsigned numSymbols, unsigned numOutputs)
      : numDims(numDims), numSymbols(numSymbols), numOutputs(numOutputs) {
    assert(numOutputs >= 1 && "The function must output something!");
  }

  void addPiece(const MultiAffineFunction &piece);
  void addPiece(const IntegerPolyhedron &domain, const Matrix &output);

  const MultiAffineFunction &getPiece(unsigned i) const { return pieces[i]; }
  unsigned getNumPieces() const { return pieces.size(); }
  unsigned getNumOutputs() const { return numOutputs; }
  unsigned getNumInputs() const { return numDims + numSymbols; }
  unsigned getNumDimIds() const { return numDims; }
  unsigned getNumSymbolIds() const { return numSymbols; }
  MultiAffineFunction &getPiece(unsigned i) { return pieces[i]; }

  /// Return the domain of this piece-wise MultiAffineFunction. This is the
  /// union of the domains of all the pieces.
  PresburgerSet getDomain() const;

  /// Check whether the `this` and the given function have compatible
  /// dimensions, i.e., the same number of dimension inputs, symbol inputs, and
  /// outputs.
  bool hasCompatibleDimensions(const MultiAffineFunction &f) const;
  bool hasCompatibleDimensions(const PWMAFunction &f) const;

  /// Return the value at the specified point and an empty optional if the
  /// point does not lie in the domain.
  ///
  /// Note: domains with local ids are not yet supported, and will assert-fail.
  Optional<SmallVector<int64_t, 8>> valueAt(ArrayRef<int64_t> point) const;

  /// Return whether `this` and `other` are equal as PWMAFunctions, i.e. whether
  /// they have the same dimensions, the same domain and they take the same
  /// value at every point in the domain.
  bool isEqual(const PWMAFunction &other) const;

  void print(raw_ostream &os) const;
  void dump() const;

private:
  /// The list of pieces in this piece-wise MultiAffineFunction.
  SmallVector<MultiAffineFunction, 4> pieces;

  /// The number of dimensions ids in the domains.
  unsigned numDims;
  /// The number of symbol ids in the domains.
  unsigned numSymbols;
  /// The number of output ids.
  unsigned numOutputs;
};

} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_PWMAFUNCTION_H
