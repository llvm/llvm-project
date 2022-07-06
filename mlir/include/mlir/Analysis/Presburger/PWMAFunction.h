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
// tuple being an affine expression in the vars of the IntegerPolyhedron.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_PWMAFUNCTION_H
#define MLIR_ANALYSIS_PRESBURGER_PWMAFUNCTION_H

#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"

namespace mlir {
namespace presburger {

/// This class represents a multi-affine function whose domain is given by an
/// IntegerPolyhedron. This can be thought of as an IntegerPolyhedron with a
/// tuple of integer values attached to every point in the polyhedron, with the
/// value of each element of the tuple given by an affine expression in the vars
/// of the polyhedron. For example we could have the domain
///
/// (x, y) : (x >= 5, y >= x)
///
/// and a tuple of three integers defined at every point in the polyhedron:
///
/// (x, y) -> (x + 2, 2*x - 3y + 5, 2*x + y).
///
/// In this way every point in the polyhedron has a tuple of integers associated
/// with it. If the integer polyhedron has local vars, then the output
/// expressions can use them as well. The output expressions are represented as
/// a matrix with one row for every element in the output vector one column for
/// each var, and an extra column at the end for the constant term.
///
/// Checking equality of two such functions is supported, as well as finding the
/// value of the function at a specified point.
class MultiAffineFunction {
public:
  MultiAffineFunction(const IntegerPolyhedron &domain, const Matrix &output)
      : domainSet(domain), output(output) {}
  MultiAffineFunction(const Matrix &output, const PresburgerSpace &space)
      : domainSet(space), output(output) {}

  unsigned getNumInputs() const { return domainSet.getNumDimAndSymbolVars(); }
  unsigned getNumOutputs() const { return output.getNumRows(); }
  bool isConsistent() const {
    return output.getNumColumns() == domainSet.getNumVars() + 1;
  }

  /// Get the space of the input domain of this function.
  const PresburgerSpace &getDomainSpace() const { return domainSet.getSpace(); }
  /// Get the input domain of this function.
  const IntegerPolyhedron &getDomain() const { return domainSet; }
  /// Get a matrix with each row representing row^th output expression.
  const Matrix &getOutputMatrix() const { return output; }
  /// Get the `i^th` output expression.
  ArrayRef<int64_t> getOutputExpr(unsigned i) const { return output.getRow(i); }

  /// Insert `num` variables of the specified kind at position `pos`.
  /// Positions are relative to the kind of variable. The coefficient columns
  /// corresponding to the added variables are initialized to zero. Return the
  /// absolute column position (i.e., not relative to the kind of variable)
  /// of the first added variable.
  unsigned insertVar(VarKind kind, unsigned pos, unsigned num = 1);

  /// Remove the specified range of vars.
  void removeVarRange(VarKind kind, unsigned varStart, unsigned varLimit);

  /// Given a MAF `other`, merges local variables such that both funcitons
  /// have union of local vars, without changing the set of points in domain or
  /// the output.
  void mergeLocalVars(MultiAffineFunction &other);

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
  Optional<SmallVector<int64_t, 8>> valueAt(ArrayRef<int64_t> point) const;

  /// Truncate the output dimensions to the first `count` dimensions.
  ///
  /// TODO: refactor so that this can be accomplished through removeVarRange.
  void truncateOutput(unsigned count);

  void print(raw_ostream &os) const;
  void dump() const;

private:
  /// The IntegerPolyhedron representing the domain over which the function is
  /// defined.
  IntegerPolyhedron domainSet;

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
/// As in PresburgerSets, the input vars are partitioned into dimension vars and
/// symbolic vars.
///
/// Support is provided to compare equality of two such functions as well as
/// finding the value of the function at a point.
class PWMAFunction {
public:
  PWMAFunction(const PresburgerSpace &space, unsigned numOutputs)
      : space(space), numOutputs(numOutputs) {
    assert(space.getNumDomainVars() == 0 &&
           "Set type space should have zero domain vars.");
    assert(space.getNumLocalVars() == 0 &&
           "PWMAFunction cannot have local vars.");
    assert(numOutputs >= 1 && "The function must output something!");
  }

  const PresburgerSpace &getSpace() const { return space; }

  void addPiece(const MultiAffineFunction &piece);
  void addPiece(const IntegerPolyhedron &domain, const Matrix &output);
  void addPiece(const PresburgerSet &domain, const Matrix &output);

  const MultiAffineFunction &getPiece(unsigned i) const { return pieces[i]; }
  unsigned getNumPieces() const { return pieces.size(); }
  unsigned getNumOutputs() const { return numOutputs; }
  unsigned getNumInputs() const { return space.getNumVars(); }
  MultiAffineFunction &getPiece(unsigned i) { return pieces[i]; }

  /// Return the domain of this piece-wise MultiAffineFunction. This is the
  /// union of the domains of all the pieces.
  PresburgerSet getDomain() const;

  /// Return the value at the specified point and an empty optional if the
  /// point does not lie in the domain.
  Optional<SmallVector<int64_t, 8>> valueAt(ArrayRef<int64_t> point) const;

  /// Return whether `this` and `other` are equal as PWMAFunctions, i.e. whether
  /// they have the same dimensions, the same domain and they take the same
  /// value at every point in the domain.
  bool isEqual(const PWMAFunction &other) const;

  /// Truncate the output dimensions to the first `count` dimensions.
  ///
  /// TODO: refactor so that this can be accomplished through removeVarRange.
  void truncateOutput(unsigned count);

  /// Return a function defined on the union of the domains of this and func,
  /// such that when only one of the functions is defined, it outputs the same
  /// as that function, and if both are defined, it outputs the lexmax/lexmin of
  /// the two outputs. On points where neither function is defined, the returned
  /// function is not defined either.
  ///
  /// Currently this does not support PWMAFunctions which have pieces containing
  /// local variables.
  /// TODO: Support local variables in peices.
  PWMAFunction unionLexMin(const PWMAFunction &func);
  PWMAFunction unionLexMax(const PWMAFunction &func);

  void print(raw_ostream &os) const;
  void dump() const;

private:
  /// Return a function defined on the union of the domains of `this` and
  /// `func`, such that when only one of the functions is defined, it outputs
  /// the same as that function, and if neither is defined, the returned
  /// function is not defined either.
  ///
  /// The provided `tiebreak` function determines which of the two functions'
  /// output should be used on inputs where both the functions are defined. More
  /// precisely, given two `MultiAffineFunction`s `mafA` and `mafB`, `tiebreak`
  /// returns the subset of the intersection of the two functions' domains where
  /// the output of `mafA` should be used.
  ///
  /// The PresburgerSet returned by `tiebreak` should be disjoint.
  /// TODO: Remove this constraint of returning disjoint set.
  PWMAFunction
  unionFunction(const PWMAFunction &func,
                llvm::function_ref<PresburgerSet(MultiAffineFunction mafA,
                                                 MultiAffineFunction mafB)>
                    tiebreak) const;

  PresburgerSpace space;

  /// The list of pieces in this piece-wise MultiAffineFunction.
  SmallVector<MultiAffineFunction, 4> pieces;

  /// The number of output vars.
  unsigned numOutputs;
};

} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_PWMAFUNCTION_H
