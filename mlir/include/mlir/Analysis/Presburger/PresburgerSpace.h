//===- PresburgerSpace.h - MLIR PresburgerSpace Class -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Classes representing space information like number of identifiers and kind of
// identifiers.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_PRESBURGERSPACE_H
#define MLIR_ANALYSIS_PRESBURGER_PRESBURGERSPACE_H

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace presburger {

class PresburgerLocalSpace;

/// Kind of identifier. Implementation wise SetDims are treated as Range
/// ids, and spaces with no distinction between dimension ids are treated
/// as relations with zero domain ids.
enum class IdKind { Symbol, Local, Domain, Range, SetDim = Range };

/// PresburgerSpace is the space of all possible values of a tuple of integer
/// valued variables/identifiers. Each identifier has one of the three types:
///
/// Dimension: Ordinary variables over which the space is represented.
///
/// Symbol: Symbol identifiers correspond to fixed but unknown values.
/// Mathematically, a space with symbolic identifiers is like a
/// family of spaces indexed by the symbolic identifiers.
///
/// Local: Local identifiers correspond to existentially quantified variables.
/// For example, consider the space: `(x, exists q)` where x is a dimension
/// identifier and q is a local identifier. Let us put the constraints:
///       `1 <= x <= 7, x = 2q`
/// on this space to get the set:
///       `(x) : (exists q : q <= x <= 7, x = 2q)`.
/// An assignment to symbolic and dimension variables is valid if there
/// exists some assignment to the local variable `q` satisfying these
/// constraints. For this example, the set is equivalent to {2, 4, 6}.
/// Mathematically, existential quantification can be thought of as the result
/// of projection. In this example, `q` is existentially quantified. This can be
/// thought of as the result of projecting out `q` from the previous example,
/// i.e. we obtained {2, 4, 6} by projecting out the second dimension from
/// {(2, 1), (4, 2), (6, 2)}.
///
/// Dimension identifiers are further divided into Domain and Range identifiers
/// to support building relations.
///
/// Spaces with distinction between domain and range identifiers should use
/// IdKind::Domain and IdKind::Range to refer to domain and range identifiers.
/// Identifiers for such spaces are stored in the following order:
///       [Domain, Range, Symbols, Locals]
///
/// Spaces with no distinction between domain and range identifiers should use
/// IdKind::SetDim to refer to dimension identifiers. Identifiers for such
/// spaces are stored in the following order:
///       [SetDim, Symbol, Locals]
///
/// PresburgerSpace does not allow identifiers of kind Local. See
/// PresburgerLocalSpace for an extension that does allow local identifiers.
class PresburgerSpace {
  friend PresburgerLocalSpace;

public:
  static PresburgerSpace getRelationSpace(unsigned numDomain, unsigned numRange,
                                          unsigned numSymbols);

  static PresburgerSpace getSetSpace(unsigned numDims, unsigned numSymbols);

  virtual ~PresburgerSpace() = default;

  unsigned getNumDomainIds() const { return numDomain; }
  unsigned getNumRangeIds() const { return numRange; }
  unsigned getNumSymbolIds() const { return numSymbols; }
  unsigned getNumSetDimIds() const { return numRange; }

  unsigned getNumDimIds() const { return numDomain + numRange; }
  unsigned getNumDimAndSymbolIds() const {
    return numDomain + numRange + numSymbols;
  }
  unsigned getNumIds() const {
    return numDomain + numRange + numSymbols + numLocals;
  }

  /// Get the number of ids of the specified kind.
  unsigned getNumIdKind(IdKind kind) const;

  /// Return the index at which the specified kind of id starts.
  unsigned getIdKindOffset(IdKind kind) const;

  /// Return the index at Which the specified kind of id ends.
  unsigned getIdKindEnd(IdKind kind) const;

  /// Get the number of elements of the specified kind in the range
  /// [idStart, idLimit).
  unsigned getIdKindOverlap(IdKind kind, unsigned idStart,
                            unsigned idLimit) const;

  /// Insert `num` identifiers of the specified kind at position `pos`.
  /// Positions are relative to the kind of identifier. Return the absolute
  /// column position (i.e., not relative to the kind of identifier) of the
  /// first added identifier.
  virtual unsigned insertId(IdKind kind, unsigned pos, unsigned num = 1);

  /// Removes identifiers in the column range [idStart, idLimit).
  virtual void removeIdRange(unsigned idStart, unsigned idLimit);

  /// Returns true if both the spaces are equal i.e. if both spaces have the
  /// same number of identifiers of each kind (excluding Local Identifiers).
  bool isEqual(const PresburgerSpace &other) const;

  /// Changes the partition between dimensions and symbols. Depending on the new
  /// symbol count, either a chunk of dimensional identifiers immediately before
  /// the split become symbols, or some of the symbols immediately after the
  /// split become dimensions.
  void setDimSymbolSeparation(unsigned newSymbolCount);

  void print(llvm::raw_ostream &os) const;
  void dump() const;

protected:
  /// Space constructor for Relation space type.
  PresburgerSpace(unsigned numDomain, unsigned numRange, unsigned numSymbols)
      : PresburgerSpace(Relation, numDomain, numRange, numSymbols,
                        /*numLocals=*/0) {}

  /// Space constructor for Set space type.
  PresburgerSpace(unsigned numDims, unsigned numSymbols)
      : PresburgerSpace(Set, /*numDomain=*/0, numDims, numSymbols,
                        /*numLocals=*/0) {}

private:
  /// Kind of space.
  enum SpaceKind { Set, Relation };

  PresburgerSpace(SpaceKind spaceKind, unsigned numDomain, unsigned numRange,
                  unsigned numSymbols, unsigned numLocals)
      : spaceKind(spaceKind), numDomain(numDomain), numRange(numRange),
        numSymbols(numSymbols), numLocals(numLocals) {}

  SpaceKind spaceKind;

  // Number of identifiers corresponding to domain identifiers.
  unsigned numDomain;

  // Number of identifiers corresponding to range identifiers.
  unsigned numRange;

  /// Number of identifiers corresponding to symbols (unknown but constant for
  /// analysis).
  unsigned numSymbols;

  /// Total number of identifiers.
  unsigned numLocals;
};

/// Extension of PresburgerSpace supporting Local identifiers.
class PresburgerLocalSpace : public PresburgerSpace {
public:
  static PresburgerLocalSpace getRelationSpace(unsigned numDomain,
                                               unsigned numRange,
                                               unsigned numSymbols,
                                               unsigned numLocals);

  static PresburgerLocalSpace getSetSpace(unsigned numDims, unsigned numSymbols,
                                          unsigned numLocals);

  unsigned getNumLocalIds() const { return numLocals; }

  /// Insert `num` identifiers of the specified kind at position `pos`.
  /// Positions are relative to the kind of identifier. Return the absolute
  /// column position (i.e., not relative to the kind of identifier) of the
  /// first added identifier.
  unsigned insertId(IdKind kind, unsigned pos, unsigned num = 1) override;

  /// Removes identifiers in the column range [idStart, idLimit).
  void removeIdRange(unsigned idStart, unsigned idLimit) override;

  void print(llvm::raw_ostream &os) const;
  void dump() const;

protected:
  /// Local Space constructor for Relation space type.
  PresburgerLocalSpace(unsigned numDomain, unsigned numRange,
                       unsigned numSymbols, unsigned numLocals)
      : PresburgerSpace(Relation, numDomain, numRange, numSymbols, numLocals) {}

  /// Local Space constructor for Set space type.
  PresburgerLocalSpace(unsigned numDims, unsigned numSymbols,
                       unsigned numLocals)
      : PresburgerSpace(Set, /*numDomain=*/0, numDims, numSymbols, numLocals) {}

  /// Returns true if both the spaces are equal i.e. if both spaces have the
  /// same number of identifiers of each kind.
  bool isEqual(const PresburgerLocalSpace &other) const;
};

} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_PRESBURGERSPACE_H
