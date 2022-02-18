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

class PresburgerLocalSpace;

/// PresburgerSpace is a tuple of identifiers with information about what kind
/// they correspond to. The identifiers can be split into three types:
///
/// Dimension: Ordinary variables over which the space is represented.
///
/// Symbol: Symbol identifiers correspond to fixed but unknown values.
/// Mathematically, a space with symbolic identifiers is like a
/// family of spaces indexed by the symbolic identifiers.
///
/// Local: Local identifiers correspond to existentially quantified variables.
///
/// Dimension identifiers are further divided into Domain and Range identifiers
/// to support building relations.
///
/// Spaces with distinction between domain and range identifiers should use
/// IdKind::Domain and IdKind::Range to refer to domain and range identifiers.
///
/// Spaces with no distinction between domain and range identifiers should use
/// IdKind::SetDim to refer to dimension identifiers.
///
/// PresburgerSpace does not support identifiers of kind Local. See
/// PresburgerLocalSpace for an extension that supports Local ids.
class PresburgerSpace {
  friend PresburgerLocalSpace;

public:
  /// Kind of identifier. Implementation wise SetDims are treated as Range
  /// ids, and spaces with no distinction between dimension ids are treated
  /// as relations with zero domain ids.
  enum IdKind { Symbol, Local, Domain, Range, SetDim = Range };

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
};

} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_PRESBURGERSPACE_H
