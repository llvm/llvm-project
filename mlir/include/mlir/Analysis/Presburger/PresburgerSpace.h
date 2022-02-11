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
/// PresburgerSpace only supports identifiers of kind Dimension and Symbol.
class PresburgerSpace {
  friend PresburgerLocalSpace;

public:
  /// Kind of identifier (column).
  enum IdKind { Dimension, Symbol, Local };

  PresburgerSpace(unsigned numDims, unsigned numSymbols)
      : numDims(numDims), numSymbols(numSymbols), numLocals(0) {}

  virtual ~PresburgerSpace() = default;

  unsigned getNumIds() const { return numDims + numSymbols + numLocals; }
  unsigned getNumDimIds() const { return numDims; }
  unsigned getNumSymbolIds() const { return numSymbols; }
  unsigned getNumDimAndSymbolIds() const { return numDims + numSymbols; }

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

private:
  PresburgerSpace(unsigned numDims, unsigned numSymbols, unsigned numLocals)
      : numDims(numDims), numSymbols(numSymbols), numLocals(numLocals) {}

  /// Number of identifiers corresponding to real dimensions.
  unsigned numDims;

  /// Number of identifiers corresponding to symbols (unknown but constant for
  /// analysis).
  unsigned numSymbols;

  /// Total number of identifiers.
  unsigned numLocals;
};

/// Extension of PresburgerSpace supporting Local identifiers.
class PresburgerLocalSpace : public PresburgerSpace {
public:
  PresburgerLocalSpace(unsigned numDims, unsigned numSymbols,
                       unsigned numLocals)
      : PresburgerSpace(numDims, numSymbols, numLocals) {}

  unsigned getNumLocalIds() const { return numLocals; }

  /// Insert `num` identifiers of the specified kind at position `pos`.
  /// Positions are relative to the kind of identifier. Return the absolute
  /// column position (i.e., not relative to the kind of identifier) of the
  /// first added identifier.
  unsigned insertId(IdKind kind, unsigned pos, unsigned num = 1) override;

  /// Removes identifiers in the column range [idStart, idLimit).
  void removeIdRange(unsigned idStart, unsigned idLimit) override;
};

} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_PRESBURGERSPACE_H
