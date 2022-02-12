//===- PresburgerSpace.cpp - MLIR PresburgerSpace Class -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include <algorithm>
#include <cassert>

using namespace mlir;

unsigned PresburgerSpace::getNumIdKind(IdKind kind) const {
  if (kind == IdKind::Dimension)
    return getNumDimIds();
  if (kind == IdKind::Symbol)
    return getNumSymbolIds();
  if (kind == IdKind::Local)
    return numLocals;
  llvm_unreachable("IdKind does not exit!");
}

unsigned PresburgerSpace::getIdKindOffset(IdKind kind) const {
  if (kind == IdKind::Dimension)
    return 0;
  if (kind == IdKind::Symbol)
    return getNumDimIds();
  if (kind == IdKind::Local)
    return getNumDimAndSymbolIds();
  llvm_unreachable("IdKind does not exit!");
}

unsigned PresburgerSpace::getIdKindEnd(IdKind kind) const {
  return getIdKindOffset(kind) + getNumIdKind(kind);
}

unsigned PresburgerSpace::getIdKindOverlap(IdKind kind, unsigned idStart,
                                           unsigned idLimit) const {
  unsigned idRangeStart = getIdKindOffset(kind);
  unsigned idRangeEnd = getIdKindEnd(kind);

  // Compute number of elements in intersection of the ranges [idStart, idLimit)
  // and [idRangeStart, idRangeEnd).
  unsigned overlapStart = std::max(idStart, idRangeStart);
  unsigned overlapEnd = std::min(idLimit, idRangeEnd);

  if (overlapStart > overlapEnd)
    return 0;
  return overlapEnd - overlapStart;
}

unsigned PresburgerSpace::insertId(IdKind kind, unsigned pos, unsigned num) {
  assert(pos <= getNumIdKind(kind));

  unsigned absolutePos = getIdKindOffset(kind) + pos;

  if (kind == IdKind::Dimension)
    numDims += num;
  else if (kind == IdKind::Symbol)
    numSymbols += num;
  else
    llvm_unreachable(
        "PresburgerSpace only supports Dimensions and Symbol identifiers!");

  return absolutePos;
}

void PresburgerSpace::removeIdRange(unsigned idStart, unsigned idLimit) {
  assert(idLimit <= getNumIds() && "invalid id limit");

  if (idStart >= idLimit)
    return;

  // We are going to be removing one or more identifiers from the range.
  assert(idStart < getNumIds() && "invalid idStart position");

  // Update members numDims, numSymbols and numIds.
  unsigned numDimsEliminated =
      getIdKindOverlap(IdKind::Dimension, idStart, idLimit);
  unsigned numSymbolsEliminated =
      getIdKindOverlap(IdKind::Symbol, idStart, idLimit);

  numDims -= numDimsEliminated;
  numSymbols -= numSymbolsEliminated;
}

unsigned PresburgerLocalSpace::insertId(IdKind kind, unsigned pos,
                                        unsigned num) {
  if (kind == IdKind::Local) {
    numLocals += num;
    return getIdKindOffset(IdKind::Local) + pos;
  }
  return PresburgerSpace::insertId(kind, pos, num);
}

void PresburgerLocalSpace::removeIdRange(unsigned idStart, unsigned idLimit) {
  assert(idLimit <= getNumIds() && "invalid id limit");

  if (idStart >= idLimit)
    return;

  // We are going to be removing one or more identifiers from the range.
  assert(idStart < getNumIds() && "invalid idStart position");

  unsigned numLocalsEliminated =
      getIdKindOverlap(IdKind::Local, idStart, idLimit);

  // Update space parameters.
  PresburgerSpace::removeIdRange(
      idStart, std::min(idLimit, PresburgerSpace::getNumIds()));

  // Update local ids.
  numLocals -= numLocalsEliminated;
}

void PresburgerSpace::setDimSymbolSeparation(unsigned newSymbolCount) {
  assert(newSymbolCount <= getNumDimAndSymbolIds() &&
         "invalid separation position");
  numDims = numDims + numSymbols - newSymbolCount;
  numSymbols = newSymbolCount;
}
