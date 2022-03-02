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
using namespace presburger;

PresburgerSpace PresburgerSpace::getRelationSpace(unsigned numDomain,
                                                  unsigned numRange,
                                                  unsigned numSymbols) {
  return PresburgerSpace(numDomain, numRange, numSymbols);
}

PresburgerSpace PresburgerSpace::getSetSpace(unsigned numDims,
                                             unsigned numSymbols) {
  return PresburgerSpace(numDims, numSymbols);
}

PresburgerLocalSpace
PresburgerLocalSpace::getRelationSpace(unsigned numDomain, unsigned numRange,
                                       unsigned numSymbols,
                                       unsigned numLocals) {
  return PresburgerLocalSpace(numDomain, numRange, numSymbols, numLocals);
}

PresburgerLocalSpace PresburgerLocalSpace::getSetSpace(unsigned numDims,
                                                       unsigned numSymbols,
                                                       unsigned numLocals) {
  return PresburgerLocalSpace(numDims, numSymbols, numLocals);
}

unsigned PresburgerSpace::getNumIdKind(IdKind kind) const {
  if (kind == IdKind::Domain)
    return getNumDomainIds();
  if (kind == IdKind::Range)
    return getNumRangeIds();
  if (kind == IdKind::Symbol)
    return getNumSymbolIds();
  if (kind == IdKind::Local)
    return numLocals;
  llvm_unreachable("IdKind does not exist!");
}

unsigned PresburgerSpace::getIdKindOffset(IdKind kind) const {
  if (kind == IdKind::Domain)
    return 0;
  if (kind == IdKind::Range)
    return getNumDomainIds();
  if (kind == IdKind::Symbol)
    return getNumDimIds();
  if (kind == IdKind::Local)
    return getNumDimAndSymbolIds();
  llvm_unreachable("IdKind does not exist!");
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

  if (kind == IdKind::Domain) {
    assert(spaceKind == Relation && "IdKind::Domain is not supported in Set.");
    numDomain += num;
  } else if (kind == IdKind::Range) {
    numRange += num;
  } else if (kind == IdKind::Symbol) {
    numSymbols += num;
  } else {
    llvm_unreachable("PresburgerSpace does not support local identifiers!");
  }

  return absolutePos;
}

void PresburgerSpace::removeIdRange(unsigned idStart, unsigned idLimit) {
  assert(idLimit <= getNumIds() && "invalid id limit");

  if (idStart >= idLimit)
    return;

  // We are going to be removing one or more identifiers from the range.
  assert(idStart < getNumIds() && "invalid idStart position");

  // Update members numDomain, numRange, numSymbols and numIds.
  unsigned numDomainEliminated = 0;
  if (spaceKind == Relation)
    numDomainEliminated = getIdKindOverlap(IdKind::Domain, idStart, idLimit);
  unsigned numRangeEliminated =
      getIdKindOverlap(IdKind::Range, idStart, idLimit);
  unsigned numSymbolsEliminated =
      getIdKindOverlap(IdKind::Symbol, idStart, idLimit);

  numDomain -= numDomainEliminated;
  numRange -= numRangeEliminated;
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
  PresburgerSpace::removeIdRange(idStart, idLimit);

  // Update local ids.
  numLocals -= numLocalsEliminated;
}

bool PresburgerSpace::isEqual(const PresburgerSpace &other) const {
  return getNumDomainIds() == other.getNumDomainIds() &&
         getNumRangeIds() == other.getNumRangeIds() &&
         getNumSymbolIds() == other.getNumSymbolIds();
}

bool PresburgerLocalSpace::isEqual(const PresburgerLocalSpace &other) const {
  return PresburgerSpace::isEqual(other) &&
         getNumLocalIds() == other.getNumLocalIds();
}

void PresburgerSpace::setDimSymbolSeparation(unsigned newSymbolCount) {
  assert(newSymbolCount <= getNumDimAndSymbolIds() &&
         "invalid separation position");
  numRange = numRange + numSymbols - newSymbolCount;
  numSymbols = newSymbolCount;
}

void PresburgerSpace::print(llvm::raw_ostream &os) const {
  if (spaceKind == Relation) {
    os << "Domain: " << getNumDomainIds() << ", "
       << "Range: " << getNumRangeIds() << ", ";
  } else {
    os << "Dimension: " << getNumDomainIds() << ", ";
  }
  os << "Symbols: " << getNumSymbolIds() << "\n";
}

void PresburgerSpace::dump() const { print(llvm::errs()); }

void PresburgerLocalSpace::print(llvm::raw_ostream &os) const {
  if (spaceKind == Relation) {
    os << "Domain: " << getNumDomainIds() << ", "
       << "Range: " << getNumRangeIds() << ", ";
  } else {
    os << "Dimension: " << getNumDomainIds() << ", ";
  }
  os << "Symbols: " << getNumSymbolIds() << ", "
     << "Locals" << getNumLocalIds() << "\n";
}

void PresburgerLocalSpace::dump() const { print(llvm::errs()); }
