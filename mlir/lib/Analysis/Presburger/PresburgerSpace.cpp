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

  if (kind == IdKind::Domain)
    numDomain += num;
  else if (kind == IdKind::Range)
    numRange += num;
  else if (kind == IdKind::Symbol)
    numSymbols += num;
  else
    llvm_unreachable("PresburgerSpace does not support local identifiers!");

  return absolutePos;
}

void PresburgerSpace::removeIdRange(IdKind kind, unsigned idStart,
                                    unsigned idLimit) {
  assert(idLimit <= getNumIdKind(kind) && "invalid id limit");

  if (idStart >= idLimit)
    return;

  unsigned numIdsEliminated = idLimit - idStart;
  if (kind == IdKind::Domain)
    numDomain -= numIdsEliminated;
  else if (kind == IdKind::Range)
    numRange -= numIdsEliminated;
  else if (kind == IdKind::Symbol)
    numSymbols -= numIdsEliminated;
  else
    llvm_unreachable("PresburgerSpace does not support local identifiers!");
}

void PresburgerSpace::truncateIdKind(IdKind kind, unsigned num) {
  unsigned curNum = getNumIdKind(kind);
  assert(num <= curNum && "Can't truncate to more ids!");
  removeIdRange(kind, num, curNum);
}

unsigned PresburgerLocalSpace::insertId(IdKind kind, unsigned pos,
                                        unsigned num) {
  if (kind == IdKind::Local) {
    numLocals += num;
    return getIdKindOffset(IdKind::Local) + pos;
  }
  return PresburgerSpace::insertId(kind, pos, num);
}

void PresburgerLocalSpace::removeIdRange(IdKind kind, unsigned idStart,
                                         unsigned idLimit) {
  assert(idLimit <= getNumIdKind(kind) && "invalid id limit");

  if (idStart >= idLimit)
    return;

  if (kind == IdKind::Local)
    numLocals -= idLimit - idStart;
  else
    PresburgerSpace::removeIdRange(kind, idStart, idLimit);
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
  os << "Domain: " << getNumDomainIds() << ", "
     << "Range: " << getNumRangeIds() << ", "
     << "Symbols: " << getNumSymbolIds() << "\n";
}

void PresburgerSpace::dump() const { print(llvm::errs()); }

void PresburgerLocalSpace::print(llvm::raw_ostream &os) const {
  PresburgerSpace::print(os);
  os << "Locals: " << getNumLocalIds() << "\n";
}

void PresburgerLocalSpace::dump() const { print(llvm::errs()); }
