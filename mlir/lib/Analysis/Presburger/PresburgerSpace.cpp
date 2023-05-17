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

PresburgerSpace PresburgerSpace::getDomainSpace() const {
  // TODO: Preserve identifiers here.
  return PresburgerSpace::getSetSpace(numDomain, numSymbols, numLocals);
}

PresburgerSpace PresburgerSpace::getRangeSpace() const {
  return PresburgerSpace::getSetSpace(numRange, numSymbols, numLocals);
}

PresburgerSpace PresburgerSpace::getSpaceWithoutLocals() const {
  PresburgerSpace space = *this;
  space.removeVarRange(VarKind::Local, 0, numLocals);
  return space;
}

unsigned PresburgerSpace::getNumVarKind(VarKind kind) const {
  if (kind == VarKind::Domain)
    return getNumDomainVars();
  if (kind == VarKind::Range)
    return getNumRangeVars();
  if (kind == VarKind::Symbol)
    return getNumSymbolVars();
  if (kind == VarKind::Local)
    return numLocals;
  llvm_unreachable("VarKind does not exist!");
}

unsigned PresburgerSpace::getVarKindOffset(VarKind kind) const {
  if (kind == VarKind::Domain)
    return 0;
  if (kind == VarKind::Range)
    return getNumDomainVars();
  if (kind == VarKind::Symbol)
    return getNumDimVars();
  if (kind == VarKind::Local)
    return getNumDimAndSymbolVars();
  llvm_unreachable("VarKind does not exist!");
}

unsigned PresburgerSpace::getVarKindEnd(VarKind kind) const {
  return getVarKindOffset(kind) + getNumVarKind(kind);
}

unsigned PresburgerSpace::getVarKindOverlap(VarKind kind, unsigned varStart,
                                            unsigned varLimit) const {
  unsigned varRangeStart = getVarKindOffset(kind);
  unsigned varRangeEnd = getVarKindEnd(kind);

  // Compute number of elements in intersection of the ranges [varStart,
  // varLimit) and [varRangeStart, varRangeEnd).
  unsigned overlapStart = std::max(varStart, varRangeStart);
  unsigned overlapEnd = std::min(varLimit, varRangeEnd);

  if (overlapStart > overlapEnd)
    return 0;
  return overlapEnd - overlapStart;
}

VarKind PresburgerSpace::getVarKindAt(unsigned pos) const {
  assert(pos < getNumVars() && "`pos` should represent a valid var position");
  if (pos < getVarKindEnd(VarKind::Domain))
    return VarKind::Domain;
  if (pos < getVarKindEnd(VarKind::Range))
    return VarKind::Range;
  if (pos < getVarKindEnd(VarKind::Symbol))
    return VarKind::Symbol;
  if (pos < getVarKindEnd(VarKind::Local))
    return VarKind::Local;
  llvm_unreachable("`pos` should represent a valid var position");
}

unsigned PresburgerSpace::insertVar(VarKind kind, unsigned pos, unsigned num) {
  assert(pos <= getNumVarKind(kind));

  unsigned absolutePos = getVarKindOffset(kind) + pos;

  if (kind == VarKind::Domain)
    numDomain += num;
  else if (kind == VarKind::Range)
    numRange += num;
  else if (kind == VarKind::Symbol)
    numSymbols += num;
  else
    numLocals += num;

  // Insert NULL identifiers if `usingIds` and variables inserted are
  // not locals.
  if (usingIds && kind != VarKind::Local)
    identifiers.insert(identifiers.begin() + absolutePos, num, nullptr);

  return absolutePos;
}

void PresburgerSpace::removeVarRange(VarKind kind, unsigned varStart,
                                     unsigned varLimit) {
  assert(varLimit <= getNumVarKind(kind) && "invalid var limit");

  if (varStart >= varLimit)
    return;

  unsigned numVarsEliminated = varLimit - varStart;
  if (kind == VarKind::Domain)
    numDomain -= numVarsEliminated;
  else if (kind == VarKind::Range)
    numRange -= numVarsEliminated;
  else if (kind == VarKind::Symbol)
    numSymbols -= numVarsEliminated;
  else
    numLocals -= numVarsEliminated;

  // Remove identifiers if `usingIds` and variables removed are not
  // locals.
  if (usingIds && kind != VarKind::Local)
    identifiers.erase(identifiers.begin() + getVarKindOffset(kind) + varStart,
                      identifiers.begin() + getVarKindOffset(kind) + varLimit);
}

void PresburgerSpace::swapVar(VarKind kindA, VarKind kindB, unsigned posA,
                              unsigned posB) {

  if (!usingIds)
    return;

  if (kindA == VarKind::Local && kindB == VarKind::Local)
    return;

  if (kindA == VarKind::Local) {
    atId(kindB, posB) = nullptr;
    return;
  }

  if (kindB == VarKind::Local) {
    atId(kindA, posA) = nullptr;
    return;
  }

  std::swap(atId(kindA, posA), atId(kindB, posB));
}

bool PresburgerSpace::isCompatible(const PresburgerSpace &other) const {
  return getNumDomainVars() == other.getNumDomainVars() &&
         getNumRangeVars() == other.getNumRangeVars() &&
         getNumSymbolVars() == other.getNumSymbolVars();
}

bool PresburgerSpace::isEqual(const PresburgerSpace &other) const {
  return isCompatible(other) && getNumLocalVars() == other.getNumLocalVars();
}

bool PresburgerSpace::isAligned(const PresburgerSpace &other) const {
  assert(isUsingIds() && other.isUsingIds() &&
         "Both spaces should be using identifiers to check for "
         "alignment.");
  return isCompatible(other) && identifiers == other.identifiers;
}

bool PresburgerSpace::isAligned(const PresburgerSpace &other,
                                VarKind kind) const {
  assert(isUsingIds() && other.isUsingIds() &&
         "Both spaces should be using identifiers to check for "
         "alignment.");

  ArrayRef<void *> kindAttachments =
      ArrayRef(identifiers).slice(getVarKindOffset(kind), getNumVarKind(kind));
  ArrayRef<void *> otherKindAttachments =
      ArrayRef(other.identifiers)
          .slice(other.getVarKindOffset(kind), other.getNumVarKind(kind));
  return kindAttachments == otherKindAttachments;
}

void PresburgerSpace::setVarSymbolSeperation(unsigned newSymbolCount) {
  assert(newSymbolCount <= getNumDimAndSymbolVars() &&
         "invalid separation position");
  numRange = numRange + numSymbols - newSymbolCount;
  numSymbols = newSymbolCount;
  // We do not need to change `identifiers` since the ordering of
  // `identifiers` remains same.
}

void PresburgerSpace::print(llvm::raw_ostream &os) const {
  os << "Domain: " << getNumDomainVars() << ", "
     << "Range: " << getNumRangeVars() << ", "
     << "Symbols: " << getNumSymbolVars() << ", "
     << "Locals: " << getNumLocalVars() << "\n";

  if (usingIds) {
#ifdef LLVM_ENABLE_ABI_BREAKING_CHECKS
    os << "TypeID of identifiers: " << idType.getAsOpaquePointer() << "\n";
#endif

    os << "(";
    for (void *identifier : identifiers)
      os << identifier << " ";
    os << ")\n";
  }
}

void PresburgerSpace::dump() const { print(llvm::errs()); }
