//===- PresburgerSpace.cpp - MLIR PresburgerSpace Class -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>

using namespace mlir;
using namespace presburger;

bool Identifier::isEqual(const Identifier &other) const {
  if (value == nullptr || other.value == nullptr)
    return false;
  assert(value == other.value && idType == other.idType &&
         "Values of Identifiers are equal but their types do not match.");
  return value == other.value;
}

void Identifier::print(llvm::raw_ostream &os) const {
  os << "Id<" << value << ">";
}

void Identifier::dump() const {
  print(llvm::errs());
  llvm::errs() << "\n";
}

PresburgerSpace PresburgerSpace::getDomainSpace() const {
  PresburgerSpace newSpace = *this;
  newSpace.removeVarRange(VarKind::Range, 0, getNumRangeVars());
  newSpace.convertVarKind(VarKind::Domain, 0, getNumDomainVars(),
                          VarKind::SetDim, 0);
  return newSpace;
}

PresburgerSpace PresburgerSpace::getRangeSpace() const {
  PresburgerSpace newSpace = *this;
  newSpace.removeVarRange(VarKind::Domain, 0, getNumDomainVars());
  return newSpace;
}

PresburgerSpace PresburgerSpace::getSpaceWithoutLocals() const {
  PresburgerSpace space = *this;
  space.removeVarRange(VarKind::Local, 0, getNumLocalVars());
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
    return getNumLocalVars();
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
    identifiers.insert(identifiers.begin() + absolutePos, num, Identifier());

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

void PresburgerSpace::convertVarKind(VarKind srcKind, unsigned srcPos,
                                     unsigned num, VarKind dstKind,
                                     unsigned dstPos) {
  assert(srcKind != dstKind && "cannot convert variables to the same kind");
  assert(srcPos + num <= getNumVarKind(srcKind) &&
         "invalid range for source variables");
  assert(dstPos <= getNumVarKind(dstKind) &&
         "invalid position for destination variables");

  // Move identifiers if `usingIds` and variables moved are not locals.
  unsigned srcOffset = getVarKindOffset(srcKind) + srcPos;
  unsigned dstOffset = getVarKindOffset(dstKind) + dstPos;
  if (isUsingIds() && srcKind != VarKind::Local && dstKind != VarKind::Local) {
    identifiers.insert(identifiers.begin() + dstOffset, num, Identifier());
    // Update srcOffset if insertion of new elements invalidates it.
    if (dstOffset < srcOffset)
      srcOffset += num;
    std::move(identifiers.begin() + srcOffset,
              identifiers.begin() + srcOffset + num,
              identifiers.begin() + dstOffset);
    identifiers.erase(identifiers.begin() + srcOffset,
                      identifiers.begin() + srcOffset + num);
  } else if (isUsingIds() && srcKind != VarKind::Local) {
    identifiers.erase(identifiers.begin() + srcOffset,
                      identifiers.begin() + srcOffset + num);
  } else if (isUsingIds() && dstKind != VarKind::Local) {
    identifiers.insert(identifiers.begin() + dstOffset, num, Identifier());
  }

  auto addVars = [&](VarKind kind, int num) {
    switch (kind) {
    case VarKind::Domain:
      numDomain += num;
      break;
    case VarKind::Range:
      numRange += num;
      break;
    case VarKind::Symbol:
      numSymbols += num;
      break;
    case VarKind::Local:
      numLocals += num;
      break;
    }
  };

  addVars(srcKind, -(signed)num);
  addVars(dstKind, num);
}

void PresburgerSpace::swapVar(VarKind kindA, VarKind kindB, unsigned posA,
                              unsigned posB) {
  if (!isUsingIds())
    return;

  if (kindA == VarKind::Local && kindB == VarKind::Local)
    return;

  if (kindA == VarKind::Local) {
    getId(kindB, posB) = Identifier();
    return;
  }

  if (kindB == VarKind::Local) {
    getId(kindA, posA) = Identifier();
    return;
  }

  std::swap(getId(kindA, posA), getId(kindB, posB));
}

bool PresburgerSpace::isCompatible(const PresburgerSpace &other) const {
  return getNumDomainVars() == other.getNumDomainVars() &&
         getNumRangeVars() == other.getNumRangeVars() &&
         getNumSymbolVars() == other.getNumSymbolVars();
}

bool PresburgerSpace::isEqual(const PresburgerSpace &other) const {
  return isCompatible(other) && getNumLocalVars() == other.getNumLocalVars();
}

/// Checks if the number of ids of the given kind in the two spaces are
/// equal and if the ids are equal. Assumes that both spaces are using
/// ids.
static bool areIdsEqual(const PresburgerSpace &spaceA,
                        const PresburgerSpace &spaceB, VarKind kind) {
  assert(spaceA.isUsingIds() && spaceB.isUsingIds() &&
         "Both spaces should be using ids");
  if (spaceA.getNumVarKind(kind) != spaceB.getNumVarKind(kind))
    return false;
  if (kind == VarKind::Local)
    return true; // No ids.
  return spaceA.getIds(kind) == spaceB.getIds(kind);
}

bool PresburgerSpace::isAligned(const PresburgerSpace &other) const {
  // If only one of the spaces is using identifiers, then they are
  // not aligned.
  if (isUsingIds() != other.isUsingIds())
    return false;
  // If both spaces are using identifiers, then they are aligned if
  // their identifiers are equal. Identifiers being equal implies
  // that the number of variables of each kind is same, which implies
  // compatiblity, so we do not check for that.
  if (isUsingIds())
    return areIdsEqual(*this, other, VarKind::Domain) &&
           areIdsEqual(*this, other, VarKind::Range) &&
           areIdsEqual(*this, other, VarKind::Symbol);
  // If neither space is using identifiers, then they are aligned if
  // they are compatible.
  return isCompatible(other);
}

bool PresburgerSpace::isAligned(const PresburgerSpace &other,
                                VarKind kind) const {
  // If only one of the spaces is using identifiers, then they are
  // not aligned.
  if (isUsingIds() != other.isUsingIds())
    return false;
  // If both spaces are using identifiers, then they are aligned if
  // their identifiers are equal. Identifiers being equal implies
  // that the number of variables of each kind is same, which implies
  // compatiblity, so we do not check for that
  if (isUsingIds())
    return areIdsEqual(*this, other, kind);
  // If neither space is using identifiers, then they are aligned if
  // the number of variable kind is equal.
  return getNumVarKind(kind) == other.getNumVarKind(kind);
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

  if (isUsingIds()) {
    auto printIds = [&](VarKind kind) {
      os << " ";
      for (Identifier id : getIds(kind)) {
        if (id.hasValue())
          id.print(os);
        else
          os << "None";
        os << " ";
      }
    };

    os << "(";
    printIds(VarKind::Domain);
    os << ") -> (";
    printIds(VarKind::Range);
    os << ") : [";
    printIds(VarKind::Symbol);
    os << "]";
  }
}

void PresburgerSpace::dump() const {
  print(llvm::errs());
  llvm::errs() << "\n";
}
