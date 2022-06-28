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

  // Insert NULL attachments if `usingAttachments` and variables inserted are
  // not locals.
  if (usingAttachments && kind != VarKind::Local)
    attachments.insert(attachments.begin() + absolutePos, num, nullptr);

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

  // Remove attachments if `usingAttachments` and variables removed are not
  // locals.
  if (usingAttachments && kind != VarKind::Local)
    attachments.erase(attachments.begin() + getVarKindOffset(kind) + varStart,
                      attachments.begin() + getVarKindOffset(kind) + varLimit);
}

void PresburgerSpace::swapVar(VarKind kindA, VarKind kindB, unsigned posA,
                              unsigned posB) {

  if (!usingAttachments)
    return;

  if (kindA == VarKind::Local && kindB == VarKind::Local)
    return;

  if (kindA == VarKind::Local) {
    atAttachment(kindB, posB) = nullptr;
    return;
  }

  if (kindB == VarKind::Local) {
    atAttachment(kindA, posA) = nullptr;
    return;
  }

  std::swap(atAttachment(kindA, posA), atAttachment(kindB, posB));
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
  assert(isUsingAttachments() && other.isUsingAttachments() &&
         "Both spaces should be using attachments to check for "
         "alignment.");
  return isCompatible(other) && attachments == other.attachments;
}

bool PresburgerSpace::isAligned(const PresburgerSpace &other,
                                VarKind kind) const {
  assert(isUsingAttachments() && other.isUsingAttachments() &&
         "Both spaces should be using attachments to check for "
         "alignment.");

  ArrayRef<void *> kindAttachments =
      makeArrayRef(attachments)
          .slice(getVarKindOffset(kind), getNumVarKind(kind));
  ArrayRef<void *> otherKindAttachments =
      makeArrayRef(other.attachments)
          .slice(other.getVarKindOffset(kind), other.getNumVarKind(kind));
  return kindAttachments == otherKindAttachments;
}

void PresburgerSpace::setVarSymbolSeperation(unsigned newSymbolCount) {
  assert(newSymbolCount <= getNumDimAndSymbolVars() &&
         "invalid separation position");
  numRange = numRange + numSymbols - newSymbolCount;
  numSymbols = newSymbolCount;
  // We do not need to change `attachments` since the ordering of
  // `attachments` remains same.
}

void PresburgerSpace::print(llvm::raw_ostream &os) const {
  os << "Domain: " << getNumDomainVars() << ", "
     << "Range: " << getNumRangeVars() << ", "
     << "Symbols: " << getNumSymbolVars() << ", "
     << "Locals: " << getNumLocalVars() << "\n";

  if (usingAttachments) {
#ifdef LLVM_ENABLE_ABI_BREAKING_CHECKS
    os << "TypeID of attachments: " << attachmentType.getAsOpaquePointer()
       << "\n";
#endif

    os << "(";
    for (void *attachment : attachments)
      os << attachment << " ";
    os << ")\n";
  }
}

void PresburgerSpace::dump() const { print(llvm::errs()); }
