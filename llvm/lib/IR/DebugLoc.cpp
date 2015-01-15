//===-- DebugLoc.cpp - Implement DebugLoc class ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DebugLoc.h"
#include "LLVMContextImpl.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/IR/DebugInfo.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// DebugLoc Implementation
//===----------------------------------------------------------------------===//

unsigned DebugLoc::getLine() const { return DILocation(Loc).getLineNumber(); }
unsigned DebugLoc::getCol() const { return DILocation(Loc).getColumnNumber(); }

MDNode *DebugLoc::getScope() const { return DILocation(Loc).getScope(); }

MDNode *DebugLoc::getInlinedAt() const {
  return DILocation(Loc).getOrigLocation();
}

/// Return both the Scope and the InlinedAt values.
void DebugLoc::getScopeAndInlinedAt(MDNode *&Scope, MDNode *&IA) const {
  Scope = getScope();
  IA = getInlinedAt();
}

MDNode *DebugLoc::getScopeNode() const {
  if (MDNode *InlinedAt = getInlinedAt())
    return DebugLoc::getFromDILocation(InlinedAt).getScopeNode();
  return getScope();
}

DebugLoc DebugLoc::getFnDebugLoc() const {
  const MDNode *Scope = getScopeNode();
  DISubprogram SP = getDISubprogram(Scope);
  if (SP.isSubprogram())
    return DebugLoc::get(SP.getScopeLineNumber(), 0, SP);

  return DebugLoc();
}

DebugLoc DebugLoc::get(unsigned Line, unsigned Col,
                       MDNode *Scope, MDNode *InlinedAt) {
  // If no scope is available, this is an unknown location.
  if (!Scope)
    return DebugLoc();

  // Saturate line and col to "unknown".
  // FIXME: Allow 16-bits for columns.
  if (Col > 255) Col = 0;
  if (Line >= (1 << 24)) Line = 0;

  return getFromDILocation(
      MDLocation::get(Scope->getContext(), Line, Col, Scope, InlinedAt));
}

/// getAsMDNode - This method converts the compressed DebugLoc node into a
/// DILocation-compatible MDNode.
MDNode *DebugLoc::getAsMDNode() const { return Loc; }

/// getFromDILocation - Translate the DILocation quad into a DebugLoc.
DebugLoc DebugLoc::getFromDILocation(MDNode *N) {
  DebugLoc Loc;
  Loc.Loc.reset(N);
  return Loc;
}

/// getFromDILexicalBlock - Translate the DILexicalBlock into a DebugLoc.
DebugLoc DebugLoc::getFromDILexicalBlock(MDNode *N) {
  DILexicalBlock LexBlock(N);
  MDNode *Scope = LexBlock.getContext();
  if (!Scope) return DebugLoc();
  return get(LexBlock.getLineNumber(), LexBlock.getColumnNumber(), Scope,
             nullptr);
}

void DebugLoc::dump() const {
#ifndef NDEBUG
  if (!isUnknown()) {
    dbgs() << getLine();
    if (getCol() != 0)
      dbgs() << ',' << getCol();
    DebugLoc InlinedAtDL = DebugLoc::getFromDILocation(getInlinedAt());
    if (!InlinedAtDL.isUnknown()) {
      dbgs() << " @ ";
      InlinedAtDL.dump();
    } else
      dbgs() << "\n";
  }
#endif
}

void DebugLoc::print(raw_ostream &OS) const {
  if (!isUnknown()) {
    // Print source line info.
    DIScope Scope(getScope());
    assert((!Scope || Scope.isScope()) &&
           "Scope of a DebugLoc should be null or a DIScope.");
    if (Scope)
      OS << Scope.getFilename();
    else
      OS << "<unknown>";
    OS << ':' << getLine();
    if (getCol() != 0)
      OS << ':' << getCol();
    DebugLoc InlinedAtDL = DebugLoc::getFromDILocation(getInlinedAt());
    if (!InlinedAtDL.isUnknown()) {
      OS << " @[ ";
      InlinedAtDL.print(OS);
      OS << " ]";
    }
  }
}
