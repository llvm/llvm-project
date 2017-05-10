//===-- DebugLoc.cpp - Implement DebugLoc class ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/IntrinsicInst.h"
#include "LLVMContextImpl.h"
#include "llvm/IR/DebugInfo.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// DebugLoc Implementation
//===----------------------------------------------------------------------===//
DebugLoc::DebugLoc(const DILocation *L) : Loc(const_cast<DILocation *>(L)) {}
DebugLoc::DebugLoc(const MDNode *L) : Loc(const_cast<MDNode *>(L)) {}

DILocation *DebugLoc::get() const {
  return cast_or_null<DILocation>(Loc.get());
}

unsigned DebugLoc::getLine() const {
  assert(get() && "Expected valid DebugLoc");
  return get()->getLine();
}

unsigned DebugLoc::getCol() const {
  assert(get() && "Expected valid DebugLoc");
  return get()->getColumn();
}

MDNode *DebugLoc::getScope() const {
  assert(get() && "Expected valid DebugLoc");
  return get()->getScope();
}

DILocation *DebugLoc::getInlinedAt() const {
  assert(get() && "Expected valid DebugLoc");
  return get()->getInlinedAt();
}

MDNode *DebugLoc::getInlinedAtScope() const {
  return cast<DILocation>(Loc)->getInlinedAtScope();
}

DebugLoc DebugLoc::getFnDebugLoc() const {
  // FIXME: Add a method on \a DILocation that does this work.
  const MDNode *Scope = getInlinedAtScope();
  if (auto *SP = getDISubprogram(Scope))
    return DebugLoc::get(SP->getScopeLine(), 0, SP);

  return DebugLoc();
}

DebugLoc DebugLoc::get(unsigned Line, unsigned Col, const MDNode *Scope,
                       const MDNode *InlinedAt) {
  // If no scope is available, this is an unknown location.
  if (!Scope)
    return DebugLoc();

  return DILocation::get(Scope->getContext(), Line, Col,
                         const_cast<MDNode *>(Scope),
                         const_cast<MDNode *>(InlinedAt));
}

DebugLoc DebugLoc::appendInlinedAt(DebugLoc DL, DILocation *InlinedAt,
                                   LLVMContext &Ctx,
                                   DenseMap<const MDNode *, MDNode *> &Cache,
                                   bool ReplaceLast) {
  SmallVector<DILocation *, 3> InlinedAtLocations;
  DILocation *Last = InlinedAt;
  DILocation *CurInlinedAt = DL;

  // Gather all the inlined-at nodes.
  while (DILocation *IA = CurInlinedAt->getInlinedAt()) {
    // Skip any we've already built nodes for.
    if (auto *Found = Cache[IA]) {
      Last = cast<DILocation>(Found);
      break;
    }

    if (ReplaceLast && !IA->getInlinedAt())
      break;
    InlinedAtLocations.push_back(IA);
    CurInlinedAt = IA;
  }

  // Starting from the top, rebuild the nodes to point to the new inlined-at
  // location (then rebuilding the rest of the chain behind it) and update the
  // map of already-constructed inlined-at nodes.
  for (const DILocation *MD : reverse(InlinedAtLocations))
    Cache[MD] = Last = DILocation::getDistinct(
        Ctx, MD->getLine(), MD->getColumn(), MD->getScope(), Last);

  return Last;
}

/// Reparent \c Scope from \c OrigSP to \c NewSP.
static DIScope *reparentScope(LLVMContext &Ctx, DIScope *Scope,
                              DISubprogram *OrigSP, DISubprogram *NewSP,
                              DenseMap<const MDNode *, MDNode *> &Cache) {
  SmallVector<DIScope *, 3> ScopeChain;
  DIScope *Last = NewSP;
  DIScope *CurScope = Scope;
  do {
    if (auto *SP = dyn_cast<DISubprogram>(CurScope)) {
      // Don't rewrite this scope chain if it doesn't lead to the replaced SP.
      if (SP != OrigSP)
        return Scope;
      Cache.insert({OrigSP, NewSP});
      break;
    }
    if (auto *Found = Cache[CurScope]) {
      Last = cast<DIScope>(Found);
      break;
    }
    ScopeChain.push_back(CurScope);
  } while ((CurScope = CurScope->getScope().resolve()));

  // Starting from the top, rebuild the nodes to point to the new inlined-at
  // location (then rebuilding the rest of the chain behind it) and update the
  // map of already-constructed inlined-at nodes.
  for (const DIScope *MD : reverse(ScopeChain)) {
    if (auto *LB = dyn_cast<DILexicalBlock>(MD))
      Cache[MD] = Last = DILexicalBlock::getDistinct(
          Ctx, Last, LB->getFile(), LB->getLine(), LB->getColumn());
    else if (auto *LB = dyn_cast<DILexicalBlockFile>(MD))
      Cache[MD] = Last = DILexicalBlockFile::getDistinct(
          Ctx, Last, LB->getFile(), LB->getDiscriminator());
    else
      llvm_unreachable("illegal parent scope");
  }
  return Last;
}

void DebugLoc::reparentDebugInfo(Instruction &I, DISubprogram *OrigSP,
                                 DISubprogram *NewSP,
                                 DenseMap<const MDNode *, MDNode *> &Cache) {
  auto DL = I.getDebugLoc();
  if (!OrigSP || !NewSP || OrigSP == NewSP || !DL)
    return;

  // Reparent the debug location.
  auto &Ctx = I.getContext();
  DILocation *InlinedAt = DL->getInlinedAt();
  if (InlinedAt) {
    while (auto *IA = InlinedAt->getInlinedAt())
      InlinedAt = IA;
    auto NewScope =
        reparentScope(Ctx, InlinedAt->getScope(), OrigSP, NewSP, Cache);
    InlinedAt =
        DebugLoc::get(InlinedAt->getLine(), InlinedAt->getColumn(), NewScope);
  }
  I.setDebugLoc(
      DebugLoc::get(DL.getLine(), DL.getCol(),
                    reparentScope(Ctx, DL->getScope(), OrigSP, NewSP, Cache),
                    DebugLoc::appendInlinedAt(DL, InlinedAt, Ctx, Cache,
                                              ReplaceLastInlinedAt)));

  // Fix up debug variables to point to NewSP.
  auto reparentVar = [&](DILocalVariable *Var) {
    return DILocalVariable::getDistinct(
        Ctx,
        cast<DILocalScope>(
            reparentScope(Ctx, Var->getScope(), OrigSP, NewSP, Cache)),
        Var->getName(), Var->getFile(), Var->getLine(), Var->getType(),
        Var->getArg(), Var->getFlags(), Var->getAlignInBits());
  };
  if (auto *DbgValue = dyn_cast<DbgValueInst>(&I)) {
    auto *Var = DbgValue->getVariable();
    I.setOperand(2, MetadataAsValue::get(Ctx, reparentVar(Var)));
  } else if (auto *DbgDeclare = dyn_cast<DbgDeclareInst>(&I)) {
    auto *Var = DbgDeclare->getVariable();
    I.setOperand(1, MetadataAsValue::get(Ctx, reparentVar(Var)));
  }
}


#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void DebugLoc::dump() const {
  if (!Loc)
    return;

  dbgs() << getLine();
  if (getCol() != 0)
    dbgs() << ',' << getCol();
  if (DebugLoc InlinedAtDL = DebugLoc(getInlinedAt())) {
    dbgs() << " @ ";
    InlinedAtDL.dump();
  } else
    dbgs() << "\n";
}
#endif

void DebugLoc::print(raw_ostream &OS) const {
  if (!Loc)
    return;

  // Print source line info.
  auto *Scope = cast<DIScope>(getScope());
  OS << Scope->getFilename();
  OS << ':' << getLine();
  if (getCol() != 0)
    OS << ':' << getCol();

  if (DebugLoc InlinedAtDL = getInlinedAt()) {
    OS << " @[ ";
    InlinedAtDL.print(OS);
    OS << " ]";
  }
}
