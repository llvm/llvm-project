//===------ VirtualInstruction.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Tools for determining which instructions are within a statement and the
// nature of their operands.
//
//===----------------------------------------------------------------------===//

#include "polly/Support/VirtualInstruction.h"
#include "polly/Support/SCEVValidator.h"

using namespace polly;
using namespace llvm;

VirtualUse VirtualUse ::create(Scop *S, Use &U, LoopInfo *LI, bool Virtual) {
  auto *UserBB = getUseBlock(U);
  auto *UserStmt = S->getStmtFor(UserBB);
  auto *UserScope = LI->getLoopFor(UserBB);
  return create(S, UserStmt, UserScope, U.get(), Virtual);
}

VirtualUse VirtualUse::create(Scop *S, ScopStmt *UserStmt, Loop *UserScope,
                              Value *Val, bool Virtual) {
  assert(!isa<StoreInst>(Val) && "a StoreInst cannot be used");

  if (isa<BasicBlock>(Val))
    return VirtualUse(UserStmt, Val, Block, nullptr, nullptr);

  if (isa<llvm::Constant>(Val))
    return VirtualUse(UserStmt, Val, Constant, nullptr, nullptr);

  // Is the value synthesizable? If the user has been pruned
  // (UserStmt == nullptr), it is either not used anywhere or is synthesizable.
  // We assume synthesizable which practically should have the same effect.
  auto *SE = S->getSE();
  if (SE->isSCEVable(Val->getType())) {
    auto *ScevExpr = SE->getSCEVAtScope(Val, UserScope);
    if (!UserStmt || canSynthesize(Val, *UserStmt->getParent(), SE, UserScope))
      return VirtualUse(UserStmt, Val, Synthesizable, ScevExpr, nullptr);
  }

  // FIXME: Inconsistency between lookupInvariantEquivClass and
  // getRequiredInvariantLoads. Querying one of them should be enough.
  auto &RIL = S->getRequiredInvariantLoads();
  if (S->lookupInvariantEquivClass(Val) || RIL.count(dyn_cast<LoadInst>(Val)))
    return VirtualUse(UserStmt, Val, Hoisted, nullptr, nullptr);

  // ReadOnly uses may have MemoryAccesses that we want to associate with the
  // use. This is why we look for a MemoryAccess here already.
  MemoryAccess *InputMA = nullptr;
  if (UserStmt && Virtual)
    InputMA = UserStmt->lookupValueReadOf(Val);

  // Uses are read-only if they have been defined before the SCoP, i.e., they
  // cannot be written to inside the SCoP. Arguments are defined before any
  // instructions, hence also before the SCoP. If the user has been pruned
  // (UserStmt == nullptr) and is not SCEVable, assume it is read-only as it is
  // neither an intra- nor an inter-use.
  if (!UserStmt || isa<Argument>(Val))
    return VirtualUse(UserStmt, Val, ReadOnly, nullptr, InputMA);

  auto Inst = cast<Instruction>(Val);
  if (!S->contains(Inst))
    return VirtualUse(UserStmt, Val, ReadOnly, nullptr, InputMA);

  // A use is inter-statement if either it is defined in another statement, or
  // there is a MemoryAccess that reads its value that has been written by
  // another statement.
  if (InputMA || (!Virtual && !UserStmt->contains(Inst->getParent())))
    return VirtualUse(UserStmt, Val, Inter, nullptr, InputMA);

  return VirtualUse(UserStmt, Val, Intra, nullptr, nullptr);
}

void VirtualUse::print(raw_ostream &OS, bool Reproducible) const {
  OS << "User: [" << User->getBaseName() << "] ";
  switch (Kind) {
  case VirtualUse::Constant:
    OS << "Constant Op:";
    break;
  case VirtualUse::Block:
    OS << "BasicBlock Op:";
    break;
  case VirtualUse::Synthesizable:
    OS << "Synthesizable Op:";
    break;
  case VirtualUse::Hoisted:
    OS << "Hoisted load Op:";
    break;
  case VirtualUse::ReadOnly:
    OS << "Read-Only Op:";
    break;
  case VirtualUse::Intra:
    OS << "Intra Op:";
    break;
  case VirtualUse::Inter:
    OS << "Inter Op:";
    break;
  }

  if (Val) {
    OS << ' ';
    if (Reproducible)
      OS << '"' << Val->getName() << '"';
    else
      Val->print(OS, true);
  }
  if (ScevExpr) {
    OS << ' ';
    ScevExpr->print(OS);
  }
  if (InputMA && !Reproducible)
    OS << ' ' << InputMA;
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VirtualUse::dump() const {
  print(errs(), false);
  errs() << '\n';
}
#endif
