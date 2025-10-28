//===------ VirtualInstruction.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tools for determining which instructions are within a statement and the
// nature of their operands.
//
//===----------------------------------------------------------------------===//

#include "polly/Support/VirtualInstruction.h"

using namespace polly;
using namespace llvm;

VirtualUse VirtualUse::create(Scop *S, const Use &U, LoopInfo *LI,
                              bool Virtual) {
  auto *UserBB = getUseBlock(U);
  Loop *UserScope = LI->getLoopFor(UserBB);
  Instruction *UI = dyn_cast<Instruction>(U.getUser());
  ScopStmt *UserStmt = S->getStmtFor(UI);

  // Uses by PHI nodes are always reading values written by other statements,
  // except it is within a region statement.
  if (PHINode *PHI = dyn_cast<PHINode>(UI)) {
    // Handle PHI in exit block.
    if (S->getRegion().getExit() == PHI->getParent())
      return VirtualUse(UserStmt, U.get(), Inter, nullptr, nullptr);

    if (UserStmt->getEntryBlock() != PHI->getParent())
      return VirtualUse(UserStmt, U.get(), Intra, nullptr, nullptr);

    // The MemoryAccess is expected to be set if @p Virtual is true.
    MemoryAccess *IncomingMA = nullptr;
    if (Virtual) {
      if (const ScopArrayInfo *SAI =
              S->getScopArrayInfoOrNull(PHI, MemoryKind::PHI)) {
        IncomingMA = S->getPHIRead(SAI);
        assert(IncomingMA->getStatement() == UserStmt);
      }
    }

    return VirtualUse(UserStmt, U.get(), Inter, nullptr, IncomingMA);
  }

  return create(S, UserStmt, UserScope, U.get(), Virtual);
}

VirtualUse VirtualUse::create(Scop *S, ScopStmt *UserStmt, Loop *UserScope,
                              Value *Val, bool Virtual) {
  assert(!isa<StoreInst>(Val) && "a StoreInst cannot be used");

  if (isa<BasicBlock>(Val))
    return VirtualUse(UserStmt, Val, Block, nullptr, nullptr);

  if (isa<llvm::Constant>(Val) || isa<MetadataAsValue>(Val) ||
      isa<InlineAsm>(Val))
    return VirtualUse(UserStmt, Val, Constant, nullptr, nullptr);

  // Is the value synthesizable? If the user has been pruned
  // (UserStmt == nullptr), it is either not used anywhere or is synthesizable.
  // We assume synthesizable which practically should have the same effect.
  auto *SE = S->getSE();
  if (SE->isSCEVable(Val->getType())) {
    const SCEV *ScevExpr = SE->getSCEVAtScope(Val, UserScope);
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
  if (InputMA || (!Virtual && UserStmt != S->getStmtFor(Inst)))
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
LLVM_DUMP_METHOD void VirtualUse::dump() const {
  print(errs(), false);
  errs() << '\n';
}
#endif

void VirtualInstruction::print(raw_ostream &OS, bool Reproducible) const {
  if (!Stmt || !Inst) {
    OS << "[null VirtualInstruction]";
    return;
  }

  OS << "[" << Stmt->getBaseName() << "]";
  Inst->print(OS, !Reproducible);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void VirtualInstruction::dump() const {
  print(errs(), false);
  errs() << '\n';
}
#endif

/// Return true if @p Inst cannot be removed, even if it is nowhere referenced.
static bool isRoot(const Instruction *Inst) {
  // The store is handled by its MemoryAccess. The load must be reached from the
  // roots in order to be marked as used.
  if (isa<LoadInst>(Inst) || isa<StoreInst>(Inst))
    return false;

  // Terminator instructions (in region statements) are required for control
  // flow.
  if (Inst->isTerminator())
    return true;

  // Writes to memory must be honored.
  if (Inst->mayWriteToMemory())
    return true;

  return false;
}

/// Return true for MemoryAccesses that cannot be removed because it represents
/// an llvm::Value that is used after the SCoP.
static bool isEscaping(MemoryAccess *MA) {
  assert(MA->isOriginalValueKind());
  Scop *S = MA->getStatement()->getParent();
  return S->isEscaping(cast<Instruction>(MA->getAccessValue()));
}

/// Add non-removable virtual instructions in @p Stmt to @p RootInsts.
static void
addInstructionRoots(ScopStmt *Stmt,
                    SmallVectorImpl<VirtualInstruction> &RootInsts) {
  if (!Stmt->isBlockStmt()) {
    // In region statements the terminator statement and all statements that
    // are not in the entry block cannot be eliminated and consequently must
    // be roots.
    RootInsts.emplace_back(Stmt,
                           Stmt->getRegion()->getEntry()->getTerminator());
    for (BasicBlock *BB : Stmt->getRegion()->blocks())
      if (Stmt->getRegion()->getEntry() != BB)
        for (Instruction &Inst : *BB)
          RootInsts.emplace_back(Stmt, &Inst);
    return;
  }

  for (Instruction *Inst : Stmt->getInstructions())
    if (isRoot(Inst))
      RootInsts.emplace_back(Stmt, Inst);
}

/// Add non-removable memory accesses in @p Stmt to @p RootInsts.
///
/// @param Local If true, all writes are assumed to escape. markAndSweep
/// algorithms can use this to be applicable to a single ScopStmt only without
/// the risk of removing definitions required by other statements.
///              If false, only writes for SCoP-escaping values are roots.  This
///              is global mode, where such writes must be marked by theirs uses
///              in order to be reachable.
static void addAccessRoots(ScopStmt *Stmt,
                           SmallVectorImpl<MemoryAccess *> &RootAccs,
                           bool Local) {
  for (auto *MA : *Stmt) {
    if (!MA->isWrite())
      continue;

    // Writes to arrays are always used.
    if (MA->isLatestArrayKind())
      RootAccs.push_back(MA);

    // Values are roots if they are escaping.
    else if (MA->isLatestValueKind()) {
      if (Local || isEscaping(MA))
        RootAccs.push_back(MA);
    }

    // Exit phis are, by definition, escaping.
    else if (MA->isLatestExitPHIKind())
      RootAccs.push_back(MA);

    // phi writes are only roots if we are not visiting the statement
    // containing the PHINode.
    else if (Local && MA->isLatestPHIKind())
      RootAccs.push_back(MA);
  }
}

/// Determine all instruction and access roots.
static void addRoots(ScopStmt *Stmt,
                     SmallVectorImpl<VirtualInstruction> &RootInsts,
                     SmallVectorImpl<MemoryAccess *> &RootAccs, bool Local) {
  addInstructionRoots(Stmt, RootInsts);
  addAccessRoots(Stmt, RootAccs, Local);
}

/// Mark accesses and instructions as used if they are reachable from a root,
/// walking the operand trees.
///
/// @param S              The SCoP to walk.
/// @param LI             The LoopInfo Analysis.
/// @param RootInsts      List of root instructions.
/// @param RootAccs       List of root accesses.
/// @param UsesInsts[out] Receives all reachable instructions, including the
/// roots.
/// @param UsedAccs[out]  Receives all reachable accesses, including the roots.
/// @param OnlyLocal      If non-nullptr, restricts walking to a single
/// statement.
static void walkReachable(Scop *S, LoopInfo *LI,
                          ArrayRef<VirtualInstruction> RootInsts,
                          ArrayRef<MemoryAccess *> RootAccs,
                          DenseSet<VirtualInstruction> &UsedInsts,
                          DenseSet<MemoryAccess *> &UsedAccs,
                          ScopStmt *OnlyLocal = nullptr) {
  UsedInsts.clear();
  UsedAccs.clear();

  SmallVector<VirtualInstruction, 32> WorklistInsts;
  SmallVector<MemoryAccess *, 32> WorklistAccs;

  WorklistInsts.append(RootInsts.begin(), RootInsts.end());
  WorklistAccs.append(RootAccs.begin(), RootAccs.end());

  auto AddToWorklist = [&](VirtualUse VUse) {
    switch (VUse.getKind()) {
    case VirtualUse::Block:
    case VirtualUse::Constant:
    case VirtualUse::Synthesizable:
    case VirtualUse::Hoisted:
      break;
    case VirtualUse::ReadOnly:
      // Read-only scalars only have MemoryAccesses if ModelReadOnlyScalars is
      // enabled.
      if (!VUse.getMemoryAccess())
        break;
      [[fallthrough]];
    case VirtualUse::Inter:
      assert(VUse.getMemoryAccess());
      WorklistAccs.push_back(VUse.getMemoryAccess());
      break;
    case VirtualUse::Intra:
      WorklistInsts.emplace_back(VUse.getUser(),
                                 cast<Instruction>(VUse.getValue()));
      break;
    }
  };

  while (true) {
    // We have two worklists to process: Only when the MemoryAccess worklist is
    // empty, we process the instruction worklist.

    while (!WorklistAccs.empty()) {
      auto *Acc = WorklistAccs.pop_back_val();

      ScopStmt *Stmt = Acc->getStatement();
      if (OnlyLocal && Stmt != OnlyLocal)
        continue;

      auto Inserted = UsedAccs.insert(Acc);
      if (!Inserted.second)
        continue;

      if (Acc->isRead()) {
        const ScopArrayInfo *SAI = Acc->getScopArrayInfo();

        if (Acc->isLatestValueKind()) {
          MemoryAccess *DefAcc = S->getValueDef(SAI);

          // Accesses to read-only values do not have a definition.
          if (DefAcc)
            WorklistAccs.push_back(S->getValueDef(SAI));
        }

        if (Acc->isLatestAnyPHIKind()) {
          auto IncomingMAs = S->getPHIIncomings(SAI);
          WorklistAccs.append(IncomingMAs.begin(), IncomingMAs.end());
        }
      }

      if (Acc->isWrite()) {
        if (Acc->isOriginalValueKind() ||
            (Acc->isOriginalArrayKind() && Acc->getAccessValue())) {
          Loop *Scope = Stmt->getSurroundingLoop();
          VirtualUse VUse =
              VirtualUse::create(S, Stmt, Scope, Acc->getAccessValue(), true);
          AddToWorklist(VUse);
        }

        if (Acc->isOriginalAnyPHIKind()) {
          for (auto Incoming : Acc->getIncoming()) {
            VirtualUse VUse = VirtualUse::create(
                S, Stmt, LI->getLoopFor(Incoming.first), Incoming.second, true);
            AddToWorklist(VUse);
          }
        }

        if (Acc->isOriginalArrayKind())
          WorklistInsts.emplace_back(Stmt, Acc->getAccessInstruction());
      }
    }

    // If both worklists are empty, stop walking.
    if (WorklistInsts.empty())
      break;

    VirtualInstruction VInst = WorklistInsts.pop_back_val();
    ScopStmt *Stmt = VInst.getStmt();
    Instruction *Inst = VInst.getInstruction();

    // Do not process statements other than the local.
    if (OnlyLocal && Stmt != OnlyLocal)
      continue;

    auto InsertResult = UsedInsts.insert(VInst);
    if (!InsertResult.second)
      continue;

    // Add all operands to the worklists.
    PHINode *PHI = dyn_cast<PHINode>(Inst);
    if (PHI && PHI->getParent() == Stmt->getEntryBlock()) {
      if (MemoryAccess *PHIRead = Stmt->lookupPHIReadOf(PHI))
        WorklistAccs.push_back(PHIRead);
    } else {
      for (VirtualUse VUse : VInst.operands())
        AddToWorklist(VUse);
    }

    // If there is an array access, also add its MemoryAccesses to the worklist.
    const MemoryAccessList *Accs = Stmt->lookupArrayAccessesFor(Inst);
    if (!Accs)
      continue;

    for (MemoryAccess *Acc : *Accs)
      WorklistAccs.push_back(Acc);
  }
}

void polly::markReachable(Scop *S, LoopInfo *LI,
                          DenseSet<VirtualInstruction> &UsedInsts,
                          DenseSet<MemoryAccess *> &UsedAccs,
                          ScopStmt *OnlyLocal) {
  SmallVector<VirtualInstruction, 32> RootInsts;
  SmallVector<MemoryAccess *, 32> RootAccs;

  if (OnlyLocal) {
    addRoots(OnlyLocal, RootInsts, RootAccs, true);
  } else {
    for (auto &Stmt : *S)
      addRoots(&Stmt, RootInsts, RootAccs, false);
  }

  walkReachable(S, LI, RootInsts, RootAccs, UsedInsts, UsedAccs, OnlyLocal);
}
