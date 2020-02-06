//===---- ReachingDefAnalysis.cpp - Reaching Def Analysis ---*- C++ -*-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/ReachingDefAnalysis.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "reaching-deps-analysis"

char ReachingDefAnalysis::ID = 0;
INITIALIZE_PASS(ReachingDefAnalysis, DEBUG_TYPE, "ReachingDefAnalysis", false,
                true)

void ReachingDefAnalysis::enterBasicBlock(
    const LoopTraversal::TraversedMBBInfo &TraversedMBB) {

  MachineBasicBlock *MBB = TraversedMBB.MBB;
  unsigned MBBNumber = MBB->getNumber();
  assert(MBBNumber < MBBReachingDefs.size() &&
         "Unexpected basic block number.");
  MBBReachingDefs[MBBNumber].resize(NumRegUnits);

  // Reset instruction counter in each basic block.
  CurInstr = 0;

  // Set up LiveRegs to represent registers entering MBB.
  // Default values are 'nothing happened a long time ago'.
  if (LiveRegs.empty())
    LiveRegs.assign(NumRegUnits, ReachingDefDefaultVal);

  // This is the entry block.
  if (MBB->pred_empty()) {
    for (const auto &LI : MBB->liveins()) {
      for (MCRegUnitIterator Unit(LI.PhysReg, TRI); Unit.isValid(); ++Unit) {
        // Treat function live-ins as if they were defined just before the first
        // instruction.  Usually, function arguments are set up immediately
        // before the call.
        LiveRegs[*Unit] = -1;
        MBBReachingDefs[MBBNumber][*Unit].push_back(LiveRegs[*Unit]);
      }
    }
    LLVM_DEBUG(dbgs() << printMBBReference(*MBB) << ": entry\n");
    return;
  }

  // Try to coalesce live-out registers from predecessors.
  for (MachineBasicBlock *pred : MBB->predecessors()) {
    assert(unsigned(pred->getNumber()) < MBBOutRegsInfos.size() &&
           "Should have pre-allocated MBBInfos for all MBBs");
    const LiveRegsDefInfo &Incoming = MBBOutRegsInfos[pred->getNumber()];
    // Incoming is null if this is a backedge from a BB
    // we haven't processed yet
    if (Incoming.empty())
      continue;

    for (unsigned Unit = 0; Unit != NumRegUnits; ++Unit) {
      // Use the most recent predecessor def for each register.
      LiveRegs[Unit] = std::max(LiveRegs[Unit], Incoming[Unit]);
      if ((LiveRegs[Unit] != ReachingDefDefaultVal))
        MBBReachingDefs[MBBNumber][Unit].push_back(LiveRegs[Unit]);
    }
  }

  LLVM_DEBUG(dbgs() << printMBBReference(*MBB)
                    << (!TraversedMBB.IsDone ? ": incomplete\n"
                                             : ": all preds known\n"));
}

void ReachingDefAnalysis::leaveBasicBlock(
    const LoopTraversal::TraversedMBBInfo &TraversedMBB) {
  assert(!LiveRegs.empty() && "Must enter basic block first.");
  unsigned MBBNumber = TraversedMBB.MBB->getNumber();
  assert(MBBNumber < MBBOutRegsInfos.size() &&
         "Unexpected basic block number.");
  // Save register clearances at end of MBB - used by enterBasicBlock().
  MBBOutRegsInfos[MBBNumber] = LiveRegs;

  // While processing the basic block, we kept `Def` relative to the start
  // of the basic block for convenience. However, future use of this information
  // only cares about the clearance from the end of the block, so adjust
  // everything to be relative to the end of the basic block.
  for (int &OutLiveReg : MBBOutRegsInfos[MBBNumber])
    OutLiveReg -= CurInstr;
  LiveRegs.clear();
}

void ReachingDefAnalysis::processDefs(MachineInstr *MI) {
  assert(!MI->isDebugInstr() && "Won't process debug instructions");

  unsigned MBBNumber = MI->getParent()->getNumber();
  assert(MBBNumber < MBBReachingDefs.size() &&
         "Unexpected basic block number.");
  const MCInstrDesc &MCID = MI->getDesc();
  for (unsigned i = 0,
                e = MI->isVariadic() ? MI->getNumOperands() : MCID.getNumDefs();
       i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || !MO.getReg())
      continue;
    if (MO.isUse())
      continue;
    for (MCRegUnitIterator Unit(MO.getReg(), TRI); Unit.isValid(); ++Unit) {
      // This instruction explicitly defines the current reg unit.
      LLVM_DEBUG(dbgs() << printReg(MO.getReg(), TRI) << ":\t" << CurInstr
                        << '\t' << *MI);

      // How many instructions since this reg unit was last written?
      LiveRegs[*Unit] = CurInstr;
      MBBReachingDefs[MBBNumber][*Unit].push_back(CurInstr);
    }
  }
  InstIds[MI] = CurInstr;
  ++CurInstr;
}

void ReachingDefAnalysis::processBasicBlock(
    const LoopTraversal::TraversedMBBInfo &TraversedMBB) {
  enterBasicBlock(TraversedMBB);
  for (MachineInstr &MI : *TraversedMBB.MBB) {
    if (!MI.isDebugInstr())
      processDefs(&MI);
  }
  leaveBasicBlock(TraversedMBB);
}

bool ReachingDefAnalysis::runOnMachineFunction(MachineFunction &mf) {
  MF = &mf;
  TRI = MF->getSubtarget().getRegisterInfo();

  LiveRegs.clear();
  NumRegUnits = TRI->getNumRegUnits();

  MBBReachingDefs.resize(mf.getNumBlockIDs());

  LLVM_DEBUG(dbgs() << "********** REACHING DEFINITION ANALYSIS **********\n");

  // Initialize the MBBOutRegsInfos
  MBBOutRegsInfos.resize(mf.getNumBlockIDs());

  // Traverse the basic blocks.
  LoopTraversal Traversal;
  LoopTraversal::TraversalOrder TraversedMBBOrder = Traversal.traverse(mf);
  for (LoopTraversal::TraversedMBBInfo TraversedMBB : TraversedMBBOrder) {
    processBasicBlock(TraversedMBB);
  }

  // Sorting all reaching defs found for a ceartin reg unit in a given BB.
  for (MBBDefsInfo &MBBDefs : MBBReachingDefs) {
    for (MBBRegUnitDefs &RegUnitDefs : MBBDefs)
      llvm::sort(RegUnitDefs);
  }

  return false;
}

void ReachingDefAnalysis::releaseMemory() {
  // Clear the internal vectors.
  MBBOutRegsInfos.clear();
  MBBReachingDefs.clear();
  InstIds.clear();
}

int ReachingDefAnalysis::getReachingDef(MachineInstr *MI, int PhysReg) const {
  assert(InstIds.count(MI) && "Unexpected machine instuction.");
  int InstId = InstIds.lookup(MI);
  int DefRes = ReachingDefDefaultVal;
  unsigned MBBNumber = MI->getParent()->getNumber();
  assert(MBBNumber < MBBReachingDefs.size() &&
         "Unexpected basic block number.");
  int LatestDef = ReachingDefDefaultVal;
  for (MCRegUnitIterator Unit(PhysReg, TRI); Unit.isValid(); ++Unit) {
    for (int Def : MBBReachingDefs[MBBNumber][*Unit]) {
      if (Def >= InstId)
        break;
      DefRes = Def;
    }
    LatestDef = std::max(LatestDef, DefRes);
  }
  return LatestDef;
}

MachineInstr* ReachingDefAnalysis::getReachingMIDef(MachineInstr *MI,
                                                    int PhysReg) const {
  return getInstFromId(MI->getParent(), getReachingDef(MI, PhysReg));
}

bool ReachingDefAnalysis::hasSameReachingDef(MachineInstr *A, MachineInstr *B,
                                             int PhysReg) const {
  MachineBasicBlock *ParentA = A->getParent();
  MachineBasicBlock *ParentB = B->getParent();
  if (ParentA != ParentB)
    return false;

  return getReachingDef(A, PhysReg) == getReachingDef(B, PhysReg);
}

MachineInstr *ReachingDefAnalysis::getInstFromId(MachineBasicBlock *MBB,
                                                 int InstId) const {
  assert(static_cast<size_t>(MBB->getNumber()) < MBBReachingDefs.size() &&
         "Unexpected basic block number.");
  assert(InstId < static_cast<int>(MBB->size()) &&
         "Unexpected instruction id.");

  if (InstId < 0)
    return nullptr;

  for (auto &MI : *MBB) {
    auto F = InstIds.find(&MI);
    if (F != InstIds.end() && F->second == InstId)
      return &MI;
  }

  return nullptr;
}

int
ReachingDefAnalysis::getClearance(MachineInstr *MI, MCPhysReg PhysReg) const {
  assert(InstIds.count(MI) && "Unexpected machine instuction.");
  return InstIds.lookup(MI) - getReachingDef(MI, PhysReg);
}

bool
ReachingDefAnalysis::hasLocalDefBefore(MachineInstr *MI, int PhysReg) const {
  return getReachingDef(MI, PhysReg) >= 0;
}

void ReachingDefAnalysis::getReachingLocalUses(MachineInstr *Def, int PhysReg,
                                               InstSet &Uses) const {
  MachineBasicBlock *MBB = Def->getParent();
  MachineBasicBlock::iterator MI = MachineBasicBlock::iterator(Def);
  while (++MI != MBB->end()) {
    if (MI->isDebugInstr())
      continue;

    // If/when we find a new reaching def, we know that there's no more uses
    // of 'Def'.
    if (getReachingMIDef(&*MI, PhysReg) != Def)
      return;

    for (auto &MO : MI->operands()) {
      if (!MO.isReg() || !MO.isUse() || MO.getReg() != PhysReg)
        continue;

      Uses.insert(&*MI);
      if (MO.isKill())
        return;
    }
  }
}

bool
ReachingDefAnalysis::getLiveInUses(MachineBasicBlock *MBB, int PhysReg,
                                   InstSet &Uses) const {
  for (auto &MI : *MBB) {
    if (MI.isDebugInstr())
      continue;
    for (auto &MO : MI.operands()) {
      if (!MO.isReg() || !MO.isUse() || MO.getReg() != PhysReg)
        continue;
      if (getReachingDef(&MI, PhysReg) >= 0)
        return false;
      Uses.insert(&MI);
    }
  }
  return isReachingDefLiveOut(&MBB->back(), PhysReg);
}

void
ReachingDefAnalysis::getGlobalUses(MachineInstr *MI, int PhysReg,
                                   InstSet &Uses) const {
  MachineBasicBlock *MBB = MI->getParent();

  // Collect the uses that each def touches within the block.
  getReachingLocalUses(MI, PhysReg, Uses);

  // Handle live-out values.
  if (auto *LiveOut = getLocalLiveOutMIDef(MI->getParent(), PhysReg)) {
    if (LiveOut != MI)
      return;

    SmallVector<MachineBasicBlock*, 4> ToVisit;
    ToVisit.insert(ToVisit.begin(), MBB->successors().begin(),
                   MBB->successors().end());
    SmallPtrSet<MachineBasicBlock*, 4>Visited;
    while (!ToVisit.empty()) {
      MachineBasicBlock *MBB = ToVisit.back();
      ToVisit.pop_back();
      if (Visited.count(MBB) || !MBB->isLiveIn(PhysReg))
        continue;
      if (getLiveInUses(MBB, PhysReg, Uses))
        ToVisit.insert(ToVisit.end(), MBB->successors().begin(),
                       MBB->successors().end());
      Visited.insert(MBB);
    }
  }
}

bool ReachingDefAnalysis::isRegUsedAfter(MachineInstr *MI, int PhysReg) const {
  MachineBasicBlock *MBB = MI->getParent();
  LivePhysRegs LiveRegs(*TRI);
  LiveRegs.addLiveOuts(*MBB);

  // Yes if the register is live out of the basic block.
  if (LiveRegs.contains(PhysReg))
    return true;

  // Walk backwards through the block to see if the register is live at some
  // point.
  for (auto Last = MBB->rbegin(), End = MBB->rend(); Last != End; ++Last) {
    LiveRegs.stepBackward(*Last);
    if (LiveRegs.contains(PhysReg))
      return InstIds.lookup(&*Last) > InstIds.lookup(MI);
  }
  return false;
}

bool ReachingDefAnalysis::isRegDefinedAfter(MachineInstr *MI,
                                            int PhysReg) const {
  MachineBasicBlock *MBB = MI->getParent();
  if (getReachingDef(MI, PhysReg) != getReachingDef(&MBB->back(), PhysReg))
    return true;

  if (auto *Def = getLocalLiveOutMIDef(MBB, PhysReg))
    return Def == getReachingMIDef(MI, PhysReg);

  return false;
}

bool
ReachingDefAnalysis::isReachingDefLiveOut(MachineInstr *MI, int PhysReg) const {
  MachineBasicBlock *MBB = MI->getParent();
  LivePhysRegs LiveRegs(*TRI);
  LiveRegs.addLiveOuts(*MBB);
  if (!LiveRegs.contains(PhysReg))
    return false;

  MachineInstr *Last = &MBB->back();
  int Def = getReachingDef(MI, PhysReg);
  if (getReachingDef(Last, PhysReg) != Def)
    return false;

  // Finally check that the last instruction doesn't redefine the register.
  for (auto &MO : Last->operands())
    if (MO.isReg() && MO.isDef() && MO.getReg() == PhysReg)
      return false;

  return true;
}

MachineInstr* ReachingDefAnalysis::getLocalLiveOutMIDef(MachineBasicBlock *MBB,
                                                        int PhysReg) const {
  LivePhysRegs LiveRegs(*TRI);
  LiveRegs.addLiveOuts(*MBB);
  if (!LiveRegs.contains(PhysReg))
    return nullptr;

  MachineInstr *Last = &MBB->back();
  int Def = getReachingDef(Last, PhysReg);
  for (auto &MO : Last->operands())
    if (MO.isReg() && MO.isDef() && MO.getReg() == PhysReg)
      return Last;

  return Def < 0 ? nullptr : getInstFromId(MBB, Def);
}

static bool mayHaveSideEffects(MachineInstr &MI) {
  return MI.mayLoadOrStore() || MI.mayRaiseFPException() ||
         MI.hasUnmodeledSideEffects() || MI.isTerminator() ||
         MI.isCall() || MI.isBarrier() || MI.isBranch() || MI.isReturn();
}

// Can we safely move 'From' to just before 'To'? To satisfy this, 'From' must
// not define a register that is used by any instructions, after and including,
// 'To'. These instructions also must not redefine any of Froms operands.
template<typename Iterator>
bool ReachingDefAnalysis::isSafeToMove(MachineInstr *From,
                                       MachineInstr *To) const {
  if (From->getParent() != To->getParent())
    return false;

  SmallSet<int, 2> Defs;
  // First check that From would compute the same value if moved.
  for (auto &MO : From->operands()) {
    if (!MO.isReg() || MO.isUndef() || !MO.getReg())
      continue;
    if (MO.isDef())
      Defs.insert(MO.getReg());
    else if (!hasSameReachingDef(From, To, MO.getReg()))
      return false;
  }

  // Now walk checking that the rest of the instructions will compute the same
  // value and that we're not overwriting anything. Don't move the instruction
  // past any memory, control-flow or other ambigious instructions.
  for (auto I = ++Iterator(From), E = Iterator(To); I != E; ++I) {
    if (mayHaveSideEffects(*I))
      return false;
    for (auto &MO : I->operands())
      if (MO.isReg() && MO.getReg() && Defs.count(MO.getReg()))
        return false;
  }
  return true;
}

bool ReachingDefAnalysis::isSafeToMoveForwards(MachineInstr *From,
                                               MachineInstr *To) const {
  return isSafeToMove<MachineBasicBlock::reverse_iterator>(From, To);
}

bool ReachingDefAnalysis::isSafeToMoveBackwards(MachineInstr *From,
                                                MachineInstr *To) const {
  return isSafeToMove<MachineBasicBlock::iterator>(From, To);
}

bool ReachingDefAnalysis::isSafeToRemove(MachineInstr *MI,
                                         InstSet &ToRemove) const {
  SmallPtrSet<MachineInstr*, 1> Ignore;
  SmallPtrSet<MachineInstr*, 2> Visited;
  return isSafeToRemove(MI, Visited, ToRemove, Ignore);
}

bool
ReachingDefAnalysis::isSafeToRemove(MachineInstr *MI, InstSet &ToRemove,
                                    InstSet &Ignore) const {
  SmallPtrSet<MachineInstr*, 2> Visited;
  return isSafeToRemove(MI, Visited, ToRemove, Ignore);
}

bool
ReachingDefAnalysis::isSafeToRemove(MachineInstr *MI, InstSet &Visited,
                                    InstSet &ToRemove, InstSet &Ignore) const {
  if (Visited.count(MI) || Ignore.count(MI))
    return true;
  else if (mayHaveSideEffects(*MI)) {
    // Unless told to ignore the instruction, don't remove anything which has
    // side effects.
    return false;
  }

  Visited.insert(MI);
  for (auto &MO : MI->operands()) {
    if (!MO.isReg() || MO.isUse() || MO.getReg() == 0)
      continue;

    SmallPtrSet<MachineInstr*, 4> Uses;
    getGlobalUses(MI, MO.getReg(), Uses);

    for (auto I : Uses) {
      if (Ignore.count(I) || ToRemove.count(I))
        continue;
      if (!isSafeToRemove(I, Visited, ToRemove, Ignore))
        return false;
    }
  }
  ToRemove.insert(MI);
  return true;
}

bool ReachingDefAnalysis::isSafeToDefRegAt(MachineInstr *MI,
                                           int PhysReg) const {
  SmallPtrSet<MachineInstr*, 1> Ignore;
  return isSafeToDefRegAt(MI, PhysReg, Ignore);
}

bool ReachingDefAnalysis::isSafeToDefRegAt(MachineInstr *MI, int PhysReg,
                                           InstSet &Ignore) const {
  // Check for any uses of the register after MI.
  if (isRegUsedAfter(MI, PhysReg)) {
    if (auto *Def = getReachingMIDef(MI, PhysReg)) {
      SmallPtrSet<MachineInstr*, 2> Uses;
      getReachingLocalUses(Def, PhysReg, Uses);
      for (auto *Use : Uses)
        if (!Ignore.count(Use))
          return false;
    } else
      return false;
  }

  MachineBasicBlock *MBB = MI->getParent();
  // Check for any defs after MI.
  if (isRegDefinedAfter(MI, PhysReg)) {
    auto I = MachineBasicBlock::iterator(MI);
    for (auto E = MBB->end(); I != E; ++I) {
      if (Ignore.count(&*I))
        continue;
      for (auto &MO : I->operands())
        if (MO.isReg() && MO.isDef() && MO.getReg() == PhysReg)
          return false;
    }
  }
  return true;
}
