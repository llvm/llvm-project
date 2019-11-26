//===---- ReachingDefAnalysis.cpp - Reaching Def Analysis ---*- C++ -*-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
  if (skipFunction(mf.getFunction()))
    return false;
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

int ReachingDefAnalysis::getReachingDef(MachineInstr *MI, int PhysReg) {
  assert(InstIds.count(MI) && "Unexpected machine instuction.");
  int InstId = InstIds[MI];
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

MachineInstr* ReachingDefAnalysis::getReachingMIDef(MachineInstr *MI, int PhysReg) {
  return getInstFromId(MI->getParent(), getReachingDef(MI, PhysReg));
}

bool ReachingDefAnalysis::hasSameReachingDef(MachineInstr *A, MachineInstr *B,
                                             int PhysReg) {
  MachineBasicBlock *ParentA = A->getParent();
  MachineBasicBlock *ParentB = B->getParent();
  if (ParentA != ParentB)
    return false;

  return getReachingDef(A, PhysReg) == getReachingDef(B, PhysReg);
}

MachineInstr *ReachingDefAnalysis::getInstFromId(MachineBasicBlock *MBB,
                                                 int InstId) {
  assert(static_cast<size_t>(MBB->getNumber()) < MBBReachingDefs.size() &&
         "Unexpected basic block number.");
  assert(InstId < static_cast<int>(MBB->size()) &&
         "Unexpected instruction id.");

  if (InstId < 0)
    return nullptr;

  for (auto &MI : *MBB) {
    if (InstIds.count(&MI) && InstIds[&MI] == InstId)
      return &MI;
  }
  return nullptr;
}

int ReachingDefAnalysis::getClearance(MachineInstr *MI, MCPhysReg PhysReg) {
  assert(InstIds.count(MI) && "Unexpected machine instuction.");
  return InstIds[MI] - getReachingDef(MI, PhysReg);
}

void ReachingDefAnalysis::getReachingLocalUses(MachineInstr *Def, int PhysReg,
                                        SmallVectorImpl<MachineInstr*> &Uses) {
  MachineBasicBlock *MBB = Def->getParent();
  MachineBasicBlock::iterator MI = MachineBasicBlock::iterator(Def);
  while (++MI != MBB->end()) {
    for (auto &MO : MI->operands()) {
      if (!MO.isReg() || !MO.isUse() || MO.getReg() != PhysReg)
        continue;

      // If/when we find a new reaching def, we know that there's no more uses
      // of 'Def'.
      if (getReachingMIDef(&*MI, PhysReg) != Def)
        return;

      Uses.push_back(&*MI);
      if (MO.isKill())
        return;
    }
  }
}

unsigned ReachingDefAnalysis::getNumUses(MachineInstr *Def, int PhysReg) {
  SmallVector<MachineInstr*, 4> Uses;
  getReachingLocalUses(Def, PhysReg, Uses);
  return Uses.size();
}

bool ReachingDefAnalysis::isRegUsedAfter(MachineInstr *MI, int PhysReg) {
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
      return InstIds[&*Last] > InstIds[MI];
  }
  return false;
}

