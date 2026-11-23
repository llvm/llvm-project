//===-- HexagonGlobalRegion.cpp - VLIW global scheduling infrastructure ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Basic infrastructure for global scheduling. Liveness update.
// This is the least complete portion. Basically it is empty infrastructure
// to be extended and improved.
// Currently in place only trace region formation routines and non fully
// functional skeleton for incremental liveness update.
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "global_sched"
#include "HexagonGlobalRegion.h"
#include "HexagonTargetMachine.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

LivenessInfo::LivenessInfo(const TargetInstrInfo *TII,
                           const TargetRegisterInfo *TRI,
                           MachineBasicBlock *MBB)
    : TII(TII), TRI(TRI) {
  LiveIns.resize(TRI->getNumRegs());
  LiveOuts.resize(TRI->getNumRegs());

  LiveIns.reset();
  LiveOuts.reset();

  // Live-ins are simple, just copy from MBB.
  for (const auto &LI : MBB->liveins())
    setUsed(LiveIns, LI.PhysReg);

  // Live-outs are concatenation of all the BB successors.
  // As of now, we are only dealing with a-cyclic regions
  // with side exits, but no side entrances.
  for (const MachineBasicBlock *Succ : MBB->successors())
    for (const auto &LI : Succ->liveins())
      setUsed(LiveOuts, LI.PhysReg);
}

// Pessimistically check if at least one def of this register in this
// instruction (bundle or not) is done under predication.
static bool isPredicatedDef(MachineInstr *MI, unsigned Reg,
                            const HexagonInstrInfo *QII) {
  if (!MI->isBundle())
    return QII->isPredicated(*MI);
  MachineBasicBlock *Parent = MI->getParent();
  if (!Parent)
    return false;
  MachineBasicBlock::instr_iterator MII = MI->getIterator();
  MachineBasicBlock::instr_iterator MIIE = Parent->instr_end();
  for (++MII; MII != MIIE && MII->isInsideBundle(); ++MII) {
    if (!QII->isPredicated(*MII))
      continue;
    for (unsigned i = 0, e = MII->getNumOperands(); i != e; ++i) {
      const MachineOperand &MO = MII->getOperand(i);
      if (!MO.isReg())
        continue;
      if (MO.isDef() && !MO.isDead() && MO.getReg() == Reg) {
        LLVM_DEBUG(dbgs() << "\t\tCond def: "; MII->dump());
        return true;
      }
    }
  }
  return false;
}

/// Determine def/use set for MI.
/// Beware, if def is conditional, like here:
/// BUNDLE %PC<imp-def>, %R0<imp-def>, %P0<imp-use,kill>, %R16<imp-use>
///   * %R0<def> = LDriuh_cdnNotPt %P0<kill,internal>, %R16, 0;
///   * %P0<def> = C2_cmpeqi %R16, 0;
/// It is not a statefull definition of R0.
///
void LivenessInfo::parseOperands(MachineInstr *MI, BitVector &Gen,
                                 BitVector &Kill, BitVector &Use) {
  const auto *QII = static_cast<const HexagonInstrInfo *>(TII);

  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg())
      continue;
    // If it is a predicated instruction, it may, or may not
    // be setting its destination, and we do not know it
    // at the compile time.
    if (MO.isDef() && !MO.isDead()) {
      if (isPredicatedDef(MI, MO.getReg(), QII))
        LLVM_DEBUG(dbgs() << "\tConditional define of "
                          << printReg(MO.getReg(), TRI) << " in ";
                   MI->dump());
      else
        setUsed(Gen, MO.getReg());
    }
    if (MO.isKill())
      setUsed(Kill, MO.getReg());
    if (MO.isUse())
      setUsed(Use, MO.getReg());
  }
}

void LivenessInfo::parseOperandsWithReset(MachineInstr *MI, BitVector &Gen,
                                          BitVector &Kill, BitVector &Use) {
  Gen.reset();
  Use.reset();
  Kill.reset();
  parseOperands(MI, Gen, Kill, Use);
}

/// setUsed - Set the register and its sub-registers as being used.
/// Taken from RegScavenger::setUsed().
void LivenessInfo::setUsed(BitVector &Set, unsigned Reg) {
  Set.set(Reg);

  for (MCSubRegIterator SubRegs(Reg, TRI); SubRegs.isValid(); ++SubRegs)
    Set.set(*SubRegs);
}

#ifndef NDEBUG
static void dumpRI(const TargetRegisterInfo *TRI, BitVector &Set) {
  for (unsigned i = 0; i < Set.size(); i++)
    if (Set.test(i))
      LLVM_DEBUG(dbgs() << " " << printReg(i, TRI));
}
#endif

// This function incrementally updates liveness for the given BB.
// First it gathers LiveOut set, and then iterates bottom-up
// over bundles/instructions while updating live set.
void LivenessInfo::UpdateLiveness(MachineBasicBlock *MBB) {
  BitVector NewLiveIns(TRI->getNumRegs());
  BitVector NewLiveOuts(TRI->getNumRegs());
  BitVector LiveIns(TRI->getNumRegs());
  BitVector LocalGen(TRI->getNumRegs());
  BitVector LocalUse(TRI->getNumRegs());
  BitVector LocalKill(TRI->getNumRegs());

  NewLiveIns.reset();
  NewLiveOuts.reset();

  LLVM_DEBUG(dbgs() << "\n\t\tUpdateLiveness for BB(" << MBB->getNumber()
                    << ")\n");

  // Original Live-ins are simple, just copy from MBB.
  for (const auto &LI : MBB->liveins())
    setUsed(NewLiveIns, LI.PhysReg);

  // Live-outs are concatenation of all the BB successors.
  // As of now, we are only dealing with a-cyclic regions
  // with side exits, but no side entrances.
  for (const MachineBasicBlock *Succ : MBB->successors())
    for (const auto &LI : Succ->liveins())
      setUsed(NewLiveOuts, LI.PhysReg);

  LiveIns = NewLiveIns;
  // This needs to be a sequential walk, not parallel update.
  LLVM_DEBUG(dbgs() << "\t\tOriginal live ins :\t"; dumpRI(TRI, NewLiveIns);
             dbgs() << "\n");
  LLVM_DEBUG(dbgs() << "\t\tOriginal live outs:\t"; dumpRI(TRI, NewLiveOuts);
             dbgs() << "\n");

  NewLiveIns = NewLiveOuts;
  // Scan BB backwards to get exposed uses.
  // TODO: Handle predicates if needed.
  std::vector<MachineInstr *> BundleList;
  for (MachineBasicBlock::iterator MI = MBB->begin(), MIE = MBB->end();
       MI != MIE; ++MI)
    if (!MI->isDebugInstr())
      BundleList.push_back(&*MI);

  while (!BundleList.empty()) {
    MachineInstr *MI = BundleList.back();
    BundleList.pop_back();
    parseOperandsWithReset(MI, LocalGen, LocalKill, LocalUse);
    LLVM_DEBUG(dbgs() << "\t\tIncr gen:\t"; dumpRI(TRI, LocalGen);
               dbgs() << "\n");
    LLVM_DEBUG(dbgs() << "\t\tIncr use:\t"; dumpRI(TRI, LocalUse);
               dbgs() << "\n");
    // NewLiveIns = (NewLiveIns - LocalGen) U LocalUse.
    BitVector NotGen(LocalGen);
    NotGen.flip();
    NewLiveIns &= NotGen;
    NewLiveIns |= LocalUse;
  }

  LLVM_DEBUG(dbgs() << "\t\tAnswer:\t"; dumpRI(TRI, NewLiveIns);
             dbgs() << "\n");

  // TODO: Consider implementing a register aliasing filter if duplicate
  // live-in entries become problematic.

  // Set new live in.
  LLVM_DEBUG(dbgs() << "\t\tNew LiveIn       :\t");

  for (unsigned i = 0; i < LiveIns.size(); ++i) {
    if (NewLiveIns.test(i))
      LLVM_DEBUG(dbgs() << " " << printReg(i, TRI));
    if (LiveIns.test(i) == NewLiveIns.test(i))
      continue;
    if (LiveIns.test(i))
      MBB->removeLiveIn(i);
    if (NewLiveIns.test(i))
      MBB->addLiveIn(i);
  }
  LLVM_DEBUG(dbgs() << "\n");
}

void LivenessInfo::dump() {
  for (unsigned i = 0; i < LiveIns.size(); i++)
    if (LiveIns.test(i))
      LLVM_DEBUG(dbgs() << "\t\tlive-in:  " << printReg(i, TRI) << "\n");
  for (unsigned i = 0; i < LiveOuts.size(); i++)
    if (LiveOuts.test(i))
      LLVM_DEBUG(dbgs() << "\t\tlive-out: " << printReg(i, TRI) << "\n");
}

///
/// BasicBlockRegion Methods.
///
BasicBlockRegion::BasicBlockRegion(const TargetInstrInfo *TII,
                                   const TargetRegisterInfo *TRI,
                                   MachineBasicBlock *MBB)
    : TII(TII), TRI(TRI) {
  // Should be the root BB.
  Elements.push_back(MBB);
  ElementIndex[MBB] = 0;
  LiveInfo[MBB] = std::make_unique<LivenessInfo>(TII, TRI, MBB);
}

BasicBlockRegion::~BasicBlockRegion() {
  LiveInfo.clear();
  Elements.clear();
  ElementIndex.clear();
}

LivenessInfo *BasicBlockRegion::getLivenessInfoForBB(MachineBasicBlock *MBB) {
  auto It = LiveInfo.find(MBB);
  assert(It != LiveInfo.end() && "Missing Liveness info");
  assert(It->second && "Missing Liveness info");
  return It->second.get();
}

void BasicBlockRegion::addBBtoRegion(MachineBasicBlock *MBB) {
  // It is OK to have duplicates if we reparse for additional BBs.
  if (LiveInfo.find(MBB) != LiveInfo.end())
    return;
  ElementIndex[MBB] = static_cast<unsigned>(Elements.size());
  Elements.push_back(MBB);
  LiveInfo[MBB] = std::make_unique<LivenessInfo>(TII, TRI, MBB);
}
