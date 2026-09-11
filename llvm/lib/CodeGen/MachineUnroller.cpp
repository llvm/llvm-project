//===- MachineUnroller.cpp - Machine loop unrolling utilities -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file implements loop unrolling functionality at the machine instruction
// (MI) level.
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineUnroller.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "machine-unroller"

// This is a utility for unrolling loops at MI level.
// It only unroll loops with the run-time trip count and
// with a single basic block.
//
// After unrolling, the loop structure will be the following:
//
// Original LoopPreheader
// Unrolled LoopPreheader
//   Unrolled Loop
// Unrolled LoopExit
// Remainder LoopPreheader
//   Remainder Loop
// Remainder LoopExit
// Original LoopExit

void MachineUnroller::init(MachineLoop *loop, unsigned unrollFactor) {
  L = loop;
  UnrollFactor = unrollFactor;
  OrigHeader = L->getHeader();
  OrigPreheader = L->getLoopPreheader();
  OrigLoopExit = L->getExitBlock();
  LoopBBs.clear();
  ExitBBLiveIns.clear();
}

bool MachineUnroller::canUnroll() {
  // Only loops with a single basic block are handled. Also, the loop must
  // be analyzable using analyzeBranch. It's the responsibility of the caller of
  // this function to make sure that these requirement are met.
  assert(L->getNumBlocks() == 1 && "Only loops with single basic block can be"
                                   "unrolled!!");
  if (!isPowerOf2_32(UnrollFactor)) {
    LLVM_DEBUG(dbgs() << "Can't Unroll!! UnrollFactor must be a power of 2.");
    return false;
  }

  if (!TII->analyzeLoopForPipelining(L->getTopBlock()))
    return false;

  // Get loop trip count. Compile-time trip count is not handled.
  LC = getLoopCount(*L->getTopBlock());
  return Register::isVirtualRegister(LC);
}

/// Create empty basic blocks for the unrolled/remainder loops and
/// add them to the CFG. Some the BBs from the original loop are reused
/// and their successors/predecessors are changed as needed.
void MachineUnroller::createUnrolledLoopStruct() {
  // Create basic blocks for the Unrolled Loop.
  ULPreheader = MF->CreateMachineBasicBlock();
  MF->insert(OrigHeader->getIterator(), ULPreheader);

  ULHeader = MF->CreateMachineBasicBlock();
  ULHeader->setAlignment(OrigHeader->getAlignment());
  MF->insert(OrigHeader->getIterator(), ULHeader);

  ULPreheader->addSuccessor(ULHeader);
  ULHeader->addSuccessor(ULHeader);
  OrigPreheader->replaceSuccessor(OrigHeader, ULPreheader);

  // Create basic blocks for the Remainder Loop. The original loop header
  // is used as the remainder loop header. The loop trip count is adjusted
  // later to the appropriate value.
  RLHeader = OrigHeader;

  ULExit = MF->CreateMachineBasicBlock();
  MF->insert(RLHeader->getIterator(), ULExit);

  RLPreheader = MF->CreateMachineBasicBlock();
  MF->insert(RLHeader->getIterator(), RLPreheader);

  RLExit = MF->CreateMachineBasicBlock();
  MF->insert(++RLHeader->getIterator(), RLExit);

  ULExit->addSuccessor(RLPreheader);
  RLPreheader->addSuccessor(RLHeader);

  ULHeader->addSuccessor(ULExit);
  OrigPreheader->addSuccessor(ULExit);
  ULExit->addSuccessor(RLExit);
  RLExit->addSuccessor(OrigLoopExit);
  RLHeader->replaceSuccessor(OrigLoopExit, RLExit);

  LoopBBs.push_back(ULPreheader);
  LoopBBs.push_back(ULHeader);
  LoopBBs.push_back(ULExit);
  LoopBBs.push_back(RLPreheader);
  LoopBBs.push_back(RLHeader);
  LoopBBs.push_back(RLExit);

  // Update the Phis in RLHeader (same as OrigHeader) and
  // OrigLoopExit to use the new predecessors.
  for (MachineBasicBlock::iterator I = RLHeader->instr_begin(),
                                   E = RLHeader->getFirstNonPHI();
       I != E; ++I) {
    MachineInstr *Phi = &*I;
    for (unsigned i = 1, e = Phi->getNumOperands(); i != e; i += 2)
      if (Phi->getOperand(i + 1).getMBB() != RLHeader)
        Phi->getOperand(i + 1).setMBB(RLPreheader);
  }

  for (MachineBasicBlock::iterator I = OrigLoopExit->instr_begin(),
                                   E = OrigLoopExit->getFirstNonPHI();
       I != E; ++I) {
    MachineInstr *Phi = &*I;
    for (unsigned i = 1, e = Phi->getNumOperands(); i != e; i += 2)
      if (Phi->getOperand(i + 1).getMBB() == RLHeader)
        Phi->getOperand(i + 1).setMBB(RLExit);
  }
}

/// Return the Phi Operand that comes from outside the loop.
static MachineOperand &getInitPhiOp(MachineInstr *Phi,
                                    MachineBasicBlock *LoopBB) {
  for (unsigned i = 1, e = Phi->getNumOperands(); i != e; i += 2)
    if (Phi->getOperand(i + 1).getMBB() != LoopBB)
      return Phi->getOperand(i);
  llvm_unreachable("Unexpected Phi structure.");
}

/// Return the Phi register value that comes from outside the loop.
static unsigned getInitPhiReg(MachineInstr *Phi, MachineBasicBlock *LoopBB) {
  for (unsigned i = 1, e = Phi->getNumOperands(); i != e; i += 2)
    if (Phi->getOperand(i + 1).getMBB() != LoopBB)
      return Phi->getOperand(i).getReg();
  llvm_unreachable("Unexpected Phi structure.");
}

/// Return the Phi Operand that comes from the loop block.
static MachineOperand &getLoopPhiOp(MachineInstr *Phi,
                                    MachineBasicBlock *LoopBB) {
  for (unsigned i = 1, e = Phi->getNumOperands(); i != e; i += 2)
    if (Phi->getOperand(i + 1).getMBB() == LoopBB)
      return Phi->getOperand(i);
  llvm_unreachable("Unexpected Phi structure.");
}

/// Return the Phi register value that comes from the loop block.
static unsigned getLoopPhiReg(MachineInstr *Phi, MachineBasicBlock *LoopBB) {
  for (unsigned i = 1, e = Phi->getNumOperands(); i != e; i += 2)
    if (Phi->getOperand(i + 1).getMBB() == LoopBB)
      return Phi->getOperand(i).getReg();
  llvm_unreachable("Unexpected Phi structure.");
}

/// Return the basic block corresponding to the Phi register value.
static MachineBasicBlock *getPhiRegBB(MachineInstr *Phi, unsigned Reg) {
  for (unsigned i = 1, e = Phi->getNumOperands(); i != e; i += 2)
    if (Phi->getOperand(i).getReg() == Reg)
      return Phi->getOperand(i + 1).getMBB();
  return 0;
}

/// Replace all uses of FromReg that appear within the specified
/// basic block with ToReg.
static void replaceRegUses(unsigned FromReg, unsigned ToReg,
                           MachineBasicBlock *MBB, MachineRegisterInfo &MRI) {
  for (MachineRegisterInfo::use_iterator I = MRI.use_begin(FromReg),
                                         E = MRI.use_end();
       I != E;) {
    MachineOperand &O = *I;
    ++I;
    MachineInstr *UseMI = O.getParent();
    if (UseMI->isPHI() && getPhiRegBB(UseMI, FromReg) != MBB)
      continue; // Don't change the register name

    if (UseMI->getParent() == MBB)
      O.setReg(ToReg);
  }
}

/// Clone the Phi instruction and set all the operands appropriately.
/// This function assumes the instruction is a Phi.
static MachineInstr *clonePHI(MachineBasicBlock *BB, MachineBasicBlock *BB1,
                              MachineBasicBlock *OrigBB, MachineInstr *Phi) {
  MachineFunction *MF = OrigBB->getParent();
  unsigned InitVal = getInitPhiReg(Phi, OrigBB);
  unsigned LoopVal = getLoopPhiReg(Phi, OrigBB);
  MachineInstr *NewMI = MF->CloneMachineInstr(Phi);
  NewMI->getOperand(1).setReg(InitVal);
  NewMI->getOperand(2).setMBB(BB1);
  NewMI->getOperand(3).setReg(LoopVal);
  NewMI->getOperand(4).setMBB(BB);
  return NewMI;
}

static bool isBlockOutsideLoop(SmallVector<MachineBasicBlock *, 4> &LoopBBs,
                               MachineBasicBlock *MBB) {
  for (auto TBB : LoopBBs)
    if (TBB == MBB)
      return false;
  return true;
}

static void
replaceRegUsesAfterLoop(unsigned FromReg, unsigned ToReg,
                        MachineRegisterInfo &MRI,
                        SmallVector<MachineBasicBlock *, 4> &LoopBBs) {
  MachineInstr *DefMI = MRI.getVRegDef(ToReg);
  for (MachineRegisterInfo::use_iterator I = MRI.use_begin(FromReg),
                                         E = MRI.use_end();
       I != E;) {
    MachineOperand &O = *I;
    ++I;
    MachineBasicBlock *UseBB = O.getParent()->getParent();
    if (isBlockOutsideLoop(LoopBBs, UseBB) && DefMI != O.getParent())
      O.setReg(ToReg);
  }
}

/// Return the register name for the latest instance of 'reg' as found
/// in the VRMap. FYI, During unrolling, different instances of 'reg'
/// (one from each iteration) are given a new name which is tracked
/// using VRMap.
unsigned MachineUnroller::getLatestInstance(unsigned reg, MachineBasicBlock *BB,
                                            ValueMapTy &VRMap) {
  auto BBI = VRMap.find(BB);
  if (BBI == VRMap.end())
    return reg;

  auto &BBMap = BBI->second;
  unsigned LatestReg = reg;
  while (true) {
    auto It = BBMap.find(LatestReg);
    if (It == BBMap.end() || LatestReg == It->second)
      return LatestReg;
    LatestReg = It->second;
  }
}

/// Update the machine instruction with new virtual registers.  This
/// function is only used to update the instructions in the unrolled
/// loop header. It may change the defintions and/or uses.
void MachineUnroller::updateInstruction(MachineInstr *NewMI, bool FirstIter,
                                        ValueMapTy &OldVRMap) {
  MachineBasicBlock *BB = NewMI->getParent();
  DenseMap<unsigned, unsigned> NewVRMap;
  DenseMap<unsigned, unsigned> &BBVRMap = VRMap[BB];
  for (unsigned i = 0, e = NewMI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = NewMI->getOperand(i);
    if (!MO.isReg() || !Register::isVirtualRegister(MO.getReg()))
      continue;
    unsigned reg = MO.getReg();
    if (MO.isDef()) {
      // Create a new virtual register for the definition.
      const TargetRegisterClass *RC = MRI->getRegClass(reg);
      unsigned NewReg = MRI->createVirtualRegister(RC);
      MO.setReg(NewReg);
      NewVRMap[reg] = NewReg;
      if (NewMI->isPHI())
        ULPhiVRMap[reg] = NewReg;
    } else if (MO.isUse()) {
      MachineInstr *DefMI = MRI->getVRegDef(reg);
      if (DefMI && DefMI->isPHI()) {
        if (NewMI->isPHI() && FirstIter)
          // Don't change the 'use' yet based on the new def reg. It will be
          // changed later to use the the last instance of the value reaching
          // from the loop after it has been unrolled.
          continue;
        else if (!FirstIter) {
          // Get mapped reg:
          // 1) If 'use' is a PHI, use the mapped reg from the previous
          //    iteration.
          // 2) If 'use' is a non-PHI, use the mapped reg from the current
          //    iteration.
          unsigned LatestReg = NewMI->isPHI()
                                   ? getLatestInstance(reg, BB, OldVRMap)
                                   : getLatestInstance(reg, BB, VRMap);
          MO.setReg(LatestReg);
          continue;
        }
      }
      if (BBVRMap.count(reg)) {
        unsigned MappedReg = BBVRMap[reg];
        if (MRI->getVRegDef(MappedReg) != NewMI)
          MO.setReg(MappedReg);
      }
    }
  }

  for (auto Val : NewVRMap)
    VRMap[BB][Val.first] = Val.second;
}

/// Return true if we can compute the amount the instruction changes
/// during each iteration. Set Delta to the amount of the change.
bool MachineUnroller::computeDelta(MachineInstr &MI, unsigned &Delta) const {
  const TargetRegisterInfo *TRI = MF->getSubtarget().getRegisterInfo();
  const MachineOperand *BaseOp;
  int64_t Offset;
  bool OffsetIsScalable;
  if (!TII->getMemOperandWithOffset(MI, BaseOp, Offset, OffsetIsScalable, TRI))
    return false;

  if (OffsetIsScalable)
    return false;

  if (!BaseOp->isReg())
    return false;

  // Check if there is a Phi. If so, get the definition in the loop.
  unsigned BaseReg = BaseOp->getReg();
  MachineInstr *BaseDef = MRI->getVRegDef(BaseReg);
  if (BaseDef && BaseDef->isPHI()) {
    if (BaseDef->getParent() != MI.getParent())
      return false;
    BaseReg = getLoopPhiReg(BaseDef, MI.getParent());
    BaseDef = MRI->getVRegDef(BaseReg);
  }
  if (!BaseDef)
    return false;

  int D = 0;
  if (!TII->getIncrementValue(*BaseDef, D))
    return false;
  // Conservative reaction to negative offsets
  if (D < 0)
    return false;
  Delta = D;
  return true;
}

/// Update the memory operand with a new offset when the unroller
/// generates a new copy of the instruction that refers to a
/// different memory location.
void MachineUnroller::updateMemOperands(MachineInstr *NewMI,
                                        MachineInstr *OldMI,
                                        unsigned iter) const {
  if (iter == 0)
    return;
  // If the instruction has memory operands, then adjust the offset
  // when the instruction appears in different iterations.
  unsigned NumRefs = NewMI->memoperands_end() - NewMI->memoperands_begin();
  if (NumRefs == 0)
    return;
  SmallVector<MachineMemOperand *, 2> NewMMOs;
  for (MachineMemOperand *MMO : NewMI->memoperands()) {
    if (MMO->isVolatile() || (MMO->isInvariant() && MMO->isDereferenceable()) ||
        (!MMO->getValue())) {
      NewMMOs.push_back(MMO);
      continue;
    }
    unsigned Delta;
    LLT valTy = MMO->getType();
    if (computeDelta(*OldMI, Delta)) {
      int64_t AdjOffset = Delta * iter;
      NewMMOs.push_back(MF->getMachineMemOperand(MMO, AdjOffset, valTy));
    } else
      NewMMOs.push_back(MF->getMachineMemOperand(MMO, 0, LLT()));
  }
  NewMI->setMemRefs(*MF, NewMMOs);
}

/// Adjust offset value for the instructions with memory operands when their
/// copies are generated after first iteration. By adjusting the offset and
/// using the right base register, we can avoid uncessary 'add' instructions
/// that are used to increment the offset for each iteration.

/// Generate instructions for the unrolled loop header.
void MachineUnroller::generateUnrolledLoop() {
  for (unsigned iter = 0; iter < UnrollFactor; iter++) {
    ValueMapTy OldVRMap = VRMap;
    for (MachineBasicBlock::iterator I = OrigHeader->instr_begin(),
                                     E = OrigHeader->getFirstTerminator();
         I != E; ++I) {
      MachineInstr *MI = &*I;
      bool FirstIter = (iter == 0);
      if (MI->isPHI() && !FirstIter) {
        // Just create a new dummy register name for the PHI def and map
        // it to LoopVal reaching from the previous iteration.
        unsigned OrigReg = MI->getOperand(0).getReg();
        const TargetRegisterClass *RC = MRI->getRegClass(OrigReg);
        unsigned NewReg = MRI->createVirtualRegister(RC);
        VRMap[ULHeader][OrigReg] = NewReg;
        unsigned LoopVal = getLoopPhiReg(MI, OrigHeader);
        if (RC == MRI->getRegClass(LoopVal)) {
          VRMap[ULHeader][NewReg] =
              getLatestInstance(LoopVal, ULHeader, OldVRMap);
          continue;
        } else {
          unsigned LatestReg = getLatestInstance(LoopVal, ULHeader, OldVRMap);
          MachineBasicBlock *BB = MI->getParent();
          MachineBasicBlock::iterator At = BB->getFirstTerminator();
          const DebugLoc &DL = BB->findDebugLoc(At);
          MachineInstr *NMI =
              BuildMI(*BB, At, DL, TII->get(TargetOpcode::COPY), NewReg)
                  .addReg(LatestReg);
          NMI->removeFromParent();
          ULHeader->push_back(NMI);
          VRMap[ULHeader][OrigReg] = NewReg;
          continue;
        }
      }
      MachineInstr *NewMI =
          MI->isPHI() ? clonePHI(ULHeader, ULPreheader, OrigHeader, MI)
                      : MF->CloneMachineInstr(MI);
      ULHeader->push_back(NewMI);
      updateInstruction(NewMI, iter == 0, OldVRMap);
      updateMemOperands(NewMI, MI, iter);
    }
  }

  // Copy any terminator instructions to the unrolled loop header.
  for (MachineBasicBlock::iterator I = OrigHeader->getFirstTerminator(),
                                   E = OrigHeader->instr_end();
       I != E; ++I) {
    MachineInstr *NewMI = MF->CloneMachineInstr(&*I);
    ULHeader->push_back(NewMI);
    updateInstruction(NewMI, false, VRMap);
  }

  // Update PHIs
  for (MachineBasicBlock::iterator I = ULHeader->instr_begin(),
                                   E = ULHeader->getFirstNonPHI();
       I != E; ++I) {
    MachineInstr *Phi = &*I;
    MachineOperand &MO = getLoopPhiOp(Phi, ULHeader);
    unsigned reg = MO.getReg();
    MO.setReg(getLatestInstance(reg, ULHeader, VRMap));
  }
}

/// Regenerate post-increment load/store instructions. Also, update the offset
/// value for the load/store instructions that use the same base address as the
/// newly created post-increment load/store.

/// Generate Phis for the exit block for the unrolled loop.
void MachineUnroller::generatePhisForULExit() {
  ValueMapTy OldVRMap = VRMap;
  for (MachineBasicBlock::iterator I = OrigHeader->instr_begin(),
                                   E = OrigHeader->getFirstNonPHI();
       I != E; ++I) {
    MachineInstr *Phi = &*I;
    assert(Phi->isPHI() && "Expecting a Phi.");
    unsigned DefReg = Phi->getOperand(0).getReg();
    const TargetRegisterClass *RC = MRI->getRegClass(DefReg);
    unsigned InitVal = getInitPhiReg(Phi, OrigHeader);
    unsigned LoopVal = getLoopPhiReg(Phi, OrigHeader);

    assert(InitVal != 0 && LoopVal != 0 && "Unexpected Phi structure.");
    MachineInstr *LoopInst = MRI->getVRegDef(LoopVal);
    unsigned PhiOp1 = InitVal;
    unsigned PhiOp2 = LoopInst->isPHI()
                          ? getLatestInstance(LoopVal, ULHeader, OldVRMap)
                          : getLatestInstance(LoopVal, ULHeader, VRMap);

    unsigned NewReg = MRI->createVirtualRegister(RC);
    MachineInstrBuilder NewPhi =
        BuildMI(*ULExit, ULExit->getFirstNonPHI(), DebugLoc(),
                TII->get(TargetOpcode::PHI), NewReg);
    NewPhi.addReg(PhiOp1).addMBB(OrigPreheader);
    NewPhi.addReg(PhiOp2).addMBB(ULHeader);
    VRMap[ULExit][DefReg] = NewReg;
    replaceRegUses(DefReg, NewReg, ULExit, *MRI);

    // Update Phi in the original loop header to use 'NewReg'
    // as the initial value.
    getInitPhiOp(Phi, OrigHeader).setReg(NewReg);
  }

  // Generate additional PHIs for the values that are live-in for
  // the original loop exit block.
  generateNewPhis(ULExit, OrigPreheader, ULHeader);
}

unsigned MachineUnroller::getMappedRegORCreate(unsigned Reg,
                                               MachineBasicBlock *BB) {
  const TargetRegisterClass *RC = MRI->getRegClass(Reg);
  if (VRMap[BB].count(Reg))
    return getLatestInstance(Reg, BB, VRMap);

  unsigned NewReg = MRI->createVirtualRegister(RC);
  BuildMI(*BB, BB->getFirstNonPHI(), DebugLoc(),
          TII->get(TargetOpcode::IMPLICIT_DEF), NewReg);
  return NewReg;
}

void MachineUnroller::generateNewPhis(MachineBasicBlock *BB,
                                      MachineBasicBlock *BB1,
                                      MachineBasicBlock *BB2) {
  for (auto Reg : ExitBBLiveIns) {
    unsigned BB1Reg = getMappedRegORCreate(Reg, BB1);
    unsigned BB2Reg = getMappedRegORCreate(Reg, BB2);
    const TargetRegisterClass *RC = MRI->getRegClass(Reg);
    unsigned NewReg = MRI->createVirtualRegister(RC);
    MachineInstrBuilder NewPhi = BuildMI(*BB, BB->getFirstNonPHI(), DebugLoc(),
                                         TII->get(TargetOpcode::PHI), NewReg);
    NewPhi.addReg(BB1Reg).addMBB(BB1);
    NewPhi.addReg(BB2Reg).addMBB(BB2);
    VRMap[BB][Reg] = NewReg;
  }
}

/// Generate Phis for the exit block for the remainder loop.
void MachineUnroller::generatePhisForRLExit() {
  // Generate PHIs for the values that are live-in for
  // the original loop exit block.
  generateNewPhis(RLExit, ULExit, RLHeader);

  for (MachineBasicBlock::iterator I = RLExit->instr_begin(),
                                   E = RLExit->getFirstNonPHI();
       I != E; ++I) {
    MachineInstr *Phi = &*I;
    unsigned OrigBBReg = 0;
    for (unsigned i = 1, e = Phi->getNumOperands(); i != e; i += 2) {
      if (Phi->getOperand(i + 1).getMBB() == OrigHeader)
        OrigBBReg = Phi->getOperand(i).getReg();
    }
    assert(OrigBBReg != 0 && "Unexpected Phi structure.");
    unsigned PhiDefReg = Phi->getOperand(0).getReg();
    replaceRegUsesAfterLoop(OrigBBReg, PhiDefReg, *MRI, LoopBBs);
  }
}

void MachineUnroller::getExitBBLiveIns() {
  for (auto I = OrigHeader->instr_begin(), E = OrigHeader->instr_end(); I != E;
       ++I) {
    MachineInstr *MI = &*I;
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg() || !MO.isDef() ||
          !Register::isVirtualRegister(MO.getReg()))
        continue;
      unsigned DefReg = MO.getReg();
      for (MachineRegisterInfo::use_iterator I = MRI->use_begin(DefReg),
                                             E = MRI->use_end();
           I != E;) {
        MachineOperand &O = *I;
        ++I;
        if (O.getParent()->getParent() != OrigHeader) {
          ExitBBLiveIns.push_back(DefReg);
          break;
        }
      }
    }
  }
}

void MachineUnroller::addBBIntoVRMap(MachineBasicBlock *BB) {
  for (auto I = BB->instr_begin(), E = BB->instr_end(); I != E; ++I) {
    MachineInstr *MI = &*I;
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg() || !Register::isVirtualRegister(MO.getReg()))
        continue;
      if (MO.isDef()) {
        unsigned DefReg = MO.getReg();
        VRMap[BB][DefReg] = DefReg;
      }
    }
  }
}

/// Remove all Phi instructions from BB.
static void cleanUpPHIs(MachineBasicBlock *BB, MachineRegisterInfo &MRI) {
  for (MachineBasicBlock::iterator MII = BB->instr_begin(),
                                   MIE = BB->getFirstNonPHI();
       MII != MIE;) {
    MachineInstr *Phi = &*MII;
    ++MII;
    unsigned InitVal = getInitPhiReg(Phi, BB);
    unsigned PhiDef = Phi->getOperand(0).getReg();
    for (MachineRegisterInfo::use_iterator I = MRI.use_begin(PhiDef),
                                           E = MRI.use_end();
         I != E;) {
      MachineOperand &O = *I;
      ++I;
      O.setReg(InitVal);
    }
    Phi->eraseFromParent();
  }
}

/// Fix all the branches for the unrolled and remainder loops. Also, update
/// the loop count.
void MachineUnroller::fixBranchesAndLoopCount(unsigned ULCount,
                                              unsigned RLCount) {
  SmallVector<MachineOperand, 4> Cond;
  MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
  bool checkBranch = TII->analyzeBranch(*ULHeader, TBB, FBB, Cond);
  assert(!checkBranch && "Can't analyze the branch in UnrolledLoop Header");
  (void)checkBranch;

  TII->removeBranch(*ULHeader);
  TII->insertBranch(*ULHeader, ULHeader, ULExit, Cond, DebugLoc());

  // Change loop count for the Unrolled loop and fixup branches.
  SmallVector<MachineOperand, 4> Cond1;
  changeLoopCount(*OrigPreheader, *ULPreheader, *ULHeader, *L->getTopBlock(),
                  ULCount, Cond1);
  TII->insertBranch(*OrigPreheader, ULExit, ULPreheader, Cond1, DebugLoc());
  Cond1.clear();
  TII->insertBranch(*ULPreheader, ULHeader, nullptr, Cond1, DebugLoc());

  // Copy instructions from the unrolled loop preheader as it may contain
  // loop setup instructions also needed for the Remainder loop.
  for (MachineBasicBlock::iterator I = ULPreheader->instr_begin(),
                                   E = ULPreheader->getFirstTerminator();
       I != E; ++I) {
    MachineInstr *MI = &*I;
    MachineInstr *NewMI = MF->CloneMachineInstr(MI);
    ULExit->push_back(NewMI);
  }

  // Change loop count for the Remainder loop and fixup branches.
  TII->removeBranch(*RLHeader);
  TII->insertBranch(*RLHeader, RLHeader, RLExit, Cond, DebugLoc());

  Cond1.clear();
  changeLoopCount(*ULExit, *RLPreheader, *RLHeader, *L->getTopBlock(), RLCount,
                  Cond1);
  TII->insertBranch(*ULExit, RLExit, RLPreheader, Cond1, DebugLoc());

  Cond1.clear();
  TII->insertBranch(*RLPreheader, RLHeader, nullptr, Cond1, DebugLoc());
  TII->insertBranch(*RLExit, OrigLoopExit, nullptr, Cond1, DebugLoc());
  if (RLHeader->succ_size() == 1)
    cleanUpPHIs(RLHeader, *MRI);
}

void MachineUnroller::preprocessPhiNodes(MachineBasicBlock &B) {
  for (MachineInstr &PI : make_range(B.begin(), B.getFirstNonPHI())) {
    MachineOperand &DefOp = PI.getOperand(0);
    assert(DefOp.getSubReg() == 0);
    auto *RC = MRI->getRegClass(DefOp.getReg());

    for (unsigned i = 1, n = PI.getNumOperands(); i != n; i += 2) {
      MachineOperand &RegOp = PI.getOperand(i);
      if (RegOp.getSubReg() == 0)
        continue;

      // If the operand uses a subregister, replace it with a new register
      // without subregisters, and generate a copy to the new register.
      unsigned NewReg = MRI->createVirtualRegister(RC);
      MachineBasicBlock &PredB = *PI.getOperand(i + 1).getMBB();
      MachineBasicBlock::iterator At = PredB.getFirstTerminator();
      const DebugLoc &DL = PredB.findDebugLoc(At);
      BuildMI(PredB, At, DL, TII->get(TargetOpcode::COPY), NewReg)
          .addReg(RegOp.getReg(), getRegState(RegOp), RegOp.getSubReg());
      RegOp.setReg(NewReg);
      RegOp.setSubReg(0);
    }
  }
}

bool MachineUnroller::unroll(MachineLoop *loop, unsigned unrollFactor) {
  init(loop, unrollFactor);
  if (!canUnroll())
    return false;

  // Remove any subregisters from input to phi nodes.
  preprocessPhiNodes(*loop->getHeader());

  // Add all the def regs in the loop header in VRMap.
  addBBIntoVRMap(OrigHeader);
  getExitBBLiveIns();

  // Create empty basic blocks for the unrolled version of the loop.
  createUnrolledLoopStruct();

  // Add instructions to compute trip counts for the unrolled and
  // remainder loops.
  TII->removeBranch(*OrigPreheader);
  unsigned ULCount = addUnrolledLoopCountMI(*OrigPreheader, LC, UnrollFactor);
  unsigned RLCount = addRemLoopCountMI(*OrigPreheader, LC, UnrollFactor);

  // Add instructions to the Unrolled loop header.
  generateUnrolledLoop();

  // Generate Phis for the unrolled loop exit block and also update
  // Phis in the remainder loop header to use the correct initial values.
  generatePhisForULExit();

  // Generate Phis for the remainder loop exit block.
  generatePhisForRLExit();

  // Optimize unrolled loop header.
  optimize(*ULHeader);

  // Update branches and adjust loop count.
  fixBranchesAndLoopCount(ULCount, RLCount);

  SmallVector<MachineBasicBlock *, 4> UpdateBBs = LoopBBs;
  UpdateBBs.insert(UpdateBBs.begin(), OrigPreheader);

  // Modify existing loop to point to the unrolled loop header.
  L->removeBlockFromLoop(OrigHeader);
  L->addBasicBlockToLoop(ULHeader, *MLI);
  return true;
}
