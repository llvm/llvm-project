//===-- SPUInstrInfo.cpp - Cell SPU Instruction Information ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Cell SPU implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "SPUInstrInfo.h"
#include "SPUInstrBuilder.h"
#include "SPUTargetMachine.h"
#include "SPUHazardRecognizers.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"

#define GET_INSTRINFO_CTOR
#include "SPUGenInstrInfo.inc"

using namespace llvm;

namespace {
  //! Predicate for an unconditional branch instruction
  inline bool isUncondBranch(const MachineInstr *I) {
    unsigned opc = I->getOpcode();

    return (opc == SPU::BR
            || opc == SPU::BRA
            || opc == SPU::BI);
  }

  //! Predicate for a conditional branch instruction
  inline bool isCondBranch(const MachineInstr *I) {
    unsigned opc = I->getOpcode();

    return (opc == SPU::BRNZr32
            || opc == SPU::BRNZv4i32
            || opc == SPU::BRZr32
            || opc == SPU::BRZv4i32
            || opc == SPU::BRHNZr16
            || opc == SPU::BRHNZv8i16
            || opc == SPU::BRHZr16
            || opc == SPU::BRHZv8i16);
  }
}

SPUInstrInfo::SPUInstrInfo(SPUTargetMachine &tm)
  : SPUGenInstrInfo(SPU::ADJCALLSTACKDOWN, SPU::ADJCALLSTACKUP),
    TM(tm),
    RI(*TM.getSubtargetImpl(), *this)
{ /* NOP */ }

/// CreateTargetHazardRecognizer - Return the hazard recognizer to use for
/// this target when scheduling the DAG.
ScheduleHazardRecognizer *SPUInstrInfo::CreateTargetHazardRecognizer(
  const TargetMachine *TM,
  const ScheduleDAG *DAG) const {
  const TargetInstrInfo *TII = TM->getInstrInfo();
  assert(TII && "No InstrInfo?");
  return new SPUHazardRecognizer(*TII);
}

unsigned
SPUInstrInfo::isLoadFromStackSlot(const MachineInstr *MI,
                                  int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case SPU::LQDv16i8:
  case SPU::LQDv8i16:
  case SPU::LQDv4i32:
  case SPU::LQDv4f32:
  case SPU::LQDv2f64:
  case SPU::LQDr128:
  case SPU::LQDr64:
  case SPU::LQDr32:
  case SPU::LQDr16: {
    const MachineOperand MOp1 = MI->getOperand(1);
    const MachineOperand MOp2 = MI->getOperand(2);
    if (MOp1.isImm() && MOp2.isFI()) {
      FrameIndex = MOp2.getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }
  }
  return 0;
}

unsigned
SPUInstrInfo::isStoreToStackSlot(const MachineInstr *MI,
                                 int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case SPU::STQDv16i8:
  case SPU::STQDv8i16:
  case SPU::STQDv4i32:
  case SPU::STQDv4f32:
  case SPU::STQDv2f64:
  case SPU::STQDr128:
  case SPU::STQDr64:
  case SPU::STQDr32:
  case SPU::STQDr16:
  case SPU::STQDr8: {
    const MachineOperand MOp1 = MI->getOperand(1);
    const MachineOperand MOp2 = MI->getOperand(2);
    if (MOp1.isImm() && MOp2.isFI()) {
      FrameIndex = MOp2.getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }
  }
  return 0;
}

void SPUInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator I, DebugLoc DL,
                               unsigned DestReg, unsigned SrcReg,
                               bool KillSrc) const
{
  // We support cross register class moves for our aliases, such as R3 in any
  // reg class to any other reg class containing R3.  This is required because
  // we instruction select bitconvert i64 -> f64 as a noop for example, so our
  // types have no specific meaning.

  BuildMI(MBB, I, DL, get(SPU::LRr128), DestReg)
    .addReg(SrcReg, getKillRegState(KillSrc));
}

void
SPUInstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator MI,
                                  unsigned SrcReg, bool isKill, int FrameIdx,
                                  const TargetRegisterClass *RC,
                                  const TargetRegisterInfo *TRI) const {
  unsigned opc;
  bool isValidFrameIdx = (FrameIdx < SPUFrameLowering::maxFrameOffset());
  if (RC == &SPU::GPRCRegClass)
    opc = isValidFrameIdx ? SPU::STQDr128 : SPU::STQXr128;
  else if (RC == &SPU::R64CRegClass)
    opc = isValidFrameIdx ? SPU::STQDr64 : SPU::STQXr64;
  else if (RC == &SPU::R64FPRegClass)
    opc = isValidFrameIdx ? SPU::STQDr64 : SPU::STQXr64;
  else if (RC == &SPU::R32CRegClass)
    opc = isValidFrameIdx ? SPU::STQDr32 : SPU::STQXr32;
  else if (RC == &SPU::R32FPRegClass)
    opc = isValidFrameIdx ? SPU::STQDr32 : SPU::STQXr32;
  else if (RC == &SPU::R16CRegClass)
    opc = isValidFrameIdx ? SPU::STQDr16 : SPU::STQXr16;
  else if (RC == &SPU::R8CRegClass)
    opc = isValidFrameIdx ? SPU::STQDr8 : SPU::STQXr8;
  else if (RC == &SPU::VECREGRegClass)
    opc = isValidFrameIdx ? SPU::STQDv16i8 : SPU::STQXv16i8;
  else
    llvm_unreachable("Unknown regclass!");

  DebugLoc DL;
  if (MI != MBB.end()) DL = MI->getDebugLoc();
  addFrameReference(BuildMI(MBB, MI, DL, get(opc))
                    .addReg(SrcReg, getKillRegState(isKill)), FrameIdx);
}

void
SPUInstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MI,
                                   unsigned DestReg, int FrameIdx,
                                   const TargetRegisterClass *RC,
                                   const TargetRegisterInfo *TRI) const {
  unsigned opc;
  bool isValidFrameIdx = (FrameIdx < SPUFrameLowering::maxFrameOffset());
  if (RC == &SPU::GPRCRegClass)
    opc = isValidFrameIdx ? SPU::LQDr128 : SPU::LQXr128;
  else if (RC == &SPU::R64CRegClass)
    opc = isValidFrameIdx ? SPU::LQDr64 : SPU::LQXr64;
  else if (RC == &SPU::R64FPRegClass)
    opc = isValidFrameIdx ? SPU::LQDr64 : SPU::LQXr64;
  else if (RC == &SPU::R32CRegClass)
    opc = isValidFrameIdx ? SPU::LQDr32 : SPU::LQXr32;
  else if (RC == &SPU::R32FPRegClass)
    opc = isValidFrameIdx ? SPU::LQDr32 : SPU::LQXr32;
  else if (RC == &SPU::R16CRegClass)
    opc = isValidFrameIdx ? SPU::LQDr16 : SPU::LQXr16;
  else if (RC == &SPU::R8CRegClass)
    opc = isValidFrameIdx ? SPU::LQDr8 : SPU::LQXr8;
  else if (RC == &SPU::VECREGRegClass)
    opc = isValidFrameIdx ? SPU::LQDv16i8 : SPU::LQXv16i8;
  else
    llvm_unreachable("Unknown regclass in loadRegFromStackSlot!");

  DebugLoc DL;
  if (MI != MBB.end()) DL = MI->getDebugLoc();
  addFrameReference(BuildMI(MBB, MI, DL, get(opc), DestReg), FrameIdx);
}

//! Branch analysis
/*!
  \note This code was kiped from PPC. There may be more branch analysis for
  CellSPU than what's currently done here.
 */
bool
SPUInstrInfo::AnalyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                            MachineBasicBlock *&FBB,
                            SmallVectorImpl<MachineOperand> &Cond,
                            bool AllowModify) const {
  // If the block has no terminators, it just falls into the block after it.
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin())
    return false;
  --I;
  while (I->isDebugValue()) {
    if (I == MBB.begin())
      return false;
    --I;
  }
  if (!isUnpredicatedTerminator(I))
    return false;

  // Get the last instruction in the block.
  MachineInstr *LastInst = I;

  // If there is only one terminator instruction, process it.
  if (I == MBB.begin() || !isUnpredicatedTerminator(--I)) {
    if (isUncondBranch(LastInst)) {
      // Check for jump tables
      if (!LastInst->getOperand(0).isMBB())
        return true;
      TBB = LastInst->getOperand(0).getMBB();
      return false;
    } else if (isCondBranch(LastInst)) {
      // Block ends with fall-through condbranch.
      TBB = LastInst->getOperand(1).getMBB();
      DEBUG(errs() << "Pushing LastInst:               ");
      DEBUG(LastInst->dump());
      Cond.push_back(MachineOperand::CreateImm(LastInst->getOpcode()));
      Cond.push_back(LastInst->getOperand(0));
      return false;
    }
    // Otherwise, don't know what this is.
    return true;
  }

  // Get the instruction before it if it's a terminator.
  MachineInstr *SecondLastInst = I;

  // If there are three terminators, we don't know what sort of block this is.
  if (SecondLastInst && I != MBB.begin() &&
      isUnpredicatedTerminator(--I))
    return true;

  // If the block ends with a conditional and unconditional branch, handle it.
  if (isCondBranch(SecondLastInst) && isUncondBranch(LastInst)) {
    TBB =  SecondLastInst->getOperand(1).getMBB();
    DEBUG(errs() << "Pushing SecondLastInst:         ");
    DEBUG(SecondLastInst->dump());
    Cond.push_back(MachineOperand::CreateImm(SecondLastInst->getOpcode()));
    Cond.push_back(SecondLastInst->getOperand(0));
    FBB = LastInst->getOperand(0).getMBB();
    return false;
  }

  // If the block ends with two unconditional branches, handle it.  The second
  // one is not executed, so remove it.
  if (isUncondBranch(SecondLastInst) && isUncondBranch(LastInst)) {
    TBB = SecondLastInst->getOperand(0).getMBB();
    I = LastInst;
    if (AllowModify)
      I->eraseFromParent();
    return false;
  }

  // Otherwise, can't handle this.
  return true;
}

// search MBB for branch hint labels and branch hit ops
static void removeHBR( MachineBasicBlock &MBB) {
  for (MachineBasicBlock::iterator I = MBB.begin(); I != MBB.end(); ++I){
    if (I->getOpcode() == SPU::HBRA ||
        I->getOpcode() == SPU::HBR_LABEL){
      I=MBB.erase(I);
      if (I == MBB.end())
        break;
    }
  }
}

unsigned
SPUInstrInfo::RemoveBranch(MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator I = MBB.end();
  removeHBR(MBB);
  if (I == MBB.begin())
    return 0;
  --I;
  while (I->isDebugValue()) {
    if (I == MBB.begin())
      return 0;
    --I;
  }
  if (!isCondBranch(I) && !isUncondBranch(I))
    return 0;

  // Remove the first branch.
  DEBUG(errs() << "Removing branch:                ");
  DEBUG(I->dump());
  I->eraseFromParent();
  I = MBB.end();
  if (I == MBB.begin())
    return 1;

  --I;
  if (!(isCondBranch(I) || isUncondBranch(I)))
    return 1;

  // Remove the second branch.
  DEBUG(errs() << "Removing second branch:         ");
  DEBUG(I->dump());
  I->eraseFromParent();
  return 2;
}

/** Find the optimal position for a hint branch instruction in a basic block.
 * This should take into account:
 *   -the branch hint delays
 *   -congestion of the memory bus
 *   -dual-issue scheduling (i.e. avoid insertion of nops)
 * Current implementation is rather simplistic.
 */
static MachineBasicBlock::iterator findHBRPosition(MachineBasicBlock &MBB)
{
   MachineBasicBlock::iterator J = MBB.end();
   for( int i=0; i<8; i++) {
     if( J == MBB.begin() ) return J;
     J--;
   }
   return J;
}

unsigned
SPUInstrInfo::InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                           MachineBasicBlock *FBB,
                           const SmallVectorImpl<MachineOperand> &Cond,
                           DebugLoc DL) const {
  // Shouldn't be a fall through.
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 2 || Cond.size() == 0) &&
         "SPU branch conditions have two components!");

  MachineInstrBuilder MIB;
  //TODO: make a more accurate algorithm.
  bool haveHBR = MBB.size()>8;

  removeHBR(MBB);
  MCSymbol *branchLabel = MBB.getParent()->getContext().CreateTempSymbol();
  // Add a label just before the branch
  if (haveHBR)
    MIB = BuildMI(&MBB, DL, get(SPU::HBR_LABEL)).addSym(branchLabel);

  // One-way branch.
  if (FBB == 0) {
    if (Cond.empty()) {
      // Unconditional branch
      MIB = BuildMI(&MBB, DL, get(SPU::BR));
      MIB.addMBB(TBB);

      DEBUG(errs() << "Inserted one-way uncond branch: ");
      DEBUG((*MIB).dump());

      // basic blocks have just one branch so it is safe to add the hint a its
      if (haveHBR) {
        MIB = BuildMI( MBB, findHBRPosition(MBB), DL, get(SPU::HBRA));
        MIB.addSym(branchLabel);
        MIB.addMBB(TBB);
      }
    } else {
      // Conditional branch
      MIB = BuildMI(&MBB, DL, get(Cond[0].getImm()));
      MIB.addReg(Cond[1].getReg()).addMBB(TBB);

      if (haveHBR) {
        MIB = BuildMI(MBB, findHBRPosition(MBB), DL, get(SPU::HBRA));
        MIB.addSym(branchLabel);
        MIB.addMBB(TBB);
      }

      DEBUG(errs() << "Inserted one-way cond branch:   ");
      DEBUG((*MIB).dump());
    }
    return 1;
  } else {
    MIB = BuildMI(&MBB, DL, get(Cond[0].getImm()));
    MachineInstrBuilder MIB2 = BuildMI(&MBB, DL, get(SPU::BR));

    // Two-way Conditional Branch.
    MIB.addReg(Cond[1].getReg()).addMBB(TBB);
    MIB2.addMBB(FBB);

    if (haveHBR) {
      MIB = BuildMI( MBB, findHBRPosition(MBB), DL, get(SPU::HBRA));
      MIB.addSym(branchLabel);
      MIB.addMBB(FBB);
    }

    DEBUG(errs() << "Inserted conditional branch:    ");
    DEBUG((*MIB).dump());
    DEBUG(errs() << "part 2: ");
    DEBUG((*MIB2).dump());
   return 2;
  }
}

//! Reverses a branch's condition, returning false on success.
bool
SPUInstrInfo::ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond)
  const {
  // Pretty brainless way of inverting the condition, but it works, considering
  // there are only two conditions...
  static struct {
    unsigned Opc;               //! The incoming opcode
    unsigned RevCondOpc;        //! The reversed condition opcode
  } revconds[] = {
    { SPU::BRNZr32, SPU::BRZr32 },
    { SPU::BRNZv4i32, SPU::BRZv4i32 },
    { SPU::BRZr32, SPU::BRNZr32 },
    { SPU::BRZv4i32, SPU::BRNZv4i32 },
    { SPU::BRHNZr16, SPU::BRHZr16 },
    { SPU::BRHNZv8i16, SPU::BRHZv8i16 },
    { SPU::BRHZr16, SPU::BRHNZr16 },
    { SPU::BRHZv8i16, SPU::BRHNZv8i16 }
  };

  unsigned Opc = unsigned(Cond[0].getImm());
  // Pretty dull mapping between the two conditions that SPU can generate:
  for (int i = sizeof(revconds)/sizeof(revconds[0]) - 1; i >= 0; --i) {
    if (revconds[i].Opc == Opc) {
      Cond[0].setImm(revconds[i].RevCondOpc);
      return false;
    }
  }

  return true;
}
