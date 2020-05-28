//===-- VEInstrInfo.cpp - VE Instruction Information ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the VE implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "VEInstrInfo.h"
#include "VE.h"
#include "VEMachineFunctionInfo.h"
#include "VESubtarget.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

#define DEBUG_TYPE "ve-instr-info"

using namespace llvm;

#define GET_INSTRINFO_CTOR_DTOR
#include "VEGenInstrInfo.inc"

// Pin the vtable to this file.
void VEInstrInfo::anchor() {}

VEInstrInfo::VEInstrInfo(VESubtarget &ST)
    : VEGenInstrInfo(VE::ADJCALLSTACKDOWN, VE::ADJCALLSTACKUP), RI() {}

static bool IsIntegerCC(unsigned CC) { return (CC < VECC::CC_AF); }

static VECC::CondCode GetOppositeBranchCondition(VECC::CondCode CC) {
  switch(CC) {
  case VECC::CC_IG:     return VECC::CC_ILE;
  case VECC::CC_IL:     return VECC::CC_IGE;
  case VECC::CC_INE:    return VECC::CC_IEQ;
  case VECC::CC_IEQ:    return VECC::CC_INE;
  case VECC::CC_IGE:    return VECC::CC_IL;
  case VECC::CC_ILE:    return VECC::CC_IG;
  case VECC::CC_AF:     return VECC::CC_AT;
  case VECC::CC_G:      return VECC::CC_LENAN;
  case VECC::CC_L:      return VECC::CC_GENAN;
  case VECC::CC_NE:     return VECC::CC_EQNAN;
  case VECC::CC_EQ:     return VECC::CC_NENAN;
  case VECC::CC_GE:     return VECC::CC_LNAN;
  case VECC::CC_LE:     return VECC::CC_GNAN;
  case VECC::CC_NUM:    return VECC::CC_NAN;
  case VECC::CC_NAN:    return VECC::CC_NUM;
  case VECC::CC_GNAN:   return VECC::CC_LE;
  case VECC::CC_LNAN:   return VECC::CC_GE;
  case VECC::CC_NENAN:  return VECC::CC_EQ;
  case VECC::CC_EQNAN:  return VECC::CC_NE;
  case VECC::CC_GENAN:  return VECC::CC_L;
  case VECC::CC_LENAN:  return VECC::CC_G;
  case VECC::CC_AT:     return VECC::CC_AF;
  }
  llvm_unreachable("Invalid cond code");
}

// Treat br.l [BRCF AT] as unconditional branch
static bool isUncondBranchOpcode(int Opc) {
  return Opc == VE::BRCFLa    || Opc == VE::BRCFWa    ||
         Opc == VE::BRCFLa_nt || Opc == VE::BRCFWa_nt ||
         Opc == VE::BRCFLa_t  || Opc == VE::BRCFWa_t  ||
         Opc == VE::BRCFDa    || Opc == VE::BRCFSa    ||
         Opc == VE::BRCFDa_nt || Opc == VE::BRCFSa_nt ||
         Opc == VE::BRCFDa_t  || Opc == VE::BRCFSa_t;
}

static bool isCondBranchOpcode(int Opc) {
  return Opc == VE::BRCFLrr    || Opc == VE::BRCFLir    ||
         Opc == VE::BRCFLrr_nt || Opc == VE::BRCFLir_nt ||
         Opc == VE::BRCFLrr_t  || Opc == VE::BRCFLir_t  ||
         Opc == VE::BRCFWrr    || Opc == VE::BRCFWir    ||
         Opc == VE::BRCFWrr_nt || Opc == VE::BRCFWir_nt ||
         Opc == VE::BRCFWrr_t  || Opc == VE::BRCFWir_t  ||
         Opc == VE::BRCFDrr    || Opc == VE::BRCFDir    ||
         Opc == VE::BRCFDrr_nt || Opc == VE::BRCFDir_nt ||
         Opc == VE::BRCFDrr_t  || Opc == VE::BRCFDir_t  ||
         Opc == VE::BRCFSrr    || Opc == VE::BRCFSir    ||
         Opc == VE::BRCFSrr_nt || Opc == VE::BRCFSir_nt ||
         Opc == VE::BRCFSrr_t  || Opc == VE::BRCFSir_t;
}

static bool isIndirectBranchOpcode(int Opc) {
  return Opc == VE::BCFLari    || Opc == VE::BCFLari    ||
         Opc == VE::BCFLari_nt || Opc == VE::BCFLari_nt ||
         Opc == VE::BCFLari_t  || Opc == VE::BCFLari_t  ||
         Opc == VE::BCFLari    || Opc == VE::BCFLari    ||
         Opc == VE::BCFLari_nt || Opc == VE::BCFLari_nt ||
         Opc == VE::BCFLari_t  || Opc == VE::BCFLari_t;
}

static void parseCondBranch(MachineInstr *LastInst, MachineBasicBlock *&Target,
                            SmallVectorImpl<MachineOperand> &Cond) {
  Cond.push_back(MachineOperand::CreateImm(LastInst->getOperand(0).getImm()));
  Cond.push_back(LastInst->getOperand(1));
  Cond.push_back(LastInst->getOperand(2));
  Target = LastInst->getOperand(3).getMBB();
}

bool VEInstrInfo::analyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                                MachineBasicBlock *&FBB,
                                SmallVectorImpl<MachineOperand> &Cond,
                                bool AllowModify) const {
  MachineBasicBlock::iterator I = MBB.getLastNonDebugInstr();
  if (I == MBB.end())
    return false;

  if (!isUnpredicatedTerminator(*I))
    return false;

  // Get the last instruction in the block.
  MachineInstr *LastInst = &*I;
  unsigned LastOpc = LastInst->getOpcode();

  // If there is only one terminator instruction, process it.
  if (I == MBB.begin() || !isUnpredicatedTerminator(*--I)) {
    if (isUncondBranchOpcode(LastOpc)) {
      TBB = LastInst->getOperand(0).getMBB();
      return false;
    }
    if (isCondBranchOpcode(LastOpc)) {
      // Block ends with fall-through condbranch.
      parseCondBranch(LastInst, TBB, Cond);
      return false;
    }
    return true; // Can't handle indirect branch.
  }

  // Get the instruction before it if it is a terminator.
  MachineInstr *SecondLastInst = &*I;
  unsigned SecondLastOpc = SecondLastInst->getOpcode();

  // If AllowModify is true and the block ends with two or more unconditional
  // branches, delete all but the first unconditional branch.
  if (AllowModify && isUncondBranchOpcode(LastOpc)) {
    while (isUncondBranchOpcode(SecondLastOpc)) {
      LastInst->eraseFromParent();
      LastInst = SecondLastInst;
      LastOpc = LastInst->getOpcode();
      if (I == MBB.begin() || !isUnpredicatedTerminator(*--I)) {
        // Return now the only terminator is an unconditional branch.
        TBB = LastInst->getOperand(0).getMBB();
        return false;
      }
      SecondLastInst = &*I;
      SecondLastOpc = SecondLastInst->getOpcode();
    }
  }

  // If there are three terminators, we don't know what sort of block this is.
  if (SecondLastInst && I != MBB.begin() && isUnpredicatedTerminator(*--I))
    return true;

  // If the block ends with a B and a Bcc, handle it.
  if (isCondBranchOpcode(SecondLastOpc) && isUncondBranchOpcode(LastOpc)) {
    parseCondBranch(SecondLastInst, TBB, Cond);
    FBB = LastInst->getOperand(0).getMBB();
    return false;
  }

  // If the block ends with two unconditional branches, handle it.  The second
  // one is not executed.
  if (isUncondBranchOpcode(SecondLastOpc) && isUncondBranchOpcode(LastOpc)) {
    TBB = SecondLastInst->getOperand(0).getMBB();
    return false;
  }

  // ...likewise if it ends with an indirect branch followed by an unconditional
  // branch.
  if (isIndirectBranchOpcode(SecondLastOpc) && isUncondBranchOpcode(LastOpc)) {
    I = LastInst;
    if (AllowModify)
      I->eraseFromParent();
    return true;
  }

  // Otherwise, can't handle this.
  return true;
}

unsigned VEInstrInfo::insertBranch(MachineBasicBlock &MBB,
                                   MachineBasicBlock *TBB,
                                   MachineBasicBlock *FBB,
                                   ArrayRef<MachineOperand> Cond,
                                   const DebugLoc &DL, int *BytesAdded) const {
  assert(TBB && "insertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 3 || Cond.size() == 0) &&
         "VE branch conditions should have three component!");
  assert(!BytesAdded && "code size not handled");
  if (Cond.empty()) {
    // Uncondition branch
    assert(!FBB && "Unconditional branch with multiple successors!");
    BuildMI(&MBB, DL, get(VE::BRCFLa_t))
        .addMBB(TBB);
    return 1;
  }

  // Conditional branch
  //   (BRCFir CC sy sz addr)
  assert(Cond[0].isImm() && Cond[2].isReg() && "not implemented");

  unsigned opc[2];
  const TargetRegisterInfo *TRI = &getRegisterInfo();
  MachineFunction *MF = MBB.getParent();
  const MachineRegisterInfo &MRI = MF->getRegInfo();
  unsigned Reg = Cond[2].getReg();
  if (IsIntegerCC(Cond[0].getImm())) {
    if (TRI->getRegSizeInBits(Reg, MRI) == 32) {
      opc[0] = VE::BRCFWir;
      opc[1] = VE::BRCFWrr;
    } else {
      opc[0] = VE::BRCFLir;
      opc[1] = VE::BRCFLrr;
    }
  } else {
    if (TRI->getRegSizeInBits(Reg, MRI) == 32) {
      opc[0] = VE::BRCFSir;
      opc[1] = VE::BRCFSrr;
    } else {
      opc[0] = VE::BRCFDir;
      opc[1] = VE::BRCFDrr;
    }
  }
  if (Cond[1].isImm()) {
      BuildMI(&MBB, DL, get(opc[0]))
          .add(Cond[0]) // condition code
          .add(Cond[1]) // lhs
          .add(Cond[2]) // rhs
          .addMBB(TBB);
  } else {
      BuildMI(&MBB, DL, get(opc[1]))
          .add(Cond[0])
          .add(Cond[1])
          .add(Cond[2])
          .addMBB(TBB);
  }

  if (!FBB)
    return 1;

  BuildMI(&MBB, DL, get(VE::BRCFLa_t))
      .addMBB(FBB);
  return 2;
}

unsigned VEInstrInfo::removeBranch(MachineBasicBlock &MBB,
                                   int *BytesRemoved) const {
  assert(!BytesRemoved && "code size not handled");

  MachineBasicBlock::iterator I = MBB.end();
  unsigned Count = 0;
  while (I != MBB.begin()) {
    --I;

    if (I->isDebugValue())
      continue;

    if (!isUncondBranchOpcode(I->getOpcode()) &&
        !isCondBranchOpcode(I->getOpcode()))
      break; // Not a branch

    I->eraseFromParent();
    I = MBB.end();
    ++Count;
  }
  return Count;
}

bool VEInstrInfo::reverseBranchCondition(
    SmallVectorImpl<MachineOperand> &Cond) const {
  VECC::CondCode CC = static_cast<VECC::CondCode>(Cond[0].getImm());
  Cond[0].setImm(GetOppositeBranchCondition(CC));
  return false;
}

static bool IsAliasOfSX(Register Reg) {
  return VE::I8RegClass.contains(Reg) || VE::I16RegClass.contains(Reg) ||
         VE::I32RegClass.contains(Reg) || VE::I64RegClass.contains(Reg) ||
         VE::F32RegClass.contains(Reg);
}

void VEInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I, const DebugLoc &DL,
                              MCRegister DestReg, MCRegister SrcReg,
                              bool KillSrc) const {

  if (IsAliasOfSX(SrcReg) && IsAliasOfSX(DestReg)) {
    BuildMI(MBB, I, DL, get(VE::ORri), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc))
        .addImm(0);
  } else {
    const TargetRegisterInfo *TRI = &getRegisterInfo();
    dbgs() << "Impossible reg-to-reg copy from " << printReg(SrcReg, TRI)
           << " to " << printReg(DestReg, TRI) << "\n";
    llvm_unreachable("Impossible reg-to-reg copy");
  }
}

/// isLoadFromStackSlot - If the specified machine instruction is a direct
/// load from a stack slot, return the virtual or physical register number of
/// the destination along with the FrameIndex of the loaded stack slot.  If
/// not, return 0.  This predicate must return 0 if the instruction has
/// any side effects other than loading from the stack slot.
unsigned VEInstrInfo::isLoadFromStackSlot(const MachineInstr &MI,
                                          int &FrameIndex) const {
  if (MI.getOpcode() == VE::LDrii ||    // I64
      MI.getOpcode() == VE::LDLSXrii || // I32
      MI.getOpcode() == VE::LDUrii      // F32
  ) {
    if (MI.getOperand(1).isFI() && MI.getOperand(2).isImm() &&
        MI.getOperand(2).getImm() == 0 && MI.getOperand(3).isImm() &&
        MI.getOperand(3).getImm() == 0) {
      FrameIndex = MI.getOperand(1).getIndex();
      return MI.getOperand(0).getReg();
    }
  }
  return 0;
}

/// isStoreToStackSlot - If the specified machine instruction is a direct
/// store to a stack slot, return the virtual or physical register number of
/// the source reg along with the FrameIndex of the loaded stack slot.  If
/// not, return 0.  This predicate must return 0 if the instruction has
/// any side effects other than storing to the stack slot.
unsigned VEInstrInfo::isStoreToStackSlot(const MachineInstr &MI,
                                         int &FrameIndex) const {
  if (MI.getOpcode() == VE::STrii ||  // I64
      MI.getOpcode() == VE::STLrii || // I32
      MI.getOpcode() == VE::STUrii    // F32
  ) {
    if (MI.getOperand(0).isFI() && MI.getOperand(1).isImm() &&
        MI.getOperand(1).getImm() == 0 && MI.getOperand(2).isImm() &&
        MI.getOperand(2).getImm() == 0) {
      FrameIndex = MI.getOperand(0).getIndex();
      return MI.getOperand(3).getReg();
    }
  }
  return 0;
}

void VEInstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator I,
                                      Register SrcReg, bool isKill, int FI,
                                      const TargetRegisterClass *RC,
                                      const TargetRegisterInfo *TRI) const {
  DebugLoc DL;
  if (I != MBB.end())
    DL = I->getDebugLoc();

  MachineFunction *MF = MBB.getParent();
  const MachineFrameInfo &MFI = MF->getFrameInfo();
  MachineMemOperand *MMO = MF->getMachineMemOperand(
      MachinePointerInfo::getFixedStack(*MF, FI), MachineMemOperand::MOStore,
      MFI.getObjectSize(FI), MFI.getObjectAlign(FI));

  // On the order of operands here: think "[FrameIdx + 0] = SrcReg".
  if (RC == &VE::I64RegClass) {
    BuildMI(MBB, I, DL, get(VE::STrii))
        .addFrameIndex(FI)
        .addImm(0)
        .addImm(0)
        .addReg(SrcReg, getKillRegState(isKill))
        .addMemOperand(MMO);
  } else if (RC == &VE::I32RegClass) {
    BuildMI(MBB, I, DL, get(VE::STLrii))
        .addFrameIndex(FI)
        .addImm(0)
        .addImm(0)
        .addReg(SrcReg, getKillRegState(isKill))
        .addMemOperand(MMO);
  } else if (RC == &VE::F32RegClass) {
    BuildMI(MBB, I, DL, get(VE::STUrii))
        .addFrameIndex(FI)
        .addImm(0)
        .addImm(0)
        .addReg(SrcReg, getKillRegState(isKill))
        .addMemOperand(MMO);
  } else
    report_fatal_error("Can't store this register to stack slot");
}

void VEInstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator I,
                                       Register DestReg, int FI,
                                       const TargetRegisterClass *RC,
                                       const TargetRegisterInfo *TRI) const {
  DebugLoc DL;
  if (I != MBB.end())
    DL = I->getDebugLoc();

  MachineFunction *MF = MBB.getParent();
  const MachineFrameInfo &MFI = MF->getFrameInfo();
  MachineMemOperand *MMO = MF->getMachineMemOperand(
      MachinePointerInfo::getFixedStack(*MF, FI), MachineMemOperand::MOLoad,
      MFI.getObjectSize(FI), MFI.getObjectAlign(FI));

  if (RC == &VE::I64RegClass) {
    BuildMI(MBB, I, DL, get(VE::LDrii), DestReg)
        .addFrameIndex(FI)
        .addImm(0)
        .addImm(0)
        .addMemOperand(MMO);
  } else if (RC == &VE::I32RegClass) {
    BuildMI(MBB, I, DL, get(VE::LDLSXrii), DestReg)
        .addFrameIndex(FI)
        .addImm(0)
        .addImm(0)
        .addMemOperand(MMO);
  } else if (RC == &VE::F32RegClass) {
    BuildMI(MBB, I, DL, get(VE::LDUrii), DestReg)
        .addFrameIndex(FI)
        .addImm(0)
        .addImm(0)
        .addMemOperand(MMO);
  } else
    report_fatal_error("Can't load this register from stack slot");
}

Register VEInstrInfo::getGlobalBaseReg(MachineFunction *MF) const {
  VEMachineFunctionInfo *VEFI = MF->getInfo<VEMachineFunctionInfo>();
  Register GlobalBaseReg = VEFI->getGlobalBaseReg();
  if (GlobalBaseReg != 0)
    return GlobalBaseReg;

  // We use %s15 (%got) as a global base register
  GlobalBaseReg = VE::SX15;

  // Insert a pseudo instruction to set the GlobalBaseReg into the first
  // MBB of the function
  MachineBasicBlock &FirstMBB = MF->front();
  MachineBasicBlock::iterator MBBI = FirstMBB.begin();
  DebugLoc dl;
  BuildMI(FirstMBB, MBBI, dl, get(VE::GETGOT), GlobalBaseReg);
  VEFI->setGlobalBaseReg(GlobalBaseReg);
  return GlobalBaseReg;
}

bool VEInstrInfo::expandPostRAPseudo(MachineInstr &MI) const {
  switch (MI.getOpcode()) {
  case VE::EXTEND_STACK: {
    return expandExtendStackPseudo(MI);
  }
  case VE::EXTEND_STACK_GUARD: {
    MI.eraseFromParent(); // The pseudo instruction is gone now.
    return true;
  }
  case VE::GETSTACKTOP: {
    return expandGetStackTopPseudo(MI);
  }
  }
  return false;
}

bool VEInstrInfo::expandExtendStackPseudo(MachineInstr &MI) const {
  MachineBasicBlock &MBB = *MI.getParent();
  MachineFunction &MF = *MBB.getParent();
  const VESubtarget &STI = MF.getSubtarget<VESubtarget>();
  const VEInstrInfo &TII = *STI.getInstrInfo();
  DebugLoc dl = MBB.findDebugLoc(MI);

  // Create following instructions and multiple basic blocks.
  //
  // thisBB:
  //   brge.l.t %sp, %sl, sinkBB
  // syscallBB:
  //   ld      %s61, 0x18(, %tp)        // load param area
  //   or      %s62, 0, %s0             // spill the value of %s0
  //   lea     %s63, 0x13b              // syscall # of grow
  //   shm.l   %s63, 0x0(%s61)          // store syscall # at addr:0
  //   shm.l   %sl, 0x8(%s61)           // store old limit at addr:8
  //   shm.l   %sp, 0x10(%s61)          // store new limit at addr:16
  //   monc                             // call monitor
  //   or      %s0, 0, %s62             // restore the value of %s0
  // sinkBB:

  // Create new MBB
  MachineBasicBlock *BB = &MBB;
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineBasicBlock *syscallMBB = MF.CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *sinkMBB = MF.CreateMachineBasicBlock(LLVM_BB);
  MachineFunction::iterator It = ++(BB->getIterator());
  MF.insert(It, syscallMBB);
  MF.insert(It, sinkMBB);

  // Transfer the remainder of BB and its successor edges to sinkMBB.
  sinkMBB->splice(sinkMBB->begin(), BB,
                  std::next(std::next(MachineBasicBlock::iterator(MI))),
                  BB->end());
  sinkMBB->transferSuccessorsAndUpdatePHIs(BB);

  // Next, add the true and fallthrough blocks as its successors.
  BB->addSuccessor(syscallMBB);
  BB->addSuccessor(sinkMBB);
  BuildMI(BB, dl, TII.get(VE::BRCFLrr_t))
      .addImm(VECC::CC_IGE)
      .addReg(VE::SX11) // %sp
      .addReg(VE::SX8)  // %sl
      .addMBB(sinkMBB);

  BB = syscallMBB;

  // Update machine-CFG edges
  BB->addSuccessor(sinkMBB);

  BuildMI(BB, dl, TII.get(VE::LDrii), VE::SX61)
      .addReg(VE::SX14)
      .addImm(0)
      .addImm(0x18);
  BuildMI(BB, dl, TII.get(VE::ORri), VE::SX62)
      .addReg(VE::SX0)
      .addImm(0);
  BuildMI(BB, dl, TII.get(VE::LEAzii), VE::SX63)
      .addImm(0)
      .addImm(0)
      .addImm(0x13b);
  BuildMI(BB, dl, TII.get(VE::SHMri))
      .addReg(VE::SX61)
      .addImm(0)
      .addReg(VE::SX63);
  BuildMI(BB, dl, TII.get(VE::SHMri))
      .addReg(VE::SX61)
      .addImm(8)
      .addReg(VE::SX8);
  BuildMI(BB, dl, TII.get(VE::SHMri))
      .addReg(VE::SX61)
      .addImm(16)
      .addReg(VE::SX11);
  BuildMI(BB, dl, TII.get(VE::MONC));

  BuildMI(BB, dl, TII.get(VE::ORri), VE::SX0)
      .addReg(VE::SX62)
      .addImm(0);

  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return true;
}

bool VEInstrInfo::expandGetStackTopPseudo(MachineInstr &MI) const {
  MachineBasicBlock *MBB = MI.getParent();
  MachineFunction &MF = *MBB->getParent();
  const VESubtarget &STI = MF.getSubtarget<VESubtarget>();
  const VEInstrInfo &TII = *STI.getInstrInfo();
  DebugLoc DL = MBB->findDebugLoc(MI);

  // Create following instruction
  //
  //   dst = %sp + target specific frame + the size of parameter area

  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const VEFrameLowering &TFL = *STI.getFrameLowering();

  // The VE ABI requires a reserved 176 bytes area at the top
  // of stack as described in VESubtarget.cpp.  So, we adjust it here.
  unsigned NumBytes = STI.getAdjustedFrameSize(0);

  // Also adds the size of parameter area.
  if (MFI.adjustsStack() && TFL.hasReservedCallFrame(MF))
    NumBytes += MFI.getMaxCallFrameSize();

  BuildMI(*MBB, MI, DL, TII.get(VE::LEArii))
      .addDef(MI.getOperand(0).getReg())
      .addReg(VE::SX11)
      .addImm(0)
      .addImm(NumBytes);

  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return true;
}
