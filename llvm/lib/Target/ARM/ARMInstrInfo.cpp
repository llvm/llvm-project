//===- ARMInstrInfo.cpp - ARM Instruction Information -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the ARM implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "ARMInstrInfo.h"
#include "ARM.h"
#include "ARMAddressingModes.h"
#include "ARMGenInstrInfo.inc"
#include "ARMMachineFunctionInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

static cl::opt<bool> EnableARM3Addr("enable-arm-3-addr-conv", cl::Hidden,
                                  cl::desc("Enable ARM 2-addr to 3-addr conv"));

static inline
const MachineInstrBuilder &AddDefaultPred(const MachineInstrBuilder &MIB) {
  return MIB.addImm((int64_t)ARMCC::AL).addReg(0);
}

static inline
const MachineInstrBuilder &AddDefaultCC(const MachineInstrBuilder &MIB) {
  return MIB.addReg(0);
}

ARMInstrInfo::ARMInstrInfo(const ARMSubtarget &STI)
  : TargetInstrInfoImpl(ARMInsts, array_lengthof(ARMInsts)),
    RI(*this, STI) {
}

const TargetRegisterClass *ARMInstrInfo::getPointerRegClass() const {
  return &ARM::GPRRegClass;
}

/// Return true if the instruction is a register to register move and
/// leave the source and dest operands in the passed parameters.
///
bool ARMInstrInfo::isMoveInstr(const MachineInstr &MI,
                               unsigned &SrcReg, unsigned &DstReg,
                               unsigned& SrcSubIdx, unsigned& DstSubIdx) const {
  SrcSubIdx = DstSubIdx = 0; // No sub-registers.

  unsigned oc = MI.getOpcode();
  switch (oc) {
  default:
    return false;
  case ARM::FCPYS:
  case ARM::FCPYD:
    SrcReg = MI.getOperand(1).getReg();
    DstReg = MI.getOperand(0).getReg();
    return true;
  case ARM::MOVr:
  case ARM::tMOVr:
    assert(MI.getDesc().getNumOperands() >= 2 &&
           MI.getOperand(0).isReg() &&
           MI.getOperand(1).isReg() &&
           "Invalid ARM MOV instruction");
    SrcReg = MI.getOperand(1).getReg();
    DstReg = MI.getOperand(0).getReg();
    return true;
  }
}

unsigned ARMInstrInfo::isLoadFromStackSlot(const MachineInstr *MI,
                                           int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case ARM::LDR:
    if (MI->getOperand(1).isFI() &&
        MI->getOperand(2).isReg() &&
        MI->getOperand(3).isImm() &&
        MI->getOperand(2).getReg() == 0 &&
        MI->getOperand(3).getImm() == 0) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  case ARM::FLDD:
  case ARM::FLDS:
    if (MI->getOperand(1).isFI() &&
        MI->getOperand(2).isImm() &&
        MI->getOperand(2).getImm() == 0) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  case ARM::tRestore:
    if (MI->getOperand(1).isFI() &&
        MI->getOperand(2).isImm() &&
        MI->getOperand(2).getImm() == 0) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }
  return 0;
}

unsigned ARMInstrInfo::isStoreToStackSlot(const MachineInstr *MI,
                                          int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case ARM::STR:
    if (MI->getOperand(1).isFI() &&
        MI->getOperand(2).isReg() &&
        MI->getOperand(3).isImm() &&
        MI->getOperand(2).getReg() == 0 &&
        MI->getOperand(3).getImm() == 0) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  case ARM::FSTD:
  case ARM::FSTS:
    if (MI->getOperand(1).isFI() &&
        MI->getOperand(2).isImm() &&
        MI->getOperand(2).getImm() == 0) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  case ARM::tSpill:
    if (MI->getOperand(1).isFI() &&
        MI->getOperand(2).isImm() &&
        MI->getOperand(2).getImm() == 0) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }
  return 0;
}

void ARMInstrInfo::reMaterialize(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator I,
                                 unsigned DestReg,
                                 const MachineInstr *Orig) const {
  if (Orig->getOpcode() == ARM::MOVi2pieces) {
    RI.emitLoadConstPool(MBB, I, DestReg, Orig->getOperand(1).getImm(),
                         Orig->getOperand(2).getImm(),
                         Orig->getOperand(3).getReg(), this, false);
    return;
  }

  MachineInstr *MI = MBB.getParent()->CloneMachineInstr(Orig);
  MI->getOperand(0).setReg(DestReg);
  MBB.insert(I, MI);
}

static unsigned getUnindexedOpcode(unsigned Opc) {
  switch (Opc) {
  default: break;
  case ARM::LDR_PRE:
  case ARM::LDR_POST:
    return ARM::LDR;
  case ARM::LDRH_PRE:
  case ARM::LDRH_POST:
    return ARM::LDRH;
  case ARM::LDRB_PRE:
  case ARM::LDRB_POST:
    return ARM::LDRB;
  case ARM::LDRSH_PRE:
  case ARM::LDRSH_POST:
    return ARM::LDRSH;
  case ARM::LDRSB_PRE:
  case ARM::LDRSB_POST:
    return ARM::LDRSB;
  case ARM::STR_PRE:
  case ARM::STR_POST:
    return ARM::STR;
  case ARM::STRH_PRE:
  case ARM::STRH_POST:
    return ARM::STRH;
  case ARM::STRB_PRE:
  case ARM::STRB_POST:
    return ARM::STRB;
  }
  return 0;
}

MachineInstr *
ARMInstrInfo::convertToThreeAddress(MachineFunction::iterator &MFI,
                                    MachineBasicBlock::iterator &MBBI,
                                    LiveVariables *LV) const {
  if (!EnableARM3Addr)
    return NULL;

  MachineInstr *MI = MBBI;
  MachineFunction &MF = *MI->getParent()->getParent();
  unsigned TSFlags = MI->getDesc().TSFlags;
  bool isPre = false;
  switch ((TSFlags & ARMII::IndexModeMask) >> ARMII::IndexModeShift) {
  default: return NULL;
  case ARMII::IndexModePre:
    isPre = true;
    break;
  case ARMII::IndexModePost:
    break;
  }

  // Try spliting an indexed load / store to a un-indexed one plus an add/sub
  // operation.
  unsigned MemOpc = getUnindexedOpcode(MI->getOpcode());
  if (MemOpc == 0)
    return NULL;

  MachineInstr *UpdateMI = NULL;
  MachineInstr *MemMI = NULL;
  unsigned AddrMode = (TSFlags & ARMII::AddrModeMask);
  const TargetInstrDesc &TID = MI->getDesc();
  unsigned NumOps = TID.getNumOperands();
  bool isLoad = !TID.mayStore();
  const MachineOperand &WB = isLoad ? MI->getOperand(1) : MI->getOperand(0);
  const MachineOperand &Base = MI->getOperand(2);
  const MachineOperand &Offset = MI->getOperand(NumOps-3);
  unsigned WBReg = WB.getReg();
  unsigned BaseReg = Base.getReg();
  unsigned OffReg = Offset.getReg();
  unsigned OffImm = MI->getOperand(NumOps-2).getImm();
  ARMCC::CondCodes Pred = (ARMCC::CondCodes)MI->getOperand(NumOps-1).getImm();
  switch (AddrMode) {
  default:
    assert(false && "Unknown indexed op!");
    return NULL;
  case ARMII::AddrMode2: {
    bool isSub = ARM_AM::getAM2Op(OffImm) == ARM_AM::sub;
    unsigned Amt = ARM_AM::getAM2Offset(OffImm);
    if (OffReg == 0) {
      int SOImmVal = ARM_AM::getSOImmVal(Amt);
      if (SOImmVal == -1)
        // Can't encode it in a so_imm operand. This transformation will
        // add more than 1 instruction. Abandon!
        return NULL;
      UpdateMI = BuildMI(MF, get(isSub ? ARM::SUBri : ARM::ADDri), WBReg)
        .addReg(BaseReg).addImm(SOImmVal)
        .addImm(Pred).addReg(0).addReg(0);
    } else if (Amt != 0) {
      ARM_AM::ShiftOpc ShOpc = ARM_AM::getAM2ShiftOpc(OffImm);
      unsigned SOOpc = ARM_AM::getSORegOpc(ShOpc, Amt);
      UpdateMI = BuildMI(MF, get(isSub ? ARM::SUBrs : ARM::ADDrs), WBReg)
        .addReg(BaseReg).addReg(OffReg).addReg(0).addImm(SOOpc)
        .addImm(Pred).addReg(0).addReg(0);
    } else 
      UpdateMI = BuildMI(MF, get(isSub ? ARM::SUBrr : ARM::ADDrr), WBReg)
        .addReg(BaseReg).addReg(OffReg)
        .addImm(Pred).addReg(0).addReg(0);
    break;
  }
  case ARMII::AddrMode3 : {
    bool isSub = ARM_AM::getAM3Op(OffImm) == ARM_AM::sub;
    unsigned Amt = ARM_AM::getAM3Offset(OffImm);
    if (OffReg == 0)
      // Immediate is 8-bits. It's guaranteed to fit in a so_imm operand.
      UpdateMI = BuildMI(MF, get(isSub ? ARM::SUBri : ARM::ADDri), WBReg)
        .addReg(BaseReg).addImm(Amt)
        .addImm(Pred).addReg(0).addReg(0);
    else
      UpdateMI = BuildMI(MF, get(isSub ? ARM::SUBrr : ARM::ADDrr), WBReg)
        .addReg(BaseReg).addReg(OffReg)
        .addImm(Pred).addReg(0).addReg(0);
    break;
  }
  }

  std::vector<MachineInstr*> NewMIs;
  if (isPre) {
    if (isLoad)
      MemMI = BuildMI(MF, get(MemOpc), MI->getOperand(0).getReg())
        .addReg(WBReg).addReg(0).addImm(0).addImm(Pred);
    else
      MemMI = BuildMI(MF, get(MemOpc)).addReg(MI->getOperand(1).getReg())
        .addReg(WBReg).addReg(0).addImm(0).addImm(Pred);
    NewMIs.push_back(MemMI);
    NewMIs.push_back(UpdateMI);
  } else {
    if (isLoad)
      MemMI = BuildMI(MF, get(MemOpc), MI->getOperand(0).getReg())
        .addReg(BaseReg).addReg(0).addImm(0).addImm(Pred);
    else
      MemMI = BuildMI(MF, get(MemOpc)).addReg(MI->getOperand(1).getReg())
        .addReg(BaseReg).addReg(0).addImm(0).addImm(Pred);
    if (WB.isDead())
      UpdateMI->getOperand(0).setIsDead();
    NewMIs.push_back(UpdateMI);
    NewMIs.push_back(MemMI);
  }
  
  // Transfer LiveVariables states, kill / dead info.
  if (LV) {
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (MO.isReg() && MO.getReg() &&
          TargetRegisterInfo::isVirtualRegister(MO.getReg())) {
        unsigned Reg = MO.getReg();
      
        LiveVariables::VarInfo &VI = LV->getVarInfo(Reg);
        if (MO.isDef()) {
          MachineInstr *NewMI = (Reg == WBReg) ? UpdateMI : MemMI;
          if (MO.isDead())
            LV->addVirtualRegisterDead(Reg, NewMI);
        }
        if (MO.isUse() && MO.isKill()) {
          for (unsigned j = 0; j < 2; ++j) {
            // Look at the two new MI's in reverse order.
            MachineInstr *NewMI = NewMIs[j];
            if (!NewMI->readsRegister(Reg))
              continue;
            LV->addVirtualRegisterKilled(Reg, NewMI);
            if (VI.removeKill(MI))
              VI.Kills.push_back(NewMI);
            break;
          }
        }
      }
    }
  }

  MFI->insert(MBBI, NewMIs[1]);
  MFI->insert(MBBI, NewMIs[0]);
  return NewMIs[0];
}

// Branch analysis.
bool ARMInstrInfo::AnalyzeBranch(MachineBasicBlock &MBB,MachineBasicBlock *&TBB,
                                 MachineBasicBlock *&FBB,
                                 SmallVectorImpl<MachineOperand> &Cond,
                                 bool AllowModify) const {
  // If the block has no terminators, it just falls into the block after it.
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin() || !isUnpredicatedTerminator(--I))
    return false;
  
  // Get the last instruction in the block.
  MachineInstr *LastInst = I;
  
  // If there is only one terminator instruction, process it.
  unsigned LastOpc = LastInst->getOpcode();
  if (I == MBB.begin() || !isUnpredicatedTerminator(--I)) {
    if (LastOpc == ARM::B || LastOpc == ARM::tB) {
      TBB = LastInst->getOperand(0).getMBB();
      return false;
    }
    if (LastOpc == ARM::Bcc || LastOpc == ARM::tBcc) {
      // Block ends with fall-through condbranch.
      TBB = LastInst->getOperand(0).getMBB();
      Cond.push_back(LastInst->getOperand(1));
      Cond.push_back(LastInst->getOperand(2));
      return false;
    }
    return true;  // Can't handle indirect branch.
  }
  
  // Get the instruction before it if it is a terminator.
  MachineInstr *SecondLastInst = I;
  
  // If there are three terminators, we don't know what sort of block this is.
  if (SecondLastInst && I != MBB.begin() && isUnpredicatedTerminator(--I))
    return true;
  
  // If the block ends with ARM::B/ARM::tB and a ARM::Bcc/ARM::tBcc, handle it.
  unsigned SecondLastOpc = SecondLastInst->getOpcode();
  if ((SecondLastOpc == ARM::Bcc && LastOpc == ARM::B) ||
      (SecondLastOpc == ARM::tBcc && LastOpc == ARM::tB)) {
    TBB =  SecondLastInst->getOperand(0).getMBB();
    Cond.push_back(SecondLastInst->getOperand(1));
    Cond.push_back(SecondLastInst->getOperand(2));
    FBB = LastInst->getOperand(0).getMBB();
    return false;
  }
  
  // If the block ends with two unconditional branches, handle it.  The second 
  // one is not executed, so remove it.
  if ((SecondLastOpc == ARM::B || SecondLastOpc==ARM::tB) &&
      (LastOpc == ARM::B || LastOpc == ARM::tB)) {
    TBB = SecondLastInst->getOperand(0).getMBB();
    I = LastInst;
    if (AllowModify)
      I->eraseFromParent();
    return false;
  }

  // Likewise if it ends with a branch table followed by an unconditional branch.
  // The branch folder can create these, and we must get rid of them for
  // correctness of Thumb constant islands.
  if ((SecondLastOpc == ARM::BR_JTr || SecondLastOpc==ARM::BR_JTm ||
       SecondLastOpc == ARM::BR_JTadd || SecondLastOpc==ARM::tBR_JTr) &&
      (LastOpc == ARM::B || LastOpc == ARM::tB)) {
    I = LastInst;
    if (AllowModify)
      I->eraseFromParent();
    return true;
  } 

  // Otherwise, can't handle this.
  return true;
}


unsigned ARMInstrInfo::RemoveBranch(MachineBasicBlock &MBB) const {
  MachineFunction &MF = *MBB.getParent();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  int BOpc   = AFI->isThumbFunction() ? ARM::tB : ARM::B;
  int BccOpc = AFI->isThumbFunction() ? ARM::tBcc : ARM::Bcc;

  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin()) return 0;
  --I;
  if (I->getOpcode() != BOpc && I->getOpcode() != BccOpc)
    return 0;
  
  // Remove the branch.
  I->eraseFromParent();
  
  I = MBB.end();
  
  if (I == MBB.begin()) return 1;
  --I;
  if (I->getOpcode() != BccOpc)
    return 1;
  
  // Remove the branch.
  I->eraseFromParent();
  return 2;
}

unsigned ARMInstrInfo::InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                                MachineBasicBlock *FBB,
                            const SmallVectorImpl<MachineOperand> &Cond) const {
  MachineFunction &MF = *MBB.getParent();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  int BOpc   = AFI->isThumbFunction() ? ARM::tB : ARM::B;
  int BccOpc = AFI->isThumbFunction() ? ARM::tBcc : ARM::Bcc;

  // Shouldn't be a fall through.
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 2 || Cond.size() == 0) &&
         "ARM branch conditions have two components!");
  
  if (FBB == 0) {
    if (Cond.empty()) // Unconditional branch?
      BuildMI(&MBB, get(BOpc)).addMBB(TBB);
    else
      BuildMI(&MBB, get(BccOpc)).addMBB(TBB)
        .addImm(Cond[0].getImm()).addReg(Cond[1].getReg());
    return 1;
  }
  
  // Two-way conditional branch.
  BuildMI(&MBB, get(BccOpc)).addMBB(TBB)
    .addImm(Cond[0].getImm()).addReg(Cond[1].getReg());
  BuildMI(&MBB, get(BOpc)).addMBB(FBB);
  return 2;
}

bool ARMInstrInfo::copyRegToReg(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator I,
                                   unsigned DestReg, unsigned SrcReg,
                                   const TargetRegisterClass *DestRC,
                                   const TargetRegisterClass *SrcRC) const {
  if (DestRC != SrcRC) {
    // Not yet supported!
    return false;
  }

  if (DestRC == ARM::GPRRegisterClass) {
    MachineFunction &MF = *MBB.getParent();
    ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
    if (AFI->isThumbFunction())
      BuildMI(MBB, I, get(ARM::tMOVr), DestReg).addReg(SrcReg);
    else
      AddDefaultCC(AddDefaultPred(BuildMI(MBB, I, get(ARM::MOVr), DestReg)
                                  .addReg(SrcReg)));
  } else if (DestRC == ARM::SPRRegisterClass)
    AddDefaultPred(BuildMI(MBB, I, get(ARM::FCPYS), DestReg)
                   .addReg(SrcReg));
  else if (DestRC == ARM::DPRRegisterClass)
    AddDefaultPred(BuildMI(MBB, I, get(ARM::FCPYD), DestReg)
                   .addReg(SrcReg));
  else
    return false;
  
  return true;
}

static const MachineInstrBuilder &ARMInstrAddOperand(MachineInstrBuilder &MIB,
                                                     MachineOperand &MO) {
  if (MO.isReg())
    MIB = MIB.addReg(MO.getReg(), MO.isDef(), MO.isImplicit());
  else if (MO.isImm())
    MIB = MIB.addImm(MO.getImm());
  else if (MO.isFI())
    MIB = MIB.addFrameIndex(MO.getIndex());
  else
    assert(0 && "Unknown operand for ARMInstrAddOperand!");

  return MIB;
}

void ARMInstrInfo::
storeRegToStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                    unsigned SrcReg, bool isKill, int FI,
                    const TargetRegisterClass *RC) const {
  if (RC == ARM::GPRRegisterClass) {
    MachineFunction &MF = *MBB.getParent();
    ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
    if (AFI->isThumbFunction())
      BuildMI(MBB, I, get(ARM::tSpill)).addReg(SrcReg, false, false, isKill)
        .addFrameIndex(FI).addImm(0);
    else
      AddDefaultPred(BuildMI(MBB, I, get(ARM::STR))
                     .addReg(SrcReg, false, false, isKill)
                     .addFrameIndex(FI).addReg(0).addImm(0));
  } else if (RC == ARM::DPRRegisterClass) {
    AddDefaultPred(BuildMI(MBB, I, get(ARM::FSTD))
                   .addReg(SrcReg, false, false, isKill)
                   .addFrameIndex(FI).addImm(0));
  } else {
    assert(RC == ARM::SPRRegisterClass && "Unknown regclass!");
    AddDefaultPred(BuildMI(MBB, I, get(ARM::FSTS))
                   .addReg(SrcReg, false, false, isKill)
                   .addFrameIndex(FI).addImm(0));
  }
}

void ARMInstrInfo::storeRegToAddr(MachineFunction &MF, unsigned SrcReg,
                                     bool isKill,
                                     SmallVectorImpl<MachineOperand> &Addr,
                                     const TargetRegisterClass *RC,
                                 SmallVectorImpl<MachineInstr*> &NewMIs) const {
  unsigned Opc = 0;
  if (RC == ARM::GPRRegisterClass) {
    ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
    if (AFI->isThumbFunction()) {
      Opc = Addr[0].isFI() ? ARM::tSpill : ARM::tSTR;
      MachineInstrBuilder MIB = 
        BuildMI(MF, get(Opc)).addReg(SrcReg, false, false, isKill);
      for (unsigned i = 0, e = Addr.size(); i != e; ++i)
        MIB = ARMInstrAddOperand(MIB, Addr[i]);
      NewMIs.push_back(MIB);
      return;
    }
    Opc = ARM::STR;
  } else if (RC == ARM::DPRRegisterClass) {
    Opc = ARM::FSTD;
  } else {
    assert(RC == ARM::SPRRegisterClass && "Unknown regclass!");
    Opc = ARM::FSTS;
  }

  MachineInstrBuilder MIB = 
    BuildMI(MF, get(Opc)).addReg(SrcReg, false, false, isKill);
  for (unsigned i = 0, e = Addr.size(); i != e; ++i)
    MIB = ARMInstrAddOperand(MIB, Addr[i]);
  AddDefaultPred(MIB);
  NewMIs.push_back(MIB);
  return;
}

void ARMInstrInfo::
loadRegFromStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                     unsigned DestReg, int FI,
                     const TargetRegisterClass *RC) const {
  if (RC == ARM::GPRRegisterClass) {
    MachineFunction &MF = *MBB.getParent();
    ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
    if (AFI->isThumbFunction())
      BuildMI(MBB, I, get(ARM::tRestore), DestReg)
        .addFrameIndex(FI).addImm(0);
    else
      AddDefaultPred(BuildMI(MBB, I, get(ARM::LDR), DestReg)
                     .addFrameIndex(FI).addReg(0).addImm(0));
  } else if (RC == ARM::DPRRegisterClass) {
    AddDefaultPred(BuildMI(MBB, I, get(ARM::FLDD), DestReg)
                   .addFrameIndex(FI).addImm(0));
  } else {
    assert(RC == ARM::SPRRegisterClass && "Unknown regclass!");
    AddDefaultPred(BuildMI(MBB, I, get(ARM::FLDS), DestReg)
                   .addFrameIndex(FI).addImm(0));
  }
}

void ARMInstrInfo::loadRegFromAddr(MachineFunction &MF, unsigned DestReg,
                                      SmallVectorImpl<MachineOperand> &Addr,
                                      const TargetRegisterClass *RC,
                                 SmallVectorImpl<MachineInstr*> &NewMIs) const {
  unsigned Opc = 0;
  if (RC == ARM::GPRRegisterClass) {
    ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
    if (AFI->isThumbFunction()) {
      Opc = Addr[0].isFI() ? ARM::tRestore : ARM::tLDR;
      MachineInstrBuilder MIB = BuildMI(MF, get(Opc), DestReg);
      for (unsigned i = 0, e = Addr.size(); i != e; ++i)
        MIB = ARMInstrAddOperand(MIB, Addr[i]);
      NewMIs.push_back(MIB);
      return;
    }
    Opc = ARM::LDR;
  } else if (RC == ARM::DPRRegisterClass) {
    Opc = ARM::FLDD;
  } else {
    assert(RC == ARM::SPRRegisterClass && "Unknown regclass!");
    Opc = ARM::FLDS;
  }

  MachineInstrBuilder MIB =  BuildMI(MF, get(Opc), DestReg);
  for (unsigned i = 0, e = Addr.size(); i != e; ++i)
    MIB = ARMInstrAddOperand(MIB, Addr[i]);
  AddDefaultPred(MIB);
  NewMIs.push_back(MIB);
  return;
}

bool ARMInstrInfo::spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                                MachineBasicBlock::iterator MI,
                                const std::vector<CalleeSavedInfo> &CSI) const {
  MachineFunction &MF = *MBB.getParent();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  if (!AFI->isThumbFunction() || CSI.empty())
    return false;

  MachineInstrBuilder MIB = BuildMI(MBB, MI, get(ARM::tPUSH));
  for (unsigned i = CSI.size(); i != 0; --i) {
    unsigned Reg = CSI[i-1].getReg();
    // Add the callee-saved register as live-in. It's killed at the spill.
    MBB.addLiveIn(Reg);
    MIB.addReg(Reg, false/*isDef*/,false/*isImp*/,true/*isKill*/);
  }
  return true;
}

bool ARMInstrInfo::restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                                                 MachineBasicBlock::iterator MI,
                                const std::vector<CalleeSavedInfo> &CSI) const {
  MachineFunction &MF = *MBB.getParent();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  if (!AFI->isThumbFunction() || CSI.empty())
    return false;

  bool isVarArg = AFI->getVarArgsRegSaveSize() > 0;
  MachineInstr *PopMI = MF.CreateMachineInstr(get(ARM::tPOP),MI->getDebugLoc());
  MBB.insert(MI, PopMI);
  for (unsigned i = CSI.size(); i != 0; --i) {
    unsigned Reg = CSI[i-1].getReg();
    if (Reg == ARM::LR) {
      // Special epilogue for vararg functions. See emitEpilogue
      if (isVarArg)
        continue;
      Reg = ARM::PC;
      PopMI->setDesc(get(ARM::tPOP_RET));
      MBB.erase(MI);
    }
    PopMI->addOperand(MachineOperand::CreateReg(Reg, true));
  }
  return true;
}

MachineInstr *ARMInstrInfo::foldMemoryOperandImpl(MachineFunction &MF,
                                                  MachineInstr *MI,
                                        const SmallVectorImpl<unsigned> &Ops,
                                                  int FI) const {
  if (Ops.size() != 1) return NULL;

  unsigned OpNum = Ops[0];
  unsigned Opc = MI->getOpcode();
  MachineInstr *NewMI = NULL;
  switch (Opc) {
  default: break;
  case ARM::MOVr: {
    if (MI->getOperand(4).getReg() == ARM::CPSR)
      // If it is updating CPSR, then it cannot be foled.
      break;
    unsigned Pred = MI->getOperand(2).getImm();
    unsigned PredReg = MI->getOperand(3).getReg();
    if (OpNum == 0) { // move -> store
      unsigned SrcReg = MI->getOperand(1).getReg();
      bool isKill = MI->getOperand(1).isKill();
      NewMI = BuildMI(MF, get(ARM::STR)).addReg(SrcReg, false, false, isKill)
        .addFrameIndex(FI).addReg(0).addImm(0).addImm(Pred).addReg(PredReg);
    } else {          // move -> load
      unsigned DstReg = MI->getOperand(0).getReg();
      bool isDead = MI->getOperand(0).isDead();
      NewMI = BuildMI(MF, get(ARM::LDR)).addReg(DstReg, true, false, false, isDead)
        .addFrameIndex(FI).addReg(0).addImm(0).addImm(Pred).addReg(PredReg);
    }
    break;
  }
  case ARM::tMOVr: {
    if (OpNum == 0) { // move -> store
      unsigned SrcReg = MI->getOperand(1).getReg();
      bool isKill = MI->getOperand(1).isKill();
      if (RI.isPhysicalRegister(SrcReg) && !RI.isLowRegister(SrcReg))
        // tSpill cannot take a high register operand.
        break;
      NewMI = BuildMI(MF, get(ARM::tSpill)).addReg(SrcReg, false, false, isKill)
        .addFrameIndex(FI).addImm(0);
    } else {          // move -> load
      unsigned DstReg = MI->getOperand(0).getReg();
      if (RI.isPhysicalRegister(DstReg) && !RI.isLowRegister(DstReg))
        // tRestore cannot target a high register operand.
        break;
      bool isDead = MI->getOperand(0).isDead();
      NewMI = BuildMI(MF, get(ARM::tRestore))
        .addReg(DstReg, true, false, false, isDead)
        .addFrameIndex(FI).addImm(0);
    }
    break;
  }
  case ARM::FCPYS: {
    unsigned Pred = MI->getOperand(2).getImm();
    unsigned PredReg = MI->getOperand(3).getReg();
    if (OpNum == 0) { // move -> store
      unsigned SrcReg = MI->getOperand(1).getReg();
      NewMI = BuildMI(MF, get(ARM::FSTS)).addReg(SrcReg).addFrameIndex(FI)
        .addImm(0).addImm(Pred).addReg(PredReg);
    } else {          // move -> load
      unsigned DstReg = MI->getOperand(0).getReg();
      NewMI = BuildMI(MF, get(ARM::FLDS), DstReg).addFrameIndex(FI)
        .addImm(0).addImm(Pred).addReg(PredReg);
    }
    break;
  }
  case ARM::FCPYD: {
    unsigned Pred = MI->getOperand(2).getImm();
    unsigned PredReg = MI->getOperand(3).getReg();
    if (OpNum == 0) { // move -> store
      unsigned SrcReg = MI->getOperand(1).getReg();
      bool isKill = MI->getOperand(1).isKill();
      NewMI = BuildMI(MF, get(ARM::FSTD)).addReg(SrcReg, false, false, isKill)
        .addFrameIndex(FI).addImm(0).addImm(Pred).addReg(PredReg);
    } else {          // move -> load
      unsigned DstReg = MI->getOperand(0).getReg();
      bool isDead = MI->getOperand(0).isDead();
      NewMI = BuildMI(MF, get(ARM::FLDD)).addReg(DstReg, true, false, false, isDead)
        .addFrameIndex(FI).addImm(0).addImm(Pred).addReg(PredReg);
    }
    break;
  }
  }

  return NewMI;
}

bool ARMInstrInfo::canFoldMemoryOperand(const MachineInstr *MI,
                                  const SmallVectorImpl<unsigned> &Ops) const {
  if (Ops.size() != 1) return false;

  unsigned OpNum = Ops[0];
  unsigned Opc = MI->getOpcode();
  switch (Opc) {
  default: break;
  case ARM::MOVr:
    // If it is updating CPSR, then it cannot be foled.
    return MI->getOperand(4).getReg() != ARM::CPSR;
  case ARM::tMOVr: {
    if (OpNum == 0) { // move -> store
      unsigned SrcReg = MI->getOperand(1).getReg();
      if (RI.isPhysicalRegister(SrcReg) && !RI.isLowRegister(SrcReg))
        // tSpill cannot take a high register operand.
        return false;
    } else {          // move -> load
      unsigned DstReg = MI->getOperand(0).getReg();
      if (RI.isPhysicalRegister(DstReg) && !RI.isLowRegister(DstReg))
        // tRestore cannot target a high register operand.
        return false;
    }
    return true;
  }
  case ARM::FCPYS:
  case ARM::FCPYD:
    return true;
  }

  return false;
}

bool ARMInstrInfo::BlockHasNoFallThrough(const MachineBasicBlock &MBB) const {
  if (MBB.empty()) return false;
  
  switch (MBB.back().getOpcode()) {
  case ARM::BX_RET:   // Return.
  case ARM::LDM_RET:
  case ARM::tBX_RET:
  case ARM::tBX_RET_vararg:
  case ARM::tPOP_RET:
  case ARM::B:
  case ARM::tB:       // Uncond branch.
  case ARM::tBR_JTr:
  case ARM::BR_JTr:   // Jumptable branch.
  case ARM::BR_JTm:   // Jumptable branch through mem.
  case ARM::BR_JTadd: // Jumptable branch add to pc.
    return true;
  default: return false;
  }
}

bool ARMInstrInfo::
ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const {
  ARMCC::CondCodes CC = (ARMCC::CondCodes)(int)Cond[0].getImm();
  Cond[0].setImm(ARMCC::getOppositeCondition(CC));
  return false;
}

bool ARMInstrInfo::isPredicated(const MachineInstr *MI) const {
  int PIdx = MI->findFirstPredOperandIdx();
  return PIdx != -1 && MI->getOperand(PIdx).getImm() != ARMCC::AL;
}

bool ARMInstrInfo::PredicateInstruction(MachineInstr *MI,
                            const SmallVectorImpl<MachineOperand> &Pred) const {
  unsigned Opc = MI->getOpcode();
  if (Opc == ARM::B || Opc == ARM::tB) {
    MI->setDesc(get(Opc == ARM::B ? ARM::Bcc : ARM::tBcc));
    MI->addOperand(MachineOperand::CreateImm(Pred[0].getImm()));
    MI->addOperand(MachineOperand::CreateReg(Pred[1].getReg(), false));
    return true;
  }

  int PIdx = MI->findFirstPredOperandIdx();
  if (PIdx != -1) {
    MachineOperand &PMO = MI->getOperand(PIdx);
    PMO.setImm(Pred[0].getImm());
    MI->getOperand(PIdx+1).setReg(Pred[1].getReg());
    return true;
  }
  return false;
}

bool
ARMInstrInfo::SubsumesPredicate(const SmallVectorImpl<MachineOperand> &Pred1,
                            const SmallVectorImpl<MachineOperand> &Pred2) const{
  if (Pred1.size() > 2 || Pred2.size() > 2)
    return false;

  ARMCC::CondCodes CC1 = (ARMCC::CondCodes)Pred1[0].getImm();
  ARMCC::CondCodes CC2 = (ARMCC::CondCodes)Pred2[0].getImm();
  if (CC1 == CC2)
    return true;

  switch (CC1) {
  default:
    return false;
  case ARMCC::AL:
    return true;
  case ARMCC::HS:
    return CC2 == ARMCC::HI;
  case ARMCC::LS:
    return CC2 == ARMCC::LO || CC2 == ARMCC::EQ;
  case ARMCC::GE:
    return CC2 == ARMCC::GT;
  case ARMCC::LE:
    return CC2 == ARMCC::LT;
  }
}

bool ARMInstrInfo::DefinesPredicate(MachineInstr *MI,
                                    std::vector<MachineOperand> &Pred) const {
  const TargetInstrDesc &TID = MI->getDesc();
  if (!TID.getImplicitDefs() && !TID.hasOptionalDef())
    return false;

  bool Found = false;
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (MO.isReg() && MO.getReg() == ARM::CPSR) {
      Pred.push_back(MO);
      Found = true;
    }
  }

  return Found;
}


/// FIXME: Works around a gcc miscompilation with -fstrict-aliasing
static unsigned getNumJTEntries(const std::vector<MachineJumpTableEntry> &JT,
                                unsigned JTI) DISABLE_INLINE;
static unsigned getNumJTEntries(const std::vector<MachineJumpTableEntry> &JT,
                                unsigned JTI) {
  return JT[JTI].MBBs.size();
}

/// GetInstSize - Return the size of the specified MachineInstr.
///
unsigned ARMInstrInfo::GetInstSizeInBytes(const MachineInstr *MI) const {
  const MachineBasicBlock &MBB = *MI->getParent();
  const MachineFunction *MF = MBB.getParent();
  const TargetAsmInfo *TAI = MF->getTarget().getTargetAsmInfo();

  // Basic size info comes from the TSFlags field.
  const TargetInstrDesc &TID = MI->getDesc();
  unsigned TSFlags = TID.TSFlags;
  
  switch ((TSFlags & ARMII::SizeMask) >> ARMII::SizeShift) {
  default: {
    // If this machine instr is an inline asm, measure it.
    if (MI->getOpcode() == ARM::INLINEASM)
      return TAI->getInlineAsmLength(MI->getOperand(0).getSymbolName());
    if (MI->isLabel())
      return 0;
    switch (MI->getOpcode()) {
    default:
      assert(0 && "Unknown or unset size field for instr!");
      break;
    case TargetInstrInfo::IMPLICIT_DEF:
    case TargetInstrInfo::DECLARE:
    case TargetInstrInfo::DBG_LABEL:
    case TargetInstrInfo::EH_LABEL:
      return 0;
    }
    break;
  }
  case ARMII::Size8Bytes: return 8;          // Arm instruction x 2.
  case ARMII::Size4Bytes: return 4;          // Arm instruction.
  case ARMII::Size2Bytes: return 2;          // Thumb instruction.
  case ARMII::SizeSpecial: {
    switch (MI->getOpcode()) {
    case ARM::CONSTPOOL_ENTRY:
      // If this machine instr is a constant pool entry, its size is recorded as
      // operand #2.
      return MI->getOperand(2).getImm();
    case ARM::BR_JTr:
    case ARM::BR_JTm:
    case ARM::BR_JTadd:
    case ARM::tBR_JTr: {
      // These are jumptable branches, i.e. a branch followed by an inlined
      // jumptable. The size is 4 + 4 * number of entries.
      unsigned NumOps = TID.getNumOperands();
      MachineOperand JTOP =
        MI->getOperand(NumOps - (TID.isPredicable() ? 3 : 2));
      unsigned JTI = JTOP.getIndex();
      const MachineJumpTableInfo *MJTI = MF->getJumpTableInfo();
      const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
      assert(JTI < JT.size());
      // Thumb instructions are 2 byte aligned, but JT entries are 4 byte
      // 4 aligned. The assembler / linker may add 2 byte padding just before
      // the JT entries.  The size does not include this padding; the
      // constant islands pass does separate bookkeeping for it.
      // FIXME: If we know the size of the function is less than (1 << 16) *2
      // bytes, we can use 16-bit entries instead. Then there won't be an
      // alignment issue.
      return getNumJTEntries(JT, JTI) * 4 + 
             (MI->getOpcode()==ARM::tBR_JTr ? 2 : 4);
    }
    default:
      // Otherwise, pseudo-instruction sizes are zero.
      return 0;
    }
  }
  }
  return 0; // Not reached
}
