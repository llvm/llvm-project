//===- AArch64InstrInfo.cpp - AArch64 Instruction Information -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the AArch64 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64InstrInfo.h"
#include "AArch64MachineFunctionInfo.h"
#include "AArch64TargetMachine.h"
#include "MCTargetDesc/AArch64MCTargetDesc.h"
#include "Utils/AArch64BaseInfo.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

#include <algorithm>

#define GET_INSTRINFO_CTOR_DTOR
#include "AArch64GenInstrInfo.inc"

using namespace llvm;

AArch64InstrInfo::AArch64InstrInfo(const AArch64Subtarget &STI)
  : AArch64GenInstrInfo(AArch64::ADJCALLSTACKDOWN, AArch64::ADJCALLSTACKUP),
    Subtarget(STI) {}

void AArch64InstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator I, DebugLoc DL,
                                   unsigned DestReg, unsigned SrcReg,
                                   bool KillSrc) const {
  unsigned Opc = 0;
  unsigned ZeroReg = 0;
  if (DestReg == AArch64::XSP || SrcReg == AArch64::XSP) {
    // E.g. ADD xDst, xsp, #0 (, lsl #0)
    BuildMI(MBB, I, DL, get(AArch64::ADDxxi_lsl0_s), DestReg)
      .addReg(SrcReg)
      .addImm(0);
    return;
  } else if (DestReg == AArch64::WSP || SrcReg == AArch64::WSP) {
    // E.g. ADD wDST, wsp, #0 (, lsl #0)
    BuildMI(MBB, I, DL, get(AArch64::ADDwwi_lsl0_s), DestReg)
      .addReg(SrcReg)
      .addImm(0);
    return;
  } else if (DestReg == AArch64::NZCV) {
    assert(AArch64::GPR64RegClass.contains(SrcReg));
    // E.g. MSR NZCV, xDST
    BuildMI(MBB, I, DL, get(AArch64::MSRix))
      .addImm(A64SysReg::NZCV)
      .addReg(SrcReg);
  } else if (SrcReg == AArch64::NZCV) {
    assert(AArch64::GPR64RegClass.contains(DestReg));
    // E.g. MRS xDST, NZCV
    BuildMI(MBB, I, DL, get(AArch64::MRSxi), DestReg)
      .addImm(A64SysReg::NZCV);
  } else if (AArch64::GPR64RegClass.contains(DestReg)) {
    if(AArch64::GPR64RegClass.contains(SrcReg)){
      Opc = AArch64::ORRxxx_lsl;
      ZeroReg = AArch64::XZR;
    } else{
      assert(AArch64::FPR64RegClass.contains(SrcReg));
      BuildMI(MBB, I, DL, get(AArch64::FMOVxd), DestReg)
        .addReg(SrcReg);
      return;
    }
  } else if (AArch64::GPR32RegClass.contains(DestReg)) {
    if(AArch64::GPR32RegClass.contains(SrcReg)){
      Opc = AArch64::ORRwww_lsl;
      ZeroReg = AArch64::WZR;
    } else{
      assert(AArch64::FPR32RegClass.contains(SrcReg));
      BuildMI(MBB, I, DL, get(AArch64::FMOVws), DestReg)
        .addReg(SrcReg);
      return;
    }
  } else if (AArch64::FPR32RegClass.contains(DestReg)) {
    if(AArch64::FPR32RegClass.contains(SrcReg)){
      BuildMI(MBB, I, DL, get(AArch64::FMOVss), DestReg)
        .addReg(SrcReg);
      return;
    }
    else {
      assert(AArch64::GPR32RegClass.contains(SrcReg));
      BuildMI(MBB, I, DL, get(AArch64::FMOVsw), DestReg)
        .addReg(SrcReg);
      return;
    }
  } else if (AArch64::FPR64RegClass.contains(DestReg)) {
    if(AArch64::FPR64RegClass.contains(SrcReg)){
      BuildMI(MBB, I, DL, get(AArch64::FMOVdd), DestReg)
        .addReg(SrcReg);
      return;
    }
    else {
      assert(AArch64::GPR64RegClass.contains(SrcReg));
      BuildMI(MBB, I, DL, get(AArch64::FMOVdx), DestReg)
        .addReg(SrcReg);
      return;
    }
  } else if (AArch64::FPR128RegClass.contains(DestReg)) {
    assert(AArch64::FPR128RegClass.contains(SrcReg));

    // If NEON is enable, we use ORR to implement this copy.
    // If NEON isn't available, emit STR and LDR to handle this.
    if(getSubTarget().hasNEON()) {
      BuildMI(MBB, I, DL, get(AArch64::ORRvvv_16B), DestReg)
        .addReg(SrcReg)
        .addReg(SrcReg);
      return;
    } else {
      BuildMI(MBB, I, DL, get(AArch64::LSFP128_PreInd_STR), AArch64::XSP)
        .addReg(SrcReg)
        .addReg(AArch64::XSP)
        .addImm(0x1ff & -16);

      BuildMI(MBB, I, DL, get(AArch64::LSFP128_PostInd_LDR), DestReg)
        .addReg(AArch64::XSP, RegState::Define)
        .addReg(AArch64::XSP)
        .addImm(16);
      return;
    }
  } else {
    llvm_unreachable("Unknown register class in copyPhysReg");
  }

  // E.g. ORR xDst, xzr, xSrc, lsl #0
  BuildMI(MBB, I, DL, get(Opc), DestReg)
    .addReg(ZeroReg)
    .addReg(SrcReg)
    .addImm(0);
}

/// Does the Opcode represent a conditional branch that we can remove and re-add
/// at the end of a basic block?
static bool isCondBranch(unsigned Opc) {
  return Opc == AArch64::Bcc || Opc == AArch64::CBZw || Opc == AArch64::CBZx ||
         Opc == AArch64::CBNZw || Opc == AArch64::CBNZx ||
         Opc == AArch64::TBZwii || Opc == AArch64::TBZxii ||
         Opc == AArch64::TBNZwii || Opc == AArch64::TBNZxii;
}

/// Takes apart a given conditional branch MachineInstr (see isCondBranch),
/// setting TBB to the destination basic block and populating the Cond vector
/// with data necessary to recreate the conditional branch at a later
/// date. First element will be the opcode, and subsequent ones define the
/// conditions being branched on in an instruction-specific manner.
static void classifyCondBranch(MachineInstr *I, MachineBasicBlock *&TBB,
                               SmallVectorImpl<MachineOperand> &Cond) {
  switch(I->getOpcode()) {
  case AArch64::Bcc:
  case AArch64::CBZw:
  case AArch64::CBZx:
  case AArch64::CBNZw:
  case AArch64::CBNZx:
    // These instructions just have one predicate operand in position 0 (either
    // a condition code or a register being compared).
    Cond.push_back(MachineOperand::CreateImm(I->getOpcode()));
    Cond.push_back(I->getOperand(0));
    TBB = I->getOperand(1).getMBB();
    return;
  case AArch64::TBZwii:
  case AArch64::TBZxii:
  case AArch64::TBNZwii:
  case AArch64::TBNZxii:
    // These have two predicate operands: a register and a bit position.
    Cond.push_back(MachineOperand::CreateImm(I->getOpcode()));
    Cond.push_back(I->getOperand(0));
    Cond.push_back(I->getOperand(1));
    TBB = I->getOperand(2).getMBB();
    return;
  default:
    llvm_unreachable("Unknown conditional branch to classify");
  }
}


bool
AArch64InstrInfo::AnalyzeBranch(MachineBasicBlock &MBB,MachineBasicBlock *&TBB,
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
  unsigned LastOpc = LastInst->getOpcode();
  if (I == MBB.begin() || !isUnpredicatedTerminator(--I)) {
    if (LastOpc == AArch64::Bimm) {
      TBB = LastInst->getOperand(0).getMBB();
      return false;
    }
    if (isCondBranch(LastOpc)) {
      classifyCondBranch(LastInst, TBB, Cond);
      return false;
    }
    return true;  // Can't handle indirect branch.
  }

  // Get the instruction before it if it is a terminator.
  MachineInstr *SecondLastInst = I;
  unsigned SecondLastOpc = SecondLastInst->getOpcode();

  // If AllowModify is true and the block ends with two or more unconditional
  // branches, delete all but the first unconditional branch.
  if (AllowModify && LastOpc == AArch64::Bimm) {
    while (SecondLastOpc == AArch64::Bimm) {
      LastInst->eraseFromParent();
      LastInst = SecondLastInst;
      LastOpc = LastInst->getOpcode();
      if (I == MBB.begin() || !isUnpredicatedTerminator(--I)) {
        // Return now the only terminator is an unconditional branch.
        TBB = LastInst->getOperand(0).getMBB();
        return false;
      } else {
        SecondLastInst = I;
        SecondLastOpc = SecondLastInst->getOpcode();
      }
    }
  }

  // If there are three terminators, we don't know what sort of block this is.
  if (SecondLastInst && I != MBB.begin() && isUnpredicatedTerminator(--I))
    return true;

  // If the block ends with a B and a Bcc, handle it.
  if (LastOpc == AArch64::Bimm) {
    if (SecondLastOpc == AArch64::Bcc) {
      TBB =  SecondLastInst->getOperand(1).getMBB();
      Cond.push_back(MachineOperand::CreateImm(AArch64::Bcc));
      Cond.push_back(SecondLastInst->getOperand(0));
      FBB = LastInst->getOperand(0).getMBB();
      return false;
    } else if (isCondBranch(SecondLastOpc)) {
      classifyCondBranch(SecondLastInst, TBB, Cond);
      FBB = LastInst->getOperand(0).getMBB();
      return false;
    }
  }

  // If the block ends with two unconditional branches, handle it.  The second
  // one is not executed, so remove it.
  if (SecondLastOpc == AArch64::Bimm && LastOpc == AArch64::Bimm) {
    TBB = SecondLastInst->getOperand(0).getMBB();
    I = LastInst;
    if (AllowModify)
      I->eraseFromParent();
    return false;
  }

  // Otherwise, can't handle this.
  return true;
}

bool AArch64InstrInfo::ReverseBranchCondition(
                                  SmallVectorImpl<MachineOperand> &Cond) const {
  switch (Cond[0].getImm()) {
  case AArch64::Bcc: {
    A64CC::CondCodes CC = static_cast<A64CC::CondCodes>(Cond[1].getImm());
    CC = A64InvertCondCode(CC);
    Cond[1].setImm(CC);
    return false;
  }
  case AArch64::CBZw:
    Cond[0].setImm(AArch64::CBNZw);
    return false;
  case AArch64::CBZx:
    Cond[0].setImm(AArch64::CBNZx);
    return false;
  case AArch64::CBNZw:
    Cond[0].setImm(AArch64::CBZw);
    return false;
  case AArch64::CBNZx:
    Cond[0].setImm(AArch64::CBZx);
    return false;
  case AArch64::TBZwii:
    Cond[0].setImm(AArch64::TBNZwii);
    return false;
  case AArch64::TBZxii:
    Cond[0].setImm(AArch64::TBNZxii);
    return false;
  case AArch64::TBNZwii:
    Cond[0].setImm(AArch64::TBZwii);
    return false;
  case AArch64::TBNZxii:
    Cond[0].setImm(AArch64::TBZxii);
    return false;
  default:
    llvm_unreachable("Unknown branch type");
  }
}


unsigned
AArch64InstrInfo::InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                               MachineBasicBlock *FBB,
                               const SmallVectorImpl<MachineOperand> &Cond,
                               DebugLoc DL) const {
  if (FBB == 0 && Cond.empty()) {
    BuildMI(&MBB, DL, get(AArch64::Bimm)).addMBB(TBB);
    return 1;
  } else if (FBB == 0) {
    MachineInstrBuilder MIB = BuildMI(&MBB, DL, get(Cond[0].getImm()));
    for (int i = 1, e = Cond.size(); i != e; ++i)
      MIB.addOperand(Cond[i]);
    MIB.addMBB(TBB);
    return 1;
  }

  MachineInstrBuilder MIB = BuildMI(&MBB, DL, get(Cond[0].getImm()));
  for (int i = 1, e = Cond.size(); i != e; ++i)
    MIB.addOperand(Cond[i]);
  MIB.addMBB(TBB);

  BuildMI(&MBB, DL, get(AArch64::Bimm)).addMBB(FBB);
  return 2;
}

unsigned AArch64InstrInfo::RemoveBranch(MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin()) return 0;
  --I;
  while (I->isDebugValue()) {
    if (I == MBB.begin())
      return 0;
    --I;
  }
  if (I->getOpcode() != AArch64::Bimm && !isCondBranch(I->getOpcode()))
    return 0;

  // Remove the branch.
  I->eraseFromParent();

  I = MBB.end();

  if (I == MBB.begin()) return 1;
  --I;
  if (!isCondBranch(I->getOpcode()))
    return 1;

  // Remove the branch.
  I->eraseFromParent();
  return 2;
}

bool
AArch64InstrInfo::expandPostRAPseudo(MachineBasicBlock::iterator MBBI) const {
  MachineInstr &MI = *MBBI;
  MachineBasicBlock &MBB = *MI.getParent();

  unsigned Opcode = MI.getOpcode();
  switch (Opcode) {
  case AArch64::TLSDESC_BLRx: {
    MachineInstr *NewMI =
      BuildMI(MBB, MBBI, MI.getDebugLoc(), get(AArch64::TLSDESCCALL))
        .addOperand(MI.getOperand(1));
    MI.setDesc(get(AArch64::BLRx));

    llvm::finalizeBundle(MBB, NewMI, *++MBBI);
    return true;
    }
  default:
    return false;
  }

  return false;
}

void
AArch64InstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator MBBI,
                                      unsigned SrcReg, bool isKill,
                                      int FrameIdx,
                                      const TargetRegisterClass *RC,
                                      const TargetRegisterInfo *TRI) const {
  DebugLoc DL = MBB.findDebugLoc(MBBI);
  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo &MFI = *MF.getFrameInfo();
  unsigned Align = MFI.getObjectAlignment(FrameIdx);

  MachineMemOperand *MMO
    = MF.getMachineMemOperand(MachinePointerInfo::getFixedStack(FrameIdx),
                              MachineMemOperand::MOStore,
                              MFI.getObjectSize(FrameIdx),
                              Align);

  unsigned StoreOp = 0;
  if (RC->hasType(MVT::i64) || RC->hasType(MVT::i32)) {
    switch(RC->getSize()) {
    case 4: StoreOp = AArch64::LS32_STR; break;
    case 8: StoreOp = AArch64::LS64_STR; break;
    default:
      llvm_unreachable("Unknown size for regclass");
    }
  } else {
    assert((RC->hasType(MVT::f32) || RC->hasType(MVT::f64) ||
            RC->hasType(MVT::f128))
           && "Expected integer or floating type for store");
    switch (RC->getSize()) {
    case 4: StoreOp = AArch64::LSFP32_STR; break;
    case 8: StoreOp = AArch64::LSFP64_STR; break;
    case 16: StoreOp = AArch64::LSFP128_STR; break;
    default:
      llvm_unreachable("Unknown size for regclass");
    }
  }

  MachineInstrBuilder NewMI = BuildMI(MBB, MBBI, DL, get(StoreOp));
  NewMI.addReg(SrcReg, getKillRegState(isKill))
    .addFrameIndex(FrameIdx)
    .addImm(0)
    .addMemOperand(MMO);

}

void
AArch64InstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator MBBI,
                                       unsigned DestReg, int FrameIdx,
                                       const TargetRegisterClass *RC,
                                       const TargetRegisterInfo *TRI) const {
  DebugLoc DL = MBB.findDebugLoc(MBBI);
  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo &MFI = *MF.getFrameInfo();
  unsigned Align = MFI.getObjectAlignment(FrameIdx);

  MachineMemOperand *MMO
    = MF.getMachineMemOperand(MachinePointerInfo::getFixedStack(FrameIdx),
                              MachineMemOperand::MOLoad,
                              MFI.getObjectSize(FrameIdx),
                              Align);

  unsigned LoadOp = 0;
  if (RC->hasType(MVT::i64) || RC->hasType(MVT::i32)) {
    switch(RC->getSize()) {
    case 4: LoadOp = AArch64::LS32_LDR; break;
    case 8: LoadOp = AArch64::LS64_LDR; break;
    default:
      llvm_unreachable("Unknown size for regclass");
    }
  } else {
    assert((RC->hasType(MVT::f32) || RC->hasType(MVT::f64)
            || RC->hasType(MVT::f128))
           && "Expected integer or floating type for store");
    switch (RC->getSize()) {
    case 4: LoadOp = AArch64::LSFP32_LDR; break;
    case 8: LoadOp = AArch64::LSFP64_LDR; break;
    case 16: LoadOp = AArch64::LSFP128_LDR; break;
    default:
      llvm_unreachable("Unknown size for regclass");
    }
  }

  MachineInstrBuilder NewMI = BuildMI(MBB, MBBI, DL, get(LoadOp), DestReg);
  NewMI.addFrameIndex(FrameIdx)
       .addImm(0)
       .addMemOperand(MMO);
}

unsigned AArch64InstrInfo::estimateRSStackLimit(MachineFunction &MF) const {
  unsigned Limit = (1 << 16) - 1;
  for (MachineFunction::iterator BB = MF.begin(),E = MF.end(); BB != E; ++BB) {
    for (MachineBasicBlock::iterator I = BB->begin(), E = BB->end();
         I != E; ++I) {
      for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
        if (!I->getOperand(i).isFI()) continue;

        // When using ADDxxi_lsl0_s to get the address of a stack object, 0xfff
        // is the largest offset guaranteed to fit in the immediate offset.
        if (I->getOpcode() == AArch64::ADDxxi_lsl0_s) {
          Limit = std::min(Limit, 0xfffu);
          break;
        }

        int AccessScale, MinOffset, MaxOffset;
        getAddressConstraints(*I, AccessScale, MinOffset, MaxOffset);
        Limit = std::min(Limit, static_cast<unsigned>(MaxOffset));

        break; // At most one FI per instruction
      }
    }
  }

  return Limit;
}
void AArch64InstrInfo::getAddressConstraints(const MachineInstr &MI,
                                             int &AccessScale, int &MinOffset,
                                             int &MaxOffset) const {
  switch (MI.getOpcode()) {
  default: llvm_unreachable("Unkown load/store kind");
  case TargetOpcode::DBG_VALUE:
    AccessScale = 1;
    MinOffset = INT_MIN;
    MaxOffset = INT_MAX;
    return;
  case AArch64::LS8_LDR: case AArch64::LS8_STR:
  case AArch64::LSFP8_LDR: case AArch64::LSFP8_STR:
  case AArch64::LDRSBw:
  case AArch64::LDRSBx:
    AccessScale = 1;
    MinOffset = 0;
    MaxOffset = 0xfff;
    return;
  case AArch64::LS16_LDR: case AArch64::LS16_STR:
  case AArch64::LSFP16_LDR: case AArch64::LSFP16_STR:
  case AArch64::LDRSHw:
  case AArch64::LDRSHx:
    AccessScale = 2;
    MinOffset = 0;
    MaxOffset = 0xfff * AccessScale;
    return;
  case AArch64::LS32_LDR:  case AArch64::LS32_STR:
  case AArch64::LSFP32_LDR: case AArch64::LSFP32_STR:
  case AArch64::LDRSWx:
  case AArch64::LDPSWx:
    AccessScale = 4;
    MinOffset = 0;
    MaxOffset = 0xfff * AccessScale;
    return;
  case AArch64::LS64_LDR: case AArch64::LS64_STR:
  case AArch64::LSFP64_LDR: case AArch64::LSFP64_STR:
  case AArch64::PRFM:
    AccessScale = 8;
    MinOffset = 0;
    MaxOffset = 0xfff * AccessScale;
    return;
  case AArch64::LSFP128_LDR: case AArch64::LSFP128_STR:
    AccessScale = 16;
    MinOffset = 0;
    MaxOffset = 0xfff * AccessScale;
    return;
  case AArch64::LSPair32_LDR: case AArch64::LSPair32_STR:
  case AArch64::LSFPPair32_LDR: case AArch64::LSFPPair32_STR:
    AccessScale = 4;
    MinOffset = -0x40 * AccessScale;
    MaxOffset = 0x3f * AccessScale;
    return;
  case AArch64::LSPair64_LDR: case AArch64::LSPair64_STR:
  case AArch64::LSFPPair64_LDR: case AArch64::LSFPPair64_STR:
    AccessScale = 8;
    MinOffset = -0x40 * AccessScale;
    MaxOffset = 0x3f * AccessScale;
    return;
  case AArch64::LSFPPair128_LDR: case AArch64::LSFPPair128_STR:
    AccessScale = 16;
    MinOffset = -0x40 * AccessScale;
    MaxOffset = 0x3f * AccessScale;
    return;
  }
}

unsigned AArch64InstrInfo::getInstSizeInBytes(const MachineInstr &MI) const {
  const MCInstrDesc &MCID = MI.getDesc();
  const MachineBasicBlock &MBB = *MI.getParent();
  const MachineFunction &MF = *MBB.getParent();
  const MCAsmInfo &MAI = *MF.getTarget().getMCAsmInfo();

  if (MCID.getSize())
    return MCID.getSize();

  if (MI.getOpcode() == AArch64::INLINEASM)
    return getInlineAsmLength(MI.getOperand(0).getSymbolName(), MAI);

  if (MI.isLabel())
    return 0;

  switch (MI.getOpcode()) {
  case TargetOpcode::BUNDLE:
    return getInstBundleLength(MI);
  case TargetOpcode::IMPLICIT_DEF:
  case TargetOpcode::KILL:
  case TargetOpcode::PROLOG_LABEL:
  case TargetOpcode::EH_LABEL:
  case TargetOpcode::DBG_VALUE:
    return 0;
  case AArch64::TLSDESCCALL:
    return 0;
  default:
    llvm_unreachable("Unknown instruction class");
  }
}

unsigned AArch64InstrInfo::getInstBundleLength(const MachineInstr &MI) const {
  unsigned Size = 0;
  MachineBasicBlock::const_instr_iterator I = MI;
  MachineBasicBlock::const_instr_iterator E = MI.getParent()->instr_end();
  while (++I != E && I->isInsideBundle()) {
    assert(!I->isBundle() && "No nested bundle!");
    Size += getInstSizeInBytes(*I);
  }
  return Size;
}

bool llvm::rewriteA64FrameIndex(MachineInstr &MI, unsigned FrameRegIdx,
                                unsigned FrameReg, int &Offset,
                                const AArch64InstrInfo &TII) {
  MachineBasicBlock &MBB = *MI.getParent();
  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo &MFI = *MF.getFrameInfo();

  MFI.getObjectOffset(FrameRegIdx);
  llvm_unreachable("Unimplemented rewriteFrameIndex");
}

void llvm::emitRegUpdate(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator MBBI,
                         DebugLoc dl, const TargetInstrInfo &TII,
                         unsigned DstReg, unsigned SrcReg, unsigned ScratchReg,
                         int64_t NumBytes, MachineInstr::MIFlag MIFlags) {
  if (NumBytes == 0 && DstReg == SrcReg)
    return;
  else if (abs64(NumBytes) & ~0xffffff) {
    // Generically, we have to materialize the offset into a temporary register
    // and subtract it. There are a couple of ways this could be done, for now
    // we'll use a movz/movk or movn/movk sequence.
    uint64_t Bits = static_cast<uint64_t>(abs64(NumBytes));
    BuildMI(MBB, MBBI, dl, TII.get(AArch64::MOVZxii), ScratchReg)
      .addImm(0xffff & Bits).addImm(0)
      .setMIFlags(MIFlags);

    Bits >>= 16;
    if (Bits & 0xffff) {
      BuildMI(MBB, MBBI, dl, TII.get(AArch64::MOVKxii), ScratchReg)
        .addReg(ScratchReg)
        .addImm(0xffff & Bits).addImm(1)
        .setMIFlags(MIFlags);
    }

    Bits >>= 16;
    if (Bits & 0xffff) {
      BuildMI(MBB, MBBI, dl, TII.get(AArch64::MOVKxii), ScratchReg)
        .addReg(ScratchReg)
        .addImm(0xffff & Bits).addImm(2)
        .setMIFlags(MIFlags);
    }

    Bits >>= 16;
    if (Bits & 0xffff) {
      BuildMI(MBB, MBBI, dl, TII.get(AArch64::MOVKxii), ScratchReg)
        .addReg(ScratchReg)
        .addImm(0xffff & Bits).addImm(3)
        .setMIFlags(MIFlags);
    }

    // ADD DST, SRC, xTMP (, lsl #0)
    unsigned AddOp = NumBytes > 0 ? AArch64::ADDxxx_uxtx : AArch64::SUBxxx_uxtx;
    BuildMI(MBB, MBBI, dl, TII.get(AddOp), DstReg)
      .addReg(SrcReg, RegState::Kill)
      .addReg(ScratchReg, RegState::Kill)
      .addImm(0)
      .setMIFlag(MIFlags);
    return;
  }

  // Now we know that the adjustment can be done in at most two add/sub
  // (immediate) instructions, which is always more efficient than a
  // literal-pool load, or even a hypothetical movz/movk/add sequence

  // Decide whether we're doing addition or subtraction
  unsigned LowOp, HighOp;
  if (NumBytes >= 0) {
    LowOp = AArch64::ADDxxi_lsl0_s;
    HighOp = AArch64::ADDxxi_lsl12_s;
  } else {
    LowOp = AArch64::SUBxxi_lsl0_s;
    HighOp = AArch64::SUBxxi_lsl12_s;
    NumBytes = abs64(NumBytes);
  }

  // If we're here, at the very least a move needs to be produced, which just
  // happens to be materializable by an ADD.
  if ((NumBytes & 0xfff) || NumBytes == 0) {
    BuildMI(MBB, MBBI, dl, TII.get(LowOp), DstReg)
      .addReg(SrcReg, RegState::Kill)
      .addImm(NumBytes & 0xfff)
      .setMIFlag(MIFlags);

    // Next update should use the register we've just defined.
    SrcReg = DstReg;
  }

  if (NumBytes & 0xfff000) {
    BuildMI(MBB, MBBI, dl, TII.get(HighOp), DstReg)
      .addReg(SrcReg, RegState::Kill)
      .addImm(NumBytes >> 12)
      .setMIFlag(MIFlags);
  }
}

void llvm::emitSPUpdate(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                        DebugLoc dl, const TargetInstrInfo &TII,
                        unsigned ScratchReg, int64_t NumBytes,
                        MachineInstr::MIFlag MIFlags) {
  emitRegUpdate(MBB, MI, dl, TII, AArch64::XSP, AArch64::XSP, AArch64::X16,
                NumBytes, MIFlags);
}


namespace {
  struct LDTLSCleanup : public MachineFunctionPass {
    static char ID;
    LDTLSCleanup() : MachineFunctionPass(ID) {}

    virtual bool runOnMachineFunction(MachineFunction &MF) {
      AArch64MachineFunctionInfo* MFI
        = MF.getInfo<AArch64MachineFunctionInfo>();
      if (MFI->getNumLocalDynamicTLSAccesses() < 2) {
        // No point folding accesses if there isn't at least two.
        return false;
      }

      MachineDominatorTree *DT = &getAnalysis<MachineDominatorTree>();
      return VisitNode(DT->getRootNode(), 0);
    }

    // Visit the dominator subtree rooted at Node in pre-order.
    // If TLSBaseAddrReg is non-null, then use that to replace any
    // TLS_base_addr instructions. Otherwise, create the register
    // when the first such instruction is seen, and then use it
    // as we encounter more instructions.
    bool VisitNode(MachineDomTreeNode *Node, unsigned TLSBaseAddrReg) {
      MachineBasicBlock *BB = Node->getBlock();
      bool Changed = false;

      // Traverse the current block.
      for (MachineBasicBlock::iterator I = BB->begin(), E = BB->end(); I != E;
           ++I) {
        switch (I->getOpcode()) {
        case AArch64::TLSDESC_BLRx:
          // Make sure it's a local dynamic access.
          if (!I->getOperand(1).isSymbol() ||
              strcmp(I->getOperand(1).getSymbolName(), "_TLS_MODULE_BASE_"))
            break;

          if (TLSBaseAddrReg)
            I = ReplaceTLSBaseAddrCall(I, TLSBaseAddrReg);
          else
            I = SetRegister(I, &TLSBaseAddrReg);
          Changed = true;
          break;
        default:
          break;
        }
      }

      // Visit the children of this block in the dominator tree.
      for (MachineDomTreeNode::iterator I = Node->begin(), E = Node->end();
           I != E; ++I) {
        Changed |= VisitNode(*I, TLSBaseAddrReg);
      }

      return Changed;
    }

    // Replace the TLS_base_addr instruction I with a copy from
    // TLSBaseAddrReg, returning the new instruction.
    MachineInstr *ReplaceTLSBaseAddrCall(MachineInstr *I,
                                         unsigned TLSBaseAddrReg) {
      MachineFunction *MF = I->getParent()->getParent();
      const AArch64TargetMachine *TM =
          static_cast<const AArch64TargetMachine *>(&MF->getTarget());
      const AArch64InstrInfo *TII = TM->getInstrInfo();

      // Insert a Copy from TLSBaseAddrReg to x0, which is where the rest of the
      // code sequence assumes the address will be.
      MachineInstr *Copy = BuildMI(*I->getParent(), I, I->getDebugLoc(),
                                   TII->get(TargetOpcode::COPY),
                                   AArch64::X0)
        .addReg(TLSBaseAddrReg);

      // Erase the TLS_base_addr instruction.
      I->eraseFromParent();

      return Copy;
    }

    // Create a virtal register in *TLSBaseAddrReg, and populate it by
    // inserting a copy instruction after I. Returns the new instruction.
    MachineInstr *SetRegister(MachineInstr *I, unsigned *TLSBaseAddrReg) {
      MachineFunction *MF = I->getParent()->getParent();
      const AArch64TargetMachine *TM =
          static_cast<const AArch64TargetMachine *>(&MF->getTarget());
      const AArch64InstrInfo *TII = TM->getInstrInfo();

      // Create a virtual register for the TLS base address.
      MachineRegisterInfo &RegInfo = MF->getRegInfo();
      *TLSBaseAddrReg = RegInfo.createVirtualRegister(&AArch64::GPR64RegClass);

      // Insert a copy from X0 to TLSBaseAddrReg for later.
      MachineInstr *Next = I->getNextNode();
      MachineInstr *Copy = BuildMI(*I->getParent(), Next, I->getDebugLoc(),
                                   TII->get(TargetOpcode::COPY),
                                   *TLSBaseAddrReg)
        .addReg(AArch64::X0);

      return Copy;
    }

    virtual const char *getPassName() const {
      return "Local Dynamic TLS Access Clean-up";
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<MachineDominatorTree>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }
  };
}

char LDTLSCleanup::ID = 0;
FunctionPass*
llvm::createAArch64CleanupLocalDynamicTLSPass() { return new LDTLSCleanup(); }
