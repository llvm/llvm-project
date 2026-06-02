//===-- EZHInstrInfo.cpp - EZH Instruction Information ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the EZH implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "EZHInstrInfo.h"
#include "EZHCondCode.h"
#include "EZHSubtarget.h"
#include "MCTargetDesc/EZHMCTargetDesc.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define GET_INSTRINFO_CTOR_DTOR
#include "EZHGenInstrInfo.inc"

#define DEBUG_TYPE "ezh-instr-info"

#define MAP_CC(OP)                                                             \
  case EZH::OP##__:                                                            \
    switch (CC) {                                                              \
    case LPCC::ICC_EU:                                                         \
      return Opc;                                                              \
    case LPCC::ICC_ZE:                                                         \
      return EZH::OP##___ze;                                                   \
    case LPCC::ICC_NZ:                                                         \
      return EZH::OP##___nz;                                                   \
    case LPCC::ICC_PO:                                                         \
      return EZH::OP##___po;                                                   \
    case LPCC::ICC_NE:                                                         \
      return EZH::OP##___ne;                                                   \
    case LPCC::ICC_AZ:                                                         \
      return EZH::OP##___az;                                                   \
    case LPCC::ICC_ZB:                                                         \
      return EZH::OP##___zb;                                                   \
    case LPCC::ICC_CA:                                                         \
      return EZH::OP##___ca;                                                   \
    case LPCC::ICC_NC:                                                         \
      return EZH::OP##___nc;                                                   \
    case LPCC::ICC_CZ:                                                         \
      return EZH::OP##___cz;                                                   \
    case LPCC::ICC_SPO:                                                        \
      return EZH::OP##___spo;                                                  \
    case LPCC::ICC_SNE:                                                        \
      return EZH::OP##___sne;                                                  \
    case LPCC::ICC_NBS:                                                        \
      return EZH::OP##___nbs;                                                  \
    case LPCC::ICC_NEX:                                                        \
      return EZH::OP##___nex;                                                  \
    case LPCC::ICC_BS:                                                         \
      return EZH::OP##___bs;                                                   \
    case LPCC::ICC_EX:                                                         \
      return EZH::OP##___ex;                                                   \
    case LPCC::UNKNOWN:                                                        \
      return Opc;                                                              \
    }                                                                          \
    break;

static unsigned getConditionalOpcode(unsigned Opc, LPCC::CondCode CC) {
  if (CC == LPCC::ICC_EU)
    return Opc;

  switch (Opc) {
    MAP_CC(ADDrr)
    MAP_CC(ADDri)
    MAP_CC(SUBrr)
    MAP_CC(SUBri)
    MAP_CC(ANDrr)
    MAP_CC(ANDri)
    MAP_CC(ORrr)
    MAP_CC(ORri)
    MAP_CC(XORrr)
    MAP_CC(XORri)
    MAP_CC(LSLi)
    MAP_CC(LSRi)
    MAP_CC(ASRi)
    MAP_CC(RORi)
    MAP_CC(MOVrr)
    MAP_CC(MOVri)
  default:
    return Opc;
  }
  return Opc;
}
#undef MAP_CC

EZHInstrInfo::EZHInstrInfo(const EZHSubtarget &STI)
    : EZHGenInstrInfo(STI, RegisterInfo, EZH::ADJCALLSTACKDOWN,
                      EZH::ADJCALLSTACKUP),
      RegisterInfo() {}

void EZHInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator Position,
                               const DebugLoc &DL, Register DestinationRegister,
                               Register SourceRegister, bool KillSource,
                               bool RenamableDest, bool RenamableSrc) const {
  if (!EZH::GPRAllRegClass.contains(DestinationRegister, SourceRegister)) {
    llvm_unreachable("Impossible reg-to-reg copy");
  }

  BuildMI(MBB, Position, DL, get(EZH::MOVrr__), DestinationRegister)
      .addReg(SourceRegister, getKillRegState(KillSource));
}

void EZHInstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator Position,
                                       Register SourceRegister, bool IsKill,
                                       int FrameIndex,
                                       const TargetRegisterClass *RegisterClass,
                                       Register /*VReg*/,
                                       MachineInstr::MIFlag Flags) const {
  DebugLoc DL;
  if (Position != MBB.end()) {
    DL = Position->getDebugLoc();
  }

  if (!EZH::GPRAllRegClass.hasSubClassEq(RegisterClass)) {
    llvm_unreachable("Can't store this register to stack slot");
  }
  BuildMI(MBB, Position, DL, get(EZH::STR))
      .addReg(SourceRegister, getKillRegState(IsKill))
      .addFrameIndex(FrameIndex)
      .addImm(0)
      .setMIFlags(Flags);
}

void EZHInstrInfo::loadRegFromStackSlot(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator Position,
    Register DestinationRegister, int FrameIndex,
    const TargetRegisterClass *RegisterClass, Register /*VReg*/,
    unsigned /*SubReg*/, MachineInstr::MIFlag Flags) const {
  DebugLoc DL;
  if (Position != MBB.end()) {
    DL = Position->getDebugLoc();
  }

  if (!EZH::GPRAllRegClass.hasSubClassEq(RegisterClass)) {
    llvm_unreachable("Can't load this register from stack slot");
  }
  BuildMI(MBB, Position, DL, get(EZH::LDR), DestinationRegister)
      .addFrameIndex(FrameIndex)
      .addImm(0)
      .setMIFlags(Flags);
}

bool EZHInstrInfo::expandPostRAPseudo(MachineInstr &MI) const { return false; }

bool EZHInstrInfo::analyzeBranch(MachineBasicBlock &MBB,
                                 MachineBasicBlock *&TrueBlock,
                                 MachineBasicBlock *&FalseBlock,
                                 SmallVectorImpl<MachineOperand> &Condition,
                                 bool AllowModify) const {
  TrueBlock = nullptr;
  FalseBlock = nullptr;
  unsigned NumTerminatorsSeen = 0;

  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin())
    return false;
  --I;

  while (I->isTerminator() || I->isDebugInstr()) {
    // Skip debug instructions.
    while (I->isDebugInstr()) {
      if (I == MBB.begin())
        return false;
      --I;
    }
    if (!I->isTerminator())
      break;

    ++NumTerminatorsSeen;

    if (I->getOpcode() == EZH::GOTO) {
      if (!I->getOperand(0).isMBB())
        return true;
      TrueBlock = I->getOperand(0).getMBB();
    } else if (I->isConditionalBranch()) {
      // Bail out if we encounter multiple conditional branches.
      if (!Condition.empty())
        return true;

      if (!I->getOperand(0).isMBB())
        return true;

      FalseBlock = TrueBlock;
      TrueBlock = I->getOperand(0).getMBB();
      Condition.push_back(MachineOperand::CreateImm(I->getOpcode()));
    } else {
      // Unrecognized terminator.
      return true;
    }

    // Cleanup code - to be run for unconditional branches and returns.
    if (I->getOpcode() == EZH::GOTO || I->isReturn()) {
      if (NumTerminatorsSeen > 1) {
        if (AllowModify) {
          MachineBasicBlock::iterator DI = std::next(I);
          while (DI != MBB.end()) {
            MachineInstr &InstToDelete = *DI;
            ++DI;
            InstToDelete.eraseFromParent();
          }
          NumTerminatorsSeen = 1;
        } else {
          return true;
        }
      }
      Condition.clear();
      FalseBlock = nullptr;
    }

    if (I == MBB.begin())
      return false;
    --I;
  }

  return false;
}

unsigned EZHInstrInfo::insertBranch(MachineBasicBlock &MBB,
                                    MachineBasicBlock *TrueBlock,
                                    MachineBasicBlock *FalseBlock,
                                    ArrayRef<MachineOperand> Condition,
                                    const DebugLoc &DL, int *BytesAdded) const {
  if (BytesAdded)
    *BytesAdded = 0;

  if (Condition.empty()) {
    BuildMI(&MBB, DL, get(EZH::GOTO)).addMBB(TrueBlock);
    if (BytesAdded)
      *BytesAdded += 4;
    return 1;
  }

  unsigned Opc = Condition[0].getImm();
  BuildMI(&MBB, DL, get(Opc)).addMBB(TrueBlock);
  if (BytesAdded)
    *BytesAdded += 4;

  if (FalseBlock) {
    BuildMI(&MBB, DL, get(EZH::GOTO)).addMBB(FalseBlock);
    if (BytesAdded)
      *BytesAdded += 4;
    return 2;
  }

  return 1;
}

unsigned EZHInstrInfo::removeBranch(MachineBasicBlock &MBB,
                                    int *BytesRemoved) const {
  if (BytesRemoved)
    *BytesRemoved = 0;

  MachineBasicBlock::iterator I = MBB.getLastNonDebugInstr();
  if (I == MBB.end())
    return 0;

  if (!I->isBranch())
    return 0;

  // Remove the last branch (unconditional or conditional).
  I->eraseFromParent();
  if (BytesRemoved)
    *BytesRemoved += 4;
  unsigned Count = 1;

  I = MBB.getLastNonDebugInstr();
  if (I == MBB.end()) {
    return Count;
  }
  if (!I->isBranch()) {
    return Count;
  }

  // Remove the joint conditional branch.
  I->eraseFromParent();
  if (BytesRemoved)
    *BytesRemoved += 4;
  return Count + 1;
}

bool EZHInstrInfo::reverseBranchCondition(
    SmallVectorImpl<MachineOperand> &Cond) const {
  assert(Cond.size() == 1 && "Invalid branch condition!");
  unsigned Opc = Cond[0].getImm();

  switch (Opc) {
  case EZH::GOTO_ze:
    Cond[0].setImm(EZH::GOTO_nz);
    return false;
  case EZH::GOTO_nz:
    Cond[0].setImm(EZH::GOTO_ze);
    return false;
  case EZH::GOTO_po:
    Cond[0].setImm(EZH::GOTO_ne);
    return false;
  case EZH::GOTO_ne:
    Cond[0].setImm(EZH::GOTO_po);
    return false;
  case EZH::GOTO_az:
    Cond[0].setImm(EZH::GOTO_zb);
    return false;
  case EZH::GOTO_zb:
    Cond[0].setImm(EZH::GOTO_az);
    return false;
  case EZH::GOTO_ca:
    Cond[0].setImm(EZH::GOTO_nc);
    return false;
  case EZH::GOTO_nc:
    Cond[0].setImm(EZH::GOTO_ca);
    return false;
  default:
    return true;
  }
}

#define CHECK_CC(OP)                                                           \
  case EZH::OP##___ze:                                                         \
  case EZH::OP##___nz:                                                         \
  case EZH::OP##___po:                                                         \
  case EZH::OP##___ne:                                                         \
  case EZH::OP##___az:                                                         \
  case EZH::OP##___zb:                                                         \
  case EZH::OP##___ca:                                                         \
  case EZH::OP##___nc:                                                         \
  case EZH::OP##___cz:                                                         \
  case EZH::OP##___spo:                                                        \
  case EZH::OP##___sne:                                                        \
  case EZH::OP##___nbs:                                                        \
  case EZH::OP##___nex:                                                        \
  case EZH::OP##___bs:                                                         \
  case EZH::OP##___ex:

bool EZHInstrInfo::isPredicated(const MachineInstr &MI) const {
  unsigned Opc = MI.getOpcode();
  switch (Opc) {
    CHECK_CC(ADDrr)
    CHECK_CC(ADDri)
    CHECK_CC(SUBrr)
    CHECK_CC(SUBri)
    CHECK_CC(ANDrr)
    CHECK_CC(ANDri)
    CHECK_CC(ORrr)
    CHECK_CC(ORri)
    CHECK_CC(XORrr)
    CHECK_CC(XORri)
    CHECK_CC(LSLi)
    CHECK_CC(LSRi)
    CHECK_CC(ASRi)
    CHECK_CC(RORi)
    return true;
  default:
    return false;
  }
}
#undef CHECK_CC

bool EZHInstrInfo::isPredicable(const MachineInstr &MI) const {
  unsigned Opc = MI.getOpcode();
  switch (Opc) {
  case EZH::ADDrr__:
  case EZH::ADDri__:
  case EZH::SUBrr__:
  case EZH::SUBri__:
  case EZH::ANDrr__:
  case EZH::ANDri__:
  case EZH::ORrr__:
  case EZH::ORri__:
  case EZH::XORrr__:
  case EZH::XORri__:
  case EZH::LSLi__:
  case EZH::LSRi__:
  case EZH::ASRi__:
  case EZH::RORi__:
  case EZH::MOVrr__:
  case EZH::MOVri__:
    return true;
  default:
    return false;
  }
}

bool EZHInstrInfo::canPredicatePredicatedInstr(const MachineInstr &MI) const {
  return false;
}

static LPCC::CondCode getCondCodeFromBranchOpc(unsigned Opc) {
  switch (Opc) {
  case EZH::GOTO_ze:
    return LPCC::ICC_ZE;
  case EZH::GOTO_nz:
    return LPCC::ICC_NZ;
  case EZH::GOTO_po:
    return LPCC::ICC_PO;
  case EZH::GOTO_ne:
    return LPCC::ICC_NE;
  case EZH::GOTO_az:
    return LPCC::ICC_AZ;
  case EZH::GOTO_zb:
    return LPCC::ICC_ZB;
  case EZH::GOTO_ca:
    return LPCC::ICC_CA;
  case EZH::GOTO_nc:
    return LPCC::ICC_NC;
  default:
    return LPCC::UNKNOWN;
  }
}

bool EZHInstrInfo::PredicateInstruction(MachineInstr &MI,
                                        ArrayRef<MachineOperand> Pred) const {
  assert(!Pred.empty() && "Empty predicate!");
  unsigned BranchOpc = Pred[0].getImm();
  LPCC::CondCode CC = getCondCodeFromBranchOpc(BranchOpc);
  if (CC == LPCC::UNKNOWN)
    return false;

  unsigned Opc = MI.getOpcode();
  unsigned NewOpc = getConditionalOpcode(Opc, CC);
  if (NewOpc == Opc)
    return false;

  MI.setDesc(get(NewOpc));

  if (MI.getNumOperands() > 0 && MI.getOperand(0).isReg() &&
      MI.getOperand(0).isDef()) {
    Register RdReg = MI.getOperand(0).getReg();
    MI.addOperand(
        MachineOperand::CreateReg(RdReg, /*isDef=*/false, /*isImp=*/true));
  }
  return true;
}

bool EZHInstrInfo::isProfitableToIfCvt(MachineBasicBlock &MBB,
                                       unsigned NumCycles,
                                       unsigned ExtraPredCycles,
                                       BranchProbability Probability) const {
  return true;
}

bool EZHInstrInfo::isProfitableToIfCvt(
    MachineBasicBlock &TMBB, unsigned NumTCycles, unsigned ExtraTCycles,
    MachineBasicBlock &FMBB, unsigned NumFCycles, unsigned ExtraFCycles,
    BranchProbability Probability) const {
  return true;
}

unsigned EZHInstrInfo::getInstSizeInBytes(const MachineInstr &MI) const {
  if (MI.isInlineAsm()) {
    const MachineFunction *MF = MI.getParent()->getParent();
    const MCAsmInfo &MAI = MF->getTarget().getMCAsmInfo();
    unsigned Size = getInlineAsmLength(MI.getOperand(0).getSymbolName(), MAI);
    return alignTo(Size, 4);
  }
  if (MI.getOpcode() == EZH::CONSTPOOL_ENTRY) {
    return MI.getOperand(2).getImm();
  }
  unsigned Size = MI.getDesc().getSize();
  if (Size > 0)
    return Size;
  if (MI.isMetaInstruction())
    return 0;
  return 4;
}

int EZHInstrInfo::getJumpTableIndex(const MachineInstr &MI) const {
  if (MI.getOpcode() == EZH::PseudoBR_JT) {
    const MachineOperand &MO = MI.getOperand(2);
    if (MO.isJTI()) {
      return MO.getIndex();
    } else if (MO.isImm()) {
      return MO.getImm();
    }
  }
  return -1;
}
