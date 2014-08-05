//===-- SIInstrInfo.cpp - SI Instruction Information  ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief SI Implementation of TargetInstrInfo.
//
//===----------------------------------------------------------------------===//


#include "SIInstrInfo.h"
#include "AMDGPUTargetMachine.h"
#include "SIDefines.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/MC/MCInstrDesc.h"

using namespace llvm;

SIInstrInfo::SIInstrInfo(const AMDGPUSubtarget &st)
  : AMDGPUInstrInfo(st),
    RI(st) { }

//===----------------------------------------------------------------------===//
// TargetInstrInfo callbacks
//===----------------------------------------------------------------------===//

void
SIInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator MI, DebugLoc DL,
                         unsigned DestReg, unsigned SrcReg,
                         bool KillSrc) const {

  // If we are trying to copy to or from SCC, there is a bug somewhere else in
  // the backend.  While it may be theoretically possible to do this, it should
  // never be necessary.
  assert(DestReg != AMDGPU::SCC && SrcReg != AMDGPU::SCC);

  static const int16_t Sub0_15[] = {
    AMDGPU::sub0, AMDGPU::sub1, AMDGPU::sub2, AMDGPU::sub3,
    AMDGPU::sub4, AMDGPU::sub5, AMDGPU::sub6, AMDGPU::sub7,
    AMDGPU::sub8, AMDGPU::sub9, AMDGPU::sub10, AMDGPU::sub11,
    AMDGPU::sub12, AMDGPU::sub13, AMDGPU::sub14, AMDGPU::sub15, 0
  };

  static const int16_t Sub0_7[] = {
    AMDGPU::sub0, AMDGPU::sub1, AMDGPU::sub2, AMDGPU::sub3,
    AMDGPU::sub4, AMDGPU::sub5, AMDGPU::sub6, AMDGPU::sub7, 0
  };

  static const int16_t Sub0_3[] = {
    AMDGPU::sub0, AMDGPU::sub1, AMDGPU::sub2, AMDGPU::sub3, 0
  };

  static const int16_t Sub0_2[] = {
    AMDGPU::sub0, AMDGPU::sub1, AMDGPU::sub2, 0
  };

  static const int16_t Sub0_1[] = {
    AMDGPU::sub0, AMDGPU::sub1, 0
  };

  unsigned Opcode;
  const int16_t *SubIndices;

  if (AMDGPU::M0 == DestReg) {
    // Check if M0 isn't already set to this value
    for (MachineBasicBlock::reverse_iterator E = MBB.rend(),
      I = MachineBasicBlock::reverse_iterator(MI); I != E; ++I) {

      if (!I->definesRegister(AMDGPU::M0))
        continue;

      unsigned Opc = I->getOpcode();
      if (Opc != TargetOpcode::COPY && Opc != AMDGPU::S_MOV_B32)
        break;

      if (!I->readsRegister(SrcReg))
        break;

      // The copy isn't necessary
      return;
    }
  }

  if (AMDGPU::SReg_32RegClass.contains(DestReg)) {
    assert(AMDGPU::SReg_32RegClass.contains(SrcReg));
    BuildMI(MBB, MI, DL, get(AMDGPU::S_MOV_B32), DestReg)
            .addReg(SrcReg, getKillRegState(KillSrc));
    return;

  } else if (AMDGPU::SReg_64RegClass.contains(DestReg)) {
    assert(AMDGPU::SReg_64RegClass.contains(SrcReg));
    BuildMI(MBB, MI, DL, get(AMDGPU::S_MOV_B64), DestReg)
            .addReg(SrcReg, getKillRegState(KillSrc));
    return;

  } else if (AMDGPU::SReg_128RegClass.contains(DestReg)) {
    assert(AMDGPU::SReg_128RegClass.contains(SrcReg));
    Opcode = AMDGPU::S_MOV_B32;
    SubIndices = Sub0_3;

  } else if (AMDGPU::SReg_256RegClass.contains(DestReg)) {
    assert(AMDGPU::SReg_256RegClass.contains(SrcReg));
    Opcode = AMDGPU::S_MOV_B32;
    SubIndices = Sub0_7;

  } else if (AMDGPU::SReg_512RegClass.contains(DestReg)) {
    assert(AMDGPU::SReg_512RegClass.contains(SrcReg));
    Opcode = AMDGPU::S_MOV_B32;
    SubIndices = Sub0_15;

  } else if (AMDGPU::VReg_32RegClass.contains(DestReg)) {
    assert(AMDGPU::VReg_32RegClass.contains(SrcReg) ||
           AMDGPU::SReg_32RegClass.contains(SrcReg));
    BuildMI(MBB, MI, DL, get(AMDGPU::V_MOV_B32_e32), DestReg)
            .addReg(SrcReg, getKillRegState(KillSrc));
    return;

  } else if (AMDGPU::VReg_64RegClass.contains(DestReg)) {
    assert(AMDGPU::VReg_64RegClass.contains(SrcReg) ||
           AMDGPU::SReg_64RegClass.contains(SrcReg));
    Opcode = AMDGPU::V_MOV_B32_e32;
    SubIndices = Sub0_1;

  } else if (AMDGPU::VReg_96RegClass.contains(DestReg)) {
    assert(AMDGPU::VReg_96RegClass.contains(SrcReg));
    Opcode = AMDGPU::V_MOV_B32_e32;
    SubIndices = Sub0_2;

  } else if (AMDGPU::VReg_128RegClass.contains(DestReg)) {
    assert(AMDGPU::VReg_128RegClass.contains(SrcReg) ||
           AMDGPU::SReg_128RegClass.contains(SrcReg));
    Opcode = AMDGPU::V_MOV_B32_e32;
    SubIndices = Sub0_3;

  } else if (AMDGPU::VReg_256RegClass.contains(DestReg)) {
    assert(AMDGPU::VReg_256RegClass.contains(SrcReg) ||
           AMDGPU::SReg_256RegClass.contains(SrcReg));
    Opcode = AMDGPU::V_MOV_B32_e32;
    SubIndices = Sub0_7;

  } else if (AMDGPU::VReg_512RegClass.contains(DestReg)) {
    assert(AMDGPU::VReg_512RegClass.contains(SrcReg) ||
           AMDGPU::SReg_512RegClass.contains(SrcReg));
    Opcode = AMDGPU::V_MOV_B32_e32;
    SubIndices = Sub0_15;

  } else {
    llvm_unreachable("Can't copy register!");
  }

  while (unsigned SubIdx = *SubIndices++) {
    MachineInstrBuilder Builder = BuildMI(MBB, MI, DL,
      get(Opcode), RI.getSubReg(DestReg, SubIdx));

    Builder.addReg(RI.getSubReg(SrcReg, SubIdx), getKillRegState(KillSrc));

    if (*SubIndices)
      Builder.addReg(DestReg, RegState::Define | RegState::Implicit);
  }
}

unsigned SIInstrInfo::commuteOpcode(unsigned Opcode) const {
  int NewOpc;

  // Try to map original to commuted opcode
  if ((NewOpc = AMDGPU::getCommuteRev(Opcode)) != -1)
    return NewOpc;

  // Try to map commuted to original opcode
  if ((NewOpc = AMDGPU::getCommuteOrig(Opcode)) != -1)
    return NewOpc;

  return Opcode;
}

void SIInstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator MI,
                                      unsigned SrcReg, bool isKill,
                                      int FrameIndex,
                                      const TargetRegisterClass *RC,
                                      const TargetRegisterInfo *TRI) const {
  MachineFunction *MF = MBB.getParent();
  SIMachineFunctionInfo *MFI = MF->getInfo<SIMachineFunctionInfo>();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  DebugLoc DL = MBB.findDebugLoc(MI);
  unsigned KillFlag = isKill ? RegState::Kill : 0;

  if (RI.hasVGPRs(RC)) {
    LLVMContext &Ctx = MF->getFunction()->getContext();
    Ctx.emitError("SIInstrInfo::storeRegToStackSlot - Can't spill VGPR!");
    BuildMI(MBB, MI, DL, get(AMDGPU::V_MOV_B32_e32), AMDGPU::VGPR0)
            .addReg(SrcReg);
  } else if (TRI->getCommonSubClass(RC, &AMDGPU::SGPR_32RegClass)) {
    unsigned Lane = MFI->SpillTracker.reserveLanes(MRI, MF);
    unsigned TgtReg = MFI->SpillTracker.LaneVGPR;

    BuildMI(MBB, MI, DL, get(AMDGPU::V_WRITELANE_B32), TgtReg)
            .addReg(SrcReg, KillFlag)
            .addImm(Lane);
    MFI->SpillTracker.addSpilledReg(FrameIndex, TgtReg, Lane);
  } else if (RI.isSGPRClass(RC)) {
    // We are only allowed to create one new instruction when spilling
    // registers, so we need to use pseudo instruction for vector
    // registers.
    //
    // Reserve a spot in the spill tracker for each sub-register of
    // the vector register.
    unsigned NumSubRegs = RC->getSize() / 4;
    unsigned FirstLane = MFI->SpillTracker.reserveLanes(MRI, MF, NumSubRegs);
    MFI->SpillTracker.addSpilledReg(FrameIndex, MFI->SpillTracker.LaneVGPR,
                                    FirstLane);

    unsigned Opcode;
    switch (RC->getSize() * 8) {
    case 64:  Opcode = AMDGPU::SI_SPILL_S64_SAVE;  break;
    case 128: Opcode = AMDGPU::SI_SPILL_S128_SAVE; break;
    case 256: Opcode = AMDGPU::SI_SPILL_S256_SAVE; break;
    case 512: Opcode = AMDGPU::SI_SPILL_S512_SAVE; break;
    default: llvm_unreachable("Cannot spill register class");
    }

    BuildMI(MBB, MI, DL, get(Opcode), MFI->SpillTracker.LaneVGPR)
            .addReg(SrcReg)
            .addImm(FrameIndex);
  } else {
    llvm_unreachable("VGPR spilling not supported");
  }
}

void SIInstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator MI,
                                       unsigned DestReg, int FrameIndex,
                                       const TargetRegisterClass *RC,
                                       const TargetRegisterInfo *TRI) const {
  MachineFunction *MF = MBB.getParent();
  SIMachineFunctionInfo *MFI = MF->getInfo<SIMachineFunctionInfo>();
  DebugLoc DL = MBB.findDebugLoc(MI);

  if (RI.hasVGPRs(RC)) {
    LLVMContext &Ctx = MF->getFunction()->getContext();
    Ctx.emitError("SIInstrInfo::loadRegToStackSlot - Can't retrieve spilled VGPR!");
    BuildMI(MBB, MI, DL, get(AMDGPU::V_MOV_B32_e32), DestReg)
            .addImm(0);
  } else if (RI.isSGPRClass(RC)){
    unsigned Opcode;
    switch(RC->getSize() * 8) {
    case 32:  Opcode = AMDGPU::SI_SPILL_S32_RESTORE; break;
    case 64:  Opcode = AMDGPU::SI_SPILL_S64_RESTORE;  break;
    case 128: Opcode = AMDGPU::SI_SPILL_S128_RESTORE; break;
    case 256: Opcode = AMDGPU::SI_SPILL_S256_RESTORE; break;
    case 512: Opcode = AMDGPU::SI_SPILL_S512_RESTORE; break;
    default: llvm_unreachable("Cannot spill register class");
    }

    SIMachineFunctionInfo::SpilledReg Spill =
        MFI->SpillTracker.getSpilledReg(FrameIndex);

    BuildMI(MBB, MI, DL, get(Opcode), DestReg)
            .addReg(Spill.VGPR)
            .addImm(FrameIndex);
  } else {
    llvm_unreachable("VGPR spilling not supported");
  }
}

static unsigned getNumSubRegsForSpillOp(unsigned Op) {

  switch (Op) {
  case AMDGPU::SI_SPILL_S512_SAVE:
  case AMDGPU::SI_SPILL_S512_RESTORE:
    return 16;
  case AMDGPU::SI_SPILL_S256_SAVE:
  case AMDGPU::SI_SPILL_S256_RESTORE:
    return 8;
  case AMDGPU::SI_SPILL_S128_SAVE:
  case AMDGPU::SI_SPILL_S128_RESTORE:
    return 4;
  case AMDGPU::SI_SPILL_S64_SAVE:
  case AMDGPU::SI_SPILL_S64_RESTORE:
    return 2;
  case AMDGPU::SI_SPILL_S32_RESTORE:
    return 1;
  default: llvm_unreachable("Invalid spill opcode");
  }
}

void SIInstrInfo::insertNOPs(MachineBasicBlock::iterator MI,
                             int Count) const {
  while (Count > 0) {
    int Arg;
    if (Count >= 8)
      Arg = 7;
    else
      Arg = Count - 1;
    Count -= 8;
    BuildMI(*MI->getParent(), MI, MI->getDebugLoc(), get(AMDGPU::S_NOP))
            .addImm(Arg);
  }
}

bool SIInstrInfo::expandPostRAPseudo(MachineBasicBlock::iterator MI) const {
  SIMachineFunctionInfo *MFI =
      MI->getParent()->getParent()->getInfo<SIMachineFunctionInfo>();
  MachineBasicBlock &MBB = *MI->getParent();
  DebugLoc DL = MBB.findDebugLoc(MI);
  switch (MI->getOpcode()) {
  default: return AMDGPUInstrInfo::expandPostRAPseudo(MI);

  // SGPR register spill
  case AMDGPU::SI_SPILL_S512_SAVE:
  case AMDGPU::SI_SPILL_S256_SAVE:
  case AMDGPU::SI_SPILL_S128_SAVE:
  case AMDGPU::SI_SPILL_S64_SAVE: {
    unsigned NumSubRegs = getNumSubRegsForSpillOp(MI->getOpcode());
    unsigned FrameIndex = MI->getOperand(2).getImm();

    for (unsigned i = 0, e = NumSubRegs; i < e; ++i) {
      SIMachineFunctionInfo::SpilledReg Spill;
      unsigned SubReg = RI.getPhysRegSubReg(MI->getOperand(1).getReg(),
                                            &AMDGPU::SGPR_32RegClass, i);
      Spill = MFI->SpillTracker.getSpilledReg(FrameIndex);

      BuildMI(MBB, MI, DL, get(AMDGPU::V_WRITELANE_B32),
              MI->getOperand(0).getReg())
              .addReg(SubReg)
              .addImm(Spill.Lane + i);
    }
    MI->eraseFromParent();
    break;
  }

  // SGPR register restore
  case AMDGPU::SI_SPILL_S512_RESTORE:
  case AMDGPU::SI_SPILL_S256_RESTORE:
  case AMDGPU::SI_SPILL_S128_RESTORE:
  case AMDGPU::SI_SPILL_S64_RESTORE:
  case AMDGPU::SI_SPILL_S32_RESTORE: {
    unsigned NumSubRegs = getNumSubRegsForSpillOp(MI->getOpcode());

    for (unsigned i = 0, e = NumSubRegs; i < e; ++i) {
      SIMachineFunctionInfo::SpilledReg Spill;
      unsigned FrameIndex = MI->getOperand(2).getImm();
      unsigned SubReg = RI.getPhysRegSubReg(MI->getOperand(0).getReg(),
                                   &AMDGPU::SGPR_32RegClass, i);
      Spill = MFI->SpillTracker.getSpilledReg(FrameIndex);

      BuildMI(MBB, MI, DL, get(AMDGPU::V_READLANE_B32), SubReg)
              .addReg(MI->getOperand(1).getReg())
              .addImm(Spill.Lane + i);
    }
    insertNOPs(MI, 3);
    MI->eraseFromParent();
    break;
  }
  case AMDGPU::SI_CONSTDATA_PTR: {
    unsigned Reg = MI->getOperand(0).getReg();
    unsigned RegLo = RI.getSubReg(Reg, AMDGPU::sub0);
    unsigned RegHi = RI.getSubReg(Reg, AMDGPU::sub1);

    BuildMI(MBB, MI, DL, get(AMDGPU::S_GETPC_B64), Reg);

    // Add 32-bit offset from this instruction to the start of the constant data.
    BuildMI(MBB, MI, DL, get(AMDGPU::S_ADD_I32), RegLo)
            .addReg(RegLo)
            .addTargetIndex(AMDGPU::TI_CONSTDATA_START)
            .addReg(AMDGPU::SCC, RegState::Define | RegState::Implicit);
    BuildMI(MBB, MI, DL, get(AMDGPU::S_ADDC_U32), RegHi)
            .addReg(RegHi)
            .addImm(0)
            .addReg(AMDGPU::SCC, RegState::Define | RegState::Implicit)
            .addReg(AMDGPU::SCC, RegState::Implicit);
    MI->eraseFromParent();
    break;
  }
  }
  return true;
}

MachineInstr *SIInstrInfo::commuteInstruction(MachineInstr *MI,
                                              bool NewMI) const {

  MachineRegisterInfo &MRI = MI->getParent()->getParent()->getRegInfo();
  if (MI->getNumOperands() < 3 || !MI->getOperand(1).isReg())
    return nullptr;

  // Cannot commute VOP2 if src0 is SGPR.
  if (isVOP2(MI->getOpcode()) && MI->getOperand(1).isReg() &&
      RI.isSGPRClass(MRI.getRegClass(MI->getOperand(1).getReg())))
   return nullptr;

  if (!MI->getOperand(2).isReg()) {
    // XXX: Commute instructions with FPImm operands
    if (NewMI || MI->getOperand(2).isFPImm() ||
       (!isVOP2(MI->getOpcode()) && !isVOP3(MI->getOpcode()))) {
      return nullptr;
    }

    // XXX: Commute VOP3 instructions with abs and neg set.
    if (isVOP3(MI->getOpcode()) &&
        (MI->getOperand(AMDGPU::getNamedOperandIdx(MI->getOpcode(),
                        AMDGPU::OpName::abs)).getImm() ||
         MI->getOperand(AMDGPU::getNamedOperandIdx(MI->getOpcode(),
                        AMDGPU::OpName::neg)).getImm()))
      return nullptr;

    unsigned Reg = MI->getOperand(1).getReg();
    unsigned SubReg = MI->getOperand(1).getSubReg();
    MI->getOperand(1).ChangeToImmediate(MI->getOperand(2).getImm());
    MI->getOperand(2).ChangeToRegister(Reg, false);
    MI->getOperand(2).setSubReg(SubReg);
  } else {
    MI = TargetInstrInfo::commuteInstruction(MI, NewMI);
  }

  if (MI)
    MI->setDesc(get(commuteOpcode(MI->getOpcode())));

  return MI;
}

MachineInstr *SIInstrInfo::buildMovInstr(MachineBasicBlock *MBB,
                                         MachineBasicBlock::iterator I,
                                         unsigned DstReg,
                                         unsigned SrcReg) const {
  return BuildMI(*MBB, I, MBB->findDebugLoc(I), get(AMDGPU::V_MOV_B32_e32),
                 DstReg) .addReg(SrcReg);
}

bool SIInstrInfo::isMov(unsigned Opcode) const {
  switch(Opcode) {
  default: return false;
  case AMDGPU::S_MOV_B32:
  case AMDGPU::S_MOV_B64:
  case AMDGPU::V_MOV_B32_e32:
  case AMDGPU::V_MOV_B32_e64:
    return true;
  }
}

bool
SIInstrInfo::isSafeToMoveRegClassDefs(const TargetRegisterClass *RC) const {
  return RC != &AMDGPU::EXECRegRegClass;
}

bool
SIInstrInfo::isTriviallyReMaterializable(const MachineInstr *MI,
                                         AliasAnalysis *AA) const {
  switch(MI->getOpcode()) {
  default: return AMDGPUInstrInfo::isTriviallyReMaterializable(MI, AA);
  case AMDGPU::S_MOV_B32:
  case AMDGPU::S_MOV_B64:
  case AMDGPU::V_MOV_B32_e32:
    return MI->getOperand(1).isImm();
  }
}

namespace llvm {
namespace AMDGPU {
// Helper function generated by tablegen.  We are wrapping this with
// an SIInstrInfo function that reutrns bool rather than int.
int isDS(uint16_t Opcode);
}
}

bool SIInstrInfo::isDS(uint16_t Opcode) const {
  return ::AMDGPU::isDS(Opcode) != -1;
}

int SIInstrInfo::isMIMG(uint16_t Opcode) const {
  return get(Opcode).TSFlags & SIInstrFlags::MIMG;
}

int SIInstrInfo::isSMRD(uint16_t Opcode) const {
  return get(Opcode).TSFlags & SIInstrFlags::SMRD;
}

bool SIInstrInfo::isVOP1(uint16_t Opcode) const {
  return get(Opcode).TSFlags & SIInstrFlags::VOP1;
}

bool SIInstrInfo::isVOP2(uint16_t Opcode) const {
  return get(Opcode).TSFlags & SIInstrFlags::VOP2;
}

bool SIInstrInfo::isVOP3(uint16_t Opcode) const {
  return get(Opcode).TSFlags & SIInstrFlags::VOP3;
}

bool SIInstrInfo::isVOPC(uint16_t Opcode) const {
  return get(Opcode).TSFlags & SIInstrFlags::VOPC;
}

bool SIInstrInfo::isSALUInstr(const MachineInstr &MI) const {
  return get(MI.getOpcode()).TSFlags & SIInstrFlags::SALU;
}

bool SIInstrInfo::isInlineConstant(const APInt &Imm) const {
  int32_t Val = Imm.getSExtValue();
  if (Val >= -16 && Val <= 64)
    return true;

  // The actual type of the operand does not seem to matter as long
  // as the bits match one of the inline immediate values.  For example:
  //
  // -nan has the hexadecimal encoding of 0xfffffffe which is -2 in decimal,
  // so it is a legal inline immediate.
  //
  // 1065353216 has the hexadecimal encoding 0x3f800000 which is 1.0f in
  // floating-point, so it is a legal inline immediate.

  return (APInt::floatToBits(0.0f) == Imm) ||
         (APInt::floatToBits(1.0f) == Imm) ||
         (APInt::floatToBits(-1.0f) == Imm) ||
         (APInt::floatToBits(0.5f) == Imm) ||
         (APInt::floatToBits(-0.5f) == Imm) ||
         (APInt::floatToBits(2.0f) == Imm) ||
         (APInt::floatToBits(-2.0f) == Imm) ||
         (APInt::floatToBits(4.0f) == Imm) ||
         (APInt::floatToBits(-4.0f) == Imm);
}

bool SIInstrInfo::isInlineConstant(const MachineOperand &MO) const {
  if (MO.isImm())
    return isInlineConstant(APInt(32, MO.getImm(), true));

  if (MO.isFPImm()) {
    APFloat FpImm = MO.getFPImm()->getValueAPF();
    return isInlineConstant(FpImm.bitcastToAPInt());
  }

  return false;
}

bool SIInstrInfo::isLiteralConstant(const MachineOperand &MO) const {
  return (MO.isImm() || MO.isFPImm()) && !isInlineConstant(MO);
}

static bool compareMachineOp(const MachineOperand &Op0,
                             const MachineOperand &Op1) {
  if (Op0.getType() != Op1.getType())
    return false;

  switch (Op0.getType()) {
  case MachineOperand::MO_Register:
    return Op0.getReg() == Op1.getReg();
  case MachineOperand::MO_Immediate:
    return Op0.getImm() == Op1.getImm();
  case MachineOperand::MO_FPImmediate:
    return Op0.getFPImm() == Op1.getFPImm();
  default:
    llvm_unreachable("Didn't expect to be comparing these operand types");
  }
}

bool SIInstrInfo::isImmOperandLegal(const MachineInstr *MI, unsigned OpNo,
                                 const MachineOperand &MO) const {
  const MCOperandInfo &OpInfo = get(MI->getOpcode()).OpInfo[OpNo];

  assert(MO.isImm() || MO.isFPImm());

  if (OpInfo.OperandType == MCOI::OPERAND_IMMEDIATE)
    return true;

  if (OpInfo.RegClass < 0)
    return false;

  return RI.regClassCanUseImmediate(OpInfo.RegClass);
}

bool SIInstrInfo::hasVALU32BitEncoding(unsigned Opcode) const {
  return AMDGPU::getVOPe32(Opcode) != -1;
}

bool SIInstrInfo::verifyInstruction(const MachineInstr *MI,
                                    StringRef &ErrInfo) const {
  uint16_t Opcode = MI->getOpcode();
  int Src0Idx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::src0);
  int Src1Idx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::src1);
  int Src2Idx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::src2);

  // Make sure the number of operands is correct.
  const MCInstrDesc &Desc = get(Opcode);
  if (!Desc.isVariadic() &&
      Desc.getNumOperands() != MI->getNumExplicitOperands()) {
     ErrInfo = "Instruction has wrong number of operands.";
     return false;
  }

  // Make sure the register classes are correct
  for (unsigned i = 0, e = Desc.getNumOperands(); i != e; ++i) {
    switch (Desc.OpInfo[i].OperandType) {
    case MCOI::OPERAND_REGISTER: {
      int RegClass = Desc.OpInfo[i].RegClass;
      if (!RI.regClassCanUseImmediate(RegClass) &&
          (MI->getOperand(i).isImm() || MI->getOperand(i).isFPImm())) {
        ErrInfo = "Expected register, but got immediate";
        return false;
      }
    }
      break;
    case MCOI::OPERAND_IMMEDIATE:
      // Check if this operand is an immediate.
      // FrameIndex operands will be replaced by immediates, so they are
      // allowed.
      if (!MI->getOperand(i).isImm() && !MI->getOperand(i).isFPImm() &&
          !MI->getOperand(i).isFI()) {
        ErrInfo = "Expected immediate, but got non-immediate";
        return false;
      }
      // Fall-through
    default:
      continue;
    }

    if (!MI->getOperand(i).isReg())
      continue;

    int RegClass = Desc.OpInfo[i].RegClass;
    if (RegClass != -1) {
      unsigned Reg = MI->getOperand(i).getReg();
      if (TargetRegisterInfo::isVirtualRegister(Reg))
        continue;

      const TargetRegisterClass *RC = RI.getRegClass(RegClass);
      if (!RC->contains(Reg)) {
        ErrInfo = "Operand has incorrect register class.";
        return false;
      }
    }
  }


  // Verify VOP*
  if (isVOP1(Opcode) || isVOP2(Opcode) || isVOP3(Opcode) || isVOPC(Opcode)) {
    unsigned ConstantBusCount = 0;
    unsigned SGPRUsed = AMDGPU::NoRegister;
    for (int i = 0, e = MI->getNumOperands(); i != e; ++i) {
      const MachineOperand &MO = MI->getOperand(i);
      if (MO.isReg() && MO.isUse() &&
          !TargetRegisterInfo::isVirtualRegister(MO.getReg())) {

        // EXEC register uses the constant bus.
        if (!MO.isImplicit() && MO.getReg() == AMDGPU::EXEC)
          ++ConstantBusCount;

        // SGPRs use the constant bus
        if (MO.getReg() == AMDGPU::M0 || MO.getReg() == AMDGPU::VCC ||
            (!MO.isImplicit() &&
            (AMDGPU::SGPR_32RegClass.contains(MO.getReg()) ||
            AMDGPU::SGPR_64RegClass.contains(MO.getReg())))) {
          if (SGPRUsed != MO.getReg()) {
            ++ConstantBusCount;
            SGPRUsed = MO.getReg();
          }
        }
      }
      // Literal constants use the constant bus.
      if (isLiteralConstant(MO))
        ++ConstantBusCount;
    }
    if (ConstantBusCount > 1) {
      ErrInfo = "VOP* instruction uses the constant bus more than once";
      return false;
    }
  }

  // Verify SRC1 for VOP2 and VOPC
  if (Src1Idx != -1 && (isVOP2(Opcode) || isVOPC(Opcode))) {
    const MachineOperand &Src1 = MI->getOperand(Src1Idx);
    if (Src1.isImm() || Src1.isFPImm()) {
      ErrInfo = "VOP[2C] src1 cannot be an immediate.";
      return false;
    }
  }

  // Verify VOP3
  if (isVOP3(Opcode)) {
    if (Src0Idx != -1 && isLiteralConstant(MI->getOperand(Src0Idx))) {
      ErrInfo = "VOP3 src0 cannot be a literal constant.";
      return false;
    }
    if (Src1Idx != -1 && isLiteralConstant(MI->getOperand(Src1Idx))) {
      ErrInfo = "VOP3 src1 cannot be a literal constant.";
      return false;
    }
    if (Src2Idx != -1 && isLiteralConstant(MI->getOperand(Src2Idx))) {
      ErrInfo = "VOP3 src2 cannot be a literal constant.";
      return false;
    }
  }

  // Verify misc. restrictions on specific instructions.
  if (Desc.getOpcode() == AMDGPU::V_DIV_SCALE_F32 ||
      Desc.getOpcode() == AMDGPU::V_DIV_SCALE_F64) {
    MI->dump();

    const MachineOperand &Src0 = MI->getOperand(2);
    const MachineOperand &Src1 = MI->getOperand(3);
    const MachineOperand &Src2 = MI->getOperand(4);
    if (Src0.isReg() && Src1.isReg() && Src2.isReg()) {
      if (!compareMachineOp(Src0, Src1) &&
          !compareMachineOp(Src0, Src2)) {
        ErrInfo = "v_div_scale_{f32|f64} require src0 = src1 or src2";
        return false;
      }
    }
  }

  return true;
}

unsigned SIInstrInfo::getVALUOp(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  default: return AMDGPU::INSTRUCTION_LIST_END;
  case AMDGPU::REG_SEQUENCE: return AMDGPU::REG_SEQUENCE;
  case AMDGPU::COPY: return AMDGPU::COPY;
  case AMDGPU::PHI: return AMDGPU::PHI;
  case AMDGPU::INSERT_SUBREG: return AMDGPU::INSERT_SUBREG;
  case AMDGPU::S_MOV_B32:
    return MI.getOperand(1).isReg() ?
           AMDGPU::COPY : AMDGPU::V_MOV_B32_e32;
  case AMDGPU::S_ADD_I32: return AMDGPU::V_ADD_I32_e32;
  case AMDGPU::S_ADDC_U32: return AMDGPU::V_ADDC_U32_e32;
  case AMDGPU::S_SUB_I32: return AMDGPU::V_SUB_I32_e32;
  case AMDGPU::S_SUBB_U32: return AMDGPU::V_SUBB_U32_e32;
  case AMDGPU::S_AND_B32: return AMDGPU::V_AND_B32_e32;
  case AMDGPU::S_OR_B32: return AMDGPU::V_OR_B32_e32;
  case AMDGPU::S_XOR_B32: return AMDGPU::V_XOR_B32_e32;
  case AMDGPU::S_MIN_I32: return AMDGPU::V_MIN_I32_e32;
  case AMDGPU::S_MIN_U32: return AMDGPU::V_MIN_U32_e32;
  case AMDGPU::S_MAX_I32: return AMDGPU::V_MAX_I32_e32;
  case AMDGPU::S_MAX_U32: return AMDGPU::V_MAX_U32_e32;
  case AMDGPU::S_ASHR_I32: return AMDGPU::V_ASHR_I32_e32;
  case AMDGPU::S_ASHR_I64: return AMDGPU::V_ASHR_I64;
  case AMDGPU::S_LSHL_B32: return AMDGPU::V_LSHL_B32_e32;
  case AMDGPU::S_LSHL_B64: return AMDGPU::V_LSHL_B64;
  case AMDGPU::S_LSHR_B32: return AMDGPU::V_LSHR_B32_e32;
  case AMDGPU::S_LSHR_B64: return AMDGPU::V_LSHR_B64;
  case AMDGPU::S_SEXT_I32_I8: return AMDGPU::V_BFE_I32;
  case AMDGPU::S_SEXT_I32_I16: return AMDGPU::V_BFE_I32;
  case AMDGPU::S_BFE_U32: return AMDGPU::V_BFE_U32;
  case AMDGPU::S_BFE_I32: return AMDGPU::V_BFE_I32;
  case AMDGPU::S_BREV_B32: return AMDGPU::V_BFREV_B32_e32;
  case AMDGPU::S_NOT_B32: return AMDGPU::V_NOT_B32_e32;
  case AMDGPU::S_NOT_B64: return AMDGPU::V_NOT_B32_e32;
  case AMDGPU::S_CMP_EQ_I32: return AMDGPU::V_CMP_EQ_I32_e32;
  case AMDGPU::S_CMP_LG_I32: return AMDGPU::V_CMP_NE_I32_e32;
  case AMDGPU::S_CMP_GT_I32: return AMDGPU::V_CMP_GT_I32_e32;
  case AMDGPU::S_CMP_GE_I32: return AMDGPU::V_CMP_GE_I32_e32;
  case AMDGPU::S_CMP_LT_I32: return AMDGPU::V_CMP_LT_I32_e32;
  case AMDGPU::S_CMP_LE_I32: return AMDGPU::V_CMP_LE_I32_e32;
  case AMDGPU::S_LOAD_DWORD_IMM:
  case AMDGPU::S_LOAD_DWORD_SGPR: return AMDGPU::BUFFER_LOAD_DWORD_ADDR64;
  case AMDGPU::S_LOAD_DWORDX2_IMM:
  case AMDGPU::S_LOAD_DWORDX2_SGPR: return AMDGPU::BUFFER_LOAD_DWORDX2_ADDR64;
  case AMDGPU::S_LOAD_DWORDX4_IMM:
  case AMDGPU::S_LOAD_DWORDX4_SGPR: return AMDGPU::BUFFER_LOAD_DWORDX4_ADDR64;
  case AMDGPU::S_BCNT1_I32_B32: return AMDGPU::V_BCNT_U32_B32_e32;
  case AMDGPU::S_FF1_I32_B32: return AMDGPU::V_FFBL_B32_e32;
  case AMDGPU::S_FLBIT_I32_B32: return AMDGPU::V_FFBH_U32_e32;
  }
}

bool SIInstrInfo::isSALUOpSupportedOnVALU(const MachineInstr &MI) const {
  return getVALUOp(MI) != AMDGPU::INSTRUCTION_LIST_END;
}

const TargetRegisterClass *SIInstrInfo::getOpRegClass(const MachineInstr &MI,
                                                      unsigned OpNo) const {
  const MachineRegisterInfo &MRI = MI.getParent()->getParent()->getRegInfo();
  const MCInstrDesc &Desc = get(MI.getOpcode());
  if (MI.isVariadic() || OpNo >= Desc.getNumOperands() ||
      Desc.OpInfo[OpNo].RegClass == -1)
    return MRI.getRegClass(MI.getOperand(OpNo).getReg());

  unsigned RCID = Desc.OpInfo[OpNo].RegClass;
  return RI.getRegClass(RCID);
}

bool SIInstrInfo::canReadVGPR(const MachineInstr &MI, unsigned OpNo) const {
  switch (MI.getOpcode()) {
  case AMDGPU::COPY:
  case AMDGPU::REG_SEQUENCE:
  case AMDGPU::PHI:
  case AMDGPU::INSERT_SUBREG:
    return RI.hasVGPRs(getOpRegClass(MI, 0));
  default:
    return RI.hasVGPRs(getOpRegClass(MI, OpNo));
  }
}

void SIInstrInfo::legalizeOpWithMove(MachineInstr *MI, unsigned OpIdx) const {
  MachineBasicBlock::iterator I = MI;
  MachineOperand &MO = MI->getOperand(OpIdx);
  MachineRegisterInfo &MRI = MI->getParent()->getParent()->getRegInfo();
  unsigned RCID = get(MI->getOpcode()).OpInfo[OpIdx].RegClass;
  const TargetRegisterClass *RC = RI.getRegClass(RCID);
  unsigned Opcode = AMDGPU::V_MOV_B32_e32;
  if (MO.isReg()) {
    Opcode = AMDGPU::COPY;
  } else if (RI.isSGPRClass(RC)) {
    Opcode = AMDGPU::S_MOV_B32;
  }

  const TargetRegisterClass *VRC = RI.getEquivalentVGPRClass(RC);
  unsigned Reg = MRI.createVirtualRegister(VRC);
  BuildMI(*MI->getParent(), I, MI->getParent()->findDebugLoc(I), get(Opcode),
          Reg).addOperand(MO);
  MO.ChangeToRegister(Reg, false);
}

unsigned SIInstrInfo::buildExtractSubReg(MachineBasicBlock::iterator MI,
                                         MachineRegisterInfo &MRI,
                                         MachineOperand &SuperReg,
                                         const TargetRegisterClass *SuperRC,
                                         unsigned SubIdx,
                                         const TargetRegisterClass *SubRC)
                                         const {
  assert(SuperReg.isReg());

  unsigned NewSuperReg = MRI.createVirtualRegister(SuperRC);
  unsigned SubReg = MRI.createVirtualRegister(SubRC);

  // Just in case the super register is itself a sub-register, copy it to a new
  // value so we don't need to worry about merging its subreg index with the
  // SubIdx passed to this function. The register coalescer should be able to
  // eliminate this extra copy.
  BuildMI(*MI->getParent(), MI, MI->getDebugLoc(), get(TargetOpcode::COPY),
          NewSuperReg)
          .addOperand(SuperReg);

  BuildMI(*MI->getParent(), MI, MI->getDebugLoc(), get(TargetOpcode::COPY),
          SubReg)
          .addReg(NewSuperReg, 0, SubIdx);
  return SubReg;
}

MachineOperand SIInstrInfo::buildExtractSubRegOrImm(
  MachineBasicBlock::iterator MII,
  MachineRegisterInfo &MRI,
  MachineOperand &Op,
  const TargetRegisterClass *SuperRC,
  unsigned SubIdx,
  const TargetRegisterClass *SubRC) const {
  if (Op.isImm()) {
    // XXX - Is there a better way to do this?
    if (SubIdx == AMDGPU::sub0)
      return MachineOperand::CreateImm(Op.getImm() & 0xFFFFFFFF);
    if (SubIdx == AMDGPU::sub1)
      return MachineOperand::CreateImm(Op.getImm() >> 32);

    llvm_unreachable("Unhandled register index for immediate");
  }

  unsigned SubReg = buildExtractSubReg(MII, MRI, Op, SuperRC,
                                       SubIdx, SubRC);
  return MachineOperand::CreateReg(SubReg, false);
}

unsigned SIInstrInfo::split64BitImm(SmallVectorImpl<MachineInstr *> &Worklist,
                                    MachineBasicBlock::iterator MI,
                                    MachineRegisterInfo &MRI,
                                    const TargetRegisterClass *RC,
                                    const MachineOperand &Op) const {
  MachineBasicBlock *MBB = MI->getParent();
  DebugLoc DL = MI->getDebugLoc();
  unsigned LoDst = MRI.createVirtualRegister(&AMDGPU::SGPR_32RegClass);
  unsigned HiDst = MRI.createVirtualRegister(&AMDGPU::SGPR_32RegClass);
  unsigned Dst = MRI.createVirtualRegister(RC);

  MachineInstr *Lo = BuildMI(*MBB, MI, DL, get(AMDGPU::S_MOV_B32),
                             LoDst)
    .addImm(Op.getImm() & 0xFFFFFFFF);
  MachineInstr *Hi = BuildMI(*MBB, MI, DL, get(AMDGPU::S_MOV_B32),
                             HiDst)
    .addImm(Op.getImm() >> 32);

  BuildMI(*MBB, MI, DL, get(TargetOpcode::REG_SEQUENCE), Dst)
    .addReg(LoDst)
    .addImm(AMDGPU::sub0)
    .addReg(HiDst)
    .addImm(AMDGPU::sub1);

  Worklist.push_back(Lo);
  Worklist.push_back(Hi);

  return Dst;
}

void SIInstrInfo::legalizeOperands(MachineInstr *MI) const {
  MachineRegisterInfo &MRI = MI->getParent()->getParent()->getRegInfo();
  int Src0Idx = AMDGPU::getNamedOperandIdx(MI->getOpcode(),
                                           AMDGPU::OpName::src0);
  int Src1Idx = AMDGPU::getNamedOperandIdx(MI->getOpcode(),
                                           AMDGPU::OpName::src1);
  int Src2Idx = AMDGPU::getNamedOperandIdx(MI->getOpcode(),
                                           AMDGPU::OpName::src2);

  // Legalize VOP2
  if (isVOP2(MI->getOpcode()) && Src1Idx != -1) {
    MachineOperand &Src0 = MI->getOperand(Src0Idx);
    MachineOperand &Src1 = MI->getOperand(Src1Idx);

    // If the instruction implicitly reads VCC, we can't have any SGPR operands,
    // so move any.
    bool ReadsVCC = MI->readsRegister(AMDGPU::VCC, &RI);
    if (ReadsVCC && Src0.isReg() &&
        RI.isSGPRClass(MRI.getRegClass(Src0.getReg()))) {
      legalizeOpWithMove(MI, Src0Idx);
      return;
    }

    if (ReadsVCC && Src1.isReg() &&
        RI.isSGPRClass(MRI.getRegClass(Src1.getReg()))) {
      legalizeOpWithMove(MI, Src1Idx);
      return;
    }

    // Legalize VOP2 instructions where src1 is not a VGPR. An SGPR input must
    // be the first operand, and there can only be one.
    if (Src1.isImm() || Src1.isFPImm() ||
        (Src1.isReg() && RI.isSGPRClass(MRI.getRegClass(Src1.getReg())))) {
      if (MI->isCommutable()) {
        if (commuteInstruction(MI))
          return;
      }
      legalizeOpWithMove(MI, Src1Idx);
    }
  }

  // XXX - Do any VOP3 instructions read VCC?
  // Legalize VOP3
  if (isVOP3(MI->getOpcode())) {
    int VOP3Idx[3] = {Src0Idx, Src1Idx, Src2Idx};
    unsigned SGPRReg = AMDGPU::NoRegister;
    for (unsigned i = 0; i < 3; ++i) {
      int Idx = VOP3Idx[i];
      if (Idx == -1)
        continue;
      MachineOperand &MO = MI->getOperand(Idx);

      if (MO.isReg()) {
        if (!RI.isSGPRClass(MRI.getRegClass(MO.getReg())))
          continue; // VGPRs are legal

        assert(MO.getReg() != AMDGPU::SCC && "SCC operand to VOP3 instruction");

        if (SGPRReg == AMDGPU::NoRegister || SGPRReg == MO.getReg()) {
          SGPRReg = MO.getReg();
          // We can use one SGPR in each VOP3 instruction.
          continue;
        }
      } else if (!isLiteralConstant(MO)) {
        // If it is not a register and not a literal constant, then it must be
        // an inline constant which is always legal.
        continue;
      }
      // If we make it this far, then the operand is not legal and we must
      // legalize it.
      legalizeOpWithMove(MI, Idx);
    }
  }

  // Legalize REG_SEQUENCE and PHI
  // The register class of the operands much be the same type as the register
  // class of the output.
  if (MI->getOpcode() == AMDGPU::REG_SEQUENCE ||
      MI->getOpcode() == AMDGPU::PHI) {
    const TargetRegisterClass *RC = nullptr, *SRC = nullptr, *VRC = nullptr;
    for (unsigned i = 1, e = MI->getNumOperands(); i != e; i+=2) {
      if (!MI->getOperand(i).isReg() ||
          !TargetRegisterInfo::isVirtualRegister(MI->getOperand(i).getReg()))
        continue;
      const TargetRegisterClass *OpRC =
              MRI.getRegClass(MI->getOperand(i).getReg());
      if (RI.hasVGPRs(OpRC)) {
        VRC = OpRC;
      } else {
        SRC = OpRC;
      }
    }

    // If any of the operands are VGPR registers, then they all most be
    // otherwise we will create illegal VGPR->SGPR copies when legalizing
    // them.
    if (VRC || !RI.isSGPRClass(getOpRegClass(*MI, 0))) {
      if (!VRC) {
        assert(SRC);
        VRC = RI.getEquivalentVGPRClass(SRC);
      }
      RC = VRC;
    } else {
      RC = SRC;
    }

    // Update all the operands so they have the same type.
    for (unsigned i = 1, e = MI->getNumOperands(); i != e; i+=2) {
      if (!MI->getOperand(i).isReg() ||
          !TargetRegisterInfo::isVirtualRegister(MI->getOperand(i).getReg()))
        continue;
      unsigned DstReg = MRI.createVirtualRegister(RC);
      MachineBasicBlock *InsertBB;
      MachineBasicBlock::iterator Insert;
      if (MI->getOpcode() == AMDGPU::REG_SEQUENCE) {
        InsertBB = MI->getParent();
        Insert = MI;
      } else {
        // MI is a PHI instruction.
        InsertBB = MI->getOperand(i + 1).getMBB();
        Insert = InsertBB->getFirstTerminator();
      }
      BuildMI(*InsertBB, Insert, MI->getDebugLoc(),
              get(AMDGPU::COPY), DstReg)
              .addOperand(MI->getOperand(i));
      MI->getOperand(i).setReg(DstReg);
    }
  }

  // Legalize INSERT_SUBREG
  // src0 must have the same register class as dst
  if (MI->getOpcode() == AMDGPU::INSERT_SUBREG) {
    unsigned Dst = MI->getOperand(0).getReg();
    unsigned Src0 = MI->getOperand(1).getReg();
    const TargetRegisterClass *DstRC = MRI.getRegClass(Dst);
    const TargetRegisterClass *Src0RC = MRI.getRegClass(Src0);
    if (DstRC != Src0RC) {
      MachineBasicBlock &MBB = *MI->getParent();
      unsigned NewSrc0 = MRI.createVirtualRegister(DstRC);
      BuildMI(MBB, MI, MI->getDebugLoc(), get(AMDGPU::COPY), NewSrc0)
              .addReg(Src0);
      MI->getOperand(1).setReg(NewSrc0);
    }
    return;
  }

  // Legalize MUBUF* instructions
  // FIXME: If we start using the non-addr64 instructions for compute, we
  // may need to legalize them here.

  int SRsrcIdx = AMDGPU::getNamedOperandIdx(MI->getOpcode(),
                                            AMDGPU::OpName::srsrc);
  int VAddrIdx = AMDGPU::getNamedOperandIdx(MI->getOpcode(),
                                             AMDGPU::OpName::vaddr);
  if (SRsrcIdx != -1 && VAddrIdx != -1) {
    const TargetRegisterClass *VAddrRC =
        RI.getRegClass(get(MI->getOpcode()).OpInfo[VAddrIdx].RegClass);

    if(VAddrRC->getSize() == 8 &&
       MRI.getRegClass(MI->getOperand(SRsrcIdx).getReg()) != VAddrRC) {
      // We have a MUBUF instruction that uses a 64-bit vaddr register and
      // srsrc has the incorrect register class.  In order to fix this, we
      // need to extract the pointer from the resource descriptor (srsrc),
      // add it to the value of vadd,  then store the result in the vaddr
      // operand.  Then, we need to set the pointer field of the resource
      // descriptor to zero.

      MachineBasicBlock &MBB = *MI->getParent();
      MachineOperand &SRsrcOp = MI->getOperand(SRsrcIdx);
      MachineOperand &VAddrOp = MI->getOperand(VAddrIdx);
      unsigned SRsrcPtrLo, SRsrcPtrHi, VAddrLo, VAddrHi;
      unsigned NewVAddrLo = MRI.createVirtualRegister(&AMDGPU::VReg_32RegClass);
      unsigned NewVAddrHi = MRI.createVirtualRegister(&AMDGPU::VReg_32RegClass);
      unsigned NewVAddr = MRI.createVirtualRegister(&AMDGPU::VReg_64RegClass);
      unsigned Zero64 = MRI.createVirtualRegister(&AMDGPU::SReg_64RegClass);
      unsigned SRsrcFormatLo = MRI.createVirtualRegister(&AMDGPU::SGPR_32RegClass);
      unsigned SRsrcFormatHi = MRI.createVirtualRegister(&AMDGPU::SGPR_32RegClass);
      unsigned NewSRsrc = MRI.createVirtualRegister(&AMDGPU::SReg_128RegClass);

      // SRsrcPtrLo = srsrc:sub0
      SRsrcPtrLo = buildExtractSubReg(MI, MRI, SRsrcOp,
          &AMDGPU::VReg_128RegClass, AMDGPU::sub0, &AMDGPU::VReg_32RegClass);

      // SRsrcPtrHi = srsrc:sub1
      SRsrcPtrHi = buildExtractSubReg(MI, MRI, SRsrcOp,
          &AMDGPU::VReg_128RegClass, AMDGPU::sub1, &AMDGPU::VReg_32RegClass);

      // VAddrLo = vaddr:sub0
      VAddrLo = buildExtractSubReg(MI, MRI, VAddrOp,
          &AMDGPU::VReg_64RegClass, AMDGPU::sub0, &AMDGPU::VReg_32RegClass);

      // VAddrHi = vaddr:sub1
      VAddrHi = buildExtractSubReg(MI, MRI, VAddrOp,
          &AMDGPU::VReg_64RegClass, AMDGPU::sub1, &AMDGPU::VReg_32RegClass);

      // NewVaddrLo = SRsrcPtrLo + VAddrLo
      BuildMI(MBB, MI, MI->getDebugLoc(), get(AMDGPU::V_ADD_I32_e32),
              NewVAddrLo)
              .addReg(SRsrcPtrLo)
              .addReg(VAddrLo)
              .addReg(AMDGPU::VCC, RegState::Define | RegState::Implicit);

      // NewVaddrHi = SRsrcPtrHi + VAddrHi
      BuildMI(MBB, MI, MI->getDebugLoc(), get(AMDGPU::V_ADDC_U32_e32),
              NewVAddrHi)
              .addReg(SRsrcPtrHi)
              .addReg(VAddrHi)
              .addReg(AMDGPU::VCC, RegState::ImplicitDefine)
              .addReg(AMDGPU::VCC, RegState::Implicit);

      // NewVaddr = {NewVaddrHi, NewVaddrLo}
      BuildMI(MBB, MI, MI->getDebugLoc(), get(AMDGPU::REG_SEQUENCE),
              NewVAddr)
              .addReg(NewVAddrLo)
              .addImm(AMDGPU::sub0)
              .addReg(NewVAddrHi)
              .addImm(AMDGPU::sub1);

      // Zero64 = 0
      BuildMI(MBB, MI, MI->getDebugLoc(), get(AMDGPU::S_MOV_B64),
              Zero64)
              .addImm(0);

      // SRsrcFormatLo = RSRC_DATA_FORMAT{31-0}
      BuildMI(MBB, MI, MI->getDebugLoc(), get(AMDGPU::S_MOV_B32),
              SRsrcFormatLo)
              .addImm(AMDGPU::RSRC_DATA_FORMAT & 0xFFFFFFFF);

      // SRsrcFormatHi = RSRC_DATA_FORMAT{63-32}
      BuildMI(MBB, MI, MI->getDebugLoc(), get(AMDGPU::S_MOV_B32),
              SRsrcFormatHi)
              .addImm(AMDGPU::RSRC_DATA_FORMAT >> 32);

      // NewSRsrc = {Zero64, SRsrcFormat}
      BuildMI(MBB, MI, MI->getDebugLoc(), get(AMDGPU::REG_SEQUENCE),
              NewSRsrc)
              .addReg(Zero64)
              .addImm(AMDGPU::sub0_sub1)
              .addReg(SRsrcFormatLo)
              .addImm(AMDGPU::sub2)
              .addReg(SRsrcFormatHi)
              .addImm(AMDGPU::sub3);

      // Update the instruction to use NewVaddr
      MI->getOperand(VAddrIdx).setReg(NewVAddr);
      // Update the instruction to use NewSRsrc
      MI->getOperand(SRsrcIdx).setReg(NewSRsrc);
    }
  }
}

void SIInstrInfo::moveSMRDToVALU(MachineInstr *MI, MachineRegisterInfo &MRI) const {
  MachineBasicBlock *MBB = MI->getParent();
  switch (MI->getOpcode()) {
    case AMDGPU::S_LOAD_DWORD_IMM:
    case AMDGPU::S_LOAD_DWORD_SGPR:
    case AMDGPU::S_LOAD_DWORDX2_IMM:
    case AMDGPU::S_LOAD_DWORDX2_SGPR:
    case AMDGPU::S_LOAD_DWORDX4_IMM:
    case AMDGPU::S_LOAD_DWORDX4_SGPR:
      unsigned NewOpcode = getVALUOp(*MI);
      unsigned RegOffset;
      unsigned ImmOffset;

      if (MI->getOperand(2).isReg()) {
        RegOffset = MI->getOperand(2).getReg();
        ImmOffset = 0;
      } else {
        assert(MI->getOperand(2).isImm());
        // SMRD instructions take a dword offsets and MUBUF instructions
        // take a byte offset.
        ImmOffset = MI->getOperand(2).getImm() << 2;
        RegOffset = MRI.createVirtualRegister(&AMDGPU::SGPR_32RegClass);
        if (isUInt<12>(ImmOffset)) {
          BuildMI(*MBB, MI, MI->getDebugLoc(), get(AMDGPU::S_MOV_B32),
                  RegOffset)
                  .addImm(0);
        } else {
          BuildMI(*MBB, MI, MI->getDebugLoc(), get(AMDGPU::S_MOV_B32),
                  RegOffset)
                  .addImm(ImmOffset);
          ImmOffset = 0;
        }
      }

      unsigned SRsrc = MRI.createVirtualRegister(&AMDGPU::SReg_128RegClass);
      unsigned DWord0 = RegOffset;
      unsigned DWord1 = MRI.createVirtualRegister(&AMDGPU::SGPR_32RegClass);
      unsigned DWord2 = MRI.createVirtualRegister(&AMDGPU::SGPR_32RegClass);
      unsigned DWord3 = MRI.createVirtualRegister(&AMDGPU::SGPR_32RegClass);

      BuildMI(*MBB, MI, MI->getDebugLoc(), get(AMDGPU::S_MOV_B32), DWord1)
              .addImm(0);
      BuildMI(*MBB, MI, MI->getDebugLoc(), get(AMDGPU::S_MOV_B32), DWord2)
              .addImm(AMDGPU::RSRC_DATA_FORMAT & 0xFFFFFFFF);
      BuildMI(*MBB, MI, MI->getDebugLoc(), get(AMDGPU::S_MOV_B32), DWord3)
              .addImm(AMDGPU::RSRC_DATA_FORMAT >> 32);
      BuildMI(*MBB, MI, MI->getDebugLoc(), get(AMDGPU::REG_SEQUENCE), SRsrc)
              .addReg(DWord0)
              .addImm(AMDGPU::sub0)
              .addReg(DWord1)
              .addImm(AMDGPU::sub1)
              .addReg(DWord2)
              .addImm(AMDGPU::sub2)
              .addReg(DWord3)
              .addImm(AMDGPU::sub3);
     MI->setDesc(get(NewOpcode));
     if (MI->getOperand(2).isReg()) {
       MI->getOperand(2).setReg(MI->getOperand(1).getReg());
     } else {
       MI->getOperand(2).ChangeToRegister(MI->getOperand(1).getReg(), false);
     }
     MI->getOperand(1).setReg(SRsrc);
     MI->addOperand(*MBB->getParent(), MachineOperand::CreateImm(ImmOffset));
  }
}

void SIInstrInfo::moveToVALU(MachineInstr &TopInst) const {
  SmallVector<MachineInstr *, 128> Worklist;
  Worklist.push_back(&TopInst);

  while (!Worklist.empty()) {
    MachineInstr *Inst = Worklist.pop_back_val();
    MachineBasicBlock *MBB = Inst->getParent();
    MachineRegisterInfo &MRI = MBB->getParent()->getRegInfo();

    unsigned Opcode = Inst->getOpcode();
    unsigned NewOpcode = getVALUOp(*Inst);

    // Handle some special cases
    switch (Opcode) {
    default:
      if (isSMRD(Inst->getOpcode())) {
        moveSMRDToVALU(Inst, MRI);
      }
      break;
    case AMDGPU::S_MOV_B64: {
      DebugLoc DL = Inst->getDebugLoc();

      // If the source operand is a register we can replace this with a
      // copy.
      if (Inst->getOperand(1).isReg()) {
        MachineInstr *Copy = BuildMI(*MBB, Inst, DL, get(TargetOpcode::COPY))
          .addOperand(Inst->getOperand(0))
          .addOperand(Inst->getOperand(1));
        Worklist.push_back(Copy);
      } else {
        // Otherwise, we need to split this into two movs, because there is
        // no 64-bit VALU move instruction.
        unsigned Reg = Inst->getOperand(0).getReg();
        unsigned Dst = split64BitImm(Worklist,
                                     Inst,
                                     MRI,
                                     MRI.getRegClass(Reg),
                                     Inst->getOperand(1));
        MRI.replaceRegWith(Reg, Dst);
      }
      Inst->eraseFromParent();
      continue;
    }
    case AMDGPU::S_AND_B64:
      splitScalar64BitBinaryOp(Worklist, Inst, AMDGPU::S_AND_B32);
      Inst->eraseFromParent();
      continue;

    case AMDGPU::S_OR_B64:
      splitScalar64BitBinaryOp(Worklist, Inst, AMDGPU::S_OR_B32);
      Inst->eraseFromParent();
      continue;

    case AMDGPU::S_XOR_B64:
      splitScalar64BitBinaryOp(Worklist, Inst, AMDGPU::S_XOR_B32);
      Inst->eraseFromParent();
      continue;

    case AMDGPU::S_NOT_B64:
      splitScalar64BitUnaryOp(Worklist, Inst, AMDGPU::S_NOT_B32);
      Inst->eraseFromParent();
      continue;

    case AMDGPU::S_BCNT1_I32_B64:
      splitScalar64BitBCNT(Worklist, Inst);
      Inst->eraseFromParent();
      continue;

    case AMDGPU::S_BFE_U64:
    case AMDGPU::S_BFE_I64:
    case AMDGPU::S_BFM_B64:
      llvm_unreachable("Moving this op to VALU not implemented");
    }

    if (NewOpcode == AMDGPU::INSTRUCTION_LIST_END) {
      // We cannot move this instruction to the VALU, so we should try to
      // legalize its operands instead.
      legalizeOperands(Inst);
      continue;
    }

    // Use the new VALU Opcode.
    const MCInstrDesc &NewDesc = get(NewOpcode);
    Inst->setDesc(NewDesc);

    // Remove any references to SCC. Vector instructions can't read from it, and
    // We're just about to add the implicit use / defs of VCC, and we don't want
    // both.
    for (unsigned i = Inst->getNumOperands() - 1; i > 0; --i) {
      MachineOperand &Op = Inst->getOperand(i);
      if (Op.isReg() && Op.getReg() == AMDGPU::SCC)
        Inst->RemoveOperand(i);
    }

    if (Opcode == AMDGPU::S_SEXT_I32_I8 || Opcode == AMDGPU::S_SEXT_I32_I16) {
      // We are converting these to a BFE, so we need to add the missing
      // operands for the size and offset.
      unsigned Size = (Opcode == AMDGPU::S_SEXT_I32_I8) ? 8 : 16;
      Inst->addOperand(Inst->getOperand(1));
      Inst->getOperand(1).ChangeToImmediate(0);
      Inst->addOperand(MachineOperand::CreateImm(0));
      Inst->addOperand(MachineOperand::CreateImm(0));
      Inst->addOperand(MachineOperand::CreateImm(0));
      Inst->addOperand(MachineOperand::CreateImm(Size));

      // XXX - Other pointless operands. There are 4, but it seems you only need
      // 3 to not hit an assertion later in MCInstLower.
      Inst->addOperand(MachineOperand::CreateImm(0));
      Inst->addOperand(MachineOperand::CreateImm(0));
    } else if (Opcode == AMDGPU::S_BCNT1_I32_B32) {
      // The VALU version adds the second operand to the result, so insert an
      // extra 0 operand.
      Inst->addOperand(MachineOperand::CreateImm(0));
    }

    addDescImplicitUseDef(NewDesc, Inst);

    if (Opcode == AMDGPU::S_BFE_I32 || Opcode == AMDGPU::S_BFE_U32) {
      const MachineOperand &OffsetWidthOp = Inst->getOperand(2);
      // If we need to move this to VGPRs, we need to unpack the second operand
      // back into the 2 separate ones for bit offset and width.
      assert(OffsetWidthOp.isImm() &&
             "Scalar BFE is only implemented for constant width and offset");
      uint32_t Imm = OffsetWidthOp.getImm();

      uint32_t Offset = Imm & 0x3f; // Extract bits [5:0].
      uint32_t BitWidth = (Imm & 0x7f0000) >> 16; // Extract bits [22:16].

      Inst->RemoveOperand(2); // Remove old immediate.
      Inst->addOperand(Inst->getOperand(1));
      Inst->getOperand(1).ChangeToImmediate(0);
      Inst->addOperand(MachineOperand::CreateImm(0));
      Inst->addOperand(MachineOperand::CreateImm(Offset));
      Inst->addOperand(MachineOperand::CreateImm(0));
      Inst->addOperand(MachineOperand::CreateImm(BitWidth));
      Inst->addOperand(MachineOperand::CreateImm(0));
      Inst->addOperand(MachineOperand::CreateImm(0));
    }

    // Update the destination register class.

    const TargetRegisterClass *NewDstRC = getOpRegClass(*Inst, 0);

    switch (Opcode) {
      // For target instructions, getOpRegClass just returns the virtual
      // register class associated with the operand, so we need to find an
      // equivalent VGPR register class in order to move the instruction to the
      // VALU.
    case AMDGPU::COPY:
    case AMDGPU::PHI:
    case AMDGPU::REG_SEQUENCE:
    case AMDGPU::INSERT_SUBREG:
      if (RI.hasVGPRs(NewDstRC))
        continue;
      NewDstRC = RI.getEquivalentVGPRClass(NewDstRC);
      if (!NewDstRC)
        continue;
      break;
    default:
      break;
    }

    unsigned DstReg = Inst->getOperand(0).getReg();
    unsigned NewDstReg = MRI.createVirtualRegister(NewDstRC);
    MRI.replaceRegWith(DstReg, NewDstReg);

    // Legalize the operands
    legalizeOperands(Inst);

    for (MachineRegisterInfo::use_iterator I = MRI.use_begin(NewDstReg),
           E = MRI.use_end(); I != E; ++I) {
      MachineInstr &UseMI = *I->getParent();
      if (!canReadVGPR(UseMI, I.getOperandNo())) {
        Worklist.push_back(&UseMI);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Indirect addressing callbacks
//===----------------------------------------------------------------------===//

unsigned SIInstrInfo::calculateIndirectAddress(unsigned RegIndex,
                                                 unsigned Channel) const {
  assert(Channel == 0);
  return RegIndex;
}

const TargetRegisterClass *SIInstrInfo::getIndirectAddrRegClass() const {
  return &AMDGPU::VReg_32RegClass;
}

void SIInstrInfo::splitScalar64BitUnaryOp(
  SmallVectorImpl<MachineInstr *> &Worklist,
  MachineInstr *Inst,
  unsigned Opcode) const {
  MachineBasicBlock &MBB = *Inst->getParent();
  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();

  MachineOperand &Dest = Inst->getOperand(0);
  MachineOperand &Src0 = Inst->getOperand(1);
  DebugLoc DL = Inst->getDebugLoc();

  MachineBasicBlock::iterator MII = Inst;

  const MCInstrDesc &InstDesc = get(Opcode);
  const TargetRegisterClass *Src0RC = Src0.isReg() ?
    MRI.getRegClass(Src0.getReg()) :
    &AMDGPU::SGPR_32RegClass;

  const TargetRegisterClass *Src0SubRC = RI.getSubRegClass(Src0RC, AMDGPU::sub0);

  MachineOperand SrcReg0Sub0 = buildExtractSubRegOrImm(MII, MRI, Src0, Src0RC,
                                                       AMDGPU::sub0, Src0SubRC);

  const TargetRegisterClass *DestRC = MRI.getRegClass(Dest.getReg());
  const TargetRegisterClass *DestSubRC = RI.getSubRegClass(DestRC, AMDGPU::sub0);

  unsigned DestSub0 = MRI.createVirtualRegister(DestRC);
  MachineInstr *LoHalf = BuildMI(MBB, MII, DL, InstDesc, DestSub0)
    .addOperand(SrcReg0Sub0);

  MachineOperand SrcReg0Sub1 = buildExtractSubRegOrImm(MII, MRI, Src0, Src0RC,
                                                       AMDGPU::sub1, Src0SubRC);

  unsigned DestSub1 = MRI.createVirtualRegister(DestSubRC);
  MachineInstr *HiHalf = BuildMI(MBB, MII, DL, InstDesc, DestSub1)
    .addOperand(SrcReg0Sub1);

  unsigned FullDestReg = MRI.createVirtualRegister(DestRC);
  BuildMI(MBB, MII, DL, get(TargetOpcode::REG_SEQUENCE), FullDestReg)
    .addReg(DestSub0)
    .addImm(AMDGPU::sub0)
    .addReg(DestSub1)
    .addImm(AMDGPU::sub1);

  MRI.replaceRegWith(Dest.getReg(), FullDestReg);

  // Try to legalize the operands in case we need to swap the order to keep it
  // valid.
  Worklist.push_back(LoHalf);
  Worklist.push_back(HiHalf);
}

void SIInstrInfo::splitScalar64BitBinaryOp(
  SmallVectorImpl<MachineInstr *> &Worklist,
  MachineInstr *Inst,
  unsigned Opcode) const {
  MachineBasicBlock &MBB = *Inst->getParent();
  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();

  MachineOperand &Dest = Inst->getOperand(0);
  MachineOperand &Src0 = Inst->getOperand(1);
  MachineOperand &Src1 = Inst->getOperand(2);
  DebugLoc DL = Inst->getDebugLoc();

  MachineBasicBlock::iterator MII = Inst;

  const MCInstrDesc &InstDesc = get(Opcode);
  const TargetRegisterClass *Src0RC = Src0.isReg() ?
    MRI.getRegClass(Src0.getReg()) :
    &AMDGPU::SGPR_32RegClass;

  const TargetRegisterClass *Src0SubRC = RI.getSubRegClass(Src0RC, AMDGPU::sub0);
  const TargetRegisterClass *Src1RC = Src1.isReg() ?
    MRI.getRegClass(Src1.getReg()) :
    &AMDGPU::SGPR_32RegClass;

  const TargetRegisterClass *Src1SubRC = RI.getSubRegClass(Src1RC, AMDGPU::sub0);

  MachineOperand SrcReg0Sub0 = buildExtractSubRegOrImm(MII, MRI, Src0, Src0RC,
                                                       AMDGPU::sub0, Src0SubRC);
  MachineOperand SrcReg1Sub0 = buildExtractSubRegOrImm(MII, MRI, Src1, Src1RC,
                                                       AMDGPU::sub0, Src1SubRC);

  const TargetRegisterClass *DestRC = MRI.getRegClass(Dest.getReg());
  const TargetRegisterClass *DestSubRC = RI.getSubRegClass(DestRC, AMDGPU::sub0);

  unsigned DestSub0 = MRI.createVirtualRegister(DestRC);
  MachineInstr *LoHalf = BuildMI(MBB, MII, DL, InstDesc, DestSub0)
    .addOperand(SrcReg0Sub0)
    .addOperand(SrcReg1Sub0);

  MachineOperand SrcReg0Sub1 = buildExtractSubRegOrImm(MII, MRI, Src0, Src0RC,
                                                       AMDGPU::sub1, Src0SubRC);
  MachineOperand SrcReg1Sub1 = buildExtractSubRegOrImm(MII, MRI, Src1, Src1RC,
                                                       AMDGPU::sub1, Src1SubRC);

  unsigned DestSub1 = MRI.createVirtualRegister(DestSubRC);
  MachineInstr *HiHalf = BuildMI(MBB, MII, DL, InstDesc, DestSub1)
    .addOperand(SrcReg0Sub1)
    .addOperand(SrcReg1Sub1);

  unsigned FullDestReg = MRI.createVirtualRegister(DestRC);
  BuildMI(MBB, MII, DL, get(TargetOpcode::REG_SEQUENCE), FullDestReg)
    .addReg(DestSub0)
    .addImm(AMDGPU::sub0)
    .addReg(DestSub1)
    .addImm(AMDGPU::sub1);

  MRI.replaceRegWith(Dest.getReg(), FullDestReg);

  // Try to legalize the operands in case we need to swap the order to keep it
  // valid.
  Worklist.push_back(LoHalf);
  Worklist.push_back(HiHalf);
}

void SIInstrInfo::splitScalar64BitBCNT(SmallVectorImpl<MachineInstr *> &Worklist,
                                       MachineInstr *Inst) const {
  MachineBasicBlock &MBB = *Inst->getParent();
  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();

  MachineBasicBlock::iterator MII = Inst;
  DebugLoc DL = Inst->getDebugLoc();

  MachineOperand &Dest = Inst->getOperand(0);
  MachineOperand &Src = Inst->getOperand(1);

  const MCInstrDesc &InstDesc = get(AMDGPU::V_BCNT_U32_B32_e32);
  const TargetRegisterClass *SrcRC = Src.isReg() ?
    MRI.getRegClass(Src.getReg()) :
    &AMDGPU::SGPR_32RegClass;

  unsigned MidReg = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  unsigned ResultReg = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);

  const TargetRegisterClass *SrcSubRC = RI.getSubRegClass(SrcRC, AMDGPU::sub0);

  MachineOperand SrcRegSub0 = buildExtractSubRegOrImm(MII, MRI, Src, SrcRC,
                                                      AMDGPU::sub0, SrcSubRC);
  MachineOperand SrcRegSub1 = buildExtractSubRegOrImm(MII, MRI, Src, SrcRC,
                                                      AMDGPU::sub1, SrcSubRC);

  MachineInstr *First = BuildMI(MBB, MII, DL, InstDesc, MidReg)
    .addOperand(SrcRegSub0)
    .addImm(0);

  MachineInstr *Second = BuildMI(MBB, MII, DL, InstDesc, ResultReg)
    .addOperand(SrcRegSub1)
    .addReg(MidReg);

  MRI.replaceRegWith(Dest.getReg(), ResultReg);

  Worklist.push_back(First);
  Worklist.push_back(Second);
}

void SIInstrInfo::addDescImplicitUseDef(const MCInstrDesc &NewDesc,
                                        MachineInstr *Inst) const {
  // Add the implict and explicit register definitions.
  if (NewDesc.ImplicitUses) {
    for (unsigned i = 0; NewDesc.ImplicitUses[i]; ++i) {
      unsigned Reg = NewDesc.ImplicitUses[i];
      Inst->addOperand(MachineOperand::CreateReg(Reg, false, true));
    }
  }

  if (NewDesc.ImplicitDefs) {
    for (unsigned i = 0; NewDesc.ImplicitDefs[i]; ++i) {
      unsigned Reg = NewDesc.ImplicitDefs[i];
      Inst->addOperand(MachineOperand::CreateReg(Reg, true, true));
    }
  }
}

MachineInstrBuilder SIInstrInfo::buildIndirectWrite(
                                   MachineBasicBlock *MBB,
                                   MachineBasicBlock::iterator I,
                                   unsigned ValueReg,
                                   unsigned Address, unsigned OffsetReg) const {
  const DebugLoc &DL = MBB->findDebugLoc(I);
  unsigned IndirectBaseReg = AMDGPU::VReg_32RegClass.getRegister(
                                      getIndirectIndexBegin(*MBB->getParent()));

  return BuildMI(*MBB, I, DL, get(AMDGPU::SI_INDIRECT_DST_V1))
          .addReg(IndirectBaseReg, RegState::Define)
          .addOperand(I->getOperand(0))
          .addReg(IndirectBaseReg)
          .addReg(OffsetReg)
          .addImm(0)
          .addReg(ValueReg);
}

MachineInstrBuilder SIInstrInfo::buildIndirectRead(
                                   MachineBasicBlock *MBB,
                                   MachineBasicBlock::iterator I,
                                   unsigned ValueReg,
                                   unsigned Address, unsigned OffsetReg) const {
  const DebugLoc &DL = MBB->findDebugLoc(I);
  unsigned IndirectBaseReg = AMDGPU::VReg_32RegClass.getRegister(
                                      getIndirectIndexBegin(*MBB->getParent()));

  return BuildMI(*MBB, I, DL, get(AMDGPU::SI_INDIRECT_SRC))
          .addOperand(I->getOperand(0))
          .addOperand(I->getOperand(1))
          .addReg(IndirectBaseReg)
          .addReg(OffsetReg)
          .addImm(0);

}

void SIInstrInfo::reserveIndirectRegisters(BitVector &Reserved,
                                            const MachineFunction &MF) const {
  int End = getIndirectIndexEnd(MF);
  int Begin = getIndirectIndexBegin(MF);

  if (End == -1)
    return;


  for (int Index = Begin; Index <= End; ++Index)
    Reserved.set(AMDGPU::VReg_32RegClass.getRegister(Index));

  for (int Index = std::max(0, Begin - 1); Index <= End; ++Index)
    Reserved.set(AMDGPU::VReg_64RegClass.getRegister(Index));

  for (int Index = std::max(0, Begin - 2); Index <= End; ++Index)
    Reserved.set(AMDGPU::VReg_96RegClass.getRegister(Index));

  for (int Index = std::max(0, Begin - 3); Index <= End; ++Index)
    Reserved.set(AMDGPU::VReg_128RegClass.getRegister(Index));

  for (int Index = std::max(0, Begin - 7); Index <= End; ++Index)
    Reserved.set(AMDGPU::VReg_256RegClass.getRegister(Index));

  for (int Index = std::max(0, Begin - 15); Index <= End; ++Index)
    Reserved.set(AMDGPU::VReg_512RegClass.getRegister(Index));
}

const MachineOperand *SIInstrInfo::getNamedOperand(const MachineInstr& MI,
                                                   unsigned OperandName) const {
  int Idx = AMDGPU::getNamedOperandIdx(MI.getOpcode(), OperandName);
  if (Idx == -1)
    return nullptr;

  return &MI.getOperand(Idx);
}
