//===-- HexagonRegisterInfo.cpp - Hexagon Register Information ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Hexagon implementation of the TargetRegisterInfo
// class.
//
//===----------------------------------------------------------------------===//

#include "HexagonRegisterInfo.h"
#include "Hexagon.h"
#include "HexagonMachineFunctionInfo.h"
#include "HexagonSubtarget.h"
#include "HexagonTargetMachine.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/MC/MachineLocation.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;

HexagonRegisterInfo::HexagonRegisterInfo()
    : HexagonGenRegisterInfo(Hexagon::R31) {}


bool HexagonRegisterInfo::isEHReturnCalleeSaveReg(unsigned R) const {
  return R == Hexagon::R0 || R == Hexagon::R1 || R == Hexagon::R2 ||
         R == Hexagon::R3 || R == Hexagon::D0 || R == Hexagon::D1;
}

bool HexagonRegisterInfo::isCalleeSaveReg(unsigned Reg) const {
  return Hexagon::R16 <= Reg && Reg <= Hexagon::R27;
}


const MCPhysReg *
HexagonRegisterInfo::getCallerSavedRegs(const MachineFunction *MF,
      const TargetRegisterClass *RC) const {
  using namespace Hexagon;

  static const MCPhysReg Int32[] = {
    R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, 0
  };
  static const MCPhysReg Int64[] = {
    D0, D1, D2, D3, D4, D5, D6, D7, 0
  };
  static const MCPhysReg Pred[] = {
    P0, P1, P2, P3, 0
  };
  static const MCPhysReg VecSgl[] = {
     V0,  V1,  V2,  V3,  V4,  V5,  V6,  V7,  V8,  V9, V10, V11, V12, V13,
    V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27,
    V28, V29, V30, V31,   0
  };
  static const MCPhysReg VecDbl[] = {
    W0, W1, W2, W3, W4, W5, W6, W7, W8, W9, W10, W11, W12, W13, W14, W15, 0
  };

  switch (RC->getID()) {
    case IntRegsRegClassID:
      return Int32;
    case DoubleRegsRegClassID:
      return Int64;
    case PredRegsRegClassID:
      return Pred;
    case VectorRegsRegClassID:
    case VectorRegs128BRegClassID:
      return VecSgl;
    case VecDblRegsRegClassID:
    case VecDblRegs128BRegClassID:
      return VecDbl;
    default:
      break;
  }

  static const MCPhysReg Empty[] = { 0 };
#ifndef NDEBUG
  dbgs() << "Register class: " << getRegClassName(RC) << "\n";
#endif
  llvm_unreachable("Unexpected register class");
  return Empty;
}


const MCPhysReg *
HexagonRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  static const MCPhysReg CalleeSavedRegsV3[] = {
    Hexagon::R16,   Hexagon::R17,   Hexagon::R18,   Hexagon::R19,
    Hexagon::R20,   Hexagon::R21,   Hexagon::R22,   Hexagon::R23,
    Hexagon::R24,   Hexagon::R25,   Hexagon::R26,   Hexagon::R27, 0
  };

  // Functions that contain a call to __builtin_eh_return also save the first 4
  // parameter registers.
  static const MCPhysReg CalleeSavedRegsV3EHReturn[] = {
    Hexagon::R0,    Hexagon::R1,    Hexagon::R2,    Hexagon::R3,
    Hexagon::R16,   Hexagon::R17,   Hexagon::R18,   Hexagon::R19,
    Hexagon::R20,   Hexagon::R21,   Hexagon::R22,   Hexagon::R23,
    Hexagon::R24,   Hexagon::R25,   Hexagon::R26,   Hexagon::R27, 0
  };

  bool HasEHReturn = MF->getInfo<HexagonMachineFunctionInfo>()->hasEHReturn();

  switch (MF->getSubtarget<HexagonSubtarget>().getHexagonArchVersion()) {
  case HexagonSubtarget::V4:
  case HexagonSubtarget::V5:
  case HexagonSubtarget::V55:
  case HexagonSubtarget::V60:
    return HasEHReturn ? CalleeSavedRegsV3EHReturn : CalleeSavedRegsV3;
  }

  llvm_unreachable("Callee saved registers requested for unknown architecture "
                   "version");
}


BitVector HexagonRegisterInfo::getReservedRegs(const MachineFunction &MF)
  const {
  BitVector Reserved(getNumRegs());
  Reserved.set(Hexagon::R29);
  Reserved.set(Hexagon::R30);
  Reserved.set(Hexagon::R31);
  Reserved.set(Hexagon::PC);
  Reserved.set(Hexagon::D14);
  Reserved.set(Hexagon::D15);
  Reserved.set(Hexagon::LC0);
  Reserved.set(Hexagon::LC1);
  Reserved.set(Hexagon::SA0);
  Reserved.set(Hexagon::SA1);
  Reserved.set(Hexagon::UGP);
  Reserved.set(Hexagon::GP);
  Reserved.set(Hexagon::CS0);
  Reserved.set(Hexagon::CS1);
  Reserved.set(Hexagon::CS);
  return Reserved;
}


void HexagonRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                              int SPAdj, unsigned FIOp,
                                              RegScavenger *RS) const {
  //
  // Hexagon_TODO: Do we need to enforce this for Hexagon?
  assert(SPAdj == 0 && "Unexpected");

  MachineInstr &MI = *II;
  MachineBasicBlock &MB = *MI.getParent();
  MachineFunction &MF = *MB.getParent();
  auto &HST = MF.getSubtarget<HexagonSubtarget>();
  auto &HII = *HST.getInstrInfo();
  auto &HFI = *HST.getFrameLowering();

  unsigned BP = 0;
  int FI = MI.getOperand(FIOp).getIndex();
  // Select the base pointer (BP) and calculate the actual offset from BP
  // to the beginning of the object at index FI.
  int Offset = HFI.getFrameIndexReference(MF, FI, BP);
  // Add the offset from the instruction.
  int RealOffset = Offset + MI.getOperand(FIOp+1).getImm();
  bool IsKill = false;

  unsigned Opc = MI.getOpcode();
  switch (Opc) {
    case Hexagon::TFR_FIA:
      MI.setDesc(HII.get(Hexagon::A2_addi));
      MI.getOperand(FIOp).ChangeToImmediate(RealOffset);
      MI.RemoveOperand(FIOp+1);
      return;
    case Hexagon::TFR_FI:
      // Set up the instruction for updating below.
      MI.setDesc(HII.get(Hexagon::A2_addi));
      break;
  }

  if (!HII.isValidOffset(Opc, RealOffset)) {
    // If the offset is not valid, calculate the address in a temporary
    // register and use it with offset 0.
    auto &MRI = MF.getRegInfo();
    unsigned TmpR = MRI.createVirtualRegister(&Hexagon::IntRegsRegClass);
    const DebugLoc &DL = MI.getDebugLoc();
    BuildMI(MB, II, DL, HII.get(Hexagon::A2_addi), TmpR)
      .addReg(BP)
      .addImm(RealOffset);
    BP = TmpR;
    RealOffset = 0;
    IsKill = true;
  }

  MI.getOperand(FIOp).ChangeToRegister(BP, false, false, IsKill);
  MI.getOperand(FIOp+1).ChangeToImmediate(RealOffset);
}


unsigned HexagonRegisterInfo::getRARegister() const {
  return Hexagon::R31;
}


unsigned HexagonRegisterInfo::getFrameRegister(const MachineFunction
                                               &MF) const {
  const HexagonFrameLowering *TFI = getFrameLowering(MF);
  if (TFI->hasFP(MF))
    return getFrameRegister();
  return getStackRegister();
}


unsigned HexagonRegisterInfo::getFrameRegister() const {
  return Hexagon::R30;
}


unsigned HexagonRegisterInfo::getStackRegister() const {
  return Hexagon::R29;
}


bool HexagonRegisterInfo::useFPForScavengingIndex(const MachineFunction &MF)
      const {
  return MF.getSubtarget<HexagonSubtarget>().getFrameLowering()->hasFP(MF);
}


unsigned HexagonRegisterInfo::getFirstCallerSavedNonParamReg() const {
  return Hexagon::R6;
}


#define GET_REGINFO_TARGET_DESC
#include "HexagonGenRegisterInfo.inc"
