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
#include "llvm/Support/CommandLine.h"
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
HexagonRegisterInfo::getCallerSavedRegs(const MachineFunction *MF) const {
  static const MCPhysReg CallerSavedRegsV4[] = {
    Hexagon::R0, Hexagon::R1, Hexagon::R2, Hexagon::R3, Hexagon::R4,
    Hexagon::R5, Hexagon::R6, Hexagon::R7, Hexagon::R8, Hexagon::R9,
    Hexagon::R10, Hexagon::R11, Hexagon::R12, Hexagon::R13, Hexagon::R14,
    Hexagon::R15, 0
  };

  auto &HST = static_cast<const HexagonSubtarget&>(MF->getSubtarget());
  switch (HST.getHexagonArchVersion()) {
  case HexagonSubtarget::V4:
  case HexagonSubtarget::V5:
  case HexagonSubtarget::V55:
  case HexagonSubtarget::V60:
    return CallerSavedRegsV4;
  }
  llvm_unreachable(
    "Callee saved registers requested for unknown archtecture version");
}


const MCPhysReg *
HexagonRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  static const MCPhysReg CalleeSavedRegsV3[] = {
    Hexagon::R16,   Hexagon::R17,   Hexagon::R18,   Hexagon::R19,
    Hexagon::R20,   Hexagon::R21,   Hexagon::R22,   Hexagon::R23,
    Hexagon::R24,   Hexagon::R25,   Hexagon::R26,   Hexagon::R27, 0
  };

  switch (MF->getSubtarget<HexagonSubtarget>().getHexagonArchVersion()) {
  case HexagonSubtarget::V4:
  case HexagonSubtarget::V5:
  case HexagonSubtarget::V55:
  case HexagonSubtarget::V60:
    return CalleeSavedRegsV3;
  }
  llvm_unreachable("Callee saved registers requested for unknown architecture "
                   "version");
}

BitVector HexagonRegisterInfo::getReservedRegs(const MachineFunction &MF)
  const {
  BitVector Reserved(getNumRegs());
  Reserved.set(HEXAGON_RESERVED_REG_1);
  Reserved.set(HEXAGON_RESERVED_REG_2);
  Reserved.set(Hexagon::R29);
  Reserved.set(Hexagon::R30);
  Reserved.set(Hexagon::R31);
  Reserved.set(Hexagon::D14);
  Reserved.set(Hexagon::D15);
  Reserved.set(Hexagon::LC0);
  Reserved.set(Hexagon::LC1);
  Reserved.set(Hexagon::SA0);
  Reserved.set(Hexagon::SA1);
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

  if (HII.isValidOffset(Opc, RealOffset)) {
    MI.getOperand(FIOp).ChangeToRegister(BP, false);
    MI.getOperand(FIOp+1).ChangeToImmediate(RealOffset);
    return;
  }

#ifndef NDEBUG
  const Function *F = MF.getFunction();
  dbgs() << "In function ";
  if (F) dbgs() << F->getName();
  else   dbgs() << "<?>";
  dbgs() << ", BB#" << MB.getNumber() << "\n" << MI;
#endif
  llvm_unreachable("Unhandled instruction");
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
