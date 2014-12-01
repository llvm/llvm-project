//===-- MipsRegisterInfo.cpp - MIPS Register Information -== --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the MIPS implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "MipsRegisterInfo.h"
#include "Mips.h"
#include "MipsAnalyzeImmediate.h"
#include "MipsInstrInfo.h"
#include "MipsMachineFunction.h"
#include "MipsSubtarget.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;

#define DEBUG_TYPE "mips-reg-info"

#define GET_REGINFO_TARGET_DESC
#include "MipsGenRegisterInfo.inc"

MipsRegisterInfo::MipsRegisterInfo(const MipsSubtarget &ST)
  : MipsGenRegisterInfo(Mips::RA), Subtarget(ST) {}

unsigned MipsRegisterInfo::getPICCallReg() { return Mips::T9; }

const TargetRegisterClass *
MipsRegisterInfo::getPointerRegClass(const MachineFunction &MF,
                                     unsigned Kind) const {
  return Subtarget.isABI_N64() ? &Mips::GPR64RegClass : &Mips::GPR32RegClass;
}

unsigned
MipsRegisterInfo::getRegPressureLimit(const TargetRegisterClass *RC,
                                      MachineFunction &MF) const {
  switch (RC->getID()) {
  default:
    return 0;
  case Mips::GPR32RegClassID:
  case Mips::GPR64RegClassID:
  case Mips::DSPRRegClassID: {
    const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();
    return 28 - TFI->hasFP(MF);
  }
  case Mips::FGR32RegClassID:
    return 32;
  case Mips::AFGR64RegClassID:
    return 16;
  case Mips::FGR64RegClassID:
    return 32;
  }
}

//===----------------------------------------------------------------------===//
// Callee Saved Registers methods
//===----------------------------------------------------------------------===//

/// Mips Callee Saved Registers
const MCPhysReg *
MipsRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  if (Subtarget.isSingleFloat())
    return CSR_SingleFloatOnly_SaveList;

  if (Subtarget.isABI_N64())
    return CSR_N64_SaveList;

  if (Subtarget.isABI_N32())
    return CSR_N32_SaveList;

  if (Subtarget.isFP64bit())
    return CSR_O32_FP64_SaveList;

  if (Subtarget.isFPXX())
    return CSR_O32_FPXX_SaveList;

  return CSR_O32_SaveList;
}

const uint32_t*
MipsRegisterInfo::getCallPreservedMask(CallingConv::ID) const {
  if (Subtarget.isSingleFloat())
    return CSR_SingleFloatOnly_RegMask;

  if (Subtarget.isABI_N64())
    return CSR_N64_RegMask;

  if (Subtarget.isABI_N32())
    return CSR_N32_RegMask;

  if (Subtarget.isFP64bit())
    return CSR_O32_FP64_RegMask;

  if (Subtarget.isFPXX())
    return CSR_O32_FPXX_RegMask;

  return CSR_O32_RegMask;
}

const uint32_t *MipsRegisterInfo::getMips16RetHelperMask() {
  return CSR_Mips16RetHelper_RegMask;
}

BitVector MipsRegisterInfo::
getReservedRegs(const MachineFunction &MF) const {
  static const MCPhysReg ReservedGPR32[] = {
    Mips::ZERO, Mips::K0, Mips::K1, Mips::SP
  };

  static const MCPhysReg ReservedGPR64[] = {
    Mips::ZERO_64, Mips::K0_64, Mips::K1_64, Mips::SP_64
  };

  BitVector Reserved(getNumRegs());
  typedef TargetRegisterClass::const_iterator RegIter;

  for (unsigned I = 0; I < array_lengthof(ReservedGPR32); ++I)
    Reserved.set(ReservedGPR32[I]);

  // Reserve registers for the NaCl sandbox.
  if (Subtarget.isTargetNaCl()) {
    Reserved.set(Mips::T6);   // Reserved for control flow mask.
    Reserved.set(Mips::T7);   // Reserved for memory access mask.
    Reserved.set(Mips::T8);   // Reserved for thread pointer.
  }

  for (unsigned I = 0; I < array_lengthof(ReservedGPR64); ++I)
    Reserved.set(ReservedGPR64[I]);

  // For mno-abicalls, GP is a program invariant!
  if (!Subtarget.isABICalls()) {
    Reserved.set(Mips::GP);
    Reserved.set(Mips::GP_64);
  }

  if (Subtarget.isFP64bit()) {
    // Reserve all registers in AFGR64.
    for (RegIter Reg = Mips::AFGR64RegClass.begin(),
         EReg = Mips::AFGR64RegClass.end(); Reg != EReg; ++Reg)
      Reserved.set(*Reg);
  } else {
    // Reserve all registers in FGR64.
    for (RegIter Reg = Mips::FGR64RegClass.begin(),
         EReg = Mips::FGR64RegClass.end(); Reg != EReg; ++Reg)
      Reserved.set(*Reg);
  }
  // Reserve FP if this function should have a dedicated frame pointer register.
  if (MF.getTarget().getFrameLowering()->hasFP(MF)) {
    if (Subtarget.inMips16Mode())
      Reserved.set(Mips::S0);
    else {
      Reserved.set(Mips::FP);
      Reserved.set(Mips::FP_64);
    }
  }

  // Reserve hardware registers.
  Reserved.set(Mips::HWR29);

  // Reserve DSP control register.
  Reserved.set(Mips::DSPPos);
  Reserved.set(Mips::DSPSCount);
  Reserved.set(Mips::DSPCarry);
  Reserved.set(Mips::DSPEFI);
  Reserved.set(Mips::DSPOutFlag);

  // Reserve MSA control registers.
  Reserved.set(Mips::MSAIR);
  Reserved.set(Mips::MSACSR);
  Reserved.set(Mips::MSAAccess);
  Reserved.set(Mips::MSASave);
  Reserved.set(Mips::MSAModify);
  Reserved.set(Mips::MSARequest);
  Reserved.set(Mips::MSAMap);
  Reserved.set(Mips::MSAUnmap);

  // Reserve RA if in mips16 mode.
  if (Subtarget.inMips16Mode()) {
    const MipsFunctionInfo *MipsFI = MF.getInfo<MipsFunctionInfo>();
    Reserved.set(Mips::RA);
    Reserved.set(Mips::RA_64);
    Reserved.set(Mips::T0);
    Reserved.set(Mips::T1);
    if (MF.getFunction()->hasFnAttribute("saveS2") || MipsFI->hasSaveS2())
      Reserved.set(Mips::S2);
  }

  // Reserve GP if small section is used.
  if (Subtarget.useSmallSection()) {
    Reserved.set(Mips::GP);
    Reserved.set(Mips::GP_64);
  }

  if (Subtarget.isABI_O32() && !Subtarget.useOddSPReg()) {
    for (const auto &Reg : Mips::OddSPRegClass)
      Reserved.set(Reg);
  }

  return Reserved;
}

bool
MipsRegisterInfo::requiresRegisterScavenging(const MachineFunction &MF) const {
  return true;
}

bool
MipsRegisterInfo::trackLivenessAfterRegAlloc(const MachineFunction &MF) const {
  return true;
}

// FrameIndex represent objects inside a abstract stack.
// We must replace FrameIndex with an stack/frame pointer
// direct reference.
void MipsRegisterInfo::
eliminateFrameIndex(MachineBasicBlock::iterator II, int SPAdj,
                    unsigned FIOperandNum, RegScavenger *RS) const {
  MachineInstr &MI = *II;
  MachineFunction &MF = *MI.getParent()->getParent();

  DEBUG(errs() << "\nFunction : " << MF.getName() << "\n";
        errs() << "<--------->\n" << MI);

  int FrameIndex = MI.getOperand(FIOperandNum).getIndex();
  uint64_t stackSize = MF.getFrameInfo()->getStackSize();
  int64_t spOffset = MF.getFrameInfo()->getObjectOffset(FrameIndex);

  DEBUG(errs() << "FrameIndex : " << FrameIndex << "\n"
               << "spOffset   : " << spOffset << "\n"
               << "stackSize  : " << stackSize << "\n");

  eliminateFI(MI, FIOperandNum, FrameIndex, stackSize, spOffset);
}

unsigned MipsRegisterInfo::
getFrameRegister(const MachineFunction &MF) const {
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();
  bool IsN64 = Subtarget.isABI_N64();

  if (Subtarget.inMips16Mode())
    return TFI->hasFP(MF) ? Mips::S0 : Mips::SP;
  else
    return TFI->hasFP(MF) ? (IsN64 ? Mips::FP_64 : Mips::FP) :
                            (IsN64 ? Mips::SP_64 : Mips::SP);

}

