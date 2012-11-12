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

#define DEBUG_TYPE "mips-reg-info"

#include "MipsRegisterInfo.h"
#include "Mips.h"
#include "MipsAnalyzeImmediate.h"
#include "MipsInstrInfo.h"
#include "MipsSubtarget.h"
#include "MipsMachineFunction.h"
#include "llvm/Constants.h"
#include "llvm/DebugInfo.h"
#include "llvm/Type.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"

#define GET_REGINFO_TARGET_DESC
#include "MipsGenRegisterInfo.inc"

using namespace llvm;

MipsRegisterInfo::MipsRegisterInfo(const MipsSubtarget &ST)
  : MipsGenRegisterInfo(Mips::RA), Subtarget(ST) {}

unsigned MipsRegisterInfo::getPICCallReg() { return Mips::T9; }

//===----------------------------------------------------------------------===//
// Callee Saved Registers methods
//===----------------------------------------------------------------------===//

/// Mips Callee Saved Registers
const uint16_t* MipsRegisterInfo::
getCalleeSavedRegs(const MachineFunction *MF) const {
  if (Subtarget.isSingleFloat())
    return CSR_SingleFloatOnly_SaveList;
  else if (!Subtarget.hasMips64())
    return CSR_O32_SaveList;
  else if (Subtarget.isABI_N32())
    return CSR_N32_SaveList;

  assert(Subtarget.isABI_N64());
  return CSR_N64_SaveList;
}

const uint32_t*
MipsRegisterInfo::getCallPreservedMask(CallingConv::ID) const {
  if (Subtarget.isSingleFloat())
    return CSR_SingleFloatOnly_RegMask;
  else if (!Subtarget.hasMips64())
    return CSR_O32_RegMask;
  else if (Subtarget.isABI_N32())
    return CSR_N32_RegMask;

  assert(Subtarget.isABI_N64());
  return CSR_N64_RegMask;
}

BitVector MipsRegisterInfo::
getReservedRegs(const MachineFunction &MF) const {
  static const uint16_t ReservedCPURegs[] = {
    Mips::ZERO, Mips::K0, Mips::K1, Mips::SP
  };

  static const uint16_t ReservedCPU64Regs[] = {
    Mips::ZERO_64, Mips::K0_64, Mips::K1_64, Mips::SP_64
  };

  BitVector Reserved(getNumRegs());
  typedef TargetRegisterClass::const_iterator RegIter;

  for (unsigned I = 0; I < array_lengthof(ReservedCPURegs); ++I)
    Reserved.set(ReservedCPURegs[I]);

  for (unsigned I = 0; I < array_lengthof(ReservedCPU64Regs); ++I)
    Reserved.set(ReservedCPU64Regs[I]);

  if (Subtarget.hasMips64()) {
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
  Reserved.set(Mips::HWR29_64);

  // Reserve DSP control register.
  Reserved.set(Mips::DSPCtrl);

  // Reserve RA if in mips16 mode.
  if (Subtarget.inMips16Mode()) {
    Reserved.set(Mips::RA);
    Reserved.set(Mips::RA_64);
  }

  // Reserve GP if small section is used.
  if (Subtarget.useSmallSection()) {
    Reserved.set(Mips::GP);
    Reserved.set(Mips::GP_64);
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
                    RegScavenger *RS) const {
  MachineInstr &MI = *II;
  MachineFunction &MF = *MI.getParent()->getParent();

  unsigned i = 0;
  while (!MI.getOperand(i).isFI()) {
    ++i;
    assert(i < MI.getNumOperands() &&
           "Instr doesn't have FrameIndex operand!");
  }

  DEBUG(errs() << "\nFunction : " << MF.getName() << "\n";
        errs() << "<--------->\n" << MI);

  int FrameIndex = MI.getOperand(i).getIndex();
  uint64_t stackSize = MF.getFrameInfo()->getStackSize();
  int64_t spOffset = MF.getFrameInfo()->getObjectOffset(FrameIndex);

  DEBUG(errs() << "FrameIndex : " << FrameIndex << "\n"
               << "spOffset   : " << spOffset << "\n"
               << "stackSize  : " << stackSize << "\n");

  eliminateFI(MI, i, FrameIndex, stackSize, spOffset);
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

unsigned MipsRegisterInfo::
getEHExceptionRegister() const {
  llvm_unreachable("What is the exception register");
}

unsigned MipsRegisterInfo::
getEHHandlerRegister() const {
  llvm_unreachable("What is the exception handler register");
}
