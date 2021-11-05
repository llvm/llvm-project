//===-- CSKYRegisterInfo.h - CSKY Register Information Impl ---*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the CSKY implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "CSKYRegisterInfo.h"
#include "CSKY.h"
#include "CSKYSubtarget.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/MC/MCContext.h"

#define GET_REGINFO_TARGET_DESC
#include "CSKYGenRegisterInfo.inc"

using namespace llvm;

CSKYRegisterInfo::CSKYRegisterInfo()
    : CSKYGenRegisterInfo(CSKY::R15, 0, 0, 0) {}

const uint32_t *
CSKYRegisterInfo::getCallPreservedMask(const MachineFunction &MF,
                                       CallingConv::ID Id) const {
  const CSKYSubtarget &STI = MF.getSubtarget<CSKYSubtarget>();
  return CSR_I32_RegMask;
}

Register CSKYRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  const TargetFrameLowering *TFI = getFrameLowering(MF);
  return TFI->hasFP(MF) ? CSKY::R8 : CSKY::R14;
}

BitVector CSKYRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  const CSKYFrameLowering *TFI = getFrameLowering(MF);
  const CSKYSubtarget &STI = MF.getSubtarget<CSKYSubtarget>();
  BitVector Reserved(getNumRegs());

  // Reserve the base register if we need to allocate
  // variable-sized objects at runtime.
  if (TFI->hasBP(MF))
    markSuperRegs(Reserved, CSKY::R7); // bp

  if (TFI->hasFP(MF))
    markSuperRegs(Reserved, CSKY::R8); // fp

  if (!STI.hasE2()) {
    for (unsigned i = 0; i < 6; i++)
      markSuperRegs(Reserved, CSKY::R8 + i); // R8 - R13
  }

  markSuperRegs(Reserved, CSKY::R14); // sp
  markSuperRegs(Reserved, CSKY::R15); // lr

  if (!STI.hasHighRegisters()) {
    for (unsigned i = 0; i < 10; i++)
      markSuperRegs(Reserved, CSKY::R16 + i); // R16 - R25
  }

  markSuperRegs(Reserved, CSKY::R26);
  markSuperRegs(Reserved, CSKY::R27);
  markSuperRegs(Reserved, CSKY::R28); // gp
  markSuperRegs(Reserved, CSKY::R29);
  markSuperRegs(Reserved, CSKY::R30);
  markSuperRegs(Reserved, CSKY::R31); // tp

  assert(checkAllSuperRegsMarked(Reserved));
  return Reserved;
}

const uint32_t *CSKYRegisterInfo::getNoPreservedMask() const {
  return CSR_NoRegs_RegMask;
}

const MCPhysReg *
CSKYRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  const CSKYSubtarget &STI = MF->getSubtarget<CSKYSubtarget>();
  if (MF->getFunction().hasFnAttribute("interrupt")) {
    return CSR_GPR_ISR_SaveList;
  }

  return CSR_I32_SaveList;
}

void CSKYRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                           int SPAdj, unsigned FIOperandNum,
                                           RegScavenger *RS) const {
  assert(SPAdj == 0 && "Unexpected non-zero SPAdj value");
}