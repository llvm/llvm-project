//===-- AGXRegisterInfo.cpp - AGX register information --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AGXRegisterInfo.h"
#include "AGX.h"
#include "AGXFrameLowering.h"
#include "AGXSubtarget.h"
#include "MCTargetDesc/AGXMCTargetDesc.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

#define GET_REGINFO_TARGET_DESC
#include "AGXGenRegisterInfo.inc"

AGXRegisterInfo::AGXRegisterInfo(const AGXSubtarget &ST)
    : AGXGenRegisterInfo(AGX::R1), ST(ST) {}

const MCPhysReg *
AGXRegisterInfo::getCalleeSavedRegs(const MachineFunction *) const {
  return CSR_AGX_NoRegs_SaveList;
}

const uint32_t *AGXRegisterInfo::getCallPreservedMask(const MachineFunction &,
                                                      CallingConv::ID) const {
  return CSR_AGX_NoRegs_RegMask;
}

BitVector AGXRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  auto ReserveWithAliases = [&](MCRegister Reg) {
    for (MCRegAliasIterator AI(Reg, this, /*IncludeSelf=*/true); AI.isValid();
         ++AI)
      Reserved.set((*AI).id());
  };

  // r0l is the execution mask nesting counter. r0h is left alone so that
  // r0 is never clobbered as a whole.
  ReserveWithAliases(AGX::R0);
  bool UsesCallABI = MF.getFunction().getCallingConv() != CallingConv::AGX_KERNEL;
  if (!UsesCallABI)
    for (const BasicBlock &BB : MF.getFunction())
      for (const Instruction &I : BB)
        if (const auto *CB = dyn_cast<CallBase>(&I)) {
          const Function *Callee = CB->getCalledFunction();
          UsesCallABI |= !Callee || !Callee->isIntrinsic();
        }
  if (agxExperimentalLeafCallsEnabled() && UsesCallABI) {
    ReserveWithAliases(AGX::R1);
    for (unsigned I = 2; I < 30; ++I)
      ReserveWithAliases(AGX::GPR32RegClass.getRegister(I));
    for (unsigned I = 30; I <= 36; ++I)
      ReserveWithAliases(AGX::GPR32RegClass.getRegister(I));
  }

  // Registers beyond the subtarget's register file.
  for (unsigned I = ST.getMaxGPRs(MF.getFunction()); I < 128; ++I)
    ReserveWithAliases(AGX::GPR32RegClass.getRegister(I));

  // Uniform registers are never allocated.
  for (MCRegister Reg : AGX::UGPR16RegClass)
    ReserveWithAliases(Reg);
  return Reserved;
}

bool AGXRegisterInfo::isUniformClass(const TargetRegisterClass *RC) {
  return AGX::UGPR16RegClass.hasSubClassEq(RC) ||
         AGX::UGPR32RegClass.hasSubClassEq(RC) ||
         AGX::UGPR64RegClass.hasSubClassEq(RC);
}

const TargetRegisterClass *
AGXRegisterInfo::getGPRClassForSize(unsigned Bits) const {
  switch (Bits) {
  case 16:
    return &AGX::GPR16RegClass;
  case 32:
    return &AGX::GPR32RegClass;
  case 64:
    return &AGX::GPR32x2RegClass;
  case 96:
    return &AGX::GPR96RegClass;
  case 128:
    return &AGX::GPR128RegClass;
  default:
    return nullptr;
  }
}

Register AGXRegisterInfo::getFrameRegister(const MachineFunction &) const {
  // Stack accesses address the thread's stack directly; there is no frame
  // register.
  return Register();
}

bool AGXRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                          int SPAdj, unsigned FIOperandNum,
                                          RegScavenger *RS) const {
  MachineInstr &MI = *II;
  MachineFunction &MF = *MI.getMF();
  MachineOperand &FIOp = MI.getOperand(FIOperandNum);
  int FI = FIOp.getIndex();
  int64_t Offset = MF.getFrameInfo().getObjectOffset(FI);
  if (agxExperimentalLeafCallsEnabled() &&
      MF.getFunction().getCallingConv() != CallingConv::AGX_KERNEL) {
    // After the kernel's region, one per depth (depth 0: an entry point of a
    // module without kernels, at offset zero)
    Offset += AGXFrameLowering::getCalleeFrameOffset(MF);
  }
  if (Offset < 0 || Offset > 0x7fff)
    report_fatal_error("AGX stack frame too large");
  FIOp.ChangeToImmediate(Offset);
  return false;
}
