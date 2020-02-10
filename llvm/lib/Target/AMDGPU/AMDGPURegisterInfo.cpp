//===-- AMDGPURegisterInfo.cpp - AMDGPU Register Information -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Parent TargetRegisterInfo class common to all hw codegen targets.
//
//===----------------------------------------------------------------------===//

#include "AMDGPURegisterInfo.h"
#include "AMDGPUTargetMachine.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"

using namespace llvm;

AMDGPURegisterInfo::AMDGPURegisterInfo() : AMDGPUGenRegisterInfo(0) {}

void AMDGPURegisterInfo::reserveRegisterTuples(BitVector &Reserved, unsigned Reg) const {
  MCRegAliasIterator R(Reg, this, true);

  for (; R.isValid(); ++R)
    Reserved.set(*R);
}

#define GET_REGINFO_TARGET_DESC
#include "AMDGPUGenRegisterInfo.inc"

// Forced to be here by one .inc
const MCPhysReg *SIRegisterInfo::getCalleeSavedRegs(
  const MachineFunction *MF) const {
  CallingConv::ID CC = MF->getFunction().getCallingConv();
  switch (CC) {
  case CallingConv::C:
  case CallingConv::Fast:
  case CallingConv::Cold:
    return CSR_AMDGPU_HighRegs_SaveList;
  default: {
    // Dummy to not crash RegisterClassInfo.
    static const MCPhysReg NoCalleeSavedReg = AMDGPU::NoRegister;
    return &NoCalleeSavedReg;
  }
  }
}

const MCPhysReg *
SIRegisterInfo::getCalleeSavedRegsViaCopy(const MachineFunction *MF) const {
  return nullptr;
}

const uint32_t *SIRegisterInfo::getCallPreservedMask(const MachineFunction &MF,
                                                     CallingConv::ID CC) const {
  switch (CC) {
  case CallingConv::C:
  case CallingConv::Fast:
  case CallingConv::Cold:
    return CSR_AMDGPU_HighRegs_RegMask;
  default:
    return nullptr;
  }
}

Register SIRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  const SIFrameLowering *TFI =
      MF.getSubtarget<GCNSubtarget>().getFrameLowering();
  const SIMachineFunctionInfo *FuncInfo = MF.getInfo<SIMachineFunctionInfo>();
  return TFI->hasFP(MF) ? FuncInfo->getFrameOffsetReg()
                        : FuncInfo->getStackPtrOffsetReg();
}

const uint32_t *SIRegisterInfo::getAllVGPRRegMask() const {
  return CSR_AMDGPU_AllVGPRs_RegMask;
}

const uint32_t *SIRegisterInfo::getAllAllocatableSRegMask() const {
  return CSR_AMDGPU_AllAllocatableSRegs_RegMask;
}
