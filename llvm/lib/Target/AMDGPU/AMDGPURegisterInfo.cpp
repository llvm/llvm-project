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

//===----------------------------------------------------------------------===//
// Function handling callbacks - Functions are a seldom used feature of GPUS, so
// they are not supported at this time.
//===----------------------------------------------------------------------===//

// Table of NumRegs sized pieces at every 32-bit offset.
static const uint16_t SubRegFromChannelTable[][32] = {
  { AMDGPU::sub0, AMDGPU::sub1, AMDGPU::sub2, AMDGPU::sub3,
    AMDGPU::sub4, AMDGPU::sub5, AMDGPU::sub6, AMDGPU::sub7,
    AMDGPU::sub8, AMDGPU::sub9, AMDGPU::sub10, AMDGPU::sub11,
    AMDGPU::sub12, AMDGPU::sub13, AMDGPU::sub14, AMDGPU::sub15,
    AMDGPU::sub16, AMDGPU::sub17, AMDGPU::sub18, AMDGPU::sub19,
    AMDGPU::sub20, AMDGPU::sub21, AMDGPU::sub22, AMDGPU::sub23,
    AMDGPU::sub24, AMDGPU::sub25, AMDGPU::sub26, AMDGPU::sub27,
    AMDGPU::sub28, AMDGPU::sub29, AMDGPU::sub30, AMDGPU::sub31
  },
  {
    AMDGPU::sub0_sub1, AMDGPU::sub1_sub2, AMDGPU::sub2_sub3, AMDGPU::sub3_sub4,
    AMDGPU::sub4_sub5, AMDGPU::sub5_sub6, AMDGPU::sub6_sub7, AMDGPU::sub7_sub8,
    AMDGPU::sub8_sub9, AMDGPU::sub9_sub10, AMDGPU::sub10_sub11, AMDGPU::sub11_sub12,
    AMDGPU::sub12_sub13, AMDGPU::sub13_sub14, AMDGPU::sub14_sub15, AMDGPU::sub15_sub16,
    AMDGPU::sub16_sub17, AMDGPU::sub17_sub18, AMDGPU::sub18_sub19, AMDGPU::sub19_sub20,
    AMDGPU::sub20_sub21, AMDGPU::sub21_sub22, AMDGPU::sub22_sub23, AMDGPU::sub23_sub24,
    AMDGPU::sub24_sub25, AMDGPU::sub25_sub26, AMDGPU::sub26_sub27, AMDGPU::sub27_sub28,
    AMDGPU::sub28_sub29, AMDGPU::sub29_sub30, AMDGPU::sub30_sub31, AMDGPU::NoSubRegister
  },
  {
    AMDGPU::sub0_sub1_sub2, AMDGPU::sub1_sub2_sub3, AMDGPU::sub2_sub3_sub4, AMDGPU::sub3_sub4_sub5,
    AMDGPU::sub4_sub5_sub6, AMDGPU::sub5_sub6_sub7, AMDGPU::sub6_sub7_sub8, AMDGPU::sub7_sub8_sub9,
    AMDGPU::sub8_sub9_sub10, AMDGPU::sub9_sub10_sub11, AMDGPU::sub10_sub11_sub12, AMDGPU::sub11_sub12_sub13,
    AMDGPU::sub12_sub13_sub14, AMDGPU::sub13_sub14_sub15, AMDGPU::sub14_sub15_sub16, AMDGPU::sub15_sub16_sub17,
    AMDGPU::sub16_sub17_sub18, AMDGPU::sub17_sub18_sub19, AMDGPU::sub18_sub19_sub20, AMDGPU::sub19_sub20_sub21,
    AMDGPU::sub20_sub21_sub22, AMDGPU::sub21_sub22_sub23, AMDGPU::sub22_sub23_sub24, AMDGPU::sub23_sub24_sub25,
    AMDGPU::sub24_sub25_sub26, AMDGPU::sub25_sub26_sub27, AMDGPU::sub26_sub27_sub28, AMDGPU::sub27_sub28_sub29,
    AMDGPU::sub28_sub29_sub30, AMDGPU::sub29_sub30_sub31, AMDGPU::NoSubRegister, AMDGPU::NoSubRegister
  },
  {
    AMDGPU::sub0_sub1_sub2_sub3, AMDGPU::sub1_sub2_sub3_sub4, AMDGPU::sub2_sub3_sub4_sub5, AMDGPU::sub3_sub4_sub5_sub6,
    AMDGPU::sub4_sub5_sub6_sub7, AMDGPU::sub5_sub6_sub7_sub8, AMDGPU::sub6_sub7_sub8_sub9, AMDGPU::sub7_sub8_sub9_sub10,
    AMDGPU::sub8_sub9_sub10_sub11, AMDGPU::sub9_sub10_sub11_sub12, AMDGPU::sub10_sub11_sub12_sub13, AMDGPU::sub11_sub12_sub13_sub14,
    AMDGPU::sub12_sub13_sub14_sub15, AMDGPU::sub13_sub14_sub15_sub16, AMDGPU::sub14_sub15_sub16_sub17, AMDGPU::sub15_sub16_sub17_sub18,
    AMDGPU::sub16_sub17_sub18_sub19, AMDGPU::sub17_sub18_sub19_sub20, AMDGPU::sub18_sub19_sub20_sub21, AMDGPU::sub19_sub20_sub21_sub22,
    AMDGPU::sub20_sub21_sub22_sub23, AMDGPU::sub21_sub22_sub23_sub24, AMDGPU::sub22_sub23_sub24_sub25, AMDGPU::sub23_sub24_sub25_sub26,
    AMDGPU::sub24_sub25_sub26_sub27, AMDGPU::sub25_sub26_sub27_sub28, AMDGPU::sub26_sub27_sub28_sub29, AMDGPU::sub27_sub28_sub29_sub30,
    AMDGPU::sub28_sub29_sub30_sub31, AMDGPU::NoSubRegister, AMDGPU::NoSubRegister, AMDGPU::NoSubRegister
  }
};

// FIXME: TableGen should generate something to make this manageable for all
// register classes. At a minimum we could use the opposite of
// composeSubRegIndices and go up from the base 32-bit subreg.
unsigned AMDGPURegisterInfo::getSubRegFromChannel(unsigned Channel, unsigned NumRegs) {
  const unsigned NumRegIndex = NumRegs - 1;

  assert(NumRegIndex < array_lengthof(SubRegFromChannelTable) &&
         "Not implemented");
  assert(Channel < array_lengthof(SubRegFromChannelTable[0]));
  return SubRegFromChannelTable[NumRegIndex][Channel];
}

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
