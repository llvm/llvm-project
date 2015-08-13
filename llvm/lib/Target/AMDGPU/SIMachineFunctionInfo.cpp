//===-- SIMachineFunctionInfo.cpp - SI Machine Function Info -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
/// \file
//===----------------------------------------------------------------------===//


#include "SIMachineFunctionInfo.h"
#include "AMDGPUSubtarget.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"

#define MAX_LANES 64

using namespace llvm;


// Pin the vtable to this file.
void SIMachineFunctionInfo::anchor() {}

SIMachineFunctionInfo::SIMachineFunctionInfo(const MachineFunction &MF)
  : AMDGPUMachineFunction(MF),
    TIDReg(AMDGPU::NoRegister),
    HasSpilledVGPRs(false),
    PSInputAddr(0),
    NumUserSGPRs(0),
    LDSWaveSpillSize(0) { }

SIMachineFunctionInfo::SpilledReg SIMachineFunctionInfo::getSpilledReg(
                                                       MachineFunction *MF,
                                                       unsigned FrameIndex,
                                                       unsigned SubIdx) {
  const MachineFrameInfo *FrameInfo = MF->getFrameInfo();
  const SIRegisterInfo *TRI = static_cast<const SIRegisterInfo *>(
      MF->getSubtarget<AMDGPUSubtarget>().getRegisterInfo());
  MachineRegisterInfo &MRI = MF->getRegInfo();
  int64_t Offset = FrameInfo->getObjectOffset(FrameIndex);
  Offset += SubIdx * 4;

  unsigned LaneVGPRIdx = Offset / (64 * 4);
  unsigned Lane = (Offset / 4) % 64;

  struct SpilledReg Spill;

  if (!LaneVGPRs.count(LaneVGPRIdx)) {
    unsigned LaneVGPR = TRI->findUnusedRegister(MRI, &AMDGPU::VGPR_32RegClass);
    LaneVGPRs[LaneVGPRIdx] = LaneVGPR;
    MRI.setPhysRegUsed(LaneVGPR);

    // Add this register as live-in to all blocks to avoid machine verifer
    // complaining about use of an undefined physical register.
    for (MachineFunction::iterator BI = MF->begin(), BE = MF->end();
         BI != BE; ++BI) {
      BI->addLiveIn(LaneVGPR);
    }
  }

  Spill.VGPR = LaneVGPRs[LaneVGPRIdx];
  Spill.Lane = Lane;
  return Spill;
}

unsigned SIMachineFunctionInfo::getMaximumWorkGroupSize(
                                              const MachineFunction &MF) const {
  const AMDGPUSubtarget &ST = MF.getSubtarget<AMDGPUSubtarget>();
  // FIXME: We should get this information from kernel attributes if it
  // is available.
  return getShaderType() == ShaderType::COMPUTE ? 256 : ST.getWavefrontSize();
}
