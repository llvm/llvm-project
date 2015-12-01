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
    ScratchRSrcReg(AMDGPU::NoRegister),
    ScratchWaveOffsetReg(AMDGPU::NoRegister),
    PrivateSegmentBufferUserSGPR(AMDGPU::NoRegister),
    DispatchPtrUserSGPR(AMDGPU::NoRegister),
    QueuePtrUserSGPR(AMDGPU::NoRegister),
    KernargSegmentPtrUserSGPR(AMDGPU::NoRegister),
    DispatchIDUserSGPR(AMDGPU::NoRegister),
    FlatScratchInitUserSGPR(AMDGPU::NoRegister),
    PrivateSegmentSizeUserSGPR(AMDGPU::NoRegister),
    GridWorkGroupCountXUserSGPR(AMDGPU::NoRegister),
    GridWorkGroupCountYUserSGPR(AMDGPU::NoRegister),
    GridWorkGroupCountZUserSGPR(AMDGPU::NoRegister),
    WorkGroupIDXSystemSGPR(AMDGPU::NoRegister),
    WorkGroupIDYSystemSGPR(AMDGPU::NoRegister),
    WorkGroupIDZSystemSGPR(AMDGPU::NoRegister),
    WorkGroupInfoSystemSGPR(AMDGPU::NoRegister),
    PrivateSegmentWaveByteOffsetSystemSGPR(AMDGPU::NoRegister),
    LDSWaveSpillSize(0),
    PSInputAddr(0),
    NumUserSGPRs(0),
    NumSystemSGPRs(0),
    HasSpilledSGPRs(false),
    HasSpilledVGPRs(false),
    PrivateSegmentBuffer(false),
    DispatchPtr(false),
    QueuePtr(false),
    DispatchID(false),
    KernargSegmentPtr(false),
    FlatScratchInit(false),
    GridWorkgroupCountX(false),
    GridWorkgroupCountY(false),
    GridWorkgroupCountZ(false),
    WorkGroupIDX(true),
    WorkGroupIDY(false),
    WorkGroupIDZ(false),
    WorkGroupInfo(false),
    PrivateSegmentWaveByteOffset(false),
    WorkItemIDX(true),
    WorkItemIDY(false),
    WorkItemIDZ(false) {
  const AMDGPUSubtarget &ST = MF.getSubtarget<AMDGPUSubtarget>();
  const Function *F = MF.getFunction();

  const MachineFrameInfo *FrameInfo = MF.getFrameInfo();

  if (getShaderType() == ShaderType::COMPUTE)
    KernargSegmentPtr = true;

  if (F->hasFnAttribute("amdgpu-work-group-id-y"))
    WorkGroupIDY = true;

  if (F->hasFnAttribute("amdgpu-work-group-id-z"))
    WorkGroupIDZ = true;

  if (F->hasFnAttribute("amdgpu-work-item-id-y"))
    WorkItemIDY = true;

  if (F->hasFnAttribute("amdgpu-work-item-id-z"))
    WorkItemIDZ = true;

  bool MaySpill = ST.isVGPRSpillingEnabled(this);
  bool HasStackObjects = FrameInfo->hasStackObjects();

  if (HasStackObjects || MaySpill)
    PrivateSegmentWaveByteOffset = true;

  if (ST.isAmdHsaOS()) {
    if (HasStackObjects || MaySpill)
      PrivateSegmentBuffer = true;

    if (F->hasFnAttribute("amdgpu-dispatch-ptr"))
      DispatchPtr = true;
  }

  // X, XY, and XYZ are the only supported combinations, so make sure Y is
  // enabled if Z is.
  if (WorkItemIDZ)
    WorkItemIDY = true;
}

unsigned SIMachineFunctionInfo::addPrivateSegmentBuffer(
  const SIRegisterInfo &TRI) {
  PrivateSegmentBufferUserSGPR = TRI.getMatchingSuperReg(
    getNextUserSGPR(), AMDGPU::sub0, &AMDGPU::SReg_128RegClass);
  NumUserSGPRs += 4;
  return PrivateSegmentBufferUserSGPR;
}

unsigned SIMachineFunctionInfo::addDispatchPtr(const SIRegisterInfo &TRI) {
  DispatchPtrUserSGPR = TRI.getMatchingSuperReg(
    getNextUserSGPR(), AMDGPU::sub0, &AMDGPU::SReg_64RegClass);
  NumUserSGPRs += 2;
  return DispatchPtrUserSGPR;
}

unsigned SIMachineFunctionInfo::addQueuePtr(const SIRegisterInfo &TRI) {
  QueuePtrUserSGPR = TRI.getMatchingSuperReg(
    getNextUserSGPR(), AMDGPU::sub0, &AMDGPU::SReg_64RegClass);
  NumUserSGPRs += 2;
  return QueuePtrUserSGPR;
}

unsigned SIMachineFunctionInfo::addKernargSegmentPtr(const SIRegisterInfo &TRI) {
  KernargSegmentPtrUserSGPR = TRI.getMatchingSuperReg(
    getNextUserSGPR(), AMDGPU::sub0, &AMDGPU::SReg_64RegClass);
  NumUserSGPRs += 2;
  return KernargSegmentPtrUserSGPR;
}

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
