//===-- SIMachineFunctionInfo.cpp -------- SI Machine Function Info -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SIMachineFunctionInfo.h"
#include "AMDGPUSubtarget.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"

#define MAX_LANES 64

using namespace llvm;

SIMachineFunctionInfo::SIMachineFunctionInfo(const MachineFunction &MF)
  : AMDGPUMachineFunction(MF),
    TIDReg(AMDGPU::NoRegister),
    ScratchRSrcReg(AMDGPU::NoRegister),
    ScratchWaveOffsetReg(AMDGPU::NoRegister),
    FrameOffsetReg(AMDGPU::NoRegister),
    StackPtrOffsetReg(AMDGPU::NoRegister),
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
    PSInputAddr(0),
    PSInputEnable(0),
    ReturnsVoid(true),
    FlatWorkGroupSizes(0, 0),
    WavesPerEU(0, 0),
    DebuggerWorkGroupIDStackObjectIndices({{0, 0, 0}}),
    DebuggerWorkItemIDStackObjectIndices({{0, 0, 0}}),
    LDSWaveSpillSize(0),
    NumUserSGPRs(0),
    NumSystemSGPRs(0),
    HasSpilledSGPRs(false),
    HasSpilledVGPRs(false),
    HasNonSpillStackObjects(false),
    NumSpilledSGPRs(0),
    NumSpilledVGPRs(0),
    PrivateSegmentBuffer(false),
    DispatchPtr(false),
    QueuePtr(false),
    KernargSegmentPtr(false),
    DispatchID(false),
    FlatScratchInit(false),
    GridWorkgroupCountX(false),
    GridWorkgroupCountY(false),
    GridWorkgroupCountZ(false),
    WorkGroupIDX(false),
    WorkGroupIDY(false),
    WorkGroupIDZ(false),
    WorkGroupInfo(false),
    PrivateSegmentWaveByteOffset(false),
    WorkItemIDX(false),
    WorkItemIDY(false),
    WorkItemIDZ(false),
    PrivateMemoryInputPtr(false) {
  const SISubtarget &ST = MF.getSubtarget<SISubtarget>();
  const Function *F = MF.getFunction();
  FlatWorkGroupSizes = ST.getFlatWorkGroupSizes(*F);
  WavesPerEU = ST.getWavesPerEU(*F);

  // Non-entry functions have no special inputs for now.
  // TODO: Return early for non-entry CCs.

  CallingConv::ID CC = F->getCallingConv();
  if (CC == CallingConv::AMDGPU_PS)
    PSInputAddr = AMDGPU::getInitialPSInputAddr(*F);

  if (AMDGPU::isKernel(CC)) {
    KernargSegmentPtr = true;
    WorkGroupIDX = true;
    WorkItemIDX = true;
  }

  if (ST.debuggerEmitPrologue()) {
    // Enable everything.
    WorkGroupIDY = true;
    WorkGroupIDZ = true;
    WorkItemIDY = true;
    WorkItemIDZ = true;
  } else {
    if (F->hasFnAttribute("amdgpu-work-group-id-y"))
      WorkGroupIDY = true;

    if (F->hasFnAttribute("amdgpu-work-group-id-z"))
      WorkGroupIDZ = true;

    if (F->hasFnAttribute("amdgpu-work-item-id-y"))
      WorkItemIDY = true;

    if (F->hasFnAttribute("amdgpu-work-item-id-z"))
      WorkItemIDZ = true;
  }

  // X, XY, and XYZ are the only supported combinations, so make sure Y is
  // enabled if Z is.
  if (WorkItemIDZ)
    WorkItemIDY = true;

  const MachineFrameInfo &FrameInfo = MF.getFrameInfo();
  bool MaySpill = ST.isVGPRSpillingEnabled(*F);
  bool HasStackObjects = FrameInfo.hasStackObjects();

  if (HasStackObjects || MaySpill)
    PrivateSegmentWaveByteOffset = true;

  if (ST.isAmdCodeObjectV2(MF)) {
    if (HasStackObjects || MaySpill)
      PrivateSegmentBuffer = true;

    if (F->hasFnAttribute("amdgpu-dispatch-ptr"))
      DispatchPtr = true;

    if (F->hasFnAttribute("amdgpu-queue-ptr"))
      QueuePtr = true;

    if (F->hasFnAttribute("amdgpu-dispatch-id"))
      DispatchID = true;
  } else if (ST.isMesaGfxShader(MF)) {
    if (HasStackObjects || MaySpill)
      PrivateMemoryInputPtr = true;
  }

  // We don't need to worry about accessing spills with flat instructions.
  // TODO: On VI where we must use flat for global, we should be able to omit
  // this if it is never used for generic access.
  if (HasStackObjects && ST.hasFlatAddressSpace() && ST.isAmdHsaOS())
    FlatScratchInit = true;
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

unsigned SIMachineFunctionInfo::addDispatchID(const SIRegisterInfo &TRI) {
  DispatchIDUserSGPR = TRI.getMatchingSuperReg(
    getNextUserSGPR(), AMDGPU::sub0, &AMDGPU::SReg_64RegClass);
  NumUserSGPRs += 2;
  return DispatchIDUserSGPR;
}

unsigned SIMachineFunctionInfo::addFlatScratchInit(const SIRegisterInfo &TRI) {
  FlatScratchInitUserSGPR = TRI.getMatchingSuperReg(
    getNextUserSGPR(), AMDGPU::sub0, &AMDGPU::SReg_64RegClass);
  NumUserSGPRs += 2;
  return FlatScratchInitUserSGPR;
}

unsigned SIMachineFunctionInfo::addPrivateMemoryPtr(const SIRegisterInfo &TRI) {
  PrivateMemoryPtrUserSGPR = TRI.getMatchingSuperReg(
    getNextUserSGPR(), AMDGPU::sub0, &AMDGPU::SReg_64RegClass);
  NumUserSGPRs += 2;
  return PrivateMemoryPtrUserSGPR;
}

/// Reserve a slice of a VGPR to support spilling for FrameIndex \p FI.
bool SIMachineFunctionInfo::allocateSGPRSpillToVGPR(MachineFunction &MF,
                                                    int FI) {
  std::vector<SpilledReg> &SpillLanes = SGPRToVGPRSpills[FI];

  // This has already been allocated.
  if (!SpillLanes.empty())
    return true;

  const SISubtarget &ST = MF.getSubtarget<SISubtarget>();
  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  MachineFrameInfo &FrameInfo = MF.getFrameInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  unsigned WaveSize = ST.getWavefrontSize();

  unsigned Size = FrameInfo.getObjectSize(FI);
  assert(Size >= 4 && Size <= 64 && "invalid sgpr spill size");
  assert(TRI->spillSGPRToVGPR() && "not spilling SGPRs to VGPRs");

  int NumLanes = Size / 4;

  // Make sure to handle the case where a wide SGPR spill may span between two
  // VGPRs.
  for (int I = 0; I < NumLanes; ++I, ++NumVGPRSpillLanes) {
    unsigned LaneVGPR;
    unsigned VGPRIndex = (NumVGPRSpillLanes % WaveSize);

    if (VGPRIndex == 0) {
      LaneVGPR = TRI->findUnusedRegister(MRI, &AMDGPU::VGPR_32RegClass, MF);
      if (LaneVGPR == AMDGPU::NoRegister) {
        // We have no VGPRs left for spilling SGPRs. Reset because we won't
        // partially spill the SGPR to VGPRs.
        SGPRToVGPRSpills.erase(FI);
        NumVGPRSpillLanes -= I;
        return false;
      }

      SpillVGPRs.push_back(LaneVGPR);

      // Add this register as live-in to all blocks to avoid machine verifer
      // complaining about use of an undefined physical register.
      for (MachineBasicBlock &BB : MF)
        BB.addLiveIn(LaneVGPR);
    } else {
      LaneVGPR = SpillVGPRs.back();
    }

    SpillLanes.push_back(SpilledReg(LaneVGPR, VGPRIndex));
  }

  return true;
}

void SIMachineFunctionInfo::removeSGPRToVGPRFrameIndices(MachineFrameInfo &MFI) {
  for (auto &R : SGPRToVGPRSpills)
    MFI.RemoveStackObject(R.first);
}
