//===- SIMachineFunctionInfo.cpp - SI Machine Function Info ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SIMachineFunctionInfo.h"
#include "AMDGPUArgumentUsageInfo.h"
#include "AMDGPUTargetMachine.h"
#include "AMDGPUSubtarget.h"
#include "SIRegisterInfo.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/Optional.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Function.h"
#include <cassert>
#include <vector>

#define MAX_LANES 64

using namespace llvm;

SIMachineFunctionInfo::SIMachineFunctionInfo(const MachineFunction &MF)
  : AMDGPUMachineFunction(MF),
    PrivateSegmentBuffer(false),
    DispatchPtr(false),
    QueuePtr(false),
    KernargSegmentPtr(false),
    DispatchID(false),
    FlatScratchInit(false),
    WorkGroupIDX(false),
    WorkGroupIDY(false),
    WorkGroupIDZ(false),
    WorkGroupInfo(false),
    PrivateSegmentWaveByteOffset(false),
    WorkItemIDX(false),
    WorkItemIDY(false),
    WorkItemIDZ(false),
    ImplicitBufferPtr(false),
    ImplicitArgPtr(false),
    GITPtrHigh(0xffffffff),
    HighBitsOf32BitAddress(0),
    GDSSize(0) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const Function &F = MF.getFunction();
  FlatWorkGroupSizes = ST.getFlatWorkGroupSizes(F);
  WavesPerEU = ST.getWavesPerEU(F);

  Occupancy = ST.computeOccupancy(MF, getLDSSize());
  CallingConv::ID CC = F.getCallingConv();
  const MachineFrameInfo &FrameInfo = MF.getFrameInfo();

  // FIXME: Should have analysis or something rather than attribute to detect
  // calls.
  const bool HasCalls = FrameInfo.hasCalls() || F.hasFnAttribute("amdgpu-calls");

  // Enable all kernel inputs if we have the fixed ABI. Don't bother if we don't
  // have any calls.
  const bool UseFixedABI = AMDGPUTargetMachine::EnableFixedFunctionABI &&
                           (!isEntryFunction() || HasCalls);

  if (CC == CallingConv::AMDGPU_KERNEL || CC == CallingConv::SPIR_KERNEL) {
    if (!F.arg_empty())
      KernargSegmentPtr = true;
    WorkGroupIDX = true;
    WorkItemIDX = true;
  } else if (CC == CallingConv::AMDGPU_PS) {
    PSInputAddr = AMDGPU::getInitialPSInputAddr(F);
  }

  if (!isEntryFunction()) {
    // Non-entry functions have no special inputs for now, other registers
    // required for scratch access.
    ScratchRSrcReg = AMDGPU::SGPR0_SGPR1_SGPR2_SGPR3;

    // TODO: Pick a high register, and shift down, similar to a kernel.
    FrameOffsetReg = AMDGPU::SGPR33;
    StackPtrOffsetReg = AMDGPU::SGPR32;

    ArgInfo.PrivateSegmentBuffer =
      ArgDescriptor::createRegister(ScratchRSrcReg);

    if (F.hasFnAttribute("amdgpu-implicitarg-ptr"))
      ImplicitArgPtr = true;
  } else {
    if (F.hasFnAttribute("amdgpu-implicitarg-ptr")) {
      KernargSegmentPtr = true;
      MaxKernArgAlign = std::max(ST.getAlignmentForImplicitArgPtr(),
                                 MaxKernArgAlign);
    }
  }

  if (UseFixedABI) {
    WorkGroupIDX = true;
    WorkGroupIDY = true;
    WorkGroupIDZ = true;
    WorkItemIDX = true;
    WorkItemIDY = true;
    WorkItemIDZ = true;
    ImplicitArgPtr = true;
  } else {
    if (F.hasFnAttribute("amdgpu-work-group-id-x"))
      WorkGroupIDX = true;

    if (F.hasFnAttribute("amdgpu-work-group-id-y"))
      WorkGroupIDY = true;

    if (F.hasFnAttribute("amdgpu-work-group-id-z"))
      WorkGroupIDZ = true;

    if (F.hasFnAttribute("amdgpu-work-item-id-x"))
      WorkItemIDX = true;

    if (F.hasFnAttribute("amdgpu-work-item-id-y"))
      WorkItemIDY = true;

    if (F.hasFnAttribute("amdgpu-work-item-id-z"))
      WorkItemIDZ = true;
  }

  bool HasStackObjects = FrameInfo.hasStackObjects();

  if (isEntryFunction()) {
    // X, XY, and XYZ are the only supported combinations, so make sure Y is
    // enabled if Z is.
    if (WorkItemIDZ)
      WorkItemIDY = true;

    PrivateSegmentWaveByteOffset = true;

    // HS and GS always have the scratch wave offset in SGPR5 on GFX9.
    if (ST.getGeneration() >= AMDGPUSubtarget::GFX9 &&
        (CC == CallingConv::AMDGPU_HS || CC == CallingConv::AMDGPU_GS))
      ArgInfo.PrivateSegmentWaveByteOffset =
          ArgDescriptor::createRegister(AMDGPU::SGPR5);
  }

  bool isAmdHsaOrMesa = ST.isAmdHsaOrMesa(F);
  if (isAmdHsaOrMesa) {
    PrivateSegmentBuffer = true;

    if (UseFixedABI) {
      DispatchPtr = true;
      QueuePtr = true;

      // FIXME: We don't need this?
      DispatchID = true;
    } else {
      if (F.hasFnAttribute("amdgpu-dispatch-ptr"))
        DispatchPtr = true;

      if (F.hasFnAttribute("amdgpu-queue-ptr"))
        QueuePtr = true;

      if (F.hasFnAttribute("amdgpu-dispatch-id"))
        DispatchID = true;
    }
  } else if (ST.isMesaGfxShader(F)) {
    ImplicitBufferPtr = true;
  }

  if (UseFixedABI || F.hasFnAttribute("amdgpu-kernarg-segment-ptr"))
    KernargSegmentPtr = true;

  if (ST.hasFlatAddressSpace() && isEntryFunction() && isAmdHsaOrMesa) {
    auto hasNonSpillStackObjects = [&]() {
      // Avoid expensive checking if there's no stack objects.
      if (!HasStackObjects)
        return false;
      for (auto OI = FrameInfo.getObjectIndexBegin(),
                OE = FrameInfo.getObjectIndexEnd(); OI != OE; ++OI)
        if (!FrameInfo.isSpillSlotObjectIndex(OI))
          return true;
      // All stack objects are spill slots.
      return false;
    };
    // TODO: This could be refined a lot. The attribute is a poor way of
    // detecting calls that may require it before argument lowering.
    if (HasCalls || hasNonSpillStackObjects())
      FlatScratchInit = true;
  }

  Attribute A = F.getFnAttribute("amdgpu-git-ptr-high");
  StringRef S = A.getValueAsString();
  if (!S.empty())
    S.consumeInteger(0, GITPtrHigh);

  A = F.getFnAttribute("amdgpu-32bit-address-high-bits");
  S = A.getValueAsString();
  if (!S.empty())
    S.consumeInteger(0, HighBitsOf32BitAddress);

  S = F.getFnAttribute("amdgpu-gds-size").getValueAsString();
  if (!S.empty())
    S.consumeInteger(0, GDSSize);
}

void SIMachineFunctionInfo::limitOccupancy(const MachineFunction &MF) {
  limitOccupancy(getMaxWavesPerEU());
  const GCNSubtarget& ST = MF.getSubtarget<GCNSubtarget>();
  limitOccupancy(ST.getOccupancyWithLocalMemSize(getLDSSize(),
                 MF.getFunction()));
}

Register SIMachineFunctionInfo::addPrivateSegmentBuffer(
  const SIRegisterInfo &TRI) {
  ArgInfo.PrivateSegmentBuffer =
    ArgDescriptor::createRegister(TRI.getMatchingSuperReg(
    getNextUserSGPR(), AMDGPU::sub0, &AMDGPU::SGPR_128RegClass));
  NumUserSGPRs += 4;
  return ArgInfo.PrivateSegmentBuffer.getRegister();
}

Register SIMachineFunctionInfo::addDispatchPtr(const SIRegisterInfo &TRI) {
  ArgInfo.DispatchPtr = ArgDescriptor::createRegister(TRI.getMatchingSuperReg(
    getNextUserSGPR(), AMDGPU::sub0, &AMDGPU::SReg_64RegClass));
  NumUserSGPRs += 2;
  return ArgInfo.DispatchPtr.getRegister();
}

Register SIMachineFunctionInfo::addQueuePtr(const SIRegisterInfo &TRI) {
  ArgInfo.QueuePtr = ArgDescriptor::createRegister(TRI.getMatchingSuperReg(
    getNextUserSGPR(), AMDGPU::sub0, &AMDGPU::SReg_64RegClass));
  NumUserSGPRs += 2;
  return ArgInfo.QueuePtr.getRegister();
}

Register SIMachineFunctionInfo::addKernargSegmentPtr(const SIRegisterInfo &TRI) {
  ArgInfo.KernargSegmentPtr
    = ArgDescriptor::createRegister(TRI.getMatchingSuperReg(
    getNextUserSGPR(), AMDGPU::sub0, &AMDGPU::SReg_64RegClass));
  NumUserSGPRs += 2;
  return ArgInfo.KernargSegmentPtr.getRegister();
}

Register SIMachineFunctionInfo::addDispatchID(const SIRegisterInfo &TRI) {
  ArgInfo.DispatchID = ArgDescriptor::createRegister(TRI.getMatchingSuperReg(
    getNextUserSGPR(), AMDGPU::sub0, &AMDGPU::SReg_64RegClass));
  NumUserSGPRs += 2;
  return ArgInfo.DispatchID.getRegister();
}

Register SIMachineFunctionInfo::addFlatScratchInit(const SIRegisterInfo &TRI) {
  ArgInfo.FlatScratchInit = ArgDescriptor::createRegister(TRI.getMatchingSuperReg(
    getNextUserSGPR(), AMDGPU::sub0, &AMDGPU::SReg_64RegClass));
  NumUserSGPRs += 2;
  return ArgInfo.FlatScratchInit.getRegister();
}

Register SIMachineFunctionInfo::addImplicitBufferPtr(const SIRegisterInfo &TRI) {
  ArgInfo.ImplicitBufferPtr = ArgDescriptor::createRegister(TRI.getMatchingSuperReg(
    getNextUserSGPR(), AMDGPU::sub0, &AMDGPU::SReg_64RegClass));
  NumUserSGPRs += 2;
  return ArgInfo.ImplicitBufferPtr.getRegister();
}

static bool isCalleeSavedReg(const MCPhysReg *CSRegs, MCPhysReg Reg) {
  for (unsigned I = 0; CSRegs[I]; ++I) {
    if (CSRegs[I] == Reg)
      return true;
  }

  return false;
}

/// \p returns true if \p NumLanes slots are available in VGPRs already used for
/// SGPR spilling.
//
// FIXME: This only works after processFunctionBeforeFrameFinalized
bool SIMachineFunctionInfo::haveFreeLanesForSGPRSpill(const MachineFunction &MF,
                                                      unsigned NumNeed) const {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  unsigned WaveSize = ST.getWavefrontSize();
  return NumVGPRSpillLanes + NumNeed <= WaveSize * SpillVGPRs.size();
}

/// Reserve a slice of a VGPR to support spilling for FrameIndex \p FI.
bool SIMachineFunctionInfo::allocateSGPRSpillToVGPR(MachineFunction &MF,
                                                    int FI) {
  std::vector<SpilledReg> &SpillLanes = SGPRToVGPRSpills[FI];

  // This has already been allocated.
  if (!SpillLanes.empty())
    return true;

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  MachineFrameInfo &FrameInfo = MF.getFrameInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  unsigned WaveSize = ST.getWavefrontSize();

  unsigned Size = FrameInfo.getObjectSize(FI);
  assert(Size >= 4 && Size <= 64 && "invalid sgpr spill size");
  assert(TRI->spillSGPRToVGPR() && "not spilling SGPRs to VGPRs");

  int NumLanes = Size / 4;

  const MCPhysReg *CSRegs = MRI.getCalleeSavedRegs();

  // Make sure to handle the case where a wide SGPR spill may span between two
  // VGPRs.
  for (int I = 0; I < NumLanes; ++I, ++NumVGPRSpillLanes) {
    Register LaneVGPR;
    unsigned VGPRIndex = (NumVGPRSpillLanes % WaveSize);

    if (VGPRIndex == 0) {
      LaneVGPR = TRI->findUnusedRegister(MRI, &AMDGPU::VGPR_32RegClass, MF);
      if (LaneVGPR == AMDGPU::NoRegister) {
        // We have no VGPRs left for spilling SGPRs. Reset because we will not
        // partially spill the SGPR to VGPRs.
        SGPRToVGPRSpills.erase(FI);
        NumVGPRSpillLanes -= I;
        return false;
      }

      Optional<int> CSRSpillFI;
      if ((FrameInfo.hasCalls() || !isEntryFunction()) && CSRegs &&
          isCalleeSavedReg(CSRegs, LaneVGPR)) {
        CSRSpillFI = FrameInfo.CreateSpillStackObject(4, 4);
      }

      SpillVGPRs.push_back(SGPRSpillVGPRCSR(LaneVGPR, CSRSpillFI));

      // Add this register as live-in to all blocks to avoid machine verifer
      // complaining about use of an undefined physical register.
      for (MachineBasicBlock &BB : MF)
        BB.addLiveIn(LaneVGPR);
    } else {
      LaneVGPR = SpillVGPRs.back().VGPR;
    }

    SpillLanes.push_back(SpilledReg(LaneVGPR, VGPRIndex));
  }

  return true;
}

/// Reserve AGPRs or VGPRs to support spilling for FrameIndex \p FI.
/// Either AGPR is spilled to VGPR to vice versa.
/// Returns true if a \p FI can be eliminated completely.
bool SIMachineFunctionInfo::allocateVGPRSpillToAGPR(MachineFunction &MF,
                                                    int FI,
                                                    bool isAGPRtoVGPR) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  MachineFrameInfo &FrameInfo = MF.getFrameInfo();
  const GCNSubtarget &ST =  MF.getSubtarget<GCNSubtarget>();

  assert(ST.hasMAIInsts() && FrameInfo.isSpillSlotObjectIndex(FI));

  auto &Spill = VGPRToAGPRSpills[FI];

  // This has already been allocated.
  if (!Spill.Lanes.empty())
    return Spill.FullyAllocated;

  unsigned Size = FrameInfo.getObjectSize(FI);
  unsigned NumLanes = Size / 4;
  Spill.Lanes.resize(NumLanes, AMDGPU::NoRegister);

  const TargetRegisterClass &RC =
      isAGPRtoVGPR ? AMDGPU::VGPR_32RegClass : AMDGPU::AGPR_32RegClass;
  auto Regs = RC.getRegisters();

  auto &SpillRegs = isAGPRtoVGPR ? SpillAGPR : SpillVGPR;
  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  Spill.FullyAllocated = true;

  // FIXME: Move allocation logic out of MachineFunctionInfo and initialize
  // once.
  BitVector OtherUsedRegs;
  OtherUsedRegs.resize(TRI->getNumRegs());

  const uint32_t *CSRMask =
      TRI->getCallPreservedMask(MF, MF.getFunction().getCallingConv());
  if (CSRMask)
    OtherUsedRegs.setBitsInMask(CSRMask);

  // TODO: Should include register tuples, but doesn't matter with current
  // usage.
  for (MCPhysReg Reg : SpillAGPR)
    OtherUsedRegs.set(Reg);
  for (MCPhysReg Reg : SpillVGPR)
    OtherUsedRegs.set(Reg);

  SmallVectorImpl<MCPhysReg>::const_iterator NextSpillReg = Regs.begin();
  for (unsigned I = 0; I < NumLanes; ++I) {
    NextSpillReg = std::find_if(
        NextSpillReg, Regs.end(), [&MRI, &OtherUsedRegs](MCPhysReg Reg) {
          return MRI.isAllocatable(Reg) && !MRI.isPhysRegUsed(Reg) &&
                 !OtherUsedRegs[Reg];
        });

    if (NextSpillReg == Regs.end()) { // Registers exhausted
      Spill.FullyAllocated = false;
      break;
    }

    OtherUsedRegs.set(*NextSpillReg);
    SpillRegs.push_back(*NextSpillReg);
    Spill.Lanes[I] = *NextSpillReg++;
  }

  return Spill.FullyAllocated;
}

void SIMachineFunctionInfo::removeDeadFrameIndices(MachineFrameInfo &MFI) {
  // The FP spill hasn't been inserted yet, so keep it around.
  for (auto &R : SGPRToVGPRSpills) {
    if (R.first != FramePointerSaveIndex)
      MFI.RemoveStackObject(R.first);
  }

  // All other SPGRs must be allocated on the default stack, so reset the stack
  // ID.
  for (int i = MFI.getObjectIndexBegin(), e = MFI.getObjectIndexEnd(); i != e;
       ++i)
    if (i != FramePointerSaveIndex)
      MFI.setStackID(i, TargetStackID::Default);

  for (auto &R : VGPRToAGPRSpills) {
    if (R.second.FullyAllocated)
      MFI.RemoveStackObject(R.first);
  }
}

MCPhysReg SIMachineFunctionInfo::getNextUserSGPR() const {
  assert(NumSystemSGPRs == 0 && "System SGPRs must be added after user SGPRs");
  return AMDGPU::SGPR0 + NumUserSGPRs;
}

MCPhysReg SIMachineFunctionInfo::getNextSystemSGPR() const {
  return AMDGPU::SGPR0 + NumUserSGPRs + NumSystemSGPRs;
}

static yaml::StringValue regToString(Register Reg,
                                     const TargetRegisterInfo &TRI) {
  yaml::StringValue Dest;
  {
    raw_string_ostream OS(Dest.Value);
    OS << printReg(Reg, &TRI);
  }
  return Dest;
}

static Optional<yaml::SIArgumentInfo>
convertArgumentInfo(const AMDGPUFunctionArgInfo &ArgInfo,
                    const TargetRegisterInfo &TRI) {
  yaml::SIArgumentInfo AI;

  auto convertArg = [&](Optional<yaml::SIArgument> &A,
                        const ArgDescriptor &Arg) {
    if (!Arg)
      return false;

    // Create a register or stack argument.
    yaml::SIArgument SA = yaml::SIArgument::createArgument(Arg.isRegister());
    if (Arg.isRegister()) {
      raw_string_ostream OS(SA.RegisterName.Value);
      OS << printReg(Arg.getRegister(), &TRI);
    } else
      SA.StackOffset = Arg.getStackOffset();
    // Check and update the optional mask.
    if (Arg.isMasked())
      SA.Mask = Arg.getMask();

    A = SA;
    return true;
  };

  bool Any = false;
  Any |= convertArg(AI.PrivateSegmentBuffer, ArgInfo.PrivateSegmentBuffer);
  Any |= convertArg(AI.DispatchPtr, ArgInfo.DispatchPtr);
  Any |= convertArg(AI.QueuePtr, ArgInfo.QueuePtr);
  Any |= convertArg(AI.KernargSegmentPtr, ArgInfo.KernargSegmentPtr);
  Any |= convertArg(AI.DispatchID, ArgInfo.DispatchID);
  Any |= convertArg(AI.FlatScratchInit, ArgInfo.FlatScratchInit);
  Any |= convertArg(AI.PrivateSegmentSize, ArgInfo.PrivateSegmentSize);
  Any |= convertArg(AI.WorkGroupIDX, ArgInfo.WorkGroupIDX);
  Any |= convertArg(AI.WorkGroupIDY, ArgInfo.WorkGroupIDY);
  Any |= convertArg(AI.WorkGroupIDZ, ArgInfo.WorkGroupIDZ);
  Any |= convertArg(AI.WorkGroupInfo, ArgInfo.WorkGroupInfo);
  Any |= convertArg(AI.PrivateSegmentWaveByteOffset,
                    ArgInfo.PrivateSegmentWaveByteOffset);
  Any |= convertArg(AI.ImplicitArgPtr, ArgInfo.ImplicitArgPtr);
  Any |= convertArg(AI.ImplicitBufferPtr, ArgInfo.ImplicitBufferPtr);
  Any |= convertArg(AI.WorkItemIDX, ArgInfo.WorkItemIDX);
  Any |= convertArg(AI.WorkItemIDY, ArgInfo.WorkItemIDY);
  Any |= convertArg(AI.WorkItemIDZ, ArgInfo.WorkItemIDZ);

  if (Any)
    return AI;

  return None;
}

yaml::SIMachineFunctionInfo::SIMachineFunctionInfo(
  const llvm::SIMachineFunctionInfo& MFI,
  const TargetRegisterInfo &TRI)
  : ExplicitKernArgSize(MFI.getExplicitKernArgSize()),
    MaxKernArgAlign(MFI.getMaxKernArgAlign()),
    LDSSize(MFI.getLDSSize()),
    IsEntryFunction(MFI.isEntryFunction()),
    NoSignedZerosFPMath(MFI.hasNoSignedZerosFPMath()),
    MemoryBound(MFI.isMemoryBound()),
    WaveLimiter(MFI.needsWaveLimiter()),
    HighBitsOf32BitAddress(MFI.get32BitAddressHighBits()),
    ScratchRSrcReg(regToString(MFI.getScratchRSrcReg(), TRI)),
    FrameOffsetReg(regToString(MFI.getFrameOffsetReg(), TRI)),
    StackPtrOffsetReg(regToString(MFI.getStackPtrOffsetReg(), TRI)),
    ArgInfo(convertArgumentInfo(MFI.getArgInfo(), TRI)),
    Mode(MFI.getMode()) {}

void yaml::SIMachineFunctionInfo::mappingImpl(yaml::IO &YamlIO) {
  MappingTraits<SIMachineFunctionInfo>::mapping(YamlIO, *this);
}

bool SIMachineFunctionInfo::initializeBaseYamlFields(
  const yaml::SIMachineFunctionInfo &YamlMFI) {
  ExplicitKernArgSize = YamlMFI.ExplicitKernArgSize;
  MaxKernArgAlign = assumeAligned(YamlMFI.MaxKernArgAlign);
  LDSSize = YamlMFI.LDSSize;
  HighBitsOf32BitAddress = YamlMFI.HighBitsOf32BitAddress;
  IsEntryFunction = YamlMFI.IsEntryFunction;
  NoSignedZerosFPMath = YamlMFI.NoSignedZerosFPMath;
  MemoryBound = YamlMFI.MemoryBound;
  WaveLimiter = YamlMFI.WaveLimiter;
  return false;
}
