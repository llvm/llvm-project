//===- SIMachineFunctionInfo.cpp - SI Machine Function Info ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SIMachineFunctionInfo.h"
#include "AMDGPUTargetMachine.h"
#include "AMDGPUSubtarget.h"
#include "SIRegisterInfo.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/Optional.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MIRParser/MIParser.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DiagnosticInfo.h"
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

  Occupancy = ST.computeOccupancy(F, getLDSSize());
  CallingConv::ID CC = F.getCallingConv();

  // FIXME: Should have analysis or something rather than attribute to detect
  // calls.
  const bool HasCalls = F.hasFnAttribute("amdgpu-calls");

  const bool IsKernel = CC == CallingConv::AMDGPU_KERNEL ||
                        CC == CallingConv::SPIR_KERNEL;

  if (IsKernel) {
    if (!F.arg_empty() || ST.getImplicitArgNumBytes(F) != 0)
      KernargSegmentPtr = true;
    WorkGroupIDX = true;
    WorkItemIDX = true;
  } else if (CC == CallingConv::AMDGPU_PS) {
    PSInputAddr = AMDGPU::getInitialPSInputAddr(F);
  }

  MayNeedAGPRs = ST.hasMAIInsts();

  if (!isEntryFunction()) {
    if (CC != CallingConv::AMDGPU_Gfx)
      ArgInfo = AMDGPUArgumentUsageInfo::FixedABIFunctionInfo;

    // TODO: Pick a high register, and shift down, similar to a kernel.
    FrameOffsetReg = AMDGPU::SGPR33;
    StackPtrOffsetReg = AMDGPU::SGPR32;

    if (!ST.enableFlatScratch()) {
      // Non-entry functions have no special inputs for now, other registers
      // required for scratch access.
      ScratchRSrcReg = AMDGPU::SGPR0_SGPR1_SGPR2_SGPR3;

      ArgInfo.PrivateSegmentBuffer =
        ArgDescriptor::createRegister(ScratchRSrcReg);
    }

    if (!F.hasFnAttribute("amdgpu-no-implicitarg-ptr"))
      ImplicitArgPtr = true;
  } else {
    ImplicitArgPtr = false;
    MaxKernArgAlign = std::max(ST.getAlignmentForImplicitArgPtr(),
                               MaxKernArgAlign);

    if (ST.hasGFX90AInsts() &&
        ST.getMaxNumVGPRs(F) <= AMDGPU::VGPR_32RegClass.getNumRegs() &&
        !mayUseAGPRs(MF))
      MayNeedAGPRs = false; // We will select all MAI with VGPR operands.
  }

  bool isAmdHsaOrMesa = ST.isAmdHsaOrMesa(F);
  if (isAmdHsaOrMesa && !ST.enableFlatScratch())
    PrivateSegmentBuffer = true;
  else if (ST.isMesaGfxShader(F))
    ImplicitBufferPtr = true;

  if (!AMDGPU::isGraphics(CC)) {
    if (IsKernel || !F.hasFnAttribute("amdgpu-no-workgroup-id-x"))
      WorkGroupIDX = true;

    if (!F.hasFnAttribute("amdgpu-no-workgroup-id-y"))
      WorkGroupIDY = true;

    if (!F.hasFnAttribute("amdgpu-no-workgroup-id-z"))
      WorkGroupIDZ = true;

    if (IsKernel || !F.hasFnAttribute("amdgpu-no-workitem-id-x"))
      WorkItemIDX = true;

    if (!F.hasFnAttribute("amdgpu-no-workitem-id-y") &&
        ST.getMaxWorkitemID(F, 1) != 0)
      WorkItemIDY = true;

    if (!F.hasFnAttribute("amdgpu-no-workitem-id-z") &&
        ST.getMaxWorkitemID(F, 2) != 0)
      WorkItemIDZ = true;

    if (!F.hasFnAttribute("amdgpu-no-dispatch-ptr"))
      DispatchPtr = true;

    if (!F.hasFnAttribute("amdgpu-no-queue-ptr"))
      QueuePtr = true;

    if (!F.hasFnAttribute("amdgpu-no-dispatch-id"))
      DispatchID = true;
  }

  // FIXME: This attribute is a hack, we just need an analysis on the function
  // to look for allocas.
  bool HasStackObjects = F.hasFnAttribute("amdgpu-stack-objects");

  // TODO: This could be refined a lot. The attribute is a poor way of
  // detecting calls or stack objects that may require it before argument
  // lowering.
  if (ST.hasFlatAddressSpace() && isEntryFunction() &&
      (isAmdHsaOrMesa || ST.enableFlatScratch()) &&
      (HasCalls || HasStackObjects || ST.enableFlatScratch()) &&
      !ST.flatScratchIsArchitected()) {
    FlatScratchInit = true;
  }

  if (isEntryFunction()) {
    // X, XY, and XYZ are the only supported combinations, so make sure Y is
    // enabled if Z is.
    if (WorkItemIDZ)
      WorkItemIDY = true;

    if (!ST.flatScratchIsArchitected()) {
      PrivateSegmentWaveByteOffset = true;

      // HS and GS always have the scratch wave offset in SGPR5 on GFX9.
      if (ST.getGeneration() >= AMDGPUSubtarget::GFX9 &&
          (CC == CallingConv::AMDGPU_HS || CC == CallingConv::AMDGPU_GS))
        ArgInfo.PrivateSegmentWaveByteOffset =
            ArgDescriptor::createRegister(AMDGPU::SGPR5);
    }
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

  // On GFX908, in order to guarantee copying between AGPRs, we need a scratch
  // VGPR available at all times.
  if (ST.hasMAIInsts() && !ST.hasGFX90AInsts()) {
    VGPRForAGPRCopy = AMDGPU::VGPR_32RegClass.getRegister(32);
  }
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

bool SIMachineFunctionInfo::isCalleeSavedReg(const MCPhysReg *CSRegs,
                                             MCPhysReg Reg) {
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
  unsigned NumLanes = Size / 4;

  if (NumLanes > WaveSize)
    return false;

  assert(Size >= 4 && "invalid sgpr spill size");
  assert(TRI->spillSGPRToVGPR() && "not spilling SGPRs to VGPRs");

  // Make sure to handle the case where a wide SGPR spill may span between two
  // VGPRs.
  for (unsigned I = 0; I < NumLanes; ++I, ++NumVGPRSpillLanes) {
    Register LaneVGPR;
    unsigned VGPRIndex = (NumVGPRSpillLanes % WaveSize);

    if (VGPRIndex == 0) {
      LaneVGPR = TRI->findUnusedRegister(MRI, &AMDGPU::VGPR_32RegClass, MF);
      if (LaneVGPR == AMDGPU::NoRegister) {
        // We have no VGPRs left for spilling SGPRs. Reset because we will not
        // partially spill the SGPR to VGPRs.
        SGPRToVGPRSpills.erase(FI);
        NumVGPRSpillLanes -= I;

        // FIXME: We can run out of free registers with split allocation if
        // IPRA is enabled and a called function already uses every VGPR.
#if 0
        DiagnosticInfoResourceLimit DiagOutOfRegs(MF.getFunction(),
                                                  "VGPRs for SGPR spilling",
                                                  0, DS_Error);
        MF.getFunction().getContext().diagnose(DiagOutOfRegs);
#endif
        return false;
      }

      Optional<int> SpillFI;
      // We need to preserve inactive lanes, so always save, even caller-save
      // registers.
      if (!isEntryFunction()) {
        SpillFI = FrameInfo.CreateSpillStackObject(4, Align(4));
      }

      SpillVGPRs.push_back(SGPRSpillVGPR(LaneVGPR, SpillFI));

      // Add this register as live-in to all blocks to avoid machine verifier
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
  for (int I = NumLanes - 1; I >= 0; --I) {
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

bool SIMachineFunctionInfo::removeDeadFrameIndices(
    MachineFrameInfo &MFI, bool ResetSGPRSpillStackIDs) {
  // Remove dead frame indices from function frame, however keep FP & BP since
  // spills for them haven't been inserted yet. And also make sure to remove the
  // frame indices from `SGPRToVGPRSpills` data structure, otherwise, it could
  // result in an unexpected side effect and bug, in case of any re-mapping of
  // freed frame indices by later pass(es) like "stack slot coloring".
  for (auto &R : make_early_inc_range(SGPRToVGPRSpills)) {
    if (R.first != FramePointerSaveIndex && R.first != BasePointerSaveIndex) {
      MFI.RemoveStackObject(R.first);
      SGPRToVGPRSpills.erase(R.first);
    }
  }

  bool HaveSGPRToMemory = false;

  if (ResetSGPRSpillStackIDs) {
    // All other SPGRs must be allocated on the default stack, so reset the
    // stack ID.
    for (int i = MFI.getObjectIndexBegin(), e = MFI.getObjectIndexEnd(); i != e;
         ++i) {
      if (i != FramePointerSaveIndex && i != BasePointerSaveIndex) {
        if (MFI.getStackID(i) == TargetStackID::SGPRSpill) {
          MFI.setStackID(i, TargetStackID::Default);
          HaveSGPRToMemory = true;
        }
      }
    }
  }

  for (auto &R : VGPRToAGPRSpills) {
    if (R.second.IsDead)
      MFI.RemoveStackObject(R.first);
  }

  return HaveSGPRToMemory;
}

void SIMachineFunctionInfo::allocateWWMReservedSpillSlots(
    MachineFrameInfo &MFI, const SIRegisterInfo &TRI) {
  assert(WWMReservedFrameIndexes.empty());

  WWMReservedFrameIndexes.resize(WWMReservedRegs.size());

  int I = 0;
  for (Register VGPR : WWMReservedRegs) {
    const TargetRegisterClass *RC = TRI.getPhysRegClass(VGPR);
    WWMReservedFrameIndexes[I++] = MFI.CreateSpillStackObject(
        TRI.getSpillSize(*RC), TRI.getSpillAlign(*RC));
  }
}

int SIMachineFunctionInfo::getScavengeFI(MachineFrameInfo &MFI,
                                         const SIRegisterInfo &TRI) {
  if (ScavengeFI)
    return *ScavengeFI;
  if (isEntryFunction()) {
    ScavengeFI = MFI.CreateFixedObject(
        TRI.getSpillSize(AMDGPU::SGPR_32RegClass), 0, false);
  } else {
    ScavengeFI = MFI.CreateStackObject(
        TRI.getSpillSize(AMDGPU::SGPR_32RegClass),
        TRI.getSpillAlign(AMDGPU::SGPR_32RegClass), false);
  }
  return *ScavengeFI;
}

MCPhysReg SIMachineFunctionInfo::getNextUserSGPR() const {
  assert(NumSystemSGPRs == 0 && "System SGPRs must be added after user SGPRs");
  return AMDGPU::SGPR0 + NumUserSGPRs;
}

MCPhysReg SIMachineFunctionInfo::getNextSystemSGPR() const {
  return AMDGPU::SGPR0 + NumUserSGPRs + NumSystemSGPRs;
}

Register
SIMachineFunctionInfo::getGITPtrLoReg(const MachineFunction &MF) const {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!ST.isAmdPalOS())
    return Register();
  Register GitPtrLo = AMDGPU::SGPR0; // Low GIT address passed in
  if (ST.hasMergedShaders()) {
    switch (MF.getFunction().getCallingConv()) {
    case CallingConv::AMDGPU_HS:
    case CallingConv::AMDGPU_GS:
      // Low GIT address is passed in s8 rather than s0 for an LS+HS or
      // ES+GS merged shader on gfx9+.
      GitPtrLo = AMDGPU::SGPR8;
      return GitPtrLo;
    default:
      return GitPtrLo;
    }
  }
  return GitPtrLo;
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
    const llvm::SIMachineFunctionInfo &MFI, const TargetRegisterInfo &TRI,
    const llvm::MachineFunction &MF)
    : ExplicitKernArgSize(MFI.getExplicitKernArgSize()),
      MaxKernArgAlign(MFI.getMaxKernArgAlign()), LDSSize(MFI.getLDSSize()),
      DynLDSAlign(MFI.getDynLDSAlign()), IsEntryFunction(MFI.isEntryFunction()),
      NoSignedZerosFPMath(MFI.hasNoSignedZerosFPMath()),
      MemoryBound(MFI.isMemoryBound()), WaveLimiter(MFI.needsWaveLimiter()),
      HasSpilledSGPRs(MFI.hasSpilledSGPRs()),
      HasSpilledVGPRs(MFI.hasSpilledVGPRs()),
      HighBitsOf32BitAddress(MFI.get32BitAddressHighBits()),
      Occupancy(MFI.getOccupancy()),
      ScratchRSrcReg(regToString(MFI.getScratchRSrcReg(), TRI)),
      FrameOffsetReg(regToString(MFI.getFrameOffsetReg(), TRI)),
      StackPtrOffsetReg(regToString(MFI.getStackPtrOffsetReg(), TRI)),
      ArgInfo(convertArgumentInfo(MFI.getArgInfo(), TRI)), Mode(MFI.getMode()) {
  for (Register Reg : MFI.WWMReservedRegs)
    WWMReservedRegs.push_back(regToString(Reg, TRI));

  auto SFI = MFI.getOptionalScavengeFI();
  if (SFI)
    ScavengeFI = yaml::FrameIndex(*SFI, MF.getFrameInfo());
}

void yaml::SIMachineFunctionInfo::mappingImpl(yaml::IO &YamlIO) {
  MappingTraits<SIMachineFunctionInfo>::mapping(YamlIO, *this);
}

bool SIMachineFunctionInfo::initializeBaseYamlFields(
    const yaml::SIMachineFunctionInfo &YamlMFI, const MachineFunction &MF,
    PerFunctionMIParsingState &PFS, SMDiagnostic &Error, SMRange &SourceRange) {
  ExplicitKernArgSize = YamlMFI.ExplicitKernArgSize;
  MaxKernArgAlign = assumeAligned(YamlMFI.MaxKernArgAlign);
  LDSSize = YamlMFI.LDSSize;
  DynLDSAlign = YamlMFI.DynLDSAlign;
  HighBitsOf32BitAddress = YamlMFI.HighBitsOf32BitAddress;
  Occupancy = YamlMFI.Occupancy;
  IsEntryFunction = YamlMFI.IsEntryFunction;
  NoSignedZerosFPMath = YamlMFI.NoSignedZerosFPMath;
  MemoryBound = YamlMFI.MemoryBound;
  WaveLimiter = YamlMFI.WaveLimiter;
  HasSpilledSGPRs = YamlMFI.HasSpilledSGPRs;
  HasSpilledVGPRs = YamlMFI.HasSpilledVGPRs;

  if (YamlMFI.ScavengeFI) {
    auto FIOrErr = YamlMFI.ScavengeFI->getFI(MF.getFrameInfo());
    if (!FIOrErr) {
      // Create a diagnostic for a the frame index.
      const MemoryBuffer &Buffer =
          *PFS.SM->getMemoryBuffer(PFS.SM->getMainFileID());

      Error = SMDiagnostic(*PFS.SM, SMLoc(), Buffer.getBufferIdentifier(), 1, 1,
                           SourceMgr::DK_Error, toString(FIOrErr.takeError()),
                           "", None, None);
      SourceRange = YamlMFI.ScavengeFI->SourceRange;
      return true;
    }
    ScavengeFI = *FIOrErr;
  } else {
    ScavengeFI = None;
  }
  return false;
}

bool SIMachineFunctionInfo::mayUseAGPRs(const MachineFunction &MF) const {
  for (const BasicBlock &BB : MF.getFunction()) {
    for (const Instruction &I : BB) {
      const auto *CB = dyn_cast<CallBase>(&I);
      if (!CB)
        continue;

      if (CB->isInlineAsm()) {
        const InlineAsm *IA = dyn_cast<InlineAsm>(CB->getCalledOperand());
        for (const auto &CI : IA->ParseConstraints()) {
          for (StringRef Code : CI.Codes) {
            Code.consume_front("{");
            if (Code.startswith("a"))
              return true;
          }
        }
        continue;
      }

      const Function *Callee =
          dyn_cast<Function>(CB->getCalledOperand()->stripPointerCasts());
      if (!Callee)
        return true;

      if (Callee->getIntrinsicID() == Intrinsic::not_intrinsic)
        return true;
    }
  }

  return false;
}

bool SIMachineFunctionInfo::usesAGPRs(const MachineFunction &MF) const {
  if (UsesAGPRs)
    return *UsesAGPRs;

  if (!mayNeedAGPRs()) {
    UsesAGPRs = false;
    return false;
  }

  if (!AMDGPU::isEntryFunctionCC(MF.getFunction().getCallingConv()) ||
      MF.getFrameInfo().hasCalls()) {
    UsesAGPRs = true;
    return true;
  }

  const MachineRegisterInfo &MRI = MF.getRegInfo();

  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
    const Register Reg = Register::index2VirtReg(I);
    const TargetRegisterClass *RC = MRI.getRegClassOrNull(Reg);
    if (RC && SIRegisterInfo::isAGPRClass(RC)) {
      UsesAGPRs = true;
      return true;
    } else if (!RC && !MRI.use_empty(Reg) && MRI.getType(Reg).isValid()) {
      // Defer caching UsesAGPRs, function might not yet been regbank selected.
      return true;
    }
  }

  for (MCRegister Reg : AMDGPU::AGPR_32RegClass) {
    if (MRI.isPhysRegUsed(Reg)) {
      UsesAGPRs = true;
      return true;
    }
  }

  UsesAGPRs = false;
  return false;
}
