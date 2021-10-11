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

  // Enable all kernel inputs if we have the fixed ABI. Don't bother if we don't
  // have any calls.
  const bool UseFixedABI = AMDGPUTargetMachine::EnableFixedFunctionABI &&
                           CC != CallingConv::AMDGPU_Gfx &&
                           (!isEntryFunction() || HasCalls);

  if (CC == CallingConv::AMDGPU_KERNEL || CC == CallingConv::SPIR_KERNEL) {
    if (!F.arg_empty() || ST.getImplicitArgNumBytes(F) != 0)
      KernargSegmentPtr = true;
    WorkGroupIDX = true;
    WorkItemIDX = true;
  } else if (CC == CallingConv::AMDGPU_PS) {
    PSInputAddr = AMDGPU::getInitialPSInputAddr(F);
  }

  if (!isEntryFunction()) {
    if (UseFixedABI)
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

  bool HasStackObjects = F.hasFnAttribute("amdgpu-stack-objects");
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

  bool isAmdHsaOrMesa = ST.isAmdHsaOrMesa(F);
  if (isAmdHsaOrMesa && !ST.enableFlatScratch())
    PrivateSegmentBuffer = true;
  else if (ST.isMesaGfxShader(F))
    ImplicitBufferPtr = true;

  if (!AMDGPU::isGraphics(CC)) {
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
  }

  // TODO: This could be refined a lot. The attribute is a poor way of
  // detecting calls or stack objects that may require it before argument
  // lowering.
  if (ST.hasFlatAddressSpace() && isEntryFunction() &&
      (isAmdHsaOrMesa || ST.enableFlatScratch()) &&
      (HasCalls || HasStackObjects || ST.enableFlatScratch()) &&
      !ST.flatScratchIsArchitected()) {
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
  SIMachineFunctionInfo *FuncInfo = MF.getInfo<SIMachineFunctionInfo>();

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

    // Reserve a VGPR (when NumVGPRSpillLanes = 0, WaveSize, 2*WaveSize, ..) and
    // when one of the two conditions is true:
    // 1. One reserved VGPR being tracked by VGPRReservedForSGPRSpill is not yet
    // reserved.
    // 2. All spill lanes of reserved VGPR(s) are full and another spill lane is
    // required.
    if (FuncInfo->VGPRReservedForSGPRSpill && NumVGPRSpillLanes < WaveSize) {
      assert(FuncInfo->VGPRReservedForSGPRSpill == SpillVGPRs.back().VGPR);
      LaneVGPR = FuncInfo->VGPRReservedForSGPRSpill;
    } else if (VGPRIndex == 0) {
      LaneVGPR = TRI->findUnusedRegister(MRI, &AMDGPU::VGPR_32RegClass, MF);
      if (LaneVGPR == AMDGPU::NoRegister) {
        // We have no VGPRs left for spilling SGPRs. Reset because we will not
        // partially spill the SGPR to VGPRs.
        SGPRToVGPRSpills.erase(FI);
        NumVGPRSpillLanes -= I;

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

/// Reserve a VGPR for spilling of SGPRs
bool SIMachineFunctionInfo::reserveVGPRforSGPRSpills(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  SIMachineFunctionInfo *FuncInfo = MF.getInfo<SIMachineFunctionInfo>();

  Register LaneVGPR = TRI->findUnusedRegister(
      MF.getRegInfo(), &AMDGPU::VGPR_32RegClass, MF, true);
  if (LaneVGPR == Register())
    return false;
  SpillVGPRs.push_back(SGPRSpillVGPR(LaneVGPR, None));
  FuncInfo->VGPRReservedForSGPRSpill = LaneVGPR;
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

void SIMachineFunctionInfo::removeDeadFrameIndices(MachineFunction &MF) {
  MachineFrameInfo &MFI = MF.getFrameInfo();
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  // Remove dead frame indices from function frame, however keep FP & BP since
  // spills for them haven't been inserted yet. And also make sure to remove the
  // frame indices from `SGPRToVGPRSpills` data structure, otherwise, it could
  // result in an unexpected side effect and bug, in case of any re-mapping of
  // freed frame indices by later pass(es) like "stack slot coloring".
  for (auto &R : make_early_inc_range(SGPRToVGPRSpills)) {
    if (R.first != FramePointerSaveIndex && R.first != BasePointerSaveIndex &&
        (!TRI->isCFISavedRegsSpillEnabled() ||
         (R.first != ReturnAddressSaveIndex && R.first != EXECSaveIndex))) {
      MFI.RemoveStackObject(R.first);
      SGPRToVGPRSpills.erase(R.first);
    }
  }

  // All other SPGRs must be allocated on the default stack, so reset the stack
  // ID.
  for (int i = MFI.getObjectIndexBegin(), e = MFI.getObjectIndexEnd(); i != e;
       ++i)
    if (i != FramePointerSaveIndex && i != BasePointerSaveIndex &&
        (!TRI->isCFISavedRegsSpillEnabled() ||
         (i != ReturnAddressSaveIndex && i != EXECSaveIndex)))
      MFI.setStackID(i, TargetStackID::Default);

  for (auto &R : VGPRToAGPRSpills) {
    if (R.second.FullyAllocated)
      MFI.RemoveStackObject(R.first);
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

// Remove VGPR which was reserved for SGPR spills if there are no spilled SGPRs
bool SIMachineFunctionInfo::removeVGPRForSGPRSpill(Register ReservedVGPR,
                                                   MachineFunction &MF) {
  for (auto *i = SpillVGPRs.begin(); i < SpillVGPRs.end(); i++) {
    if (i->VGPR == ReservedVGPR) {
      SpillVGPRs.erase(i);

      for (MachineBasicBlock &MBB : MF) {
        MBB.removeLiveIn(ReservedVGPR);
        MBB.sortUniqueLiveIns();
      }
      this->VGPRReservedForSGPRSpill = AMDGPU::NoRegister;
      return true;
    }
  }
  return false;
}
