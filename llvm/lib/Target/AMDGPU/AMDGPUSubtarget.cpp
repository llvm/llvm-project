//===-- AMDGPUSubtarget.cpp - AMDGPU Subtarget Information ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Implements the AMDGPU specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUSubtarget.h"
#include "AMDGPUCallLowering.h"
#include "AMDGPUInstructionSelector.h"
#include "AMDGPULegalizerInfo.h"
#include "AMDGPURegisterBankInfo.h"
#include "R600Subtarget.h"
#include "SIMachineFunctionInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/CodeGen/GlobalISel/InlineAsmLowering.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/IntrinsicsR600.h"
#include "llvm/IR/MDBuilder.h"
#include <algorithm>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-subtarget"

AMDGPUSubtarget::AMDGPUSubtarget(Triple TT) : TargetTriple(std::move(TT)) {}

bool AMDGPUSubtarget::useRealTrue16Insts() const {
  return hasTrue16BitInsts() && EnableRealTrue16Insts;
}

// Returns the maximum per-workgroup LDS allocation size (in bytes) that still
// allows the given function to achieve an occupancy of NWaves waves per
// SIMD / EU, taking into account only the function's *maximum* workgroup size.
unsigned
AMDGPUSubtarget::getMaxLocalMemSizeWithWaveCount(unsigned NWaves,
                                                 const Function &F) const {
  const unsigned WaveSize = getWavefrontSize();
  const unsigned WorkGroupSize = getFlatWorkGroupSizes(F).second;
  const unsigned WavesPerWorkgroup =
      std::max(1u, (WorkGroupSize + WaveSize - 1) / WaveSize);

  const unsigned WorkGroupsPerCU =
      std::max(1u, (NWaves * getEUsPerCU()) / WavesPerWorkgroup);

  return getLocalMemorySize() / WorkGroupsPerCU;
}

std::pair<unsigned, unsigned> AMDGPUSubtarget::getOccupancyWithWorkGroupSizes(
    uint32_t LDSBytes, std::pair<unsigned, unsigned> FlatWorkGroupSizes) const {

  // FIXME: We should take into account the LDS allocation granularity.
  const unsigned MaxWGsLDS = getLocalMemorySize() / std::max(LDSBytes, 1u);

  // Queried LDS size may be larger than available on a CU, in which case we
  // consider the only achievable occupancy to be 1, in line with what we
  // consider the occupancy to be when the number of requested registers in a
  // particular bank is higher than the number of available ones in that bank.
  if (!MaxWGsLDS)
    return {1, 1};

  const unsigned WaveSize = getWavefrontSize(), WavesPerEU = getMaxWavesPerEU();

  auto PropsFromWGSize = [=](unsigned WGSize)
      -> std::tuple<const unsigned, const unsigned, unsigned> {
    unsigned WavesPerWG = divideCeil(WGSize, WaveSize);
    unsigned WGsPerCU = std::min(getMaxWorkGroupsPerCU(WGSize), MaxWGsLDS);
    return {WavesPerWG, WGsPerCU, WavesPerWG * WGsPerCU};
  };

  // The maximum group size will generally yield the minimum number of
  // workgroups, maximum number of waves, and minimum occupancy. The opposite is
  // generally true for the minimum group size. LDS or barrier ressource
  // limitations can flip those minimums/maximums.
  const auto [MinWGSize, MaxWGSize] = FlatWorkGroupSizes;
  auto [MinWavesPerWG, MaxWGsPerCU, MaxWavesPerCU] = PropsFromWGSize(MinWGSize);
  auto [MaxWavesPerWG, MinWGsPerCU, MinWavesPerCU] = PropsFromWGSize(MaxWGSize);

  // It is possible that we end up with flipped minimum and maximum number of
  // waves per CU when the number of minimum/maximum concurrent groups on the CU
  // is limited by LDS usage or barrier resources.
  if (MinWavesPerCU >= MaxWavesPerCU) {
    std::swap(MinWavesPerCU, MaxWavesPerCU);
  } else {
    const unsigned WaveSlotsPerCU = WavesPerEU * getEUsPerCU();

    // Look for a potential smaller group size than the maximum which decreases
    // the concurrent number of waves on the CU for the same number of
    // concurrent workgroups on the CU.
    unsigned MinWavesPerCUForWGSize =
        divideCeil(WaveSlotsPerCU, MinWGsPerCU + 1) * MinWGsPerCU;
    if (MinWavesPerCU > MinWavesPerCUForWGSize) {
      unsigned ExcessSlots = MinWavesPerCU - MinWavesPerCUForWGSize;
      if (unsigned ExcessSlotsPerWG = ExcessSlots / MinWGsPerCU) {
        // There may exist a smaller group size than the maximum that achieves
        // the minimum number of waves per CU. This group size is the largest
        // possible size that requires MaxWavesPerWG - E waves where E is
        // maximized under the following constraints.
        // 1. 0 <= E <= ExcessSlotsPerWG
        // 2. (MaxWavesPerWG - E) * WaveSize >= MinWGSize
        MinWavesPerCU -= MinWGsPerCU * std::min(ExcessSlotsPerWG,
                                                MaxWavesPerWG - MinWavesPerWG);
      }
    }

    // Look for a potential larger group size than the minimum which increases
    // the concurrent number of waves on the CU for the same number of
    // concurrent workgroups on the CU.
    unsigned LeftoverSlots = WaveSlotsPerCU - MaxWGsPerCU * MinWavesPerWG;
    if (unsigned LeftoverSlotsPerWG = LeftoverSlots / MaxWGsPerCU) {
      // There may exist a larger group size than the minimum that achieves the
      // maximum number of waves per CU. This group size is the smallest
      // possible size that requires MinWavesPerWG + L waves where L is
      // maximized under the following constraints.
      // 1. 0 <= L <= LeftoverSlotsPerWG
      // 2. (MinWavesPerWG + L - 1) * WaveSize <= MaxWGSize
      MaxWavesPerCU += MaxWGsPerCU * std::min(LeftoverSlotsPerWG,
                                              ((MaxWGSize - 1) / WaveSize) + 1 -
                                                  MinWavesPerWG);
    }
  }

  // Return the minimum/maximum number of waves on any EU, assuming that all
  // wavefronts are spread across all EUs as evenly as possible.
  return {std::clamp(MinWavesPerCU / getEUsPerCU(), 1U, WavesPerEU),
          std::clamp(divideCeil(MaxWavesPerCU, getEUsPerCU()), 1U, WavesPerEU)};
}

std::pair<unsigned, unsigned> AMDGPUSubtarget::getOccupancyWithWorkGroupSizes(
    const MachineFunction &MF) const {
  const auto *MFI = MF.getInfo<SIMachineFunctionInfo>();
  return getOccupancyWithWorkGroupSizes(MFI->getLDSSize(), MF.getFunction());
}

std::pair<unsigned, unsigned>
AMDGPUSubtarget::getDefaultFlatWorkGroupSize(CallingConv::ID CC) const {
  switch (CC) {
  case CallingConv::AMDGPU_VS:
  case CallingConv::AMDGPU_LS:
  case CallingConv::AMDGPU_HS:
  case CallingConv::AMDGPU_ES:
  case CallingConv::AMDGPU_GS:
  case CallingConv::AMDGPU_PS:
    return std::pair(1, getWavefrontSize());
  default:
    return std::pair(1u, getMaxFlatWorkGroupSize());
  }
}

std::pair<unsigned, unsigned> AMDGPUSubtarget::getFlatWorkGroupSizes(
  const Function &F) const {
  // Default minimum/maximum flat work group sizes.
  std::pair<unsigned, unsigned> Default =
    getDefaultFlatWorkGroupSize(F.getCallingConv());

  // Requested minimum/maximum flat work group sizes.
  std::pair<unsigned, unsigned> Requested = AMDGPU::getIntegerPairAttribute(
    F, "amdgpu-flat-work-group-size", Default);

  // Make sure requested minimum is less than requested maximum.
  if (Requested.first > Requested.second)
    return Default;

  // Make sure requested values do not violate subtarget's specifications.
  if (Requested.first < getMinFlatWorkGroupSize())
    return Default;
  if (Requested.second > getMaxFlatWorkGroupSize())
    return Default;

  return Requested;
}

std::pair<unsigned, unsigned> AMDGPUSubtarget::getEffectiveWavesPerEU(
    std::pair<unsigned, unsigned> RequestedWavesPerEU,
    std::pair<unsigned, unsigned> FlatWorkGroupSizes, unsigned LDSBytes) const {
  // Default minimum/maximum number of waves per EU. The range of flat workgroup
  // sizes limits the achievable maximum, and we aim to support enough waves per
  // EU so that we can concurrently execute all waves of a single workgroup of
  // maximum size on a CU.
  std::pair<unsigned, unsigned> Default = {
      getWavesPerEUForWorkGroup(FlatWorkGroupSizes.second),
      getOccupancyWithWorkGroupSizes(LDSBytes, FlatWorkGroupSizes).second};
  Default.first = std::min(Default.first, Default.second);

  // Make sure requested minimum is within the default range and lower than the
  // requested maximum. The latter must not violate target specification.
  if (RequestedWavesPerEU.first < Default.first ||
      RequestedWavesPerEU.first > Default.second ||
      RequestedWavesPerEU.first > RequestedWavesPerEU.second ||
      RequestedWavesPerEU.second > getMaxWavesPerEU())
    return Default;

  // We cannot exceed maximum occupancy implied by flat workgroup size and LDS.
  RequestedWavesPerEU.second =
      std::min(RequestedWavesPerEU.second, Default.second);
  return RequestedWavesPerEU;
}

std::pair<unsigned, unsigned>
AMDGPUSubtarget::getWavesPerEU(const Function &F) const {
  // Default/requested minimum/maximum flat work group sizes.
  std::pair<unsigned, unsigned> FlatWorkGroupSizes = getFlatWorkGroupSizes(F);
  // Minimum number of bytes allocated in the LDS.
  unsigned LDSBytes =
      AMDGPU::getIntegerPairAttribute(F, "amdgpu-lds-size", {0, UINT32_MAX},
                                      /*OnlyFirstRequired=*/true)
          .first;
  return getWavesPerEU(FlatWorkGroupSizes, LDSBytes, F);
}

std::pair<unsigned, unsigned>
AMDGPUSubtarget::getWavesPerEU(std::pair<unsigned, unsigned> FlatWorkGroupSizes,
                               unsigned LDSBytes, const Function &F) const {
  // Default minimum/maximum number of waves per execution unit.
  std::pair<unsigned, unsigned> Default(1, getMaxWavesPerEU());

  // Requested minimum/maximum number of waves per execution unit.
  std::pair<unsigned, unsigned> Requested =
      AMDGPU::getIntegerPairAttribute(F, "amdgpu-waves-per-eu", Default, true);
  return getEffectiveWavesPerEU(Requested, FlatWorkGroupSizes, LDSBytes);
}

static unsigned getReqdWorkGroupSize(const Function &Kernel, unsigned Dim) {
  auto *Node = Kernel.getMetadata("reqd_work_group_size");
  if (Node && Node->getNumOperands() == 3)
    return mdconst::extract<ConstantInt>(Node->getOperand(Dim))->getZExtValue();
  return std::numeric_limits<unsigned>::max();
}

bool AMDGPUSubtarget::isMesaKernel(const Function &F) const {
  return isMesa3DOS() && !AMDGPU::isShader(F.getCallingConv());
}

unsigned AMDGPUSubtarget::getMaxWorkitemID(const Function &Kernel,
                                           unsigned Dimension) const {
  unsigned ReqdSize = getReqdWorkGroupSize(Kernel, Dimension);
  if (ReqdSize != std::numeric_limits<unsigned>::max())
    return ReqdSize - 1;
  return getFlatWorkGroupSizes(Kernel).second - 1;
}

bool AMDGPUSubtarget::isSingleLaneExecution(const Function &Func) const {
  for (int I = 0; I < 3; ++I) {
    if (getMaxWorkitemID(Func, I) > 0)
      return false;
  }

  return true;
}

bool AMDGPUSubtarget::makeLIDRangeMetadata(Instruction *I) const {
  Function *Kernel = I->getParent()->getParent();
  unsigned MinSize = 0;
  unsigned MaxSize = getFlatWorkGroupSizes(*Kernel).second;
  bool IdQuery = false;

  // If reqd_work_group_size is present it narrows value down.
  if (auto *CI = dyn_cast<CallInst>(I)) {
    const Function *F = CI->getCalledFunction();
    if (F) {
      unsigned Dim = UINT_MAX;
      switch (F->getIntrinsicID()) {
      case Intrinsic::amdgcn_workitem_id_x:
      case Intrinsic::r600_read_tidig_x:
        IdQuery = true;
        [[fallthrough]];
      case Intrinsic::r600_read_local_size_x:
        Dim = 0;
        break;
      case Intrinsic::amdgcn_workitem_id_y:
      case Intrinsic::r600_read_tidig_y:
        IdQuery = true;
        [[fallthrough]];
      case Intrinsic::r600_read_local_size_y:
        Dim = 1;
        break;
      case Intrinsic::amdgcn_workitem_id_z:
      case Intrinsic::r600_read_tidig_z:
        IdQuery = true;
        [[fallthrough]];
      case Intrinsic::r600_read_local_size_z:
        Dim = 2;
        break;
      default:
        break;
      }

      if (Dim <= 3) {
        unsigned ReqdSize = getReqdWorkGroupSize(*Kernel, Dim);
        if (ReqdSize != std::numeric_limits<unsigned>::max())
          MinSize = MaxSize = ReqdSize;
      }
    }
  }

  if (!MaxSize)
    return false;

  // Range metadata is [Lo, Hi). For ID query we need to pass max size
  // as Hi. For size query we need to pass Hi + 1.
  if (IdQuery)
    MinSize = 0;
  else
    ++MaxSize;

  APInt Lower{32, MinSize};
  APInt Upper{32, MaxSize};
  if (auto *CI = dyn_cast<CallBase>(I)) {
    ConstantRange Range(Lower, Upper);
    CI->addRangeRetAttr(Range);
  } else {
    MDBuilder MDB(I->getContext());
    MDNode *MaxWorkGroupSizeRange = MDB.createRange(Lower, Upper);
    I->setMetadata(LLVMContext::MD_range, MaxWorkGroupSizeRange);
  }
  return true;
}

unsigned AMDGPUSubtarget::getImplicitArgNumBytes(const Function &F) const {
  assert(AMDGPU::isKernel(F.getCallingConv()));

  // We don't allocate the segment if we know the implicit arguments weren't
  // used, even if the ABI implies we need them.
  if (F.hasFnAttribute("amdgpu-no-implicitarg-ptr"))
    return 0;

  if (isMesaKernel(F))
    return 16;

  // Assume all implicit inputs are used by default
  const Module *M = F.getParent();
  unsigned NBytes =
      AMDGPU::getAMDHSACodeObjectVersion(*M) >= AMDGPU::AMDHSA_COV5 ? 256 : 56;
  return F.getFnAttributeAsParsedInteger("amdgpu-implicitarg-num-bytes",
                                         NBytes);
}

uint64_t AMDGPUSubtarget::getExplicitKernArgSize(const Function &F,
                                                 Align &MaxAlign) const {
  assert(F.getCallingConv() == CallingConv::AMDGPU_KERNEL ||
         F.getCallingConv() == CallingConv::SPIR_KERNEL);

  const DataLayout &DL = F.getDataLayout();
  uint64_t ExplicitArgBytes = 0;
  MaxAlign = Align(1);

  for (const Argument &Arg : F.args()) {
    if (Arg.hasAttribute("amdgpu-hidden-argument"))
      continue;

    const bool IsByRef = Arg.hasByRefAttr();
    Type *ArgTy = IsByRef ? Arg.getParamByRefType() : Arg.getType();
    Align Alignment = DL.getValueOrABITypeAlignment(
        IsByRef ? Arg.getParamAlign() : std::nullopt, ArgTy);
    uint64_t AllocSize = DL.getTypeAllocSize(ArgTy);
    ExplicitArgBytes = alignTo(ExplicitArgBytes, Alignment) + AllocSize;
    MaxAlign = std::max(MaxAlign, Alignment);
  }

  return ExplicitArgBytes;
}

unsigned AMDGPUSubtarget::getKernArgSegmentSize(const Function &F,
                                                Align &MaxAlign) const {
  if (F.getCallingConv() != CallingConv::AMDGPU_KERNEL &&
      F.getCallingConv() != CallingConv::SPIR_KERNEL)
    return 0;

  uint64_t ExplicitArgBytes = getExplicitKernArgSize(F, MaxAlign);

  unsigned ExplicitOffset = getExplicitKernelArgOffset();

  uint64_t TotalSize = ExplicitOffset + ExplicitArgBytes;
  unsigned ImplicitBytes = getImplicitArgNumBytes(F);
  if (ImplicitBytes != 0) {
    const Align Alignment = getAlignmentForImplicitArgPtr();
    TotalSize = alignTo(ExplicitArgBytes, Alignment) + ImplicitBytes;
    MaxAlign = std::max(MaxAlign, Alignment);
  }

  // Being able to dereference past the end is useful for emitting scalar loads.
  return alignTo(TotalSize, 4);
}

AMDGPUDwarfFlavour AMDGPUSubtarget::getAMDGPUDwarfFlavour() const {
  return getWavefrontSize() == 32 ? AMDGPUDwarfFlavour::Wave32
                                  : AMDGPUDwarfFlavour::Wave64;
}

const AMDGPUSubtarget &AMDGPUSubtarget::get(const MachineFunction &MF) {
  if (MF.getTarget().getTargetTriple().isAMDGCN())
    return static_cast<const AMDGPUSubtarget&>(MF.getSubtarget<GCNSubtarget>());
  return static_cast<const AMDGPUSubtarget &>(MF.getSubtarget<R600Subtarget>());
}

const AMDGPUSubtarget &AMDGPUSubtarget::get(const TargetMachine &TM, const Function &F) {
  if (TM.getTargetTriple().isAMDGCN())
    return static_cast<const AMDGPUSubtarget&>(TM.getSubtarget<GCNSubtarget>(F));
  return static_cast<const AMDGPUSubtarget &>(
      TM.getSubtarget<R600Subtarget>(F));
}

// FIXME: This has no reason to be in subtarget
SmallVector<unsigned>
AMDGPUSubtarget::getMaxNumWorkGroups(const Function &F) const {
  return AMDGPU::getIntegerVecAttribute(F, "amdgpu-max-num-workgroups", 3,
                                        std::numeric_limits<uint32_t>::max());
}
