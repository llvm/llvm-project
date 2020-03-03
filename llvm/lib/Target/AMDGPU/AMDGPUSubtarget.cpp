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
#include "AMDGPU.h"
#include "AMDGPUTargetMachine.h"
#include "AMDGPUCallLowering.h"
#include "AMDGPUInstructionSelector.h"
#include "AMDGPULegalizerInfo.h"
#include "AMDGPURegisterBankInfo.h"
#include "SIMachineFunctionInfo.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include <algorithm>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#define AMDGPUSubtarget GCNSubtarget
#include "AMDGPUGenSubtargetInfo.inc"
#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#undef AMDGPUSubtarget
#include "R600GenSubtargetInfo.inc"

static cl::opt<bool> DisablePowerSched(
  "amdgpu-disable-power-sched",
  cl::desc("Disable scheduling to minimize mAI power bursts"),
  cl::init(false));

static cl::opt<bool> EnableVGPRIndexMode(
  "amdgpu-vgpr-index-mode",
  cl::desc("Use GPR indexing mode instead of movrel for vector indexing"),
  cl::init(false));

GCNSubtarget::~GCNSubtarget() = default;

R600Subtarget &
R600Subtarget::initializeSubtargetDependencies(const Triple &TT,
                                               StringRef GPU, StringRef FS) {
  SmallString<256> FullFS("+promote-alloca,");
  FullFS += FS;
  ParseSubtargetFeatures(GPU, FullFS);

  // FIXME: I don't think think Evergreen has any useful support for
  // denormals, but should be checked. Should we issue a warning somewhere
  // if someone tries to enable these?
  if (getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS) {
    FP32Denormals = false;
  }

  HasMulU24 = getGeneration() >= EVERGREEN;
  HasMulI24 = hasCaymanISA();

  return *this;
}

GCNSubtarget &
GCNSubtarget::initializeSubtargetDependencies(const Triple &TT,
                                              StringRef GPU, StringRef FS) {
  // Determine default and user-specified characteristics
  // On SI+, we want FP64 denormals to be on by default. FP32 denormals can be
  // enabled, but some instructions do not respect them and they run at the
  // double precision rate, so don't enable by default.
  //
  // We want to be able to turn these off, but making this a subtarget feature
  // for SI has the unhelpful behavior that it unsets everything else if you
  // disable it.
  //
  // Similarly we want enable-prt-strict-null to be on by default and not to
  // unset everything else if it is disabled

  // Assuming ECC is enabled is the conservative default.
  SmallString<256> FullFS("+promote-alloca,+load-store-opt,+sram-ecc,+xnack,");

  if (isAmdHsaOS()) // Turn on FlatForGlobal for HSA.
    FullFS += "+flat-for-global,+unaligned-buffer-access,+trap-handler,";

  // FIXME: I don't think think Evergreen has any useful support for
  // denormals, but should be checked. Should we issue a warning somewhere
  // if someone tries to enable these?
  if (getGeneration() >= AMDGPUSubtarget::SOUTHERN_ISLANDS) {
    FullFS += "+fp64-fp16-denormals,";
  } else {
    FullFS += "-fp32-denormals,";
  }

  FullFS += "+enable-prt-strict-null,"; // This is overridden by a disable in FS

  // Disable mutually exclusive bits.
  if (FS.find_lower("+wavefrontsize") != StringRef::npos) {
    if (FS.find_lower("wavefrontsize16") == StringRef::npos)
      FullFS += "-wavefrontsize16,";
    if (FS.find_lower("wavefrontsize32") == StringRef::npos)
      FullFS += "-wavefrontsize32,";
    if (FS.find_lower("wavefrontsize64") == StringRef::npos)
      FullFS += "-wavefrontsize64,";
  }

  FullFS += FS;

  ParseSubtargetFeatures(GPU, FullFS);

  // We don't support FP64 for EG/NI atm.
  assert(!hasFP64() || (getGeneration() >= AMDGPUSubtarget::SOUTHERN_ISLANDS));

  // Unless +-flat-for-global is specified, turn on FlatForGlobal for all OS-es
  // on VI and newer hardware to avoid assertion failures due to missing ADDR64
  // variants of MUBUF instructions.
  if (!hasAddr64() && !FS.contains("flat-for-global")) {
    FlatForGlobal = true;
  }

  // Set defaults if needed.
  if (MaxPrivateElementSize == 0)
    MaxPrivateElementSize = 4;

  if (LDSBankCount == 0)
    LDSBankCount = 32;

  if (TT.getArch() == Triple::amdgcn) {
    if (LocalMemorySize == 0)
      LocalMemorySize = 32768;

    // Do something sensible for unspecified target.
    if (!HasMovrel && !HasVGPRIndexMode)
      HasMovrel = true;
  }

  // Don't crash on invalid devices.
  if (WavefrontSize == 0)
    WavefrontSize = 64;

  HasFminFmaxLegacy = getGeneration() < AMDGPUSubtarget::VOLCANIC_ISLANDS;

  // Disable XNACK on targets where it is not enabled by default unless it is
  // explicitly requested.
  if (!FS.contains("+xnack") && DoesNotSupportXNACK && EnableXNACK) {
    ToggleFeature(AMDGPU::FeatureXNACK);
    EnableXNACK = false;
  }

  // ECC is on by default, but turn it off if the hardware doesn't support it
  // anyway. This matters for the gfx9 targets with d16 loads, but don't support
  // ECC.
  if (DoesNotSupportSRAMECC && EnableSRAMECC) {
    ToggleFeature(AMDGPU::FeatureSRAMECC);
    EnableSRAMECC = false;
  }

  return *this;
}

AMDGPUSubtarget::AMDGPUSubtarget(const Triple &TT) :
  TargetTriple(TT),
  Has16BitInsts(false),
  HasMadMixInsts(false),
  FP32Denormals(false),
  FPExceptions(false),
  HasSDWA(false),
  HasVOP3PInsts(false),
  HasMulI24(true),
  HasMulU24(true),
  HasInv2PiInlineImm(false),
  HasFminFmaxLegacy(true),
  EnablePromoteAlloca(false),
  HasTrigReducedRange(false),
  MaxWavesPerEU(10),
  LocalMemorySize(0),
  WavefrontSize(0)
  { }

GCNSubtarget::GCNSubtarget(const Triple &TT, StringRef GPU, StringRef FS,
                           const GCNTargetMachine &TM) :
    AMDGPUGenSubtargetInfo(TT, GPU, FS),
    AMDGPUSubtarget(TT),
    TargetTriple(TT),
    Gen(TT.getOS() == Triple::AMDHSA ? SEA_ISLANDS : SOUTHERN_ISLANDS),
    InstrItins(getInstrItineraryForCPU(GPU)),
    LDSBankCount(0),
    MaxPrivateElementSize(0),

    FastFMAF32(false),
    HalfRate64Ops(false),

    FP64FP16Denormals(false),
    FlatForGlobal(false),
    AutoWaitcntBeforeBarrier(false),
    CodeObjectV3(false),
    UnalignedScratchAccess(false),
    UnalignedBufferAccess(false),

    HasApertureRegs(false),
    EnableXNACK(false),
    DoesNotSupportXNACK(false),
    EnableCuMode(false),
    TrapHandler(false),

    EnableLoadStoreOpt(false),
    EnableUnsafeDSOffsetFolding(false),
    EnableSIScheduler(false),
    EnableDS128(false),
    EnablePRTStrictNull(false),
    DumpCode(false),

    FP64(false),
    GCN3Encoding(false),
    CIInsts(false),
    GFX8Insts(false),
    GFX9Insts(false),
    GFX10Insts(false),
    GFX7GFX8GFX9Insts(false),
    SGPRInitBug(false),
    HasSMemRealTime(false),
    HasIntClamp(false),
    HasFmaMixInsts(false),
    HasMovrel(false),
    HasVGPRIndexMode(false),
    HasScalarStores(false),
    HasScalarAtomics(false),
    HasSDWAOmod(false),
    HasSDWAScalar(false),
    HasSDWASdst(false),
    HasSDWAMac(false),
    HasSDWAOutModsVOPC(false),
    HasDPP(false),
    HasDPP8(false),
    HasR128A16(false),
    HasGFX10A16(false),
    HasNSAEncoding(false),
    HasDLInsts(false),
    HasDot1Insts(false),
    HasDot2Insts(false),
    HasDot3Insts(false),
    HasDot4Insts(false),
    HasDot5Insts(false),
    HasDot6Insts(false),
    HasMAIInsts(false),
    HasPkFmacF16Inst(false),
    HasAtomicFaddInsts(false),
    EnableSRAMECC(false),
    DoesNotSupportSRAMECC(false),
    HasNoSdstCMPX(false),
    HasVscnt(false),
    HasRegisterBanking(false),
    HasVOP3Literal(false),
    HasNoDataDepHazard(false),
    FlatAddressSpace(false),
    FlatInstOffsets(false),
    FlatGlobalInsts(false),
    FlatScratchInsts(false),
    ScalarFlatScratchInsts(false),
    AddNoCarryInsts(false),
    HasUnpackedD16VMem(false),
    LDSMisalignedBug(false),
    HasMFMAInlineLiteralBug(false),

    ScalarizeGlobal(false),

    HasVcmpxPermlaneHazard(false),
    HasVMEMtoScalarWriteHazard(false),
    HasSMEMtoVectorWriteHazard(false),
    HasInstFwdPrefetchBug(false),
    HasVcmpxExecWARHazard(false),
    HasLdsBranchVmemWARHazard(false),
    HasNSAtoVMEMBug(false),
    HasOffset3fBug(false),
    HasFlatSegmentOffsetBug(false),

    FeatureDisable(false),
    InstrInfo(initializeSubtargetDependencies(TT, GPU, FS)),
    TLInfo(TM, *this),
    FrameLowering(TargetFrameLowering::StackGrowsUp, getStackAlignment(), 0) {
  MaxWavesPerEU = AMDGPU::IsaInfo::getMaxWavesPerEU(this);
  CallLoweringInfo.reset(new AMDGPUCallLowering(*getTargetLowering()));
  Legalizer.reset(new AMDGPULegalizerInfo(*this, TM));
  RegBankInfo.reset(new AMDGPURegisterBankInfo(*this));
  InstSelector.reset(new AMDGPUInstructionSelector(
  *this, *static_cast<AMDGPURegisterBankInfo *>(RegBankInfo.get()), TM));
}

unsigned GCNSubtarget::getConstantBusLimit(unsigned Opcode) const {
  if (getGeneration() < GFX10)
    return 1;

  switch (Opcode) {
  case AMDGPU::V_LSHLREV_B64:
  case AMDGPU::V_LSHLREV_B64_gfx10:
  case AMDGPU::V_LSHL_B64:
  case AMDGPU::V_LSHRREV_B64:
  case AMDGPU::V_LSHRREV_B64_gfx10:
  case AMDGPU::V_LSHR_B64:
  case AMDGPU::V_ASHRREV_I64:
  case AMDGPU::V_ASHRREV_I64_gfx10:
  case AMDGPU::V_ASHR_I64:
    return 1;
  }

  return 2;
}

unsigned AMDGPUSubtarget::getMaxLocalMemSizeWithWaveCount(unsigned NWaves,
  const Function &F) const {
  if (NWaves == 1)
    return getLocalMemorySize();
  unsigned WorkGroupSize = getFlatWorkGroupSizes(F).second;
  unsigned WorkGroupsPerCu = getMaxWorkGroupsPerCU(WorkGroupSize);
  if (!WorkGroupsPerCu)
    return 0;
  unsigned MaxWaves = getMaxWavesPerEU();
  return getLocalMemorySize() * MaxWaves / WorkGroupsPerCu / NWaves;
}

// FIXME: Should return min,max range.
unsigned AMDGPUSubtarget::getOccupancyWithLocalMemSize(uint32_t Bytes,
  const Function &F) const {
  const unsigned MaxWorkGroupSize = getFlatWorkGroupSizes(F).second;
  const unsigned MaxWorkGroupsPerCu = getMaxWorkGroupsPerCU(MaxWorkGroupSize);
  if (!MaxWorkGroupsPerCu)
    return 0;

  const unsigned WaveSize = getWavefrontSize();

  // FIXME: Do we need to account for alignment requirement of LDS rounding the
  // size up?
  // Compute restriction based on LDS usage
  unsigned NumGroups = getLocalMemorySize() / (Bytes ? Bytes : 1u);

  // This can be queried with more LDS than is possible, so just assume the
  // worst.
  if (NumGroups == 0)
    return 1;

  NumGroups = std::min(MaxWorkGroupsPerCu, NumGroups);

  // Round to the number of waves.
  const unsigned MaxGroupNumWaves = (MaxWorkGroupSize + WaveSize - 1) / WaveSize;
  unsigned MaxWaves = NumGroups * MaxGroupNumWaves;

  // Clamp to the maximum possible number of waves.
  MaxWaves = std::min(MaxWaves, getMaxWavesPerEU());

  // FIXME: Needs to be a multiple of the group size?
  //MaxWaves = MaxGroupNumWaves * (MaxWaves / MaxGroupNumWaves);

  assert(MaxWaves > 0 && MaxWaves <= getMaxWavesPerEU() &&
         "computed invalid occupancy");
  return MaxWaves;
}

unsigned
AMDGPUSubtarget::getOccupancyWithLocalMemSize(const MachineFunction &MF) const {
  const auto *MFI = MF.getInfo<SIMachineFunctionInfo>();
  return getOccupancyWithLocalMemSize(MFI->getLDSSize(), MF.getFunction());
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
    return std::make_pair(1, getWavefrontSize());
  default:
    return std::make_pair(1u, getMaxFlatWorkGroupSize());
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

std::pair<unsigned, unsigned> AMDGPUSubtarget::getWavesPerEU(
  const Function &F) const {
  // Default minimum/maximum number of waves per execution unit.
  std::pair<unsigned, unsigned> Default(1, getMaxWavesPerEU());

  // Default/requested minimum/maximum flat work group sizes.
  std::pair<unsigned, unsigned> FlatWorkGroupSizes = getFlatWorkGroupSizes(F);

  // If minimum/maximum flat work group sizes were explicitly requested using
  // "amdgpu-flat-work-group-size" attribute, then set default minimum/maximum
  // number of waves per execution unit to values implied by requested
  // minimum/maximum flat work group sizes.
  unsigned MinImpliedByFlatWorkGroupSize =
    getMaxWavesPerEU(FlatWorkGroupSizes.second);
  bool RequestedFlatWorkGroupSize = false;

  if (F.hasFnAttribute("amdgpu-flat-work-group-size")) {
    Default.first = MinImpliedByFlatWorkGroupSize;
    RequestedFlatWorkGroupSize = true;
  }

  // Requested minimum/maximum number of waves per execution unit.
  std::pair<unsigned, unsigned> Requested = AMDGPU::getIntegerPairAttribute(
    F, "amdgpu-waves-per-eu", Default, true);

  // Make sure requested minimum is less than requested maximum.
  if (Requested.second && Requested.first > Requested.second)
    return Default;

  // Make sure requested values do not violate subtarget's specifications.
  if (Requested.first < getMinWavesPerEU() ||
      Requested.first > getMaxWavesPerEU())
    return Default;
  if (Requested.second > getMaxWavesPerEU())
    return Default;

  // Make sure requested values are compatible with values implied by requested
  // minimum/maximum flat work group sizes.
  if (RequestedFlatWorkGroupSize &&
      Requested.first < MinImpliedByFlatWorkGroupSize)
    return Default;

  return Requested;
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
        LLVM_FALLTHROUGH;
      case Intrinsic::r600_read_local_size_x:
        Dim = 0;
        break;
      case Intrinsic::amdgcn_workitem_id_y:
      case Intrinsic::r600_read_tidig_y:
        IdQuery = true;
        LLVM_FALLTHROUGH;
      case Intrinsic::r600_read_local_size_y:
        Dim = 1;
        break;
      case Intrinsic::amdgcn_workitem_id_z:
      case Intrinsic::r600_read_tidig_z:
        IdQuery = true;
        LLVM_FALLTHROUGH;
      case Intrinsic::r600_read_local_size_z:
        Dim = 2;
        break;
      default:
        break;
      }
      if (Dim <= 3) {
        if (auto Node = Kernel->getMetadata("reqd_work_group_size"))
          if (Node->getNumOperands() == 3)
            MinSize = MaxSize = mdconst::extract<ConstantInt>(
                                  Node->getOperand(Dim))->getZExtValue();
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

  MDBuilder MDB(I->getContext());
  MDNode *MaxWorkGroupSizeRange = MDB.createRange(APInt(32, MinSize),
                                                  APInt(32, MaxSize));
  I->setMetadata(LLVMContext::MD_range, MaxWorkGroupSizeRange);
  return true;
}

uint64_t AMDGPUSubtarget::getExplicitKernArgSize(const Function &F,
                                                 Align &MaxAlign) const {
  assert(F.getCallingConv() == CallingConv::AMDGPU_KERNEL ||
         F.getCallingConv() == CallingConv::SPIR_KERNEL);

  const DataLayout &DL = F.getParent()->getDataLayout();
  uint64_t ExplicitArgBytes = 0;
  MaxAlign = Align(1);

  for (const Argument &Arg : F.args()) {
    Type *ArgTy = Arg.getType();

    const Align Alignment(DL.getABITypeAlignment(ArgTy));
    uint64_t AllocSize = DL.getTypeAllocSize(ArgTy);
    ExplicitArgBytes = alignTo(ExplicitArgBytes, Alignment) + AllocSize;
    MaxAlign = std::max(MaxAlign, Alignment);
  }

  return ExplicitArgBytes;
}

unsigned AMDGPUSubtarget::getKernArgSegmentSize(const Function &F,
                                                Align &MaxAlign) const {
  uint64_t ExplicitArgBytes = getExplicitKernArgSize(F, MaxAlign);

  unsigned ExplicitOffset = getExplicitKernelArgOffset(F);

  uint64_t TotalSize = ExplicitOffset + ExplicitArgBytes;
  unsigned ImplicitBytes = getImplicitArgNumBytes(F);
  if (ImplicitBytes != 0) {
    const Align Alignment = getAlignmentForImplicitArgPtr();
    TotalSize = alignTo(ExplicitArgBytes, Alignment) + ImplicitBytes;
  }

  // Being able to dereference past the end is useful for emitting scalar loads.
  return alignTo(TotalSize, 4);
}

R600Subtarget::R600Subtarget(const Triple &TT, StringRef GPU, StringRef FS,
                             const TargetMachine &TM) :
  R600GenSubtargetInfo(TT, GPU, FS),
  AMDGPUSubtarget(TT),
  InstrInfo(*this),
  FrameLowering(TargetFrameLowering::StackGrowsUp, getStackAlignment(), 0),
  FMA(false),
  CaymanISA(false),
  CFALUBug(false),
  HasVertexCache(false),
  R600ALUInst(false),
  FP64(false),
  TexVTXClauseSize(0),
  Gen(R600),
  TLInfo(TM, initializeSubtargetDependencies(TT, GPU, FS)),
  InstrItins(getInstrItineraryForCPU(GPU)) { }

void GCNSubtarget::overrideSchedPolicy(MachineSchedPolicy &Policy,
                                      unsigned NumRegionInstrs) const {
  // Track register pressure so the scheduler can try to decrease
  // pressure once register usage is above the threshold defined by
  // SIRegisterInfo::getRegPressureSetLimit()
  Policy.ShouldTrackPressure = true;

  // Enabling both top down and bottom up scheduling seems to give us less
  // register spills than just using one of these approaches on its own.
  Policy.OnlyTopDown = false;
  Policy.OnlyBottomUp = false;

  // Enabling ShouldTrackLaneMasks crashes the SI Machine Scheduler.
  if (!enableSIScheduler())
    Policy.ShouldTrackLaneMasks = true;
}

bool GCNSubtarget::hasMadF16() const {
  return InstrInfo.pseudoToMCOpcode(AMDGPU::V_MAD_F16) != -1;
}

bool GCNSubtarget::useVGPRIndexMode() const {
  return !hasMovrel() || (EnableVGPRIndexMode && hasVGPRIndexMode());
}

unsigned GCNSubtarget::getOccupancyWithNumSGPRs(unsigned SGPRs) const {
  if (getGeneration() >= AMDGPUSubtarget::GFX10)
    return getMaxWavesPerEU();

  if (getGeneration() >= AMDGPUSubtarget::VOLCANIC_ISLANDS) {
    if (SGPRs <= 80)
      return 10;
    if (SGPRs <= 88)
      return 9;
    if (SGPRs <= 100)
      return 8;
    return 7;
  }
  if (SGPRs <= 48)
    return 10;
  if (SGPRs <= 56)
    return 9;
  if (SGPRs <= 64)
    return 8;
  if (SGPRs <= 72)
    return 7;
  if (SGPRs <= 80)
    return 6;
  return 5;
}

unsigned GCNSubtarget::getOccupancyWithNumVGPRs(unsigned VGPRs) const {
  unsigned MaxWaves = getMaxWavesPerEU();
  unsigned Granule = getVGPRAllocGranule();
  if (VGPRs < Granule)
    return MaxWaves;
  unsigned RoundedRegs = ((VGPRs + Granule - 1) / Granule) * Granule;
  return std::min(std::max(getTotalNumVGPRs() / RoundedRegs, 1u), MaxWaves);
}

unsigned GCNSubtarget::getReservedNumSGPRs(const MachineFunction &MF) const {
  const SIMachineFunctionInfo &MFI = *MF.getInfo<SIMachineFunctionInfo>();
  if (getGeneration() >= AMDGPUSubtarget::GFX10)
    return 2; // VCC. FLAT_SCRATCH and XNACK are no longer in SGPRs.

  if (MFI.hasFlatScratchInit()) {
    if (getGeneration() >= AMDGPUSubtarget::VOLCANIC_ISLANDS)
      return 6; // FLAT_SCRATCH, XNACK, VCC (in that order).
    if (getGeneration() == AMDGPUSubtarget::SEA_ISLANDS)
      return 4; // FLAT_SCRATCH, VCC (in that order).
  }

  if (isXNACKEnabled())
    return 4; // XNACK, VCC (in that order).
  return 2; // VCC.
}

unsigned GCNSubtarget::computeOccupancy(const MachineFunction &MF,
                                        unsigned LDSSize,
                                        unsigned NumSGPRs,
                                        unsigned NumVGPRs) const {
  unsigned Occupancy =
    std::min(getMaxWavesPerEU(),
             getOccupancyWithLocalMemSize(LDSSize, MF.getFunction()));
  if (NumSGPRs)
    Occupancy = std::min(Occupancy, getOccupancyWithNumSGPRs(NumSGPRs));
  if (NumVGPRs)
    Occupancy = std::min(Occupancy, getOccupancyWithNumVGPRs(NumVGPRs));
  return Occupancy;
}

unsigned GCNSubtarget::getMaxNumSGPRs(const MachineFunction &MF) const {
  const Function &F = MF.getFunction();
  const SIMachineFunctionInfo &MFI = *MF.getInfo<SIMachineFunctionInfo>();

  // Compute maximum number of SGPRs function can use using default/requested
  // minimum number of waves per execution unit.
  std::pair<unsigned, unsigned> WavesPerEU = MFI.getWavesPerEU();
  unsigned MaxNumSGPRs = getMaxNumSGPRs(WavesPerEU.first, false);
  unsigned MaxAddressableNumSGPRs = getMaxNumSGPRs(WavesPerEU.first, true);

  // Check if maximum number of SGPRs was explicitly requested using
  // "amdgpu-num-sgpr" attribute.
  if (F.hasFnAttribute("amdgpu-num-sgpr")) {
    unsigned Requested = AMDGPU::getIntegerAttribute(
      F, "amdgpu-num-sgpr", MaxNumSGPRs);

    // Make sure requested value does not violate subtarget's specifications.
    if (Requested && (Requested <= getReservedNumSGPRs(MF)))
      Requested = 0;

    // If more SGPRs are required to support the input user/system SGPRs,
    // increase to accommodate them.
    //
    // FIXME: This really ends up using the requested number of SGPRs + number
    // of reserved special registers in total. Theoretically you could re-use
    // the last input registers for these special registers, but this would
    // require a lot of complexity to deal with the weird aliasing.
    unsigned InputNumSGPRs = MFI.getNumPreloadedSGPRs();
    if (Requested && Requested < InputNumSGPRs)
      Requested = InputNumSGPRs;

    // Make sure requested value is compatible with values implied by
    // default/requested minimum/maximum number of waves per execution unit.
    if (Requested && Requested > getMaxNumSGPRs(WavesPerEU.first, false))
      Requested = 0;
    if (WavesPerEU.second &&
        Requested && Requested < getMinNumSGPRs(WavesPerEU.second))
      Requested = 0;

    if (Requested)
      MaxNumSGPRs = Requested;
  }

  if (hasSGPRInitBug())
    MaxNumSGPRs = AMDGPU::IsaInfo::FIXED_NUM_SGPRS_FOR_INIT_BUG;

  return std::min(MaxNumSGPRs - getReservedNumSGPRs(MF),
                  MaxAddressableNumSGPRs);
}

unsigned GCNSubtarget::getMaxNumVGPRs(const MachineFunction &MF) const {
  const Function &F = MF.getFunction();
  const SIMachineFunctionInfo &MFI = *MF.getInfo<SIMachineFunctionInfo>();

  // Compute maximum number of VGPRs function can use using default/requested
  // minimum number of waves per execution unit.
  std::pair<unsigned, unsigned> WavesPerEU = MFI.getWavesPerEU();
  unsigned MaxNumVGPRs = getMaxNumVGPRs(WavesPerEU.first);

  // Check if maximum number of VGPRs was explicitly requested using
  // "amdgpu-num-vgpr" attribute.
  if (F.hasFnAttribute("amdgpu-num-vgpr")) {
    unsigned Requested = AMDGPU::getIntegerAttribute(
      F, "amdgpu-num-vgpr", MaxNumVGPRs);

    // Make sure requested value is compatible with values implied by
    // default/requested minimum/maximum number of waves per execution unit.
    if (Requested && Requested > getMaxNumVGPRs(WavesPerEU.first))
      Requested = 0;
    if (WavesPerEU.second &&
        Requested && Requested < getMinNumVGPRs(WavesPerEU.second))
      Requested = 0;

    if (Requested)
      MaxNumVGPRs = Requested;
  }

  return MaxNumVGPRs;
}

void GCNSubtarget::adjustSchedDependency(SUnit *Src, SUnit *Dst,
                                         SDep &Dep) const {
  if (Dep.getKind() != SDep::Kind::Data || !Dep.getReg() ||
      !Src->isInstr() || !Dst->isInstr())
    return;

  MachineInstr *SrcI = Src->getInstr();
  MachineInstr *DstI = Dst->getInstr();

  if (SrcI->isBundle()) {
    const SIRegisterInfo *TRI = getRegisterInfo();
    auto Reg = Dep.getReg();
    MachineBasicBlock::const_instr_iterator I(SrcI->getIterator());
    MachineBasicBlock::const_instr_iterator E(SrcI->getParent()->instr_end());
    unsigned Lat = 0;
    for (++I; I != E && I->isBundledWithPred(); ++I) {
      if (I->modifiesRegister(Reg, TRI))
        Lat = InstrInfo.getInstrLatency(getInstrItineraryData(), *I);
      else if (Lat)
        --Lat;
    }
    Dep.setLatency(Lat);
  } else if (DstI->isBundle()) {
    const SIRegisterInfo *TRI = getRegisterInfo();
    auto Reg = Dep.getReg();
    MachineBasicBlock::const_instr_iterator I(DstI->getIterator());
    MachineBasicBlock::const_instr_iterator E(DstI->getParent()->instr_end());
    unsigned Lat = InstrInfo.getInstrLatency(getInstrItineraryData(), *SrcI);
    for (++I; I != E && I->isBundledWithPred() && Lat; ++I) {
      if (I->readsRegister(Reg, TRI))
        break;
      --Lat;
    }
    Dep.setLatency(Lat);
  }
}

namespace {
struct FillMFMAShadowMutation : ScheduleDAGMutation {
  const SIInstrInfo *TII;

  ScheduleDAGMI *DAG;

  FillMFMAShadowMutation(const SIInstrInfo *tii) : TII(tii) {}

  bool isSALU(const SUnit *SU) const {
    const MachineInstr *MI = SU->getInstr();
    return MI && TII->isSALU(*MI) && !MI->isTerminator();
  }

  bool isVALU(const SUnit *SU) const {
    const MachineInstr *MI = SU->getInstr();
    return MI && TII->isVALU(*MI);
  }

  bool canAddEdge(const SUnit *Succ, const SUnit *Pred) const {
    if (Pred->NodeNum < Succ->NodeNum)
      return true;

    SmallVector<const SUnit*, 64> Succs({Succ}), Preds({Pred});

    for (unsigned I = 0; I < Succs.size(); ++I) {
      for (const SDep &SI : Succs[I]->Succs) {
        const SUnit *SU = SI.getSUnit();
        if (SU != Succs[I] && llvm::find(Succs, SU) == Succs.end())
          Succs.push_back(SU);
      }
    }

    SmallPtrSet<const SUnit*, 32> Visited;
    while (!Preds.empty()) {
      const SUnit *SU = Preds.pop_back_val();
      if (llvm::find(Succs, SU) != Succs.end())
        return false;
      Visited.insert(SU);
      for (const SDep &SI : SU->Preds)
        if (SI.getSUnit() != SU && !Visited.count(SI.getSUnit()))
          Preds.push_back(SI.getSUnit());
    }

    return true;
  }

  // Link as much SALU intructions in chain as possible. Return the size
  // of the chain. Links up to MaxChain instructions.
  unsigned linkSALUChain(SUnit *From, SUnit *To, unsigned MaxChain,
                         SmallPtrSetImpl<SUnit *> &Visited) const {
    SmallVector<SUnit *, 8> Worklist({To});
    unsigned Linked = 0;

    while (!Worklist.empty() && MaxChain-- > 0) {
      SUnit *SU = Worklist.pop_back_val();
      if (!Visited.insert(SU).second)
        continue;

      LLVM_DEBUG(dbgs() << "Inserting edge from\n" ; DAG->dumpNode(*From);
                 dbgs() << "to\n"; DAG->dumpNode(*SU); dbgs() << '\n');

      if (SU->addPred(SDep(From, SDep::Artificial), false))
        ++Linked;

      for (SDep &SI : From->Succs) {
        SUnit *SUv = SI.getSUnit();
        if (SUv != From && isVALU(SUv) && canAddEdge(SUv, SU))
          SUv->addPred(SDep(SU, SDep::Artificial), false);
      }

      for (SDep &SI : SU->Succs) {
        SUnit *Succ = SI.getSUnit();
        if (Succ != SU && isSALU(Succ) && canAddEdge(From, Succ))
          Worklist.push_back(Succ);
      }
    }

    return Linked;
  }

  void apply(ScheduleDAGInstrs *DAGInstrs) override {
    const GCNSubtarget &ST = DAGInstrs->MF.getSubtarget<GCNSubtarget>();
    if (!ST.hasMAIInsts() || DisablePowerSched)
      return;
    DAG = static_cast<ScheduleDAGMI*>(DAGInstrs);
    const TargetSchedModel *TSchedModel = DAGInstrs->getSchedModel();
    if (!TSchedModel || DAG->SUnits.empty())
      return;

    // Scan for MFMA long latency instructions and try to add a dependency
    // of available SALU instructions to give them a chance to fill MFMA
    // shadow. That is desirable to fill MFMA shadow with SALU instructions
    // rather than VALU to prevent power consumption bursts and throttle.
    auto LastSALU = DAG->SUnits.begin();
    auto E = DAG->SUnits.end();
    SmallPtrSet<SUnit*, 32> Visited;
    for (SUnit &SU : DAG->SUnits) {
      MachineInstr &MAI = *SU.getInstr();
      if (!TII->isMAI(MAI) ||
           MAI.getOpcode() == AMDGPU::V_ACCVGPR_WRITE_B32 ||
           MAI.getOpcode() == AMDGPU::V_ACCVGPR_READ_B32)
        continue;

      unsigned Lat = TSchedModel->computeInstrLatency(&MAI) - 1;

      LLVM_DEBUG(dbgs() << "Found MFMA: "; DAG->dumpNode(SU);
                 dbgs() << "Need " << Lat
                        << " instructions to cover latency.\n");

      // Find up to Lat independent scalar instructions as early as
      // possible such that they can be scheduled after this MFMA.
      for ( ; Lat && LastSALU != E; ++LastSALU) {
        if (Visited.count(&*LastSALU))
          continue;

        if (!isSALU(&*LastSALU) || !canAddEdge(&*LastSALU, &SU))
          continue;

        Lat -= linkSALUChain(&SU, &*LastSALU, Lat, Visited);
      }
    }
  }
};
} // namespace

void GCNSubtarget::getPostRAMutations(
    std::vector<std::unique_ptr<ScheduleDAGMutation>> &Mutations) const {
  Mutations.push_back(std::make_unique<FillMFMAShadowMutation>(&InstrInfo));
}

const AMDGPUSubtarget &AMDGPUSubtarget::get(const MachineFunction &MF) {
  if (MF.getTarget().getTargetTriple().getArch() == Triple::amdgcn)
    return static_cast<const AMDGPUSubtarget&>(MF.getSubtarget<GCNSubtarget>());
  else
    return static_cast<const AMDGPUSubtarget&>(MF.getSubtarget<R600Subtarget>());
}

const AMDGPUSubtarget &AMDGPUSubtarget::get(const TargetMachine &TM, const Function &F) {
  if (TM.getTargetTriple().getArch() == Triple::amdgcn)
    return static_cast<const AMDGPUSubtarget&>(TM.getSubtarget<GCNSubtarget>(F));
  else
    return static_cast<const AMDGPUSubtarget&>(TM.getSubtarget<R600Subtarget>(F));
}
