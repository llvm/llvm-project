//===-- AMDGPUSubtarget.cpp - AMDGPU Subtarget Information ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Implements the AMDGPU specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUSubtarget.h"
#include "R600ISelLowering.h"
#include "R600InstrInfo.h"
#include "SIFrameLowering.h"
#include "SIISelLowering.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/CodeGen/MachineScheduler.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-subtarget"

#define GET_SUBTARGETINFO_ENUM
#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "AMDGPUGenSubtargetInfo.inc"

AMDGPUSubtarget::~AMDGPUSubtarget() {}

AMDGPUSubtarget &
AMDGPUSubtarget::initializeSubtargetDependencies(const Triple &TT,
                                                 StringRef GPU, StringRef FS) {
  // Determine default and user-specified characteristics
  // On SI+, we want FP64 denormals to be on by default. FP32 denormals can be
  // enabled, but some instructions do not respect them and they run at the
  // double precision rate, so don't enable by default.
  //
  // We want to be able to turn these off, but making this a subtarget feature
  // for SI has the unhelpful behavior that it unsets everything else if you
  // disable it.

  SmallString<256> FullFS("+promote-alloca,+fp64-denormals,+load-store-opt,");
  if (isAmdHsaOS()) // Turn on FlatForGlobal for HSA.
    FullFS += "+flat-for-global,+unaligned-buffer-access,";
  FullFS += FS;

  ParseSubtargetFeatures(GPU, FullFS);

  // FIXME: I don't think think Evergreen has any useful support for
  // denormals, but should be checked. Should we issue a warning somewhere
  // if someone tries to enable these?
  if (getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS) {
    FP32Denormals = false;
    FP64Denormals = false;
  }

  // Set defaults if needed.
  if (MaxPrivateElementSize == 0)
    MaxPrivateElementSize = 4;

  return *this;
}

AMDGPUSubtarget::AMDGPUSubtarget(const Triple &TT, StringRef GPU, StringRef FS,
                                 const TargetMachine &TM)
  : AMDGPUGenSubtargetInfo(TT, GPU, FS),
    TargetTriple(TT),
    Gen(TT.getArch() == Triple::amdgcn ? SOUTHERN_ISLANDS : R600),
    IsaVersion(ISAVersion0_0_0),
    WavefrontSize(64),
    LocalMemorySize(0),
    LDSBankCount(0),
    MaxPrivateElementSize(0),

    FastFMAF32(false),
    HalfRate64Ops(false),

    FP32Denormals(false),
    FP64Denormals(false),
    FPExceptions(false),
    FlatForGlobal(false),
    UnalignedBufferAccess(false),

    EnableXNACK(false),
    DebuggerInsertNops(false),
    DebuggerReserveRegs(false),
    DebuggerEmitPrologue(false),

    EnableVGPRSpilling(false),
    EnablePromoteAlloca(false),
    EnableLoadStoreOpt(false),
    EnableUnsafeDSOffsetFolding(false),
    EnableSIScheduler(false),
    DumpCode(false),

    FP64(false),
    IsGCN(false),
    GCN1Encoding(false),
    GCN3Encoding(false),
    CIInsts(false),
    SGPRInitBug(false),
    HasSMemRealTime(false),
    Has16BitInsts(false),
    FlatAddressSpace(false),

    R600ALUInst(false),
    CaymanISA(false),
    CFALUBug(false),
    HasVertexCache(false),
    TexVTXClauseSize(0),

    FeatureDisable(false),
    InstrItins(getInstrItineraryForCPU(GPU)) {
  initializeSubtargetDependencies(TT, GPU, FS);
}

// FIXME: These limits are for SI. Did they change with the larger maximum LDS
// size?
unsigned AMDGPUSubtarget::getMaxLocalMemSizeWithWaveCount(unsigned NWaves) const {
  switch (NWaves) {
  case 10:
    return 1638;
  case 9:
    return 1820;
  case 8:
    return 2048;
  case 7:
    return 2340;
  case 6:
    return 2730;
  case 5:
    return 3276;
  case 4:
    return 4096;
  case 3:
    return 5461;
  case 2:
    return 8192;
  default:
    return getLocalMemorySize();
  }
}

unsigned AMDGPUSubtarget::getOccupancyWithLocalMemSize(uint32_t Bytes) const {
  if (Bytes <= 1638)
    return 10;

  if (Bytes <= 1820)
    return 9;

  if (Bytes <= 2048)
    return 8;

  if (Bytes <= 2340)
    return 7;

  if (Bytes <= 2730)
    return 6;

  if (Bytes <= 3276)
    return 5;

  if (Bytes <= 4096)
    return 4;

  if (Bytes <= 5461)
    return 3;

  if (Bytes <= 8192)
    return 2;

  return 1;
}

R600Subtarget::R600Subtarget(const Triple &TT, StringRef GPU, StringRef FS,
                             const TargetMachine &TM) :
  AMDGPUSubtarget(TT, GPU, FS, TM),
  InstrInfo(*this),
  FrameLowering(TargetFrameLowering::StackGrowsUp, getStackAlignment(), 0),
  TLInfo(TM, *this) {}

SISubtarget::SISubtarget(const Triple &TT, StringRef GPU, StringRef FS,
                         const TargetMachine &TM) :
  AMDGPUSubtarget(TT, GPU, FS, TM),
  InstrInfo(*this),
  FrameLowering(TargetFrameLowering::StackGrowsUp, getStackAlignment(), 0),
  TLInfo(TM, *this),
  GISel() {}

unsigned R600Subtarget::getStackEntrySize() const {
  switch (getWavefrontSize()) {
  case 16:
    return 8;
  case 32:
    return hasCaymanISA() ? 4 : 8;
  case 64:
    return 4;
  default:
    llvm_unreachable("Illegal wavefront size.");
  }
}

void SISubtarget::overrideSchedPolicy(MachineSchedPolicy &Policy,
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

bool SISubtarget::isVGPRSpillingEnabled(const Function& F) const {
  return EnableVGPRSpilling || !AMDGPU::isShader(F.getCallingConv());
}

unsigned SISubtarget::getAmdKernelCodeChipID() const {
  switch (getGeneration()) {
  case SEA_ISLANDS:
    return 12;
  default:
    llvm_unreachable("ChipID unknown");
  }
}

AMDGPU::IsaVersion SISubtarget::getIsaVersion() const {
  return AMDGPU::getIsaVersion(getFeatureBits());
}
