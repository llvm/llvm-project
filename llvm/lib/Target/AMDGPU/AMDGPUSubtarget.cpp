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
#include "AMDGPUCallLowering.h"
#include "R600ISelLowering.h"
#include "R600InstrInfo.h"
#include "R600MachineScheduler.h"
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

#ifdef LLVM_BUILD_GLOBAL_ISEL
namespace {
struct AMDGPUGISelActualAccessor : public GISelAccessor {
  std::unique_ptr<CallLowering> CallLoweringInfo;
  const CallLowering *getCallLowering() const override {
    return CallLoweringInfo.get();
  }
};
} // End anonymous namespace.
#endif

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
    FullFS += "+flat-for-global,";
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
                                 TargetMachine &TM)
    : AMDGPUGenSubtargetInfo(TT, GPU, FS),
      DumpCode(false), R600ALUInst(false), HasVertexCache(false),
      TexVTXClauseSize(0),
      Gen(TT.getArch() == Triple::amdgcn ? SOUTHERN_ISLANDS : R600),
      FP64(false),
      FP64Denormals(false), FP32Denormals(false), FPExceptions(false),
      FastFMAF32(false), HalfRate64Ops(false), CaymanISA(false),
      FlatAddressSpace(false), FlatForGlobal(false), EnableIRStructurizer(true),
      EnablePromoteAlloca(false),
      EnableIfCvt(true), EnableLoadStoreOpt(false),
      EnableUnsafeDSOffsetFolding(false),
      EnableXNACK(false),
      WavefrontSize(64), CFALUBug(false),
      LocalMemorySize(0), MaxPrivateElementSize(0),
      EnableVGPRSpilling(false), SGPRInitBug(false), IsGCN(false),
      GCN1Encoding(false), GCN3Encoding(false), CIInsts(false),
      HasSMemRealTime(false), Has16BitInsts(false),
      LDSBankCount(0),
      IsaVersion(ISAVersion0_0_0),
      EnableSIScheduler(false),
      DebuggerInsertNops(false), DebuggerReserveRegs(false),
      FrameLowering(nullptr),
      GISel(),
      InstrItins(getInstrItineraryForCPU(GPU)), TargetTriple(TT) {

  initializeSubtargetDependencies(TT, GPU, FS);

  // Scratch is allocated in 256 dword per wave blocks.
  const unsigned StackAlign = 4 * 256 / getWavefrontSize();

  if (getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS) {
    InstrInfo.reset(new R600InstrInfo(*this));
    TLInfo.reset(new R600TargetLowering(TM, *this));

    // FIXME: Should have R600 specific FrameLowering
    FrameLowering.reset(new AMDGPUFrameLowering(
                          TargetFrameLowering::StackGrowsUp,
                          StackAlign,
                          0));
  } else {
    InstrInfo.reset(new SIInstrInfo(*this));
    TLInfo.reset(new SITargetLowering(TM, *this));
    FrameLowering.reset(new SIFrameLowering(
                          TargetFrameLowering::StackGrowsUp,
                          StackAlign,
                          0));
#ifndef LLVM_BUILD_GLOBAL_ISEL
    GISelAccessor *GISel = new GISelAccessor();
#else
    AMDGPUGISelActualAccessor *GISel =
        new AMDGPUGISelActualAccessor();
    GISel->CallLoweringInfo.reset(
        new AMDGPUCallLowering(*getTargetLowering()));
#endif
    setGISelAccessor(*GISel);
  }
}

const CallLowering *AMDGPUSubtarget::getCallLowering() const {
  assert(GISel && "Access to GlobalISel APIs not set");
  return GISel->getCallLowering();
}

unsigned AMDGPUSubtarget::getStackEntrySize() const {
  assert(getGeneration() <= NORTHERN_ISLANDS);
  switch(getWavefrontSize()) {
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

unsigned AMDGPUSubtarget::getAmdKernelCodeChipID() const {
  switch(getGeneration()) {
  default: llvm_unreachable("ChipID unknown");
  case SEA_ISLANDS: return 12;
  }
}

AMDGPU::IsaVersion AMDGPUSubtarget::getIsaVersion() const {
  return AMDGPU::getIsaVersion(getFeatureBits());
}

bool AMDGPUSubtarget::isVGPRSpillingEnabled(const Function& F) const {
  return !AMDGPU::isShader(F.getCallingConv()) || EnableVGPRSpilling;
}

void AMDGPUSubtarget::overrideSchedPolicy(MachineSchedPolicy &Policy,
                                          MachineInstr *begin,
                                          MachineInstr *end,
                                          unsigned NumRegionInstrs) const {
  if (getGeneration() >= SOUTHERN_ISLANDS) {

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
}

