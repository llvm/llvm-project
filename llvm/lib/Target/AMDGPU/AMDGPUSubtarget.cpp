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
#include "R600MachineScheduler.h"
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

  SmallString<256> FullFS("+promote-alloca,+fp64-denormals,");
  FullFS += FS;

  if (GPU == "" && TT.getArch() == Triple::amdgcn)
    GPU = "SI";

  ParseSubtargetFeatures(GPU, FullFS);

  // FIXME: I don't think think Evergreen has any useful support for
  // denormals, but should be checked. Should we issue a warning somewhere
  // if someone tries to enable these?
  if (getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS) {
    FP32Denormals = false;
    FP64Denormals = false;
  }
  return *this;
}

AMDGPUSubtarget::AMDGPUSubtarget(const Triple &TT, StringRef GPU, StringRef FS,
                                 TargetMachine &TM)
    : AMDGPUGenSubtargetInfo(TT, GPU, FS), DevName(GPU), Is64bit(false),
      DumpCode(false), R600ALUInst(false), HasVertexCache(false),
      TexVTXClauseSize(0), Gen(AMDGPUSubtarget::R600), FP64(false),
      FP64Denormals(false), FP32Denormals(false), FastFMAF32(false),
      CaymanISA(false), FlatAddressSpace(false), EnableIRStructurizer(true),
      EnablePromoteAlloca(false), EnableIfCvt(true), EnableLoadStoreOpt(false),
      EnableUnsafeDSOffsetFolding(false),
      WavefrontSize(0), CFALUBug(false), LocalMemorySize(0),
      EnableVGPRSpilling(false), SGPRInitBug(false), IsGCN(false),
      GCN1Encoding(false), GCN3Encoding(false), CIInsts(false), LDSBankCount(0),
      IsaVersion(ISAVersion0_0_0), EnableHugeScratchBuffer(false),
      FrameLowering(TargetFrameLowering::StackGrowsUp,
                    64 * 16, // Maximum stack alignment (long16)
                    0),
      InstrItins(getInstrItineraryForCPU(GPU)), TargetTriple(TT) {

  initializeSubtargetDependencies(TT, GPU, FS);

  if (getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS) {
    InstrInfo.reset(new R600InstrInfo(*this));
    TLInfo.reset(new R600TargetLowering(TM, *this));
  } else {
    InstrInfo.reset(new SIInstrInfo(*this));
    TLInfo.reset(new SITargetLowering(TM, *this));
  }
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

unsigned AMDGPUSubtarget::getAmdKernelCodeChipID() const {
  switch(getGeneration()) {
  default: llvm_unreachable("ChipID unknown");
  case SEA_ISLANDS: return 12;
  }
}

AMDGPU::IsaVersion AMDGPUSubtarget::getIsaVersion() const {
  return AMDGPU::getIsaVersion(getFeatureBits());
}

bool AMDGPUSubtarget::isVGPRSpillingEnabled(
                                       const SIMachineFunctionInfo *MFI) const {
  return MFI->getShaderType() == ShaderType::COMPUTE || EnableVGPRSpilling;
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
  }
}

