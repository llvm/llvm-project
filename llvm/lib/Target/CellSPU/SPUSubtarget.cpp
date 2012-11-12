//===-- SPUSubtarget.cpp - STI Cell SPU Subtarget Information -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CellSPU-specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "SPUSubtarget.h"
#include "SPU.h"
#include "SPURegisterInfo.h"
#include "llvm/Support/TargetRegistry.h"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "SPUGenSubtargetInfo.inc"

using namespace llvm;

SPUSubtarget::SPUSubtarget(const std::string &TT, const std::string &CPU,
                           const std::string &FS) :
  SPUGenSubtargetInfo(TT, CPU, FS),
  StackAlignment(16),
  ProcDirective(SPU::DEFAULT_PROC),
  UseLargeMem(false)
{
  // Should be the target SPU processor type. For now, since there's only
  // one, simply default to the current "v0" default:
  std::string default_cpu("v0");

  // Parse features string.
  ParseSubtargetFeatures(default_cpu, FS);

  // Initialize scheduling itinerary for the specified CPU.
  InstrItins = getInstrItineraryForCPU(default_cpu);
}

/// SetJITMode - This is called to inform the subtarget info that we are
/// producing code for the JIT.
void SPUSubtarget::SetJITMode() {
}

/// Enable PostRA scheduling for optimization levels -O2 and -O3.
bool SPUSubtarget::enablePostRAScheduler(
                       CodeGenOpt::Level OptLevel,
                       TargetSubtargetInfo::AntiDepBreakMode& Mode,
                       RegClassVector& CriticalPathRCs) const {
  Mode = TargetSubtargetInfo::ANTIDEP_CRITICAL;
  // CriticalPathsRCs seems to be the set of
  // RegisterClasses that antidep breakings are performed for.
  // Do it for all register classes 
  CriticalPathRCs.clear();
  CriticalPathRCs.push_back(&SPU::R8CRegClass);
  CriticalPathRCs.push_back(&SPU::R16CRegClass);
  CriticalPathRCs.push_back(&SPU::R32CRegClass);
  CriticalPathRCs.push_back(&SPU::R32FPRegClass);
  CriticalPathRCs.push_back(&SPU::R64CRegClass);
  CriticalPathRCs.push_back(&SPU::VECREGRegClass);
  return OptLevel >= CodeGenOpt::Default;
}
