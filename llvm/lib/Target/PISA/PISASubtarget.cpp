//===-- PISASubtarget.cpp - PISA Subtarget Information --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PISASubtarget.h"
#include "PISA.h"
#include "PISALegalizerInfo.h"
#include "PISARegisterBankInfo.h"
#include "PISATargetMachine.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/TargetParser/Host.h"

using namespace llvm;

#define DEBUG_TYPE "pisa-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "PISAGenSubtargetInfo.inc"

PISASubtarget::PISASubtarget(const Triple &TT, const std::string &CPU,
                             const std::string &FS, const PISATargetMachine &TM)
    : PISAGenSubtargetInfo(TT, CPU, /*TuneCPU=*/CPU, FS), InstrInfo(*this),
      FrameLowering(initSubtargetDependencies(CPU, FS)), TLInfo(TM, *this) {

  CallLoweringInfo = std::make_unique<PISACallLowering>(TLInfo);
  InlineAsmLoweringInfo = std::make_unique<InlineAsmLowering>(&TLInfo);
  Legalizer = std::make_unique<PISALegalizerInfo>(*this);
  RegBankInfo = std::make_unique<PISARegisterBankInfo>();
  LLT::setUseExtended(true); // enable bfloat support
  // The instruction selector is created in a subsequent change.
}

PISASubtarget &PISASubtarget::initSubtargetDependencies(StringRef CPU,
                                                        StringRef FS) {
  ParseSubtargetFeatures(CPU, /*TuneCPU=*/CPU, FS);
  if (CPU.empty())
    CPU = PISA::stripCPUPrefix(PISA::getDefaultCPUName());
  PISATarget = PISA::getPISATargetInfo(CPU);
  return *this;
}

// Determine compatibility of instruction's PISA target, specified via
// "let Predicates = []", vs. platform's target, specified via -mcpu=
bool PISASubtarget::supportsPISATarget(StringRef Name) const {
  auto InstrPISATarget = PISA::getPISATargetInfo(Name);
  return isCompatiblePISATargetInfo(PISATarget, InstrPISATarget);
}
