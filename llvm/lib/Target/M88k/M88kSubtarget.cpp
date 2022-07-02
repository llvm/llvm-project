//===-- M88kSubtarget.cpp - M88k Subtarget Information ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the M88k specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "M88kSubtarget.h"
#include "GISel/M88kCallLowering.h"
#include "GISel/M88kLegalizerInfo.h"
#include "GISel/M88kRegisterBankInfo.h"
#include "M88k.h"
#include "M88kTargetMachine.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

#define DEBUG_TYPE "m88k-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "M88kGenSubtargetInfo.inc"

M88kSubtarget &M88kSubtarget::initializeSubtargetDependencies(
    StringRef CPUString, StringRef TuneCPUString, StringRef FS) {
  if (CPUString.empty())
    CPUString = "mc88100"; // TODO Is default to mc88100 the correct choice?

  if (TuneCPUString.empty())
    TuneCPUString = CPUString;

  ParseSubtargetFeatures(CPUString, TuneCPUString, FS);

  return *this;
}

M88kSubtarget::M88kSubtarget(const Triple &TT, const std::string &CPU,
                             const std::string &TuneCPU, const std::string &FS,
                             const TargetMachine &TM)
    : M88kGenSubtargetInfo(TT, CPU, TuneCPU, FS), TargetTriple(TT),
      InstrInfo(initializeSubtargetDependencies(CPU, TuneCPU, FS)),
      TLInfo(TM, *this), FrameLowering(*this) {
  // GlobalISEL
  CallLoweringInfo.reset(new M88kCallLowering(*getTargetLowering()));
  Legalizer.reset(new M88kLegalizerInfo(*this));
  auto *RBI = new M88kRegisterBankInfo(*getRegisterInfo());
  RegBankInfo.reset(RBI);
  InstSelector.reset(createM88kInstructionSelector(
      *static_cast<const M88kTargetMachine *>(&TM), *this, *RBI));
}

Optional<unsigned> M88kSubtarget::getCacheSize(unsigned Level) const {
  if (Level > 0)
    return Optional<unsigned>();
  switch (M88kProc) {
  case MC88100:
    return 16 * 1024; // 16k sharec I+D cache.
  case MC88110:
    return 8 * 1024; // 8k sharec D cache.
  default:
    return Optional<unsigned>();
  }
}

Optional<unsigned> M88kSubtarget::getCacheAssociativity(unsigned Level) const {
  if (Level > 0)
    return Optional<unsigned>();
  switch (M88kProc) {
  case MC88100:
    return 4; // Cache is 4-way associative.
  case MC88110:
    return 2; // Cache is 4-way associative.
  default:
    return Optional<unsigned>();
  }
}

Optional<unsigned> M88kSubtarget::getCacheLineSize(unsigned Level) const {
  if (Level > 0)
    return Optional<unsigned>();
  switch (M88kProc) {
  case MC88100:
    return 16; // 4 bytes.
  case MC88110:
    return 32; // 8 words.
  default:
    return Optional<unsigned>();
  }
}
