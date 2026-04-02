//===-- ConnexSubtarget.cpp - Connex Subtarget Information ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Connex specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "ConnexSubtarget.h"
#include "Connex.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

#define DEBUG_TYPE "connex-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "ConnexGenSubtargetInfo.inc"

void ConnexSubtarget::anchor() {}

// Inspired from lib/Target/BPF/BPFSubtarget.h (from Oct 2025)
ConnexSubtarget &ConnexSubtarget::initializeSubtargetDependencies(StringRef CPU,
                                                                 StringRef FS) {
  /*
  initializeEnvironment();
  initSubtargetFeatures(CPU, FS);
  */
  ParseSubtargetFeatures(CPU, /*TuneCPU*/ CPU, FS);
  return *this;
}

ConnexSubtarget::ConnexSubtarget(const Triple &TT, const std::string &CPU,
                                 const std::string &FS, const TargetMachine &TM)
    : ConnexGenSubtargetInfo(TT, CPU, /*TuneCPU*/ CPU, FS),
      InstrInfo(initializeSubtargetDependencies(CPU, FS)),
      FrameLowering(*this), TLInfo(TM, *this), TSInfo2() {}
