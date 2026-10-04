//===-- PISASubtarget.cpp - PISA Subtarget Information --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PISASubtarget.h"
#include "PISA.h"
#include "PISATargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "pisa-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "PISAGenSubtargetInfo.inc"

PISASubtarget::PISASubtarget(const Triple &TT, const std::string &CPU,
                             const std::string &FS, const PISATargetMachine &TM)
    : PISAGenSubtargetInfo(TT, CPU, /*TuneCPU=*/CPU, FS), InstrInfo(*this),
      FrameLowering(initSubtargetDependencies(CPU, FS)) {}

PISASubtarget &PISASubtarget::initSubtargetDependencies(StringRef CPU,
                                                        StringRef FS) {
  ParseSubtargetFeatures(CPU, /*TuneCPU=*/CPU, FS);
  return *this;
}
