//===-- AGXSubtarget.cpp - AGX subtarget information ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AGXSubtarget.h"
#include "AGXTargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "agx-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "AGXGenSubtargetInfo.inc"

AGXSubtarget &AGXSubtarget::initializeSubtargetDependencies(StringRef CPU,
                                                            StringRef FS) {
  if (CPU.empty())
    CPU = "generic";
  ParseSubtargetFeatures(CPU, /*TuneCPU=*/CPU, FS);
  return *this;
}

AGXSubtarget::AGXSubtarget(const Triple &TT, StringRef CPU, StringRef FS,
                           const AGXTargetMachine &TM)
    : AGXGenSubtargetInfo(TT, CPU.empty() ? "generic" : CPU,
                          /*TuneCPU=*/CPU.empty() ? "generic" : CPU, FS),
      InstrInfo(initializeSubtargetDependencies(CPU, FS)),
      FrameLowering(*this), TLInfo(TM, *this) {}
