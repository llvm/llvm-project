//===-- SPIRVSubtarget.cpp - SPIR-V Subtarget Information ------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SPIR-V specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "SPIRVSubtarget.h"
#include "SPIRV.h"
#include "SPIRVRegisterBankInfo.h"
#include "SPIRVTargetMachine.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Host.h"

using namespace llvm;

#define DEBUG_TYPE "spirv-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "SPIRVGenSubtargetInfo.inc"

// Compare version numbers, but allow 0 to mean unspecified.
static bool isAtLeastVer(uint32_t Target, uint32_t VerToCompareTo) {
  return Target == 0 || Target >= VerToCompareTo;
}

static unsigned computePointerSize(const Triple &TT) {
  const auto Arch = TT.getArch();
  // TODO: unify this with pointers legalization.
  assert(TT.isSPIRV());
  return Arch == Triple::spirv32 ? 32 : 64;
}

SPIRVSubtarget::SPIRVSubtarget(const Triple &TT, const std::string &CPU,
                               const std::string &FS,
                               const SPIRVTargetMachine &TM)
    : SPIRVGenSubtargetInfo(TT, CPU, /*TuneCPU=*/CPU, FS),
      PointerSize(computePointerSize(TT)), SPIRVVersion(0), InstrInfo(),
      FrameLowering(initSubtargetDependencies(CPU, FS)), TLInfo(TM, *this) {
  CallLoweringInfo = std::make_unique<SPIRVCallLowering>(TLInfo);
  RegBankInfo = std::make_unique<SPIRVRegisterBankInfo>();
}

SPIRVSubtarget &SPIRVSubtarget::initSubtargetDependencies(StringRef CPU,
                                                          StringRef FS) {
  ParseSubtargetFeatures(CPU, /*TuneCPU=*/CPU, FS);
  if (SPIRVVersion == 0)
    SPIRVVersion = 14;
  return *this;
}

// If the SPIR-V version is >= 1.4 we can call OpPtrEqual and OpPtrNotEqual.
bool SPIRVSubtarget::canDirectlyComparePointers() const {
  return isAtLeastVer(SPIRVVersion, 14);
}
