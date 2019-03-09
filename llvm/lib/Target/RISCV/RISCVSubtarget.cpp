//===-- RISCVSubtarget.cpp - RISCV Subtarget Information ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the RISCV specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "RISCVSubtarget.h"
#include "RISCV.h"
#include "RISCVFrameLowering.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "RISCVGenSubtargetInfo.inc"

void RISCVSubtarget::anchor() {}

RISCVSubtarget &RISCVSubtarget::initializeSubtargetDependencies(
    const Triple &TT, StringRef CPU, StringRef FS, StringRef ABIName) {
  // Determine default and user-specified characteristics
  bool Is64Bit = TT.isArch64Bit();
  std::string CPUName = CPU;
  if (CPUName.empty())
    CPUName = Is64Bit ? "generic-rv64" : "generic-rv32";
  ParseSubtargetFeatures(CPUName, FS);
  if (Is64Bit) {
    XLenVT = MVT::i64;
    XLen = 64;
  }

  TargetABI = RISCVABI::computeTargetABI(TT, getFeatureBits(), ABIName);
  return *this;
}

RISCVSubtarget::RISCVSubtarget(const Triple &TT, StringRef CPU, StringRef FS,
                               StringRef ABIName, const TargetMachine &TM)
    : RISCVGenSubtargetInfo(TT, CPU, FS),
      FrameLowering(initializeSubtargetDependencies(TT, CPU, FS, ABIName)),
      InstrInfo(), RegInfo(getHwMode()), TLInfo(TM, *this) {}
