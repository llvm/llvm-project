//===-- LoongArchSubtarget.cpp - LoongArch Subtarget Information -*- C++ -*--=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the LoongArch specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "LoongArchSubtarget.h"
#include "LoongArchFrameLowering.h"
#include "MCTargetDesc/LoongArchBaseInfo.h"

using namespace llvm;

#define DEBUG_TYPE "loongarch-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "LoongArchGenSubtargetInfo.inc"

void LoongArchSubtarget::anchor() {}

LoongArchSubtarget &LoongArchSubtarget::initializeSubtargetDependencies(
    const Triple &TT, StringRef CPU, StringRef TuneCPU, StringRef FS,
    StringRef ABIName) {
  bool Is64Bit = TT.isArch64Bit();
  if (CPU.empty() || CPU == "generic")
    CPU = Is64Bit ? "generic-la64" : "generic-la32";

  if (TuneCPU.empty())
    TuneCPU = CPU;

  ParseSubtargetFeatures(CPU, TuneCPU, FS);
  if (Is64Bit) {
    GRLenVT = MVT::i64;
    GRLen = 64;
  }

  if (HasLA32 == HasLA64)
    report_fatal_error("Please use one feature of 32bit and 64bit.");

  if (Is64Bit && HasLA32)
    report_fatal_error("Feature 32bit should be used for loongarch32 target.");

  if (!Is64Bit && HasLA64)
    report_fatal_error("Feature 64bit should be used for loongarch64 target.");

  TargetABI = LoongArchABI::computeTargetABI(TT, ABIName);

  return *this;
}

LoongArchSubtarget::LoongArchSubtarget(const Triple &TT, StringRef CPU,
                                       StringRef TuneCPU, StringRef FS,
                                       StringRef ABIName,
                                       const TargetMachine &TM)
    : LoongArchGenSubtargetInfo(TT, CPU, TuneCPU, FS),
      FrameLowering(
          initializeSubtargetDependencies(TT, CPU, TuneCPU, FS, ABIName)),
      InstrInfo(*this), RegInfo(getHwMode()), TLInfo(TM, *this) {}
