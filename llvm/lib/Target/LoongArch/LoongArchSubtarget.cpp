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
  if (CPU.empty())
    CPU = Is64Bit ? "generic-la64" : "generic-la32";

  if (TuneCPU.empty())
    TuneCPU = CPU;

  ParseSubtargetFeatures(CPU, TuneCPU, FS);
  if (Is64Bit) {
    GRLenVT = MVT::i64;
    GRLen = 64;
  }

  // TODO: ILP32{S,F} LP64{S,F}
  TargetABI = Is64Bit ? LoongArchABI::ABI_LP64D : LoongArchABI::ABI_ILP32D;
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
