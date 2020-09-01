//===-- MSP430MCTargetDesc.cpp - MSP430 Target Descriptions ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides MSP430 specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "MSP430MCTargetDesc.h"
#include "MSP430InstPrinter.h"
#include "MSP430MCAsmInfo.h"
#include "TargetInfo/MSP430TargetInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

#define GET_INSTRINFO_MC_DESC
#include "MSP430GenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "MSP430GenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "MSP430GenRegisterInfo.inc"

static MCInstrInfo *createMSP430MCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitMSP430MCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createMSP430MCRegisterInfo(const Triple &TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitMSP430MCRegisterInfo(X, MSP430::PC);
  return X;
}

static MCSubtargetInfo *
createMSP430MCSubtargetInfo(const Triple &TT, StringRef CPU, StringRef FS) {
  return createMSP430MCSubtargetInfoImpl(TT, CPU, /*TuneCPU*/ CPU, FS);
}

static MCInstPrinter *createMSP430MCInstPrinter(const Triple &T,
                                                unsigned SyntaxVariant,
                                                const MCAsmInfo &MAI,
                                                const MCInstrInfo &MII,
                                                const MCRegisterInfo &MRI) {
  if (SyntaxVariant == 0)
    return new MSP430InstPrinter(MAI, MII, MRI);
  return nullptr;
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeMSP430TargetMC() {
  Target &T = getTheMSP430Target();

  RegisterMCAsmInfo<MSP430MCAsmInfo> X(T);
  TargetRegistry::RegisterMCInstrInfo(T, createMSP430MCInstrInfo);
  TargetRegistry::RegisterMCRegInfo(T, createMSP430MCRegisterInfo);
  TargetRegistry::RegisterMCSubtargetInfo(T, createMSP430MCSubtargetInfo);
  TargetRegistry::RegisterMCInstPrinter(T, createMSP430MCInstPrinter);
  TargetRegistry::RegisterMCCodeEmitter(T, createMSP430MCCodeEmitter);
  TargetRegistry::RegisterMCAsmBackend(T, createMSP430MCAsmBackend);
  TargetRegistry::RegisterObjectTargetStreamer(
      T, createMSP430ObjectTargetStreamer);
}
