//===-- SuperHMCTargetDesc.cpp - SuperH Target Descriptions ---------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides SuperH target specific descriptions.
///
//===----------------------------------------------------------------------===//

#include "SuperHMCTargetDesc.h"
#include "SuperHInstPrinter.h"
#include "SuperHMCAsmInfo.h"
#include "TargetInfo/SuperHTargetInfo.h"

#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MachineLocation.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"

using namespace llvm;

#define GET_INSTRINFO_MC_DESC
#define ENABLE_INSTR_PREDICATE_VERIFIER
#include "SuperHGenInstrInfo.inc"

#define GET_REGINFO_MC_DESC
#include "SuperHGenRegisterInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "SuperHGenSubtargetInfo.inc"

static MCInstrInfo *createSuperHMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitSuperHMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createSuperHMCRegisterInfo(const Triple &TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitSuperHMCRegisterInfo(X, SH::R0);
  return X;
}

static MCSubtargetInfo *
createSuperHMCSubtargetInfo(const Triple &TT, StringRef CPU, StringRef FS) {
  return createSuperHMCSubtargetInfoImpl(TT, CPU, CPU, FS);
}

static MCInstPrinter *createSuperHMCInstPrinter(const Triple &T,
                                                unsigned SyntaxVariant,
                                                const MCAsmInfo &MAI,
                                                const MCInstrInfo &MII,
                                                const MCRegisterInfo &MRI) {
  return new SuperHInstPrinter(MAI, MII, MRI);
}

static MCAsmInfo *createSuperHMCAsmInfo(const MCRegisterInfo &MRI,
                                        const Triple &TT,
                                        const MCTargetOptions &Options) {
  MCAsmInfo *MAI = new SuperHMCAsmInfo(TT, Options);
  return MAI;
}

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeSuperHTargetMC() {
  for (Target *T : {&getTheSuperHTarget(), &getTheSuperHLETarget()}) {

    // Register the MC instruction info.
    TargetRegistry::RegisterMCInstrInfo(*T, createSuperHMCInstrInfo);

    // Register the MC register info.
    TargetRegistry::RegisterMCRegInfo(*T, createSuperHMCRegisterInfo);

    // Register the MC subtarget info.
    TargetRegistry::RegisterMCSubtargetInfo(*T, createSuperHMCSubtargetInfo);
    
    // Register the MC asm info.
    TargetRegistry::RegisterMCAsmInfo(*T, createSuperHMCAsmInfo);

    // Register the MCInstPrinter.
    TargetRegistry::RegisterMCInstPrinter(*T, createSuperHMCInstPrinter);

    // Register the AsmBackend
    TargetRegistry::RegisterMCAsmBackend(*T, createSuperHAsmBackend);
  }
}