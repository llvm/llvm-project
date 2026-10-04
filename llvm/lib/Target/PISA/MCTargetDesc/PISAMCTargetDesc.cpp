//===-- PISAMCTargetDesc.cpp - PISA Target Descriptions -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PISAMCTargetDesc.h"
#include "PISAInstPrinter.h"
#include "PISAMCAsmInfo.h"
#include "TargetInfo/PISATargetInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"

#define GET_INSTRINFO_MC_DESC
#define ENABLE_INSTR_PREDICATE_VERIFIER
#include "PISAGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "PISAGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "PISAGenRegisterInfo.inc"

using namespace llvm;

static MCInstrInfo *createPISAMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitPISAMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createPISAMCRegisterInfo(const Triple &TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitPISAMCRegisterInfo(X, PISA::DummyReg);
  return X;
}

static MCSubtargetInfo *createPISAMCSubtargetInfo(const Triple &TT,
                                                  StringRef CPU, StringRef FS) {
  return createPISAMCSubtargetInfoImpl(TT, CPU, /*TuneCPU=*/CPU, FS);
}

static MCInstPrinter *createPISAMCInstPrinter(const Triple &T,
                                              unsigned SyntaxVariant,
                                              const MCAsmInfo &MAI,
                                              const MCInstrInfo &MII,
                                              const MCRegisterInfo &MRI) {
  assert(SyntaxVariant == 0);
  return new PISAInstPrinter(MAI, MII, MRI);
}

// NOLINTNEXTLINE(readability-identifier-naming)
extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializePISATargetMC() {
  Target *T = &getThePISATarget();
  RegisterMCAsmInfo<PISAMCAsmInfo> X(*T);
  TargetRegistry::RegisterMCInstrInfo(*T, createPISAMCInstrInfo);
  TargetRegistry::RegisterMCRegInfo(*T, createPISAMCRegisterInfo);
  TargetRegistry::RegisterMCSubtargetInfo(*T, createPISAMCSubtargetInfo);
  TargetRegistry::RegisterMCInstPrinter(*T, createPISAMCInstPrinter);
}
