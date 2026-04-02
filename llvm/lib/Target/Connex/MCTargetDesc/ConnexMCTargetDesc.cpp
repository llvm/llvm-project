//===-- ConnexMCTargetDesc.cpp - Connex Target Descriptions ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides Connex specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "ConnexMCTargetDesc.h"
#include "Connex.h"
#include "ConnexInstPrinter.h"
#include "ConnexMCAsmInfo.h"
#include "TargetInfo/ConnexTargetInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/ErrorHandling.h"

#define GET_INSTRINFO_MC_DESC
#include "ConnexGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "ConnexGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "ConnexGenRegisterInfo.inc"

using namespace llvm;

static MCInstrInfo *createConnexMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitConnexMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createConnexMCRegisterInfo(const Triple &TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitConnexMCRegisterInfo(X, Connex::R11 /* RAReg doesn't exist */);
  return X;
}

static MCSubtargetInfo *
createConnexMCSubtargetInfo(const Triple &TT, StringRef CPU, StringRef FS) {
  return createConnexMCSubtargetInfoImpl(TT, CPU, CPU, FS);
}

static MCStreamer *createConnexMCStreamer(const Triple &T, MCContext &Ctx,
                                          std::unique_ptr<MCAsmBackend> &&MAB,
                                          std::unique_ptr<MCObjectWriter> &&OW,
                                          std::unique_ptr<MCCodeEmitter>
                                              &&Emitter) {
  return createELFStreamer(Ctx, std::move(MAB), std::move(OW),
                           std::move(Emitter));
}

static MCInstPrinter *createConnexMCInstPrinter(const Triple &T,
                                                unsigned SyntaxVariant,
                                                const MCAsmInfo &MAI,
                                                const MCInstrInfo &MII,
                                                const MCRegisterInfo &MRI) {
  if (SyntaxVariant == 0)
    return new ConnexInstPrinter(MAI, MII, MRI);
  return nullptr;
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeConnexTargetMC() {
  for (Target *T : {&getTheConnexTarget()}) {
    // Register the MC asm info.
    RegisterMCAsmInfo<ConnexMCAsmInfo> X(*T);

    // Register the MC instruction info.
    TargetRegistry::RegisterMCInstrInfo(*T, createConnexMCInstrInfo);

    // Register the MC register info.
    TargetRegistry::RegisterMCRegInfo(*T, createConnexMCRegisterInfo);

    // Register the MC subtarget info.
    TargetRegistry::RegisterMCSubtargetInfo(*T, createConnexMCSubtargetInfo);

    // Register the object streamer
    TargetRegistry::RegisterELFStreamer(*T, createConnexMCStreamer);

    // Register the MCInstPrinter.
    TargetRegistry::RegisterMCInstPrinter(*T, createConnexMCInstPrinter);
  }

  // Register the MC code emitter
  TargetRegistry::RegisterMCCodeEmitter(getTheConnexTarget(),
                                        createConnexMCCodeEmitter);

  // Register the ASM Backend
  TargetRegistry::RegisterMCAsmBackend(getTheConnexTarget(), createConnexAsmBackend);
}
