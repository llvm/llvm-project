//===-- XtensaMCTargetDesc.cpp - Xtebsa target descriptions ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "XtensaMCTargetDesc.h"
#include "XtensaMCAsmInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

#define GET_INSTRINFO_MC_DESC
#include "XtensaGenInstrInfo.inc"

#define GET_REGINFO_MC_DESC
#include "XtensaGenRegisterInfo.inc"

using namespace llvm;

static MCAsmInfo *createXtensaMCAsmInfo(const MCRegisterInfo &MRI,
                                        const Triple &TT) {
  MCAsmInfo *MAI = new XtensaMCAsmInfo(TT);
  return MAI;
}

static MCInstrInfo *createXtensaMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitXtensaMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createXtensaMCRegisterInfo(const Triple &TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitXtensaMCRegisterInfo(X, Xtensa::SP);
  return X;
}

extern "C" void LLVMInitializeXtensaTargetMC() {
  // Register the MCAsmInfo.
  TargetRegistry::RegisterMCAsmInfo(TheXtensaTarget, createXtensaMCAsmInfo);

  // Register the MCCodeEmitter.
  TargetRegistry::RegisterMCCodeEmitter(TheXtensaTarget,
                                        createXtensaMCCodeEmitter);

  // Register the MCInstrInfo.
  TargetRegistry::RegisterMCInstrInfo(TheXtensaTarget, createXtensaMCInstrInfo);

  // Register the MCRegisterInfo.
  TargetRegistry::RegisterMCRegInfo(TheXtensaTarget,
                                    createXtensaMCRegisterInfo);

  // Register the MCAsmBackend.
  TargetRegistry::RegisterMCAsmBackend(TheXtensaTarget,
                                       createXtensaMCAsmBackend);
}
