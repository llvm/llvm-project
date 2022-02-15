//===-- LoongArchMCTargetDesc.cpp - LoongArch Target Descriptions ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides LoongArch specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "LoongArchMCTargetDesc.h"
#include "LoongArchBaseInfo.h"
#include "LoongArchInstPrinter.h"
#include "LoongArchMCAsmInfo.h"
#include "TargetInfo/LoongArchTargetInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"

#define GET_INSTRINFO_MC_DESC
#include "LoongArchGenInstrInfo.inc"

#define GET_REGINFO_MC_DESC
#include "LoongArchGenRegisterInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "LoongArchGenSubtargetInfo.inc"

using namespace llvm;

static MCRegisterInfo *createLoongArchMCRegisterInfo(const Triple &TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitLoongArchMCRegisterInfo(X, LoongArch::R1);
  return X;
}

static MCInstrInfo *createLoongArchMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitLoongArchMCInstrInfo(X);
  return X;
}

static MCSubtargetInfo *
createLoongArchMCSubtargetInfo(const Triple &TT, StringRef CPU, StringRef FS) {
  if (CPU.empty())
    CPU = TT.isArch64Bit() ? "la464" : "generic-la32";
  return createLoongArchMCSubtargetInfoImpl(TT, CPU, /*TuneCPU*/ CPU, FS);
}

static MCAsmInfo *createLoongArchMCAsmInfo(const MCRegisterInfo &MRI,
                                           const Triple &TT,
                                           const MCTargetOptions &Options) {
  MCAsmInfo *MAI = new LoongArchMCAsmInfo(TT);

  MCRegister SP = MRI.getDwarfRegNum(LoongArch::R2, true);
  MCCFIInstruction Inst = MCCFIInstruction::cfiDefCfa(nullptr, SP, 0);
  MAI->addInitialFrameState(Inst);

  return MAI;
}

static MCInstPrinter *createLoongArchMCInstPrinter(const Triple &T,
                                                   unsigned SyntaxVariant,
                                                   const MCAsmInfo &MAI,
                                                   const MCInstrInfo &MII,
                                                   const MCRegisterInfo &MRI) {
  return new LoongArchInstPrinter(MAI, MII, MRI);
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeLoongArchTargetMC() {
  for (Target *T : {&getTheLoongArch32Target(), &getTheLoongArch64Target()}) {
    // Register the MC register info.
    TargetRegistry::RegisterMCRegInfo(*T, createLoongArchMCRegisterInfo);

    // Register the MC instruction info.
    TargetRegistry::RegisterMCInstrInfo(*T, createLoongArchMCInstrInfo);

    // Register the MC subtarget info.
    TargetRegistry::RegisterMCSubtargetInfo(*T, createLoongArchMCSubtargetInfo);

    // Register the MC asm info.
    TargetRegistry::RegisterMCAsmInfo(*T, createLoongArchMCAsmInfo);

    // Register the MC Code Emitter
    TargetRegistry::RegisterMCCodeEmitter(*T, createLoongArchMCCodeEmitter);

    // Register the asm backend.
    TargetRegistry::RegisterMCAsmBackend(*T, createLoongArchAsmBackend);

    // Register the MCInstPrinter.
    TargetRegistry::RegisterMCInstPrinter(*T, createLoongArchMCInstPrinter);
  }
}
