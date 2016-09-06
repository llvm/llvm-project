//===-- BPFMCTargetDesc.cpp - BPF Target Descriptions ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides BPF specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "BPF.h"
#include "BPFMCTargetDesc.h"
#include "BPFMCAsmInfo.h"
#include "InstPrinter/BPFInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

#define GET_INSTRINFO_MC_DESC
#include "BPFGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "BPFGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "BPFGenRegisterInfo.inc"

using namespace llvm;

static MCInstrInfo *createBPFMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitBPFMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createBPFMCRegisterInfo(const Triple &TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitBPFMCRegisterInfo(X, BPF::R11 /* RAReg doesn't exist */);
  return X;
}

static MCSubtargetInfo *createBPFMCSubtargetInfo(const Triple &TT,
                                                 StringRef CPU, StringRef FS) {
  return createBPFMCSubtargetInfoImpl(TT, CPU, FS);
}

static MCStreamer *createBPFMCStreamer(const Triple &T,
                                       MCContext &Ctx, MCAsmBackend &MAB,
                                       raw_pwrite_stream &OS, MCCodeEmitter *Emitter,
                                       bool RelaxAll) {
  return createELFStreamer(Ctx, MAB, OS, Emitter, RelaxAll);
}

static MCInstPrinter *createBPFMCInstPrinter(const Triple &T,
                                             unsigned SyntaxVariant,
                                             const MCAsmInfo &MAI,
                                             const MCInstrInfo &MII,
                                             const MCRegisterInfo &MRI) {
  if (SyntaxVariant == 0)
    return new BPFInstPrinter(MAI, MII, MRI);
  return 0;
}

extern "C" void LLVMInitializeBPFTargetMC() {
  for (Target *T : {&TheBPFleTarget, &TheBPFbeTarget, &TheBPFTarget}) {
    // Register the MC asm info.
    RegisterMCAsmInfo<BPFMCAsmInfo> X(*T);

    // Register the MC instruction info.
    TargetRegistry::RegisterMCInstrInfo(*T, createBPFMCInstrInfo);

    // Register the MC register info.
    TargetRegistry::RegisterMCRegInfo(*T, createBPFMCRegisterInfo);

    // Register the MC subtarget info.
    TargetRegistry::RegisterMCSubtargetInfo(*T,
                                            createBPFMCSubtargetInfo);

    // Register the object streamer
    TargetRegistry::RegisterELFStreamer(*T, createBPFMCStreamer);

    // Register the MCInstPrinter.
    TargetRegistry::RegisterMCInstPrinter(*T, createBPFMCInstPrinter);
  }

  // Register the MC code emitter
  TargetRegistry::RegisterMCCodeEmitter(TheBPFleTarget, createBPFMCCodeEmitter);
  TargetRegistry::RegisterMCCodeEmitter(TheBPFbeTarget, createBPFbeMCCodeEmitter);

  // Register the ASM Backend
  TargetRegistry::RegisterMCAsmBackend(TheBPFleTarget, createBPFAsmBackend);
  TargetRegistry::RegisterMCAsmBackend(TheBPFbeTarget, createBPFbeAsmBackend);

  if (sys::IsLittleEndianHost) {
    TargetRegistry::RegisterMCCodeEmitter(TheBPFTarget, createBPFMCCodeEmitter);
    TargetRegistry::RegisterMCAsmBackend(TheBPFTarget, createBPFAsmBackend);
  } else {
    TargetRegistry::RegisterMCCodeEmitter(TheBPFTarget, createBPFbeMCCodeEmitter);
    TargetRegistry::RegisterMCAsmBackend(TheBPFTarget, createBPFbeAsmBackend);
  }
}
