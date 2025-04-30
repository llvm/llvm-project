//===-- Next32MCTargetDesc.cpp - Next32 Target Descriptions ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides Next32 specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/Next32MCTargetDesc.h"
#include "MCTargetDesc/Next32InstPrinter.h"
#include "MCTargetDesc/Next32MCAsmInfo.h"
#include "Next32.h"
#include "Next32ELFStreamer.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/TargetParser/Host.h"

#define GET_INSTRINFO_MC_DESC
#define ENABLE_INSTR_PREDICATE_VERIFIER
#include "Next32GenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "Next32GenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "Next32GenRegisterInfo.inc"

using namespace llvm;

static MCInstrInfo *createNext32MCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitNext32MCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createNext32MCRegisterInfo(const Triple &TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitNext32MCRegisterInfo(X, Next32::R11 /* RAReg doesn't exist */);
  return X;
}

static MCSubtargetInfo *
createNext32MCSubtargetInfo(const Triple &TT, StringRef CPU, StringRef FS) {
  return createNext32MCSubtargetInfoImpl(TT, CPU, /*TuneCPU*/ CPU, FS);
}

static MCStreamer *createNext32MCStreamer(
    const Triple &T, MCContext &Ctx, std::unique_ptr<MCAsmBackend> &&MAB,
    std::unique_ptr<MCObjectWriter> &&OW,
    std::unique_ptr<MCCodeEmitter> &&Emitter) {
  return createNext32ELFStreamer(Ctx, std::move(MAB), std::move(OW),
                                 std::move(Emitter));
}

static MCInstPrinter *createNext32MCInstPrinter(const Triple &T,
                                                unsigned SyntaxVariant,
                                                const MCAsmInfo &MAI,
                                                const MCInstrInfo &MII,
                                                const MCRegisterInfo &MRI) {
  if (SyntaxVariant == 0)
    return new Next32InstPrinter(MAI, MII, MRI);
  return nullptr;
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeNext32TargetMC() {
  Target *T = &getTheNext32Target();

  // Register the MC asm info.
  RegisterMCAsmInfo<Next32MCAsmInfo> X(*T);

  // Register the MC instruction info.
  TargetRegistry::RegisterMCInstrInfo(*T, createNext32MCInstrInfo);

  // Register the MC register info.
  TargetRegistry::RegisterMCRegInfo(*T, createNext32MCRegisterInfo);

  // Register the MC subtarget info.
  TargetRegistry::RegisterMCSubtargetInfo(*T, createNext32MCSubtargetInfo);

  // Register the object streamer
  TargetRegistry::RegisterELFStreamer(*T, createNext32MCStreamer);

  // Register the MCInstPrinter.
  TargetRegistry::RegisterMCInstPrinter(*T, createNext32MCInstPrinter);

  // Register the MC code emitter
  TargetRegistry::RegisterMCCodeEmitter(*T, createNext32MCCodeEmitter);

  // Register the ASM Backend
  TargetRegistry::RegisterMCAsmBackend(*T, createNext32AsmBackend);
}
