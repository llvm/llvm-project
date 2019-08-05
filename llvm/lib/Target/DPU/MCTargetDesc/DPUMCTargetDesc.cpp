//===-- DPUMCTargetDesc.cpp - DPU Target Descriptions ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides DPU specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "InstPrinter/DPUInstPrinter.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

#define GET_REGINFO_ENUM

#include "DPUGenRegisterInfo.inc"

#define GET_INSTRINFO_ENUM

#include "DPUGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM

#include "DPUGenSubtargetInfo.inc"

#define GET_INSTRINFO_MC_DESC

#include "DPUGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC

#include "DPUGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC

#include "DPUGenRegisterInfo.inc"
#include "DPUMCAsmInfo.h"
#include "DPUMCTargetDesc.h"

#define DPU_ABI_VERSION 2

using namespace llvm;

static MCInstPrinter *createDPUMCInstPrinter(const Triple &T,
                                             unsigned SyntaxVariant,
                                             const MCAsmInfo &MAI,
                                             const MCInstrInfo &MII,
                                             const MCRegisterInfo &MRI) {
  if (SyntaxVariant == 0)
    return new DPUInstPrinter(MAI, MII, MRI);
  return 0;
}

static MCRegisterInfo *createDPUMCRegisterInfo(const Triple &TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitDPUMCRegisterInfo(X, DPU::R0);
  return X;
}

static MCInstrInfo *createDPUMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitDPUMCInstrInfo(X);
  return X;
}

static MCSubtargetInfo *createDPUMCSubtargetInfo(const Triple &TT,
                                                 StringRef CPU, StringRef FS) {
  return createDPUMCSubtargetInfoImpl(TT, CPU, FS);
}

class DPUTargetStreamer : public MCTargetStreamer {
public:
  MCELFStreamer &getStreamer() {
    return static_cast<MCELFStreamer &>(Streamer);
  }
  DPUTargetStreamer(MCStreamer &S) : MCTargetStreamer(S) {
    MCAssembler &MCA = getStreamer().getAssembler();
    MCA.setELFHeaderEFlags(
        EF_EABI_DPU_SET(MCA.getELFHeaderEFlags(), DPU_ABI_VERSION));
  }
};

static MCTargetStreamer *
createDPUTargetStreamer(MCStreamer &S, const MCSubtargetInfo &STI) {
  return new DPUTargetStreamer(S);
}

extern "C" void LLVMInitializeDPUTargetMC() {
  // Register the MC asm info.
  RegisterMCAsmInfo<DPUMCAsmInfo> X(TheDPUTarget);

  // Register the MC instruction info.
  TargetRegistry::RegisterMCInstrInfo(TheDPUTarget, createDPUMCInstrInfo);

  // Need the MC register info to emit debug information.
  TargetRegistry::RegisterMCRegInfo(TheDPUTarget, createDPUMCRegisterInfo);

  // Register the MC subtarget info.
  TargetRegistry::RegisterMCSubtargetInfo(TheDPUTarget,
                                          createDPUMCSubtargetInfo);

  // Register the MC code emitter
  TargetRegistry::RegisterMCCodeEmitter(TheDPUTarget, createDPUMCCodeEmitter);

  // Register the MCInstPrinter.
  TargetRegistry::RegisterMCInstPrinter(TheDPUTarget, createDPUMCInstPrinter);

  // Register the Asm Backend.
  TargetRegistry::RegisterMCAsmBackend(TheDPUTarget, createDPUAsmBackend);

  // Register the obj target streamer
  TargetRegistry::RegisterObjectTargetStreamer(TheDPUTarget,
                                               createDPUTargetStreamer);
}
