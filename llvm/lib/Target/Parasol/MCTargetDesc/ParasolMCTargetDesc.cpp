//===-- ParasolMCTargetDesc.cpp - Parasol Target Descriptions
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//
//
// This file provides Parasol specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "ParasolMCTargetDesc.h"
#include "ParasolInstPrinter.h"
#include "ParasolMCAsmInfo.h"
#include "TargetInfo/ParasolTargetInfo.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

#define GET_INSTRINFO_MC_DESC
#include "ParasolGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "ParasolGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "ParasolGenRegisterInfo.inc"

static MCInstrInfo *createParasolMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitParasolMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createParasolMCRegisterInfo(const Triple &TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  // SUNSCREEN TODO: What number do I need to put here?
  InitParasolMCRegisterInfo(X, 0);
  return X;
}

static MCSubtargetInfo *
createParasolMCSubtargetInfo(const Triple &TT, StringRef CPU, StringRef FS) {
  return createParasolMCSubtargetInfoImpl(TT, "parasol", "", FS);
}

static MCInstPrinter *createParasolMCInstPrinter(const Triple &T,
                                                 unsigned SyntaxVariant,
                                                 const MCAsmInfo &MAI,
                                                 const MCInstrInfo &MII,
                                                 const MCRegisterInfo &MRI) {
  return new ParasolInstPrinter(MAI, MII, MRI);
}

static MCAsmInfo *createParasolMCAsmInfo(const MCRegisterInfo &MRI,
                                         const Triple &TT,
                                         const MCTargetOptions &Options) {
  MCAsmInfo *MAI = new ParasolMCAsmInfo(TT);

  unsigned WP = MRI.getDwarfRegNum(Parasol::X2, true);
  // TODO: The original instruction was
  // MCCFIInstruction Inst = MCCFIInstruction::createDefCfa(nullptr, WP, 0);
  // but createDefCfa doesn't exist anymore.
  // createDefCfa:
  // https://cs6340.cc.gatech.edu/LLVM8Doxygen/classllvm_1_1MCCFIInstruction.html#a7914e845a8171a2e975cb6a8122d166c
  // and we are replacing it with
  // createDefCfaRegister:
  // https://llvm.org/doxygen/classllvm_1_1MCCFIInstruction.html#a03445be1c81520587d5bb31b353f5558
  MCCFIInstruction Inst = MCCFIInstruction::createDefCfaRegister(nullptr, WP);
  MAI->addInitialFrameState(Inst);

  return MAI;
}

extern "C" void LLVMInitializeParasolTargetMC() {
  for (Target *T : {&getTheParasolTarget()}) {
    // Register the MC asm info.
    TargetRegistry::RegisterMCAsmInfo(*T, createParasolMCAsmInfo);

    // Register the MC instruction info.
    TargetRegistry::RegisterMCInstrInfo(*T, createParasolMCInstrInfo);

    // Register the MC register info.
    TargetRegistry::RegisterMCRegInfo(*T, createParasolMCRegisterInfo);

    // Register the MC subtarget info.
    TargetRegistry::RegisterMCSubtargetInfo(*T, createParasolMCSubtargetInfo);

    // Register the MCInstPrinter.
    TargetRegistry::RegisterMCInstPrinter(*T, createParasolMCInstPrinter);

    // Register the MC Code Emitter
    TargetRegistry::RegisterMCCodeEmitter(*T, createParasolMCCodeEmitter);

    // Register the asm backend.
    TargetRegistry::RegisterMCAsmBackend(*T, createParasolAsmBackend);
  }
}
