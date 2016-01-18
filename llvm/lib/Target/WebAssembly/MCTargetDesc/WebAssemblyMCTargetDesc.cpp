//===-- WebAssemblyMCTargetDesc.cpp - WebAssembly Target Descriptions -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides WebAssembly-specific target descriptions.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblyMCTargetDesc.h"
#include "InstPrinter/WebAssemblyInstPrinter.h"
#include "WebAssemblyMCAsmInfo.h"
#include "llvm/MC/MCCodeGenInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-mc-target-desc"

#define GET_INSTRINFO_MC_DESC
#include "WebAssemblyGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "WebAssemblyGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "WebAssemblyGenRegisterInfo.inc"

static MCAsmInfo *createWebAssemblyMCAsmInfo(const MCRegisterInfo &MRI,
                                             const Triple &TT) {
  return new WebAssemblyMCAsmInfo(TT);
}

static MCInstrInfo *createWebAssemblyMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitWebAssemblyMCInstrInfo(X);
  return X;
}

static MCInstPrinter *
createWebAssemblyMCInstPrinter(const Triple &T, unsigned SyntaxVariant,
                               const MCAsmInfo &MAI, const MCInstrInfo &MII,
                               const MCRegisterInfo &MRI) {
  assert(SyntaxVariant == 0);
  return new WebAssemblyInstPrinter(MAI, MII, MRI);
}

// Force static initialization.
extern "C" void LLVMInitializeWebAssemblyTargetMC() {
  for (Target *T : {&TheWebAssemblyTarget32, &TheWebAssemblyTarget64}) {
    // Register the MC asm info.
    RegisterMCAsmInfoFn X(*T, createWebAssemblyMCAsmInfo);

    // Register the MC instruction info.
    TargetRegistry::RegisterMCInstrInfo(*T, createWebAssemblyMCInstrInfo);

    // Register the MCInstPrinter.
    TargetRegistry::RegisterMCInstPrinter(*T, createWebAssemblyMCInstPrinter);
  }
}
