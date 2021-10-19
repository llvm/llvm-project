//===-- CSKYMCTargetDesc.cpp - CSKY Target Descriptions -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file provides CSKY specific target descriptions.
///
//===----------------------------------------------------------------------===//

#include "CSKYMCTargetDesc.h"
#include "CSKYAsmBackend.h"
#include "CSKYInstPrinter.h"
#include "CSKYMCAsmInfo.h"
#include "CSKYMCCodeEmitter.h"
#include "TargetInfo/CSKYTargetInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"

#define GET_INSTRINFO_MC_DESC
#include "CSKYGenInstrInfo.inc"

#define GET_REGINFO_MC_DESC
#include "CSKYGenRegisterInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "CSKYGenSubtargetInfo.inc"

using namespace llvm;

static MCAsmInfo *createCSKYMCAsmInfo(const MCRegisterInfo &MRI,
                                      const Triple &TT,
                                      const MCTargetOptions &Options) {
  MCAsmInfo *MAI = new CSKYMCAsmInfo(TT);

  // Initial state of the frame pointer is SP.
  unsigned Reg = MRI.getDwarfRegNum(CSKY::R14, true);
  MCCFIInstruction Inst = MCCFIInstruction::cfiDefCfa(nullptr, Reg, 0);
  MAI->addInitialFrameState(Inst);
  return MAI;
}

static MCInstrInfo *createCSKYMCInstrInfo() {
  MCInstrInfo *Info = new MCInstrInfo();
  InitCSKYMCInstrInfo(Info);
  return Info;
}

static MCInstPrinter *createCSKYMCInstPrinter(const Triple &T,
                                              unsigned SyntaxVariant,
                                              const MCAsmInfo &MAI,
                                              const MCInstrInfo &MII,
                                              const MCRegisterInfo &MRI) {
  return new CSKYInstPrinter(MAI, MII, MRI);
}

static MCRegisterInfo *createCSKYMCRegisterInfo(const Triple &TT) {
  MCRegisterInfo *Info = new MCRegisterInfo();
  InitCSKYMCRegisterInfo(Info, CSKY::R15);
  return Info;
}

static MCSubtargetInfo *createCSKYMCSubtargetInfo(const Triple &TT,
                                                  StringRef CPU, StringRef FS) {
  std::string CPUName = std::string(CPU);
  if (CPUName.empty())
    CPUName = "generic";
  return createCSKYMCSubtargetInfoImpl(TT, CPUName, /*TuneCPU=*/CPUName, FS);
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeCSKYTargetMC() {
  auto &CSKYTarget = getTheCSKYTarget();
  TargetRegistry::RegisterMCAsmBackend(CSKYTarget, createCSKYAsmBackend);
  TargetRegistry::RegisterMCAsmInfo(CSKYTarget, createCSKYMCAsmInfo);
  TargetRegistry::RegisterMCInstrInfo(CSKYTarget, createCSKYMCInstrInfo);
  TargetRegistry::RegisterMCRegInfo(CSKYTarget, createCSKYMCRegisterInfo);
  TargetRegistry::RegisterMCCodeEmitter(CSKYTarget, createCSKYMCCodeEmitter);
  TargetRegistry::RegisterMCInstPrinter(CSKYTarget, createCSKYMCInstPrinter);
  TargetRegistry::RegisterMCSubtargetInfo(CSKYTarget,
                                          createCSKYMCSubtargetInfo);
}
