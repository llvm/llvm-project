//===-- XCoreMCTargetDesc.cpp - XCore Target Descriptions -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides XCore specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "XCoreMCTargetDesc.h"
#include "XCoreMCAsmInfo.h"
#include "llvm/MC/MCCodeGenInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

#define GET_INSTRINFO_MC_DESC
#include "XCoreGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "XCoreGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "XCoreGenRegisterInfo.inc"

using namespace llvm;

static MCInstrInfo *createXCoreMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitXCoreMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createXCoreMCRegisterInfo(StringRef TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitXCoreMCRegisterInfo(X, XCore::LR);
  return X;
}

static MCSubtargetInfo *createXCoreMCSubtargetInfo(StringRef TT, StringRef CPU,
                                                   StringRef FS) {
  MCSubtargetInfo *X = new MCSubtargetInfo();
  InitXCoreMCSubtargetInfo(X, TT, CPU, FS);
  return X;
}

static MCAsmInfo *createXCoreMCAsmInfo(const Target &T, StringRef TT) {
  MCAsmInfo *MAI = new XCoreMCAsmInfo(T, TT);

  // Initial state of the frame pointer is SP.
  MachineLocation Dst(MachineLocation::VirtualFP);
  MachineLocation Src(XCore::SP, 0);
  MAI->addInitialFrameState(0, Dst, Src);

  return MAI;
}

static MCCodeGenInfo *createXCoreMCCodeGenInfo(StringRef TT, Reloc::Model RM,
                                               CodeModel::Model CM,
                                               CodeGenOpt::Level OL) {
  MCCodeGenInfo *X = new MCCodeGenInfo();
  X->InitMCCodeGenInfo(RM, CM, OL);
  return X;
}

// Force static initialization.
extern "C" void LLVMInitializeXCoreTargetMC() {
  // Register the MC asm info.
  RegisterMCAsmInfoFn X(TheXCoreTarget, createXCoreMCAsmInfo);

  // Register the MC codegen info.
  TargetRegistry::RegisterMCCodeGenInfo(TheXCoreTarget,
                                        createXCoreMCCodeGenInfo);

  // Register the MC instruction info.
  TargetRegistry::RegisterMCInstrInfo(TheXCoreTarget, createXCoreMCInstrInfo);

  // Register the MC register info.
  TargetRegistry::RegisterMCRegInfo(TheXCoreTarget, createXCoreMCRegisterInfo);

  // Register the MC subtarget info.
  TargetRegistry::RegisterMCSubtargetInfo(TheXCoreTarget,
                                          createXCoreMCSubtargetInfo);
}
