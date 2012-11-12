//===-- SparcMCTargetDesc.cpp - Sparc Target Descriptions -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides Sparc specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "SparcMCTargetDesc.h"
#include "SparcMCAsmInfo.h"
#include "llvm/MC/MCCodeGenInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

#define GET_INSTRINFO_MC_DESC
#include "SparcGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "SparcGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "SparcGenRegisterInfo.inc"

using namespace llvm;

static MCInstrInfo *createSparcMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitSparcMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createSparcMCRegisterInfo(StringRef TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitSparcMCRegisterInfo(X, SP::I7);
  return X;
}

static MCSubtargetInfo *createSparcMCSubtargetInfo(StringRef TT, StringRef CPU,
                                                   StringRef FS) {
  MCSubtargetInfo *X = new MCSubtargetInfo();
  InitSparcMCSubtargetInfo(X, TT, CPU, FS);
  return X;
}

static MCCodeGenInfo *createSparcMCCodeGenInfo(StringRef TT, Reloc::Model RM,
                                               CodeModel::Model CM,
                                               CodeGenOpt::Level OL) {
  MCCodeGenInfo *X = new MCCodeGenInfo();
  X->InitMCCodeGenInfo(RM, CM, OL);
  return X;
}

extern "C" void LLVMInitializeSparcTargetMC() {
  // Register the MC asm info.
  RegisterMCAsmInfo<SparcELFMCAsmInfo> X(TheSparcTarget);
  RegisterMCAsmInfo<SparcELFMCAsmInfo> Y(TheSparcV9Target);

  // Register the MC codegen info.
  TargetRegistry::RegisterMCCodeGenInfo(TheSparcTarget,
                                       createSparcMCCodeGenInfo);
  TargetRegistry::RegisterMCCodeGenInfo(TheSparcV9Target,
                                       createSparcMCCodeGenInfo);

  // Register the MC instruction info.
  TargetRegistry::RegisterMCInstrInfo(TheSparcTarget, createSparcMCInstrInfo);

  // Register the MC register info.
  TargetRegistry::RegisterMCRegInfo(TheSparcTarget, createSparcMCRegisterInfo);

  // Register the MC subtarget info.
  TargetRegistry::RegisterMCSubtargetInfo(TheSparcTarget,
                                          createSparcMCSubtargetInfo);
}
