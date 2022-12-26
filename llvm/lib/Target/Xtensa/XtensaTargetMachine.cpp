//===- XtensaTargetMachine.cpp - Define TargetMachine for Xtensa ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the info about Xtensa target spec.
//
//===----------------------------------------------------------------------===//

#include "XtensaTargetMachine.h"
#include "TargetInfo/XtensaTargetInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Scalar.h"
#include <optional>

using namespace llvm;

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeXtensaTarget() {
  // Register the target.
  RegisterTargetMachine<XtensaTargetMachine> A(getTheXtensaTarget());
}

static std::string computeDataLayout(const Triple &TT, StringRef CPU,
                                     const TargetOptions &Options,
                                     bool IsLittle) {
  std::string Ret = "e-m:e-p:32:32-i8:8:32-i16:16:32-i64:64-n32";
  return Ret;
}

static Reloc::Model getEffectiveRelocModel(bool JIT,
                                           std::optional<Reloc::Model> RM) {
  if (!RM || JIT)
     return Reloc::Static;
  return *RM;
}

XtensaTargetMachine::XtensaTargetMachine(const Target &T, const Triple &TT,
                                         StringRef CPU, StringRef FS,
                                         const TargetOptions &Options,
                                         std::optional<Reloc::Model> RM,
                                         std::optional<CodeModel::Model> CM,
                                         CodeGenOpt::Level OL, bool JIT,
                                         bool IsLittle)
    : LLVMTargetMachine(T, computeDataLayout(TT, CPU, Options, IsLittle), TT,
                        CPU, FS, Options, getEffectiveRelocModel(JIT, RM),
                        getEffectiveCodeModel(CM, CodeModel::Small), OL),
      TLOF(std::make_unique<TargetLoweringObjectFileELF>()) {
  initAsmInfo();
}

XtensaTargetMachine::XtensaTargetMachine(const Target &T, const Triple &TT,
                                         StringRef CPU, StringRef FS,
                                         const TargetOptions &Options,
                                         std::optional<Reloc::Model> RM,
                                         std::optional<CodeModel::Model> CM,
                                         CodeGenOpt::Level OL, bool JIT)
    : XtensaTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, JIT, true) {}

TargetPassConfig *XtensaTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new TargetPassConfig(*this, PM);
}
