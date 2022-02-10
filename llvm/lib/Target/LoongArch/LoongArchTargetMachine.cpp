//===-- LoongArchTargetMachine.cpp - Define TargetMachine for LoongArch ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the info about LoongArch target spec.
//
//===----------------------------------------------------------------------===//

#include "LoongArchTargetMachine.h"
#include "TargetInfo/LoongArchTargetInfo.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

#define DEBUG_TYPE "loongarch"

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeLoongArchTarget() {
  // Register the target.
  RegisterTargetMachine<LoongArchTargetMachine> X(getTheLoongArch32Target());
  RegisterTargetMachine<LoongArchTargetMachine> Y(getTheLoongArch64Target());
}

// FIXME: This is just a placeholder to make current commit buildable. Body of
// this function will be filled in later commits.
static std::string computeDataLayout(const Triple &TT) {
  std::string Ret;
  Ret += "e";
  return Ret;
}

static Reloc::Model getEffectiveRelocModel(const Triple &TT,
                                           Optional<Reloc::Model> RM) {
  if (!RM.hasValue())
    return Reloc::Static;
  return *RM;
}

LoongArchTargetMachine::LoongArchTargetMachine(
    const Target &T, const Triple &TT, StringRef CPU, StringRef FS,
    const TargetOptions &Options, Optional<Reloc::Model> RM,
    Optional<CodeModel::Model> CM, CodeGenOpt::Level OL, bool JIT)
    : LLVMTargetMachine(T, computeDataLayout(TT), TT, CPU, FS, Options,
                        getEffectiveRelocModel(TT, RM),
                        getEffectiveCodeModel(CM, CodeModel::Small), OL) {
  initAsmInfo();
}

LoongArchTargetMachine::~LoongArchTargetMachine() = default;
