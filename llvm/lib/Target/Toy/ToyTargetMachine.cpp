//===-- ToyTargetMachine.cpp - Define TargetMachine for Toy ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "ToyTargetMachine.h"
#include "TargetInfo/ToyTargetInfo.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

extern "C" void LLVMInitializeToyTarget() {
  // Register the target.
  RegisterTargetMachine<ToyTargetMachine> X(getTheToyTarget());
}

static std::string computeDataLayout() {
  // TODO: Appropriately fill-up this stub function later
  return "";
}

static Reloc::Model getEffectiveRelocModel(Optional<Reloc::Model> RM) {
  // TODO: Appropriately fill-up this stub function later
  return *RM;
}

static CodeModel::Model getEffectiveToyCodeModel() {
  // TODO: Appropriately fill-up this stub function later
  return CodeModel::Small;
}

ToyTargetMachine::ToyTargetMachine(
    // TODO: Appropriately fill-up this stub constructor later
    const Target &T, const Triple &TT, StringRef &CPU, StringRef &FS,
    const TargetOptions &Options, Optional<Reloc::Model> &RM,
    Optional<CodeModel::Model> &CM, CodeGenOpt::Level &OL, bool &JIT)
    : LLVMTargetMachine(T, computeDataLayout(), TT, CPU, FS, Options,
                        getEffectiveRelocModel(RM),
                        getEffectiveToyCodeModel(), OL) {
}

ToyTargetMachine::~ToyTargetMachine() {}
