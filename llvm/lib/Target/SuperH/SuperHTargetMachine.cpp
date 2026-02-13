//===-- SuperHTargetMachine.cpp - Define TargetMachine for SuperH -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "SuperHTargetMachine.h"
#include "TargetInfo/SuperHTargetInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"
#include <optional>

using namespace llvm;

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void LLVMInitializeSuperHTarget() {
    RegisterTargetMachine<SuperHTargetMachine> SH(getTheSuperHTarget());
}

SuperHTargetMachine::~SuperHTargetMachine() { }

/// Create a SuperH architecture model.
SuperHTargetMachine::SuperHTargetMachine(const Target &T, const Triple &TT,
                                           StringRef CPU, StringRef FS,
                                           const TargetOptions &Options,
                                           std::optional<Reloc::Model> RM,
                                           std::optional<CodeModel::Model> CM,
                                           CodeGenOptLevel OL, bool JIT)
    : CodeGenTargetMachineImpl(
        T, TT.computeDataLayout(), TT, CPU, FS, Options,
        RM.value_or(Reloc::Static), getEffectiveCodeModel(CM, CodeModel::Small), 
        OL) {

}