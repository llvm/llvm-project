//===-- Nios2TargetMachine.cpp - Define TargetMachine for Nios2 -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements the info about Nios2 target spec.
//
//===----------------------------------------------------------------------===//

#include "Nios2TargetMachine.h"
#include "Nios2.h"

using namespace llvm;

#define DEBUG_TYPE "nios2"

extern "C" void LLVMInitializeNios2Target() {
  // Register the target.
}

static std::string computeDataLayout(const Triple &TT, StringRef CPU,
                                     const TargetOptions &Options) {
  return "e-p:32:32:32-i8:8:32-i16:16:32-n32";
}

static Reloc::Model getEffectiveRelocModel(CodeModel::Model CM,
                                           Optional<Reloc::Model> RM) {
  if (!RM.hasValue() || CM == CodeModel::JITDefault)
    return Reloc::Static;
  return *RM;
}

Nios2TargetMachine::Nios2TargetMachine(const Target &T, const Triple &TT,
                                       StringRef CPU, StringRef FS,
                                       const TargetOptions &Options,
                                       Optional<Reloc::Model> RM,
                                       CodeModel::Model CM,
                                       CodeGenOpt::Level OL)
    : LLVMTargetMachine(T, computeDataLayout(TT, CPU, Options), TT, CPU, FS,
                        Options, getEffectiveRelocModel(CM, RM), CM, OL) {}

Nios2TargetMachine::~Nios2TargetMachine() {}
