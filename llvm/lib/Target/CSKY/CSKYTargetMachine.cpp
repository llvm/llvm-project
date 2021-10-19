//===--- CSKYTargetMachine.cpp - Define TargetMachine for CSKY ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the info about CSKY target spec.
//
//===----------------------------------------------------------------------===//

#include "CSKYTargetMachine.h"
#include "TargetInfo/CSKYTargetInfo.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeCSKYTarget() {
  RegisterTargetMachine<CSKYTargetMachine> X(getTheCSKYTarget());
}

static std::string computeDataLayout(const Triple &TT) {
  std::string Ret;

  // Only support little endian for now.
  // TODO: Add support for big endian.
  Ret += "e";

  // CSKY is always 32-bit target with the CSKYv2 ABI as prefer now.
  // It's a 4-byte aligned stack with ELF mangling only.
  Ret += "-m:e-S32-p:32:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:32"
         "-v128:32:32-a:0:32-Fi32-n32";

  return Ret;
}

CSKYTargetMachine::CSKYTargetMachine(const Target &T, const Triple &TT,
                                     StringRef CPU, StringRef FS,
                                     const TargetOptions &Options,
                                     Optional<Reloc::Model> RM,
                                     Optional<CodeModel::Model> CM,
                                     CodeGenOpt::Level OL, bool JIT)
    : LLVMTargetMachine(T, computeDataLayout(TT), TT, CPU, FS, Options,
                        RM.getValueOr(Reloc::Static),
                        getEffectiveCodeModel(CM, CodeModel::Small), OL),
      TLOF(std::make_unique<TargetLoweringObjectFileELF>()) {
  initAsmInfo();
}

namespace {
class CSKYPassConfig : public TargetPassConfig {
public:
  CSKYPassConfig(CSKYTargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {}

  CSKYTargetMachine &getCSKYTargetMachine() const {
    return getTM<CSKYTargetMachine>();
  }
};

} // namespace

TargetPassConfig *CSKYTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new CSKYPassConfig(*this, PM);
}
