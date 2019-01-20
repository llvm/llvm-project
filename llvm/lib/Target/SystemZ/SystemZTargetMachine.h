//=- SystemZTargetMachine.h - Define TargetMachine for SystemZ ----*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the SystemZ specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZTARGETMACHINE_H
#define LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZTARGETMACHINE_H

#include "SystemZSubtarget.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetMachine.h"
#include <memory>

namespace llvm {

class SystemZTargetMachine : public LLVMTargetMachine {
  std::unique_ptr<TargetLoweringObjectFile> TLOF;
  SystemZSubtarget Subtarget;

public:
  SystemZTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                       StringRef FS, const TargetOptions &Options,
                       Optional<Reloc::Model> RM, Optional<CodeModel::Model> CM,
                       CodeGenOpt::Level OL, bool JIT);
  ~SystemZTargetMachine() override;

  const SystemZSubtarget *getSubtargetImpl() const { return &Subtarget; }

  const SystemZSubtarget *getSubtargetImpl(const Function &) const override {
    return &Subtarget;
  }

  // Override LLVMTargetMachine
  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;
  TargetTransformInfo getTargetTransformInfo(const Function &F) override;

  TargetLoweringObjectFile *getObjFileLowering() const override {
    return TLOF.get();
  }

  bool targetSchedulesPostRAScheduling() const override { return true; };
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZTARGETMACHINE_H
