//===-- BPFTargetMachine.h - Define TargetMachine for BPF --- C++ ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the BPF specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_BPF_BPFTARGETMACHINE_H
#define LLVM_LIB_TARGET_BPF_BPFTARGETMACHINE_H

#include "BPFSubtarget.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
class BPFTargetMachine : public LLVMTargetMachine {
  std::unique_ptr<TargetLoweringObjectFile> TLOF;
  BPFSubtarget Subtarget;

public:
  BPFTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                   StringRef FS, const TargetOptions &Options,
                   Optional<Reloc::Model> RM, Optional<CodeModel::Model> CM,
                   CodeGenOpt::Level OL, bool JIT);

  const BPFSubtarget *getSubtargetImpl() const { return &Subtarget; }
  const BPFSubtarget *getSubtargetImpl(const Function &) const override {
    return &Subtarget;
  }

  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;

  TargetTransformInfo getTargetTransformInfo(const Function &F) override;

  TargetLoweringObjectFile *getObjFileLowering() const override {
    return TLOF.get();
  }

  void adjustPassManager(PassManagerBuilder &) override;
  void registerPassBuilderCallbacks(PassBuilder &PB,
                                    bool DebugPassManager) override;
};
}

#endif
