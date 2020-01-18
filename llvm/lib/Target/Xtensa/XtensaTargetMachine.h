//===-- XtensaTargetMachine.h - Define TargetMachine for Xtensa -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Xtensa specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_XTENSA_XTENSATARGETMACHINE_H
#define LLVM_LIB_TARGET_XTENSA_XTENSATARGETMACHINE_H

#include "XtensaSubtarget.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class TargetFrameLowering;

extern Target TheXtensaTarget;

class XtensaTargetMachine : public LLVMTargetMachine {
  std::unique_ptr<TargetLoweringObjectFile> TLOF;

public:
  XtensaTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                      StringRef FS, const TargetOptions &Options,
                      Optional<Reloc::Model> RM, Optional<CodeModel::Model> CM,
                      CodeGenOpt::Level OL, bool JIT, bool isLittle);

  XtensaTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                      StringRef FS, const TargetOptions &Options,
                      Optional<Reloc::Model> RM, Optional<CodeModel::Model> CM,
                      CodeGenOpt::Level OL, bool JIT);

  // Override TargetMachine.
  const XtensaSubtarget *getSubtargetImpl() const { return &Subtarget; }
  const XtensaSubtarget *getSubtargetImpl(const Function &F) const override;
  // Override LLVMTargetMachine
  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;
  TargetLoweringObjectFile *getObjFileLowering() const override {
    return TLOF.get();
  }

protected:
  XtensaSubtarget Subtarget;
};
} // end namespace llvm

#endif /* LLVM_LIB_TARGET_XTENSA_XTENSATARGETMACHINE_H */
