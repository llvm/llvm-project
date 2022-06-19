//===-- AVRTargetMachine.h - Define TargetMachine for AVR -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the AVR specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_AVR_TARGET_MACHINE_H
#define LLVM_AVR_TARGET_MACHINE_H

#include "llvm/IR/DataLayout.h"
#include "llvm/Target/TargetMachine.h"

#include "AVRFrameLowering.h"
#include "AVRISelLowering.h"
#include "AVRInstrInfo.h"
#include "AVRSelectionDAGInfo.h"
#include "AVRSubtarget.h"

namespace llvm {

/// A generic AVR implementation.
class AVRTargetMachine : public LLVMTargetMachine {
public:
  AVRTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                   StringRef FS, const TargetOptions &Options,
                   Optional<Reloc::Model> RM, Optional<CodeModel::Model> CM,
                   CodeGenOpt::Level OL, bool JIT);

  const AVRSubtarget *getSubtargetImpl() const;
  const AVRSubtarget *getSubtargetImpl(const Function &) const override;

  TargetLoweringObjectFile *getObjFileLowering() const override {
    return this->TLOF.get();
  }

  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;

private:
  std::unique_ptr<TargetLoweringObjectFile> TLOF;
  AVRSubtarget SubTarget;
};

} // end namespace llvm

#endif // LLVM_AVR_TARGET_MACHINE_H
