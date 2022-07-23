//===-- M88kTargetMachine.h - Define TargetMachine for M88k -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the M88k specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_M88K_M88KTARGETMACHINE_H
#define LLVM_LIB_TARGET_M88K_M88KTARGETMACHINE_H

#include "M88kSubtarget.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class M88kTargetMachine : public LLVMTargetMachine {
  std::unique_ptr<TargetLoweringObjectFile> TLOF;
  mutable StringMap<std::unique_ptr<M88kSubtarget>> SubtargetMap;

public:
  M88kTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                    StringRef FS, const TargetOptions &Options,
                    Optional<Reloc::Model> RM, Optional<CodeModel::Model> CM,
                    CodeGenOpt::Level OL, bool JIT);
  ~M88kTargetMachine() override;
  const M88kSubtarget *getSubtargetImpl(const Function &) const override;

  // DO NOT IMPLEMENT: There is no such thing as a valid default subtarget,
  // subtargets are per-function entities based on the target-specific
  // attributes of each function.
  const M88kSubtarget *getSubtargetImpl() const = delete;

  // Override LLVMTargetMachine
  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;
  // TargetTransformInfo getTargetTransformInfo(const Function &F) override;
  TargetLoweringObjectFile *getObjFileLowering() const override {
    return TLOF.get();
  }

  // Returns true if signed division instruction should be used on MC88100.
  bool useDivInstr() const;

  // Returns true if check for zero division should be omitted on MC88100.
  bool noZeroDivCheck() const;
};

} // end namespace llvm

#endif
