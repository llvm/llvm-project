//===-- Nios2TargetMachine.h - Define TargetMachine for Nios2 ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Nios2 specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NIOS2_NIOS2TARGETMACHINE_H
#define LLVM_LIB_TARGET_NIOS2_NIOS2TARGETMACHINE_H

#include "Nios2Subtarget.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
class Nios2TargetMachine : public LLVMTargetMachine {
  Nios2Subtarget DefaultSubtarget;

  mutable StringMap<std::unique_ptr<Nios2Subtarget>> SubtargetMap;

public:
  Nios2TargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                     StringRef FS, const TargetOptions &Options,
                     Optional<Reloc::Model> RM, Optional<CodeModel::Model> CM,
                     CodeGenOpt::Level OL, bool JIT);
  ~Nios2TargetMachine() override;

  const Nios2Subtarget *getSubtargetImpl() const { return &DefaultSubtarget; }

  const Nios2Subtarget *getSubtargetImpl(const Function &F) const override;

  // Pass Pipeline Configuration
  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;
};
} // namespace llvm

#endif
