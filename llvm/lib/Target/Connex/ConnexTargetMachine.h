//===-- ConnexTargetMachine.h - Define TargetMachine for Connex --- C++ ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Connex specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_CONNEX_CONNEXTARGETMACHINE_H
#define LLVM_LIB_TARGET_CONNEX_CONNEXTARGETMACHINE_H

#include "ConnexSubtarget.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetTransformInfo.h"
// Inspired from BPF/BPFTargetMachine.h (Oct 2025)
#include "llvm/CodeGen/CodeGenTargetMachineImpl.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetMachine.h" // This was before
#include <memory>
#include <optional>

namespace llvm {
class ConnexTargetMachine : public CodeGenTargetMachineImpl {
  std::unique_ptr<TargetLoweringObjectFile> TLOF;
  ConnexSubtarget Subtarget;

public:
  ConnexTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                      StringRef FS, const TargetOptions &Options,
                      std::optional<Reloc::Model> RM,
                      std::optional<CodeModel::Model> CM,
                      CodeGenOptLevel OL,
                      bool JIT);

  const ConnexSubtarget *getSubtargetImpl() const { return &Subtarget; }
  const ConnexSubtarget *getSubtargetImpl(const Function &) const override {
    return &Subtarget;
  }

  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;

  TargetTransformInfo getTargetTransformInfo(const Function &F) const override;

  TargetLoweringObjectFile *getObjFileLowering() const override {
    return TLOF.get();
  }
};
} // namespace llvm

#endif
