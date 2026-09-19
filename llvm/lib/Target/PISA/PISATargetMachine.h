//===-- PISATargetMachine.h - Define TargetMachine for PISA ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PISA_PISATARGETMACHINE_H
#define LLVM_LIB_TARGET_PISA_PISATARGETMACHINE_H

#include "PISASubtarget.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/CodeGen/CodeGenTargetMachineImpl.h"
#include <memory>
#include <optional>

namespace llvm {
class PISATargetMachine : public CodeGenTargetMachineImpl {
  std::unique_ptr<TargetLoweringObjectFile> TLOF;
  PISASubtarget Subtarget;
  mutable StringMap<std::unique_ptr<PISASubtarget>> SubtargetMap;

public:
  PISATargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                    StringRef FS, const TargetOptions &Options,
                    std::optional<Reloc::Model> RM,
                    std::optional<CodeModel::Model> CM, CodeGenOptLevel OL,
                    bool JIT);

  const PISASubtarget *getSubtargetImpl(const Function &F) const override;
  // DO NOT IMPLEMENT: subtargets are per-function entities based on the
  // target-specific attributes of each function.
  const PISASubtarget *getSubtargetImpl() const = delete;

  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;

  bool usesPhysRegsForValues() const override { return false; }

  TargetLoweringObjectFile *getObjFileLowering() const override {
    return TLOF.get();
  }
};
} // namespace llvm

#endif // LLVM_LIB_TARGET_PISA_PISATARGETMACHINE_H
