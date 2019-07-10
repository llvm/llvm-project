//===-- DPUTargetMachine.h - Define TargetMachine for DPUs    ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the DPU specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIB_TARGET_DPU_DPUTARGETMACHINE_H
#define LLVM_LIB_TARGET_DPU_DPUTARGETMACHINE_H

#include "llvm/IR/DataLayout.h"
#include "llvm/Target/TargetMachine.h"

#include "DPUSubtarget.h"

// we reserve 8 bytes for debugging purpose. See function
// DPURegisterInfo::eliminateFrameIndex for more information
#define STACK_SIZE_FOR_D22 (8)

namespace llvm {

class DPUTargetMachine : public LLVMTargetMachine {
  std::unique_ptr<TargetLoweringObjectFile> TLOF;

private:
  DPUSubtarget Subtarget;

public:
  DPUTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                   StringRef FS, const TargetOptions &Options,
                   Optional<Reloc::Model> RM, Optional<CodeModel::Model> CM,
                   CodeGenOpt::Level &OL, bool JIT);

  const TargetSubtargetInfo *
  getSubtargetImpl(const Function &F) const override {
    return &Subtarget;
  }

  // Pass Pipeline Configuration
  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;

  TargetLoweringObjectFile *getObjFileLowering() const override {
    return TLOF.get();
  }
};
} // namespace llvm
#endif
