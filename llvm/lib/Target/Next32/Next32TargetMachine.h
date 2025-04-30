//===-- Next32TargetMachine.h - Define TargetMachine for Next32 --- C++ ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Next32 specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_Next32_Next32TARGETMACHINE_H
#define LLVM_LIB_TARGET_Next32_Next32TARGETMACHINE_H

#include "Next32Subtarget.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
class Next32TargetMachine : public LLVMTargetMachine {
  std::unique_ptr<TargetLoweringObjectFile> TLOF;
  Next32Subtarget Subtarget;

public:
  Next32TargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                      StringRef FS, const TargetOptions &Options,
                      std::optional<Reloc::Model> RM,
                      std::optional<CodeModel::Model> CM, CodeGenOptLevel OL,
                      bool JIT);

  const Next32Subtarget *getSubtargetImpl() const { return &Subtarget; }
  const Next32Subtarget *getSubtargetImpl(const Function &) const override {
    return &Subtarget;
  }

  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;

  TargetLoweringObjectFile *getObjFileLowering() const override {
    return TLOF.get();
  }

  virtual bool usesVRegsForVariadicDefs() const override { return true; }

  MachineFunctionInfo *
  createMachineFunctionInfo(BumpPtrAllocator &Allocator, const Function &F,
                            const TargetSubtargetInfo *STI) const override;
  TargetTransformInfo getTargetTransformInfo(const Function &F) const override;
};
} // namespace llvm

#endif
