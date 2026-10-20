//===-- Toy64TargetMachine.h - Define Toy64 TargetMachine ------*- C++ -*-===//

#ifndef LLVM_LIB_TARGET_TOY64_TOY64TARGETMACHINE_H
#define LLVM_LIB_TARGET_TOY64_TOY64TARGETMACHINE_H

#include "Toy64Subtarget.h"

#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/CodeGen/CodeGenTargetMachineImpl.h"

#include <memory>
#include <optional>

namespace llvm {

class Toy64TargetMachine : public CodeGenTargetMachineImpl {
  Toy64Subtarget Subtarget;

  std::unique_ptr<TargetLoweringObjectFile> TLOF;

public:
  Toy64TargetMachine(
      const Target &T,
      const Triple &TT,
      StringRef CPU,
      StringRef FS,
      const TargetOptions &Options,
      std::optional<Reloc::Model> RM,
      std::optional<CodeModel::Model> CM,
      CodeGenOptLevel OL,
      bool JIT);

  const Toy64Subtarget *
  getSubtargetImpl(const Function &F) const override {
    return &Subtarget;
  }

  TargetLoweringObjectFile *
  getObjFileLowering() const override {
    return TLOF.get();
  }

  TargetPassConfig *
  createPassConfig(
      PassManagerBase &PM) override;
};

} // namespace llvm

#endif