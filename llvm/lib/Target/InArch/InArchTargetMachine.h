#ifndef LLVM_LIB_TARGET_INARCH_INARCHTARGETMACHINE_H
#define LLVM_LIB_TARGET_INARCH_INARCHTARGETMACHINE_H

#include "InArchInstrInfo.h"
#include "InArchSubtarget.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class InArchTargetMachine : public LLVMTargetMachine {
  std::unique_ptr<TargetLoweringObjectFile> TLOF;
  InArchSubtarget Subtarget;

public:
  InArchTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                   StringRef FS, const TargetOptions &Options,
                     std::optional<Reloc::Model> RM,
                     std::optional<CodeModel::Model> CM, CodeGenOptLevel OL,
                     bool JIT);
  ~InArchTargetMachine() override;

  const InArchSubtarget *getSubtargetImpl() const { return &Subtarget; }
  const InArchSubtarget *getSubtargetImpl(const Function &) const override {
    return &Subtarget;
  }

  // Pass Pipeline Configuration
  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;
  TargetLoweringObjectFile *getObjFileLowering() const override {
    return TLOF.get();
  }
};
} // end namespace llvm

#endif // LLVM_LIB_TARGET_INARCH_INARCHTARGETMACHINE_H