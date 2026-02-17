#ifndef LLVM_LIB_TARGET_SC32_SC32TARGETMACHINE_H
#define LLVM_LIB_TARGET_SC32_SC32TARGETMACHINE_H

#include "llvm/CodeGen/CodeGenTargetMachineImpl.h"

namespace llvm {

class SC32TargetMachine : public CodeGenTargetMachineImpl {
private:
  std::unique_ptr<TargetLoweringObjectFile> TLOF;
  std::unique_ptr<TargetSubtargetInfo> Subtarget;

public:
  SC32TargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                    StringRef FS, const TargetOptions &Options,
                    std::optional<Reloc::Model> RM,
                    std::optional<CodeModel::Model> CM, CodeGenOptLevel OL,
                    bool JIT);

  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;

  TargetLoweringObjectFile *getObjFileLowering() const override;

  const TargetSubtargetInfo *getSubtargetImpl(const Function &) const override;
};

} // namespace llvm

#endif
