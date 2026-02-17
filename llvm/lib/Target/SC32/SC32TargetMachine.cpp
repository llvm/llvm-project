#include "SC32TargetMachine.h"
#include "SC32PassConfig.h"
#include "SC32Subtarget.h"
#include "TargetInfo/SC32TargetInfo.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"

using namespace llvm;

SC32TargetMachine::SC32TargetMachine(const Target &T, const Triple &TT,
                                     StringRef CPU, StringRef FS,
                                     const TargetOptions &Options,
                                     std::optional<Reloc::Model> RM,
                                     std::optional<CodeModel::Model> CM,
                                     CodeGenOptLevel OL, bool JIT)
    : CodeGenTargetMachineImpl(T, TT.computeDataLayout(), TT, CPU, FS, Options,
                               RM.value_or(Reloc::Static),
                               getEffectiveCodeModel(CM, CodeModel::Small), OL),
      TLOF(std::make_unique<TargetLoweringObjectFileELF>()),
      Subtarget(std::make_unique<SC32Subtarget>(*this)) {
  this->Options.EmitAddrsig = false;

  initAsmInfo();
}

TargetPassConfig *SC32TargetMachine::createPassConfig(PassManagerBase &PM) {
  return new SC32PassConfig(*this, PM);
}

TargetLoweringObjectFile *SC32TargetMachine::getObjFileLowering() const {
  return TLOF.get();
}

const TargetSubtargetInfo *
SC32TargetMachine::getSubtargetImpl(const Function &) const {
  return Subtarget.get();
}

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void LLVMInitializeSC32Target() {
  RegisterTargetMachine<SC32TargetMachine>{getTheSC32Target()};
}
