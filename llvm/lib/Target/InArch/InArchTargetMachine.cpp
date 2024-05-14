//===----------------------------------------------------------------------===//
//
// Implements the info about InArch target spec.
//
//===----------------------------------------------------------------------===//

#include "InArchTargetMachine.h"
#include "InArch.h"
//#include "InArchTargetTransformInfo.h"
#include "TargetInfo/InArchTargetInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetOptions.h"

#define DEBUG_TYPE "InArch"

using namespace llvm;

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeInArchTarget() {
  // Register the target.
  INARCH_DUMP_CYAN
  RegisterTargetMachine<InArchTargetMachine> A(getTheInArchTarget());
}

InArchTargetMachine::InArchTargetMachine(const Target &T, const Triple &TT,
                                   StringRef CPU, StringRef FS,
                                   const TargetOptions &Options,
                                   std::optional<Reloc::Model> RM,
                                   std::optional<CodeModel::Model> CM,
                                   CodeGenOptLevel OL, bool JIT)
    : LLVMTargetMachine(T, "e-m:e-p:32:32-i8:8:32-i16:16:32-i64:64-n32", TT,
                        CPU, FS, Options, Reloc::Static,
                                                getEffectiveCodeModel(CM, CodeModel::Small), OL),
      TLOF(std::make_unique<TargetLoweringObjectFileELF>()),
      Subtarget(TT, std::string(CPU), std::string(FS), *this) {
  initAsmInfo();
}

InArchTargetMachine::~InArchTargetMachine() = default;

namespace {

/// InArch Code Generator Pass Configuration Options.
class InArchPassConfig : public TargetPassConfig {
public:
  InArchPassConfig(InArchTargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {}

  InArchTargetMachine &getInArchTargetMachine() const {
    return getTM<InArchTargetMachine>();
  }

  bool addInstSelector() override;
};

} // end anonymous namespace

TargetPassConfig *InArchTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new InArchPassConfig(*this, PM);
}

bool InArchPassConfig::addInstSelector() {
  addPass(createInArchISelDag(getInArchTargetMachine()));
  return false;
}