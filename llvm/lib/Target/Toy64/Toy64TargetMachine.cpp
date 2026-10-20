//===-- Toy64TargetMachine.cpp - Define Toy64 TargetMachine ---------------===//
#include "Toy64.h"

#include "Toy64TargetMachine.h"

#include "TargetInfo/Toy64TargetInfo.h"

#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/Compiler.h"


using namespace llvm;

//===----------------------------------------------------------------------===//
// Target registration
//===----------------------------------------------------------------------===//

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeToy64Target() {

  PassRegistry &PR =
      *PassRegistry::getPassRegistry();

  initializeToy64DAGToDAGISelLegacyPass(PR);

  RegisterTargetMachine<Toy64TargetMachine>
      X(getTheToy64Target());
}

//===----------------------------------------------------------------------===//
// Target defaults
//===----------------------------------------------------------------------===//

static Reloc::Model
getToy64EffectiveRelocModel(
    std::optional<Reloc::Model> RM) {
  return RM.value_or(Reloc::Static);
}


namespace {

class Toy64PassConfig : public TargetPassConfig {
  Toy64TargetMachine &ToyTM;

public:
  Toy64PassConfig(
      Toy64TargetMachine &TM,
      PassManagerBase &PM)
      : TargetPassConfig(TM, PM),
        ToyTM(TM) {}

  bool addInstSelector() override {
    addPass(
        createToy64ISelDagLegacyPass(ToyTM));

    return false;
  }
};

} // end anonymous namespace

TargetPassConfig *
Toy64TargetMachine::createPassConfig(
    PassManagerBase &PM) {

  return new Toy64PassConfig(
      *this,
      PM);
}

//===----------------------------------------------------------------------===//
// Toy64TargetMachine
//===----------------------------------------------------------------------===//

Toy64TargetMachine::Toy64TargetMachine(
    const Target &T,
    const Triple &TT,
    StringRef CPU,
    StringRef FS,
    const TargetOptions &Options,
    std::optional<Reloc::Model> RM,
    std::optional<CodeModel::Model> CM,
    CodeGenOptLevel OL,
    bool JIT)
    : CodeGenTargetMachineImpl(
          T,
          TT,
          CPU,
          FS,
          Options,
          getToy64EffectiveRelocModel(RM),
          getEffectiveCodeModel(CM, CodeModel::Small),
          OL),
          Subtarget(TT, CPU, FS, *this),
          TLOF(std::make_unique<TargetLoweringObjectFileELF>()) {

  initAsmInfo();
}