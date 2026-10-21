#include "GenpayTargetMachine.h"
#include "Genpay.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/PassRegistry.h"
#include "llvm/Target/TargetLoweringObjectFile.h"

using namespace llvm;

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeGenpayTarget() {
  RegisterTargetMachine<GenpayTargetMachine> X(getTheGenpayTarget());
  PassRegistry &PR = *PassRegistry::getPassRegistry();
  initializeGenpayAsmPrinterPass(PR);
  initializeGenpayDAGToDAGISelLegacyPass(PR);
}

static std::string computeDataLayout() {
  return "e"        // little endian
         "-m:e"     // ELF name mangling
         "-p:64:64" // 64-bit pointers
         "-i64:64"  // i64 legal, 64-bit aligned
         "-a:0:64"  // aggregate alignment
         "-n64"     // native integer width
         "-S64";    // natural stack alignment
}

GenpayTargetMachine::GenpayTargetMachine(
    const Target &T, const Triple &TT, StringRef CPU, StringRef FS,
    const TargetOptions &Options, std::optional<Reloc::Model> RM,
    std::optional<CodeModel::Model> CM, CodeGenOptLevel OL, bool JIT)
    : CodeGenTargetMachineImpl(T, computeDataLayout(), TT, CPU, FS, Options,
                               RM.value_or(Reloc::Static),
                               getEffectiveCodeModel(CM, CodeModel::Small),
                               OL),
      Subtarget(TT, CPU, FS, *this, Options, getCodeModel(), OL),
      TLOF(std::make_unique<TargetLoweringObjectFileELF>()) {
  initAsmInfo();
}

namespace {
class GenpayPassConfig : public TargetPassConfig {
public:
  GenpayPassConfig(GenpayTargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {}

  GenpayTargetMachine &getGenpayTargetMachine() const {
    return getTM<GenpayTargetMachine>();
  }

  bool addInstSelector() override {
    addPass(createGenpayISelDag(getGenpayTargetMachine()));
    return false;
  }
};
} // namespace

TargetPassConfig *
GenpayTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new GenpayPassConfig(*this, PM);
}
