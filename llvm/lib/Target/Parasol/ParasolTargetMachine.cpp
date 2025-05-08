//===-- ParasolTargetMachine.cpp - Define TargetMachine for Parasol -------===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//
//
// Implements the info about Parasol target spec.
//
//===----------------------------------------------------------------------===//

#include "ParasolTargetMachine.h"
#include "Parasol.h"
#include "ParasolISelDAGToDAG.h"
#include "ParasolSubtarget.h"
#include "ParasolTargetObjectFile.h"
#include "TargetInfo/ParasolTargetInfo.h"
#include "llvm/CodeGen/GlobalISel/IRTranslator.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelect.h"
#include "llvm/CodeGen/GlobalISel/Legalizer.h"
#include "llvm/CodeGen/GlobalISel/RegBankSelect.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/MC/TargetRegistry.h"
#include <optional>

using namespace llvm;

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeParasolTarget() {
  // Register the target.
  //- Little endian Target Machine
  RegisterTargetMachine<ParasolTargetMachine> X(getTheParasolTarget());

  PassRegistry &PR = *PassRegistry::getPassRegistry();
  initializeGlobalISel(PR);

  // Maybe implement?
  // initializeParasolCheckAndAdjustIRPass(PR);
  // initializeParasolMIPeepholePass(PR);
  // initializeParasolDAGToDAGISelPass(PR);
}

// If you update this string, also update in clang at
// clang/lib/Basic/Targets/Parasol.cpp
// otherwise you will get a mismatch between clang and llc.
static std::string computeDataLayout() {
  std::string Ret = "";

  // Little endian
  Ret += "e";

  // ELF name mangling
  Ret += "-m:e";

  // 32-bit pointers, 32-bit aligned
  Ret += "-p:32:32";

  // 64-bit integers, 64 bit aligned
  Ret += "-i64:64";

  // 1, 8, 16, and 32-bit native integer width i.e register are 32-bit
  Ret += "-n1:8:16:32";

  // 128-bit natural stack alignment
  Ret += "-S128";

  return Ret;
}

static Reloc::Model getEffectiveRelocModel(std::optional<CodeModel::Model> CM,
                                           std::optional<Reloc::Model> RM) {
  return RM.value_or(Reloc::Static);
}

ParasolTargetMachine::ParasolTargetMachine(const Target &T, const Triple &TT,
                                           StringRef CPU, StringRef FS,
                                           const TargetOptions &Options,
                                           std::optional<Reloc::Model> RM,
                                           std::optional<CodeModel::Model> CM,
                                           CodeGenOptLevel OL, bool JIT)
    : LLVMTargetMachine(T, computeDataLayout(), TT, CPU, FS, Options,
                        getEffectiveRelocModel(CM, RM),
                        getEffectiveCodeModel(CM, CodeModel::Medium), OL),
      TLOF(std::make_unique<ParasolTargetObjectFile>()) {
  // initAsmInfo will display features by llc -march=parasol on 3.7
  initAsmInfo();
}

const ParasolSubtarget *
ParasolTargetMachine::getSubtargetImpl(const Function &F) const {
  Attribute CPUAttr = F.getFnAttribute("target-cpu");
  Attribute FSAttr = F.getFnAttribute("target-features");

  std::string CPU = !CPUAttr.hasAttribute(Attribute::None)
                        ? CPUAttr.getValueAsString().str()
                        : TargetCPU;
  std::string FS = !FSAttr.hasAttribute(Attribute::None)
                       ? FSAttr.getValueAsString().str()
                       : TargetFS;

  auto &I = SubtargetMap[CPU + FS];
  if (!I) {
    // This needs to be done before we create a new subtarget since any
    // creation will depend on the TM and the code generation flags on the
    // function that reside in TargetOptions.
    resetTargetOptions(F);
    I = std::make_unique<ParasolSubtarget>(TargetTriple, CPU, FS, *this);
  }
  return I.get();
}

namespace {
class ParasolPassConfig : public TargetPassConfig {
public:
  ParasolPassConfig(ParasolTargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {}

  ParasolTargetMachine &getParasolTargetMachine() const {
    return getTM<ParasolTargetMachine>();
  }

  bool addInstSelector() override;
  void addPreEmitPass() override;
  void addIRPasses() override;

  bool addIRTranslator() override;
  bool addLegalizeMachineIR() override;
  bool addRegBankSelect() override;
  bool addGlobalInstructionSelect() override;
};
} // namespace

TargetPassConfig *ParasolTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new ParasolPassConfig(*this, PM);
}

// Install an instruction selector pass using
// the ISelDag to gen Parasol code.
bool ParasolPassConfig::addInstSelector() {
  addPass(new ParasolDAGToDAGISel(getParasolTargetMachine()));
  return false;
}

// Implemented by targets that want to run passes immediately before
// machine code is emitted. return true if -print-machineinstrs should
// print out the code after the passes.
void ParasolPassConfig::addPreEmitPass() {}

void ParasolPassConfig::addIRPasses() {
  // TargetPassConfig::addIRPasses();
  // addPass(createAnnotateEncryptionPass());
}

bool ParasolPassConfig::addIRTranslator() {
  addPass(new IRTranslator());
  return false;
}

bool ParasolPassConfig::addLegalizeMachineIR() {
  addPass(new Legalizer());
  return false;
}

bool ParasolPassConfig::addRegBankSelect() {
  addPass(new RegBankSelect());
  return false;
}

bool ParasolPassConfig::addGlobalInstructionSelect() {
  addPass(new InstructionSelect(getOptLevel()));
  return false;
}
