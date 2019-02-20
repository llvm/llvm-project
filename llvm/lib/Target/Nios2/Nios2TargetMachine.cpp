//===-- Nios2TargetMachine.cpp - Define TargetMachine for Nios2 -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements the info about Nios2 target spec.
//
//===----------------------------------------------------------------------===//

#include "Nios2TargetMachine.h"
#include "Nios2.h"
#include "Nios2TargetObjectFile.h"

#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

#define DEBUG_TYPE "nios2"

extern "C" void LLVMInitializeNios2Target() {
  // Register the target.
  RegisterTargetMachine<Nios2TargetMachine> X(getTheNios2Target());
}

static std::string computeDataLayout() {
  return "e-p:32:32:32-i8:8:32-i16:16:32-n32";
}

static Reloc::Model getEffectiveRelocModel(Optional<Reloc::Model> RM) {
  if (!RM.hasValue())
    return Reloc::Static;
  return *RM;
}

Nios2TargetMachine::Nios2TargetMachine(const Target &T, const Triple &TT,
                                       StringRef CPU, StringRef FS,
                                       const TargetOptions &Options,
                                       Optional<Reloc::Model> RM,
                                       Optional<CodeModel::Model> CM,
                                       CodeGenOpt::Level OL, bool JIT)
    : LLVMTargetMachine(T, computeDataLayout(), TT, CPU, FS, Options,
                        getEffectiveRelocModel(RM),
                        getEffectiveCodeModel(CM, CodeModel::Small), OL),
      TLOF(make_unique<Nios2TargetObjectFile>()),
      Subtarget(TT, CPU, FS, *this) {
  initAsmInfo();
}

Nios2TargetMachine::~Nios2TargetMachine() {}

const Nios2Subtarget *
Nios2TargetMachine::getSubtargetImpl(const Function &F) const {
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
    I = llvm::make_unique<Nios2Subtarget>(TargetTriple, CPU, FS, *this);
  }
  return I.get();
}

namespace {
/// Nios2 Code Generator Pass Configuration Options.
class Nios2PassConfig : public TargetPassConfig {
public:
  Nios2PassConfig(Nios2TargetMachine &TM, PassManagerBase *PM)
      : TargetPassConfig(TM, *PM) {}

  Nios2TargetMachine &getNios2TargetMachine() const {
    return getTM<Nios2TargetMachine>();
  }

  void addCodeGenPrepare() override;
  bool addInstSelector() override;
  void addIRPasses() override;
};
} // namespace

TargetPassConfig *Nios2TargetMachine::createPassConfig(PassManagerBase &PM) {
  return new Nios2PassConfig(*this, &PM);
}

void Nios2PassConfig::addCodeGenPrepare() {
  TargetPassConfig::addCodeGenPrepare();
}

void Nios2PassConfig::addIRPasses() { TargetPassConfig::addIRPasses(); }

// Install an instruction selector pass using
// the ISelDag to gen Nios2 code.
bool Nios2PassConfig::addInstSelector() {
  addPass(createNios2ISelDag(getNios2TargetMachine(), getOptLevel()));
  return false;
}
