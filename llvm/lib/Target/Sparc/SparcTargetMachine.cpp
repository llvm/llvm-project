//===-- SparcTargetMachine.cpp - Define TargetMachine for Sparc -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "SparcTargetMachine.h"
#include "SparcTargetObjectFile.h"
#include "Sparc.h"
#include "LeonPasses.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

extern "C" void LLVMInitializeSparcTarget() {
  // Register the target.
  RegisterTargetMachine<SparcV8TargetMachine> X(getTheSparcTarget());
  RegisterTargetMachine<SparcV9TargetMachine> Y(getTheSparcV9Target());
  RegisterTargetMachine<SparcelTargetMachine> Z(getTheSparcelTarget());
}

static std::string computeDataLayout(const Triple &T, bool is64Bit) {
  // Sparc is typically big endian, but some are little.
  std::string Ret = T.getArch() == Triple::sparcel ? "e" : "E";
  Ret += "-m:e";

  // Some ABIs have 32bit pointers.
  if (!is64Bit)
    Ret += "-p:32:32";

  // Alignments for 64 bit integers.
  Ret += "-i64:64";

  // On SparcV9 128 floats are aligned to 128 bits, on others only to 64.
  // On SparcV9 registers can hold 64 or 32 bits, on others only 32.
  if (is64Bit)
    Ret += "-n32:64";
  else
    Ret += "-f128:64-n32";

  if (is64Bit)
    Ret += "-S128";
  else
    Ret += "-S64";

  return Ret;
}

static Reloc::Model getEffectiveRelocModel(Optional<Reloc::Model> RM) {
  if (!RM.hasValue())
    return Reloc::Static;
  return *RM;
}

/// Create an ILP32 architecture model
SparcTargetMachine::SparcTargetMachine(const Target &T, const Triple &TT,
                                       StringRef CPU, StringRef FS,
                                       const TargetOptions &Options,
                                       Optional<Reloc::Model> RM,
                                       CodeModel::Model CM,
                                       CodeGenOpt::Level OL, bool is64bit)
    : LLVMTargetMachine(T, computeDataLayout(TT, is64bit), TT, CPU, FS, Options,
                        getEffectiveRelocModel(RM), CM, OL),
      TLOF(make_unique<SparcELFTargetObjectFile>()),
      Subtarget(TT, CPU, FS, *this, is64bit), is64Bit(is64bit) {
  initAsmInfo();
}

SparcTargetMachine::~SparcTargetMachine() {}

const SparcSubtarget * 
SparcTargetMachine::getSubtargetImpl(const Function &F) const {
  Attribute CPUAttr = F.getFnAttribute("target-cpu");
  Attribute FSAttr = F.getFnAttribute("target-features");

  std::string CPU = !CPUAttr.hasAttribute(Attribute::None)
                        ? CPUAttr.getValueAsString().str()
                        : TargetCPU;
  std::string FS = !FSAttr.hasAttribute(Attribute::None)
                       ? FSAttr.getValueAsString().str()
                       : TargetFS;

  // FIXME: This is related to the code below to reset the target options,
  // we need to know whether or not the soft float flag is set on the
  // function, so we can enable it as a subtarget feature.
  bool softFloat =
      F.hasFnAttribute("use-soft-float") &&
      F.getFnAttribute("use-soft-float").getValueAsString() == "true";

  if (softFloat)         
    FS += FS.empty() ? "+soft-float" : ",+soft-float";

  auto &I = SubtargetMap[CPU + FS];
  if (!I) {
    // This needs to be done before we create a new subtarget since any
    // creation will depend on the TM and the code generation flags on the
    // function that reside in TargetOptions.
    resetTargetOptions(F);
    I = llvm::make_unique<SparcSubtarget>(TargetTriple, CPU, FS, *this,
                                          this->is64Bit);
  }
  return I.get();
}

namespace {
/// Sparc Code Generator Pass Configuration Options.
class SparcPassConfig : public TargetPassConfig {
public:
  SparcPassConfig(SparcTargetMachine &TM, PassManagerBase &PM)
    : TargetPassConfig(TM, PM) {}

  SparcTargetMachine &getSparcTargetMachine() const {
    return getTM<SparcTargetMachine>();
  }

  void addIRPasses() override;
  bool addInstSelector() override;
  void addPreEmitPass() override;
};
} // namespace

TargetPassConfig *SparcTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new SparcPassConfig(*this, PM);
}

void SparcPassConfig::addIRPasses() {
  addPass(createAtomicExpandPass());

  TargetPassConfig::addIRPasses();
}

bool SparcPassConfig::addInstSelector() {
  addPass(createSparcISelDag(getSparcTargetMachine()));
  return false;
}

void SparcPassConfig::addPreEmitPass(){
  addPass(createSparcDelaySlotFillerPass());

  if (this->getSparcTargetMachine().getSubtargetImpl()->insertNOPLoad())
  {
    addPass(new InsertNOPLoad());
  }
  if (this->getSparcTargetMachine().getSubtargetImpl()->fixFSMULD())
  {
    addPass(new FixFSMULD());
  }
  if (this->getSparcTargetMachine().getSubtargetImpl()->replaceFMULS())
  {
    addPass(new ReplaceFMULS());
  }
  if (this->getSparcTargetMachine().getSubtargetImpl()->detectRoundChange()) {
    addPass(new DetectRoundChange());
  }
  if (this->getSparcTargetMachine().getSubtargetImpl()->fixAllFDIVSQRT())
  {
    addPass(new FixAllFDIVSQRT());
  }
}

void SparcV8TargetMachine::anchor() { }

SparcV8TargetMachine::SparcV8TargetMachine(const Target &T, const Triple &TT,
                                           StringRef CPU, StringRef FS,
                                           const TargetOptions &Options,
                                           Optional<Reloc::Model> RM,
                                           CodeModel::Model CM,
                                           CodeGenOpt::Level OL)
    : SparcTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, false) {}

void SparcV9TargetMachine::anchor() { }

SparcV9TargetMachine::SparcV9TargetMachine(const Target &T, const Triple &TT,
                                           StringRef CPU, StringRef FS,
                                           const TargetOptions &Options,
                                           Optional<Reloc::Model> RM,
                                           CodeModel::Model CM,
                                           CodeGenOpt::Level OL)
    : SparcTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, true) {}

void SparcelTargetMachine::anchor() {}

SparcelTargetMachine::SparcelTargetMachine(const Target &T, const Triple &TT,
                                           StringRef CPU, StringRef FS,
                                           const TargetOptions &Options,
                                           Optional<Reloc::Model> RM,
                                           CodeModel::Model CM,
                                           CodeGenOpt::Level OL)
    : SparcTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, false) {}
