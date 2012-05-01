//===-- SPUTargetMachine.cpp - Define TargetMachine for Cell SPU ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Top-level implementation for the Cell SPU target.
//
//===----------------------------------------------------------------------===//

#include "SPUTargetMachine.h"
#include "SPU.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/SchedulerRegistry.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

extern "C" void LLVMInitializeCellSPUTarget() {
  // Register the target.
  RegisterTargetMachine<SPUTargetMachine> X(TheCellSPUTarget);
}

const std::pair<unsigned, int> *
SPUFrameLowering::getCalleeSaveSpillSlots(unsigned &NumEntries) const {
  NumEntries = 1;
  return &LR[0];
}

SPUTargetMachine::SPUTargetMachine(const Target &T, StringRef TT,
                                   StringRef CPU, StringRef FS,
                                   const TargetOptions &Options,
                                   Reloc::Model RM, CodeModel::Model CM,
                                   CodeGenOpt::Level OL)
  : LLVMTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL),
    Subtarget(TT, CPU, FS),
    DataLayout(Subtarget.getTargetDataString()),
    InstrInfo(*this),
    FrameLowering(Subtarget),
    TLInfo(*this),
    TSInfo(*this),
    InstrItins(Subtarget.getInstrItineraryData()) {
}

//===----------------------------------------------------------------------===//
// Pass Pipeline Configuration
//===----------------------------------------------------------------------===//

namespace {
/// SPU Code Generator Pass Configuration Options.
class SPUPassConfig : public TargetPassConfig {
public:
  SPUPassConfig(SPUTargetMachine *TM, PassManagerBase &PM)
    : TargetPassConfig(TM, PM) {}

  SPUTargetMachine &getSPUTargetMachine() const {
    return getTM<SPUTargetMachine>();
  }

  virtual bool addInstSelector();
  virtual bool addPreEmitPass();
};
} // namespace

TargetPassConfig *SPUTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new SPUPassConfig(this, PM);
}

bool SPUPassConfig::addInstSelector() {
  // Install an instruction selector.
  PM->add(createSPUISelDag(getSPUTargetMachine()));
  return false;
}

// passes to run just before printing the assembly
bool SPUPassConfig::addPreEmitPass() {
  // load the TCE instruction scheduler, if available via
  // loaded plugins
  typedef llvm::FunctionPass* (*BuilderFunc)(const char*);
  BuilderFunc schedulerCreator =
    (BuilderFunc)(intptr_t)sys::DynamicLibrary::SearchForAddressOfSymbol(
          "createTCESchedulerPass");
  if (schedulerCreator != NULL)
      PM->add(schedulerCreator("cellspu"));

  //align instructions with nops/lnops for dual issue
  PM->add(createSPUNopFillerPass(getSPUTargetMachine()));
  return true;
}
