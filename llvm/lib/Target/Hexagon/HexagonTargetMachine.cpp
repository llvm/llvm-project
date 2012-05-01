//===-- HexagonTargetMachine.cpp - Define TargetMachine for Hexagon -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements the info about Hexagon target spec.
//
//===----------------------------------------------------------------------===//

#include "HexagonTargetMachine.h"
#include "Hexagon.h"
#include "HexagonISelLowering.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/PassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

static cl::
opt<bool> DisableHardwareLoops(
                        "disable-hexagon-hwloops", cl::Hidden,
                        cl::desc("Disable Hardware Loops for Hexagon target"));

/// HexagonTargetMachineModule - Note that this is used on hosts that
/// cannot link in a library unless there are references into the
/// library.  In particular, it seems that it is not possible to get
/// things to work on Win32 without this.  Though it is unused, do not
/// remove it.
extern "C" int HexagonTargetMachineModule;
int HexagonTargetMachineModule = 0;

extern "C" void LLVMInitializeHexagonTarget() {
  // Register the target.
  RegisterTargetMachine<HexagonTargetMachine> X(TheHexagonTarget);
}


/// HexagonTargetMachine ctor - Create an ILP32 architecture model.
///

/// Hexagon_TODO: Do I need an aggregate alignment?
///
HexagonTargetMachine::HexagonTargetMachine(const Target &T, StringRef TT,
                                           StringRef CPU, StringRef FS,
                                           const TargetOptions &Options,
                                           Reloc::Model RM,
                                           CodeModel::Model CM,
                                           CodeGenOpt::Level OL)
  : LLVMTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL),
    DataLayout("e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-a0:0") ,
    Subtarget(TT, CPU, FS), InstrInfo(Subtarget), TLInfo(*this),
    TSInfo(*this),
    FrameLowering(Subtarget),
    InstrItins(&Subtarget.getInstrItineraryData()) {
  setMCUseCFI(false);
}

// addPassesForOptimizations - Allow the backend (target) to add Target
// Independent Optimization passes to the Pass Manager.
bool HexagonTargetMachine::addPassesForOptimizations(PassManagerBase &PM) {

  PM.add(createConstantPropagationPass());
  PM.add(createLoopSimplifyPass());
  PM.add(createDeadCodeEliminationPass());
  PM.add(createConstantPropagationPass());
  PM.add(createLoopUnrollPass());
  PM.add(createLoopStrengthReducePass(getTargetLowering()));
  return true;
}

namespace {
/// Hexagon Code Generator Pass Configuration Options.
class HexagonPassConfig : public TargetPassConfig {
public:
  HexagonPassConfig(HexagonTargetMachine *TM, PassManagerBase &PM)
    : TargetPassConfig(TM, PM) {}

  HexagonTargetMachine &getHexagonTargetMachine() const {
    return getTM<HexagonTargetMachine>();
  }

  virtual bool addInstSelector();
  virtual bool addPreRegAlloc();
  virtual bool addPostRegAlloc();
  virtual bool addPreSched2();
  virtual bool addPreEmitPass();
};
} // namespace

TargetPassConfig *HexagonTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new HexagonPassConfig(this, PM);
}

bool HexagonPassConfig::addInstSelector() {
  PM->add(createHexagonRemoveExtendOps(getHexagonTargetMachine()));
  PM->add(createHexagonISelDag(getHexagonTargetMachine()));
  PM->add(createHexagonPeephole());
  return false;
}


bool HexagonPassConfig::addPreRegAlloc() {
  if (!DisableHardwareLoops) {
    PM->add(createHexagonHardwareLoops());
  }

  return false;
}

bool HexagonPassConfig::addPostRegAlloc() {
  PM->add(createHexagonCFGOptimizer(getHexagonTargetMachine()));
  return true;
}


bool HexagonPassConfig::addPreSched2() {
  addPass(IfConverterID);
  return true;
}

bool HexagonPassConfig::addPreEmitPass() {

  if (!DisableHardwareLoops) {
    PM->add(createHexagonFixupHwLoops());
  }

  // Expand Spill code for predicate registers.
  PM->add(createHexagonExpandPredSpillCode(getHexagonTargetMachine()));

  // Split up TFRcondsets into conditional transfers.
  PM->add(createHexagonSplitTFRCondSets(getHexagonTargetMachine()));

  return false;
}
