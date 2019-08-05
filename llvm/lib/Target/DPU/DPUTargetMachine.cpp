//===-- DPUTargetMachine.cpp - Define TargetMachine for Sparc   -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DPUTargetMachine.h"
#include "DPU.h"
#include "DPUISelDAGToDAG.h"
#include "DPUMacroFusion.h"
#include "DPUTargetTransformInfo.h"
#include "MCTargetDesc/DPUMCAsmInfo.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

namespace llvm {
extern Target TheDPUTarget;
}

extern "C" void LLVMInitializeDPUTarget() {
  // Register the target.
  RegisterTargetMachine<DPUTargetMachine> X(TheDPUTarget);
}

static std::string computeDataLayout(const Triple &TT, StringRef CPU,
                                     const TargetOptions &Options) {
  return "e-m:e-p:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:64:64-n32";
}

static Reloc::Model getEffectiveRelocModel(Optional<Reloc::Model> RM) {
  if (!RM.hasValue())
    return Reloc::Static;
  return *RM;
}

DPUTargetMachine::DPUTargetMachine(const Target &T, const Triple &TT,
                                   StringRef CPU, StringRef FS,
                                   const TargetOptions &Options,
                                   Optional<Reloc::Model> RM,
                                   Optional<CodeModel::Model> CM,
                                   CodeGenOpt::Level &OL, bool JIT)
    : LLVMTargetMachine(T, computeDataLayout(TT, CPU, Options), TT, CPU, FS,
                        Options, getEffectiveRelocModel(RM),
                        getEffectiveCodeModel(CM, CodeModel::Small), OL),
      TLOF(make_unique<TargetLoweringObjectFileELF>()),
      Subtarget(TT, CPU, FS, *this) {
  initAsmInfo();
}

TargetTransformInfo
DPUTargetMachine::getTargetTransformInfo(const Function &F) {
  return TargetTransformInfo(DPUTTIImpl(this, F));
}

namespace {
/// DPU Code Generator Pass Configuration Options.
class DPUPassConfig : public TargetPassConfig {
public:
  DPUPassConfig(DPUTargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {}

  DPUTargetMachine &getDPUTargetMachine() const {
    return getTM<DPUTargetMachine>();
  }

  ScheduleDAGInstrs *
  createMachineScheduler(MachineSchedContext *C) const override {
    ScheduleDAGMILive *DAG = createGenericSchedLive(C);
    DAG->addMutation(createDPUMacroFusionDAGMutation());
    return DAG;
  }

  void addIRPasses() override;

  bool addInstSelector() override;

  void addPreEmitPass() override;
  void addPreEmitPass2() override;
};
} // namespace

TargetPassConfig *DPUTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new DPUPassConfig(*this, PM);
}

void DPUPassConfig::addIRPasses() {
  addPass(createAtomicExpandPass());
  TargetPassConfig::addIRPasses();
}

bool DPUPassConfig::addInstSelector() {
  addPass(createDPUISelDag(getDPUTargetMachine(), getOptLevel()));
  return false;
}

void DPUPassConfig::addPreEmitPass() {
  DPUTargetMachine &TM = getDPUTargetMachine();
  addPass(createDPUMergeComboInstrPass(TM));
}

void DPUPassConfig::addPreEmitPass2() {
  DPUTargetMachine &TM = getDPUTargetMachine();
  addPass(createDPUResolveMacroInstrPass(TM));
}
