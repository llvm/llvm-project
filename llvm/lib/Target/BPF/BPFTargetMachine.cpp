//===-- BPFTargetMachine.cpp - Define TargetMachine for BPF ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the info about BPF target spec.
//
//===----------------------------------------------------------------------===//

#include "BPFTargetMachine.h"
#include "BPF.h"
#include "BPFTargetTransformInfo.h"
#include "MCTargetDesc/BPFMCAsmInfo.h"
#include "TargetInfo/BPFTargetInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Utils/SimplifyCFGOptions.h"
#include <optional>
using namespace llvm;

static cl::
opt<bool> DisableMIPeephole("disable-bpf-peephole", cl::Hidden,
                            cl::desc("Disable machine peepholes for BPF"));

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeBPFTarget() {
  // Register the target.
  RegisterTargetMachine<BPFTargetMachine> X(getTheBPFleTarget());
  RegisterTargetMachine<BPFTargetMachine> Y(getTheBPFbeTarget());
  RegisterTargetMachine<BPFTargetMachine> Z(getTheBPFTarget());

  PassRegistry &PR = *PassRegistry::getPassRegistry();
  initializeBPFAbstractMemberAccessLegacyPassPass(PR);
  initializeBPFPreserveDITypePass(PR);
  initializeBPFIRPeepholePass(PR);
  initializeBPFAdjustOptPass(PR);
  initializeBPFCheckAndAdjustIRPass(PR);
  initializeBPFMIPeepholePass(PR);
  initializeBPFMIPeepholeTruncElimPass(PR);
  initializeBPFDAGToDAGISelPass(PR);
}

// DataLayout: little or big endian
static std::string computeDataLayout(const Triple &TT) {
  if (TT.getArch() == Triple::bpfeb)
    return "E-m:e-p:64:64-i64:64-i128:128-n32:64-S128";
  else
    return "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128";
}

static Reloc::Model getEffectiveRelocModel(std::optional<Reloc::Model> RM) {
  return RM.value_or(Reloc::PIC_);
}

BPFTargetMachine::BPFTargetMachine(const Target &T, const Triple &TT,
                                   StringRef CPU, StringRef FS,
                                   const TargetOptions &Options,
                                   std::optional<Reloc::Model> RM,
                                   std::optional<CodeModel::Model> CM,
                                   CodeGenOpt::Level OL, bool JIT)
    : LLVMTargetMachine(T, computeDataLayout(TT), TT, CPU, FS, Options,
                        getEffectiveRelocModel(RM),
                        getEffectiveCodeModel(CM, CodeModel::Small), OL),
      TLOF(std::make_unique<TargetLoweringObjectFileELF>()),
      Subtarget(TT, std::string(CPU), std::string(FS), *this) {
  initAsmInfo();

  BPFMCAsmInfo *MAI =
      static_cast<BPFMCAsmInfo *>(const_cast<MCAsmInfo *>(AsmInfo.get()));
  MAI->setDwarfUsesRelocationsAcrossSections(!Subtarget.getUseDwarfRIS());
}

namespace {
// BPF Code Generator Pass Configuration Options.
class BPFPassConfig : public TargetPassConfig {
public:
  BPFPassConfig(BPFTargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {}

  BPFTargetMachine &getBPFTargetMachine() const {
    return getTM<BPFTargetMachine>();
  }

  void addIRPasses() override;
  bool addInstSelector() override;
  void addMachineSSAOptimization() override;
  void addPreEmitPass() override;
};
}

TargetPassConfig *BPFTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new BPFPassConfig(*this, PM);
}

void BPFTargetMachine::registerPassBuilderCallbacks(PassBuilder &PB) {
  PB.registerPipelineStartEPCallback(
      [=](ModulePassManager &MPM, OptimizationLevel) {
        FunctionPassManager FPM;
        FPM.addPass(BPFAbstractMemberAccessPass(this));
        FPM.addPass(BPFPreserveDITypePass());
        FPM.addPass(BPFIRPeepholePass());
        MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
      });
  PB.registerPeepholeEPCallback([=](FunctionPassManager &FPM,
                                    OptimizationLevel Level) {
    FPM.addPass(SimplifyCFGPass(SimplifyCFGOptions().hoistCommonInsts(true)));
  });
  PB.registerPipelineEarlySimplificationEPCallback(
      [=](ModulePassManager &MPM, OptimizationLevel) {
        MPM.addPass(BPFAdjustOptPass());
      });
}

void BPFPassConfig::addIRPasses() {
  addPass(createBPFCheckAndAdjustIR());
  TargetPassConfig::addIRPasses();
}

TargetTransformInfo
BPFTargetMachine::getTargetTransformInfo(const Function &F) const {
  return TargetTransformInfo(BPFTTIImpl(this, F));
}

// Install an instruction selector pass using
// the ISelDag to gen BPF code.
bool BPFPassConfig::addInstSelector() {
  addPass(createBPFISelDag(getBPFTargetMachine()));

  return false;
}

void BPFPassConfig::addMachineSSAOptimization() {
  addPass(createBPFMISimplifyPatchablePass());

  // The default implementation must be called first as we want eBPF
  // Peephole ran at last.
  TargetPassConfig::addMachineSSAOptimization();

  const BPFSubtarget *Subtarget = getBPFTargetMachine().getSubtargetImpl();
  if (!DisableMIPeephole) {
    if (Subtarget->getHasAlu32())
      addPass(createBPFMIPeepholePass());
    addPass(createBPFMIPeepholeTruncElimPass());
  }
}

void BPFPassConfig::addPreEmitPass() {
  addPass(createBPFMIPreEmitCheckingPass());
  if (getOptLevel() != CodeGenOpt::None)
    if (!DisableMIPeephole)
      addPass(createBPFMIPreEmitPeepholePass());
}
