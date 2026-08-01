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
#include "BPFTargetLoweringObjectFile.h"
#include "BPFTargetTransformInfo.h"
#include "MCTargetDesc/BPFMCAsmInfo.h"
#include "TargetInfo/BPFTargetInfo.h"
#include "llvm/CodeGen/GlobalISel/IRTranslator.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelect.h"
#include "llvm/CodeGen/GlobalISel/Legalizer.h"
#include "llvm/CodeGen/GlobalISel/RegBankSelect.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Utils/SimplifyCFGOptions.h"
#include <optional>
using namespace llvm;

cl::opt<bool> DisableMIPeephole("disable-bpf-peephole", cl::Hidden,
                                cl::desc("Disable machine peepholes for BPF"));

static cl::opt<bool>
    DisableCheckUnreachable("bpf-disable-trap-unreachable", cl::Hidden,
                            cl::desc("Disable Trap Unreachable for BPF"));

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void LLVMInitializeBPFTarget() {
  // Register the target.
  RegisterTargetMachine<BPFTargetMachine> X(getTheBPFleTarget());
  RegisterTargetMachine<BPFTargetMachine> Y(getTheBPFbeTarget());
  RegisterTargetMachine<BPFTargetMachine> Z(getTheBPFTarget());

  PassRegistry &PR = *PassRegistry::getPassRegistry();
  initializeGlobalISel(PR);
  initializeBPFAsmPrinterPass(PR);
  initializeBPFCheckAndAdjustIRLegacyPass(PR);
  initializeBPFMIPeepholeLegacyPass(PR);
  initializeBPFMIExpandStackArgPseudosLegacyPass(PR);
  initializeBPFMIPreEmitPeepholeLegacyPass(PR);
  initializeBPFDAGToDAGISelLegacyPass(PR);
  initializeBPFMISimplifyPatchableLegacyPass(PR);
  initializeBPFMIPreEmitCheckingLegacyPass(PR);
}

static Reloc::Model getEffectiveRelocModel(std::optional<Reloc::Model> RM) {
  return RM.value_or(Reloc::PIC_);
}

BPFTargetMachine::BPFTargetMachine(const Target &T, const Triple &TT,
                                   StringRef CPU, StringRef FS,
                                   const TargetOptions &Options,
                                   std::optional<Reloc::Model> RM,
                                   std::optional<CodeModel::Model> CM,
                                   CodeGenOptLevel OL, bool JIT)
    : CodeGenTargetMachineImpl(T, TT.computeDataLayout(), TT, CPU, FS, Options,
                               getEffectiveRelocModel(RM),
                               getEffectiveCodeModel(CM, CodeModel::Small), OL),
      TLOF(std::make_unique<BPFTargetLoweringObjectFileELF>()),
      Subtarget(TT, std::string(CPU), std::string(FS), *this) {
  if (!DisableCheckUnreachable) {
    this->Options.TrapUnreachable = true;
    this->Options.NoTrapAfterNoreturn = true;
  }

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

  bool addIRTranslator() override;
  bool addLegalizeMachineIR() override;
  bool addRegBankSelect() override;
  bool addGlobalInstructionSelect() override;
};
}

TargetPassConfig *BPFTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new BPFPassConfig(*this, PM);
}

void BPFPassConfig::addIRPasses() {
  addPass(createAtomicExpandLegacyPass());
  addPass(createBPFCheckAndAdjustIRLegacyPass());

  TargetPassConfig::addIRPasses();
}

TargetTransformInfo
BPFTargetMachine::getTargetTransformInfo(const Function &F) const {
  return TargetTransformInfo(std::make_unique<BPFTTIImpl>(this, F));
}

// Install an instruction selector pass using
// the ISelDag to gen BPF code.
bool BPFPassConfig::addInstSelector() {
  addPass(createBPFISelDag(getBPFTargetMachine()));

  return false;
}

void BPFPassConfig::addMachineSSAOptimization() {
  addPass(createBPFMISimplifyPatchableLegacyPass());

  // The default implementation must be called first as we want eBPF
  // Peephole ran at last.
  TargetPassConfig::addMachineSSAOptimization();

  const BPFSubtarget *Subtarget = getBPFTargetMachine().getSubtargetImpl();
  if (!DisableMIPeephole) {
    if (Subtarget->getHasAlu32())
      addPass(createBPFMIPeepholeLegacyPass());
  }
}

void BPFPassConfig::addPreEmitPass() {
  addPass(createBPFMIPreEmitCheckingLegacyPass());
  if (!DisableMIPeephole) {
    addPass(createBPFMIExpandStackArgPseudosLegacyPass());
    addPass(createBPFMIPreEmitPeepholeLegacyPass());
  }
}

bool BPFPassConfig::addIRTranslator() {
  addPass(new IRTranslator());
  return false;
}

bool BPFPassConfig::addLegalizeMachineIR() {
  addPass(new Legalizer());
  return false;
}

bool BPFPassConfig::addRegBankSelect() {
  addPass(new RegBankSelect());
  return false;
}

bool BPFPassConfig::addGlobalInstructionSelect() {
  addPass(new InstructionSelect(getOptLevel()));
  return false;
}
