//===-- SuperHTargetMachine.cpp - Define TargetMachine for SuperH
//-----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "SuperHTargetMachine.h"
#include "SuperH.h"
#include "SuperHSubtarget.h"
#include "SuperHMachineFunctionInfo.h"
#include "TargetInfo/SuperHTargetInfo.h"
#include "llvm/CodeGen/BranchFoldingPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/Compiler.h"
#include <optional>

using namespace llvm;

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void LLVMInitializeSuperHTarget() {
  RegisterTargetMachine<SuperHTargetMachine> SH(getTheSuperHTarget());
  RegisterTargetMachine<SuperHTargetMachine> SHLE(getTheSuperHLETarget());

  PassRegistry &Registry = *PassRegistry::getPassRegistry();
  initializeSuperHAsmPrinterPass(Registry);
  initializeSuperHFillDelaySlotsPass(Registry);
  initializeSuperHConstantIslandsPass(Registry);
  initializeSuperHDAGToDAGISelLegacyPass(Registry);
}

//
//      PASS CONFIG
//

namespace {
class SuperHPassConfig : public TargetPassConfig {
public:
  SuperHPassConfig(SuperHTargetMachine &TM, PassManagerBase &PM)
    : TargetPassConfig(TM, PM) {}

  bool addInstSelector() override;
  void addPreSched2() override;
  void addPreEmitPass() override;
  void addPreEmitPass2() override;
  SuperHTargetMachine &getSuperHTargetMachine() const {
    return getTM<SuperHTargetMachine>();
  }
};

bool SuperHPassConfig::addInstSelector() {
  addPass(createSuperHISelDag(getSuperHTargetMachine(), getOptLevel()));
  return false;
}

void SuperHPassConfig::addPreSched2() {
}

void SuperHPassConfig::addPreEmitPass() {
  addPass(&BranchFolderPassID);
  addPass(&IfConverterID);
  addPass(createSuperHFillDelaySlotsPass());
}

void SuperHPassConfig::addPreEmitPass2() {

  // Inserts Constant Islands. Block sizes cannot be increased after this point,
  // as this may push the branch ranges and load offsets of accessing constant
  // pools out of range.
  addPass(createSuperHConstantIslandPass());
}

} // namespace


//
//      TARGET MACHINE
//

SuperHTargetMachine::~SuperHTargetMachine() {}

/// Create a SuperH architecture model.
SuperHTargetMachine::SuperHTargetMachine(const Target &T, const Triple &TT,
                                         StringRef CPU, StringRef FS,
                                         const TargetOptions &Options,
                                         std::optional<Reloc::Model> RM,
                                         std::optional<CodeModel::Model> CM,
                                         CodeGenOptLevel OL, bool JIT)
    : CodeGenTargetMachineImpl(T, TT.computeDataLayout(), TT, CPU, FS, Options,
                               RM.value_or(Reloc::Static),
                               getEffectiveCodeModel(CM, CodeModel::Small),
                               OL), TLOF(std::make_unique<TargetLoweringObjectFileELF>()) {

  initAsmInfo();
}

TargetPassConfig *SuperHTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new SuperHPassConfig(*this, PM);
}

const TargetSubtargetInfo *
SuperHTargetMachine::getSubtargetImpl(const Function &F) const {
  Attribute CPUAttr = F.getFnAttribute("target-cpu");
  Attribute TuneAttr = F.getFnAttribute("tune-cpu");
  Attribute FSAttr = F.getFnAttribute("target-features");

  std::string CPU =
      CPUAttr.isValid() ? CPUAttr.getValueAsString().str() : TargetCPU;
  std::string TuneCPU =
      TuneAttr.isValid() ? TuneAttr.getValueAsString().str() : CPU;
  std::string FS =
      FSAttr.isValid() ? FSAttr.getValueAsString().str() : TargetFS;

  resetTargetOptions(F);
  if (!ST) {
    ST = std::make_unique<SuperHSubtarget>(CPU, TuneCPU, FS, *this);
  }
  return ST.get();
}

MachineFunctionInfo *
SuperHTargetMachine::createMachineFunctionInfo(BumpPtrAllocator &Allocator, const Function &F,
                          const TargetSubtargetInfo *STI) const {
  return SuperHMachineFunctionInfo::create<SuperHMachineFunctionInfo>(Allocator, F,
                                                                  STI);
}