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
#include "SuperHMachineFunctionInfo.h"
#include "SuperHSubtarget.h"
#include "TargetInfo/SuperHTargetInfo.h"
#include "llvm/CodeGen/BranchFoldingPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/Compiler.h"
#include <memory>
#include <optional>

using namespace llvm;

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void LLVMInitializeSuperHTarget() {
  RegisterTargetMachine<SuperHTargetMachine> SH(getTheSuperHTarget());
  RegisterTargetMachine<SuperHTargetMachine> SHLE(getTheSuperHLETarget());

  PassRegistry &Registry = *PassRegistry::getPassRegistry();
  initializeSuperHAsmPrinterPass(Registry);
  initializeSuperHExpandPseudoPass(Registry);
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
  void addPostRegAlloc() override;
  void addPreEmitPass2() override;
  SuperHTargetMachine &getSuperHTargetMachine() const {
    return getTM<SuperHTargetMachine>();
  }
};

bool SuperHPassConfig::addInstSelector() {
  addPass(createSuperHISelDag(getSuperHTargetMachine(), getOptLevel()));
  return false;
}

void SuperHPassConfig::addPostRegAlloc() {
  addPass(createSuperHFrameFixupPass());
}

void SuperHPassConfig::addPreEmitPass2() {
  addPass(createSuperHExpandPseudoPass());
  addPass(createSuperHFillDelaySlotsPass());

  // Inserts Constant Islands. Block sizes cannot be increased after this point,
  // as this may push the branch ranges and load offsets of accessing constant
  // pools out of range.
  addPass(createSuperHConstantIslandPass());
}

} // namespace



//
//      TARGET MACHINE
//

/// Processes a CPU name.
static StringRef getCPU(StringRef CPU, const Triple &TT) {
#define CASE(ARCH) \
  case Triple::SuperHSubArch_ ## ARCH: return "sh" # ARCH;

  switch(TT.getSubArch()) {
  CASE(1);
  CASE(2);
  CASE(2a);
  CASE(2e);
  CASE(3);
  CASE(3e);
  CASE(4);
  CASE(4a);
  default:
    if (CPU.empty() || CPU == "generic") {
      return "sh4";
    }
    return CPU;
  }
#undef CASE
}

SuperHTargetMachine::~SuperHTargetMachine() {}

/// Create a SuperH architecture model.
SuperHTargetMachine::SuperHTargetMachine(const Target &T, const Triple &TT,
                                         StringRef CPU, StringRef FS,
                                         const TargetOptions &Options,
                                         std::optional<Reloc::Model> RM,
                                         std::optional<CodeModel::Model> CM,
                                         CodeGenOptLevel OL, bool JIT)
    : CodeGenTargetMachineImpl(T, TT, CPU, FS, Options,
                               RM.value_or(Reloc::Static),
                               getEffectiveCodeModel(CM, CodeModel::Small), OL),
      TLOF(std::make_unique<TargetLoweringObjectFileELF>()),
      ST(std::make_unique<SuperHSubtarget>(std::string(getCPU(CPU, TT)), std::string(FS), *this)) {
  initAsmInfo();
}

TargetPassConfig *SuperHTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new SuperHPassConfig(*this, PM);
}

const SuperHSubtarget *
SuperHTargetMachine::getSubtargetImpl() const {
  return ST.get();
}

const SuperHSubtarget *
SuperHTargetMachine::getSubtargetImpl(const Function &F) const {
  return ST.get();
}

MachineFunctionInfo *SuperHTargetMachine::createMachineFunctionInfo(
    BumpPtrAllocator &Allocator, const Function &F,
    const TargetSubtargetInfo *STI) const {
  return SuperHMachineFunctionInfo::create<SuperHMachineFunctionInfo>(Allocator,
                                                                      F, STI);
}