//===-- VETargetMachine.cpp - Define TargetMachine for VE -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "VETargetMachine.h"
#include "TargetInfo/VETargetInfo.h"
#include "VE.h"
#include "VEMachineFunctionInfo.h"
#include "VETargetTransformInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "ve"

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void LLVMInitializeVETarget() {
  // Register the target.
  RegisterTargetMachine<VETargetMachine> X(getTheVETarget());

  PassRegistry &PR = *PassRegistry::getPassRegistry();
  initializeVEAsmPrinterPass(PR);
  initializeVEDAGToDAGISelLegacyPass(PR);
}

static Reloc::Model getEffectiveRelocModel(std::optional<Reloc::Model> RM) {
  return RM.value_or(Reloc::Static);
}

namespace {
class VEELFTargetObjectFile : public TargetLoweringObjectFileELF {
  void Initialize(MCContext &Ctx, const TargetMachine &TM) override {
    TargetLoweringObjectFileELF::Initialize(Ctx, TM);
    InitializeELF(TM.Options.UseInitArray);
  }
};
} // namespace

static std::unique_ptr<TargetLoweringObjectFile> createTLOF() {
  return std::make_unique<VEELFTargetObjectFile>();
}

/// Create an Aurora VE architecture model
VETargetMachine::VETargetMachine(const Target &T, const Triple &TT,
                                 StringRef CPU, StringRef FS,
                                 const TargetOptions &Options,
                                 std::optional<Reloc::Model> RM,
                                 std::optional<CodeModel::Model> CM,
                                 CodeGenOptLevel OL, bool JIT)
    : CodeGenTargetMachineImpl(T, TT.computeDataLayout(), TT, CPU, FS, Options,
                               getEffectiveRelocModel(RM),
                               getEffectiveCodeModel(CM, CodeModel::Small), OL),
      TLOF(createTLOF()),
      Subtarget(TT, std::string(CPU), std::string(FS), *this) {
  initAsmInfo();
}

VETargetMachine::~VETargetMachine() = default;

TargetTransformInfo
VETargetMachine::getTargetTransformInfo(const Function &F) const {
  return TargetTransformInfo(std::make_unique<VETTIImpl>(this, F));
}

MachineFunctionInfo *VETargetMachine::createMachineFunctionInfo(
    BumpPtrAllocator &Allocator, const Function &F,
    const TargetSubtargetInfo *STI) const {
  return VEMachineFunctionInfo::create<VEMachineFunctionInfo>(Allocator, F,
                                                              STI);
}

namespace {
/// VE Code Generator Pass Configuration Options.
class VEPassConfig : public TargetPassConfig {
public:
  VEPassConfig(VETargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {}

  VETargetMachine &getVETargetMachine() const {
    return getTM<VETargetMachine>();
  }

  void addIRPasses() override;
  bool addInstSelector() override;
  void addPreEmitPass() override;
};
} // namespace

TargetPassConfig *VETargetMachine::createPassConfig(PassManagerBase &PM) {
  return new VEPassConfig(*this, PM);
}

void VEPassConfig::addIRPasses() {
  // VE requires atomic expand pass.
  addPass(createAtomicExpandLegacyPass());
  TargetPassConfig::addIRPasses();
}

bool VEPassConfig::addInstSelector() {
  addPass(createVEISelDag(getVETargetMachine()));
  return false;
}

void VEPassConfig::addPreEmitPass() {
  // LVLGen should be called after scheduling and register allocation
  addPass(createLVLGenPass());
}
