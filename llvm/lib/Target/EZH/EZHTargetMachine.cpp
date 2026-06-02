//===-- EZHTargetMachine.cpp - Define TargetMachine for EZH ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "EZHTargetMachine.h"
#include "EZH.h"
#include "EZHMachineFunctionInfo.h"
#include "EZHTargetObjectFile.h"
#include "TargetInfo/EZHTargetInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/MC/TargetRegistry.h"
#include <optional>

using namespace llvm;

namespace {
class EZHTTIImpl : public BasicTTIImplBase<EZHTTIImpl> {
  using BaseT = BasicTTIImplBase<EZHTTIImpl>;
  using TTI = TargetTransformInfo;
  friend BaseT;
  friend TargetTransformInfoImplBase;

  const EZHSubtarget *ST;
  const EZHTargetLowering *TLI;

  const EZHSubtarget *getST() const { return ST; }
  const EZHTargetLowering *getTLI() const { return TLI; }

public:
  explicit EZHTTIImpl(const EZHTargetMachine *TM, const Function &F)
      : BaseT(TM, F.getDataLayout()), ST(TM->getSubtargetImpl(F)),
        TLI(ST->getTargetLowering()) {}
};
} // namespace

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void LLVMInitializeEZHTarget() {
  RegisterTargetMachine<EZHTargetMachine> registered_target(getTheEZHTarget());
}

static Reloc::Model getEffectiveRelocModel(std::optional<Reloc::Model> RM) {
  return Reloc::Static;
}

EZHTargetMachine::EZHTargetMachine(const Target &T, const Triple &TT,
                                   StringRef Cpu, StringRef FeatureString,
                                   const TargetOptions &Options,
                                   std::optional<Reloc::Model> RM,
                                   std::optional<CodeModel::Model> CodeModel,
                                   CodeGenOptLevel OptLevel, bool JIT)
    : CodeGenTargetMachineImpl(
          T, TT.computeDataLayout(), TT, Cpu, FeatureString, Options,
          getEffectiveRelocModel(RM),
          getEffectiveCodeModel(CodeModel, CodeModel::Medium), OptLevel),
      Subtarget(TT, Cpu, FeatureString, *this, Options, getCodeModel(),
                OptLevel),
      TLOF(new EZHTargetObjectFile()) {
  initAsmInfo();
}

TargetTransformInfo
EZHTargetMachine::getTargetTransformInfo(const Function &F) const {
  return TargetTransformInfo(std::make_unique<EZHTTIImpl>(this, F));
}

MachineFunctionInfo *EZHTargetMachine::createMachineFunctionInfo(
    BumpPtrAllocator &Allocator, const Function &F,
    const TargetSubtargetInfo *STI) const {
  return EZHMachineFunctionInfo::create<EZHMachineFunctionInfo>(Allocator, F,
                                                                STI);
}

namespace {
class EZHPassConfig : public TargetPassConfig {
public:
  EZHPassConfig(EZHTargetMachine &TM, PassManagerBase *PassManager)
      : TargetPassConfig(TM, *PassManager) {}

  EZHTargetMachine &getEZHTargetMachine() const {
    return getTM<EZHTargetMachine>();
  }

  bool addInstSelector() override;
  void addPostRegAlloc() override;
  void addPreSched2() override;
  void addPreEmitPass() override;
  void addPreEmitPass2() override;
};
} // namespace

TargetPassConfig *
EZHTargetMachine::createPassConfig(PassManagerBase &PassManager) {
  return new EZHPassConfig(*this, &PassManager);
}

bool EZHPassConfig::addInstSelector() {
  addPass(createEZHISelDag(getEZHTargetMachine()));
  return false;
}

void EZHPassConfig::addPostRegAlloc() {}

void EZHPassConfig::addPreSched2() {
  if (getOptLevel() != CodeGenOptLevel::None)
    addPass(&IfConverterID);
}

void EZHPassConfig::addPreEmitPass() {}

void EZHPassConfig::addPreEmitPass2() {
  addPass(createEZHBitSliceInjectionPass());
  addPass(createEZHConstantIslandPass());
}
