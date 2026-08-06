//===-- PISATargetMachine.cpp - Define TargetMachine for PISA -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PISATargetMachine.h"
#include "PISA.h"
#include "PISAMachineFunctionInfo.h"
#include "PISATargetObjectFile.h"
#include "TargetInfo/PISATargetInfo.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/GlobalISel/IRTranslator.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelect.h"
#include "llvm/CodeGen/GlobalISel/Legalizer.h"
#include "llvm/CodeGen/GlobalISel/RegBankSelect.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsPISA.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/PISAAddrSpace.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;

// NOLINTNEXTLINE(readability-identifier-naming)
extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializePISATarget() {
  // Register the target.
  RegisterTargetMachine<PISATargetMachine> Y(getThePISATarget());

  PassRegistry &PR = *PassRegistry::getPassRegistry();
  initializeGlobalISel(PR);
  initializePISAPreLegalizerCombinerPass(PR);
  initializePISAPostLegalizerCombinerPass(PR);
}

MachineFunctionInfo *PISATargetMachine::createMachineFunctionInfo(
    BumpPtrAllocator &Allocator, const Function &F,
    const TargetSubtargetInfo *STI) const {
  return PISAMachineFunctionInfo::create<PISAMachineFunctionInfo>(
      Allocator, F, static_cast<const PISASubtarget *>(STI));
}

static Reloc::Model getEffectiveRelocModel(std::optional<Reloc::Model> RM) {
  if (!RM)
    return Reloc::PIC_;
  return *RM;
}

// Pin PISATargetObjectFile's vtables to this file.
PISATargetObjectFile::~PISATargetObjectFile() {}

PISATargetMachine::PISATargetMachine(const Target &T, const Triple &TT,
                                     StringRef CPU, StringRef FS,
                                     const TargetOptions &Options,
                                     std::optional<Reloc::Model> RM,
                                     std::optional<CodeModel::Model> CM,
                                     CodeGenOptLevel OL, bool JIT)
    : CodeGenTargetMachineImpl(T, TT.computeDataLayout(), TT, CPU, FS, Options,
                               getEffectiveRelocModel(RM),
                               getEffectiveCodeModel(CM, CodeModel::Small), OL),
      TLOF(std::make_unique<PISATargetObjectFile>()),
      Subtarget(TT, CPU.str(), FS.str(), *this) {
  initAsmInfo();
  setGlobalISel(true);
  setFastISel(false);
  setO0WantsFastISel(false);
  setRequiresStructuredCFG(false);
}

const PISASubtarget *
PISATargetMachine::getSubtargetImpl(const Function &F) const {
  Attribute CPUAttr = F.getFnAttribute("target-cpu");
  Attribute FSAttr = F.getFnAttribute("target-features");

  StringRef CPU =
      CPUAttr.isValid() ? CPUAttr.getValueAsString() : getTargetCPU();
  StringRef FS =
      FSAttr.isValid() ? FSAttr.getValueAsString() : getTargetFeatureString();

  SmallString<128> Key(CPU);
  Key.append(FS);

  auto &I = SubtargetMap[Key];
  if (!I) {
    I = std::make_unique<PISASubtarget>(TargetTriple, CPU.str(), FS.str(),
                                        *this);
  }
  return I.get();
}

unsigned PISATargetMachine::getAssumedAddrSpace(const Value *V) const {
  const auto *Ld = dyn_cast<LoadInst>(V);
  if (!Ld || Ld->getPointerOperand()->getType()->getPointerAddressSpace() !=
                 unsigned(PISAAS::AddressSpace::CONSTANT))
    return ~0U;
  return unsigned(PISAAS::AddressSpace::GLOBAL);
}

std::pair<const Value *, unsigned>
PISATargetMachine::getPredicatedAddrSpace(const Value *V) const {
  auto *II = dyn_cast<IntrinsicInst>(V);
  if (!II)
    return std::make_pair(nullptr, -1);

  switch (II->getIntrinsicID()) {
  case Intrinsic::pisa_isaddr_private:
    return std::make_pair(II->getArgOperand(0),
                          unsigned(PISAAS::AddressSpace::PRIVATE));
  case Intrinsic::pisa_isaddr_global:
    return std::make_pair(II->getArgOperand(0),
                          unsigned(PISAAS::AddressSpace::GLOBAL));
  case Intrinsic::pisa_isaddr_shared:
    return std::make_pair(II->getArgOperand(0),
                          unsigned(PISAAS::AddressSpace::SHARED));
  default:
    break;
  }
  return std::make_pair(nullptr, -1);
}

TargetTransformInfo
PISATargetMachine::getTargetTransformInfo(const Function &F) const {
  return TargetTransformInfo(F.getDataLayout());
}

namespace {
// PISA Code Generator Pass Configuration Options.
//
// PISA is a virtual-register-only target: it maintains virtual registers
// throughout the pipeline and does not run register allocation. This
// configuration wires up the GlobalISel selection stages and disables the
// standard machine passes that assume physical registers exist.
class PISAPassConfig : public TargetPassConfig {
public:
  PISAPassConfig(PISATargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {
    disablePass(&GCLoweringID);
    disablePass(&ShadowStackGCLoweringID);
  }

  PISATargetMachine &getPISATargetMachine() const {
    return getTM<PISATargetMachine>();
  }

  void addIRPasses() override {
    TargetPassConfig::addIRPasses();

    // Disable passes that assume physical registers exist.
    disablePass(&PrologEpilogCodeInserterID);
    disablePass(&MachineLateInstrsCleanupID);
    disablePass(&MachineCopyPropagationID);
    disablePass(&TailDuplicateLegacyID);
    disablePass(&StackMapLivenessID);
    disablePass(&LiveDebugValuesID);
    disablePass(&PostRAMachineSinkingID);
    disablePass(&PostRASchedulerID);
    disablePass(&FuncletLayoutID);
    disablePass(&PatchableFunctionID);
    disablePass(&ShrinkWrapID);
    disablePass(&RemoveLoadsIntoFakeUsesID);
    disablePass(&GCMachineCodeAnalysisID);
  }

  bool addIRTranslator() override {
    addPass(new IRTranslator(getOptLevel()));
    return false;
  }

  void addPreLegalizeMachineIR() override {
    if (getOptLevel() != CodeGenOptLevel::None)
      addPass(createPISAPreLegalizerCombiner());
  }

  bool addLegalizeMachineIR() override {
    addPass(new Legalizer());
    return false;
  }

  void addPreRegBankSelect() override {
    if (getOptLevel() != CodeGenOptLevel::None) {
      addPass(&MachineCSELegacyID);
      addPass(createPISAPostLegalizerCombiner());
    }
  }

  bool addRegBankSelect() override {
    addPass(new RegBankSelect());
    return false;
  }

  // Instruction selection (addGlobalInstructionSelect) is added in a
  // subsequent change together with the PISA instruction selector.

  // PISA does not allocate physical registers.
  FunctionPass *createTargetRegisterAllocator(bool) override { return nullptr; }
  bool addRegAssignAndRewriteFast() override { return false; }
  bool addRegAssignAndRewriteOptimized() override { return false; }
};
} // namespace

TargetPassConfig *PISATargetMachine::createPassConfig(PassManagerBase &PM) {
  return new PISAPassConfig(*this, PM);
}
