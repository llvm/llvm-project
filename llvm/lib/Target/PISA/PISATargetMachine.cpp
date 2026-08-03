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
#include "llvm/CodeGen/DeadMachineInstructionElim.h"
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
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/PISAAddrSpace.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Vectorize/LoadStoreVectorizer.h"

using namespace llvm;

static cl::opt<bool>
    EnableLoadStoreVectorizer("pisa-load-store-vectorizer",
                              cl::desc("Enable load store vectorizer"),
                              cl::init(true), cl::Hidden);

// NOLINTNEXTLINE(readability-identifier-naming)
extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializePISATarget() {
  // Register the target.
  RegisterTargetMachine<PISATargetMachine> Y(getThePISATarget());

  PassRegistry &PR = *PassRegistry::getPassRegistry();
  initializeGlobalISel(PR);
  initializePISALegalizeCallsPass(PR);
  initializePISAEmitIntrinsicsPass(PR);
  initializePISAExpandIntrinsicsPass(PR);
  initializePISALegalizeSubregAccessPass(PR);
  initializePISAPreLegalizerCombinerPass(PR);
  initializePISAPostLegalizerCombinerPass(PR);
  initializePISALegalizePredicatesPass(PR);
  initializePISAReplaceIntrinsicsPass(PR);
  initializePISAPropagateNullPointersPass(PR);
  initializePISACacheHintSelectorPass(PR);
  initializePISAVerifierPass(PR);
  initializePISAOptimizeRedundantCopiesPass(PR);
  initializePISAOptimizeSubregAccessPass(PR);
  initializePISAInsertLifetimeStartPass(PR);
  initializePISAMarkConvergentNoMergePass(PR);
  initializePISAScopeSelectorPass(PR);
  initializePISAVerifyTypesPass(PR);
  initializePISAKernelByValArgsLoweringLegacyPass(PR);
  initializePISALayoutPass(PR);
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
// throughout the pipeline and does not run register allocation.
class PISAPassConfig : public TargetPassConfig {
public:
  PISAPassConfig(PISATargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {
    disablePass(&GCLoweringID);
    disablePass(&ShadowStackGCLoweringID);
    disablePass(&XRayInstrumentationID);
  }

  PISATargetMachine &getPISATargetMachine() const {
    return getTM<PISATargetMachine>();
  }

  void addIRPasses() override;
  void addISelPrepare() override;
  bool addIRTranslator() override;
  void addPreLegalizeMachineIR() override;
  bool addLegalizeMachineIR() override;
  void addPreRegBankSelect() override;
  bool addRegBankSelect() override;
  bool addGlobalInstructionSelect() override;
  FunctionPass *createTargetRegisterAllocator(bool) override { return nullptr; }
  bool addRegAssignAndRewriteFast() override { return false; }
  bool addRegAssignAndRewriteOptimized() override { return false; }
  void addPreRegAlloc() override;
  void addPostRegAlloc() override;
  void addPreEmitPass() override;
};
} // namespace

void PISAPassConfig::addIRPasses() {
  addPass(createPISAVerifierPass());

  // Legalize atomics with LLVM's AtomicExpandPass, driven by the
  // PISATargetLowering atomic hooks. Keep it first in addIRPasses().
  addPass(createAtomicExpandLegacyPass());

  TargetPassConfig::addIRPasses();

  addPass(createPISAPropagateNullPointersPass());
  addPass(createPISAKernelByValArgsLoweringLegacyPass());
  addPass(createPISAExpandIntrinsicsPass());
  addPass(createPISALegalizeCallsPass());

  if ((getOptLevel() != CodeGenOptLevel::None) && EnableLoadStoreVectorizer)
    addPass(createLoadStoreVectorizerPass());

  // A temporary solution to prevent divergent barrier calls.
  addPass(createPISALayoutPass());

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

void PISAPassConfig::addISelPrepare() {
  addPass(createPISAEmitIntrinsicsPass());
  TargetPassConfig::addISelPrepare();
}

bool PISAPassConfig::addIRTranslator() {
  addPass(new IRTranslatorLegacy(getOptLevel()));
  addPass(createPISAVerifyTypesPass());
  addPass(createPISAReplaceIntrinsicsPass());
  return false;
}

void PISAPassConfig::addPreLegalizeMachineIR() {
  if (getOptLevel() != CodeGenOptLevel::None) {
    addPass(createPISALegalizePredicatesPass());
    addPass(createPISAPreLegalizerCombiner());
    addPass(createPISAVerifyTypesPass());
  }
}

bool PISAPassConfig::addLegalizeMachineIR() {
  addPass(new LegalizerLegacy());
  addPass(createPISAVerifyTypesPass());
  return false;
}

void PISAPassConfig::addPreRegBankSelect() {
  if (getOptLevel() != CodeGenOptLevel::None) {
    addPass(&MachineCSELegacyID);
    addPass(createPISAPostLegalizerCombiner());
    addPass(createPISAVerifyTypesPass());
  }
}

bool PISAPassConfig::addRegBankSelect() {
  addPass(new RegBankSelect());
  return false;
}

bool PISAPassConfig::addGlobalInstructionSelect() {
  // The MachineCSE pass doesn't detect common subexpressions on instructions
  // using IMPLICIT_DEF instructions. These are sometimes inserted by
  // InstructionSelect, so we run MachineCSE before that to ensure good CSE.
  if (getOptLevel() != CodeGenOptLevel::None)
    addPass(&MachineCSELegacyID);

  addPass(createPISAVerifyTypesPass());
  addPass(new InstructionSelect());
  addPass(createPISAScopeSelectorPass());
  addPass(createPISACacheHintSelectorPass());
  // G_BUILD_VECTOR will produce IMPLICIT_DEFS that must be removed.
  if (getOptLevel() == CodeGenOptLevel::None)
    addPass(&ProcessImplicitDefsID);

  return false;
}

void PISAPassConfig::addPreRegAlloc() {
  if (getOptLevel() != CodeGenOptLevel::None)
    addPass(&LiveRangeShrinkID);
  TargetPassConfig::addPreRegAlloc();
}

void PISAPassConfig::addPostRegAlloc() {
  addPass(createPISALegalizeSubregAccess());
  if (getOptLevel() != CodeGenOptLevel::None) {
    addPass(createPISAOptimizeSubregAccess());
    addPass(createPISAOptimizeRedundantCopies());
    addPass(&DeadMachineInstructionElimID);
  }
  addPass(createPISAMarkConvergentNoMerge());
  // The machine block placement pass is able to rearrange blocks in a way that
  // breaks control flow for kernels with disabled IFP.
  disablePass(&MachineBlockPlacementID);
  TargetPassConfig::addPostRegAlloc();
}

void PISAPassConfig::addPreEmitPass() {
  // The lifetime.start marker names a *post-coalescing* virtual register, so it
  // must run after Register Coalescer and PISAOptimizeRedundantCopies.
  if (getOptLevel() != CodeGenOptLevel::None)
    addPass(createPISAInsertLifetimeStart());
  TargetPassConfig::addPreEmitPass();
}

TargetPassConfig *PISATargetMachine::createPassConfig(PassManagerBase &PM) {
  return new PISAPassConfig(*this, PM);
}
