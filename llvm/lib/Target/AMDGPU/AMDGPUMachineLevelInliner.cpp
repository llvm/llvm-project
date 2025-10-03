//===-- AMDGPUMachineLevelInliner.cpp - AMDGPU Machine Level Inliner ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUMachineLevelInliner.h"
#include "AMDGPU.h"
#include "AMDGPUMachineModuleInfo.h"
#include "AMDGPUSubtarget.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManagers.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassTimingInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-machine-level-inliner"

namespace {
class AMDGPUInliningPassManager : public FPPassManager {
public:
  static char ID;

  explicit AMDGPUInliningPassManager() : FPPassManager(ID) {}

  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool doFinalization(Module &M) override;

  StringRef getPassName() const override {
    return "AMDGPU Inlining Pass Manager";
  }
};

/// AMDGPUInliningAnchor - A machine function pass that serves as an anchor for
/// setting up the AMDGPU inlining pass manager infrastructure. It makes sure
/// the inliner is run via an AMDGPUInliningPassManager. It can be run well in
/// advance of the inliner as long as there are only FunctionPasses in between.
class AMDGPUInliningAnchor : public MachineFunctionPass {
public:
  static char ID; // Pass identification

  AMDGPUInliningAnchor() : MachineFunctionPass(ID) {}

  // We don't really need to process any functions here.
  bool runOnMachineFunction(MachineFunction &MF) override { return false; }

  void getAnalysisUsage(AnalysisUsage &AU) const override;
  StringRef getPassName() const override;

  /// Prepare the pass manager stack for the inliner. This will push an
  /// `AMDGPUInliningPassManager` onto the stack.
  void preparePassManager(PMStack &Stack) override;
};

} // end anonymous namespace.

// Pass identification
char AMDGPUMachineLevelInliner::ID = 0;
char AMDGPUInliningPassManager::ID = 0;
char AMDGPUInliningAnchor::ID = 0;

char &llvm::AMDGPUMachineLevelInlinerID = AMDGPUMachineLevelInliner::ID;
char &llvm::AMDGPUInliningAnchorID = AMDGPUInliningAnchor::ID;

INITIALIZE_PASS_BEGIN(AMDGPUMachineLevelInliner, DEBUG_TYPE,
                      "AMDGPU Machine Level Inliner", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineModuleInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AMDGPUInliningAnchor)
INITIALIZE_PASS_END(AMDGPUMachineLevelInliner, DEBUG_TYPE,
                    "AMDGPU Machine Level Inliner", false, false)

INITIALIZE_PASS_BEGIN(AMDGPUInliningAnchor, "amdgpu-inlining-anchor",
                      "AMDGPU Inlining Anchor", false, true)
INITIALIZE_PASS_DEPENDENCY(MachineModuleInfoWrapperPass)
INITIALIZE_PASS_END(AMDGPUInliningAnchor, "amdgpu-inlining-anchor",
                    "AMDGPU Inlining Anchor", false, true)

AMDGPUMachineLevelInliner::AMDGPUMachineLevelInliner()
    : MachineFunctionPass(ID) {
  initializeAMDGPUMachineLevelInlinerPass(*PassRegistry::getPassRegistry());
}

void AMDGPUMachineLevelInliner::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<MachineModuleInfoWrapperPass>();
  AU.addRequired<AMDGPUInliningAnchor>();
  AU.addPreserved<MachineModuleInfoWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool AMDGPUMachineLevelInliner::runOnMachineFunction(MachineFunction &MF) {
  MachineModuleInfo &MMI = getAnalysis<MachineModuleInfoWrapperPass>().getMMI();

  Function &F = MF.getFunction();
  if (shouldInlineCallsTo(F)) {
    // Mark the function as machine-inlined in AMDGPUMachineModuleInfo. This
    // tells the inlining pass manager to stop processing it.
    auto &AMMMI = MMI.getObjFileInfo<AMDGPUMachineModuleInfo>();
    AMMMI.addMachineInlinedFunction(F);

    return false;
  }

  return false;
}

FunctionPass *llvm::createAMDGPUMachineLevelInlinerPass() {
  return new AMDGPUMachineLevelInliner();
}

// The implementation here follows FPPassManager::runOnFunction but with some
// simplifications since we know we're not running this on LLVM IR (so the
// Function itself will never be changed, only its corresponding
// MachineFunction). It also checks after every pass if the function has been
// inlined, and stops running passes on it if that's the case.
bool AMDGPUInliningPassManager::runOnFunction(Function &F) {
  if (F.isDeclaration())
    return false;

  MachineModuleInfo &MMI = getAnalysis<MachineModuleInfoWrapperPass>().getMMI();
  auto &AMMMI = MMI.getObjFileInfo<AMDGPUMachineModuleInfo>();

  // Don't run anything on functions that have already been inlined.
  if (AMMMI.isMachineInlinedFunction(F))
    return false;

  bool Changed = false;
  populateInheritedAnalysis(TPM->activeStack);

  // Store name outside of loop to avoid redundant calls.
  const StringRef Name = F.getName();
  llvm::TimeTraceScope FunctionScope("OptFunction", Name);

  for (Pass *P : PassVector) {
    FunctionPass *FP = static_cast<FunctionPass *>(P);
    bool LocalChanged = false;

    // Call getPassName only when required. The call itself is fairly cheap, but
    // still virtual and repeated calling adds unnecessary overhead.
    llvm::TimeTraceScope PassScope(
        "RunPass", [FP]() { return std::string(FP->getPassName()); });

    dumpPassInfo(FP, EXECUTION_MSG, ON_FUNCTION_MSG, Name);
    dumpRequiredSet(FP);

    initializeAnalysisImpl(FP);

    {
      PassManagerPrettyStackEntry X(FP, F);
      TimeRegion PassTimer(getPassTimer(FP));

      LocalChanged |= FP->runOnFunction(F);
    }

    Changed |= LocalChanged;
    if (LocalChanged)
      dumpPassInfo(FP, MODIFICATION_MSG, ON_FUNCTION_MSG, Name);
    dumpPreservedSet(FP);
    dumpUsedSet(FP);

    // If the pass has marked the function for inlining, skip remaining passes.
    if (AMMMI.isMachineInlinedFunction(F))
      break;

    verifyPreservedAnalysis(FP);
    if (LocalChanged)
      removeNotPreservedAnalysis(FP);
    recordAvailableAnalysis(FP);
    removeDeadPasses(FP, Name, ON_FUNCTION_MSG);
  }

  return Changed;
}

bool AMDGPUInliningPassManager::doFinalization(Module &M) {
  MachineModuleInfo &MMI = getAnalysis<MachineModuleInfoWrapperPass>().getMMI();
  auto &AMMMI = MMI.getObjFileInfo<AMDGPUMachineModuleInfo>();

  // Free MachineFunction for all inlined functions. Other machine functions are
  // being freed via the FreeMachineFunction pass which runs at the end of
  // the pass pipeline.
  // TODO: This is a good place to run the rest of the pass pipeline for
  // functions that have been only partially inlined and which still need to be
  // emitted. This way they can be in their inlining-ready form until we're done
  // processing all their callers, and then still go through the rest of the
  // pipeline.
  for (Function *F : AMMMI.getMachineInlinedFunctions())
    MMI.deleteMachineFunctionFor(*F);

  return FPPassManager::doFinalization(M);
}

void AMDGPUInliningPassManager::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<MachineModuleInfoWrapperPass>();
  AU.addPreserved<MachineModuleInfoWrapperPass>();
  ModulePass::getAnalysisUsage(AU);
}

FunctionPass *llvm::createAMDGPUInliningAnchorPass() {
  return new AMDGPUInliningAnchor();
}

void AMDGPUInliningAnchor::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<MachineModuleInfoWrapperPass>();
  AU.setPreservesAll();
}

StringRef AMDGPUInliningAnchor::getPassName() const {
  return "AMDGPU Inlining Anchor";
}

void AMDGPUInliningAnchor::preparePassManager(PMStack &PMS) {
  // Replace the top FunctionPass manager (if there is one) with an
  // AMDGPUInliningPassManager.
  while (!PMS.empty() &&
         PMS.top()->getPassManagerType() > PMT_FunctionPassManager)
    PMS.pop();

  assert(!PMS.empty() && "Unable to create AMDGPU Inlining Pass Manager");
  PMDataManager *PMD = PMS.top();

  // Nothing to do if it's already an AMDGPUInliningPassManager.
  if (PMD->getAsPass()->getPassID() == &AMDGPUInliningPassManager::ID)
    return;

  // If we have a different FunctionPass manager, pop it.
  if (PMD->getPassManagerType() == PMT_FunctionPassManager) {
    PMS.pop();
    PMD = PMS.top();
  }

  // Create and push our custom AMDGPUInliningPassManager.
  auto *PM = new AMDGPUInliningPassManager();
  PM->populateInheritedAnalysis(PMS);

  PMTopLevelManager *TPM = PMD->getTopLevelManager();
  TPM->addIndirectPassManager(PM);

  PM->assignPassManager(PMS, PMD->getPassManagerType());

  PMS.push(PM);
}
