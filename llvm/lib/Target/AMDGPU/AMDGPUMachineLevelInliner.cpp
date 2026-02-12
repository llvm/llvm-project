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
  if (mayInlineCallsTo(F)) {
    // Mark the function as machine-inlined in AMDGPUMachineModuleInfo. This
    // tells the inlining pass manager to stop processing it.
    auto &AMMMI = MMI.getObjFileInfo<AMDGPUMachineModuleInfo>();
    AMMMI.addMachineInliningCandidate(F);

    return false;
  }

  bool Changed = false;

  // Can't inline anything if there aren't any calls.
  MachineFrameInfo &MFI = MF.getFrameInfo();
  if (!MFI.hasCalls() && !MFI.hasTailCall())
    return false;

  // Collect calls to inline.
  SmallVector<MachineInstr *, 4> CallsToInline;
  const SIInstrInfo *TII = MF.getSubtarget<GCNSubtarget>().getInstrInfo();

  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      if (!MI.isCall())
        continue;

      const MachineOperand *CalleeOp =
          TII->getNamedOperand(MI, AMDGPU::OpName::callee);
      if (!CalleeOp || !CalleeOp->isGlobal())
        continue;

      auto *CalledFunc = dyn_cast<Function>(CalleeOp->getGlobal());
      assert(CalledFunc && "Expected global callee operand");

      // Partial inlining is not supported yet, because the inlining pass
      // manager does not run the rest of the pass pipeline on functions that
      // get inlined (including outputting code for them).
      if (CalledFunc == &F)
        report_fatal_error("Recursive calls in whole wave functions are "
                           "not supported yet");

      if (mayInlineCallsTo(*CalledFunc))
        CallsToInline.push_back(&MI);
    }
  }

  // Perform the actual inlining.
  for (MachineInstr *CallMI : CallsToInline) {
    const MachineOperand *CalleeOp =
        TII->getNamedOperand(*CallMI, AMDGPU::OpName::callee);
    assert(CalleeOp && CalleeOp->isGlobal() &&
           isa<Function>(CalleeOp->getGlobal()));
    auto *Callee = cast<Function>(CalleeOp->getGlobal());

    MachineFunction *CalleeMF = MMI.getMachineFunction(*Callee);
    assert(CalleeMF && "Couldn't get MachineFunction for callee");
    assert(!CalleeMF->empty() && "Machine function body is empty");

    LLVM_DEBUG(dbgs() << "    Inlining machine call to: " << Callee->getName()
                      << " (" << CalleeMF->size() << " basic blocks)\n");

    inlineMachineFunction(&MF, CallMI, CalleeMF, TII);
    cleanupAfterInlining(&MF, CallMI, TII);
    Changed = true;
  }

  return Changed;
}

void AMDGPUMachineLevelInliner::inlineMachineFunction(MachineFunction *CallerMF,
                                                      MachineInstr *CallMI,
                                                      MachineFunction *CalleeMF,
                                                      const SIInstrInfo *TII) {

  MachineBasicBlock *CallMBB = CallMI->getParent();
  MachineBasicBlock *ContinuationMBB =
      CallMBB->splitAt(*CallMI, /*UpdateLiveIns=*/true);

  // Splitting marks the ContinuationMBB as a successor, but we want to
  // fallthrough to the body of the inlined function instead.
  CallMBB->removeSuccessor(ContinuationMBB);

  // First we clone all the blocks and build a map, so we can patch up the
  // control flow while cloning their content in a second pass.
  DenseMap<const MachineBasicBlock *, MachineBasicBlock *> ClonedBlocks;
  for (const MachineBasicBlock &OrigMBB : *CalleeMF) {
    MachineBasicBlock *ClonedMBB =
        CallerMF->CreateMachineBasicBlock(OrigMBB.getBasicBlock());
    CallerMF->insert(ContinuationMBB->getIterator(), ClonedMBB);
    ClonedBlocks[&OrigMBB] = ClonedMBB;
  }

  MachineBasicBlock *ClonedEntry = ClonedBlocks[&CalleeMF->front()];
  CallMBB->addSuccessor(ClonedEntry);

  for (const MachineBasicBlock &OrigMBB : *CalleeMF) {
    MachineBasicBlock *ClonedMBB = ClonedBlocks[&OrigMBB];

    for (MachineBasicBlock *OrigSucc : OrigMBB.successors())
      ClonedMBB->addSuccessor(ClonedBlocks[OrigSucc]);

    for (auto &LiveIn : OrigMBB.liveins())
      ClonedMBB->addLiveIn(LiveIn);

    // Also add the registers that are live across the call. We can get them
    // from ContinuationMBB because it was split with `UpdateLiveIns` set to
    // true.
    for (const auto &LiveIn : ContinuationMBB->liveins())
      ClonedMBB->addLiveIn(LiveIn);
    ClonedMBB->sortUniqueLiveIns();

    for (const MachineInstr &OrigMI : OrigMBB) {
      // Bundled instructions are handled by the bundle header.
      if (OrigMI.isBundledWithPred())
        continue;

      if (OrigMI.isReturn()) {
        if (OrigMI.isCall())
          reportFatalInternalError("Tail calls not supported yet"); // FIXME
        TII->insertBranch(*ClonedMBB, ContinuationMBB, nullptr,
                          SmallVector<MachineOperand, 0>(), DebugLoc());
        ClonedMBB->addSuccessor(ContinuationMBB);
      } else {
        MachineInstr &ClonedMI = CallerMF->cloneMachineInstrBundle(
            *ClonedMBB, ClonedMBB->end(), OrigMI);
        ClonedMI.dropMemRefs(*CallerMF); // FIXME: Update them instead.

        for (MachineOperand &MO : ClonedMI.operands())
          if (MO.isMBB())
            MO.setMBB(ClonedBlocks[MO.getMBB()]);
      }
    }
  }
}

void AMDGPUMachineLevelInliner::cleanupAfterInlining(
    MachineFunction *CallerMF, MachineInstr *CallMI,
    const SIInstrInfo *TII) const {
  MachineRegisterInfo &MRI = CallerMF->getRegInfo();
  const TargetRegisterInfo *TRI = CallerMF->getSubtarget().getRegisterInfo();

  // Clean up instructions setting up the callee operand (this is important
  // because we won't be generating any code for that symbol, so we don't want
  // references to it dangling around).
  const MachineOperand *CalleeGlobalOp =
      TII->getNamedOperand(*CallMI, AMDGPU::OpName::callee);
  const MachineOperand *CalleeRegOp =
      TII->getNamedOperand(*CallMI, AMDGPU::OpName::src0);

  assert(CalleeGlobalOp && CalleeRegOp &&
         "Couldn't get operands for call inst");
  assert(CalleeGlobalOp->isGlobal() && "Unexpected operand kind");
  assert(CalleeRegOp->isReg() && "Unexpected operand kind");

  const GlobalValue *CalleeGV = CalleeGlobalOp->getGlobal();
  Register CalleeReg = CalleeRegOp->getReg();

  SmallVector<MachineInstr *, 4> ToErase;
  ToErase.push_back(CallMI);

  // Check each subreg of the callee register (e.g., s0 and s1 for s[0:1]).
  for (MCSubRegIterator SR(CalleeReg, TRI, /*IncludeSelf=*/true); SR.isValid();
       ++SR) {
    MCPhysReg SubReg = *SR;

    // Usually the instructions setting up the callee are a S_MOV_B32
    // referencing the global op. Look for them and remove them. In the general
    // case, we'll want to check that these instructions have no other uses, but
    // for now this should be safe because the addresses of whole wave functions
    // may not be used for anything other than direct calls.
    for (MachineInstr &DefMI : MRI.def_instructions(SubReg)) {
      // Check if this def instruction references the callee global
      for (const MachineOperand &MO : DefMI.operands()) {
        if (MO.isGlobal() && MO.getGlobal() == CalleeGV) {
          ToErase.push_back(&DefMI);
          break;
        }
      }
    }
  }

  for (MachineInstr *MI : ToErase)
    MI->eraseFromParent();
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
  if (AMMMI.isMachineInliningCandidate(F))
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
    if (AMMMI.isMachineInliningCandidate(F))
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
  for (Function *F : AMMMI.getMachineInliningCandidates())
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
