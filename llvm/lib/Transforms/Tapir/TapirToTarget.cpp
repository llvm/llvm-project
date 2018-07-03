//===- TapirToTarget.cpp - Convert Tapir into parallel-runtime calls ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass converts functions that use Tapir instructions to call out to a
// target parallel runtime system.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Tapir.h"
#include "llvm/Transforms/Tapir/LoweringUtils.h"

#define DEBUG_TYPE "tapir2target"

using namespace llvm;

static cl::opt<TapirTargetType> ClTapirTarget(
    "tapir-target", cl::desc("Target runtime for Tapir"),
    cl::init(TapirTargetType::Cilk),
    cl::values(clEnumValN(TapirTargetType::None,
                          "none", "None"),
               clEnumValN(TapirTargetType::Serial,
                          "serial", "Serial code"),
               clEnumValN(TapirTargetType::Cilk,
                          "cilk", "Cilk Plus"),
               clEnumValN(TapirTargetType::OpenMP,
                          "openmp", "OpenMP"),
               clEnumValN(TapirTargetType::CilkR,
                          "cilkr", "CilkR")));

namespace {

struct LowerTapirToTarget : public ModulePass {
  static char ID; // Pass identification, replacement for typeid
  TapirTarget* tapirTarget;
  explicit LowerTapirToTarget(TapirTarget* tapirTarget = nullptr)
      : ModulePass(ID), tapirTarget(tapirTarget) {
    if (!this->tapirTarget)
      this->tapirTarget = getTapirTargetFromType(ClTapirTarget);
    assert(this->tapirTarget);
    initializeLowerTapirToTargetPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "Simple Lowering of Tapir to Target ABI";
  }

  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<TaskInfoWrapperPass>();
  }
private:
  ValueToValueMapTy DetachCtxToStackFrame;
  bool unifyReturns(Function &F);
  SmallVectorImpl<Function *> *processFunction(Function &F, DominatorTree &DT,
                                               AssumptionCache &AC);
};
}  // End of anonymous namespace

char LowerTapirToTarget::ID = 0;
INITIALIZE_PASS_BEGIN(LowerTapirToTarget, "tapir2target",
                      "Lower Tapir to Target ABI", false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TaskInfoWrapperPass)
INITIALIZE_PASS_END(LowerTapirToTarget, "tapir2target",
                    "Lower Tapir to Target ABI", false, false)


bool LowerTapirToTarget::unifyReturns(Function &F) {
  SmallVector<BasicBlock *, 4> ReturningBlocks;
  for (BasicBlock &BB : F)
    if (isa<ReturnInst>(BB.getTerminator()))
      ReturningBlocks.push_back(&BB);

  // If this function already has a single return, then terminate early.
  if (ReturningBlocks.size() == 1)
    return false;

  BasicBlock *NewRetBlock = BasicBlock::Create(F.getContext(),
                                               "UnifiedReturnBlock", &F);
  PHINode *PN = nullptr;
  if (F.getReturnType()->isVoidTy()) {
    ReturnInst::Create(F.getContext(), nullptr, NewRetBlock);
  } else {
    // If the function doesn't return void... add a PHI node to the block...
    PN = PHINode::Create(F.getReturnType(), ReturningBlocks.size(),
                         "UnifiedRetVal");
    NewRetBlock->getInstList().push_back(PN);
    ReturnInst::Create(F.getContext(), PN, NewRetBlock);
  }

  // Loop over all of the blocks, replacing the return instruction with an
  // unconditional branch.
  //
  for (BasicBlock *BB : ReturningBlocks) {
    // Add an incoming element to the PHI node for every return instruction that
    // is merging into this new block...
    if (PN)
      PN->addIncoming(BB->getTerminator()->getOperand(0), BB);

    BB->getInstList().pop_back();  // Remove the return insn
    BranchInst::Create(NewRetBlock, BB);
  }
  return true;
}

SmallVectorImpl<Function *> *LowerTapirToTarget::processFunction(
    Function &F, DominatorTree &DT, AssumptionCache &AC) {
  if (unifyReturns(F))
    DT.recalculate(F);

  DEBUG(dbgs() << "Tapir: Processing function " << F.getName() << "\n");

  tapirTarget->preProcessFunction(F);

  // Get the detaches and syncs to process
  SmallVector<DetachInst *, 8> Detaches;
  SmallVector<SyncInst *, 8> Syncs;
  SmallVector<CallInst *, 8> GrainsizeCalls;
  {
    SmallVector<BasicBlock *, 32> WorkList;
    SmallPtrSet<BasicBlock *, 32> Visited;
    WorkList.push_back(&(F.getEntryBlock()));
    while (!WorkList.empty()) {
      BasicBlock *CurBB = WorkList.pop_back_val();
      if (!Visited.insert(CurBB).second)
        continue;

      // Record calls to get Tapir-loop grainsizes.
      for (Instruction &I : *CurBB)
        if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(&I)) {
          if (Intrinsic::tapir_loop_grainsize == II->getIntrinsicID())
            GrainsizeCalls.push_back(II);
        }

      // Record detach instructions in this function.
      if (DetachInst *DI = dyn_cast<DetachInst>(CurBB->getTerminator())) {
        Detaches.push_back(DI);
        // Skip pushing the detached task onto the work list.  Any nested tasks
        // will be handled on a subsequent run of processFunction() on the
        // generated helper.
        WorkList.push_back(DI->getContinue());
        if (DI->hasUnwindDest())
          WorkList.push_back(DI->getUnwindDest());
        continue;
      }

      // Record sync instructions in this function.
      if (SyncInst *SI = dyn_cast<SyncInst>(CurBB->getTerminator()))
        Syncs.push_back(SI);

      for (BasicBlock *Succ : successors(CurBB))
        WorkList.push_back(Succ);
    }
  }

  // Lower Tapir instructions in this function.  Collect the set of helper
  // functions generated by this process.
  bool Changed = false;

  // Lower calls to get Tapir-loop grainsizes.
  while (!GrainsizeCalls.empty()) {
    CallInst *GrainsizeCall = GrainsizeCalls.pop_back_val();
    DEBUG(dbgs() << "Lowering grainsize call " << *GrainsizeCall << "\n");
    tapirTarget->lowerGrainsizeCall(GrainsizeCall);
    Changed = true;
  }

  SmallVector<Function *, 4> *NewHelpers = new SmallVector<Function *, 4>();
  // Process the set of detaches backwards, in order to process the innermost
  // detached tasks first.
  while (!Detaches.empty()) {
    DetachInst *DI = Detaches.pop_back_val();
    // Lower a detach instruction, and collect the helper function generated in
    // this process for executing the detached task.
    Function *Helper = tapirTarget->createDetach(*DI, DetachCtxToStackFrame,
                                                 DT, AC);
    NewHelpers->push_back(Helper);
    Changed = true;
  }

  // Process the set of syncs.
  while (!Syncs.empty()) {
    SyncInst *SI = Syncs.pop_back_val();
    tapirTarget->createSync(*SI, DetachCtxToStackFrame);
    Changed = true;
  }

  if (!Changed) return NewHelpers;

  if (verifyFunction(F, &errs()))
    llvm_unreachable("Tapir lowering produced bad IR!");

  tapirTarget->postProcessFunction(F);
  for (Function *H : *NewHelpers)
    tapirTarget->postProcessHelper(*H);

  return NewHelpers;
}

static int64_t countRedundantSyncs(TaskInfo &TI, BasicBlock *B) {
  Spindle *S = TI.getSpindleFor(B);
  if (!S->isPhi())
    return 1;

  SmallVector<Spindle *, 8> WorkList;
  SmallVector<Spindle *, 8> ToProcess;
  SmallPtrSet<Spindle *, 8> Visited;
  DenseMap<Spindle *, unsigned> MightBeSynced;
  WorkList.push_back(S);

  while (!WorkList.empty()) {
    Spindle *SP = WorkList.pop_back_val();

    if (!Visited.insert(SP).second) continue;
    if (!SP->isPhi()) {
      MightBeSynced[SP] = SP->isSync();
      continue;
    }

    bool Queue = true;
    for (BasicBlock *PredB : predecessors(SP->getEntry())) {
      // if (isa<ReattachInst>(PredB->getTerminator()) ||
      //     isDetachedRethrow(PredB->getTerminator())) {
      if (TI.getTaskFor(PredB) != TI.getTaskFor(SP)) {
        Queue = false;
        MightBeSynced[SP] = 0;
        break;
      }
    }
    if (!Queue) continue;

    ToProcess.push_back(SP);
    for (Spindle *PredS : TI.predecessors(SP))
      WorkList.push_back(PredS);
  }

  for (Spindle *S : reverse(ToProcess)) {
    MightBeSynced[S] = 0;
    for (Spindle *Pred : TI.predecessors(S)) {
      if (MightBeSynced[Pred])
        MightBeSynced[S] |= MightBeSynced[Pred];
    }
  }
  return MightBeSynced[S];
}

static void analyzeSyncs(TaskInfo &TI, Function *F) {
  int64_t numRedundantSyncs = 0;
  for (BasicBlock &B : *F) {
    if (!isa<SyncInst>(B.getTerminator()))
      continue;

    numRedundantSyncs += countRedundantSyncs(TI, &B);
  }
  dbgs() << "Counted " << numRedundantSyncs << " partially redundant syncs.\n";
}

bool LowerTapirToTarget::runOnModule(Module &M) {
  if (skipModule(M))
    return false;

  // Add functions that detach to the work list.
  SmallVector<Function *, 4> WorkList;
  for (Function &F : M) {
    if (tapirTarget->shouldProcessFunction(F)) {
      dbgs() << "Checking " << F.getName() << "\n";
      TaskInfo &TI = getAnalysis<TaskInfoWrapperPass>(F).getTaskInfo();
      analyzeSyncs(TI, &F);
      WorkList.push_back(&F);
    }
  }

  if (WorkList.empty())
    return false;

  bool Changed = false;
  std::unique_ptr<SmallVectorImpl<Function *>> NewHelpers;
  while (!WorkList.empty()) {
    // Process the next function.
    Function *F = WorkList.pop_back_val();
    DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>(*F).getDomTree();
    AssumptionCacheTracker &ACT = getAnalysis<AssumptionCacheTracker>();
    NewHelpers.reset(processFunction(*F, DT, ACT.getAssumptionCache(*F)));
    Changed |= !NewHelpers->empty();
    // Check the generated helper functions to see if any need to be processed,
    // that is, to see if any of them themselves detach a subtask.
    for (Function *Helper : *NewHelpers)
      if (tapirTarget->shouldProcessFunction(*Helper))
        WorkList.push_back(Helper);
  }
  return Changed;
}

// createLowerTapirToTargetPass - Provide an entry point to create this pass.
//
namespace llvm {
ModulePass *createLowerTapirToTargetPass(TapirTarget* tapirTarget) {
  return new LowerTapirToTarget(tapirTarget);
}
}
