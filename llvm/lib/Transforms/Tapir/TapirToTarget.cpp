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

#include "llvm/Transforms/Tapir/TapirToTarget.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Tapir.h"
#include "llvm/Transforms/Tapir/LoweringUtils.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

#define DEBUG_TYPE "tapir2target"

using namespace llvm;

class TapirToTargetImpl {
public:
  TapirToTargetImpl(Module &M,
                    function_ref<DominatorTree &(Function &)> GetDT,
                    function_ref<TaskInfo &(Function &)> GetTI,
                    function_ref<AssumptionCache &(Function &)> GetAC,
                    TapirTarget *Target)
      : Target(Target), M(M), GetDT(GetDT), GetTI(GetTI), GetAC(GetAC) {
    assert(this->Target);
  }
  ~TapirToTargetImpl() {
    if (Target)
      delete Target;
  }

  bool run();

private:
  bool unifyReturns(Function &F);
  void processFunction(Function &F, SmallVectorImpl<Function *> &NewHelpers);
  TaskOutlineMapTy outlineAllTasks(Function &F, DominatorTree &DT,
                                   AssumptionCache &AC, TaskInfo &TI);
  bool processSimpleABI(Function &F);
  bool processRootTask(Function &F, TaskOutlineMapTy &TaskToOutline,
                       DominatorTree &DT, AssumptionCache &AC, TaskInfo &TI);
  bool processOutlinedTask(
      Task *T, TaskOutlineMapTy &TaskToOutline, DominatorTree &DT,
      AssumptionCache &AC, TaskInfo &TI);

private:
  TapirTarget *Target = nullptr;

  Module &M;

  function_ref<DominatorTree &(Function &)> GetDT;
  function_ref<TaskInfo &(Function &)> GetTI;
  function_ref<AssumptionCache &(Function &)> GetAC;
};

bool TapirToTargetImpl::unifyReturns(Function &F) {
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

// Outline all tasks in this function in post order.
TaskOutlineMapTy
TapirToTargetImpl::outlineAllTasks(Function &F, DominatorTree &DT,
                                   AssumptionCache &AC, TaskInfo &TI) {
  TaskOutlineMapTy TaskToOutline;

  // Determine the inputs for all tasks.
  DenseMap<Task *, ValueSet> TaskInputs = findAllTaskInputs(F, DT, TI);
  // Traverse the tasks in this function in post order.
  for (Task *T : post_order(TI.getRootTask())) {
    // At this point, all subtasks of T must have been processed.  Replace their
    // detaches with calls.
    for (Task *SubT : T->subtasks())
      TaskToOutline[SubT].replaceReplCall(
          replaceDetachWithCallToOutline(SubT, TaskToOutline[SubT]));

    // Outline the task, if necessary, and add the outlined function to the
    // mapping.
    if (T->isRootTask())
      break;

    // If task T tracks any exception-handling spindles for its subtasks, remove
    // any dependencies from those shared-EH spindles to T.
    for (Spindle *SharedEH : T->shared_eh_spindles()) {
      // Remove blocks in shared-EH spindles from PHI's in T.
      for (Spindle::SpindleEdge &SuccEdge : SharedEH->out_edges()) {
        Spindle *Succ = SuccEdge.first;
        BasicBlock *Exit = SuccEdge.second;
        if (Succ->getParentTask() != T || T->containsSharedEH(Succ))
          continue;
        Succ->getEntry()->removePredecessor(Exit);
      }
    }

    ValueToValueMapTy VMap;
    TaskToOutline[T] = outlineTask(T, TaskInputs[T], VMap, &AC, &DT);
    // Update subtask outline info to reflect the fact that their spawner was
    // outlined.
    for (Task *SubT : T->subtasks())
      TaskToOutline[SubT].remapOutlineInfo(VMap);
  }

  return TaskToOutline;
}

bool TapirToTargetImpl::processSimpleABI(Function &F) {
  // Get the simple Tapir instructions to process, including syncs and
  // loop-grainsize calls.
  SmallVector<SyncInst *, 8> Syncs;
  SmallVector<CallInst *, 8> GrainsizeCalls;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      // Record calls to get Tapir-loop grainsizes.
      if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(&I))
        if (Intrinsic::tapir_loop_grainsize == II->getIntrinsicID())
          GrainsizeCalls.push_back(II);

      // Record sync instructions in this function.
      if (SyncInst *SI = dyn_cast<SyncInst>(&I))
        Syncs.push_back(SI);
    }
  }

  // Lower simple Tapir instructions in this function.  Collect the set of
  // helper functions generated by this process.
  bool Changed = false;

  // Lower calls to get Tapir-loop grainsizes.
  while (!GrainsizeCalls.empty()) {
    CallInst *GrainsizeCall = GrainsizeCalls.pop_back_val();
    LLVM_DEBUG(dbgs() << "Lowering grainsize call " << *GrainsizeCall << "\n");
    Target->lowerGrainsizeCall(GrainsizeCall);
    Changed = true;
  }

  // Process the set of syncs.
  while (!Syncs.empty()) {
    SyncInst *SI = Syncs.pop_back_val();
    Target->createSync(*SI);
    Changed = true;
  }

  return Changed;
}

bool TapirToTargetImpl::processRootTask(
    Function &F, TaskOutlineMapTy &TaskToOutline, DominatorTree &DT,
    AssumptionCache &AC, TaskInfo &TI) {
  bool Changed = false;
  if (!TI.isSerial()) {
    Changed = true;
    Target->processSpawner(F);

    for (Task *SubT : TI.getRootTask()->subtasks())
      Target->processSubTaskCall(TaskToOutline[SubT], DT);
  }
  Changed |= processSimpleABI(F);
  return Changed;
}

bool TapirToTargetImpl::processOutlinedTask(
    Task *T, TaskOutlineMapTy &TaskToOutline, DominatorTree &DT,
    AssumptionCache &AC, TaskInfo &TI) {
  Function &F = *TaskToOutline[T].Outline;
  Target->processOutlinedTask(F);
  if (!T->isSerial()) {
    Target->processSpawner(F);

    for (Task *SubT : T->subtasks())
      Target->processSubTaskCall(TaskToOutline[SubT], DT);
  }
  processSimpleABI(F);
  return true;
}

void TapirToTargetImpl::processFunction(
    Function &F, SmallVectorImpl<Function *> &NewHelpers) {
  unifyReturns(F);

  LLVM_DEBUG(dbgs() << "Tapir: Processing function " << F.getName() << "\n");

  Target->preProcessFunction(F);

  // Get the necessary analysis results.
  DominatorTree &DT = GetDT(F);
  TaskInfo &TI = GetTI(F);
  AssumptionCache &AC = GetAC(F);

  // Outline all tasks in a target-oblivious manner.
  TaskOutlineMapTy TaskToOutline = outlineAllTasks(F, DT, AC, TI);

  if (verifyFunction(F, &errs()))
    llvm_unreachable("Outlining tasks produced bad IR!");

  // Perform target-specific processing of this function and all newly created
  // helpers.
  for (Task *T : post_order(TI.getRootTask())) {
    if (T->isRootTask())
      processRootTask(F, TaskToOutline, DT, AC, TI);
    else {
      processOutlinedTask(T, TaskToOutline, DT, AC, TI);
      NewHelpers.push_back(TaskToOutline[T].Outline);
    }
  }
  Target->postProcessFunction(F);
  for (Function *H : NewHelpers)
    Target->postProcessHelper(*H);

  if (verifyFunction(F, &errs()))
    llvm_unreachable("Tapir lowering produced bad IR!");

  return;
}

bool TapirToTargetImpl::run() {
  // Add functions that detach to the work list.
  SmallVector<Function *, 4> WorkList;
  for (Function &F : M)
    if (Target->shouldProcessFunction(F))
      WorkList.push_back(&F);

  if (WorkList.empty())
    return false;

  bool Changed = false;
  while (!WorkList.empty()) {
    // Process the next function.
    Function *F = WorkList.pop_back_val();
    SmallVector<Function *, 4> NewHelpers;
    processFunction(*F, NewHelpers);
    Changed |= !NewHelpers.empty();
    // Check the generated helper functions to see if any need to be processed,
    // that is, to see if any of them themselves detach a subtask.
    for (Function *Helper : NewHelpers)
      if (Target->shouldProcessFunction(*Helper))
        WorkList.push_back(Helper);
  }
  return Changed;
}

PreservedAnalyses TapirToTargetPass::run(Module &M, ModuleAnalysisManager &AM) {
  auto &TLI = AM.getResult<TargetLibraryAnalysis>(M);
  auto &FAM =
    AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto GetDT =
    [&FAM](Function &F) -> DominatorTree & {
      return FAM.getResult<DominatorTreeAnalysis>(F);
    };
  auto GetTI =
    [&FAM](Function &F) -> TaskInfo & {
      return FAM.getResult<TaskAnalysis>(F);
    };
  auto GetAC =
    [&FAM](Function &F) -> AssumptionCache & {
      return FAM.getResult<AssumptionAnalysis>(F);
    };

  bool Changed = false;
  TapirTargetID TargetID = TLI.getTapirTarget();
  Changed |= TapirToTargetImpl(M, GetDT, GetTI, GetAC,
                               getTapirTargetFromID(TargetID)).run();

  if (Changed)
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}

namespace {
struct LowerTapirToTarget : public ModulePass {
  static char ID; // Pass identification, replacement for typeid
  explicit LowerTapirToTarget()
      : ModulePass(ID) {
    initializeLowerTapirToTargetPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "LowerTapirToTarget";
  }

  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addRequired<TaskInfoWrapperPass>();
  }
};
}  // End of anonymous namespace

char LowerTapirToTarget::ID = 0;
INITIALIZE_PASS_BEGIN(LowerTapirToTarget, "tapir2target",
                      "Lower Tapir to Target ABI", false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TaskInfoWrapperPass)
INITIALIZE_PASS_END(LowerTapirToTarget, "tapir2target",
                    "Lower Tapir to Target ABI", false, false)

bool LowerTapirToTarget::runOnModule(Module &M) {
  if (skipModule(M))
    return false;
  auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
  TapirTargetID TargetID = TLI.getTapirTarget();

  auto GetDT =
    [this](Function &F) -> DominatorTree & {
      return this->getAnalysis<DominatorTreeWrapperPass>(F).getDomTree();
    };
  auto GetTI =
    [this](Function &F) -> TaskInfo & {
      return this->getAnalysis<TaskInfoWrapperPass>(F).getTaskInfo();
    };
  AssumptionCacheTracker *ACT = &getAnalysis<AssumptionCacheTracker>();
  auto GetAC =
    [&ACT](Function &F) -> AssumptionCache & {
      return ACT->getAssumptionCache(F);
    };

  bool Changed = false;
  Changed |= TapirToTargetImpl(M, GetDT, GetTI, GetAC,
                               getTapirTargetFromID(TargetID)).run();
  return Changed;
}

// createLowerTapirToTargetPass - Provide an entry point to create this pass.
//
namespace llvm {
ModulePass *createLowerTapirToTargetPass() {
  return new LowerTapirToTarget();
}
}
