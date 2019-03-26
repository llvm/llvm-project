//===- DRFScopedNoAliasAA.cpp - DRF-based scoped-noalias metadata ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Adds scoped-noalias metadata to memory accesses based on Tapir's parallel
// control flow constructs and the assumption that the function is data-race
// free.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/DRFScopedNoAliasAA.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Tapir.h"

#define DEBUG_TYPE "drf-scoped-noalias"

using namespace llvm;

/// Process Tapir loops within the given function for loop spawning.
class DRFScopedNoAliasImpl {
public:
  DRFScopedNoAliasImpl(Function &F, TaskInfo &TI, AliasAnalysis &AA,
                       LoopInfo *LI)
      : F(F), TI(TI), AA(AA), LI(LI) {
    TI.evaluateParallelState<MaybeParallelTasks>(MPTasks);
  }

  bool run();

private:
  bool populateTaskScopeNoAlias();

  bool populateSubTaskScopeNoAlias(
      const Task *T, MDBuilder &MDB, SmallVectorImpl<Metadata *> &CurrScopes,
      SmallVectorImpl<Metadata *> &CurrNoAlias,
      DenseMap<const Task *, Metadata *> &TaskToScope);

  bool populateTaskScopeNoAliasInBlock(
      const Task *T, BasicBlock *BB, MDBuilder &MDB,
      SmallVectorImpl<Metadata *> &Scopes,
      SmallVectorImpl<Metadata *> &NoAlias);

  Function &F;
  TaskInfo &TI;
  AliasAnalysis &AA;
  LoopInfo *LI;

  MaybeParallelTasks MPTasks;
};

namespace {
struct DRFScopedNoAliasWrapperPass : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  explicit DRFScopedNoAliasWrapperPass() : FunctionPass(ID) {
    initializeDRFScopedNoAliasWrapperPassPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "Assume DRF to Add Scoped-No-Alias Metadata";
  }

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addPreserved<LoopInfoWrapperPass>();
    AU.addRequired<TaskInfoWrapperPass>();
    AU.addPreserved<TaskInfoWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addPreserved<BasicAAWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
  }
};
}  // End of anonymous namespace

char DRFScopedNoAliasWrapperPass::ID = 0;
INITIALIZE_PASS_BEGIN(DRFScopedNoAliasWrapperPass, "drf-scoped-noalias",
                      "Add DRF-based scoped-noalias metadata",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TaskInfoWrapperPass)
INITIALIZE_PASS_END(DRFScopedNoAliasWrapperPass, "drf-scoped-noalias",
                    "Add DRF-based scoped-noalias metadata",
                    false, false)

bool DRFScopedNoAliasImpl::populateTaskScopeNoAliasInBlock(
    const Task *T, BasicBlock *BB, MDBuilder &MDB,
    SmallVectorImpl<Metadata *> &Scopes, SmallVectorImpl<Metadata *> &NoAlias) {
  LLVM_DEBUG(dbgs() << "Processing block " << BB->getName() << " in task "
             << T->getEntry()->getName() << "\n");
  for (Instruction &I : *BB) {
    bool IsArgMemOnlyCall = false, IsFuncCall = false;
    SmallVector<const Value *, 2> PtrArgs;

    if (const LoadInst *LI = dyn_cast<LoadInst>(&I))
      PtrArgs.push_back(LI->getPointerOperand());
    else if (const StoreInst *SI = dyn_cast<StoreInst>(&I))
      PtrArgs.push_back(SI->getPointerOperand());
    else if (const VAArgInst *VAAI = dyn_cast<VAArgInst>(&I))
      PtrArgs.push_back(VAAI->getPointerOperand());
    else if (const AtomicCmpXchgInst *CXI = dyn_cast<AtomicCmpXchgInst>(&I))
      PtrArgs.push_back(CXI->getPointerOperand());
    else if (const AtomicRMWInst *RMWI = dyn_cast<AtomicRMWInst>(&I))
      PtrArgs.push_back(RMWI->getPointerOperand());
    else if (ImmutableCallSite ICS = ImmutableCallSite(&I)) {
      // We don't need to worry about callsites that don't access memory.
      if (ICS.doesNotAccessMemory())
        continue;

      IsFuncCall = true;
      FunctionModRefBehavior MRB = AA.getModRefBehavior(ICS);
      if (MRB == FMRB_OnlyAccessesArgumentPointees ||
          MRB == FMRB_OnlyReadsArgumentPointees)
        IsArgMemOnlyCall = true;

      for (Value *Arg : ICS.args()) {
        // We need to check the underlying objects of all arguments, not just
        // the pointer arguments, because we might be passing pointers as
        // integers, etc.
        // However, if we know that the call only accesses pointer arguments,
        // then we only need to check the pointer arguments.
        if (IsArgMemOnlyCall && !Arg->getType()->isPointerTy())
          continue;

        PtrArgs.push_back(Arg);
      }
    }

    // If we found no pointers, then this instruction is not suitable for
    // pairing with an instruction to receive aliasing metadata.  However, if
    // this is a call, this we might just alias with none of the noalias
    // arguments.
    if (PtrArgs.empty() && !IsFuncCall)
      continue;

    // It is possible that there is only one underlying object, but you need to
    // go through several PHIs to see it, and thus could be repeated in the
    // Objects list.
    bool UsesObjectOutsideTask = false;
    for (const Value *V : PtrArgs) {
      SmallVector<Value *, 4> Objects;
      const DataLayout &DL = F.getParent()->getDataLayout();
      GetUnderlyingObjects(const_cast<Value*>(V), Objects, DL, LI);

      for (Value *O : Objects) {
        LLVM_DEBUG(dbgs() << "Checking object " << *O << "\n");
        // Check if this value is a constant that cannot be derived from any
        // pointer value (we need to exclude constant expressions, for example,
        // that are formed from arithmetic on global symbols).
        bool IsNonPtrConst = isa<ConstantInt>(V) || isa<ConstantFP>(V) ||
                             isa<ConstantPointerNull>(V) ||
                             isa<ConstantDataVector>(V) || isa<UndefValue>(V);
        if (IsNonPtrConst)
          continue;

        // Check if this object was created in this task.
        if (Instruction *OI = dyn_cast<Instruction>(O))
          if (TI.getTaskFor(OI->getParent()) == T)
            continue;

        // This object exists outside the task.
        UsesObjectOutsideTask = true;
        break;
      }
      // Quit early if a pointer argument is found that refers to an object
      // allocated outside of this task.
      if (UsesObjectOutsideTask)
        break;
    }

    // If this instruction does not refer to an object outside of the task,
    // don't add noalias metadata.
    if (!UsesObjectOutsideTask) {
      LLVM_DEBUG(dbgs() << "Instruction " << I
                 << " does not use object outside of task "
                 << T->getEntry()->getName() << "\n");
      continue;
    }

    if (!NoAlias.empty())
      I.setMetadata(LLVMContext::MD_noalias,
                     MDNode::concatenate(
                         I.getMetadata(LLVMContext::MD_noalias),
                         MDNode::get(F.getContext(), NoAlias)));

    if (!Scopes.empty())
      I.setMetadata(
          LLVMContext::MD_alias_scope,
          MDNode::concatenate(I.getMetadata(LLVMContext::MD_alias_scope),
                              MDNode::get(F.getContext(), Scopes)));
  }
  return true;
}
                                     
bool DRFScopedNoAliasImpl::populateSubTaskScopeNoAlias(
    const Task *T, MDBuilder &MDB, SmallVectorImpl<Metadata *> &CurrScopes,
    SmallVectorImpl<Metadata *> &CurrNoAlias,
    DenseMap<const Task *, Metadata *> &TaskToScope) {
  bool Changed = false;
  size_t OrigNoAliasSize = CurrNoAlias.size();

  // FIXME? Separately handle shared EH spindles.
  for (Spindle *S : depth_first<InTask<Spindle *>>(T->getEntrySpindle())) {
    for (const Task *MPT : MPTasks.TaskList[S]) {
      // Don't record noalias scopes for maybe-parallel tasks that enclose the
      // spindle.  These cases arise from parallel loops, which need special
      // alias analysis anyway (e.g., LoopAccessAnalysis).
      if (!MPT->encloses(S->getEntry()))
        CurrNoAlias.push_back(TaskToScope[MPT]);
    }
    // Populate instructions in spindle with scoped-noalias information.
    for (BasicBlock *BB : S->blocks())
      Changed |=
        populateTaskScopeNoAliasInBlock(T, BB, MDB, CurrScopes, CurrNoAlias);

    // Remove the noalias scopes for this spindle.
    CurrNoAlias.erase(CurrNoAlias.begin() + OrigNoAliasSize, CurrNoAlias.end());

    // For each successor spindle in a subtask, recursively populate the
    // scoped-noalias information in that subtask.
    for (Spindle *Succ : successors(S)) {
      if (S->succInSubTask(Succ)) {
        CurrScopes.push_back(TaskToScope[Succ->getParentTask()]);
        populateSubTaskScopeNoAlias(Succ->getParentTask(), MDB, CurrScopes,
                                    CurrNoAlias, TaskToScope);
        CurrScopes.pop_back();
      }
    }
  }

  return Changed;
}

static void createTaskDomainsAndFullScopes(
    const Task *T, MDBuilder &MDB, const Twine ParentName,
    DenseMap<const Task *, MDNode *> &TaskToDomain,
    DenseMap<const Task *, Metadata *> &TaskToScope) {
  // Within the domain of T, create a scope and domain for each subtask.
  for (const Task *SubT : T->subtasks()) {
    const Twine Name = ParentName + "_" + SubT->getEntry()->getName();

    MDNode *NewScope =
      MDB.createAnonymousAliasScope(TaskToDomain[T], ("taskscp_" + Name).str());
    TaskToScope[SubT] = NewScope;
    MDNode *NewDomain =
      MDB.createAnonymousAliasScopeDomain(("taskdom_" + Name).str());
    TaskToDomain[SubT] = NewDomain;

    // Recursively create domains and scopes for subtasks.
    createTaskDomainsAndFullScopes(SubT, MDB, Name, TaskToDomain, TaskToScope);
  }
}

bool DRFScopedNoAliasImpl::populateTaskScopeNoAlias() {
  // Create a domain for the task scopes.
  MDBuilder MDB(F.getContext());
  if (TI.isSerial())
    return false;

  DenseMap<const Task *, MDNode *> TaskToDomain;
  DenseMap<const Task *, Metadata *> TaskToScope;

  // Create a domain and scope for the root task.
  MDNode *NewDomain =
    MDB.createAnonymousAliasScopeDomain(("dom_" + F.getName()).str());
  TaskToDomain[TI.getRootTask()] = NewDomain;
  MDNode *NewScope =
    MDB.createAnonymousAliasScope(NewDomain, ("scp_" + F.getName()).str());
  TaskToScope[TI.getRootTask()] = NewScope;

  // Recursively create task domains and scopes for subtasks.
  createTaskDomainsAndFullScopes(TI.getRootTask(), MDB, F.getName(),
                                 TaskToDomain, TaskToScope);

  SmallVector<Metadata *, 4> Scopes, NoAlias;
  return populateSubTaskScopeNoAlias(TI.getRootTask(), MDB, Scopes, NoAlias,
                                     TaskToScope);
}

bool DRFScopedNoAliasImpl::run() {
  return populateTaskScopeNoAlias();
}

bool DRFScopedNoAliasWrapperPass::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  TaskInfo &TI = getAnalysis<TaskInfoWrapperPass>().getTaskInfo();
  AliasAnalysis &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
  LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  return DRFScopedNoAliasImpl(F, TI, AA, &LI).run();
}

// createDRFScopedNoAliasPass - Provide an entry point to create this pass.
//
namespace llvm {
FunctionPass *createDRFScopedNoAliasWrapperPass() {
  return new DRFScopedNoAliasWrapperPass();
}
} // end namespace llvm

PreservedAnalyses DRFScopedNoAliasPass::run(Function &F,
                                            FunctionAnalysisManager &AM) {
  TaskInfo &TI = AM.getResult<TaskAnalysis>(F);
  AliasAnalysis &AA = AM.getResult<AAManager>(F);
  LoopInfo &LI = AM.getResult<LoopAnalysis>(F);

  DRFScopedNoAliasImpl(F, TI, AA, &LI).run();

  PreservedAnalyses PA;
  PA.preserve<LoopAnalysis>();
  PA.preserve<TaskAnalysis>();
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<BasicAA>();
  PA.preserve<GlobalsAA>();
  return PA;
}
