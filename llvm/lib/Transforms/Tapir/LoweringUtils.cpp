//===- LoweringUtils.cpp - Utility functions for lowering Tapir -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements several utility functions for lowering Tapir.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Tapir/CilkABI.h"
#include "llvm/Transforms/Tapir/CilkRABI.h"
#include "llvm/Transforms/Tapir/OpenMPABI.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

#define DEBUG_TYPE "tapirlowering"

static cl::opt<bool> StructTaskArgs(
    "use-struct-for-task-args", cl::init(false), cl::Hidden,
    cl::desc("Use a struct to store arguments for detached tasks"));

TapirTarget *llvm::getTapirTargetFromID(TapirTargetID ID) {
  switch (ID) {
  case TapirTargetID::Cilk:
    return new CilkABI();
  case TapirTargetID::OpenMP:
    return new OpenMPABI();
  case TapirTargetID::CilkR:
  case TapirTargetID::Cheetah:
    return new CilkRABI();
  case TapirTargetID::None:
  case TapirTargetID::Serial:
    return nullptr;
  default:
    llvm_unreachable("Invalid TapirTargetID");
  }
}

//----------------------------------------------------------------------------//
// Lowering utilities for Tapir tasks.

/// Helper function to find the inputs and outputs to task T, based only the
/// blocks in T and no subtask of T.
static void
findTaskInputsOutputs(Task *T, ValueSet &Inputs, ValueSet &Outputs,
                      DominatorTree &DT) {
  // Get the sync region for this task's detach, so we can filter it out of
  // this inputs.
  const Value *SyncRegion = nullptr;
  SmallPtrSet<BasicBlock *, 8> UnwindPHIs;
  if (DetachInst *DI = T->getDetach()) {
    SyncRegion = DI->getSyncRegion();
    // Get the PHI nodes that directly or indirectly use the landing pad of the
    // unwind destination of this task's detach.
    getDetachUnwindPHIUses(DI, UnwindPHIs);
  }

  for (Spindle *S : depth_first<InTask<Spindle *>>(T->getEntrySpindle())) {
    LLVM_DEBUG(dbgs() <<
               "Examining spindle for inputs/outputs: " << *S << "\n");
    for (BasicBlock *BB : S->blocks()) {
      // Skip basic blocks that are successors of detached rethrows.  They're
      // dead anyway.
      if (isSuccessorOfDetachedRethrow(BB))
        continue;

      // If a used value is defined outside the region, it's an input.  If an
      // instruction is used outside the region, it's an output.
      for (Instruction &II : *BB) {
        // Examine all operands of this instruction.
        for (User::op_iterator OI = II.op_begin(), OE = II.op_end(); OI != OE;
             ++OI) {
          // PHI nodes in the entry block of a shared-EH exit will be
          // rewritten in any cloned helper, so we skip operands of these PHI
          // nodes for blocks not in this task.
          if (S->isSharedEH() && S->isEntry(BB))
            if (PHINode *PN = dyn_cast<PHINode>(&II)) {
              LLVM_DEBUG(dbgs() <<
                         "\tPHI node in shared-EH spindle: " << *PN << "\n");
              if (!T->simplyEncloses(PN->getIncomingBlock(*OI))) {
                LLVM_DEBUG(dbgs() << "skipping\n");
                continue;
              }
            }
          // If the operand is the sync region of this task's detach, skip it.
          if (SyncRegion == *OI)
            continue;
          // If this operand is defined in the parent, it's an input.
          if (T->definedInParent(*OI))
            Inputs.insert(*OI);
        }
        // Examine all uses of this instruction
        for (User *U : II.users()) {
          // If we find a live use outside of the task, it's an output.
          if (Instruction *I = dyn_cast<Instruction>(U)) {
            // Skip uses in PHI nodes that depend on the unwind landing pad of
            // the detach.
            if (UnwindPHIs.count(I->getParent()))
              continue;
            if (!T->encloses(I->getParent()) &&
                DT.isReachableFromEntry(I->getParent()))
              Outputs.insert(&II);
          }
        }
      }
    }
  }
}

/// Determine the inputs for all tasks in this function.  Returns a map from
/// tasks to their inputs.
///
/// Aggregating all of this work into a single routine allows us to avoid
/// redundant traversals of basic blocks in nested tasks.
DenseMap<Task *, ValueSet>
llvm::findAllTaskInputs(Function &F, DominatorTree &DT, TaskInfo &TI) {
  DenseMap<Task *, ValueSet> TaskInputs;
  for (Task *T : post_order(TI.getRootTask())) {
    // Skip the root task
    if (T->isRootTask()) break;

    LLVM_DEBUG(dbgs() << "Finding inputs/outputs for task @ "
          << T->getEntry()->getName() << "\n");
    ValueSet Inputs, Outputs;
    // Check all inputs of subtasks to determine if they're inputs to this task.
    for (Task *SubT : T->subtasks()) {
      LLVM_DEBUG(dbgs() <<
                 "\tsubtask @ " << SubT->getEntry()->getName() << "\n");

      if (TaskInputs.count(SubT))
        for (Value *V : TaskInputs[SubT])
          if (T->definedInParent(V))
            Inputs.insert(V);
    }

    LLVM_DEBUG({
        dbgs() << "Subtask Inputs:\n";
        for (Value *V : Inputs)
          dbgs() << "\t" << *V << "\n";
        dbgs() << "Subtask Outputs:\n";
        for (Value *V : Outputs)
          dbgs() << "\t" << *V << "\n";
      });
    assert(Outputs.empty() && "Task should have no outputs.");

    // Find additional inputs and outputs of task T by examining blocks in T and
    // not in any subtask of T.
    findTaskInputsOutputs(T, Inputs, Outputs, DT);

    LLVM_DEBUG({
        dbgs() << "Inputs:\n";
        for (Value *V : Inputs)
          dbgs() << "\t" << *V << "\n";
        dbgs() << "Outputs:\n";
        for (Value *V : Outputs)
          dbgs() << "\t" << *V << "\n";
      });
    assert(Outputs.empty() && "Task should have no outputs.");

    // Map the computed inputs to this task.
    TaskInputs[T] = Inputs;
  }
  return TaskInputs;
}

/// Create a structure for storing all arguments to a task.
///
/// NOTE: This function inserts the struct for task arguments in the same
/// location as the Reference compiler and other compilers that lower parallel
/// constructs in the front end.  This location is NOT the correct place,
/// however, for handling tasks that are spawned inside of a serial loop.
std::pair<AllocaInst *, Instruction *>
llvm::createTaskArgsStruct(ValueSet &Inputs, Task *T,
                           Instruction *StorePt, Instruction *LoadPt) {
  assert(T && T->getParentTask() && "Expected spawned task.");
  assert(T->encloses(LoadPt->getParent()) &&
         "Loads of struct arguments must be inside task.");
  assert(!T->encloses(StorePt->getParent()) &&
         "Store of struct arguments must be outside task.");
  assert(T->getParentTask()->encloses(StorePt->getParent()) &&
         "Store of struct arguments expected to be in parent task.");
  SmallVector<Value *, 8> InputsToSort;
  {
    for (Value *V : Inputs)
      InputsToSort.push_back(V);
    Function *F = T->getEntry()->getParent();
    const DataLayout &DL = F->getParent()->getDataLayout();
    std::sort(InputsToSort.begin(), InputsToSort.end(),
              [&DL](const Value *A, const Value *B) {
                return DL.getTypeSizeInBits(A->getType()) >
                  DL.getTypeSizeInBits(B->getType());
              });
  }

  // Get vector of struct inputs and their types.
  SmallVector<Value *, 8> StructInputs;
  SmallVector<Type *, 8> StructIT;
  for (Value *V : InputsToSort) {
    StructInputs.push_back(V);
    StructIT.push_back(V->getType());
  }

  // Create an alloca for this struct in the parent task's entry block.
  AllocaInst *Closure;
  StructType *ST = StructType::get(T->getEntry()->getContext(), StructIT);
  LLVM_DEBUG(dbgs() << "Closure struct type " << *ST << "\n");
  {
    BasicBlock *AllocaInsertBlk = T->getParentTask()->getEntry();
    IRBuilder<> Builder(&*AllocaInsertBlk->getFirstInsertionPt());
    Closure = Builder.CreateAlloca(ST);
  }

  // Add code to store values into struct immediately before detach.
  Instruction *ArgsStart = StorePt;
  IRBuilder<> B(StorePt);
  if (!StructInputs.empty())
    ArgsStart =
      B.CreateStore(StructInputs[0], B.CreateConstGEP2_32(ST, Closure, 0, 0));
  for (unsigned i = 1; i < StructInputs.size(); ++i)
    B.CreateStore(StructInputs[i], B.CreateConstGEP2_32(ST, Closure, 0, i));

  // Add code to load values from struct in task entry and use those loaded
  // values.
  IRBuilder<> B2(LoadPt);
  for (unsigned i = 0; i < StructInputs.size(); ++i) {
    auto STGEP = cast<Instruction>(B2.CreateConstGEP2_32(ST, Closure, 0, i));
    auto STLoad = B2.CreateLoad(STGEP);

    // Update all uses of the struct inputs in the loop body.
    auto UI = StructInputs[i]->use_begin(), E = StructInputs[i]->use_end();
    for (; UI != E;) {
      Use &U = *UI;
      ++UI;
      auto *Usr = dyn_cast<Instruction>(U.getUser());
      if (!Usr || !T->encloses(Usr->getParent()))
        continue;
      U.set(STLoad);
    }
  }

  return std::make_pair(Closure, ArgsStart);
}

Instruction *llvm::fixupHelperInputs(
    Function &F, Task *T, ValueSet &TaskInputs, ValueSet &HelperArgs,
    Instruction *StorePt, Instruction *LoadPt) {
  if (StructTaskArgs) {
    std::pair<AllocaInst *, Instruction *> ArgsStructInfo =
      createTaskArgsStruct(TaskInputs, T, StorePt, LoadPt);
    HelperArgs.insert(ArgsStructInfo.first);
    return ArgsStructInfo.second;
  }

  // Scan for any sret parameters in TaskInputs and add them first.  These
  // parameters must appear first or second in the prototype of the Helper
  // function.
  Value *SRetInput = nullptr;
  if (F.hasStructRetAttr()) {
    Function::arg_iterator ArgIter = F.arg_begin();
    if (F.hasParamAttribute(0, Attribute::StructRet))
      if (TaskInputs.count(&*ArgIter))
        SRetInput = &*ArgIter;
    if (F.hasParamAttribute(1, Attribute::StructRet)) {
      ++ArgIter;
      if (TaskInputs.count(&*ArgIter))
        SRetInput = &*ArgIter;
    }
  }
  if (SRetInput) {
    LLVM_DEBUG(dbgs() << "sret input " << *SRetInput << "\n");
    HelperArgs.insert(SRetInput);
  }

  // Sort the inputs to the task with largest arguments first, in order to
  // improve packing or arguments in memory.
  SmallVector<Value *, 8> InputsToSort;
  for (Value *V : TaskInputs)
    if (V != SRetInput)
      InputsToSort.push_back(V);
  LLVM_DEBUG({
      dbgs() << "After sorting:\n";
      for (Value *V : InputsToSort)
        dbgs() << "\t" << *V << "\n";
    });
  const DataLayout &DL = F.getParent()->getDataLayout();
  std::sort(InputsToSort.begin(), InputsToSort.end(),
            [&DL](const Value *A, const Value *B) {
              return DL.getTypeSizeInBits(A->getType()) >
                DL.getTypeSizeInBits(B->getType());
            });

  // Add the remaining inputs.
  for (Value *V : InputsToSort)
    if (!HelperArgs.count(V))
      HelperArgs.insert(V);

  return StorePt;
}

bool llvm::isSuccessorOfDetachedRethrow(const BasicBlock *B) {
  for (const BasicBlock *Pred : predecessors(B))
    if (isDetachedRethrow(Pred->getTerminator()))
      return true;
  return false;
}

void llvm::getTaskBlocks(Task *T, std::vector<BasicBlock *> &TaskBlocks,
                         SmallPtrSetImpl<BasicBlock *> &ReattachBlocks,
                         SmallPtrSetImpl<BasicBlock *> &DetachedRethrowBlocks,
                         SmallPtrSetImpl<BasicBlock *> &SharedEHEntries) {
  for (Spindle *S : depth_first<InTask<Spindle *>>(T->getEntrySpindle())) {
    // Record the entry blocks of any shared-EH spindles.
    if (S->isSharedEH())
      SharedEHEntries.insert(S->getEntry());

    for (BasicBlock *B : S->blocks()) {
      // Skip basic blocks that are successors of detached rethrows.  They're
      // dead anyway.
      if (isSuccessorOfDetachedRethrow(B))
        continue;

      TaskBlocks.push_back(B);

      // Record the blocks terminated by reattaches and detached rethrows.
      if (isa<ReattachInst>(B->getTerminator()))
        ReattachBlocks.insert(B);
      if (isDetachedRethrow(B->getTerminator()))
        DetachedRethrowBlocks.insert(B);
    }
  }
}

Function *llvm::createHelperForTask(
    Function &F, Task *T, ValueSet &Args, ValueToValueMapTy &VMap,
    AssumptionCache *AC, DominatorTree *DT) {
  // Collect all basic blocks in this task.
  std::vector<BasicBlock *> TaskBlocks;
  // Reattach instructions and detached rethrows in this task might need special
  // handling.
  SmallPtrSet<BasicBlock *, 4> ReattachBlocks;
  SmallPtrSet<BasicBlock *, 4> DetachedRethrowBlocks;
  // Entry blocks of shared-EH spindles may contain PHI nodes that need to be
  // rewritten in the cloned helper.
  SmallPtrSet<BasicBlock *, 4> SharedEHEntries;
  getTaskBlocks(T, TaskBlocks, ReattachBlocks, DetachedRethrowBlocks,
                SharedEHEntries);

  SmallVector<ReturnInst *, 4> Returns;  // Ignore returns cloned.
  ValueSet Outputs;
  DetachInst *DI = T->getDetach();

  Twine NameSuffix = ".otd" + Twine(T->getTaskDepth());
  Function *Helper =
    CreateHelper(Args, Outputs, TaskBlocks, T->getEntry(),
                 DI->getParent(), DI->getContinue(), VMap, F.getParent(),
                 F.getSubprogram() != nullptr, Returns,
                 NameSuffix.str(), &ReattachBlocks,
                 &DetachedRethrowBlocks, &SharedEHEntries, nullptr, nullptr,
                 nullptr, nullptr, nullptr);

  assert(Returns.empty() && "Returns cloned when cloning detached CFG.");

  // Use a fast calling convention for the helper.
  Helper->setCallingConv(CallingConv::Fast);
  // Inlining the helper function is not legal.
  Helper->addFnAttr(Attribute::NoInline);
  // Note that the address of the helper is unimportant.
  Helper->setUnnamedAddr(GlobalValue::UnnamedAddr::Local);
  // The helper is private to this module.
  Helper->setLinkage(GlobalValue::InternalLinkage);

  // Add alignment assumptions to arguments of helper, based on alignment of
  // values in old function.
  AddAlignmentAssumptions(&F, Args, VMap, DI, AC, DT);

  // Move allocas in the newly cloned detached CFG to the entry block of the
  // helper.
  {
    // Collect the end instructions of the task.
    SmallVector<Instruction *, 4> TaskEnds;
    for (BasicBlock *EndBlock : ReattachBlocks)
      TaskEnds.push_back(cast<BasicBlock>(VMap[EndBlock])->getTerminator());
    for (BasicBlock *EndBlock : DetachedRethrowBlocks)
      TaskEnds.push_back(cast<BasicBlock>(VMap[EndBlock])->getTerminator());

    // Move allocas in cloned detached block to entry of helper function.
    BasicBlock *ClonedDetachedBlock = cast<BasicBlock>(VMap[T->getEntry()]);
    MoveStaticAllocasInBlock(&Helper->getEntryBlock(), ClonedDetachedBlock,
                             TaskEnds);

    // We do not need to add new llvm.stacksave/llvm.stackrestore intrinsics,
    // because calling and returning from the helper will automatically manage
    // the stack appropriately.
  }

  return Helper;
}

/// Helper function to unlink task T's exception-handling blocks from T's
/// parent.
static void unlinkTaskEHFromParent(Task *T) {
  DetachInst *DI = T->getDetach();

  // Get the PHI's that use the landing pad of the detach's unwind.
  SmallPtrSet<BasicBlock *, 8> UnwindPHIs;
  getDetachUnwindPHIUses(DI, UnwindPHIs);

  SmallVector<Instruction *, 8> ToRemove;
  // Look through PHI's that use the landing pad of the detach's unwind, and
  // update those PHI's to not refer to task T.
  for (BasicBlock *BB : UnwindPHIs) {
    for (BasicBlock *Pred : predecessors(BB)) {
      // Ignore the shared-EH spindles in T, because those might be used by
      // other subtasks of T's parent.  The shared-EH spindles tracked by T's
      // parent will be handled once all subtasks of T's parent have been
      // processed.
      if (T->simplyEncloses(Pred) && !T->encloses(BB) &&
          T->getParentTask()->encloses(BB)) {
        // Update the PHI nodes in BB.
        BB->removePredecessor(Pred);
        // Remove the edge from Pred to BB.
        IRBuilder<> B(Pred->getTerminator());
        Instruction *Unreach = B.CreateUnreachable();
        Unreach->setDebugLoc(Pred->getTerminator()->getDebugLoc());
        ToRemove.push_back(Pred->getTerminator());
      }
    }
  }

  // Remove the terminators we no longer need.
  for (Instruction *I : ToRemove)
    I->eraseFromParent();
}

/// Replace the detach that spawns T with a call to the outlined function.
Instruction *llvm::replaceDetachWithCallToOutline(Task *T,
                                                  TaskOutlineInfo &Out) {
  // Remove any dependencies from T's exception-handling code to T's parent.
  unlinkTaskEHFromParent(T);

  DetachInst *DI = T->getDetach();

  // Add call to new helper function in original function.
  if (!Out.ReplUnwind) {
    // Common case.  Insert a call to the outline immediately before the detach.
    CallInst *TopCall;
    // Create call instruction.
    IRBuilder<> Builder(Out.ReplCall);
    TopCall = Builder.CreateCall(Out.Outline, Out.OutlineInputs);
    // Use a fast calling convention for the outline.
    TopCall->setCallingConv(CallingConv::Fast);
    TopCall->setDebugLoc(DI->getDebugLoc());
    // Replace the detach with an unconditional branch to its continuation.
    ReplaceInstWithInst(DI, BranchInst::Create(Out.ReplRet));
    return TopCall;
  } else {
    // The detach might catch an exception from the task.  Replace the detach
    // with an invoke of the outline.
    InvokeInst *TopCall;
    // Create invoke instruction.  The ordinary return of the invoke is the
    // detach's continuation, and the unwind return is the detach's unwind.
    TopCall = InvokeInst::Create(Out.Outline, Out.ReplRet, Out.ReplUnwind,
                                 Out.OutlineInputs);
    // Use a fast calling convention for the outline.
    TopCall->setCallingConv(CallingConv::Fast);
    TopCall->setDebugLoc(DI->getDebugLoc());
    // Replace the detach with the invoke.
    ReplaceInstWithInst(Out.ReplCall, TopCall);
    return TopCall;
  }
}

TaskOutlineInfo llvm::outlineTask(
    Task *T, ValueSet &Inputs, ValueToValueMapTy &VMap, AssumptionCache *AC,
    DominatorTree *DT) {
  assert(!T->isRootTask() && "Cannot outline the root task.");
  Function &F = *T->getEntry()->getParent();
  DetachInst *DI = T->getDetach();

  // Convert the inputs of the task to inputs to the helper.
  ValueSet HelperArgs;
  Instruction *ArgsStart =
    fixupHelperInputs(F, T, Inputs, HelperArgs, DI,
                      T->getEntry()->getFirstNonPHIOrDbgOrLifetime());
  SmallVector<Value *, 8> HelperInputs;
  for (Value *V : HelperArgs)
    HelperInputs.push_back(V);

  // Clone the blocks into a helper function.
  Function *Helper = createHelperForTask(F, T, HelperArgs, VMap, AC, DT);
  return TaskOutlineInfo(Helper, HelperInputs, ArgsStart, DI, DI->getContinue(),
                         DI->getUnwindDest());
}

//----------------------------------------------------------------------------//
// Methods for lowering Tapir loops

/// Returns true if the value V used inside the body of Tapir loop L is defined
/// outside of L.
static bool taskInputDefinedOutsideLoop(const Value *V, const Loop *L) {
  if (isa<Argument>(V))
    return true;

  const BasicBlock *Header = L->getHeader();
  const BasicBlock *Latch = L->getLoopLatch();
  if (const Instruction *I = dyn_cast<Instruction>(V))
    if ((Header != I->getParent()) && (Latch != I->getParent()))
      return true;
  return false;
}

static bool definedOutsideBlocks(const Value *V,
                                 SmallPtrSetImpl<BasicBlock *> &Blocks) {
  if (isa<Argument>(V)) return true;
  if (const Instruction *I = dyn_cast<Instruction>(V))
    return !Blocks.count(I->getParent());
  return false;
}

ValueSet llvm::getTapirLoopInputs(TapirLoopInfo *TL, ValueSet &TaskInputs) {
  Loop *L = TL->getLoop();
  Task *T = TL->getTask();
  ValueSet LoopInputs;

  for (Value *V : TaskInputs)
    if (taskInputDefinedOutsideLoop(V, L))
      LoopInputs.insert(V);

  const Value *SyncRegion = T->getDetach()->getSyncRegion();

  SmallPtrSet<BasicBlock *, 2> BlocksToCheck;
  BlocksToCheck.insert(L->getHeader());
  BlocksToCheck.insert(L->getLoopLatch());
  for (BasicBlock *BB : BlocksToCheck) {
    for (Instruction &II : *BB) {
      // Skip the condition of this loop, since we will process that specially.
      if (TL->getCondition() == &II) continue;
      // Examine all operands of this instruction.
      for (User::op_iterator OI = II.op_begin(), OE = II.op_end(); OI != OE;
           ++OI) {
        // If the operand is the sync region of this task's detach, skip it.
        if (SyncRegion == *OI)
          continue;
        LLVM_DEBUG({
            if (Instruction *OP = dyn_cast<Instruction>(*OI))
              assert(!T->encloses(OP->getParent()) &&
                     "Loop control uses value defined in body task.");
          });
        // If this operand is not defined in the header or latch, it's an input.
        if (definedOutsideBlocks(*OI, BlocksToCheck))
          LoopInputs.insert(*OI);
      }
    }
  }

  return LoopInputs;
}

/// Replace the detach that spawns T with a call to the outlined function.
Instruction *llvm::replaceLoopWithCallToOutline(TapirLoopInfo *TL,
                                                TaskOutlineInfo &Out) {
  // Remove any dependencies from the detach unwind of T code to T's parent.
  unlinkTaskEHFromParent(TL->getTask());

  LLVM_DEBUG({
      dbgs() << "Creating call with arguments:\n";
      for (Value *V : Out.OutlineInputs)
        dbgs() << "\t" << *V << "\n";
    });

  Loop *L = TL->getLoop();
  // Add call to new helper function in original function.
  if (!Out.ReplUnwind) {
    // Common case.  Insert a call to the outline immediately before the detach.
    CallInst *TopCall;
    // Create call instruction.
    IRBuilder<> Builder(Out.ReplCall);
    TopCall = Builder.CreateCall(Out.Outline, Out.OutlineInputs);
    // Use a fast calling convention for the outline.
    TopCall->setCallingConv(CallingConv::Fast);
    TopCall->setDebugLoc(TL->getDebugLoc());
    // Replace the loop with an unconditional branch to its exit.
    L->getHeader()->removePredecessor(Out.ReplCall->getParent());
    ReplaceInstWithInst(Out.ReplCall, BranchInst::Create(Out.ReplRet));
    return TopCall;
  } else {
    // The detach might catch an exception from the task.  Replace the detach
    // with an invoke of the outline.
    InvokeInst *TopCall;

    // Create invoke instruction.  The ordinary return of the invoke is the
    // detach's continuation, and the unwind return is the detach's unwind.
    TopCall = InvokeInst::Create(Out.Outline, Out.ReplRet, Out.ReplUnwind,
                                 Out.OutlineInputs);
    // Use a fast calling convention for the outline.
    TopCall->setCallingConv(CallingConv::Fast);
    TopCall->setDebugLoc(TL->getDebugLoc());
    // Replace the loop with the invoke.
    L->getHeader()->removePredecessor(Out.ReplCall->getParent());
    ReplaceInstWithInst(Out.ReplCall, TopCall);
    // Add invoke parent as a predecessor for all Phi nodes in ReplUnwind.
    for (PHINode &Phi : Out.ReplUnwind->phis())
      Phi.addIncoming(Phi.getIncomingValueForBlock(L->getHeader()),
                      TopCall->getParent());
    return TopCall;
  }
}

//----------------------------------------------------------------------------//
// Old lowering utils

/// Extracts a detached task into a separate function.  Inserts a call or invoke
/// in place of the original detach instruction.  Returns a pointer to the
/// extracted function.  If CallSite is not null, then sets *CallSite to point
/// to the new call or invoke instruction.
Function *llvm::extractDetachBodyToFunction(
    DetachInst &Detach, DominatorTree &DT, AssumptionCache &AC,
    Instruction **CallSite) {
  BasicBlock *Detacher = Detach.getParent();
  Function &F = *(Detacher->getParent());

  BasicBlock *Detached = Detach.getDetached();
  BasicBlock *Continue = Detach.getContinue();
  BasicBlock *Unwind = nullptr;
  if (Detach.hasUnwindDest())
    Unwind = Detach.getUnwindDest();
  Value *SyncRegion = Detach.getSyncRegion();

  SmallPtrSet<BasicBlock *, 8> FunctionPieces;
  SmallPtrSet<BasicBlock *, 4> ExitBlocks;
  SmallPtrSet<BasicBlock *, 4> TaskReturns;

  assert(Detached->getUniquePredecessor() &&
         "Entry block of detached task has multiple predecessors.");
  assert(Detached->getUniquePredecessor() == Detacher &&
         "Broken CFG.");

  GetDetachedCFG(Detach, DT, FunctionPieces, ExitBlocks, TaskReturns);

  // Replace the reattach instructions terminating the task with branches.  This
  // replacement prevents the outlining process from mistakenly finding the sync
  // region as a input to the task.
  for (BasicBlock *TaskRet : TaskReturns) {
    TerminatorInst *TI = TaskRet->getTerminator();
    if (isa<ReattachInst>(TI))
      ReplaceInstWithInst(TI, BranchInst::Create(Continue));
  }

  // Check the detached task's predecessors.  Perform simple cleanup if need be.
  LLVM_DEBUG(dbgs() << "Function pieces:");
  for (BasicBlock *BB : FunctionPieces) {
    LLVM_DEBUG(dbgs() << *BB);
    if (ExitBlocks.count(BB)) continue;
    for (BasicBlock *Pred : predecessors(BB)) {
      if (Pred == Detach.getParent()) {
        // Verify the CFG structure of the entry block of the detached task.
        assert(BB == Detached &&
               "Detach instruction has multiple successors in detached task.");
        BasicBlockEdge DetachEdge(Detach.getParent(), Detached);
        assert(DT.dominates(DetachEdge, BB) &&
               "Entry of detached task reachable by non-detach edge.");
        continue;
      }

      // Remove predecessors of the detached task that are not reachable.  Such
      // blocks can arise when the inliner is run without other optimizations,
      // e.g., to handle always_inline functions.
      if (!DT.isReachableFromEntry(Pred))
        BB->removePredecessor(Pred);

      // There should be no other predecessor blocks of the detached task.
      LLVM_DEBUG({
          if (!(FunctionPieces.count(Pred) || !DT.isReachableFromEntry(Pred)))
            dbgs() << "Problem block found: " << *BB
                   << "reachable via " << *Pred;
        });
      assert((FunctionPieces.count(Pred) || !DT.isReachableFromEntry(Pred)) &&
             "Block inside of detached task is reachable from outside the task");
    }
  }

  // Get the inputs and outputs for the detached CFG.
  SetVector<Value *> BodyInputs, Outputs;
  findInputsOutputs(FunctionPieces, BodyInputs, Outputs, &ExitBlocks, &DT);
  LLVM_DEBUG({
      for (Value *V : Outputs)
        dbgs() << "EL output: " << *V << "\n";
    });
  assert(Outputs.empty() &&
         "All results from detached task should be passed by memory already.");

  // Fix up the inputs.
  SetVector<Value *> Inputs;
  SmallVector<Value *, 8> StructInputs;
  SmallVector<Type *, 8> StructIT;
  AllocaInst *Closure;
  {
    // Scan for any sret parameters in BodyInputs and add them first.
    Value *SRetInput = nullptr;
    if (F.hasStructRetAttr() && !StructTaskArgs) {
      Function::arg_iterator ArgIter = F.arg_begin();
      if (F.hasParamAttribute(0, Attribute::StructRet))
	if (BodyInputs.count(&*ArgIter))
	  SRetInput = &*ArgIter;
      if (F.hasParamAttribute(1, Attribute::StructRet)) {
	++ArgIter;
	if (BodyInputs.count(&*ArgIter))
	  SRetInput = &*ArgIter;
      }
    }
    if (SRetInput) {
      LLVM_DEBUG(dbgs() << "sret input " << *SRetInput << "\n");
      Inputs.insert(SRetInput);
      StructInputs.push_back(SRetInput);
      StructIT.push_back(SRetInput->getType());
    }
    // Add the remaining inputs.
    for (Value *V : BodyInputs) {
      if (V == SyncRegion) continue;
      if (!Inputs.count(V)) {
	Inputs.insert(V);
        StructInputs.push_back(V);
        StructIT.push_back(V->getType());
      }
    }
  }
  SetVector<Value *> HelperInputs;
  if (StructTaskArgs) {
    StructType *ST = StructType::get(F.getContext(), StructIT);
    LLVM_DEBUG(dbgs() << "Closure struct type " << *ST << "\n");
    {
      BasicBlock *AllocaInsertBlk = &F.getEntryBlock();
      IRBuilder<> Builder(&*AllocaInsertBlk->getFirstInsertionPt());
      Closure = Builder.CreateAlloca(ST);
    }
    IRBuilder<> B(Detacher->getTerminator());
    IRBuilder<> B2(Detached->getFirstNonPHIOrDbgOrLifetime());
    for (unsigned i = 0; i < StructInputs.size(); ++i) {
      B.CreateStore(StructInputs[i], B.CreateConstGEP2_32(ST, Closure, 0, i));
    }
    for (unsigned i = 0; i < StructInputs.size(); ++i) {
      auto STGEP = cast<Instruction>(B2.CreateConstGEP2_32(ST, Closure, 0, i));
      auto STLoad = B2.CreateLoad(STGEP);

      // Update all uses of the struct inputs in the loop body.
      auto UI = StructInputs[i]->use_begin(), E = StructInputs[i]->use_end();
      for (; UI != E;) {
        Use &U = *UI;
        ++UI;
        auto *Usr = dyn_cast<Instruction>(U.getUser());
        if (!Usr || !FunctionPieces.count(Usr->getParent()))
          continue;
        U.set(STLoad);
      }
    }
    HelperInputs.insert(Closure);
  } else {
    HelperInputs = Inputs;
  }

  // Clone the detached CFG into a helper function.
  ValueToValueMapTy VMap;
  Function *Extracted;
  {
    SmallVector<ReturnInst *, 4> Returns;  // Ignore returns cloned.
    std::vector<BasicBlock *> Blocks(FunctionPieces.begin(),
                                     FunctionPieces.end());

    Extracted = CreateHelper(HelperInputs, Outputs, Blocks,
                             Detached, Detacher, Continue,
                             VMap, F.getParent(),
                             F.getSubprogram() != nullptr, Returns, ".cilk",
                             &ExitBlocks, nullptr,
                             nullptr, nullptr, nullptr, nullptr);

    assert(Returns.empty() && "Returns cloned when cloning detached CFG.");

    // Use a fast calling convention for the helper.
    Extracted->setCallingConv(CallingConv::Fast);
    Extracted->addFnAttr(Attribute::NoInline);
  }

  // Add alignment assumptions to arguments of helper, based on alignment of
  // values in old function.
  AddAlignmentAssumptions(&F, HelperInputs, VMap, &Detach, &AC, &DT);

  // Add call to new helper function in original function.
  if (!Unwind) {
    CallInst *TopCall;
    // Create call instruction.
    IRBuilder<> Builder(&Detach);
    TopCall = Builder.CreateCall(Extracted, HelperInputs.getArrayRef());
    // Use a fast calling convention for the helper.
    TopCall->setCallingConv(CallingConv::Fast);
    TopCall->setDebugLoc(Detach.getDebugLoc());
    if (CallSite)
      *CallSite = TopCall;
  } else {
    InvokeInst *TopCall;
    BasicBlock *CallBlock = Detach.getParent();
    // Create the normal destination for the invoke.
    BasicBlock *CallDest = SplitBlock(CallBlock, &Detach, &DT);
    // Create invoke instruction.
    TopCall = InvokeInst::Create(Extracted, CallDest, Unwind,
                                 HelperInputs.getArrayRef());
    ReplaceInstWithInst(CallBlock->getTerminator(), TopCall);
    // Use a fast calling convention for the helper.
    TopCall->setCallingConv(CallingConv::Fast);
    TopCall->setDebugLoc(Detach.getDebugLoc());
    if (CallSite)
      *CallSite = TopCall;
  }

  // Move allocas in the newly cloned detached CFG to the entry block of the
  // helper.
  {
    // Collect the end instructions of the task.
    SmallVector<Instruction *, 4> TaskEnds;
    for (BasicBlock *TaskRetBlock : TaskReturns)
      TaskEnds.push_back(cast<BasicBlock>(VMap[TaskRetBlock])->getTerminator());
    // for (BasicBlock *Pred : predecessors(Continue)) {
    //   if (!isa<ReattachInst>(Pred->getTerminator())) continue;
    //   if (FunctionPieces.count(Pred))
    //     ReattachPoints.push_back(cast<BasicBlock>(VMap[Pred])->getTerminator());
    // }

    // Move allocas in cloned detached block to entry of helper function.
    BasicBlock *ClonedDetachedBlock = cast<BasicBlock>(VMap[Detached]);
    MoveStaticAllocasInBlock(&Extracted->getEntryBlock(), ClonedDetachedBlock,
                             TaskEnds);

    // We do not need to add new llvm.stacksave/llvm.stackrestore intrinsics,
    // because calling and returning from the helper will automatically manage
    // the stack appropriately.
  }
  return Extracted;
}

bool TapirTarget::shouldProcessFunction(const Function &F) {
  if (F.getName() == "main")
    return true;

  if (canDetach(&F))
    return true;

  for (const Instruction &I : instructions(&F))
    if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(&I))
      if (Intrinsic::tapir_loop_grainsize == II->getIntrinsicID())
        return true;

  return false;
}
