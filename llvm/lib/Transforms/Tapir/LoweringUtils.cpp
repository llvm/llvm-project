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

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Tapir/CilkABI.h"
#include "llvm/Transforms/Tapir/CilkRABI.h"
#include "llvm/Transforms/Tapir/OpenMPABI.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

#define DEBUG_TYPE "tapir"

static cl::opt<bool> StructTaskArgs(
    "use-struct-for-task-args", cl::init(false), cl::Hidden,
    cl::desc("Use a struct to store arguments for detached tasks"));

TapirTarget *llvm::getTapirTargetFromType(TapirTargetType Type) {
  switch (Type) {
  case TapirTargetType::Cilk:
    return new CilkABI();
  case TapirTargetType::OpenMP:
    return new OpenMPABI();
  case TapirTargetType::CilkR:
    return new CilkRABI();
  case TapirTargetType::None:
  case TapirTargetType::Serial:
    return nullptr;
  default:
    llvm_unreachable("Invalid TapirTargetType");
  }
}

bool llvm::verifyDetachedCFG(const DetachInst &Detach, DominatorTree &DT,
                             bool error) {
  BasicBlock *Spawned  = Detach.getDetached();
  BasicBlock *Continue = Detach.getContinue();
  Value *SyncRegion = Detach.getSyncRegion();
  BasicBlockEdge DetachEdge(Detach.getParent(), Spawned);

  SmallVector<BasicBlock *, 32> Todo;
  SmallPtrSet<BasicBlock *, 32> FunctionPieces;
  SmallVector<BasicBlock *, 4> WorkListEH;
  Todo.push_back(Spawned);

  while (!Todo.empty()) {
    BasicBlock *BB = Todo.pop_back_val();

    if (!FunctionPieces.insert(BB).second)
      continue;

    TerminatorInst* Term = BB->getTerminator();
    if (Term == nullptr) return false;
    if (ReattachInst* Inst = dyn_cast<ReattachInst>(Term)) {
      // Only analyze reattaches going to the same continuation.
      if (Inst->getSuccessor(0) != Continue) continue;
      continue;
    } else if (DetachInst* Inst = dyn_cast<DetachInst>(Term)) {
      assert(Inst != &Detach && "Found recursive Detach!");
      Todo.push_back(Inst->getDetached());
      Todo.push_back(Inst->getContinue());
      if (Inst->hasUnwindDest())
        Todo.push_back(Inst->getUnwindDest());
      continue;
    } else if (SyncInst* Inst = dyn_cast<SyncInst>(Term)) {
      // Only sync inner elements, consider as branch
      Todo.push_back(Inst->getSuccessor(0));
      continue;
    } else if (isa<BranchInst>(Term) || isa<SwitchInst>(Term) ||
               isa<InvokeInst>(Term)) {
      if (isDetachedRethrow(Term, SyncRegion))
        continue;
      else {
        for (BasicBlock *Succ : successors(BB)) {
          if (!DT.dominates(DetachEdge, Succ))
            // We assume that this block is an exception-handling block and save
            // it for later processing.
            WorkListEH.push_back(Succ);
          else
            Todo.push_back(Succ);
        }
      }
      continue;
    } else if (isa<UnreachableInst>(Term)) {
      continue;
    } else {
      llvm_unreachable("Detached block did not absolutely terminate in reattach");
    }
  }
  {
    SmallPtrSet<BasicBlock *, 4> Visited;
    while (!WorkListEH.empty()) {
      BasicBlock *BB = WorkListEH.pop_back_val();
      if (!Visited.insert(BB).second)
        continue;

      // Make sure that the control flow through these exception-handling blocks
      // cannot re-enter the blocks being outlined.
      assert(!FunctionPieces.count(BB) &&
             "EH blocks for a detached region reenter that region.");

      // Make sure that the control flow through these exception-handling blocks
      // doesn't perform an ordinary return or resume.
      assert(!isa<ReturnInst>(BB->getTerminator()) &&
             "EH block terminated by return.");
      assert(!isa<ResumeInst>(BB->getTerminator()) &&
             "EH block terminated by resume.");

      // Make sure that the control flow through these exception-handling blocks
      // doesn't reattach to the detached CFG's continuation.
      DEBUG({
          if (ReattachInst *RI = dyn_cast<ReattachInst>(BB->getTerminator()))
            assert(RI->getSuccessor(0) != Continue &&
                   "Exit block reaches a reattach to the continuation.");
        });

      // Stop searching down this path upon finding a detached rethrow.
      if (isDetachedRethrow(BB->getTerminator(), SyncRegion))
        continue;

      for (BasicBlock *Succ : successors(BB))
        WorkListEH.push_back(Succ);
    }
  }
  return true;
}

bool llvm::populateDetachedCFG(
    const DetachInst &Detach, DominatorTree &DT,
    SmallPtrSetImpl<BasicBlock *> &FunctionPieces,
    SmallPtrSetImpl<BasicBlock *> &ExitBlocks) {
  SmallVector<BasicBlock *, 32> Todo;
  SmallVector<BasicBlock *, 4> WorkListEH;

  DEBUG(dbgs() << "Tapir: Populating CFG detached by " << Detach << "\n");

  BasicBlock *Detached = Detach.getDetached();
  BasicBlock *Continue = Detach.getContinue();
  Value *SyncRegion = Detach.getSyncRegion();
  BasicBlockEdge DetachEdge(Detach.getParent(), Detached);
  Todo.push_back(Detached);

  while (!Todo.empty()) {
    BasicBlock *BB = Todo.pop_back_val();

    if (!FunctionPieces.insert(BB).second)
      continue;

    DEBUG(dbgs() << "  Found block " << BB->getName() << "\n");

    TerminatorInst *Term = BB->getTerminator();
    if (Term == nullptr) return false;
    if (isa<ReattachInst>(Term)) {
      // Only analyze reattaches going to the same continuation.
      if (Term->getSuccessor(0) != Continue) continue;
      // Replace the reattach with a branch.  This replacement prevents the
      // outlining process from mistakenly finding the sync region as a input to
      // the task.
      ReplaceInstWithInst(Term, BranchInst::Create(Continue));
      continue;
    } else if (DetachInst* Inst = dyn_cast<DetachInst>(Term)) {
      assert(Inst != &Detach && "Found recursive Detach!");
      Todo.push_back(Inst->getDetached());
      Todo.push_back(Inst->getContinue());
      if (Inst->hasUnwindDest())
        Todo.push_back(Inst->getUnwindDest());
      continue;
    } else if (isa<SyncInst>(Term)) {
      // Only sync inner elements, consider as branch
      Todo.push_back(Term->getSuccessor(0));
      continue;
    } else if (isa<BranchInst>(Term) || isa<SwitchInst>(Term) ||
               isa<InvokeInst>(Term)) {
      if (isDetachedRethrow(Term, SyncRegion)) {
        DEBUG(dbgs() << "  Exit block " << BB->getName() << "\n");
        ExitBlocks.insert(BB);
      } else {
        for (BasicBlock *Succ : successors(BB)) {
          if (!DT.dominates(DetachEdge, Succ)) {
            // We assume that this block is an exception-handling block and save
            // it for later processing.
            DEBUG(dbgs() << "  Exit block to search " << Succ->getName() << "\n");
            ExitBlocks.insert(Succ);
            WorkListEH.push_back(Succ);
          } else {
            DEBUG(dbgs() << "Adding successor " << Succ->getName() << "\n");
            Todo.push_back(Succ);
          }
        }
      }
      // We don't bother cloning unreachable exits from the detached CFG at this
      // point.  We're cloning the entire detached CFG anyway when we outline
      // the function.
      continue;
    } else if (isa<UnreachableInst>(Term)) {
      continue;
    } else {
      llvm_unreachable("Detached block did not absolutely terminate in reattach");
    }
  }

  // Find the exit-handling blocks.
  {
    SmallPtrSet<BasicBlock *, 4> Visited;
    while (!WorkListEH.empty()) {
      BasicBlock *BB = WorkListEH.pop_back_val();
      if (!Visited.insert(BB).second)
        continue;

      // Make sure that the control flow through these exception-handling blocks
      // cannot re-enter the blocks being outlined.
      assert(!FunctionPieces.count(BB) &&
             "EH blocks for a detached task reenter that task.");

      // Make sure that the control flow through these exception-handling blocks
      // doesn't perform an ordinary return or resume.
      assert(!isa<ReturnInst>(BB->getTerminator()) &&
             "EH block terminated by return.");
      assert(!isa<ResumeInst>(BB->getTerminator()) &&
             "EH block terminated by resume.");

      // Make sure that the control flow through these exception-handling blocks
      // doesn't reattach to the detached CFG's continuation.
      DEBUG({
          if (ReattachInst *RI = dyn_cast<ReattachInst>(BB->getTerminator()))
            assert(RI->getSuccessor(0) != Continue &&
                   "Exit block reaches a reattach to the continuation.");
        });

      // Stop searching down this path upon finding a detached rethrow.
      if (isDetachedRethrow(BB->getTerminator(), SyncRegion))
        continue;

      for (BasicBlock *Succ : successors(BB)) {
        ExitBlocks.insert(Succ);
        WorkListEH.push_back(Succ);
      }
    }

    // Visited now contains exception-handling blocks that we want to clone as
    // part of outlining.
    for (BasicBlock *EHBlock : Visited)
      FunctionPieces.insert(EHBlock);
  }

  DEBUG({
      dbgs() << "Exit blocks:";
      for (BasicBlock *Exit : ExitBlocks) {
        if (DT.dominates(DetachEdge, Exit))
          dbgs() << "(dominated)";
        else
          dbgs() << "(shared)";
        dbgs() << *Exit;
      }
      dbgs() << "\n";
    });

  return true;
}

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

  assert(Detached->getUniquePredecessor() &&
         "Entry block of detached task has multiple predecessors.");
  assert(Detached->getUniquePredecessor() == Detacher &&
         "Broken CFG.");

  if (!populateDetachedCFG(Detach, DT, FunctionPieces, ExitBlocks))
    return nullptr;

  // Check the detached task's predecessors.  Perform simple cleanup if need be.
  DEBUG(dbgs() << "Function pieces:");
  for (BasicBlock *BB : FunctionPieces) {
    DEBUG(dbgs() << *BB);
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
      DEBUG({
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
  DEBUG({
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
      DEBUG(dbgs() << "sret input " << *SRetInput << "\n");
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
    DEBUG(dbgs() << "Closure struct type " << *ST << "\n");
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
        // if (StructInputs[i] == LimitVar && Usr == NewCond)
        //   continue;
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
                             &ExitBlocks, nullptr, nullptr, nullptr, nullptr,
                             nullptr);

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
    // Collect reattach instructions.
    SmallVector<Instruction *, 4> ReattachPoints;
    for (BasicBlock *Pred : predecessors(Continue)) {
      if (!isa<ReattachInst>(Pred->getTerminator())) continue;
      if (FunctionPieces.count(Pred))
        ReattachPoints.push_back(cast<BasicBlock>(VMap[Pred])->getTerminator());
    }

    // Move allocas in cloned detached block to entry of helper function.
    BasicBlock *ClonedDetachedBlock = cast<BasicBlock>(VMap[Detached]);
    MoveStaticAllocasInBlock(&Extracted->getEntryBlock(), ClonedDetachedBlock,
                             ReattachPoints);

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
