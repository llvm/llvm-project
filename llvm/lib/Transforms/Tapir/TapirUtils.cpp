//===- TapirUtils.cpp - Utility functions for handling Tapir --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements several utility functions for operating with Tapir.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Transforms/Tapir/CilkABI.h"
#include "llvm/Transforms/Tapir/OpenMPABI.h"
#include "llvm/Transforms/Tapir/CilkRABI.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/EscapeEnumerator.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

#define DEBUG_TYPE "tapir"

TapirTarget *llvm::getTapirTargetFromType(TapirTargetType Type) {
  switch(Type) {
  case TapirTargetType::Cilk:
    return new CilkABI();
  case TapirTargetType::OpenMP:
    return new OpenMPABI();
  case TapirTargetType::CilkR:
    return new CilkRABI();
  case TapirTargetType::None:
  case TapirTargetType::Serial:
    return nullptr;
  }

  llvm_unreachable("Invalid TapirTargetType");
}

bool llvm::verifyDetachedCFG(const DetachInst &Detach, DominatorTree &DT,
                             bool error) {
  BasicBlock *Spawned  = Detach.getDetached();
  BasicBlock *Continue = Detach.getContinue();
  BasicBlockEdge DetachEdge(Detach.getParent(), Spawned);

  SmallVector<BasicBlock *, 32> Todo;
  SmallPtrSet<BasicBlock *, 32> functionPieces;
  SmallVector<BasicBlock *, 4> WorkListEH;
  Todo.push_back(Spawned);

  while (!Todo.empty()) {
    BasicBlock *BB = Todo.pop_back_val();

    if (!functionPieces.insert(BB).second)
      continue;

    TerminatorInst* Term = BB->getTerminator();
    if (Term == nullptr) return false;
    if (ReattachInst* Inst = dyn_cast<ReattachInst>(Term)) {
      //only analyze reattaches going to the same continuation
      if (Inst->getSuccessor(0) != Continue) continue;
      continue;
    } else if (DetachInst* Inst = dyn_cast<DetachInst>(Term)) {
      assert(Inst != &Detach && "Found recursive Detach!");
      Todo.push_back(Inst->getSuccessor(0));
      Todo.push_back(Inst->getSuccessor(1));
      continue;
    } else if (SyncInst* Inst = dyn_cast<SyncInst>(Term)) {
      //only sync inner elements, consider as branch
      Todo.push_back(Inst->getSuccessor(0));
      continue;
    } else if (isa<BranchInst>(Term) || isa<SwitchInst>(Term) ||
               isa<InvokeInst>(Term)) {
      for (BasicBlock *Succ : successors(BB)) {
        if (!DT.dominates(DetachEdge, Succ))
          // We assume that this block is an exception-handling block and save
          // it for later processing.
          WorkListEH.push_back(Succ);
        else
          Todo.push_back(Succ);
      }
      continue;
    } else if (isa<UnreachableInst>(Term) || isa<ResumeInst>(Term)) {
      continue;
    } else {
      DEBUG(Term->dump());
      DEBUG(Term->getParent()->getParent()->dump());
      assert(!error && "Detached block did not absolutely terminate in reattach");
      return false;
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
      assert(!functionPieces.count(BB) &&
             "EH blocks for a detached region reenter that region.");

      // Make sure that the control flow through these exception-handling blocks
      // doesn't perform an ordinary return.
      assert(!isa<ReturnInst>(BB->getTerminator()) &&
             "EH block terminated by return.");

      // Make sure that the control flow through these exception-handling blocks
      // doesn't reattach to the detached CFG's continuation.
      if (ReattachInst *RI = dyn_cast<ReattachInst>(BB->getTerminator()))
        assert(RI->getSuccessor(0) != Continue &&
               "Exit block reaches a reattach to the continuation.");

      for (BasicBlock *Succ : successors(BB))
        WorkListEH.push_back(Succ);
    }
  }
  return true;
}

bool llvm::populateDetachedCFG(
    const DetachInst &Detach, DominatorTree &DT,
    SmallPtrSetImpl<BasicBlock *> &functionPieces,
    SmallVectorImpl<BasicBlock *> &reattachB,
    SmallPtrSetImpl<BasicBlock *> &ExitBlocks,
    int replaceOrDelete, bool error) {
  SmallVector<BasicBlock *, 32> Todo;
  SmallVector<BasicBlock *, 4> WorkListEH;

  BasicBlock *Spawned  = Detach.getDetached();
  BasicBlock *Continue = Detach.getContinue();
  BasicBlockEdge DetachEdge(Detach.getParent(), Spawned);
  Todo.push_back(Spawned);

  while (!Todo.empty()) {
    BasicBlock *BB = Todo.pop_back_val();

    if (!functionPieces.insert(BB).second)
      continue;

    TerminatorInst *Term = BB->getTerminator();
    if (Term == nullptr) return false;
    if (isa<ReattachInst>(Term)) {
      // only analyze reattaches going to the same continuation
      if (Term->getSuccessor(0) != Continue) continue;
      if (replaceOrDelete == 1) {
        BranchInst* toReplace = BranchInst::Create(Continue);
        ReplaceInstWithInst(Term, toReplace);
        reattachB.push_back(BB);
      } else if (replaceOrDelete == 2) {
          BasicBlock::iterator BI = Continue->begin();
          while (PHINode *P = dyn_cast<PHINode>(BI)) {
            P->removeIncomingValue(Term->getParent());
            ++BI;
          }
          Term->eraseFromParent();
      }
      continue;
    } else if (isa<DetachInst>(Term)) {
      assert(Term != &Detach && "Found recursive detach!");
      Todo.push_back(Term->getSuccessor(0));
      Todo.push_back(Term->getSuccessor(1));
      continue;
    } else if (isa<SyncInst>(Term)) {
      //only sync inner elements, consider as branch
      Todo.push_back(Term->getSuccessor(0));
      continue;
    } else if (isa<BranchInst>(Term) || isa<SwitchInst>(Term) ||
               isa<InvokeInst>(Term)) {
      for (BasicBlock *Succ : successors(BB)) {
        if (!DT.dominates(DetachEdge, Succ)) {
          // We assume that this block is an exception-handling block and save
          // it for later processing.
          ExitBlocks.insert(Succ);
          WorkListEH.push_back(Succ);
        } else {
          Todo.push_back(Succ);
        }
      }
      // We don't bother cloning unreachable exits from the detached CFG at this
      // point.  We're cloning the entire detached CFG anyway when we outline
      // the function.
      continue;
    } else if (isa<UnreachableInst>(Term) || isa<ResumeInst>(Term)) {
      continue;
    } else {
      DEBUG(Term->dump());
      DEBUG(Term->getParent()->getParent()->dump());
      assert(!error && "Detached block did not absolutely terminate in reattach");
      return false;
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
      assert(!functionPieces.count(BB) &&
             "EH blocks for a detached region reenter that region.");

      // Make sure that the control flow through these exception-handling blocks
      // doesn't perform an ordinary return.
      assert(!isa<ReturnInst>(BB->getTerminator()) &&
             "EH block terminated by return.");

      // Make sure that the control flow through these exception-handling blocks
      // doesn't reattach to the detached CFG's continuation.
      if (ReattachInst *RI = dyn_cast<ReattachInst>(BB->getTerminator()))
        assert(RI->getSuccessor(0) != Continue &&
               "Exit block reaches a reattach to the continuation.");

      // if (isa<ResumeInst>(BB-getTerminator()))
      //   ResumeBlocks.push_back(BB);

      for (BasicBlock *Succ : successors(BB)) {
        ExitBlocks.insert(Succ);
        WorkListEH.push_back(Succ);
      }
    }

    // Visited now contains exception-handling blocks that we want to clone as
    // part of outlining.
    for (BasicBlock *EHBlock : Visited)
      functionPieces.insert(EHBlock);
  }

  return true;
}

//Returns true if success
Function *llvm::extractDetachBodyToFunction(DetachInst &detach,
                                            DominatorTree &DT,
                                            AssumptionCache &AC,
                                            CallInst **call) {
  BasicBlock *Detacher = detach.getParent();
  Function &F = *(Detacher->getParent());

  BasicBlock *Spawned  = detach.getDetached();
  BasicBlock *Continue = detach.getContinue();

  SmallPtrSet<BasicBlock *, 32> functionPieces;
  SmallVector<BasicBlock *, 32> reattachB;
  SmallPtrSet<BasicBlock *, 4> ExitBlocks;

  assert(Spawned->getUniquePredecessor() &&
         "Entry block of detached CFG has multiple predecessors.");
  assert(Spawned->getUniquePredecessor() == Detacher &&
         "Broken CFG.");

  if (!populateDetachedCFG(detach, DT, functionPieces, reattachB,
                           ExitBlocks, /*change to branch reattach*/1))
    return nullptr;

  // Check the spawned block's predecessors.
  for (BasicBlock *BB : functionPieces) {
    int detached_count = 0;
    if (ExitBlocks.count(BB))
      continue;
    for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI) {
      BasicBlock *Pred = *PI;
      if (detached_count == 0 && BB == Spawned && Pred == detach.getParent()) {
        detached_count = 1;
        continue;
      }
      assert(functionPieces.count(Pred) &&
             "Block inside of detached context branched into from outside branch context");
    }
  }

  // Get the inputs and outputs for the detached CFG.
  SetVector<Value *> Inputs, Outputs;
  findInputsOutputs(functionPieces, Inputs, Outputs, &ExitBlocks);
  assert(Outputs.empty() &&
         "All results from detached CFG should be passed by memory already.");

  // Clone the detached CFG into a helper function.
  ValueToValueMapTy VMap;
  Function *extracted;
  {
    SmallVector<ReturnInst *, 4> Returns;  // Ignore returns cloned.
    std::vector<BasicBlock *> blocks(functionPieces.begin(), functionPieces.end());

    extracted = CreateHelper(Inputs, Outputs, blocks,
                             Spawned, Detacher, Continue,
                             VMap, F.getParent(),
                             F.getSubprogram() != nullptr, Returns, ".cilk",
                             &ExitBlocks, nullptr, nullptr, nullptr, nullptr);

    assert(Returns.empty() && "Returns cloned when cloning detached CFG.");

    // Use a fast calling convention for the helper.
    extracted->setCallingConv(CallingConv::Fast);
    extracted->addFnAttr(Attribute::NoInline);
  }

  // Add alignment assumptions to arguments of helper, based on alignment of
  // values in old function.
  AddAlignmentAssumptions(&F, Inputs, VMap, &detach, &AC, &DT);

  // Add call to new helper function in original function.
  CallInst *TopCall;
  {
    // Create call instruction.
    IRBuilder<> Builder(&detach);
    TopCall = Builder.CreateCall(extracted, Inputs.getArrayRef());
    // Use a fast calling convention for the helper.
    TopCall->setCallingConv(CallingConv::Fast);
    TopCall->setDebugLoc(detach.getDebugLoc());
  }
  if (call)
    *call = TopCall;

  // Move allocas in the newly cloned detached CFG to the entry block of the
  // helper.
  {
    // Collect reattach instructions.
    SmallVector<Instruction *, 4> ReattachPoints;
    for (pred_iterator PI = pred_begin(Continue), PE = pred_end(Continue);
         PI != PE; ++PI) {
      BasicBlock *Pred = *PI;
      if (!isa<ReattachInst>(Pred->getTerminator())) continue;
      if (functionPieces.count(Pred))
        ReattachPoints.push_back(cast<BasicBlock>(VMap[Pred])->getTerminator());
    }

    // Move allocas in cloned detached block to entry of helper function.
    BasicBlock *ClonedDetachedBlock = cast<BasicBlock>(VMap[Spawned]);
    MoveStaticAllocasInBlock(&extracted->getEntryBlock(), ClonedDetachedBlock,
                             ReattachPoints);

    // We should not need to add new llvm.stacksave/llvm.stackrestore
    // intrinsics, because calling and returning from the helper will
    // automatically manage the stack.
  }

  for(BasicBlock* BB : reattachB) {
    auto term = BB->getTerminator();
    BasicBlock::iterator BI = term->getSuccessor(0)->begin();
    while (PHINode *P = dyn_cast<PHINode>(BI)) {
      P->removeIncomingValue(BB);
      ++BI;
    }
    IRBuilder<> b(term);
    b.CreateUnreachable();
    term->eraseFromParent();
  }
  return extracted;
}
