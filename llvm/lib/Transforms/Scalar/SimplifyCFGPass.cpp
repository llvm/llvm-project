//===- SimplifyCFGPass.cpp - CFG Simplification Pass ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements dead code elimination and basic block merging, along
// with a collection of other peephole control flow optimizations.  For example:
//
//   * Removes basic blocks with no predecessors.
//   * Merges a basic block into its predecessor if there is only one and the
//     predecessor only has one successor.
//   * Eliminates PHI nodes for basic blocks with a single predecessor.
//   * Eliminates a basic block that only contains an unconditional branch.
//   * Changes invoke instructions to nounwind functions to be calls.
//   * Change things like "if (x) if (y)" into "if (x&y)".
//   * etc..
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "simplifycfg"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Module.h"
#include "llvm/Attributes.h"
#include "llvm/Support/CFG.h"
#include "llvm/Pass.h"
#include "llvm/DataLayout.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/TargetTransformInfo.h"
using namespace llvm;

STATISTIC(NumSimpl, "Number of blocks simplified");

namespace {
  struct CFGSimplifyPass : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    CFGSimplifyPass() : FunctionPass(ID) {
      initializeCFGSimplifyPassPass(*PassRegistry::getPassRegistry());
    }

    virtual bool runOnFunction(Function &F);
  };
}

char CFGSimplifyPass::ID = 0;
INITIALIZE_PASS(CFGSimplifyPass, "simplifycfg",
                "Simplify the CFG", false, false)

// Public interface to the CFGSimplification pass
FunctionPass *llvm::createCFGSimplificationPass() {
  return new CFGSimplifyPass();
}

/// changeToUnreachable - Insert an unreachable instruction before the specified
/// instruction, making it and the rest of the code in the block dead.
static void changeToUnreachable(Instruction *I, bool UseLLVMTrap) {
  BasicBlock *BB = I->getParent();
  // Loop over all of the successors, removing BB's entry from any PHI
  // nodes.
  for (succ_iterator SI = succ_begin(BB), SE = succ_end(BB); SI != SE; ++SI)
    (*SI)->removePredecessor(BB);

  // Insert a call to llvm.trap right before this.  This turns the undefined
  // behavior into a hard fail instead of falling through into random code.
  if (UseLLVMTrap) {
    Function *TrapFn =
      Intrinsic::getDeclaration(BB->getParent()->getParent(), Intrinsic::trap);
    CallInst *CallTrap = CallInst::Create(TrapFn, "", I);
    CallTrap->setDebugLoc(I->getDebugLoc());
  }
  new UnreachableInst(I->getContext(), I);

  // All instructions after this are dead.
  BasicBlock::iterator BBI = I, BBE = BB->end();
  while (BBI != BBE) {
    if (!BBI->use_empty())
      BBI->replaceAllUsesWith(UndefValue::get(BBI->getType()));
    BB->getInstList().erase(BBI++);
  }
}

/// changeToCall - Convert the specified invoke into a normal call.
static void changeToCall(InvokeInst *II) {
  SmallVector<Value*, 8> Args(II->op_begin(), II->op_end() - 3);
  CallInst *NewCall = CallInst::Create(II->getCalledValue(), Args, "", II);
  NewCall->takeName(II);
  NewCall->setCallingConv(II->getCallingConv());
  NewCall->setAttributes(II->getAttributes());
  NewCall->setDebugLoc(II->getDebugLoc());
  II->replaceAllUsesWith(NewCall);

  // Follow the call by a branch to the normal destination.
  BranchInst::Create(II->getNormalDest(), II);

  // Update PHI nodes in the unwind destination
  II->getUnwindDest()->removePredecessor(II->getParent());
  II->eraseFromParent();
}

static bool markAliveBlocks(BasicBlock *BB,
                            SmallPtrSet<BasicBlock*, 128> &Reachable) {

  SmallVector<BasicBlock*, 128> Worklist;
  Worklist.push_back(BB);
  bool Changed = false;
  do {
    BB = Worklist.pop_back_val();

    if (!Reachable.insert(BB))
      continue;

    // Do a quick scan of the basic block, turning any obviously unreachable
    // instructions into LLVM unreachable insts.  The instruction combining pass
    // canonicalizes unreachable insts into stores to null or undef.
    for (BasicBlock::iterator BBI = BB->begin(), E = BB->end(); BBI != E;++BBI){
      if (CallInst *CI = dyn_cast<CallInst>(BBI)) {
        if (CI->doesNotReturn()) {
          // If we found a call to a no-return function, insert an unreachable
          // instruction after it.  Make sure there isn't *already* one there
          // though.
          ++BBI;
          if (!isa<UnreachableInst>(BBI)) {
            // Don't insert a call to llvm.trap right before the unreachable.
            changeToUnreachable(BBI, false);
            Changed = true;
          }
          break;
        }
      }

      // Store to undef and store to null are undefined and used to signal that
      // they should be changed to unreachable by passes that can't modify the
      // CFG.
      if (StoreInst *SI = dyn_cast<StoreInst>(BBI)) {
        // Don't touch volatile stores.
        if (SI->isVolatile()) continue;

        Value *Ptr = SI->getOperand(1);

        if (isa<UndefValue>(Ptr) ||
            (isa<ConstantPointerNull>(Ptr) &&
             SI->getPointerAddressSpace() == 0)) {
          changeToUnreachable(SI, true);
          Changed = true;
          break;
        }
      }
    }

    // Turn invokes that call 'nounwind' functions into ordinary calls.
    if (InvokeInst *II = dyn_cast<InvokeInst>(BB->getTerminator())) {
      Value *Callee = II->getCalledValue();
      if (isa<ConstantPointerNull>(Callee) || isa<UndefValue>(Callee)) {
        changeToUnreachable(II, true);
        Changed = true;
      } else if (II->doesNotThrow()) {
        if (II->use_empty() && II->onlyReadsMemory()) {
          // jump to the normal destination branch.
          BranchInst::Create(II->getNormalDest(), II);
          II->getUnwindDest()->removePredecessor(II->getParent());
          II->eraseFromParent();
        } else
          changeToCall(II);
        Changed = true;
      }
    }

    Changed |= ConstantFoldTerminator(BB, true);
    for (succ_iterator SI = succ_begin(BB), SE = succ_end(BB); SI != SE; ++SI)
      Worklist.push_back(*SI);
  } while (!Worklist.empty());
  return Changed;
}

/// removeUnreachableBlocksFromFn - Remove blocks that are not reachable, even
/// if they are in a dead cycle.  Return true if a change was made, false
/// otherwise.
static bool removeUnreachableBlocksFromFn(Function &F) {
  SmallPtrSet<BasicBlock*, 128> Reachable;
  bool Changed = markAliveBlocks(F.begin(), Reachable);

  // If there are unreachable blocks in the CFG...
  if (Reachable.size() == F.size())
    return Changed;

  assert(Reachable.size() < F.size());
  NumSimpl += F.size()-Reachable.size();

  // Loop over all of the basic blocks that are not reachable, dropping all of
  // their internal references...
  for (Function::iterator BB = ++F.begin(), E = F.end(); BB != E; ++BB) {
    if (Reachable.count(BB))
      continue;

    for (succ_iterator SI = succ_begin(BB), SE = succ_end(BB); SI != SE; ++SI)
      if (Reachable.count(*SI))
        (*SI)->removePredecessor(BB);
    BB->dropAllReferences();
  }

  for (Function::iterator I = ++F.begin(); I != F.end();)
    if (!Reachable.count(I))
      I = F.getBasicBlockList().erase(I);
    else
      ++I;

  return true;
}

/// mergeEmptyReturnBlocks - If we have more than one empty (other than phi
/// node) return blocks, merge them together to promote recursive block merging.
static bool mergeEmptyReturnBlocks(Function &F) {
  bool Changed = false;

  BasicBlock *RetBlock = 0;

  // Scan all the blocks in the function, looking for empty return blocks.
  for (Function::iterator BBI = F.begin(), E = F.end(); BBI != E; ) {
    BasicBlock &BB = *BBI++;

    // Only look at return blocks.
    ReturnInst *Ret = dyn_cast<ReturnInst>(BB.getTerminator());
    if (Ret == 0) continue;

    // Only look at the block if it is empty or the only other thing in it is a
    // single PHI node that is the operand to the return.
    if (Ret != &BB.front()) {
      // Check for something else in the block.
      BasicBlock::iterator I = Ret;
      --I;
      // Skip over debug info.
      while (isa<DbgInfoIntrinsic>(I) && I != BB.begin())
        --I;
      if (!isa<DbgInfoIntrinsic>(I) &&
          (!isa<PHINode>(I) || I != BB.begin() ||
           Ret->getNumOperands() == 0 ||
           Ret->getOperand(0) != I))
        continue;
    }

    // If this is the first returning block, remember it and keep going.
    if (RetBlock == 0) {
      RetBlock = &BB;
      continue;
    }

    // Otherwise, we found a duplicate return block.  Merge the two.
    Changed = true;

    // Case when there is no input to the return or when the returned values
    // agree is trivial.  Note that they can't agree if there are phis in the
    // blocks.
    if (Ret->getNumOperands() == 0 ||
        Ret->getOperand(0) ==
          cast<ReturnInst>(RetBlock->getTerminator())->getOperand(0)) {
      BB.replaceAllUsesWith(RetBlock);
      BB.eraseFromParent();
      continue;
    }

    // If the canonical return block has no PHI node, create one now.
    PHINode *RetBlockPHI = dyn_cast<PHINode>(RetBlock->begin());
    if (RetBlockPHI == 0) {
      Value *InVal = cast<ReturnInst>(RetBlock->getTerminator())->getOperand(0);
      pred_iterator PB = pred_begin(RetBlock), PE = pred_end(RetBlock);
      RetBlockPHI = PHINode::Create(Ret->getOperand(0)->getType(),
                                    std::distance(PB, PE), "merge",
                                    &RetBlock->front());

      for (pred_iterator PI = PB; PI != PE; ++PI)
        RetBlockPHI->addIncoming(InVal, *PI);
      RetBlock->getTerminator()->setOperand(0, RetBlockPHI);
    }

    // Turn BB into a block that just unconditionally branches to the return
    // block.  This handles the case when the two return blocks have a common
    // predecessor but that return different things.
    RetBlockPHI->addIncoming(Ret->getOperand(0), &BB);
    BB.getTerminator()->eraseFromParent();
    BranchInst::Create(RetBlock, &BB);
  }

  return Changed;
}

/// iterativelySimplifyCFG - Call SimplifyCFG on all the blocks in the function,
/// iterating until no more changes are made.
static bool iterativelySimplifyCFG(Function &F, const DataLayout *TD,
                                   const TargetTransformInfo *TTI) {
  bool Changed = false;
  bool LocalChange = true;
  while (LocalChange) {
    LocalChange = false;

    // Loop over all of the basic blocks and remove them if they are unneeded...
    //
    for (Function::iterator BBIt = F.begin(); BBIt != F.end(); ) {
      if (SimplifyCFG(BBIt++, TD, TTI)) {
        LocalChange = true;
        ++NumSimpl;
      }
    }
    Changed |= LocalChange;
  }
  return Changed;
}

// It is possible that we may require multiple passes over the code to fully
// simplify the CFG.
//
bool CFGSimplifyPass::runOnFunction(Function &F) {
  const DataLayout *TD = getAnalysisIfAvailable<DataLayout>();
  const TargetTransformInfo *TTI =
      getAnalysisIfAvailable<TargetTransformInfo>();
  bool EverChanged = removeUnreachableBlocksFromFn(F);
  EverChanged |= mergeEmptyReturnBlocks(F);
  EverChanged |= iterativelySimplifyCFG(F, TD, TTI);

  // If neither pass changed anything, we're done.
  if (!EverChanged) return false;

  // iterativelySimplifyCFG can (rarely) make some loops dead.  If this happens,
  // removeUnreachableBlocksFromFn is needed to nuke them, which means we should
  // iterate between the two optimizations.  We structure the code like this to
  // avoid reruning iterativelySimplifyCFG if the second pass of
  // removeUnreachableBlocksFromFn doesn't do anything.
  if (!removeUnreachableBlocksFromFn(F))
    return true;

  do {
    EverChanged = iterativelySimplifyCFG(F, TD, TTI);
    EverChanged |= removeUnreachableBlocksFromFn(F);
  } while (EverChanged);

  return true;
}
