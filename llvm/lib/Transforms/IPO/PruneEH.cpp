//===- PruneEH.cpp - Pass which deletes unused exception handlers ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple interprocedural pass which walks the
// call-graph, turning invoke instructions into calls, iff the callee cannot
// throw an exception, and marking functions 'nounwind' if they cannot throw.
// It implements this as a bottom-up traversal of the call-graph.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "prune-eh"
#include "llvm/Transforms/IPO.h"
#include "llvm/CallGraphSCCPass.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/LLVMContext.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CFG.h"
#include <algorithm>
using namespace llvm;

STATISTIC(NumRemoved, "Number of invokes removed");
STATISTIC(NumUnreach, "Number of noreturn calls optimized");

namespace {
  struct PruneEH : public CallGraphSCCPass {
    static char ID; // Pass identification, replacement for typeid
    PruneEH() : CallGraphSCCPass(ID) {
      initializePruneEHPass(*PassRegistry::getPassRegistry());
    }

    // runOnSCC - Analyze the SCC, performing the transformation if possible.
    bool runOnSCC(CallGraphSCC &SCC);

    bool SimplifyFunction(Function *F);
    void DeleteBasicBlock(BasicBlock *BB);
  };
}

char PruneEH::ID = 0;
INITIALIZE_PASS_BEGIN(PruneEH, "prune-eh",
                "Remove unused exception handling info", false, false)
INITIALIZE_AG_DEPENDENCY(CallGraph)
INITIALIZE_PASS_END(PruneEH, "prune-eh",
                "Remove unused exception handling info", false, false)

Pass *llvm::createPruneEHPass() { return new PruneEH(); }


bool PruneEH::runOnSCC(CallGraphSCC &SCC) {
  SmallPtrSet<CallGraphNode *, 8> SCCNodes;
  CallGraph &CG = getAnalysis<CallGraph>();
  bool MadeChange = false;

  // Fill SCCNodes with the elements of the SCC.  Used for quickly
  // looking up whether a given CallGraphNode is in this SCC.
  for (CallGraphSCC::iterator I = SCC.begin(), E = SCC.end(); I != E; ++I)
    SCCNodes.insert(*I);

  // First pass, scan all of the functions in the SCC, simplifying them
  // according to what we know.
  for (CallGraphSCC::iterator I = SCC.begin(), E = SCC.end(); I != E; ++I)
    if (Function *F = (*I)->getFunction())
      MadeChange |= SimplifyFunction(F);

  // Next, check to see if any callees might throw or if there are any external
  // functions in this SCC: if so, we cannot prune any functions in this SCC.
  // Definitions that are weak and not declared non-throwing might be 
  // overridden at linktime with something that throws, so assume that.
  // If this SCC includes the unwind instruction, we KNOW it throws, so
  // obviously the SCC might throw.
  //
  bool SCCMightUnwind = false, SCCMightReturn = false;
  for (CallGraphSCC::iterator I = SCC.begin(), E = SCC.end(); 
       (!SCCMightUnwind || !SCCMightReturn) && I != E; ++I) {
    Function *F = (*I)->getFunction();
    if (F == 0) {
      SCCMightUnwind = true;
      SCCMightReturn = true;
    } else if (F->isDeclaration() || F->mayBeOverridden()) {
      SCCMightUnwind |= !F->doesNotThrow();
      SCCMightReturn |= !F->doesNotReturn();
    } else {
      bool CheckUnwind = !SCCMightUnwind && !F->doesNotThrow();
      bool CheckReturn = !SCCMightReturn && !F->doesNotReturn();

      if (!CheckUnwind && !CheckReturn)
        continue;

      // Check to see if this function performs an unwind or calls an
      // unwinding function.
      for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB) {
        if (CheckUnwind && isa<ResumeInst>(BB->getTerminator())) {
          // Uses unwind / resume!
          SCCMightUnwind = true;
        } else if (CheckReturn && isa<ReturnInst>(BB->getTerminator())) {
          SCCMightReturn = true;
        }

        // Invoke instructions don't allow unwinding to continue, so we are
        // only interested in call instructions.
        if (CheckUnwind && !SCCMightUnwind)
          for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
            if (CallInst *CI = dyn_cast<CallInst>(I)) {
              if (CI->doesNotThrow()) {
                // This call cannot throw.
              } else if (Function *Callee = CI->getCalledFunction()) {
                CallGraphNode *CalleeNode = CG[Callee];
                // If the callee is outside our current SCC then we may
                // throw because it might.
                if (!SCCNodes.count(CalleeNode)) {
                  SCCMightUnwind = true;
                  break;
                }
              } else {
                // Indirect call, it might throw.
                SCCMightUnwind = true;
                break;
              }
            }
        if (SCCMightUnwind && SCCMightReturn) break;
      }
    }
  }

  // If the SCC doesn't unwind or doesn't throw, note this fact.
  if (!SCCMightUnwind || !SCCMightReturn)
    for (CallGraphSCC::iterator I = SCC.begin(), E = SCC.end(); I != E; ++I) {
      AttrBuilder NewAttributes;

      if (!SCCMightUnwind)
        NewAttributes.addAttribute(Attributes::NoUnwind);
      if (!SCCMightReturn)
        NewAttributes.addAttribute(Attributes::NoReturn);

      Function *F = (*I)->getFunction();
      const AttrListPtr &PAL = F->getAttributes();
      const AttrListPtr &NPAL = PAL.addAttr(F->getContext(), ~0,
                                            Attributes::get(F->getContext(),
                                                            NewAttributes));
      if (PAL != NPAL) {
        MadeChange = true;
        F->setAttributes(NPAL);
      }
    }

  for (CallGraphSCC::iterator I = SCC.begin(), E = SCC.end(); I != E; ++I) {
    // Convert any invoke instructions to non-throwing functions in this node
    // into call instructions with a branch.  This makes the exception blocks
    // dead.
    if (Function *F = (*I)->getFunction())
      MadeChange |= SimplifyFunction(F);
  }

  return MadeChange;
}


// SimplifyFunction - Given information about callees, simplify the specified
// function if we have invokes to non-unwinding functions or code after calls to
// no-return functions.
bool PruneEH::SimplifyFunction(Function *F) {
  bool MadeChange = false;
  for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB) {
    if (InvokeInst *II = dyn_cast<InvokeInst>(BB->getTerminator()))
      if (II->doesNotThrow()) {
        SmallVector<Value*, 8> Args(II->op_begin(), II->op_end() - 3);
        // Insert a call instruction before the invoke.
        CallInst *Call = CallInst::Create(II->getCalledValue(), Args, "", II);
        Call->takeName(II);
        Call->setCallingConv(II->getCallingConv());
        Call->setAttributes(II->getAttributes());
        Call->setDebugLoc(II->getDebugLoc());

        // Anything that used the value produced by the invoke instruction
        // now uses the value produced by the call instruction.  Note that we
        // do this even for void functions and calls with no uses so that the
        // callgraph edge is updated.
        II->replaceAllUsesWith(Call);
        BasicBlock *UnwindBlock = II->getUnwindDest();
        UnwindBlock->removePredecessor(II->getParent());

        // Insert a branch to the normal destination right before the
        // invoke.
        BranchInst::Create(II->getNormalDest(), II);

        // Finally, delete the invoke instruction!
        BB->getInstList().pop_back();

        // If the unwind block is now dead, nuke it.
        if (pred_begin(UnwindBlock) == pred_end(UnwindBlock))
          DeleteBasicBlock(UnwindBlock);  // Delete the new BB.

        ++NumRemoved;
        MadeChange = true;
      }

    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; )
      if (CallInst *CI = dyn_cast<CallInst>(I++))
        if (CI->doesNotReturn() && !isa<UnreachableInst>(I)) {
          // This call calls a function that cannot return.  Insert an
          // unreachable instruction after it and simplify the code.  Do this
          // by splitting the BB, adding the unreachable, then deleting the
          // new BB.
          BasicBlock *New = BB->splitBasicBlock(I);

          // Remove the uncond branch and add an unreachable.
          BB->getInstList().pop_back();
          new UnreachableInst(BB->getContext(), BB);

          DeleteBasicBlock(New);  // Delete the new BB.
          MadeChange = true;
          ++NumUnreach;
          break;
        }
  }

  return MadeChange;
}

/// DeleteBasicBlock - remove the specified basic block from the program,
/// updating the callgraph to reflect any now-obsolete edges due to calls that
/// exist in the BB.
void PruneEH::DeleteBasicBlock(BasicBlock *BB) {
  assert(pred_begin(BB) == pred_end(BB) && "BB is not dead!");
  CallGraph &CG = getAnalysis<CallGraph>();

  CallGraphNode *CGN = CG[BB->getParent()];
  for (BasicBlock::iterator I = BB->end(), E = BB->begin(); I != E; ) {
    --I;
    if (CallInst *CI = dyn_cast<CallInst>(I)) {
      if (!isa<IntrinsicInst>(I))
        CGN->removeCallEdgeFor(CI);
    } else if (InvokeInst *II = dyn_cast<InvokeInst>(I))
      CGN->removeCallEdgeFor(II);
    if (!I->use_empty())
      I->replaceAllUsesWith(UndefValue::get(I->getType()));
  }

  // Get the list of successors of this block.
  std::vector<BasicBlock*> Succs(succ_begin(BB), succ_end(BB));

  for (unsigned i = 0, e = Succs.size(); i != e; ++i)
    Succs[i]->removePredecessor(BB);

  BB->eraseFromParent();
}
