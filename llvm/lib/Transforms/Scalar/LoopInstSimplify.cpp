//===- LoopInstSimplify.cpp - Loop Instruction Simplification Pass --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass performs lightweight instruction simplification on loop bodies.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "loop-instsimplify"
#include "llvm/Instructions.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Support/Debug.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumSimplified, "Number of redundant instructions simplified");

namespace {
  class LoopInstSimplify : public LoopPass {
  public:
    static char ID; // Pass ID, replacement for typeid
    LoopInstSimplify() : LoopPass(ID) {
      initializeLoopInstSimplifyPass(*PassRegistry::getPassRegistry());
    }

    bool runOnLoop(Loop*, LPPassManager&);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<LoopInfo>();
      AU.addRequiredID(LoopSimplifyID);
      AU.addPreservedID(LoopSimplifyID);
      AU.addPreservedID(LCSSAID);
      AU.addPreserved("scalar-evolution");
      AU.addRequired<TargetLibraryInfo>();
    }
  };
}

char LoopInstSimplify::ID = 0;
INITIALIZE_PASS_BEGIN(LoopInstSimplify, "loop-instsimplify",
                "Simplify instructions in loops", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfo)
INITIALIZE_PASS_DEPENDENCY(DominatorTree)
INITIALIZE_PASS_DEPENDENCY(LoopInfo)
INITIALIZE_PASS_DEPENDENCY(LCSSA)
INITIALIZE_PASS_END(LoopInstSimplify, "loop-instsimplify",
                "Simplify instructions in loops", false, false)

Pass *llvm::createLoopInstSimplifyPass() {
  return new LoopInstSimplify();
}

bool LoopInstSimplify::runOnLoop(Loop *L, LPPassManager &LPM) {
  DominatorTree *DT = getAnalysisIfAvailable<DominatorTree>();
  LoopInfo *LI = &getAnalysis<LoopInfo>();
  const DataLayout *TD = getAnalysisIfAvailable<DataLayout>();
  const TargetLibraryInfo *TLI = &getAnalysis<TargetLibraryInfo>();

  SmallVector<BasicBlock*, 8> ExitBlocks;
  L->getUniqueExitBlocks(ExitBlocks);
  array_pod_sort(ExitBlocks.begin(), ExitBlocks.end());

  SmallPtrSet<const Instruction*, 8> S1, S2, *ToSimplify = &S1, *Next = &S2;

  // The bit we are stealing from the pointer represents whether this basic
  // block is the header of a subloop, in which case we only process its phis.
  typedef PointerIntPair<BasicBlock*, 1> WorklistItem;
  SmallVector<WorklistItem, 16> VisitStack;
  SmallPtrSet<BasicBlock*, 32> Visited;

  bool Changed = false;
  bool LocalChanged;
  do {
    LocalChanged = false;

    VisitStack.clear();
    Visited.clear();

    VisitStack.push_back(WorklistItem(L->getHeader(), false));

    while (!VisitStack.empty()) {
      WorklistItem Item = VisitStack.pop_back_val();
      BasicBlock *BB = Item.getPointer();
      bool IsSubloopHeader = Item.getInt();

      // Simplify instructions in the current basic block.
      for (BasicBlock::iterator BI = BB->begin(), BE = BB->end(); BI != BE;) {
        Instruction *I = BI++;

        // The first time through the loop ToSimplify is empty and we try to
        // simplify all instructions. On later iterations ToSimplify is not
        // empty and we only bother simplifying instructions that are in it.
        if (!ToSimplify->empty() && !ToSimplify->count(I))
          continue;

        // Don't bother simplifying unused instructions.
        if (!I->use_empty()) {
          Value *V = SimplifyInstruction(I, TD, TLI, DT);
          if (V && LI->replacementPreservesLCSSAForm(I, V)) {
            // Mark all uses for resimplification next time round the loop.
            for (Value::use_iterator UI = I->use_begin(), UE = I->use_end();
                 UI != UE; ++UI)
              Next->insert(cast<Instruction>(*UI));

            I->replaceAllUsesWith(V);
            LocalChanged = true;
            ++NumSimplified;
          }
        }
        LocalChanged |= RecursivelyDeleteTriviallyDeadInstructions(I, TLI);

        if (IsSubloopHeader && !isa<PHINode>(I))
          break;
      }

      // Add all successors to the worklist, except for loop exit blocks and the
      // bodies of subloops. We visit the headers of loops so that we can process
      // their phis, but we contract the rest of the subloop body and only follow
      // edges leading back to the original loop.
      for (succ_iterator SI = succ_begin(BB), SE = succ_end(BB); SI != SE;
           ++SI) {
        BasicBlock *SuccBB = *SI;
        if (!Visited.insert(SuccBB))
          continue;

        const Loop *SuccLoop = LI->getLoopFor(SuccBB);
        if (SuccLoop && SuccLoop->getHeader() == SuccBB
                     && L->contains(SuccLoop)) {
          VisitStack.push_back(WorklistItem(SuccBB, true));

          SmallVector<BasicBlock*, 8> SubLoopExitBlocks;
          SuccLoop->getExitBlocks(SubLoopExitBlocks);

          for (unsigned i = 0; i < SubLoopExitBlocks.size(); ++i) {
            BasicBlock *ExitBB = SubLoopExitBlocks[i];
            if (LI->getLoopFor(ExitBB) == L && Visited.insert(ExitBB))
              VisitStack.push_back(WorklistItem(ExitBB, false));
          }

          continue;
        }

        bool IsExitBlock = std::binary_search(ExitBlocks.begin(),
                                              ExitBlocks.end(), SuccBB);
        if (IsExitBlock)
          continue;

        VisitStack.push_back(WorklistItem(SuccBB, false));
      }
    }

    // Place the list of instructions to simplify on the next loop iteration
    // into ToSimplify.
    std::swap(ToSimplify, Next);
    Next->clear();

    Changed |= LocalChanged;
  } while (LocalChanged);

  return Changed;
}
