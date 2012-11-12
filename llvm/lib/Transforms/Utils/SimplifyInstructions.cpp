//===------ SimplifyInstructions.cpp - Remove redundant instructions ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a utility pass used for testing the InstructionSimplify analysis.
// The analysis is applied to every instruction, and if it simplifies then the
// instruction is replaced by the simplification.  If you are looking for a pass
// that performs serious instruction folding, use the instcombine pass instead.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "instsimplify"
#include "llvm/Function.h"
#include "llvm/Pass.h"
#include "llvm/Type.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
using namespace llvm;

STATISTIC(NumSimplified, "Number of redundant instructions removed");

namespace {
  struct InstSimplifier : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    InstSimplifier() : FunctionPass(ID) {
      initializeInstSimplifierPass(*PassRegistry::getPassRegistry());
    }

    void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<TargetLibraryInfo>();
    }

    /// runOnFunction - Remove instructions that simplify.
    bool runOnFunction(Function &F) {
      const DominatorTree *DT = getAnalysisIfAvailable<DominatorTree>();
      const DataLayout *TD = getAnalysisIfAvailable<DataLayout>();
      const TargetLibraryInfo *TLI = &getAnalysis<TargetLibraryInfo>();
      SmallPtrSet<const Instruction*, 8> S1, S2, *ToSimplify = &S1, *Next = &S2;
      bool Changed = false;

      do {
        for (df_iterator<BasicBlock*> DI = df_begin(&F.getEntryBlock()),
             DE = df_end(&F.getEntryBlock()); DI != DE; ++DI)
          for (BasicBlock::iterator BI = DI->begin(), BE = DI->end(); BI != BE;) {
            Instruction *I = BI++;
            // The first time through the loop ToSimplify is empty and we try to
            // simplify all instructions.  On later iterations ToSimplify is not
            // empty and we only bother simplifying instructions that are in it.
            if (!ToSimplify->empty() && !ToSimplify->count(I))
              continue;
            // Don't waste time simplifying unused instructions.
            if (!I->use_empty())
              if (Value *V = SimplifyInstruction(I, TD, TLI, DT)) {
                // Mark all uses for resimplification next time round the loop.
                for (Value::use_iterator UI = I->use_begin(), UE = I->use_end();
                     UI != UE; ++UI)
                  Next->insert(cast<Instruction>(*UI));
                I->replaceAllUsesWith(V);
                ++NumSimplified;
                Changed = true;
              }
            Changed |= RecursivelyDeleteTriviallyDeadInstructions(I, TLI);
          }

        // Place the list of instructions to simplify on the next loop iteration
        // into ToSimplify.
        std::swap(ToSimplify, Next);
        Next->clear();
      } while (!ToSimplify->empty());

      return Changed;
    }
  };
}

char InstSimplifier::ID = 0;
INITIALIZE_PASS_BEGIN(InstSimplifier, "instsimplify",
                      "Remove redundant instructions", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfo)
INITIALIZE_PASS_END(InstSimplifier, "instsimplify",
                    "Remove redundant instructions", false, false)
char &llvm::InstructionSimplifierID = InstSimplifier::ID;

// Public interface to the simplify instructions pass.
FunctionPass *llvm::createInstructionSimplifierPass() {
  return new InstSimplifier();
}
