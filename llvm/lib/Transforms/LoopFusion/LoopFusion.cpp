/*===- LoopFusion.cpp -
  This program is the implementation of a pass for loop fusion in LLVM compiler.
Two loops, which are adjacent and have the same condition and increments with
respect to the loop variable may be fused, i.e, their bodies may be executed one
after the other with in a single loop. The decision to fuse the loops is taken
based on the legality and profitability of the fusion. It should not be
performed if the resulting code has anti-dependency or if the execution time of
the program increases. Algorithm:
1. Check 2 loops are can fuse.
2. Replace the use of induction variable of 2nd loop with that of 1st loop.
3. Combine the bodies of loop1 and loop2.
3. Set the succesor of 1st loop’s header to exit block of 2nd loop.
4. Delete the unwanted basic blocks of 2nd loop.
===-------------------------------------------------------------------------------------------===*/

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <algorithm>
using namespace llvm;

#define DEBUG_TYPE "hello"

namespace {
// Scev - The second implementation with getAnalysisUsage implemented.
struct LoopFusion : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  LoopFusion() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {

    SmallVector<Loop *> LoopVector;
    LoopInfo *LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();

    for (auto *L : *LI) {
      LoopVector.push_back(L);
    }

    int LoopCount = LoopVector.size();
    if (LoopCount < 2) {
      llvm::errs() << "The program contains less no of loops to fuse\n";
      return false;
    }

    // Check for each combinations of loops are fusable
    for (int i = 0; i < LoopCount; i++) {
      for (int j = i + 1; j < LoopCount; j++) {
        // Function to perform basic checks on the two loops
        if (fuseCheck(LoopVector[j], LoopVector[i]))
          // Function to perform fusing on the two loops
          fuseBody(LoopVector[j], LoopVector[i], F);
        break;
      }
    }

    return false;
  }

  void fuseBody(Loop *Loop1, Loop *Loop2, Function &F) {
    BasicBlock *Header1 = nullptr;
    BasicBlock *Latch1 = nullptr;
    BasicBlock *Exit2 = nullptr;

    BasicBlock *Body1 = getBody(Loop1);
    BasicBlock *Body2 = getBody(Loop2);
    Header1 = Loop1->getHeader();
    Latch1 = Loop1->getLoopLatch();
    Exit2 = Loop2->getExitBlock();

    assert(Body1 && Body2 && Header1 && Latch1 && Exit2 &&
           "NULL Pointer encountered\n");

    PHINode *Phi1 = Loop1->getCanonicalInductionVariable();
    PHINode *Phi2 = Loop2->getCanonicalInductionVariable();

    // Replace the use of induction variable of 2nd loop with that of 1st loop.
    Phi2->replaceAllUsesWith(Phi1);

    for (BasicBlock &BB : F) {

      if (isa<ReturnInst>(BB.getTerminator()))
        continue;
      // Get the branch Instruction every block.
      BranchInst *BI = dyn_cast<BranchInst>(BB.getTerminator());
      // Set the successor of first Body block to Body of the second block.
      if (&BB == Body1) {
        BI->setSuccessor(0, Body2);
      }
      // Set the successor of second body block to Latch of the first block.
      if (&BB == Body2) {
        BI->setSuccessor(0, Latch1);
      }
      // Set the successor of first header block to exit of the second as its
      // contains return insn.
      if (&BB == Header1) {
        BI->setSuccessor(1, Exit2);
      }
    }
    // Function to remove un-wanted basic blocks.
    EliminateUnreachableBlocks(F);
  }

  // Function to get Loop Body Blocks.
  BasicBlock *getBody(Loop *L) {
    BasicBlock *NullBB = nullptr;
    for (BasicBlock *BB : L->getBlocks()) {
      BasicBlock *HeaderBlock = L->getHeader();
      if ((HeaderBlock != BB) && !(L->isLoopLatch(BB))) {
        return BB;
      }
    }
    return NullBB;
  }

  bool adjacent(Loop *Loop1, Loop *Loop2) {

    BasicBlock *Bb1 = Loop1->getExitBlock();
    BasicBlock *Bb2 = Loop2->getLoopPreheader();

    //  If exit block and preHeader are not same.
    if (Bb1 != Bb2) {
      return false;
    }
    return true;
  }

  // Helper function to check and fuse two loops.
  bool fuseCheck(Loop *L1, Loop *L2) {

    // Check if the two loops are adjacent.
    if (!adjacent(L1, L2)) {
      llvm::errs() << "The two loops  " << L1->getName() << " and "
                   << L2->getName() << "  are not adjacent. CANNOT fuse.\n";
      return false;
    }

    // Check if the start integer is same.
    if (startValue(*L1) != startValue(*L2)) {
      llvm::errs() << "The loop check starting values of 2 loops "
                   << L1->getName() << " and " << L2->getName()
                   << "  are not same. CANNOT fuse.\n";
      return false;
    }

    // Check if the limit integer is same.
    if (limitValue(L1) != limitValue(L2)) {
      llvm::errs() << "The loop check limiting value of 2 loops "
                   << L1->getName() << " and " << L2->getName()
                   << "  are not same. CANNOT fuse.\n";
      return false;
    }
    llvm::errs() << "The two loops " << *L1 << " and  " << *L2
                 << " are being fused.\n";
    return true;
  }

  // Check if the start value is same.
  Value *startValue(Loop &LoopV) {
    for (auto &IndVar : LoopV.getHeader()->phis()) {
      Value *V = IndVar.getOperand(1);
      return V;
    }
    return nullptr;
  }

  // Check if the limit value is same.
  Value *limitValue(Loop *LoopV) {
    Value *End, *ContEnd;

    BasicBlock *BB = LoopV->getHeader();
    for (auto I = BB->begin(), E = BB->end(); I != E; ++I) {
      // Check instruction is compare
      if (isa<ICmpInst>(I)) {
        ContEnd = I->getOperand(1);
        // Check end value is constant
        if (dyn_cast<Constant>(ContEnd)) {
          return ContEnd;
        } else {
          for (Use &U : LoopV->getHeader()->getFirstNonPHI()->operands()) {
            if (!dyn_cast<PHINode>(U.get())) {
              Instruction *I = dyn_cast<Instruction>(U.get());
              for (Use &U : I->operands())
                End = U.get();
              return End;
            }
          }
        }
      }
    }

    return nullptr;
  }

  // We don't modify the program, so we preserve all analyses.
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
  }
};
} // namespace

char LoopFusion::ID = 0;
static RegisterPass<LoopFusion>
    X("loopfusion",
      "LoopFusion Implementation Pass (with getAnalysisUsage implemented)");