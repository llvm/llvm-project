//===- DCE.cpp - Code to perform dead code elimination --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements dead inst elimination and dead code elimination.
//
// Dead Inst Elimination performs a single pass over the function removing
// instructions that are obviously dead.  Dead Code Elimination is similar, but
// it rechecks instructions that were used by removed instructions to see if
// they are newly dead.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "dce"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Instruction.h"
#include "llvm/Pass.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/ADT/Statistic.h"
#include <set>
using namespace llvm;

STATISTIC(DIEEliminated, "Number of insts removed by DIE pass");
STATISTIC(DCEEliminated, "Number of insts removed");

namespace {
  //===--------------------------------------------------------------------===//
  // DeadInstElimination pass implementation
  //
  struct VISIBILITY_HIDDEN DeadInstElimination : public BasicBlockPass {
    static char ID; // Pass identification, replacement for typeid
    DeadInstElimination() : BasicBlockPass(intptr_t(&ID)) {}
    virtual bool runOnBasicBlock(BasicBlock &BB) {
      bool Changed = false;
      for (BasicBlock::iterator DI = BB.begin(); DI != BB.end(); ) {
        Instruction *Inst = DI++;
        if (isInstructionTriviallyDead(Inst)) {
          Inst->eraseFromParent();
          Changed = true;
          ++DIEEliminated;
        }
      }
      return Changed;
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
    }
  };
}

char DeadInstElimination::ID = 0;
static RegisterPass<DeadInstElimination>
X("die", "Dead Instruction Elimination");

Pass *llvm::createDeadInstEliminationPass() {
  return new DeadInstElimination();
}


namespace {
  //===--------------------------------------------------------------------===//
  // DeadCodeElimination pass implementation
  //
  struct DCE : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    DCE() : FunctionPass(&ID) {}

    virtual bool runOnFunction(Function &F);

     virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
    }
 };
}

char DCE::ID = 0;
static RegisterPass<DCE> Y("dce", "Dead Code Elimination");

bool DCE::runOnFunction(Function &F) {
  // Start out with all of the instructions in the worklist...
  std::vector<Instruction*> WorkList;
  for (inst_iterator i = inst_begin(F), e = inst_end(F); i != e; ++i)
    WorkList.push_back(&*i);

  // Loop over the worklist finding instructions that are dead.  If they are
  // dead make them drop all of their uses, making other instructions
  // potentially dead, and work until the worklist is empty.
  //
  bool MadeChange = false;
  while (!WorkList.empty()) {
    Instruction *I = WorkList.back();
    WorkList.pop_back();

    if (isInstructionTriviallyDead(I)) {       // If the instruction is dead.
      // Loop over all of the values that the instruction uses, if there are
      // instructions being used, add them to the worklist, because they might
      // go dead after this one is removed.
      //
      for (User::op_iterator OI = I->op_begin(), E = I->op_end(); OI != E; ++OI)
        if (Instruction *Used = dyn_cast<Instruction>(*OI))
          WorkList.push_back(Used);

      // Remove the instruction.
      I->eraseFromParent();

      // Remove the instruction from the worklist if it still exists in it.
      for (std::vector<Instruction*>::iterator WI = WorkList.begin();
           WI != WorkList.end(); ) {
        if (*WI == I)
          WI = WorkList.erase(WI);
        else
          ++WI;
      }

      MadeChange = true;
      ++DCEEliminated;
    }
  }
  return MadeChange;
}

FunctionPass *llvm::createDeadCodeEliminationPass() {
  return new DCE();
}

