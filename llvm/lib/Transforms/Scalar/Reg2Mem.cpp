//===- Reg2Mem.cpp - Convert registers to allocas -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file demotes all registers to memory references.  It is intended to be
// the inverse of PromoteMemoryToRegister.  By converting to loads, the only
// values live across basic blocks are allocas and loads before phi nodes.
// It is intended that this should make CFG hacking much easier.
// To make later hacking easier, the entry block is split into two, such that
// all introduced allocas and nothing else are in the entry block.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/Reg2Mem.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include <list>
using namespace llvm;

#define DEBUG_TYPE "reg2mem"

STATISTIC(NumRegsDemoted, "Number of registers demoted");
STATISTIC(NumPhisDemoted, "Number of phi-nodes demoted");

static bool valueEscapes(const Instruction &Inst) {
  if (!Inst.getType()->isSized())
    return false;

  const BasicBlock *BB = Inst.getParent();
  for (const User *U : Inst.users()) {
    const Instruction *UI = cast<Instruction>(U);
    if (UI->getParent() != BB || isa<PHINode>(UI))
      return true;
  }
  return false;
}

static bool runPass(Function &F) {
  // Insert all new allocas into entry block.
  BasicBlock *BBEntry = &F.getEntryBlock();
  assert(pred_empty(BBEntry) &&
         "Entry block to function must not have predecessors!");

  // Find first non-alloca instruction and create insertion point. This is
  // safe if block is well-formed: it always have terminator, otherwise
  // we'll get and assertion.
  BasicBlock::iterator I = BBEntry->begin();
  while (isa<AllocaInst>(I)) ++I;

  CastInst *AllocaInsertionPoint = new BitCastInst(
      Constant::getNullValue(Type::getInt32Ty(F.getContext())),
      Type::getInt32Ty(F.getContext()), "reg2mem alloca point", &*I);

  // Find the escaped instructions. But don't create stack slots for
  // allocas in entry block.
  std::list<Instruction*> WorkList;
  for (Instruction &I : instructions(F))
    if (!(isa<AllocaInst>(I) && I.getParent() == BBEntry) && valueEscapes(I))
      WorkList.push_front(&I);

  // Demote escaped instructions
  NumRegsDemoted += WorkList.size();
  for (Instruction *I : WorkList)
    DemoteRegToStack(*I, false, AllocaInsertionPoint);

  WorkList.clear();

  // Find all phi's
  for (BasicBlock &BB : F)
    for (auto &Phi : BB.phis())
      WorkList.push_front(&Phi);

  // Demote phi nodes
  NumPhisDemoted += WorkList.size();
  for (Instruction *I : WorkList)
    DemotePHIToStack(cast<PHINode>(I), AllocaInsertionPoint);

  return true;
}

PreservedAnalyses RegToMemPass::run(Function &F, FunctionAnalysisManager &AM) {
  auto *DT = &AM.getResult<DominatorTreeAnalysis>(F);
  auto *LI = &AM.getResult<LoopAnalysis>(F);
  unsigned N = SplitAllCriticalEdges(F, CriticalEdgeSplittingOptions(DT, LI));
  bool Changed = runPass(F);
  if (N == 0 && !Changed)
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<LoopAnalysis>();
  return PA;
}

namespace {
struct RegToMemLegacy : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  RegToMemLegacy() : FunctionPass(ID) {
    initializeRegToMemLegacyPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequiredID(BreakCriticalEdgesID);
    AU.addPreservedID(BreakCriticalEdgesID);
  }

  bool runOnFunction(Function &F) override {
    if (F.isDeclaration() || skipFunction(F))
      return false;
    return runPass(F);
  }
};
} // namespace

char RegToMemLegacy::ID = 0;
INITIALIZE_PASS_BEGIN(RegToMemLegacy, "reg2mem",
                      "Demote all values to stack slots", false, false)
INITIALIZE_PASS_DEPENDENCY(BreakCriticalEdges)
INITIALIZE_PASS_END(RegToMemLegacy, "reg2mem",
                    "Demote all values to stack slots", false, false)

// createDemoteRegisterToMemory - Provide an entry point to create this pass.
char &llvm::DemoteRegisterToMemoryID = RegToMemLegacy::ID;
FunctionPass *llvm::createDemoteRegisterToMemoryPass() {
  return new RegToMemLegacy();
}
