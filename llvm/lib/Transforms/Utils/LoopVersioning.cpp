//===- LoopVersioning.cpp - Utility to version a loop ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a utility class to perform loop versioning.  The versioned
// loop speculates that otherwise may-aliasing memory accesses don't overlap and
// emits checks to prove this.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/LoopVersioning.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

LoopVersioning::LoopVersioning(const LoopAccessInfo &LAI, Loop *L, LoopInfo *LI,
                               DominatorTree *DT, ScalarEvolution *SE,
                               bool UseLAIChecks)
    : VersionedLoop(L), NonVersionedLoop(nullptr), LAI(LAI), LI(LI), DT(DT),
      SE(SE) {
  assert(L->getExitBlock() && "No single exit block");
  assert(L->getLoopPreheader() && "No preheader");
  if (UseLAIChecks) {
    setAliasChecks(LAI.getRuntimePointerChecking()->getChecks());
    setSCEVChecks(LAI.Preds);
  }
}

void LoopVersioning::setAliasChecks(
    const SmallVector<RuntimePointerChecking::PointerCheck, 4> Checks) {
  AliasChecks = std::move(Checks);
}

void LoopVersioning::setSCEVChecks(SCEVUnionPredicate Check) {
  Preds = std::move(Check);
}

void LoopVersioning::versionLoop(
    const SmallVectorImpl<Instruction *> &DefsUsedOutside) {
  Instruction *FirstCheckInst;
  Instruction *MemRuntimeCheck;
  Value *SCEVRuntimeCheck;
  Value *RuntimeCheck = nullptr;

  // Add the memcheck in the original preheader (this is empty initially).
  BasicBlock *RuntimeCheckBB = VersionedLoop->getLoopPreheader();
  std::tie(FirstCheckInst, MemRuntimeCheck) =
      LAI.addRuntimeChecks(RuntimeCheckBB->getTerminator(), AliasChecks);
  assert(MemRuntimeCheck && "called even though needsAnyChecking = false");

  const SCEVUnionPredicate &Pred = LAI.Preds;
  SCEVExpander Exp(*SE, RuntimeCheckBB->getModule()->getDataLayout(),
                   "scev.check");
  SCEVRuntimeCheck =
      Exp.expandCodeForPredicate(&Pred, RuntimeCheckBB->getTerminator());
  auto *CI = dyn_cast<ConstantInt>(SCEVRuntimeCheck);

  // Discard the SCEV runtime check if it is always true.
  if (CI && CI->isZero())
    SCEVRuntimeCheck = nullptr;

  if (MemRuntimeCheck && SCEVRuntimeCheck) {
    RuntimeCheck = BinaryOperator::Create(Instruction::Or, MemRuntimeCheck,
                                          SCEVRuntimeCheck, "ldist.safe");
    if (auto *I = dyn_cast<Instruction>(RuntimeCheck))
      I->insertBefore(RuntimeCheckBB->getTerminator());
  } else
    RuntimeCheck = MemRuntimeCheck ? MemRuntimeCheck : SCEVRuntimeCheck;

  assert(RuntimeCheck && "called even though we don't need "
                         "any runtime checks");

  // Rename the block to make the IR more readable.
  RuntimeCheckBB->setName(VersionedLoop->getHeader()->getName() +
                          ".lver.check");

  // Create empty preheader for the loop (and after cloning for the
  // non-versioned loop).
  BasicBlock *PH =
      SplitBlock(RuntimeCheckBB, RuntimeCheckBB->getTerminator(), DT, LI);
  PH->setName(VersionedLoop->getHeader()->getName() + ".ph");

  // Clone the loop including the preheader.
  //
  // FIXME: This does not currently preserve SimplifyLoop because the exit
  // block is a join between the two loops.
  SmallVector<BasicBlock *, 8> NonVersionedLoopBlocks;
  NonVersionedLoop =
      cloneLoopWithPreheader(PH, RuntimeCheckBB, VersionedLoop, VMap,
                             ".lver.orig", LI, DT, NonVersionedLoopBlocks);
  remapInstructionsInBlocks(NonVersionedLoopBlocks, VMap);

  // Insert the conditional branch based on the result of the memchecks.
  Instruction *OrigTerm = RuntimeCheckBB->getTerminator();
  BranchInst::Create(NonVersionedLoop->getLoopPreheader(),
                     VersionedLoop->getLoopPreheader(), RuntimeCheck, OrigTerm);
  OrigTerm->eraseFromParent();

  // The loops merge in the original exit block.  This is now dominated by the
  // memchecking block.
  DT->changeImmediateDominator(VersionedLoop->getExitBlock(), RuntimeCheckBB);

  // Adds the necessary PHI nodes for the versioned loops based on the
  // loop-defined values used outside of the loop.
  addPHINodes(DefsUsedOutside);
}

void LoopVersioning::addPHINodes(
    const SmallVectorImpl<Instruction *> &DefsUsedOutside) {
  BasicBlock *PHIBlock = VersionedLoop->getExitBlock();
  assert(PHIBlock && "No single successor to loop exit block");

  for (auto *Inst : DefsUsedOutside) {
    auto *NonVersionedLoopInst = cast<Instruction>(VMap[Inst]);
    PHINode *PN;

    // First see if we have a single-operand PHI with the value defined by the
    // original loop.
    for (auto I = PHIBlock->begin(); (PN = dyn_cast<PHINode>(I)); ++I) {
      assert(PN->getNumOperands() == 1 &&
             "Exit block should only have on predecessor");
      if (PN->getIncomingValue(0) == Inst)
        break;
    }
    // If not create it.
    if (!PN) {
      PN = PHINode::Create(Inst->getType(), 2, Inst->getName() + ".lver",
                           &PHIBlock->front());
      for (auto *User : Inst->users())
        if (!VersionedLoop->contains(cast<Instruction>(User)->getParent()))
          User->replaceUsesOfWith(Inst, PN);
      PN->addIncoming(Inst, VersionedLoop->getExitingBlock());
    }
    // Add the new incoming value from the non-versioned loop.
    PN->addIncoming(NonVersionedLoopInst, NonVersionedLoop->getExitingBlock());
  }
}
