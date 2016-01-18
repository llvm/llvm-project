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
#include "llvm/IR/Dominators.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

LoopVersioning::LoopVersioning(
    SmallVector<RuntimePointerChecking::PointerCheck, 4> Checks,
    const LoopAccessInfo &LAI, Loop *L, LoopInfo *LI, DominatorTree *DT)
    : VersionedLoop(L), NonVersionedLoop(nullptr), Checks(std::move(Checks)),
      LAI(LAI), LI(LI), DT(DT) {
  assert(L->getExitBlock() && "No single exit block");
  assert(L->getLoopPreheader() && "No preheader");
}

LoopVersioning::LoopVersioning(const LoopAccessInfo &LAInfo, Loop *L,
                               LoopInfo *LI, DominatorTree *DT)
    : VersionedLoop(L), NonVersionedLoop(nullptr),
      Checks(LAInfo.getRuntimePointerChecking()->getChecks()), LAI(LAInfo),
      LI(LI), DT(DT) {
  assert(L->getExitBlock() && "No single exit block");
  assert(L->getLoopPreheader() && "No preheader");
}

void LoopVersioning::versionLoop(
    const SmallVectorImpl<Instruction *> &DefsUsedOutside) {
  Instruction *FirstCheckInst;
  Instruction *MemRuntimeCheck;
  // Add the memcheck in the original preheader (this is empty initially).
  BasicBlock *MemCheckBB = VersionedLoop->getLoopPreheader();
  std::tie(FirstCheckInst, MemRuntimeCheck) =
      LAI.addRuntimeChecks(MemCheckBB->getTerminator(), Checks);
  assert(MemRuntimeCheck && "called even though needsAnyChecking = false");

  // Rename the block to make the IR more readable.
  MemCheckBB->setName(VersionedLoop->getHeader()->getName() + ".lver.memcheck");

  // Create empty preheader for the loop (and after cloning for the
  // non-versioned loop).
  BasicBlock *PH = SplitBlock(MemCheckBB, MemCheckBB->getTerminator(), DT, LI);
  PH->setName(VersionedLoop->getHeader()->getName() + ".ph");

  // Clone the loop including the preheader.
  //
  // FIXME: This does not currently preserve SimplifyLoop because the exit
  // block is a join between the two loops.
  SmallVector<BasicBlock *, 8> NonVersionedLoopBlocks;
  NonVersionedLoop =
      cloneLoopWithPreheader(PH, MemCheckBB, VersionedLoop, VMap, ".lver.orig",
                             LI, DT, NonVersionedLoopBlocks);
  remapInstructionsInBlocks(NonVersionedLoopBlocks, VMap);

  // Insert the conditional branch based on the result of the memchecks.
  Instruction *OrigTerm = MemCheckBB->getTerminator();
  BranchInst::Create(NonVersionedLoop->getLoopPreheader(),
                     VersionedLoop->getLoopPreheader(), MemRuntimeCheck,
                     OrigTerm);
  OrigTerm->eraseFromParent();

  // The loops merge in the original exit block.  This is now dominated by the
  // memchecking block.
  DT->changeImmediateDominator(VersionedLoop->getExitBlock(), MemCheckBB);

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
