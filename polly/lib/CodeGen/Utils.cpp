//===--- Utils.cpp - Utility functions for the code generation --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains utility functions for the code generation.
//
//===----------------------------------------------------------------------===//

#include "polly/CodeGen/Utils.h"
#include "polly/ScopInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

BasicBlock *polly::executeScopConditionally(Scop &S, Pass *PassInfo) {
  BasicBlock *StartBlock, *SplitBlock, *NewBlock;
  Region &R = S.getRegion();
  IRBuilder<> Builder(R.getEntry());
  DominatorTree &DT = PassInfo->getAnalysis<DominatorTree>();
  RegionInfo &RI = PassInfo->getAnalysis<RegionInfo>();

  // Split the entry edge of the region and generate a new basic block on this
  // edge. This function also updates ScopInfo and RegionInfo.
  NewBlock = SplitEdge(R.getEnteringBlock(), R.getEntry(), PassInfo);
  if (DT.dominates(R.getEntry(), NewBlock)) {
    BasicBlock *OldBlock = R.getEntry();
    std::string OldName = OldBlock->getName();

    // Update ScopInfo.
    for (Scop::iterator SI = S.begin(), SE = S.end(); SI != SE; ++SI)
      if ((*SI)->getBasicBlock() == OldBlock) {
        (*SI)->setBasicBlock(NewBlock);
        break;
      }

    // Update RegionInfo.
    SplitBlock = OldBlock;
    OldBlock->setName("polly.split");
    NewBlock->setName(OldName);
    R.replaceEntryRecursive(NewBlock);
    RI.setRegionFor(NewBlock, &R);
  } else {
    RI.setRegionFor(NewBlock, R.getParent());
    SplitBlock = NewBlock;
  }

  SplitBlock->setName("polly.split_new_and_old");
  Function *F = SplitBlock->getParent();
  StartBlock = BasicBlock::Create(F->getContext(), "polly.start", F);
  SplitBlock->getTerminator()->eraseFromParent();
  Builder.SetInsertPoint(SplitBlock);
  Builder.CreateCondBr(Builder.getTrue(), StartBlock, R.getEntry());
  DT.addNewBlock(StartBlock, SplitBlock);
  Builder.SetInsertPoint(StartBlock);

  BasicBlock *MergeBlock;

  if (R.getExit()->getSinglePredecessor())
    // No splitEdge required.  A block with a single predecessor cannot have
    // PHI nodes that would complicate life.
    MergeBlock = R.getExit();
  else {
    MergeBlock = SplitEdge(R.getExitingBlock(), R.getExit(), PassInfo);
    // SplitEdge will never split R.getExit(), as R.getExit() has more than
    // one predecessor. Hence, mergeBlock is always a newly generated block.
    R.replaceExitRecursive(MergeBlock);
    RI.setRegionFor(MergeBlock, &R);
  }

  Builder.CreateBr(MergeBlock);
  MergeBlock->setName("polly.merge_new_and_old");

  if (DT.dominates(SplitBlock, MergeBlock))
    DT.changeImmediateDominator(MergeBlock, SplitBlock);
  return StartBlock;
}
