//===--- SelectOptimize.cpp - Convert select to branches if profitable ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass converts selects to conditional jumps when profitable.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSchedule.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "select-optimize"

STATISTIC(NumSelectsConverted, "Number of selects converted");

namespace {

class SelectOptimize : public FunctionPass {
  const TargetMachine *TM = nullptr;
  const TargetSubtargetInfo *TSI;
  const TargetLowering *TLI = nullptr;
  const LoopInfo *LI;
  std::unique_ptr<BlockFrequencyInfo> BFI;
  std::unique_ptr<BranchProbabilityInfo> BPI;

public:
  static char ID;
  SelectOptimize() : FunctionPass(ID) {
    initializeSelectOptimizePass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.addRequired<LoopInfoWrapperPass>();
  }

private:
  // Select groups consist of consecutive select instructions with the same
  // condition.
  using SelectGroup = SmallVector<SelectInst *, 2>;
  using SelectGroups = SmallVector<SelectGroup, 2>;

  bool optimizeSelects(Function &F);
  void convertProfitableSIGroups(SelectGroups &ProfSIGroups);
  void collectSelectGroups(BasicBlock &BB, SelectGroups &SIGroups);
  bool isSelectKindSupported(SelectInst *SI);
};
} // namespace

char SelectOptimize::ID = 0;

INITIALIZE_PASS_BEGIN(SelectOptimize, DEBUG_TYPE, "Optimize selects", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(SelectOptimize, DEBUG_TYPE, "Optimize selects", false,
                    false)

FunctionPass *llvm::createSelectOptimizePass() { return new SelectOptimize(); }

bool SelectOptimize::runOnFunction(Function &F) {
  TM = &getAnalysis<TargetPassConfig>().getTM<TargetMachine>();
  TSI = TM->getSubtargetImpl(F);
  TLI = TSI->getTargetLowering();
  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  BPI.reset(new BranchProbabilityInfo(F, *LI));
  BFI.reset(new BlockFrequencyInfo(F, *BPI, *LI));

  return optimizeSelects(F);
}

bool SelectOptimize::optimizeSelects(Function &F) {
  // Collect all the select groups.
  SelectGroups SIGroups;
  for (BasicBlock &BB : F) {
    collectSelectGroups(BB, SIGroups);
  }

  // Determine for which select groups it is profitable converting to branches.
  SelectGroups ProfSIGroups;
  // For now assume that all select groups can be profitably converted to
  // branches.
  for (SelectGroup &ASI : SIGroups) {
    ProfSIGroups.push_back(ASI);
  }

  // Convert to branches the select groups that were deemed
  // profitable-to-convert.
  convertProfitableSIGroups(ProfSIGroups);

  // Code modified if at least one select group was converted.
  return !ProfSIGroups.empty();
}

/// If \p isTrue is true, return the true value of \p SI, otherwise return
/// false value of \p SI. If the true/false value of \p SI is defined by any
/// select instructions in \p Selects, look through the defining select
/// instruction until the true/false value is not defined in \p Selects.
static Value *
getTrueOrFalseValue(SelectInst *SI, bool isTrue,
                    const SmallPtrSet<const Instruction *, 2> &Selects) {
  Value *V = nullptr;
  for (SelectInst *DefSI = SI; DefSI != nullptr && Selects.count(DefSI);
       DefSI = dyn_cast<SelectInst>(V)) {
    assert(DefSI->getCondition() == SI->getCondition() &&
           "The condition of DefSI does not match with SI");
    V = (isTrue ? DefSI->getTrueValue() : DefSI->getFalseValue());
  }
  assert(V && "Failed to get select true/false value");
  return V;
}

void SelectOptimize::convertProfitableSIGroups(SelectGroups &ProfSIGroups) {
  for (SelectGroup &ASI : ProfSIGroups) {
    // TODO: eliminate the redundancy of logic transforming selects to branches
    // by removing CodeGenPrepare::optimizeSelectInst and optimizing here
    // selects for all cases (with and without profile information).

    // Transform a sequence like this:
    //    start:
    //       %cmp = cmp uge i32 %a, %b
    //       %sel = select i1 %cmp, i32 %c, i32 %d
    //
    // Into:
    //    start:
    //       %cmp = cmp uge i32 %a, %b
    //       %cmp.frozen = freeze %cmp
    //       br i1 %cmp.frozen, label %select.end, label %select.false
    //    select.false:
    //       br label %select.end
    //    select.end:
    //       %sel = phi i32 [ %c, %start ], [ %d, %select.false ]
    //
    // %cmp should be frozen, otherwise it may introduce undefined behavior.

    // We split the block containing the select(s) into two blocks.
    SelectInst *SI = ASI.front();
    SelectInst *LastSI = ASI.back();
    BasicBlock *StartBlock = SI->getParent();
    BasicBlock::iterator SplitPt = ++(BasicBlock::iterator(LastSI));
    BasicBlock *EndBlock = StartBlock->splitBasicBlock(SplitPt, "select.end");
    BFI->setBlockFreq(EndBlock, BFI->getBlockFreq(StartBlock).getFrequency());
    // Delete the unconditional branch that was just created by the split.
    StartBlock->getTerminator()->eraseFromParent();

    // Move any debug/pseudo instructions that were in-between the select
    // group to the newly-created end block.
    SmallVector<Instruction *, 2> DebugPseudoINS;
    auto DIt = SI->getIterator();
    while (&*DIt != LastSI) {
      if (DIt->isDebugOrPseudoInst())
        DebugPseudoINS.push_back(&*DIt);
      DIt++;
    }
    for (auto DI : DebugPseudoINS) {
      DI->moveBefore(&*EndBlock->getFirstInsertionPt());
    }

    // These are the new basic blocks for the conditional branch.
    // For now, no instruction sinking to the true/false blocks.
    // Thus both True and False blocks will be empty.
    BasicBlock *TrueBlock = nullptr, *FalseBlock = nullptr;

    // Use the 'false' side for a new input value to the PHI.
    FalseBlock = BasicBlock::Create(SI->getContext(), "select.false",
                                    EndBlock->getParent(), EndBlock);
    auto *FalseBranch = BranchInst::Create(EndBlock, FalseBlock);
    FalseBranch->setDebugLoc(SI->getDebugLoc());

    // For the 'true' side the path originates from the start block from the
    // point view of the new PHI.
    TrueBlock = StartBlock;

    // Insert the real conditional branch based on the original condition.
    BasicBlock *TT, *FT;
    TT = EndBlock;
    FT = FalseBlock;
    IRBuilder<> IB(SI);
    auto *CondFr =
        IB.CreateFreeze(SI->getCondition(), SI->getName() + ".frozen");
    IB.CreateCondBr(CondFr, TT, FT, SI);

    SmallPtrSet<const Instruction *, 2> INS;
    INS.insert(ASI.begin(), ASI.end());
    // Use reverse iterator because later select may use the value of the
    // earlier select, and we need to propagate value through earlier select
    // to get the PHI operand.
    for (auto It = ASI.rbegin(); It != ASI.rend(); ++It) {
      SelectInst *SI = *It;
      // The select itself is replaced with a PHI Node.
      PHINode *PN = PHINode::Create(SI->getType(), 2, "", &EndBlock->front());
      PN->takeName(SI);
      PN->addIncoming(getTrueOrFalseValue(SI, true, INS), TrueBlock);
      PN->addIncoming(getTrueOrFalseValue(SI, false, INS), FalseBlock);
      PN->setDebugLoc(SI->getDebugLoc());

      SI->replaceAllUsesWith(PN);
      SI->eraseFromParent();
      INS.erase(SI);
      ++NumSelectsConverted;
    }
  }
}

void SelectOptimize::collectSelectGroups(BasicBlock &BB,
                                         SelectGroups &SIGroups) {
  BasicBlock::iterator BBIt = BB.begin();
  while (BBIt != BB.end()) {
    Instruction *I = &*BBIt++;
    if (SelectInst *SI = dyn_cast<SelectInst>(I)) {
      SelectGroup SIGroup;
      SIGroup.push_back(SI);
      while (BBIt != BB.end()) {
        Instruction *NI = &*BBIt;
        SelectInst *NSI = dyn_cast<SelectInst>(NI);
        if (NSI && SI->getCondition() == NSI->getCondition()) {
          SIGroup.push_back(NSI);
        } else if (!NI->isDebugOrPseudoInst()) {
          // Debug/pseudo instructions should be skipped and not prevent the
          // formation of a select group.
          break;
        }
        ++BBIt;
      }

      // If the select type is not supported, no point optimizing it.
      // Instruction selection will take care of it.
      if (!isSelectKindSupported(SI))
        continue;

      SIGroups.push_back(SIGroup);
    }
  }
}

bool SelectOptimize::isSelectKindSupported(SelectInst *SI) {
  bool VectorCond = !SI->getCondition()->getType()->isIntegerTy(1);
  if (VectorCond)
    return false;
  TargetLowering::SelectSupportKind SelectKind;
  if (SI->getType()->isVectorTy())
    SelectKind = TargetLowering::ScalarCondVectorVal;
  else
    SelectKind = TargetLowering::ScalarValSelect;
  return TLI->isSelectSupported(SelectKind);
}
