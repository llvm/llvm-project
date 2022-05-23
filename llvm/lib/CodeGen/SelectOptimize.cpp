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
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSchedule.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/SizeOpts.h"
#include <algorithm>
#include <memory>
#include <queue>
#include <stack>
#include <string>

using namespace llvm;

#define DEBUG_TYPE "select-optimize"

STATISTIC(NumSelectOptAnalyzed,
          "Number of select groups considered for conversion to branch");
STATISTIC(NumSelectConvertedExpColdOperand,
          "Number of select groups converted due to expensive cold operand");
STATISTIC(NumSelectConvertedHighPred,
          "Number of select groups converted due to high-predictability");
STATISTIC(NumSelectUnPred,
          "Number of select groups not converted due to unpredictability");
STATISTIC(NumSelectColdBB,
          "Number of select groups not converted due to cold basic block");
STATISTIC(NumSelectsConverted, "Number of selects converted");

static cl::opt<unsigned> ColdOperandThreshold(
    "cold-operand-threshold",
    cl::desc("Maximum frequency of path for an operand to be considered cold."),
    cl::init(20), cl::Hidden);

static cl::opt<unsigned> ColdOperandMaxCostMultiplier(
    "cold-operand-max-cost-multiplier",
    cl::desc("Maximum cost multiplier of TCC_expensive for the dependence "
             "slice of a cold operand to be considered inexpensive."),
    cl::init(1), cl::Hidden);

namespace {

class SelectOptimize : public FunctionPass {
  const TargetMachine *TM = nullptr;
  const TargetSubtargetInfo *TSI;
  const TargetLowering *TLI = nullptr;
  const TargetTransformInfo *TTI = nullptr;
  const LoopInfo *LI;
  DominatorTree *DT;
  std::unique_ptr<BlockFrequencyInfo> BFI;
  std::unique_ptr<BranchProbabilityInfo> BPI;
  ProfileSummaryInfo *PSI;
  OptimizationRemarkEmitter *ORE;

public:
  static char ID;

  SelectOptimize() : FunctionPass(ID) {
    initializeSelectOptimizePass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<ProfileSummaryInfoWrapperPass>();
    AU.addRequired<TargetPassConfig>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<OptimizationRemarkEmitterWrapperPass>();
  }

private:
  // Select groups consist of consecutive select instructions with the same
  // condition.
  using SelectGroup = SmallVector<SelectInst *, 2>;
  using SelectGroups = SmallVector<SelectGroup, 2>;

  // Converts select instructions of a function to conditional jumps when deemed
  // profitable. Returns true if at least one select was converted.
  bool optimizeSelects(Function &F);

  // Heuristics for determining which select instructions can be profitably
  // conveted to branches. Separate heuristics for selects in inner-most loops
  // and the rest of code regions (base heuristics for non-inner-most loop
  // regions).
  void optimizeSelectsBase(Function &F, SelectGroups &ProfSIGroups);
  void optimizeSelectsInnerLoops(Function &F, SelectGroups &ProfSIGroups);

  // Converts to branches the select groups that were deemed
  // profitable-to-convert.
  void convertProfitableSIGroups(SelectGroups &ProfSIGroups);

  // Splits selects of a given basic block into select groups.
  void collectSelectGroups(BasicBlock &BB, SelectGroups &SIGroups);

  // Determines for which select groups it is profitable converting to branches
  // (base heuristics).
  void findProfitableSIGroupsBase(SelectGroups &SIGroups,
                                  SelectGroups &ProfSIGroups);
  // Determines if a select group should be converted to a branch (base
  // heuristics).
  bool isConvertToBranchProfitableBase(const SmallVector<SelectInst *, 2> &ASI);

  // Returns true if there are expensive instructions in the cold value
  // operand's (if any) dependence slice of any of the selects of the given
  // group.
  bool hasExpensiveColdOperand(const SmallVector<SelectInst *, 2> &ASI);

  // For a given source instruction, collect its backwards dependence slice
  // consisting of instructions exclusively computed for producing the operands
  // of the source instruction.
  void getExclBackwardsSlice(Instruction *I,
                             SmallVector<Instruction *, 2> &Slice);

  // Returns true if the condition of the select is highly predictable.
  bool isSelectHighlyPredictable(const SelectInst *SI);

  // Returns true if the target architecture supports lowering a given select.
  bool isSelectKindSupported(SelectInst *SI);
};
} // namespace

char SelectOptimize::ID = 0;

INITIALIZE_PASS_BEGIN(SelectOptimize, DEBUG_TYPE, "Optimize selects", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ProfileSummaryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(OptimizationRemarkEmitterWrapperPass)
INITIALIZE_PASS_END(SelectOptimize, DEBUG_TYPE, "Optimize selects", false,
                    false)

FunctionPass *llvm::createSelectOptimizePass() { return new SelectOptimize(); }

bool SelectOptimize::runOnFunction(Function &F) {
  TM = &getAnalysis<TargetPassConfig>().getTM<TargetMachine>();
  TSI = TM->getSubtargetImpl(F);
  TLI = TSI->getTargetLowering();

  // If none of the select types is supported then skip this pass.
  // This is an optimization pass. Legality issues will be handled by
  // instruction selection.
  if (!TLI->isSelectSupported(TargetLowering::ScalarValSelect) &&
      !TLI->isSelectSupported(TargetLowering::ScalarCondVectorVal) &&
      !TLI->isSelectSupported(TargetLowering::VectorMaskSelect))
    return false;

  TTI = &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  BPI.reset(new BranchProbabilityInfo(F, *LI));
  BFI.reset(new BlockFrequencyInfo(F, *BPI, *LI));
  PSI = &getAnalysis<ProfileSummaryInfoWrapperPass>().getPSI();
  ORE = &getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE();

  // When optimizing for size, selects are preferable over branches.
  if (F.hasOptSize() || llvm::shouldOptimizeForSize(&F, PSI, BFI.get()))
    return false;

  return optimizeSelects(F);
}

bool SelectOptimize::optimizeSelects(Function &F) {
  // Determine for which select groups it is profitable converting to branches.
  SelectGroups ProfSIGroups;
  // Base heuristics apply only to non-loops and outer loops.
  optimizeSelectsBase(F, ProfSIGroups);
  // Separate heuristics for inner-most loops.
  optimizeSelectsInnerLoops(F, ProfSIGroups);

  // Convert to branches the select groups that were deemed
  // profitable-to-convert.
  convertProfitableSIGroups(ProfSIGroups);

  // Code modified if at least one select group was converted.
  return !ProfSIGroups.empty();
}

void SelectOptimize::optimizeSelectsBase(Function &F,
                                         SelectGroups &ProfSIGroups) {
  // Collect all the select groups.
  SelectGroups SIGroups;
  for (BasicBlock &BB : F) {
    // Base heuristics apply only to non-loops and outer loops.
    Loop *L = LI->getLoopFor(&BB);
    if (L && L->isInnermost())
      continue;
    collectSelectGroups(BB, SIGroups);
  }

  // Determine for which select groups it is profitable converting to branches.
  findProfitableSIGroupsBase(SIGroups, ProfSIGroups);
}

void SelectOptimize::optimizeSelectsInnerLoops(Function &F,
                                               SelectGroups &ProfSIGroups) {}

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

void SelectOptimize::findProfitableSIGroupsBase(SelectGroups &SIGroups,
                                                SelectGroups &ProfSIGroups) {
  for (SelectGroup &ASI : SIGroups) {
    ++NumSelectOptAnalyzed;
    if (isConvertToBranchProfitableBase(ASI))
      ProfSIGroups.push_back(ASI);
  }
}

bool SelectOptimize::isConvertToBranchProfitableBase(
    const SmallVector<SelectInst *, 2> &ASI) {
  SelectInst *SI = ASI.front();
  OptimizationRemark OR(DEBUG_TYPE, "SelectOpti", SI);
  OptimizationRemarkMissed ORmiss(DEBUG_TYPE, "SelectOpti", SI);

  // Skip cold basic blocks. Better to optimize for size for cold blocks.
  if (PSI->isColdBlock(SI->getParent(), BFI.get())) {
    ++NumSelectColdBB;
    ORmiss << "Not converted to branch because of cold basic block. ";
    ORE->emit(ORmiss);
    return false;
  }

  // If unpredictable, branch form is less profitable.
  if (SI->getMetadata(LLVMContext::MD_unpredictable)) {
    ++NumSelectUnPred;
    ORmiss << "Not converted to branch because of unpredictable branch. ";
    ORE->emit(ORmiss);
    return false;
  }

  // If highly predictable, branch form is more profitable, unless a
  // predictable select is inexpensive in the target architecture.
  if (isSelectHighlyPredictable(SI) && TLI->isPredictableSelectExpensive()) {
    ++NumSelectConvertedHighPred;
    OR << "Converted to branch because of highly predictable branch. ";
    ORE->emit(OR);
    return true;
  }

  // Look for expensive instructions in the cold operand's (if any) dependence
  // slice of any of the selects in the group.
  if (hasExpensiveColdOperand(ASI)) {
    ++NumSelectConvertedExpColdOperand;
    OR << "Converted to branch because of expensive cold operand.";
    ORE->emit(OR);
    return true;
  }

  ORmiss << "Not profitable to convert to branch (base heuristic).";
  ORE->emit(ORmiss);
  return false;
}

static InstructionCost divideNearest(InstructionCost Numerator,
                                     uint64_t Denominator) {
  return (Numerator + (Denominator / 2)) / Denominator;
}

bool SelectOptimize::hasExpensiveColdOperand(
    const SmallVector<SelectInst *, 2> &ASI) {
  bool ColdOperand = false;
  uint64_t TrueWeight, FalseWeight, TotalWeight;
  if (ASI.front()->extractProfMetadata(TrueWeight, FalseWeight)) {
    uint64_t MinWeight = std::min(TrueWeight, FalseWeight);
    TotalWeight = TrueWeight + FalseWeight;
    // Is there a path with frequency <ColdOperandThreshold% (default:20%) ?
    ColdOperand = TotalWeight * ColdOperandThreshold > 100 * MinWeight;
  } else if (PSI->hasProfileSummary()) {
    OptimizationRemarkMissed ORmiss(DEBUG_TYPE, "SelectOpti", ASI.front());
    ORmiss << "Profile data available but missing branch-weights metadata for "
              "select instruction. ";
    ORE->emit(ORmiss);
  }
  if (!ColdOperand)
    return false;
  // Check if the cold path's dependence slice is expensive for any of the
  // selects of the group.
  for (SelectInst *SI : ASI) {
    Instruction *ColdI = nullptr;
    uint64_t HotWeight;
    if (TrueWeight < FalseWeight) {
      ColdI = dyn_cast<Instruction>(SI->getTrueValue());
      HotWeight = FalseWeight;
    } else {
      ColdI = dyn_cast<Instruction>(SI->getFalseValue());
      HotWeight = TrueWeight;
    }
    if (ColdI) {
      SmallVector<Instruction *, 2> ColdSlice;
      getExclBackwardsSlice(ColdI, ColdSlice);
      InstructionCost SliceCost = 0;
      for (auto *ColdII : ColdSlice) {
        SliceCost +=
            TTI->getInstructionCost(ColdII, TargetTransformInfo::TCK_Latency);
      }
      // The colder the cold value operand of the select is the more expensive
      // the cmov becomes for computing the cold value operand every time. Thus,
      // the colder the cold operand is the more its cost counts.
      // Get nearest integer cost adjusted for coldness.
      InstructionCost AdjSliceCost =
          divideNearest(SliceCost * HotWeight, TotalWeight);
      if (AdjSliceCost >=
          ColdOperandMaxCostMultiplier * TargetTransformInfo::TCC_Expensive)
        return true;
    }
  }
  return false;
}

// For a given source instruction, collect its backwards dependence slice
// consisting of instructions exclusively computed for the purpose of producing
// the operands of the source instruction. As an approximation
// (sufficiently-accurate in practice), we populate this set with the
// instructions of the backwards dependence slice that only have one-use and
// form an one-use chain that leads to the source instruction.
void SelectOptimize::getExclBackwardsSlice(
    Instruction *I, SmallVector<Instruction *, 2> &Slice) {
  SmallPtrSet<Instruction *, 2> Visited;
  std::queue<Instruction *> Worklist;
  Worklist.push(I);
  while (!Worklist.empty()) {
    Instruction *II = Worklist.front();
    Worklist.pop();

    // Avoid cycles.
    if (Visited.count(II))
      continue;
    Visited.insert(II);

    if (!II->hasOneUse())
      continue;

    // Avoid considering instructions with less frequency than the source
    // instruction (i.e., avoid colder code regions of the dependence slice).
    if (BFI->getBlockFreq(II->getParent()) < BFI->getBlockFreq(I->getParent()))
      continue;

    // Eligible one-use instruction added to the dependence slice.
    Slice.push_back(II);

    // Explore all the operands of the current instruction to expand the slice.
    for (unsigned k = 0; k < II->getNumOperands(); ++k)
      if (auto *OpI = dyn_cast<Instruction>(II->getOperand(k)))
        Worklist.push(OpI);
  }
}

bool SelectOptimize::isSelectHighlyPredictable(const SelectInst *SI) {
  uint64_t TrueWeight, FalseWeight;
  if (SI->extractProfMetadata(TrueWeight, FalseWeight)) {
    uint64_t Max = std::max(TrueWeight, FalseWeight);
    uint64_t Sum = TrueWeight + FalseWeight;
    if (Sum != 0) {
      auto Probability = BranchProbability::getBranchProbability(Max, Sum);
      if (Probability > TTI->getPredictableBranchThreshold())
        return true;
    }
  }
  return false;
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
