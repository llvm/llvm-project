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

#include "llvm/ADT/Optional.h"
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
#include "llvm/Support/ScaledNumber.h"
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
STATISTIC(NumSelectConvertedLoop,
          "Number of select groups converted due to loop-level analysis");
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

static cl::opt<unsigned>
    GainGradientThreshold("select-opti-loop-gradient-gain-threshold",
                          cl::desc("Gradient gain threshold (%)."),
                          cl::init(25), cl::Hidden);

static cl::opt<unsigned>
    GainCycleThreshold("select-opti-loop-cycle-gain-threshold",
                       cl::desc("Minimum gain per loop (in cycles) threshold."),
                       cl::init(4), cl::Hidden);

static cl::opt<unsigned> GainRelativeThreshold(
    "select-opti-loop-relative-gain-threshold",
    cl::desc(
        "Minimum relative gain per loop threshold (1/X). Defaults to 12.5%"),
    cl::init(8), cl::Hidden);

static cl::opt<unsigned> MispredictDefaultRate(
    "mispredict-default-rate", cl::Hidden, cl::init(25),
    cl::desc("Default mispredict rate (initialized to 25%)."));

static cl::opt<bool>
    DisableLoopLevelHeuristics("disable-loop-level-heuristics", cl::Hidden,
                               cl::init(false),
                               cl::desc("Disable loop-level heuristics."));

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
  TargetSchedModel TSchedModel;

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

  using Scaled64 = ScaledNumber<uint64_t>;

  struct CostInfo {
    /// Predicated cost (with selects as conditional moves).
    Scaled64 PredCost;
    /// Non-predicated cost (with selects converted to branches).
    Scaled64 NonPredCost;
  };

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
  // (base and inner-most-loop heuristics).
  void findProfitableSIGroupsBase(SelectGroups &SIGroups,
                                  SelectGroups &ProfSIGroups);
  void findProfitableSIGroupsInnerLoops(const Loop *L, SelectGroups &SIGroups,
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
  void getExclBackwardsSlice(Instruction *I, std::stack<Instruction *> &Slice,
                             bool ForSinking = false);

  // Returns true if the condition of the select is highly predictable.
  bool isSelectHighlyPredictable(const SelectInst *SI);

  // Loop-level checks to determine if a non-predicated version (with branches)
  // of the given loop is more profitable than its predicated version.
  bool checkLoopHeuristics(const Loop *L, const CostInfo LoopDepth[2]);

  // Computes instruction and loop-critical-path costs for both the predicated
  // and non-predicated version of the given loop.
  bool computeLoopCosts(const Loop *L, const SelectGroups &SIGroups,
                        DenseMap<const Instruction *, CostInfo> &InstCostMap,
                        CostInfo *LoopCost);

  // Returns a set of all the select instructions in the given select groups.
  SmallPtrSet<const Instruction *, 2> getSIset(const SelectGroups &SIGroups);

  // Returns the latency cost of a given instruction.
  Optional<uint64_t> computeInstCost(const Instruction *I);

  // Returns the misprediction cost of a given select when converted to branch.
  Scaled64 getMispredictionCost(const SelectInst *SI, const Scaled64 CondCost);

  // Returns the cost of a branch when the prediction is correct.
  Scaled64 getPredictedPathCost(Scaled64 TrueCost, Scaled64 FalseCost,
                                const SelectInst *SI);

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
  TSchedModel.init(TSI);

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
                                               SelectGroups &ProfSIGroups) {
  SmallVector<Loop *, 4> Loops(LI->begin(), LI->end());
  // Need to check size on each iteration as we accumulate child loops.
  for (unsigned long i = 0; i < Loops.size(); ++i)
    for (Loop *ChildL : Loops[i]->getSubLoops())
      Loops.push_back(ChildL);

  for (Loop *L : Loops) {
    if (!L->isInnermost())
      continue;

    SelectGroups SIGroups;
    for (BasicBlock *BB : L->getBlocks())
      collectSelectGroups(*BB, SIGroups);

    findProfitableSIGroupsInnerLoops(L, SIGroups, ProfSIGroups);
  }
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
    // The code transformation here is a modified version of the sinking
    // transformation in CodeGenPrepare::optimizeSelectInst with a more
    // aggressive strategy of which instructions to sink.
    //
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
    //       br i1 %cmp.frozen, label %select.true, label %select.false
    //    select.true:
    //       br label %select.end
    //    select.false:
    //       br label %select.end
    //    select.end:
    //       %sel = phi i32 [ %c, %select.true ], [ %d, %select.false ]
    //
    // %cmp should be frozen, otherwise it may introduce undefined behavior.
    // In addition, we may sink instructions that produce %c or %d into the
    // destination(s) of the new branch.
    // If the true or false blocks do not contain a sunken instruction, that
    // block and its branch may be optimized away. In that case, one side of the
    // first branch will point directly to select.end, and the corresponding PHI
    // predecessor block will be the start block.

    // Find all the instructions that can be soundly sunk to the true/false
    // blocks. These are instructions that are computed solely for producing the
    // operands of the select instructions in the group and can be sunk without
    // breaking the semantics of the LLVM IR (e.g., cannot sink instructions
    // with side effects).
    SmallVector<std::stack<Instruction *>, 2> TrueSlices, FalseSlices;
    typedef std::stack<Instruction *>::size_type StackSizeType;
    StackSizeType maxTrueSliceLen = 0, maxFalseSliceLen = 0;
    for (SelectInst *SI : ASI) {
      // For each select, compute the sinkable dependence chains of the true and
      // false operands.
      if (auto *TI = dyn_cast<Instruction>(SI->getTrueValue())) {
        std::stack<Instruction *> TrueSlice;
        getExclBackwardsSlice(TI, TrueSlice, true);
        maxTrueSliceLen = std::max(maxTrueSliceLen, TrueSlice.size());
        TrueSlices.push_back(TrueSlice);
      }
      if (auto *FI = dyn_cast<Instruction>(SI->getFalseValue())) {
        std::stack<Instruction *> FalseSlice;
        getExclBackwardsSlice(FI, FalseSlice, true);
        maxFalseSliceLen = std::max(maxFalseSliceLen, FalseSlice.size());
        FalseSlices.push_back(FalseSlice);
      }
    }
    // In the case of multiple select instructions in the same group, the order
    // of non-dependent instructions (instructions of different dependence
    // slices) in the true/false blocks appears to affect performance.
    // Interleaving the slices seems to experimentally be the optimal approach.
    // This interleaving scheduling allows for more ILP (with a natural downside
    // of increasing a bit register pressure) compared to a simple ordering of
    // one whole chain after another. One would expect that this ordering would
    // not matter since the scheduling in the backend of the compiler  would
    // take care of it, but apparently the scheduler fails to deliver optimal
    // ILP with a naive ordering here.
    SmallVector<Instruction *, 2> TrueSlicesInterleaved, FalseSlicesInterleaved;
    for (StackSizeType IS = 0; IS < maxTrueSliceLen; ++IS) {
      for (auto &S : TrueSlices) {
        if (!S.empty()) {
          TrueSlicesInterleaved.push_back(S.top());
          S.pop();
        }
      }
    }
    for (StackSizeType IS = 0; IS < maxFalseSliceLen; ++IS) {
      for (auto &S : FalseSlices) {
        if (!S.empty()) {
          FalseSlicesInterleaved.push_back(S.top());
          S.pop();
        }
      }
    }

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
    // At least one will become an actual new basic block.
    BasicBlock *TrueBlock = nullptr, *FalseBlock = nullptr;
    BranchInst *TrueBranch = nullptr, *FalseBranch = nullptr;
    if (!TrueSlicesInterleaved.empty()) {
      TrueBlock = BasicBlock::Create(LastSI->getContext(), "select.true.sink",
                                     EndBlock->getParent(), EndBlock);
      TrueBranch = BranchInst::Create(EndBlock, TrueBlock);
      TrueBranch->setDebugLoc(LastSI->getDebugLoc());
      for (Instruction *TrueInst : TrueSlicesInterleaved)
        TrueInst->moveBefore(TrueBranch);
    }
    if (!FalseSlicesInterleaved.empty()) {
      FalseBlock = BasicBlock::Create(LastSI->getContext(), "select.false.sink",
                                      EndBlock->getParent(), EndBlock);
      FalseBranch = BranchInst::Create(EndBlock, FalseBlock);
      FalseBranch->setDebugLoc(LastSI->getDebugLoc());
      for (Instruction *FalseInst : FalseSlicesInterleaved)
        FalseInst->moveBefore(FalseBranch);
    }
    // If there was nothing to sink, then arbitrarily choose the 'false' side
    // for a new input value to the PHI.
    if (TrueBlock == FalseBlock) {
      assert(TrueBlock == nullptr &&
             "Unexpected basic block transform while optimizing select");

      FalseBlock = BasicBlock::Create(SI->getContext(), "select.false",
                                      EndBlock->getParent(), EndBlock);
      auto *FalseBranch = BranchInst::Create(EndBlock, FalseBlock);
      FalseBranch->setDebugLoc(SI->getDebugLoc());
    }

    // Insert the real conditional branch based on the original condition.
    // If we did not create a new block for one of the 'true' or 'false' paths
    // of the condition, it means that side of the branch goes to the end block
    // directly and the path originates from the start block from the point of
    // view of the new PHI.
    BasicBlock *TT, *FT;
    if (TrueBlock == nullptr) {
      TT = EndBlock;
      FT = FalseBlock;
      TrueBlock = StartBlock;
    } else if (FalseBlock == nullptr) {
      TT = TrueBlock;
      FT = EndBlock;
      FalseBlock = StartBlock;
    } else {
      TT = TrueBlock;
      FT = FalseBlock;
    }
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

void SelectOptimize::findProfitableSIGroupsInnerLoops(
    const Loop *L, SelectGroups &SIGroups, SelectGroups &ProfSIGroups) {
  NumSelectOptAnalyzed += SIGroups.size();
  // For each select group in an inner-most loop,
  // a branch is more preferable than a select/conditional-move if:
  // i) conversion to branches for all the select groups of the loop satisfies
  //    loop-level heuristics including reducing the loop's critical path by
  //    some threshold (see SelectOptimize::checkLoopHeuristics); and
  // ii) the total cost of the select group is cheaper with a branch compared
  //     to its predicated version. The cost is in terms of latency and the cost
  //     of a select group is the cost of its most expensive select instruction
  //     (assuming infinite resources and thus fully leveraging available ILP).

  DenseMap<const Instruction *, CostInfo> InstCostMap;
  CostInfo LoopCost[2] = {{Scaled64::getZero(), Scaled64::getZero()},
                          {Scaled64::getZero(), Scaled64::getZero()}};
  if (!computeLoopCosts(L, SIGroups, InstCostMap, LoopCost) ||
      !checkLoopHeuristics(L, LoopCost)) {
    return;
  }

  for (SelectGroup &ASI : SIGroups) {
    // Assuming infinite resources, the cost of a group of instructions is the
    // cost of the most expensive instruction of the group.
    Scaled64 SelectCost = Scaled64::getZero(), BranchCost = Scaled64::getZero();
    for (SelectInst *SI : ASI) {
      SelectCost = std::max(SelectCost, InstCostMap[SI].PredCost);
      BranchCost = std::max(BranchCost, InstCostMap[SI].NonPredCost);
    }
    if (BranchCost < SelectCost) {
      OptimizationRemark OR(DEBUG_TYPE, "SelectOpti", ASI.front());
      OR << "Profitable to convert to branch (loop analysis). BranchCost="
         << BranchCost.toString() << ", SelectCost=" << SelectCost.toString()
         << ". ";
      ORE->emit(OR);
      ++NumSelectConvertedLoop;
      ProfSIGroups.push_back(ASI);
    } else {
      OptimizationRemarkMissed ORmiss(DEBUG_TYPE, "SelectOpti", ASI.front());
      ORmiss << "Select is more profitable (loop analysis). BranchCost="
             << BranchCost.toString()
             << ", SelectCost=" << SelectCost.toString() << ". ";
      ORE->emit(ORmiss);
    }
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
      std::stack<Instruction *> ColdSlice;
      getExclBackwardsSlice(ColdI, ColdSlice);
      InstructionCost SliceCost = 0;
      while (!ColdSlice.empty()) {
        SliceCost += TTI->getInstructionCost(ColdSlice.top(),
                                             TargetTransformInfo::TCK_Latency);
        ColdSlice.pop();
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
void SelectOptimize::getExclBackwardsSlice(Instruction *I,
                                           std::stack<Instruction *> &Slice,
                                           bool ForSinking) {
  SmallPtrSet<Instruction *, 2> Visited;
  std::queue<Instruction *> Worklist;
  Worklist.push(I);
  while (!Worklist.empty()) {
    Instruction *II = Worklist.front();
    Worklist.pop();

    // Avoid cycles.
    if (!Visited.insert(II).second)
      continue;

    if (!II->hasOneUse())
      continue;

    // Cannot soundly sink instructions with side-effects.
    // Terminator or phi instructions cannot be sunk.
    // Avoid sinking other select instructions (should be handled separetely).
    if (ForSinking && (II->isTerminator() || II->mayHaveSideEffects() ||
                       isa<SelectInst>(II) || isa<PHINode>(II)))
      continue;

    // Avoid considering instructions with less frequency than the source
    // instruction (i.e., avoid colder code regions of the dependence slice).
    if (BFI->getBlockFreq(II->getParent()) < BFI->getBlockFreq(I->getParent()))
      continue;

    // Eligible one-use instruction added to the dependence slice.
    Slice.push(II);

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

bool SelectOptimize::checkLoopHeuristics(const Loop *L,
                                         const CostInfo LoopCost[2]) {
  // Loop-level checks to determine if a non-predicated version (with branches)
  // of the loop is more profitable than its predicated version.

  if (DisableLoopLevelHeuristics)
    return true;

  OptimizationRemarkMissed ORmissL(DEBUG_TYPE, "SelectOpti",
                                   L->getHeader()->getFirstNonPHI());

  if (LoopCost[0].NonPredCost > LoopCost[0].PredCost ||
      LoopCost[1].NonPredCost >= LoopCost[1].PredCost) {
    ORmissL << "No select conversion in the loop due to no reduction of loop's "
               "critical path. ";
    ORE->emit(ORmissL);
    return false;
  }

  Scaled64 Gain[2] = {LoopCost[0].PredCost - LoopCost[0].NonPredCost,
                      LoopCost[1].PredCost - LoopCost[1].NonPredCost};

  // Profitably converting to branches need to reduce the loop's critical path
  // by at least some threshold (absolute gain of GainCycleThreshold cycles and
  // relative gain of 12.5%).
  if (Gain[1] < Scaled64::get(GainCycleThreshold) ||
      Gain[1] * Scaled64::get(GainRelativeThreshold) < LoopCost[1].PredCost) {
    Scaled64 RelativeGain = Scaled64::get(100) * Gain[1] / LoopCost[1].PredCost;
    ORmissL << "No select conversion in the loop due to small reduction of "
               "loop's critical path. Gain="
            << Gain[1].toString()
            << ", RelativeGain=" << RelativeGain.toString() << "%. ";
    ORE->emit(ORmissL);
    return false;
  }

  // If the loop's critical path involves loop-carried dependences, the gradient
  // of the gain needs to be at least GainGradientThreshold% (defaults to 25%).
  // This check ensures that the latency reduction for the loop's critical path
  // keeps decreasing with sufficient rate beyond the two analyzed loop
  // iterations.
  if (Gain[1] > Gain[0]) {
    Scaled64 GradientGain = Scaled64::get(100) * (Gain[1] - Gain[0]) /
                            (LoopCost[1].PredCost - LoopCost[0].PredCost);
    if (GradientGain < Scaled64::get(GainGradientThreshold)) {
      ORmissL << "No select conversion in the loop due to small gradient gain. "
                 "GradientGain="
              << GradientGain.toString() << "%. ";
      ORE->emit(ORmissL);
      return false;
    }
  }
  // If the gain decreases it is not profitable to convert.
  else if (Gain[1] < Gain[0]) {
    ORmissL
        << "No select conversion in the loop due to negative gradient gain. ";
    ORE->emit(ORmissL);
    return false;
  }

  // Non-predicated version of the loop is more profitable than its
  // predicated version.
  return true;
}

// Computes instruction and loop-critical-path costs for both the predicated
// and non-predicated version of the given loop.
// Returns false if unable to compute these costs due to invalid cost of loop
// instruction(s).
bool SelectOptimize::computeLoopCosts(
    const Loop *L, const SelectGroups &SIGroups,
    DenseMap<const Instruction *, CostInfo> &InstCostMap, CostInfo *LoopCost) {
  const auto &SIset = getSIset(SIGroups);
  // Compute instruction and loop-critical-path costs across two iterations for
  // both predicated and non-predicated version.
  const unsigned Iterations = 2;
  for (unsigned Iter = 0; Iter < Iterations; ++Iter) {
    // Cost of the loop's critical path.
    CostInfo &MaxCost = LoopCost[Iter];
    for (BasicBlock *BB : L->getBlocks()) {
      for (const Instruction &I : *BB) {
        if (I.isDebugOrPseudoInst())
          continue;
        // Compute the predicated and non-predicated cost of the instruction.
        Scaled64 IPredCost = Scaled64::getZero(),
                 INonPredCost = Scaled64::getZero();

        // Assume infinite resources that allow to fully exploit the available
        // instruction-level parallelism.
        // InstCost = InstLatency + max(Op1Cost, Op2Cost, … OpNCost)
        for (const Use &U : I.operands()) {
          auto UI = dyn_cast<Instruction>(U.get());
          if (!UI)
            continue;
          if (InstCostMap.count(UI)) {
            IPredCost = std::max(IPredCost, InstCostMap[UI].PredCost);
            INonPredCost = std::max(INonPredCost, InstCostMap[UI].NonPredCost);
          }
        }
        auto ILatency = computeInstCost(&I);
        if (!ILatency) {
          OptimizationRemarkMissed ORmissL(DEBUG_TYPE, "SelectOpti", &I);
          ORmissL << "Invalid instruction cost preventing analysis and "
                     "optimization of the inner-most loop containing this "
                     "instruction. ";
          ORE->emit(ORmissL);
          return false;
        }
        IPredCost += Scaled64::get(*ILatency);
        INonPredCost += Scaled64::get(*ILatency);

        // For a select that can be converted to branch,
        // compute its cost as a branch (non-predicated cost).
        //
        // BranchCost = PredictedPathCost + MispredictCost
        // PredictedPathCost = TrueOpCost * TrueProb + FalseOpCost * FalseProb
        // MispredictCost = max(MispredictPenalty, CondCost) * MispredictRate
        if (SIset.contains(&I)) {
          auto SI = dyn_cast<SelectInst>(&I);

          Scaled64 TrueOpCost = Scaled64::getZero(),
                   FalseOpCost = Scaled64::getZero();
          if (auto *TI = dyn_cast<Instruction>(SI->getTrueValue()))
            if (InstCostMap.count(TI))
              TrueOpCost = InstCostMap[TI].NonPredCost;
          if (auto *FI = dyn_cast<Instruction>(SI->getFalseValue()))
            if (InstCostMap.count(FI))
              FalseOpCost = InstCostMap[FI].NonPredCost;
          Scaled64 PredictedPathCost =
              getPredictedPathCost(TrueOpCost, FalseOpCost, SI);

          Scaled64 CondCost = Scaled64::getZero();
          if (auto *CI = dyn_cast<Instruction>(SI->getCondition()))
            if (InstCostMap.count(CI))
              CondCost = InstCostMap[CI].NonPredCost;
          Scaled64 MispredictCost = getMispredictionCost(SI, CondCost);

          INonPredCost = PredictedPathCost + MispredictCost;
        }

        InstCostMap[&I] = {IPredCost, INonPredCost};
        MaxCost.PredCost = std::max(MaxCost.PredCost, IPredCost);
        MaxCost.NonPredCost = std::max(MaxCost.NonPredCost, INonPredCost);
      }
    }
  }
  return true;
}

SmallPtrSet<const Instruction *, 2>
SelectOptimize::getSIset(const SelectGroups &SIGroups) {
  SmallPtrSet<const Instruction *, 2> SIset;
  for (const SelectGroup &ASI : SIGroups)
    for (const SelectInst *SI : ASI)
      SIset.insert(SI);
  return SIset;
}

Optional<uint64_t> SelectOptimize::computeInstCost(const Instruction *I) {
  InstructionCost ICost =
      TTI->getInstructionCost(I, TargetTransformInfo::TCK_Latency);
  if (auto OC = ICost.getValue())
    return Optional<uint64_t>(*OC);
  return Optional<uint64_t>(None);
}

ScaledNumber<uint64_t>
SelectOptimize::getMispredictionCost(const SelectInst *SI,
                                     const Scaled64 CondCost) {
  uint64_t MispredictPenalty = TSchedModel.getMCSchedModel()->MispredictPenalty;

  // Account for the default misprediction rate when using a branch
  // (conservatively set to 25% by default).
  uint64_t MispredictRate = MispredictDefaultRate;
  // If the select condition is obviously predictable, then the misprediction
  // rate is zero.
  if (isSelectHighlyPredictable(SI))
    MispredictRate = 0;

  // CondCost is included to account for cases where the computation of the
  // condition is part of a long dependence chain (potentially loop-carried)
  // that would delay detection of a misprediction and increase its cost.
  Scaled64 MispredictCost =
      std::max(Scaled64::get(MispredictPenalty), CondCost) *
      Scaled64::get(MispredictRate);
  MispredictCost /= Scaled64::get(100);

  return MispredictCost;
}

// Returns the cost of a branch when the prediction is correct.
// TrueCost * TrueProbability + FalseCost * FalseProbability.
ScaledNumber<uint64_t>
SelectOptimize::getPredictedPathCost(Scaled64 TrueCost, Scaled64 FalseCost,
                                     const SelectInst *SI) {
  Scaled64 PredPathCost;
  uint64_t TrueWeight, FalseWeight;
  if (SI->extractProfMetadata(TrueWeight, FalseWeight)) {
    uint64_t SumWeight = TrueWeight + FalseWeight;
    if (SumWeight != 0) {
      PredPathCost = TrueCost * Scaled64::get(TrueWeight) +
                     FalseCost * Scaled64::get(FalseWeight);
      PredPathCost /= Scaled64::get(SumWeight);
      return PredPathCost;
    }
  }
  // Without branch weight metadata, we assume 75% for the one path and 25% for
  // the other, and pick the result with the biggest cost.
  PredPathCost = std::max(TrueCost * Scaled64::get(3) + FalseCost,
                          FalseCost * Scaled64::get(3) + TrueCost);
  PredPathCost /= Scaled64::get(4);
  return PredPathCost;
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
