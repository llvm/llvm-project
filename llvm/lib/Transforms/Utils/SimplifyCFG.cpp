//===- SimplifyCFG.cpp - Code to perform CFG simplification ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Peephole optimize the CFG.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/GuardUtils.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/MemoryModelRelaxationAnnotations.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/ProfDataUtils.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LockstepReverseIterator.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <algorithm>
#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "simplifycfg"

cl::opt<bool> llvm::RequireAndPreserveDomTree(
    "simplifycfg-require-and-preserve-domtree", cl::Hidden,

    cl::desc(
        "Temporary development switch used to gradually uplift SimplifyCFG "
        "into preserving DomTree,"));

// Chosen as 2 so as to be cheap, but still to have enough power to fold
// a select, so the "clamp" idiom (of a min followed by a max) will be caught.
// To catch this, we need to fold a compare and a select, hence '2' being the
// minimum reasonable default.
static cl::opt<unsigned> PHINodeFoldingThreshold(
    "phi-node-folding-threshold", cl::Hidden, cl::init(2),
    cl::desc(
        "Control the amount of phi node folding to perform (default = 2)"));

static cl::opt<unsigned> TwoEntryPHINodeFoldingThreshold(
    "two-entry-phi-node-folding-threshold", cl::Hidden, cl::init(4),
    cl::desc("Control the maximal total instruction cost that we are willing "
             "to speculatively execute to fold a 2-entry PHI node into a "
             "select (default = 4)"));

static cl::opt<bool>
    HoistCommon("simplifycfg-hoist-common", cl::Hidden, cl::init(true),
                cl::desc("Hoist common instructions up to the parent block"));

static cl::opt<bool> HoistLoadsWithCondFaulting(
    "simplifycfg-hoist-loads-with-cond-faulting", cl::Hidden, cl::init(true),
    cl::desc("Hoist loads if the target supports conditional faulting"));

static cl::opt<bool> HoistStoresWithCondFaulting(
    "simplifycfg-hoist-stores-with-cond-faulting", cl::Hidden, cl::init(true),
    cl::desc("Hoist stores if the target supports conditional faulting"));

static cl::opt<unsigned> HoistLoadsStoresWithCondFaultingThreshold(
    "hoist-loads-stores-with-cond-faulting-threshold", cl::Hidden, cl::init(6),
    cl::desc("Control the maximal conditional load/store that we are willing "
             "to speculatively execute to eliminate conditional branch "
             "(default = 6)"));

static cl::opt<unsigned>
    HoistCommonSkipLimit("simplifycfg-hoist-common-skip-limit", cl::Hidden,
                         cl::init(20),
                         cl::desc("Allow reordering across at most this many "
                                  "instructions when hoisting"));

static cl::opt<bool>
    SinkCommon("simplifycfg-sink-common", cl::Hidden, cl::init(true),
               cl::desc("Sink common instructions down to the end block"));

static cl::opt<bool> HoistCondStores(
    "simplifycfg-hoist-cond-stores", cl::Hidden, cl::init(true),
    cl::desc("Hoist conditional stores if an unconditional store precedes"));

static cl::opt<bool> MergeCondStores(
    "simplifycfg-merge-cond-stores", cl::Hidden, cl::init(true),
    cl::desc("Hoist conditional stores even if an unconditional store does not "
             "precede - hoist multiple conditional stores into a single "
             "predicated store"));

static cl::opt<bool> MergeCondStoresAggressively(
    "simplifycfg-merge-cond-stores-aggressively", cl::Hidden, cl::init(false),
    cl::desc("When merging conditional stores, do so even if the resultant "
             "basic blocks are unlikely to be if-converted as a result"));

static cl::opt<bool> SpeculateOneExpensiveInst(
    "speculate-one-expensive-inst", cl::Hidden, cl::init(true),
    cl::desc("Allow exactly one expensive instruction to be speculatively "
             "executed"));

static cl::opt<unsigned> MaxSpeculationDepth(
    "max-speculation-depth", cl::Hidden, cl::init(10),
    cl::desc("Limit maximum recursion depth when calculating costs of "
             "speculatively executed instructions"));

static cl::opt<int>
    MaxSmallBlockSize("simplifycfg-max-small-block-size", cl::Hidden,
                      cl::init(10),
                      cl::desc("Max size of a block which is still considered "
                               "small enough to thread through"));

// Two is chosen to allow one negation and a logical combine.
static cl::opt<unsigned>
    BranchFoldThreshold("simplifycfg-branch-fold-threshold", cl::Hidden,
                        cl::init(2),
                        cl::desc("Maximum cost of combining conditions when "
                                 "folding branches"));

static cl::opt<unsigned> BranchFoldToCommonDestVectorMultiplier(
    "simplifycfg-branch-fold-common-dest-vector-multiplier", cl::Hidden,
    cl::init(2),
    cl::desc("Multiplier to apply to threshold when determining whether or not "
             "to fold branch to common destination when vector operations are "
             "present"));

static cl::opt<bool> EnableMergeCompatibleInvokes(
    "simplifycfg-merge-compatible-invokes", cl::Hidden, cl::init(true),
    cl::desc("Allow SimplifyCFG to merge invokes together when appropriate"));

static cl::opt<unsigned> MaxSwitchCasesPerResult(
    "max-switch-cases-per-result", cl::Hidden, cl::init(16),
    cl::desc("Limit cases to analyze when converting a switch to select"));

static cl::opt<unsigned> MaxJumpThreadingLiveBlocks(
    "max-jump-threading-live-blocks", cl::Hidden, cl::init(24),
    cl::desc("Limit number of blocks a define in a threaded block is allowed "
             "to be live in"));

STATISTIC(NumBitMaps, "Number of switch instructions turned into bitmaps");
STATISTIC(NumLinearMaps,
          "Number of switch instructions turned into linear mapping");
STATISTIC(NumLookupTables,
          "Number of switch instructions turned into lookup tables");
STATISTIC(
    NumLookupTablesHoles,
    "Number of switch instructions turned into lookup tables (holes checked)");
STATISTIC(NumTableCmpReuses, "Number of reused switch table lookup compares");
STATISTIC(NumFoldValueComparisonIntoPredecessors,
          "Number of value comparisons folded into predecessor basic blocks");
STATISTIC(NumFoldBranchToCommonDest,
          "Number of branches folded into predecessor basic block");
STATISTIC(
    NumHoistCommonCode,
    "Number of common instruction 'blocks' hoisted up to the begin block");
STATISTIC(NumHoistCommonInstrs,
          "Number of common instructions hoisted up to the begin block");
STATISTIC(NumSinkCommonCode,
          "Number of common instruction 'blocks' sunk down to the end block");
STATISTIC(NumSinkCommonInstrs,
          "Number of common instructions sunk down to the end block");
STATISTIC(NumSpeculations, "Number of speculative executed instructions");
STATISTIC(NumInvokes,
          "Number of invokes with empty resume blocks simplified into calls");
STATISTIC(NumInvokesMerged, "Number of invokes that were merged together");
STATISTIC(NumInvokeSetsFormed, "Number of invoke sets that were formed");

namespace {

// The first field contains the value that the switch produces when a certain
// case group is selected, and the second field is a vector containing the
// cases composing the case group.
using SwitchCaseResultVectorTy =
    SmallVector<std::pair<Constant *, SmallVector<ConstantInt *, 4>>, 2>;

// The first field contains the phi node that generates a result of the switch
// and the second field contains the value generated for a certain case in the
// switch for that PHI.
using SwitchCaseResultsTy = SmallVector<std::pair<PHINode *, Constant *>, 4>;

/// ValueEqualityComparisonCase - Represents a case of a switch.
struct ValueEqualityComparisonCase {
  ConstantInt *Value;
  BasicBlock *Dest;

  ValueEqualityComparisonCase(ConstantInt *Value, BasicBlock *Dest)
      : Value(Value), Dest(Dest) {}

  bool operator<(ValueEqualityComparisonCase RHS) const {
    // Comparing pointers is ok as we only rely on the order for uniquing.
    return Value < RHS.Value;
  }

  bool operator==(BasicBlock *RHSDest) const { return Dest == RHSDest; }
};

class SimplifyCFGOpt {
  const TargetTransformInfo &TTI;
  DomTreeUpdater *DTU;
  const DataLayout &DL;
  ArrayRef<WeakVH> LoopHeaders;
  const SimplifyCFGOptions &Options;
  bool Resimplify;

  Value *isValueEqualityComparison(Instruction *TI);
  BasicBlock *getValueEqualityComparisonCases(
      Instruction *TI, std::vector<ValueEqualityComparisonCase> &Cases);
  bool simplifyEqualityComparisonWithOnlyPredecessor(Instruction *TI,
                                                     BasicBlock *Pred,
                                                     IRBuilder<> &Builder);
  bool performValueComparisonIntoPredecessorFolding(Instruction *TI, Value *&CV,
                                                    Instruction *PTI,
                                                    IRBuilder<> &Builder);
  bool foldValueComparisonIntoPredecessors(Instruction *TI,
                                           IRBuilder<> &Builder);

  bool simplifyResume(ResumeInst *RI, IRBuilder<> &Builder);
  bool simplifySingleResume(ResumeInst *RI);
  bool simplifyCommonResume(ResumeInst *RI);
  bool simplifyCleanupReturn(CleanupReturnInst *RI);
  bool simplifyUnreachable(UnreachableInst *UI);
  bool simplifySwitch(SwitchInst *SI, IRBuilder<> &Builder);
  bool simplifyDuplicateSwitchArms(SwitchInst *SI, DomTreeUpdater *DTU);
  bool simplifyIndirectBr(IndirectBrInst *IBI);
  bool simplifyBranch(BranchInst *Branch, IRBuilder<> &Builder);
  bool simplifyUncondBranch(BranchInst *BI, IRBuilder<> &Builder);
  bool simplifyCondBranch(BranchInst *BI, IRBuilder<> &Builder);
  bool foldCondBranchOnValueKnownInPredecessor(BranchInst *BI);

  bool tryToSimplifyUncondBranchWithICmpInIt(ICmpInst *ICI,
                                             IRBuilder<> &Builder);

  bool hoistCommonCodeFromSuccessors(Instruction *TI, bool AllInstsEqOnly);
  bool hoistSuccIdenticalTerminatorToSwitchOrIf(
      Instruction *TI, Instruction *I1,
      SmallVectorImpl<Instruction *> &OtherSuccTIs);
  bool speculativelyExecuteBB(BranchInst *BI, BasicBlock *ThenBB);
  bool simplifyTerminatorOnSelect(Instruction *OldTerm, Value *Cond,
                                  BasicBlock *TrueBB, BasicBlock *FalseBB,
                                  uint32_t TrueWeight, uint32_t FalseWeight);
  bool simplifyBranchOnICmpChain(BranchInst *BI, IRBuilder<> &Builder,
                                 const DataLayout &DL);
  bool simplifySwitchOnSelect(SwitchInst *SI, SelectInst *Select);
  bool simplifyIndirectBrOnSelect(IndirectBrInst *IBI, SelectInst *SI);
  bool turnSwitchRangeIntoICmp(SwitchInst *SI, IRBuilder<> &Builder);

public:
  SimplifyCFGOpt(const TargetTransformInfo &TTI, DomTreeUpdater *DTU,
                 const DataLayout &DL, ArrayRef<WeakVH> LoopHeaders,
                 const SimplifyCFGOptions &Opts)
      : TTI(TTI), DTU(DTU), DL(DL), LoopHeaders(LoopHeaders), Options(Opts) {
    assert((!DTU || !DTU->hasPostDomTree()) &&
           "SimplifyCFG is not yet capable of maintaining validity of a "
           "PostDomTree, so don't ask for it.");
  }

  bool simplifyOnce(BasicBlock *BB);
  bool run(BasicBlock *BB);

  // Helper to set Resimplify and return change indication.
  bool requestResimplify() {
    Resimplify = true;
    return true;
  }
};

} // end anonymous namespace

/// Return true if all the PHI nodes in the basic block \p BB
/// receive compatible (identical) incoming values when coming from
/// all of the predecessor blocks that are specified in \p IncomingBlocks.
///
/// Note that if the values aren't exactly identical, but \p EquivalenceSet
/// is provided, and *both* of the values are present in the set,
/// then they are considered equal.
static bool incomingValuesAreCompatible(
    BasicBlock *BB, ArrayRef<BasicBlock *> IncomingBlocks,
    SmallPtrSetImpl<Value *> *EquivalenceSet = nullptr) {
  assert(IncomingBlocks.size() == 2 &&
         "Only for a pair of incoming blocks at the time!");

  // FIXME: it is okay if one of the incoming values is an `undef` value,
  //        iff the other incoming value is guaranteed to be a non-poison value.
  // FIXME: it is okay if one of the incoming values is a `poison` value.
  return all_of(BB->phis(), [IncomingBlocks, EquivalenceSet](PHINode &PN) {
    Value *IV0 = PN.getIncomingValueForBlock(IncomingBlocks[0]);
    Value *IV1 = PN.getIncomingValueForBlock(IncomingBlocks[1]);
    if (IV0 == IV1)
      return true;
    if (EquivalenceSet && EquivalenceSet->contains(IV0) &&
        EquivalenceSet->contains(IV1))
      return true;
    return false;
  });
}

/// Return true if it is safe to merge these two
/// terminator instructions together.
static bool
safeToMergeTerminators(Instruction *SI1, Instruction *SI2,
                       SmallSetVector<BasicBlock *, 4> *FailBlocks = nullptr) {
  if (SI1 == SI2)
    return false; // Can't merge with self!

  // It is not safe to merge these two switch instructions if they have a common
  // successor, and if that successor has a PHI node, and if *that* PHI node has
  // conflicting incoming values from the two switch blocks.
  BasicBlock *SI1BB = SI1->getParent();
  BasicBlock *SI2BB = SI2->getParent();

  SmallPtrSet<BasicBlock *, 16> SI1Succs(llvm::from_range, successors(SI1BB));
  bool Fail = false;
  for (BasicBlock *Succ : successors(SI2BB)) {
    if (!SI1Succs.count(Succ))
      continue;
    if (incomingValuesAreCompatible(Succ, {SI1BB, SI2BB}))
      continue;
    Fail = true;
    if (FailBlocks)
      FailBlocks->insert(Succ);
    else
      break;
  }

  return !Fail;
}

/// Update PHI nodes in Succ to indicate that there will now be entries in it
/// from the 'NewPred' block. The values that will be flowing into the PHI nodes
/// will be the same as those coming in from ExistPred, an existing predecessor
/// of Succ.
static void addPredecessorToBlock(BasicBlock *Succ, BasicBlock *NewPred,
                                  BasicBlock *ExistPred,
                                  MemorySSAUpdater *MSSAU = nullptr) {
  for (PHINode &PN : Succ->phis())
    PN.addIncoming(PN.getIncomingValueForBlock(ExistPred), NewPred);
  if (MSSAU)
    if (auto *MPhi = MSSAU->getMemorySSA()->getMemoryAccess(Succ))
      MPhi->addIncoming(MPhi->getIncomingValueForBlock(ExistPred), NewPred);
}

/// Compute an abstract "cost" of speculating the given instruction,
/// which is assumed to be safe to speculate. TCC_Free means cheap,
/// TCC_Basic means less cheap, and TCC_Expensive means prohibitively
/// expensive.
static InstructionCost computeSpeculationCost(const User *I,
                                              const TargetTransformInfo &TTI) {
  return TTI.getInstructionCost(I, TargetTransformInfo::TCK_SizeAndLatency);
}

/// If we have a merge point of an "if condition" as accepted above,
/// return true if the specified value dominates the block.  We don't handle
/// the true generality of domination here, just a special case which works
/// well enough for us.
///
/// If AggressiveInsts is non-null, and if V does not dominate BB, we check to
/// see if V (which must be an instruction) and its recursive operands
/// that do not dominate BB have a combined cost lower than Budget and
/// are non-trapping.  If both are true, the instruction is inserted into the
/// set and true is returned.
///
/// The cost for most non-trapping instructions is defined as 1 except for
/// Select whose cost is 2.
///
/// After this function returns, Cost is increased by the cost of
/// V plus its non-dominating operands.  If that cost is greater than
/// Budget, false is returned and Cost is undefined.
static bool dominatesMergePoint(
    Value *V, BasicBlock *BB, Instruction *InsertPt,
    SmallPtrSetImpl<Instruction *> &AggressiveInsts, InstructionCost &Cost,
    InstructionCost Budget, const TargetTransformInfo &TTI, AssumptionCache *AC,
    SmallPtrSetImpl<Instruction *> &ZeroCostInstructions, unsigned Depth = 0) {
  // It is possible to hit a zero-cost cycle (phi/gep instructions for example),
  // so limit the recursion depth.
  // TODO: While this recursion limit does prevent pathological behavior, it
  // would be better to track visited instructions to avoid cycles.
  if (Depth == MaxSpeculationDepth)
    return false;

  Instruction *I = dyn_cast<Instruction>(V);
  if (!I) {
    // Non-instructions dominate all instructions and can be executed
    // unconditionally.
    return true;
  }
  BasicBlock *PBB = I->getParent();

  // We don't want to allow weird loops that might have the "if condition" in
  // the bottom of this block.
  if (PBB == BB)
    return false;

  // If this instruction is defined in a block that contains an unconditional
  // branch to BB, then it must be in the 'conditional' part of the "if
  // statement".  If not, it definitely dominates the region.
  BranchInst *BI = dyn_cast<BranchInst>(PBB->getTerminator());
  if (!BI || BI->isConditional() || BI->getSuccessor(0) != BB)
    return true;

  // If we have seen this instruction before, don't count it again.
  if (AggressiveInsts.count(I))
    return true;

  // Okay, it looks like the instruction IS in the "condition".  Check to
  // see if it's a cheap instruction to unconditionally compute, and if it
  // only uses stuff defined outside of the condition.  If so, hoist it out.
  if (!isSafeToSpeculativelyExecute(I, InsertPt, AC))
    return false;

  // Overflow arithmetic instruction plus extract value are usually generated
  // when a division is being replaced. But, in this case, the zero check may
  // still be kept in the code. In that case it would be worth to hoist these
  // two instruction out of the basic block. Let's treat this pattern as one
  // single cheap instruction here!
  WithOverflowInst *OverflowInst;
  if (match(I, m_ExtractValue<1>(m_OneUse(m_WithOverflowInst(OverflowInst))))) {
    ZeroCostInstructions.insert(OverflowInst);
    Cost += 1;
  } else if (!ZeroCostInstructions.contains(I))
    Cost += computeSpeculationCost(I, TTI);

  // Allow exactly one instruction to be speculated regardless of its cost
  // (as long as it is safe to do so).
  // This is intended to flatten the CFG even if the instruction is a division
  // or other expensive operation. The speculation of an expensive instruction
  // is expected to be undone in CodeGenPrepare if the speculation has not
  // enabled further IR optimizations.
  if (Cost > Budget &&
      (!SpeculateOneExpensiveInst || !AggressiveInsts.empty() || Depth > 0 ||
       !Cost.isValid()))
    return false;

  // Okay, we can only really hoist these out if their operands do
  // not take us over the cost threshold.
  for (Use &Op : I->operands())
    if (!dominatesMergePoint(Op, BB, InsertPt, AggressiveInsts, Cost, Budget,
                             TTI, AC, ZeroCostInstructions, Depth + 1))
      return false;
  // Okay, it's safe to do this!  Remember this instruction.
  AggressiveInsts.insert(I);
  return true;
}

/// Extract ConstantInt from value, looking through IntToPtr
/// and PointerNullValue. Return NULL if value is not a constant int.
static ConstantInt *getConstantInt(Value *V, const DataLayout &DL) {
  // Normal constant int.
  ConstantInt *CI = dyn_cast<ConstantInt>(V);
  if (CI || !isa<Constant>(V) || !V->getType()->isPointerTy() ||
      DL.isNonIntegralPointerType(V->getType()))
    return CI;

  // This is some kind of pointer constant. Turn it into a pointer-sized
  // ConstantInt if possible.
  IntegerType *PtrTy = cast<IntegerType>(DL.getIntPtrType(V->getType()));

  // Null pointer means 0, see SelectionDAGBuilder::getValue(const Value*).
  if (isa<ConstantPointerNull>(V))
    return ConstantInt::get(PtrTy, 0);

  // IntToPtr const int.
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V))
    if (CE->getOpcode() == Instruction::IntToPtr)
      if (ConstantInt *CI = dyn_cast<ConstantInt>(CE->getOperand(0))) {
        // The constant is very likely to have the right type already.
        if (CI->getType() == PtrTy)
          return CI;
        else
          return cast<ConstantInt>(
              ConstantFoldIntegerCast(CI, PtrTy, /*isSigned=*/false, DL));
      }
  return nullptr;
}

namespace {

/// Given a chain of or (||) or and (&&) comparison of a value against a
/// constant, this will try to recover the information required for a switch
/// structure.
/// It will depth-first traverse the chain of comparison, seeking for patterns
/// like %a == 12 or %a < 4 and combine them to produce a set of integer
/// representing the different cases for the switch.
/// Note that if the chain is composed of '||' it will build the set of elements
/// that matches the comparisons (i.e. any of this value validate the chain)
/// while for a chain of '&&' it will build the set elements that make the test
/// fail.
struct ConstantComparesGatherer {
  const DataLayout &DL;

  /// Value found for the switch comparison
  Value *CompValue = nullptr;

  /// Extra clause to be checked before the switch
  Value *Extra = nullptr;

  /// Set of integers to match in switch
  SmallVector<ConstantInt *, 8> Vals;

  /// Number of comparisons matched in the and/or chain
  unsigned UsedICmps = 0;

  /// If the elements in Vals matches the comparisons
  bool IsEq = false;

  // Used to check if the first matched CompValue shall be the Extra check.
  bool IgnoreFirstMatch = false;
  bool MultipleMatches = false;

  /// Construct and compute the result for the comparison instruction Cond
  ConstantComparesGatherer(Instruction *Cond, const DataLayout &DL) : DL(DL) {
    gather(Cond);
    if (CompValue || !MultipleMatches)
      return;
    Extra = nullptr;
    Vals.clear();
    UsedICmps = 0;
    IgnoreFirstMatch = true;
    gather(Cond);
  }

  ConstantComparesGatherer(const ConstantComparesGatherer &) = delete;
  ConstantComparesGatherer &
  operator=(const ConstantComparesGatherer &) = delete;

private:
  /// Try to set the current value used for the comparison, it succeeds only if
  /// it wasn't set before or if the new value is the same as the old one
  bool setValueOnce(Value *NewVal) {
    if (IgnoreFirstMatch) {
      IgnoreFirstMatch = false;
      return false;
    }
    if (CompValue && CompValue != NewVal) {
      MultipleMatches = true;
      return false;
    }
    CompValue = NewVal;
    return true;
  }

  /// Try to match Instruction "I" as a comparison against a constant and
  /// populates the array Vals with the set of values that match (or do not
  /// match depending on isEQ).
  /// Return false on failure. On success, the Value the comparison matched
  /// against is placed in CompValue.
  /// If CompValue is already set, the function is expected to fail if a match
  /// is found but the value compared to is different.
  bool matchInstruction(Instruction *I, bool isEQ) {
    Value *Val;
    if (match(I, m_NUWTrunc(m_Value(Val)))) {
      // If we already have a value for the switch, it has to match!
      if (!setValueOnce(Val))
        return false;
      UsedICmps++;
      Vals.push_back(ConstantInt::get(cast<IntegerType>(Val->getType()), isEQ));
      return true;
    }
    // If this is an icmp against a constant, handle this as one of the cases.
    ICmpInst *ICI;
    ConstantInt *C;
    if (!((ICI = dyn_cast<ICmpInst>(I)) &&
          (C = getConstantInt(I->getOperand(1), DL)))) {
      return false;
    }

    Value *RHSVal;
    const APInt *RHSC;

    // Pattern match a special case
    // (x & ~2^z) == y --> x == y || x == y|2^z
    // This undoes a transformation done by instcombine to fuse 2 compares.
    if (ICI->getPredicate() == (isEQ ? ICmpInst::ICMP_EQ : ICmpInst::ICMP_NE)) {
      // It's a little bit hard to see why the following transformations are
      // correct. Here is a CVC3 program to verify them for 64-bit values:

      /*
         ONE  : BITVECTOR(64) = BVZEROEXTEND(0bin1, 63);
         x    : BITVECTOR(64);
         y    : BITVECTOR(64);
         z    : BITVECTOR(64);
         mask : BITVECTOR(64) = BVSHL(ONE, z);
         QUERY( (y & ~mask = y) =>
                ((x & ~mask = y) <=> (x = y OR x = (y |  mask)))
         );
         QUERY( (y |  mask = y) =>
                ((x |  mask = y) <=> (x = y OR x = (y & ~mask)))
         );
      */

      // Please note that each pattern must be a dual implication (<--> or
      // iff). One directional implication can create spurious matches. If the
      // implication is only one-way, an unsatisfiable condition on the left
      // side can imply a satisfiable condition on the right side. Dual
      // implication ensures that satisfiable conditions are transformed to
      // other satisfiable conditions and unsatisfiable conditions are
      // transformed to other unsatisfiable conditions.

      // Here is a concrete example of a unsatisfiable condition on the left
      // implying a satisfiable condition on the right:
      //
      // mask = (1 << z)
      // (x & ~mask) == y  --> (x == y || x == (y | mask))
      //
      // Substituting y = 3, z = 0 yields:
      // (x & -2) == 3 --> (x == 3 || x == 2)

      // Pattern match a special case:
      /*
        QUERY( (y & ~mask = y) =>
               ((x & ~mask = y) <=> (x = y OR x = (y |  mask)))
        );
      */
      if (match(ICI->getOperand(0),
                m_And(m_Value(RHSVal), m_APInt(RHSC)))) {
        APInt Mask = ~*RHSC;
        if (Mask.isPowerOf2() && (C->getValue() & ~Mask) == C->getValue()) {
          // If we already have a value for the switch, it has to match!
          if (!setValueOnce(RHSVal))
            return false;

          Vals.push_back(C);
          Vals.push_back(
              ConstantInt::get(C->getContext(),
                               C->getValue() | Mask));
          UsedICmps++;
          return true;
        }
      }

      // Pattern match a special case:
      /*
        QUERY( (y |  mask = y) =>
               ((x |  mask = y) <=> (x = y OR x = (y & ~mask)))
        );
      */
      if (match(ICI->getOperand(0),
                m_Or(m_Value(RHSVal), m_APInt(RHSC)))) {
        APInt Mask = *RHSC;
        if (Mask.isPowerOf2() && (C->getValue() | Mask) == C->getValue()) {
          // If we already have a value for the switch, it has to match!
          if (!setValueOnce(RHSVal))
            return false;

          Vals.push_back(C);
          Vals.push_back(ConstantInt::get(C->getContext(),
                                          C->getValue() & ~Mask));
          UsedICmps++;
          return true;
        }
      }

      // If we already have a value for the switch, it has to match!
      if (!setValueOnce(ICI->getOperand(0)))
        return false;

      UsedICmps++;
      Vals.push_back(C);
      return true;
    }

    // If we have "x ult 3", for example, then we can add 0,1,2 to the set.
    ConstantRange Span =
        ConstantRange::makeExactICmpRegion(ICI->getPredicate(), C->getValue());

    // Shift the range if the compare is fed by an add. This is the range
    // compare idiom as emitted by instcombine.
    Value *CandidateVal = I->getOperand(0);
    if (match(I->getOperand(0), m_Add(m_Value(RHSVal), m_APInt(RHSC)))) {
      Span = Span.subtract(*RHSC);
      CandidateVal = RHSVal;
    }

    // If this is an and/!= check, then we are looking to build the set of
    // value that *don't* pass the and chain. I.e. to turn "x ugt 2" into
    // x != 0 && x != 1.
    if (!isEQ)
      Span = Span.inverse();

    // If there are a ton of values, we don't want to make a ginormous switch.
    if (Span.isSizeLargerThan(8) || Span.isEmptySet()) {
      return false;
    }

    // If we already have a value for the switch, it has to match!
    if (!setValueOnce(CandidateVal))
      return false;

    // Add all values from the range to the set
    for (APInt Tmp = Span.getLower(); Tmp != Span.getUpper(); ++Tmp)
      Vals.push_back(ConstantInt::get(I->getContext(), Tmp));

    UsedICmps++;
    return true;
  }

  /// Given a potentially 'or'd or 'and'd together collection of icmp
  /// eq/ne/lt/gt instructions that compare a value against a constant, extract
  /// the value being compared, and stick the list constants into the Vals
  /// vector.
  /// One "Extra" case is allowed to differ from the other.
  void gather(Value *V) {
    Value *Op0, *Op1;
    if (match(V, m_LogicalOr(m_Value(Op0), m_Value(Op1))))
      IsEq = true;
    else if (match(V, m_LogicalAnd(m_Value(Op0), m_Value(Op1))))
      IsEq = false;
    else
      return;
    // Keep a stack (SmallVector for efficiency) for depth-first traversal
    SmallVector<Value *, 8> DFT{Op0, Op1};
    SmallPtrSet<Value *, 8> Visited{V, Op0, Op1};

    while (!DFT.empty()) {
      V = DFT.pop_back_val();

      if (Instruction *I = dyn_cast<Instruction>(V)) {
        // If it is a || (or && depending on isEQ), process the operands.
        if (IsEq ? match(I, m_LogicalOr(m_Value(Op0), m_Value(Op1)))
                 : match(I, m_LogicalAnd(m_Value(Op0), m_Value(Op1)))) {
          if (Visited.insert(Op1).second)
            DFT.push_back(Op1);
          if (Visited.insert(Op0).second)
            DFT.push_back(Op0);

          continue;
        }

        // Try to match the current instruction
        if (matchInstruction(I, IsEq))
          // Match succeed, continue the loop
          continue;
      }

      // One element of the sequence of || (or &&) could not be match as a
      // comparison against the same value as the others.
      // We allow only one "Extra" case to be checked before the switch
      if (!Extra) {
        Extra = V;
        continue;
      }
      // Failed to parse a proper sequence, abort now
      CompValue = nullptr;
      break;
    }
  }
};

} // end anonymous namespace

static void eraseTerminatorAndDCECond(Instruction *TI,
                                      MemorySSAUpdater *MSSAU = nullptr) {
  Instruction *Cond = nullptr;
  if (SwitchInst *SI = dyn_cast<SwitchInst>(TI)) {
    Cond = dyn_cast<Instruction>(SI->getCondition());
  } else if (BranchInst *BI = dyn_cast<BranchInst>(TI)) {
    if (BI->isConditional())
      Cond = dyn_cast<Instruction>(BI->getCondition());
  } else if (IndirectBrInst *IBI = dyn_cast<IndirectBrInst>(TI)) {
    Cond = dyn_cast<Instruction>(IBI->getAddress());
  }

  TI->eraseFromParent();
  if (Cond)
    RecursivelyDeleteTriviallyDeadInstructions(Cond, nullptr, MSSAU);
}

/// Return true if the specified terminator checks
/// to see if a value is equal to constant integer value.
Value *SimplifyCFGOpt::isValueEqualityComparison(Instruction *TI) {
  Value *CV = nullptr;
  if (SwitchInst *SI = dyn_cast<SwitchInst>(TI)) {
    // Do not permit merging of large switch instructions into their
    // predecessors unless there is only one predecessor.
    if (!SI->getParent()->hasNPredecessorsOrMore(128 / SI->getNumSuccessors()))
      CV = SI->getCondition();
  } else if (BranchInst *BI = dyn_cast<BranchInst>(TI))
    if (BI->isConditional() && BI->getCondition()->hasOneUse()) {
      if (ICmpInst *ICI = dyn_cast<ICmpInst>(BI->getCondition())) {
        if (ICI->isEquality() && getConstantInt(ICI->getOperand(1), DL))
          CV = ICI->getOperand(0);
      } else if (auto *Trunc = dyn_cast<TruncInst>(BI->getCondition())) {
        if (Trunc->hasNoUnsignedWrap())
          CV = Trunc->getOperand(0);
      }
    }

  // Unwrap any lossless ptrtoint cast.
  if (CV) {
    if (PtrToIntInst *PTII = dyn_cast<PtrToIntInst>(CV)) {
      Value *Ptr = PTII->getPointerOperand();
      if (PTII->getType() == DL.getIntPtrType(Ptr->getType()))
        CV = Ptr;
    }
  }
  return CV;
}

/// Given a value comparison instruction,
/// decode all of the 'cases' that it represents and return the 'default' block.
BasicBlock *SimplifyCFGOpt::getValueEqualityComparisonCases(
    Instruction *TI, std::vector<ValueEqualityComparisonCase> &Cases) {
  if (SwitchInst *SI = dyn_cast<SwitchInst>(TI)) {
    Cases.reserve(SI->getNumCases());
    for (auto Case : SI->cases())
      Cases.push_back(ValueEqualityComparisonCase(Case.getCaseValue(),
                                                  Case.getCaseSuccessor()));
    return SI->getDefaultDest();
  }

  BranchInst *BI = cast<BranchInst>(TI);
  Value *Cond = BI->getCondition();
  ICmpInst::Predicate Pred;
  ConstantInt *C;
  if (auto *ICI = dyn_cast<ICmpInst>(Cond)) {
    Pred = ICI->getPredicate();
    C = getConstantInt(ICI->getOperand(1), DL);
  } else {
    Pred = ICmpInst::ICMP_NE;
    auto *Trunc = cast<TruncInst>(Cond);
    C = ConstantInt::get(cast<IntegerType>(Trunc->getOperand(0)->getType()), 0);
  }
  BasicBlock *Succ = BI->getSuccessor(Pred == ICmpInst::ICMP_NE);
  Cases.push_back(ValueEqualityComparisonCase(C, Succ));
  return BI->getSuccessor(Pred == ICmpInst::ICMP_EQ);
}

/// Given a vector of bb/value pairs, remove any entries
/// in the list that match the specified block.
static void
eliminateBlockCases(BasicBlock *BB,
                    std::vector<ValueEqualityComparisonCase> &Cases) {
  llvm::erase(Cases, BB);
}

/// Return true if there are any keys in C1 that exist in C2 as well.
static bool valuesOverlap(std::vector<ValueEqualityComparisonCase> &C1,
                          std::vector<ValueEqualityComparisonCase> &C2) {
  std::vector<ValueEqualityComparisonCase> *V1 = &C1, *V2 = &C2;

  // Make V1 be smaller than V2.
  if (V1->size() > V2->size())
    std::swap(V1, V2);

  if (V1->empty())
    return false;
  if (V1->size() == 1) {
    // Just scan V2.
    ConstantInt *TheVal = (*V1)[0].Value;
    for (const ValueEqualityComparisonCase &VECC : *V2)
      if (TheVal == VECC.Value)
        return true;
  }

  // Otherwise, just sort both lists and compare element by element.
  array_pod_sort(V1->begin(), V1->end());
  array_pod_sort(V2->begin(), V2->end());
  unsigned i1 = 0, i2 = 0, e1 = V1->size(), e2 = V2->size();
  while (i1 != e1 && i2 != e2) {
    if ((*V1)[i1].Value == (*V2)[i2].Value)
      return true;
    if ((*V1)[i1].Value < (*V2)[i2].Value)
      ++i1;
    else
      ++i2;
  }
  return false;
}

// Set branch weights on SwitchInst. This sets the metadata if there is at
// least one non-zero weight.
static void setBranchWeights(SwitchInst *SI, ArrayRef<uint32_t> Weights,
                             bool IsExpected) {
  // Check that there is at least one non-zero weight. Otherwise, pass
  // nullptr to setMetadata which will erase the existing metadata.
  MDNode *N = nullptr;
  if (llvm::any_of(Weights, [](uint32_t W) { return W != 0; }))
    N = MDBuilder(SI->getParent()->getContext())
            .createBranchWeights(Weights, IsExpected);
  SI->setMetadata(LLVMContext::MD_prof, N);
}

// Similar to the above, but for branch and select instructions that take
// exactly 2 weights.
static void setBranchWeights(Instruction *I, uint32_t TrueWeight,
                             uint32_t FalseWeight, bool IsExpected) {
  assert(isa<BranchInst>(I) || isa<SelectInst>(I));
  // Check that there is at least one non-zero weight. Otherwise, pass
  // nullptr to setMetadata which will erase the existing metadata.
  MDNode *N = nullptr;
  if (TrueWeight || FalseWeight)
    N = MDBuilder(I->getParent()->getContext())
            .createBranchWeights(TrueWeight, FalseWeight, IsExpected);
  I->setMetadata(LLVMContext::MD_prof, N);
}

/// If TI is known to be a terminator instruction and its block is known to
/// only have a single predecessor block, check to see if that predecessor is
/// also a value comparison with the same value, and if that comparison
/// determines the outcome of this comparison. If so, simplify TI. This does a
/// very limited form of jump threading.
bool SimplifyCFGOpt::simplifyEqualityComparisonWithOnlyPredecessor(
    Instruction *TI, BasicBlock *Pred, IRBuilder<> &Builder) {
  Value *PredVal = isValueEqualityComparison(Pred->getTerminator());
  if (!PredVal)
    return false; // Not a value comparison in predecessor.

  Value *ThisVal = isValueEqualityComparison(TI);
  assert(ThisVal && "This isn't a value comparison!!");
  if (ThisVal != PredVal)
    return false; // Different predicates.

  // TODO: Preserve branch weight metadata, similarly to how
  // foldValueComparisonIntoPredecessors preserves it.

  // Find out information about when control will move from Pred to TI's block.
  std::vector<ValueEqualityComparisonCase> PredCases;
  BasicBlock *PredDef =
      getValueEqualityComparisonCases(Pred->getTerminator(), PredCases);
  eliminateBlockCases(PredDef, PredCases); // Remove default from cases.

  // Find information about how control leaves this block.
  std::vector<ValueEqualityComparisonCase> ThisCases;
  BasicBlock *ThisDef = getValueEqualityComparisonCases(TI, ThisCases);
  eliminateBlockCases(ThisDef, ThisCases); // Remove default from cases.

  // If TI's block is the default block from Pred's comparison, potentially
  // simplify TI based on this knowledge.
  if (PredDef == TI->getParent()) {
    // If we are here, we know that the value is none of those cases listed in
    // PredCases.  If there are any cases in ThisCases that are in PredCases, we
    // can simplify TI.
    if (!valuesOverlap(PredCases, ThisCases))
      return false;

    if (isa<BranchInst>(TI)) {
      // Okay, one of the successors of this condbr is dead.  Convert it to a
      // uncond br.
      assert(ThisCases.size() == 1 && "Branch can only have one case!");
      // Insert the new branch.
      Instruction *NI = Builder.CreateBr(ThisDef);
      (void)NI;

      // Remove PHI node entries for the dead edge.
      ThisCases[0].Dest->removePredecessor(PredDef);

      LLVM_DEBUG(dbgs() << "Threading pred instr: " << *Pred->getTerminator()
                        << "Through successor TI: " << *TI << "Leaving: " << *NI
                        << "\n");

      eraseTerminatorAndDCECond(TI);

      if (DTU)
        DTU->applyUpdates(
            {{DominatorTree::Delete, PredDef, ThisCases[0].Dest}});

      return true;
    }

    SwitchInstProfUpdateWrapper SI = *cast<SwitchInst>(TI);
    // Okay, TI has cases that are statically dead, prune them away.
    SmallPtrSet<Constant *, 16> DeadCases;
    for (const ValueEqualityComparisonCase &Case : PredCases)
      DeadCases.insert(Case.Value);

    LLVM_DEBUG(dbgs() << "Threading pred instr: " << *Pred->getTerminator()
                      << "Through successor TI: " << *TI);

    SmallDenseMap<BasicBlock *, int, 8> NumPerSuccessorCases;
    for (SwitchInst::CaseIt i = SI->case_end(), e = SI->case_begin(); i != e;) {
      --i;
      auto *Successor = i->getCaseSuccessor();
      if (DTU)
        ++NumPerSuccessorCases[Successor];
      if (DeadCases.count(i->getCaseValue())) {
        Successor->removePredecessor(PredDef);
        SI.removeCase(i);
        if (DTU)
          --NumPerSuccessorCases[Successor];
      }
    }

    if (DTU) {
      std::vector<DominatorTree::UpdateType> Updates;
      for (const std::pair<BasicBlock *, int> &I : NumPerSuccessorCases)
        if (I.second == 0)
          Updates.push_back({DominatorTree::Delete, PredDef, I.first});
      DTU->applyUpdates(Updates);
    }

    LLVM_DEBUG(dbgs() << "Leaving: " << *TI << "\n");
    return true;
  }

  // Otherwise, TI's block must correspond to some matched value.  Find out
  // which value (or set of values) this is.
  ConstantInt *TIV = nullptr;
  BasicBlock *TIBB = TI->getParent();
  for (const auto &[Value, Dest] : PredCases)
    if (Dest == TIBB) {
      if (TIV)
        return false; // Cannot handle multiple values coming to this block.
      TIV = Value;
    }
  assert(TIV && "No edge from pred to succ?");

  // Okay, we found the one constant that our value can be if we get into TI's
  // BB.  Find out which successor will unconditionally be branched to.
  BasicBlock *TheRealDest = nullptr;
  for (const auto &[Value, Dest] : ThisCases)
    if (Value == TIV) {
      TheRealDest = Dest;
      break;
    }

  // If not handled by any explicit cases, it is handled by the default case.
  if (!TheRealDest)
    TheRealDest = ThisDef;

  SmallPtrSet<BasicBlock *, 2> RemovedSuccs;

  // Remove PHI node entries for dead edges.
  BasicBlock *CheckEdge = TheRealDest;
  for (BasicBlock *Succ : successors(TIBB))
    if (Succ != CheckEdge) {
      if (Succ != TheRealDest)
        RemovedSuccs.insert(Succ);
      Succ->removePredecessor(TIBB);
    } else
      CheckEdge = nullptr;

  // Insert the new branch.
  Instruction *NI = Builder.CreateBr(TheRealDest);
  (void)NI;

  LLVM_DEBUG(dbgs() << "Threading pred instr: " << *Pred->getTerminator()
                    << "Through successor TI: " << *TI << "Leaving: " << *NI
                    << "\n");

  eraseTerminatorAndDCECond(TI);
  if (DTU) {
    SmallVector<DominatorTree::UpdateType, 2> Updates;
    Updates.reserve(RemovedSuccs.size());
    for (auto *RemovedSucc : RemovedSuccs)
      Updates.push_back({DominatorTree::Delete, TIBB, RemovedSucc});
    DTU->applyUpdates(Updates);
  }
  return true;
}

namespace {

/// This class implements a stable ordering of constant
/// integers that does not depend on their address.  This is important for
/// applications that sort ConstantInt's to ensure uniqueness.
struct ConstantIntOrdering {
  bool operator()(const ConstantInt *LHS, const ConstantInt *RHS) const {
    return LHS->getValue().ult(RHS->getValue());
  }
};

} // end anonymous namespace

static int constantIntSortPredicate(ConstantInt *const *P1,
                                    ConstantInt *const *P2) {
  const ConstantInt *LHS = *P1;
  const ConstantInt *RHS = *P2;
  if (LHS == RHS)
    return 0;
  return LHS->getValue().ult(RHS->getValue()) ? 1 : -1;
}

/// Get Weights of a given terminator, the default weight is at the front
/// of the vector. If TI is a conditional eq, we need to swap the branch-weight
/// metadata.
static void getBranchWeights(Instruction *TI,
                             SmallVectorImpl<uint64_t> &Weights) {
  MDNode *MD = TI->getMetadata(LLVMContext::MD_prof);
  assert(MD && "Invalid branch-weight metadata");
  extractFromBranchWeightMD64(MD, Weights);

  // If TI is a conditional eq, the default case is the false case,
  // and the corresponding branch-weight data is at index 2. We swap the
  // default weight to be the first entry.
  if (BranchInst *BI = dyn_cast<BranchInst>(TI)) {
    assert(Weights.size() == 2);
    auto *ICI = dyn_cast<ICmpInst>(BI->getCondition());
    if (!ICI)
      return;

    if (ICI->getPredicate() == ICmpInst::ICMP_EQ)
      std::swap(Weights.front(), Weights.back());
  }
}

/// Keep halving the weights until all can fit in uint32_t.
static void fitWeights(MutableArrayRef<uint64_t> Weights) {
  uint64_t Max = *llvm::max_element(Weights);
  if (Max > UINT_MAX) {
    unsigned Offset = 32 - llvm::countl_zero(Max);
    for (uint64_t &I : Weights)
      I >>= Offset;
  }
}

static void cloneInstructionsIntoPredecessorBlockAndUpdateSSAUses(
    BasicBlock *BB, BasicBlock *PredBlock, ValueToValueMapTy &VMap) {
  Instruction *PTI = PredBlock->getTerminator();

  // If we have bonus instructions, clone them into the predecessor block.
  // Note that there may be multiple predecessor blocks, so we cannot move
  // bonus instructions to a predecessor block.
  for (Instruction &BonusInst : *BB) {
    if (BonusInst.isTerminator())
      continue;

    Instruction *NewBonusInst = BonusInst.clone();

    if (!NewBonusInst->getDebugLoc().isSameSourceLocation(PTI->getDebugLoc())) {
      // Unless the instruction has the same !dbg location as the original
      // branch, drop it. When we fold the bonus instructions we want to make
      // sure we reset their debug locations in order to avoid stepping on
      // dead code caused by folding dead branches.
      NewBonusInst->setDebugLoc(DebugLoc::getDropped());
    } else if (const DebugLoc &DL = NewBonusInst->getDebugLoc()) {
      mapAtomInstance(DL, VMap);
    }

    RemapInstruction(NewBonusInst, VMap,
                     RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);

    // If we speculated an instruction, we need to drop any metadata that may
    // result in undefined behavior, as the metadata might have been valid
    // only given the branch precondition.
    // Similarly strip attributes on call parameters that may cause UB in
    // location the call is moved to.
    NewBonusInst->dropUBImplyingAttrsAndMetadata();

    NewBonusInst->insertInto(PredBlock, PTI->getIterator());
    auto Range = NewBonusInst->cloneDebugInfoFrom(&BonusInst);
    RemapDbgRecordRange(NewBonusInst->getModule(), Range, VMap,
                        RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);

    NewBonusInst->takeName(&BonusInst);
    BonusInst.setName(NewBonusInst->getName() + ".old");
    VMap[&BonusInst] = NewBonusInst;

    // Update (liveout) uses of bonus instructions,
    // now that the bonus instruction has been cloned into predecessor.
    // Note that we expect to be in a block-closed SSA form for this to work!
    for (Use &U : make_early_inc_range(BonusInst.uses())) {
      auto *UI = cast<Instruction>(U.getUser());
      auto *PN = dyn_cast<PHINode>(UI);
      if (!PN) {
        assert(UI->getParent() == BB && BonusInst.comesBefore(UI) &&
               "If the user is not a PHI node, then it should be in the same "
               "block as, and come after, the original bonus instruction.");
        continue; // Keep using the original bonus instruction.
      }
      // Is this the block-closed SSA form PHI node?
      if (PN->getIncomingBlock(U) == BB)
        continue; // Great, keep using the original bonus instruction.
      // The only other alternative is an "use" when coming from
      // the predecessor block - here we should refer to the cloned bonus instr.
      assert(PN->getIncomingBlock(U) == PredBlock &&
             "Not in block-closed SSA form?");
      U.set(NewBonusInst);
    }
  }

  // Key Instructions: We may have propagated atom info into the pred. If the
  // pred's terminator already has atom info do nothing as merging would drop
  // one atom group anyway. If it doesn't, propagte the remapped atom group
  // from BB's terminator.
  if (auto &PredDL = PTI->getDebugLoc()) {
    auto &DL = BB->getTerminator()->getDebugLoc();
    if (!PredDL->getAtomGroup() && DL && DL->getAtomGroup() &&
        PredDL.isSameSourceLocation(DL)) {
      PTI->setDebugLoc(DL);
      RemapSourceAtom(PTI, VMap);
    }
  }
}

bool SimplifyCFGOpt::performValueComparisonIntoPredecessorFolding(
    Instruction *TI, Value *&CV, Instruction *PTI, IRBuilder<> &Builder) {
  BasicBlock *BB = TI->getParent();
  BasicBlock *Pred = PTI->getParent();

  SmallVector<DominatorTree::UpdateType, 32> Updates;

  // Figure out which 'cases' to copy from SI to PSI.
  std::vector<ValueEqualityComparisonCase> BBCases;
  BasicBlock *BBDefault = getValueEqualityComparisonCases(TI, BBCases);

  std::vector<ValueEqualityComparisonCase> PredCases;
  BasicBlock *PredDefault = getValueEqualityComparisonCases(PTI, PredCases);

  // Based on whether the default edge from PTI goes to BB or not, fill in
  // PredCases and PredDefault with the new switch cases we would like to
  // build.
  SmallMapVector<BasicBlock *, int, 8> NewSuccessors;

  // Update the branch weight metadata along the way
  SmallVector<uint64_t, 8> Weights;
  bool PredHasWeights = hasBranchWeightMD(*PTI);
  bool SuccHasWeights = hasBranchWeightMD(*TI);

  if (PredHasWeights) {
    getBranchWeights(PTI, Weights);
    // branch-weight metadata is inconsistent here.
    if (Weights.size() != 1 + PredCases.size())
      PredHasWeights = SuccHasWeights = false;
  } else if (SuccHasWeights)
    // If there are no predecessor weights but there are successor weights,
    // populate Weights with 1, which will later be scaled to the sum of
    // successor's weights
    Weights.assign(1 + PredCases.size(), 1);

  SmallVector<uint64_t, 8> SuccWeights;
  if (SuccHasWeights) {
    getBranchWeights(TI, SuccWeights);
    // branch-weight metadata is inconsistent here.
    if (SuccWeights.size() != 1 + BBCases.size())
      PredHasWeights = SuccHasWeights = false;
  } else if (PredHasWeights)
    SuccWeights.assign(1 + BBCases.size(), 1);

  if (PredDefault == BB) {
    // If this is the default destination from PTI, only the edges in TI
    // that don't occur in PTI, or that branch to BB will be activated.
    std::set<ConstantInt *, ConstantIntOrdering> PTIHandled;
    for (unsigned i = 0, e = PredCases.size(); i != e; ++i)
      if (PredCases[i].Dest != BB)
        PTIHandled.insert(PredCases[i].Value);
      else {
        // The default destination is BB, we don't need explicit targets.
        std::swap(PredCases[i], PredCases.back());

        if (PredHasWeights || SuccHasWeights) {
          // Increase weight for the default case.
          Weights[0] += Weights[i + 1];
          std::swap(Weights[i + 1], Weights.back());
          Weights.pop_back();
        }

        PredCases.pop_back();
        --i;
        --e;
      }

    // Reconstruct the new switch statement we will be building.
    if (PredDefault != BBDefault) {
      PredDefault->removePredecessor(Pred);
      if (DTU && PredDefault != BB)
        Updates.push_back({DominatorTree::Delete, Pred, PredDefault});
      PredDefault = BBDefault;
      ++NewSuccessors[BBDefault];
    }

    unsigned CasesFromPred = Weights.size();
    uint64_t ValidTotalSuccWeight = 0;
    for (unsigned i = 0, e = BBCases.size(); i != e; ++i)
      if (!PTIHandled.count(BBCases[i].Value) && BBCases[i].Dest != BBDefault) {
        PredCases.push_back(BBCases[i]);
        ++NewSuccessors[BBCases[i].Dest];
        if (SuccHasWeights || PredHasWeights) {
          // The default weight is at index 0, so weight for the ith case
          // should be at index i+1. Scale the cases from successor by
          // PredDefaultWeight (Weights[0]).
          Weights.push_back(Weights[0] * SuccWeights[i + 1]);
          ValidTotalSuccWeight += SuccWeights[i + 1];
        }
      }

    if (SuccHasWeights || PredHasWeights) {
      ValidTotalSuccWeight += SuccWeights[0];
      // Scale the cases from predecessor by ValidTotalSuccWeight.
      for (unsigned i = 1; i < CasesFromPred; ++i)
        Weights[i] *= ValidTotalSuccWeight;
      // Scale the default weight by SuccDefaultWeight (SuccWeights[0]).
      Weights[0] *= SuccWeights[0];
    }
  } else {
    // If this is not the default destination from PSI, only the edges
    // in SI that occur in PSI with a destination of BB will be
    // activated.
    std::set<ConstantInt *, ConstantIntOrdering> PTIHandled;
    std::map<ConstantInt *, uint64_t> WeightsForHandled;
    for (unsigned i = 0, e = PredCases.size(); i != e; ++i)
      if (PredCases[i].Dest == BB) {
        PTIHandled.insert(PredCases[i].Value);

        if (PredHasWeights || SuccHasWeights) {
          WeightsForHandled[PredCases[i].Value] = Weights[i + 1];
          std::swap(Weights[i + 1], Weights.back());
          Weights.pop_back();
        }

        std::swap(PredCases[i], PredCases.back());
        PredCases.pop_back();
        --i;
        --e;
      }

    // Okay, now we know which constants were sent to BB from the
    // predecessor.  Figure out where they will all go now.
    for (const ValueEqualityComparisonCase &Case : BBCases)
      if (PTIHandled.count(Case.Value)) {
        // If this is one we are capable of getting...
        if (PredHasWeights || SuccHasWeights)
          Weights.push_back(WeightsForHandled[Case.Value]);
        PredCases.push_back(Case);
        ++NewSuccessors[Case.Dest];
        PTIHandled.erase(Case.Value); // This constant is taken care of
      }

    // If there are any constants vectored to BB that TI doesn't handle,
    // they must go to the default destination of TI.
    for (ConstantInt *I : PTIHandled) {
      if (PredHasWeights || SuccHasWeights)
        Weights.push_back(WeightsForHandled[I]);
      PredCases.push_back(ValueEqualityComparisonCase(I, BBDefault));
      ++NewSuccessors[BBDefault];
    }
  }

  // Okay, at this point, we know which new successor Pred will get.  Make
  // sure we update the number of entries in the PHI nodes for these
  // successors.
  SmallPtrSet<BasicBlock *, 2> SuccsOfPred;
  if (DTU) {
    SuccsOfPred = {llvm::from_range, successors(Pred)};
    Updates.reserve(Updates.size() + NewSuccessors.size());
  }
  for (const std::pair<BasicBlock *, int /*Num*/> &NewSuccessor :
       NewSuccessors) {
    for (auto I : seq(NewSuccessor.second)) {
      (void)I;
      addPredecessorToBlock(NewSuccessor.first, Pred, BB);
    }
    if (DTU && !SuccsOfPred.contains(NewSuccessor.first))
      Updates.push_back({DominatorTree::Insert, Pred, NewSuccessor.first});
  }

  Builder.SetInsertPoint(PTI);
  // Convert pointer to int before we switch.
  if (CV->getType()->isPointerTy()) {
    CV =
        Builder.CreatePtrToInt(CV, DL.getIntPtrType(CV->getType()), "magicptr");
  }

  // Now that the successors are updated, create the new Switch instruction.
  SwitchInst *NewSI = Builder.CreateSwitch(CV, PredDefault, PredCases.size());
  NewSI->setDebugLoc(PTI->getDebugLoc());
  for (ValueEqualityComparisonCase &V : PredCases)
    NewSI->addCase(V.Value, V.Dest);

  if (PredHasWeights || SuccHasWeights) {
    // Halve the weights if any of them cannot fit in an uint32_t
    fitWeights(Weights);

    SmallVector<uint32_t, 8> MDWeights(Weights.begin(), Weights.end());

    setBranchWeights(NewSI, MDWeights, /*IsExpected=*/false);
  }

  eraseTerminatorAndDCECond(PTI);

  // Okay, last check.  If BB is still a successor of PSI, then we must
  // have an infinite loop case.  If so, add an infinitely looping block
  // to handle the case to preserve the behavior of the code.
  BasicBlock *InfLoopBlock = nullptr;
  for (unsigned i = 0, e = NewSI->getNumSuccessors(); i != e; ++i)
    if (NewSI->getSuccessor(i) == BB) {
      if (!InfLoopBlock) {
        // Insert it at the end of the function, because it's either code,
        // or it won't matter if it's hot. :)
        InfLoopBlock =
            BasicBlock::Create(BB->getContext(), "infloop", BB->getParent());
        BranchInst::Create(InfLoopBlock, InfLoopBlock);
        if (DTU)
          Updates.push_back(
              {DominatorTree::Insert, InfLoopBlock, InfLoopBlock});
      }
      NewSI->setSuccessor(i, InfLoopBlock);
    }

  if (DTU) {
    if (InfLoopBlock)
      Updates.push_back({DominatorTree::Insert, Pred, InfLoopBlock});

    Updates.push_back({DominatorTree::Delete, Pred, BB});

    DTU->applyUpdates(Updates);
  }

  ++NumFoldValueComparisonIntoPredecessors;
  return true;
}

/// The specified terminator is a value equality comparison instruction
/// (either a switch or a branch on "X == c").
/// See if any of the predecessors of the terminator block are value comparisons
/// on the same value.  If so, and if safe to do so, fold them together.
bool SimplifyCFGOpt::foldValueComparisonIntoPredecessors(Instruction *TI,
                                                         IRBuilder<> &Builder) {
  BasicBlock *BB = TI->getParent();
  Value *CV = isValueEqualityComparison(TI); // CondVal
  assert(CV && "Not a comparison?");

  bool Changed = false;

  SmallSetVector<BasicBlock *, 16> Preds(pred_begin(BB), pred_end(BB));
  while (!Preds.empty()) {
    BasicBlock *Pred = Preds.pop_back_val();
    Instruction *PTI = Pred->getTerminator();

    // Don't try to fold into itself.
    if (Pred == BB)
      continue;

    // See if the predecessor is a comparison with the same value.
    Value *PCV = isValueEqualityComparison(PTI); // PredCondVal
    if (PCV != CV)
      continue;

    SmallSetVector<BasicBlock *, 4> FailBlocks;
    if (!safeToMergeTerminators(TI, PTI, &FailBlocks)) {
      for (auto *Succ : FailBlocks) {
        if (!SplitBlockPredecessors(Succ, TI->getParent(), ".fold.split", DTU))
          return false;
      }
    }

    performValueComparisonIntoPredecessorFolding(TI, CV, PTI, Builder);
    Changed = true;
  }
  return Changed;
}

// If we would need to insert a select that uses the value of this invoke
// (comments in hoistSuccIdenticalTerminatorToSwitchOrIf explain why we would
// need to do this), we can't hoist the invoke, as there is nowhere to put the
// select in this case.
static bool isSafeToHoistInvoke(BasicBlock *BB1, BasicBlock *BB2,
                                Instruction *I1, Instruction *I2) {
  for (BasicBlock *Succ : successors(BB1)) {
    for (const PHINode &PN : Succ->phis()) {
      Value *BB1V = PN.getIncomingValueForBlock(BB1);
      Value *BB2V = PN.getIncomingValueForBlock(BB2);
      if (BB1V != BB2V && (BB1V == I1 || BB2V == I2)) {
        return false;
      }
    }
  }
  return true;
}

// Get interesting characteristics of instructions that
// `hoistCommonCodeFromSuccessors` didn't hoist. They restrict what kind of
// instructions can be reordered across.
enum SkipFlags {
  SkipReadMem = 1,
  SkipSideEffect = 2,
  SkipImplicitControlFlow = 4
};

static unsigned skippedInstrFlags(Instruction *I) {
  unsigned Flags = 0;
  if (I->mayReadFromMemory())
    Flags |= SkipReadMem;
  // We can't arbitrarily move around allocas, e.g. moving allocas (especially
  // inalloca) across stacksave/stackrestore boundaries.
  if (I->mayHaveSideEffects() || isa<AllocaInst>(I))
    Flags |= SkipSideEffect;
  if (!isGuaranteedToTransferExecutionToSuccessor(I))
    Flags |= SkipImplicitControlFlow;
  return Flags;
}

// Returns true if it is safe to reorder an instruction across preceding
// instructions in a basic block.
static bool isSafeToHoistInstr(Instruction *I, unsigned Flags) {
  // Don't reorder a store over a load.
  if ((Flags & SkipReadMem) && I->mayWriteToMemory())
    return false;

  // If we have seen an instruction with side effects, it's unsafe to reorder an
  // instruction which reads memory or itself has side effects.
  if ((Flags & SkipSideEffect) &&
      (I->mayReadFromMemory() || I->mayHaveSideEffects() || isa<AllocaInst>(I)))
    return false;

  // Reordering across an instruction which does not necessarily transfer
  // control to the next instruction is speculation.
  if ((Flags & SkipImplicitControlFlow) && !isSafeToSpeculativelyExecute(I))
    return false;

  // Hoisting of llvm.deoptimize is only legal together with the next return
  // instruction, which this pass is not always able to do.
  if (auto *CB = dyn_cast<CallBase>(I))
    if (CB->getIntrinsicID() == Intrinsic::experimental_deoptimize)
      return false;

  // It's also unsafe/illegal to hoist an instruction above its instruction
  // operands
  BasicBlock *BB = I->getParent();
  for (Value *Op : I->operands()) {
    if (auto *J = dyn_cast<Instruction>(Op))
      if (J->getParent() == BB)
        return false;
  }

  return true;
}

static bool passingValueIsAlwaysUndefined(Value *V, Instruction *I, bool PtrValueMayBeModified = false);

/// Helper function for hoistCommonCodeFromSuccessors. Return true if identical
/// instructions \p I1 and \p I2 can and should be hoisted.
static bool shouldHoistCommonInstructions(Instruction *I1, Instruction *I2,
                                          const TargetTransformInfo &TTI) {
  // If we're going to hoist a call, make sure that the two instructions
  // we're commoning/hoisting are both marked with musttail, or neither of
  // them is marked as such. Otherwise, we might end up in a situation where
  // we hoist from a block where the terminator is a `ret` to a block where
  // the terminator is a `br`, and `musttail` calls expect to be followed by
  // a return.
  auto *C1 = dyn_cast<CallInst>(I1);
  auto *C2 = dyn_cast<CallInst>(I2);
  if (C1 && C2)
    if (C1->isMustTailCall() != C2->isMustTailCall())
      return false;

  if (!TTI.isProfitableToHoist(I1) || !TTI.isProfitableToHoist(I2))
    return false;

  // If any of the two call sites has nomerge or convergent attribute, stop
  // hoisting.
  if (const auto *CB1 = dyn_cast<CallBase>(I1))
    if (CB1->cannotMerge() || CB1->isConvergent())
      return false;
  if (const auto *CB2 = dyn_cast<CallBase>(I2))
    if (CB2->cannotMerge() || CB2->isConvergent())
      return false;

  return true;
}

/// Hoists DbgVariableRecords from \p I1 and \p OtherInstrs that are identical
/// in lock-step to \p TI. This matches how dbg.* intrinsics are hoisting in
/// hoistCommonCodeFromSuccessors. e.g. The input:
///    I1                DVRs: { x, z },
///    OtherInsts: { I2  DVRs: { x, y, z } }
/// would result in hoisting only DbgVariableRecord x.
static void hoistLockstepIdenticalDbgVariableRecords(
    Instruction *TI, Instruction *I1,
    SmallVectorImpl<Instruction *> &OtherInsts) {
  if (!I1->hasDbgRecords())
    return;
  using CurrentAndEndIt =
      std::pair<DbgRecord::self_iterator, DbgRecord::self_iterator>;
  // Vector of {Current, End} iterators.
  SmallVector<CurrentAndEndIt> Itrs;
  Itrs.reserve(OtherInsts.size() + 1);
  // Helper lambdas for lock-step checks:
  // Return true if this Current == End.
  auto atEnd = [](const CurrentAndEndIt &Pair) {
    return Pair.first == Pair.second;
  };
  // Return true if all Current are identical.
  auto allIdentical = [](const SmallVector<CurrentAndEndIt> &Itrs) {
    return all_of(make_first_range(ArrayRef(Itrs).drop_front()),
                  [&](DbgRecord::self_iterator I) {
                    return Itrs[0].first->isIdenticalToWhenDefined(*I);
                  });
  };

  // Collect the iterators.
  Itrs.push_back(
      {I1->getDbgRecordRange().begin(), I1->getDbgRecordRange().end()});
  for (Instruction *Other : OtherInsts) {
    if (!Other->hasDbgRecords())
      return;
    Itrs.push_back(
        {Other->getDbgRecordRange().begin(), Other->getDbgRecordRange().end()});
  }

  // Iterate in lock-step until any of the DbgRecord lists are exausted. If
  // the lock-step DbgRecord are identical, hoist all of them to TI.
  // This replicates the dbg.* intrinsic behaviour in
  // hoistCommonCodeFromSuccessors.
  while (none_of(Itrs, atEnd)) {
    bool HoistDVRs = allIdentical(Itrs);
    for (CurrentAndEndIt &Pair : Itrs) {
      // Increment Current iterator now as we may be about to move the
      // DbgRecord.
      DbgRecord &DR = *Pair.first++;
      if (HoistDVRs) {
        DR.removeFromParent();
        TI->getParent()->insertDbgRecordBefore(&DR, TI->getIterator());
      }
    }
  }
}

static bool areIdenticalUpToCommutativity(const Instruction *I1,
                                          const Instruction *I2) {
  if (I1->isIdenticalToWhenDefined(I2, /*IntersectAttrs=*/true))
    return true;

  if (auto *Cmp1 = dyn_cast<CmpInst>(I1))
    if (auto *Cmp2 = dyn_cast<CmpInst>(I2))
      return Cmp1->getPredicate() == Cmp2->getSwappedPredicate() &&
             Cmp1->getOperand(0) == Cmp2->getOperand(1) &&
             Cmp1->getOperand(1) == Cmp2->getOperand(0);

  if (I1->isCommutative() && I1->isSameOperationAs(I2)) {
    return I1->getOperand(0) == I2->getOperand(1) &&
           I1->getOperand(1) == I2->getOperand(0) &&
           equal(drop_begin(I1->operands(), 2), drop_begin(I2->operands(), 2));
  }

  return false;
}

/// If the target supports conditional faulting,
/// we look for the following pattern:
/// \code
///   BB:
///     ...
///     %cond = icmp ult %x, %y
///     br i1 %cond, label %TrueBB, label %FalseBB
///   FalseBB:
///     store i32 1, ptr %q, align 4
///     ...
///   TrueBB:
///     %maskedloadstore = load i32, ptr %b, align 4
///     store i32 %maskedloadstore, ptr %p, align 4
///     ...
/// \endcode
///
/// and transform it into:
///
/// \code
///   BB:
///     ...
///     %cond = icmp ult %x, %y
///     %maskedloadstore = cload i32, ptr %b, %cond
///     cstore i32 %maskedloadstore, ptr %p, %cond
///     cstore i32 1, ptr %q, ~%cond
///     br i1 %cond, label %TrueBB, label %FalseBB
///   FalseBB:
///     ...
///   TrueBB:
///     ...
/// \endcode
///
/// where cload/cstore are represented by llvm.masked.load/store intrinsics,
/// e.g.
///
/// \code
///   %vcond = bitcast i1 %cond to <1 x i1>
///   %v0 = call <1 x i32> @llvm.masked.load.v1i32.p0
///                         (ptr %b, i32 4, <1 x i1> %vcond, <1 x i32> poison)
///   %maskedloadstore = bitcast <1 x i32> %v0 to i32
///   call void @llvm.masked.store.v1i32.p0
///                          (<1 x i32> %v0, ptr %p, i32 4, <1 x i1> %vcond)
///   %cond.not = xor i1 %cond, true
///   %vcond.not = bitcast i1 %cond.not to <1 x i>
///   call void @llvm.masked.store.v1i32.p0
///              (<1 x i32> <i32 1>, ptr %q, i32 4, <1x i1> %vcond.not)
/// \endcode
///
/// So we need to turn hoisted load/store into cload/cstore.
///
/// \param BI The branch instruction.
/// \param SpeculatedConditionalLoadsStores The load/store instructions that
///                                         will be speculated.
/// \param Invert indicates if speculates FalseBB. Only used in triangle CFG.
static void hoistConditionalLoadsStores(
    BranchInst *BI,
    SmallVectorImpl<Instruction *> &SpeculatedConditionalLoadsStores,
    std::optional<bool> Invert, Instruction *Sel) {
  auto &Context = BI->getParent()->getContext();
  auto *VCondTy = FixedVectorType::get(Type::getInt1Ty(Context), 1);
  auto *Cond = BI->getOperand(0);
  // Construct the condition if needed.
  BasicBlock *BB = BI->getParent();
  Value *Mask = nullptr;
  Value *MaskFalse = nullptr;
  Value *MaskTrue = nullptr;
  if (Invert.has_value()) {
    IRBuilder<> Builder(Sel ? Sel : SpeculatedConditionalLoadsStores.back());
    Mask = Builder.CreateBitCast(
        *Invert ? Builder.CreateXor(Cond, ConstantInt::getTrue(Context)) : Cond,
        VCondTy);
  } else {
    IRBuilder<> Builder(BI);
    MaskFalse = Builder.CreateBitCast(
        Builder.CreateXor(Cond, ConstantInt::getTrue(Context)), VCondTy);
    MaskTrue = Builder.CreateBitCast(Cond, VCondTy);
  }
  auto PeekThroughBitcasts = [](Value *V) {
    while (auto *BitCast = dyn_cast<BitCastInst>(V))
      V = BitCast->getOperand(0);
    return V;
  };
  for (auto *I : SpeculatedConditionalLoadsStores) {
    IRBuilder<> Builder(Invert.has_value() ? I : BI);
    if (!Invert.has_value())
      Mask = I->getParent() == BI->getSuccessor(0) ? MaskTrue : MaskFalse;
    // We currently assume conditional faulting load/store is supported for
    // scalar types only when creating new instructions. This can be easily
    // extended for vector types in the future.
    assert(!getLoadStoreType(I)->isVectorTy() && "not implemented");
    auto *Op0 = I->getOperand(0);
    CallInst *MaskedLoadStore = nullptr;
    if (auto *LI = dyn_cast<LoadInst>(I)) {
      // Handle Load.
      auto *Ty = I->getType();
      PHINode *PN = nullptr;
      Value *PassThru = nullptr;
      if (Invert.has_value())
        for (User *U : I->users()) {
          if ((PN = dyn_cast<PHINode>(U))) {
            PassThru = Builder.CreateBitCast(
                PeekThroughBitcasts(PN->getIncomingValueForBlock(BB)),
                FixedVectorType::get(Ty, 1));
          } else if (auto *Ins = cast<Instruction>(U);
                     Sel && Ins->getParent() == BB) {
            // This happens when store or/and a speculative instruction between
            // load and store were hoisted to the BB. Make sure the masked load
            // inserted before its use.
            // We assume there's one of such use.
            Builder.SetInsertPoint(Ins);
          }
        }
      MaskedLoadStore = Builder.CreateMaskedLoad(
          FixedVectorType::get(Ty, 1), Op0, LI->getAlign(), Mask, PassThru);
      Value *NewLoadStore = Builder.CreateBitCast(MaskedLoadStore, Ty);
      if (PN)
        PN->setIncomingValue(PN->getBasicBlockIndex(BB), NewLoadStore);
      I->replaceAllUsesWith(NewLoadStore);
    } else {
      // Handle Store.
      auto *StoredVal = Builder.CreateBitCast(
          PeekThroughBitcasts(Op0), FixedVectorType::get(Op0->getType(), 1));
      MaskedLoadStore = Builder.CreateMaskedStore(
          StoredVal, I->getOperand(1), cast<StoreInst>(I)->getAlign(), Mask);
    }
    // For non-debug metadata, only !annotation, !range, !nonnull and !align are
    // kept when hoisting (see Instruction::dropUBImplyingAttrsAndMetadata).
    //
    // !nonnull, !align : Not support pointer type, no need to keep.
    // !range: Load type is changed from scalar to vector, but the metadata on
    //         vector specifies a per-element range, so the semantics stay the
    //         same. Keep it.
    // !annotation: Not impact semantics. Keep it.
    if (const MDNode *Ranges = I->getMetadata(LLVMContext::MD_range))
      MaskedLoadStore->addRangeRetAttr(getConstantRangeFromMetadata(*Ranges));
    I->dropUBImplyingAttrsAndUnknownMetadata({LLVMContext::MD_annotation});
    // FIXME: DIAssignID is not supported for masked store yet.
    // (Verifier::visitDIAssignIDMetadata)
    at::deleteAssignmentMarkers(I);
    I->eraseMetadataIf([](unsigned MDKind, MDNode *Node) {
      return Node->getMetadataID() == Metadata::DIAssignIDKind;
    });
    MaskedLoadStore->copyMetadata(*I);
    I->eraseFromParent();
  }
}

static bool isSafeCheapLoadStore(const Instruction *I,
                                 const TargetTransformInfo &TTI) {
  // Not handle volatile or atomic.
  bool IsStore = false;
  if (auto *L = dyn_cast<LoadInst>(I)) {
    if (!L->isSimple() || !HoistLoadsWithCondFaulting)
      return false;
  } else if (auto *S = dyn_cast<StoreInst>(I)) {
    if (!S->isSimple() || !HoistStoresWithCondFaulting)
      return false;
    IsStore = true;
  } else
    return false;

  // llvm.masked.load/store use i32 for alignment while load/store use i64.
  // That's why we have the alignment limitation.
  // FIXME: Update the prototype of the intrinsics?
  return TTI.hasConditionalLoadStoreForType(getLoadStoreType(I), IsStore) &&
         getLoadStoreAlignment(I) < Value::MaximumAlignment;
}

/// Hoist any common code in the successor blocks up into the block. This
/// function guarantees that BB dominates all successors. If AllInstsEqOnly is
/// given, only perform hoisting in case all successors blocks contain matching
/// instructions only. In that case, all instructions can be hoisted and the
/// original branch will be replaced and selects for PHIs are added.
bool SimplifyCFGOpt::hoistCommonCodeFromSuccessors(Instruction *TI,
                                                   bool AllInstsEqOnly) {
  // This does very trivial matching, with limited scanning, to find identical
  // instructions in the two blocks. In particular, we don't want to get into
  // O(N1*N2*...) situations here where Ni are the sizes of these successors. As
  // such, we currently just scan for obviously identical instructions in an
  // identical order, possibly separated by the same number of non-identical
  // instructions.
  BasicBlock *BB = TI->getParent();
  unsigned int SuccSize = succ_size(BB);
  if (SuccSize < 2)
    return false;

  // If either of the blocks has it's address taken, then we can't do this fold,
  // because the code we'd hoist would no longer run when we jump into the block
  // by it's address.
  for (auto *Succ : successors(BB))
    if (Succ->hasAddressTaken() || !Succ->getSinglePredecessor())
      return false;

  // The second of pair is a SkipFlags bitmask.
  using SuccIterPair = std::pair<BasicBlock::iterator, unsigned>;
  SmallVector<SuccIterPair, 8> SuccIterPairs;
  for (auto *Succ : successors(BB)) {
    BasicBlock::iterator SuccItr = Succ->begin();
    if (isa<PHINode>(*SuccItr))
      return false;
    SuccIterPairs.push_back(SuccIterPair(SuccItr, 0));
  }

  if (AllInstsEqOnly) {
    // Check if all instructions in the successor blocks match. This allows
    // hoisting all instructions and removing the blocks we are hoisting from,
    // so does not add any new instructions.
    SmallVector<BasicBlock *> Succs = to_vector(successors(BB));
    // Check if sizes and terminators of all successors match.
    bool AllSame = none_of(Succs, [&Succs](BasicBlock *Succ) {
      Instruction *Term0 = Succs[0]->getTerminator();
      Instruction *Term = Succ->getTerminator();
      return !Term->isSameOperationAs(Term0) ||
             !equal(Term->operands(), Term0->operands()) ||
             Succs[0]->size() != Succ->size();
    });
    if (!AllSame)
      return false;
    if (AllSame) {
      LockstepReverseIterator<true> LRI(Succs);
      while (LRI.isValid()) {
        Instruction *I0 = (*LRI)[0];
        if (any_of(*LRI, [I0](Instruction *I) {
              return !areIdenticalUpToCommutativity(I0, I);
            })) {
          return false;
        }
        --LRI;
      }
    }
    // Now we know that all instructions in all successors can be hoisted. Let
    // the loop below handle the hoisting.
  }

  // Count how many instructions were not hoisted so far. There's a limit on how
  // many instructions we skip, serving as a compilation time control as well as
  // preventing excessive increase of life ranges.
  unsigned NumSkipped = 0;
  // If we find an unreachable instruction at the beginning of a basic block, we
  // can still hoist instructions from the rest of the basic blocks.
  if (SuccIterPairs.size() > 2) {
    erase_if(SuccIterPairs,
             [](const auto &Pair) { return isa<UnreachableInst>(Pair.first); });
    if (SuccIterPairs.size() < 2)
      return false;
  }

  bool Changed = false;

  for (;;) {
    auto *SuccIterPairBegin = SuccIterPairs.begin();
    auto &BB1ItrPair = *SuccIterPairBegin++;
    auto OtherSuccIterPairRange =
        iterator_range(SuccIterPairBegin, SuccIterPairs.end());
    auto OtherSuccIterRange = make_first_range(OtherSuccIterPairRange);

    Instruction *I1 = &*BB1ItrPair.first;

    bool AllInstsAreIdentical = true;
    bool HasTerminator = I1->isTerminator();
    for (auto &SuccIter : OtherSuccIterRange) {
      Instruction *I2 = &*SuccIter;
      HasTerminator |= I2->isTerminator();
      if (AllInstsAreIdentical && (!areIdenticalUpToCommutativity(I1, I2) ||
                                   MMRAMetadata(*I1) != MMRAMetadata(*I2)))
        AllInstsAreIdentical = false;
    }

    SmallVector<Instruction *, 8> OtherInsts;
    for (auto &SuccIter : OtherSuccIterRange)
      OtherInsts.push_back(&*SuccIter);

    // If we are hoisting the terminator instruction, don't move one (making a
    // broken BB), instead clone it, and remove BI.
    if (HasTerminator) {
      // Even if BB, which contains only one unreachable instruction, is ignored
      // at the beginning of the loop, we can hoist the terminator instruction.
      // If any instructions remain in the block, we cannot hoist terminators.
      if (NumSkipped || !AllInstsAreIdentical) {
        hoistLockstepIdenticalDbgVariableRecords(TI, I1, OtherInsts);
        return Changed;
      }

      return hoistSuccIdenticalTerminatorToSwitchOrIf(TI, I1, OtherInsts) ||
             Changed;
    }

    if (AllInstsAreIdentical) {
      unsigned SkipFlagsBB1 = BB1ItrPair.second;
      AllInstsAreIdentical =
          isSafeToHoistInstr(I1, SkipFlagsBB1) &&
          all_of(OtherSuccIterPairRange, [=](const auto &Pair) {
            Instruction *I2 = &*Pair.first;
            unsigned SkipFlagsBB2 = Pair.second;
            // Even if the instructions are identical, it may not
            // be safe to hoist them if we have skipped over
            // instructions with side effects or their operands
            // weren't hoisted.
            return isSafeToHoistInstr(I2, SkipFlagsBB2) &&
                   shouldHoistCommonInstructions(I1, I2, TTI);
          });
    }

    if (AllInstsAreIdentical) {
      BB1ItrPair.first++;
      // For a normal instruction, we just move one to right before the
      // branch, then replace all uses of the other with the first.  Finally,
      // we remove the now redundant second instruction.
      hoistLockstepIdenticalDbgVariableRecords(TI, I1, OtherInsts);
      // We've just hoisted DbgVariableRecords; move I1 after them (before TI)
      // and leave any that were not hoisted behind (by calling moveBefore
      // rather than moveBeforePreserving).
      I1->moveBefore(TI->getIterator());
      for (auto &SuccIter : OtherSuccIterRange) {
        Instruction *I2 = &*SuccIter++;
        assert(I2 != I1);
        if (!I2->use_empty())
          I2->replaceAllUsesWith(I1);
        I1->andIRFlags(I2);
        if (auto *CB = dyn_cast<CallBase>(I1)) {
          bool Success = CB->tryIntersectAttributes(cast<CallBase>(I2));
          assert(Success && "We should not be trying to hoist callbases "
                            "with non-intersectable attributes");
          // For NDEBUG Compile.
          (void)Success;
        }

        combineMetadataForCSE(I1, I2, true);
        // I1 and I2 are being combined into a single instruction.  Its debug
        // location is the merged locations of the original instructions.
        I1->applyMergedLocation(I1->getDebugLoc(), I2->getDebugLoc());
        I2->eraseFromParent();
      }
      if (!Changed)
        NumHoistCommonCode += SuccIterPairs.size();
      Changed = true;
      NumHoistCommonInstrs += SuccIterPairs.size();
    } else {
      if (NumSkipped >= HoistCommonSkipLimit) {
        hoistLockstepIdenticalDbgVariableRecords(TI, I1, OtherInsts);
        return Changed;
      }
      // We are about to skip over a pair of non-identical instructions. Record
      // if any have characteristics that would prevent reordering instructions
      // across them.
      for (auto &SuccIterPair : SuccIterPairs) {
        Instruction *I = &*SuccIterPair.first++;
        SuccIterPair.second |= skippedInstrFlags(I);
      }
      ++NumSkipped;
    }
  }
}

bool SimplifyCFGOpt::hoistSuccIdenticalTerminatorToSwitchOrIf(
    Instruction *TI, Instruction *I1,
    SmallVectorImpl<Instruction *> &OtherSuccTIs) {

  auto *BI = dyn_cast<BranchInst>(TI);

  bool Changed = false;
  BasicBlock *TIParent = TI->getParent();
  BasicBlock *BB1 = I1->getParent();

  // Use only for an if statement.
  auto *I2 = *OtherSuccTIs.begin();
  auto *BB2 = I2->getParent();
  if (BI) {
    assert(OtherSuccTIs.size() == 1);
    assert(BI->getSuccessor(0) == I1->getParent());
    assert(BI->getSuccessor(1) == I2->getParent());
  }

  // In the case of an if statement, we try to hoist an invoke.
  // FIXME: Can we define a safety predicate for CallBr?
  // FIXME: Test case llvm/test/Transforms/SimplifyCFG/2009-06-15-InvokeCrash.ll
  // removed in 4c923b3b3fd0ac1edebf0603265ca3ba51724937 commit?
  if (isa<InvokeInst>(I1) && (!BI || !isSafeToHoistInvoke(BB1, BB2, I1, I2)))
    return false;

  // TODO: callbr hoisting currently disabled pending further study.
  if (isa<CallBrInst>(I1))
    return false;

  for (BasicBlock *Succ : successors(BB1)) {
    for (PHINode &PN : Succ->phis()) {
      Value *BB1V = PN.getIncomingValueForBlock(BB1);
      for (Instruction *OtherSuccTI : OtherSuccTIs) {
        Value *BB2V = PN.getIncomingValueForBlock(OtherSuccTI->getParent());
        if (BB1V == BB2V)
          continue;

        // In the case of an if statement, check for
        // passingValueIsAlwaysUndefined here because we would rather eliminate
        // undefined control flow then converting it to a select.
        if (!BI || passingValueIsAlwaysUndefined(BB1V, &PN) ||
            passingValueIsAlwaysUndefined(BB2V, &PN))
          return false;
      }
    }
  }

  // Hoist DbgVariableRecords attached to the terminator to match dbg.*
  // intrinsic hoisting behaviour in hoistCommonCodeFromSuccessors.
  hoistLockstepIdenticalDbgVariableRecords(TI, I1, OtherSuccTIs);
  // Clone the terminator and hoist it into the pred, without any debug info.
  Instruction *NT = I1->clone();
  NT->insertInto(TIParent, TI->getIterator());
  if (!NT->getType()->isVoidTy()) {
    I1->replaceAllUsesWith(NT);
    for (Instruction *OtherSuccTI : OtherSuccTIs)
      OtherSuccTI->replaceAllUsesWith(NT);
    NT->takeName(I1);
  }
  Changed = true;
  NumHoistCommonInstrs += OtherSuccTIs.size() + 1;

  // Ensure terminator gets a debug location, even an unknown one, in case
  // it involves inlinable calls.
  SmallVector<DebugLoc, 4> Locs;
  Locs.push_back(I1->getDebugLoc());
  for (auto *OtherSuccTI : OtherSuccTIs)
    Locs.push_back(OtherSuccTI->getDebugLoc());
  NT->setDebugLoc(DebugLoc::getMergedLocations(Locs));

  // PHIs created below will adopt NT's merged DebugLoc.
  IRBuilder<NoFolder> Builder(NT);

  // In the case of an if statement, hoisting one of the terminators from our
  // successor is a great thing. Unfortunately, the successors of the if/else
  // blocks may have PHI nodes in them.  If they do, all PHI entries for BB1/BB2
  // must agree for all PHI nodes, so we insert select instruction to compute
  // the final result.
  if (BI) {
    std::map<std::pair<Value *, Value *>, SelectInst *> InsertedSelects;
    for (BasicBlock *Succ : successors(BB1)) {
      for (PHINode &PN : Succ->phis()) {
        Value *BB1V = PN.getIncomingValueForBlock(BB1);
        Value *BB2V = PN.getIncomingValueForBlock(BB2);
        if (BB1V == BB2V)
          continue;

        // These values do not agree.  Insert a select instruction before NT
        // that determines the right value.
        SelectInst *&SI = InsertedSelects[std::make_pair(BB1V, BB2V)];
        if (!SI) {
          // Propagate fast-math-flags from phi node to its replacement select.
          SI = cast<SelectInst>(Builder.CreateSelectFMF(
              BI->getCondition(), BB1V, BB2V,
              isa<FPMathOperator>(PN) ? &PN : nullptr,
              BB1V->getName() + "." + BB2V->getName(), BI));
        }

        // Make the PHI node use the select for all incoming values for BB1/BB2
        for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i)
          if (PN.getIncomingBlock(i) == BB1 || PN.getIncomingBlock(i) == BB2)
            PN.setIncomingValue(i, SI);
      }
    }
  }

  SmallVector<DominatorTree::UpdateType, 4> Updates;

  // Update any PHI nodes in our new successors.
  for (BasicBlock *Succ : successors(BB1)) {
    addPredecessorToBlock(Succ, TIParent, BB1);
    if (DTU)
      Updates.push_back({DominatorTree::Insert, TIParent, Succ});
  }

  if (DTU)
    for (BasicBlock *Succ : successors(TI))
      Updates.push_back({DominatorTree::Delete, TIParent, Succ});

  eraseTerminatorAndDCECond(TI);
  if (DTU)
    DTU->applyUpdates(Updates);
  return Changed;
}

// TODO: Refine this. This should avoid cases like turning constant memcpy sizes
// into variables.
static bool replacingOperandWithVariableIsCheap(const Instruction *I,
                                                int OpIdx) {
  // Divide/Remainder by constant is typically much cheaper than by variable.
  if (I->isIntDivRem())
    return OpIdx != 1;
  return !isa<IntrinsicInst>(I);
}

// All instructions in Insts belong to different blocks that all unconditionally
// branch to a common successor. Analyze each instruction and return true if it
// would be possible to sink them into their successor, creating one common
// instruction instead. For every value that would be required to be provided by
// PHI node (because an operand varies in each input block), add to PHIOperands.
static bool canSinkInstructions(
    ArrayRef<Instruction *> Insts,
    DenseMap<const Use *, SmallVector<Value *, 4>> &PHIOperands) {
  // Prune out obviously bad instructions to move. Each instruction must have
  // the same number of uses, and we check later that the uses are consistent.
  std::optional<unsigned> NumUses;
  for (auto *I : Insts) {
    // These instructions may change or break semantics if moved.
    if (isa<PHINode>(I) || I->isEHPad() || isa<AllocaInst>(I) ||
        I->getType()->isTokenTy())
      return false;

    // Do not try to sink an instruction in an infinite loop - it can cause
    // this algorithm to infinite loop.
    if (I->getParent()->getSingleSuccessor() == I->getParent())
      return false;

    // Conservatively return false if I is an inline-asm instruction. Sinking
    // and merging inline-asm instructions can potentially create arguments
    // that cannot satisfy the inline-asm constraints.
    // If the instruction has nomerge or convergent attribute, return false.
    if (const auto *C = dyn_cast<CallBase>(I))
      if (C->isInlineAsm() || C->cannotMerge() || C->isConvergent())
        return false;

    if (!NumUses)
      NumUses = I->getNumUses();
    else if (NumUses != I->getNumUses())
      return false;
  }

  const Instruction *I0 = Insts.front();
  const auto I0MMRA = MMRAMetadata(*I0);
  for (auto *I : Insts) {
    if (!I->isSameOperationAs(I0, Instruction::CompareUsingIntersectedAttrs))
      return false;

    // Treat MMRAs conservatively. This pass can be quite aggressive and
    // could drop a lot of MMRAs otherwise.
    if (MMRAMetadata(*I) != I0MMRA)
      return false;
  }

  // Uses must be consistent: If I0 is used in a phi node in the sink target,
  // then the other phi operands must match the instructions from Insts. This
  // also has to hold true for any phi nodes that would be created as a result
  // of sinking. Both of these cases are represented by PhiOperands.
  for (const Use &U : I0->uses()) {
    auto It = PHIOperands.find(&U);
    if (It == PHIOperands.end())
      // There may be uses in other blocks when sinking into a loop header.
      return false;
    if (!equal(Insts, It->second))
      return false;
  }

  // For calls to be sinkable, they must all be indirect, or have same callee.
  // I.e. if we have two direct calls to different callees, we don't want to
  // turn that into an indirect call. Likewise, if we have an indirect call,
  // and a direct call, we don't actually want to have a single indirect call.
  if (isa<CallBase>(I0)) {
    auto IsIndirectCall = [](const Instruction *I) {
      return cast<CallBase>(I)->isIndirectCall();
    };
    bool HaveIndirectCalls = any_of(Insts, IsIndirectCall);
    bool AllCallsAreIndirect = all_of(Insts, IsIndirectCall);
    if (HaveIndirectCalls) {
      if (!AllCallsAreIndirect)
        return false;
    } else {
      // All callees must be identical.
      Value *Callee = nullptr;
      for (const Instruction *I : Insts) {
        Value *CurrCallee = cast<CallBase>(I)->getCalledOperand();
        if (!Callee)
          Callee = CurrCallee;
        else if (Callee != CurrCallee)
          return false;
      }
    }
  }

  for (unsigned OI = 0, OE = I0->getNumOperands(); OI != OE; ++OI) {
    Value *Op = I0->getOperand(OI);
    auto SameAsI0 = [&I0, OI](const Instruction *I) {
      assert(I->getNumOperands() == I0->getNumOperands());
      return I->getOperand(OI) == I0->getOperand(OI);
    };
    if (!all_of(Insts, SameAsI0)) {
      if ((isa<Constant>(Op) && !replacingOperandWithVariableIsCheap(I0, OI)) ||
          !canReplaceOperandWithVariable(I0, OI))
        // We can't create a PHI from this GEP.
        return false;
      auto &Ops = PHIOperands[&I0->getOperandUse(OI)];
      for (auto *I : Insts)
        Ops.push_back(I->getOperand(OI));
    }
  }
  return true;
}

// Assuming canSinkInstructions(Blocks) has returned true, sink the last
// instruction of every block in Blocks to their common successor, commoning
// into one instruction.
static void sinkLastInstruction(ArrayRef<BasicBlock*> Blocks) {
  auto *BBEnd = Blocks[0]->getTerminator()->getSuccessor(0);

  // canSinkInstructions returning true guarantees that every block has at
  // least one non-terminator instruction.
  SmallVector<Instruction*,4> Insts;
  for (auto *BB : Blocks) {
    Instruction *I = BB->getTerminator();
    I = I->getPrevNode();
    Insts.push_back(I);
  }

  // We don't need to do any more checking here; canSinkInstructions should
  // have done it all for us.
  SmallVector<Value*, 4> NewOperands;
  Instruction *I0 = Insts.front();
  for (unsigned O = 0, E = I0->getNumOperands(); O != E; ++O) {
    // This check is different to that in canSinkInstructions. There, we
    // cared about the global view once simplifycfg (and instcombine) have
    // completed - it takes into account PHIs that become trivially
    // simplifiable.  However here we need a more local view; if an operand
    // differs we create a PHI and rely on instcombine to clean up the very
    // small mess we may make.
    bool NeedPHI = any_of(Insts, [&I0, O](const Instruction *I) {
      return I->getOperand(O) != I0->getOperand(O);
    });
    if (!NeedPHI) {
      NewOperands.push_back(I0->getOperand(O));
      continue;
    }

    // Create a new PHI in the successor block and populate it.
    auto *Op = I0->getOperand(O);
    assert(!Op->getType()->isTokenTy() && "Can't PHI tokens!");
    auto *PN =
        PHINode::Create(Op->getType(), Insts.size(), Op->getName() + ".sink");
    PN->insertBefore(BBEnd->begin());
    for (auto *I : Insts)
      PN->addIncoming(I->getOperand(O), I->getParent());
    NewOperands.push_back(PN);
  }

  // Arbitrarily use I0 as the new "common" instruction; remap its operands
  // and move it to the start of the successor block.
  for (unsigned O = 0, E = I0->getNumOperands(); O != E; ++O)
    I0->getOperandUse(O).set(NewOperands[O]);

  I0->moveBefore(*BBEnd, BBEnd->getFirstInsertionPt());

  // Update metadata and IR flags, and merge debug locations.
  for (auto *I : Insts)
    if (I != I0) {
      // The debug location for the "common" instruction is the merged locations
      // of all the commoned instructions.  We start with the original location
      // of the "common" instruction and iteratively merge each location in the
      // loop below.
      // This is an N-way merge, which will be inefficient if I0 is a CallInst.
      // However, as N-way merge for CallInst is rare, so we use simplified API
      // instead of using complex API for N-way merge.
      I0->applyMergedLocation(I0->getDebugLoc(), I->getDebugLoc());
      combineMetadataForCSE(I0, I, true);
      I0->andIRFlags(I);
      if (auto *CB = dyn_cast<CallBase>(I0)) {
        bool Success = CB->tryIntersectAttributes(cast<CallBase>(I));
        assert(Success && "We should not be trying to sink callbases "
                          "with non-intersectable attributes");
        // For NDEBUG Compile.
        (void)Success;
      }
    }

  for (User *U : make_early_inc_range(I0->users())) {
    // canSinkLastInstruction checked that all instructions are only used by
    // phi nodes in a way that allows replacing the phi node with the common
    // instruction.
    auto *PN = cast<PHINode>(U);
    PN->replaceAllUsesWith(I0);
    PN->eraseFromParent();
  }

  // Finally nuke all instructions apart from the common instruction.
  for (auto *I : Insts) {
    if (I == I0)
      continue;
    // The remaining uses are debug users, replace those with the common inst.
    // In most (all?) cases this just introduces a use-before-def.
    assert(I->user_empty() && "Inst unexpectedly still has non-dbg users");
    I->replaceAllUsesWith(I0);
    I->eraseFromParent();
  }
}

/// Check whether BB's predecessors end with unconditional branches. If it is
/// true, sink any common code from the predecessors to BB.
static bool sinkCommonCodeFromPredecessors(BasicBlock *BB,
                                           DomTreeUpdater *DTU) {
  // We support two situations:
  //   (1) all incoming arcs are unconditional
  //   (2) there are non-unconditional incoming arcs
  //
  // (2) is very common in switch defaults and
  // else-if patterns;
  //
  //   if (a) f(1);
  //   else if (b) f(2);
  //
  // produces:
  //
  //       [if]
  //      /    \
  //    [f(1)] [if]
  //      |     | \
  //      |     |  |
  //      |  [f(2)]|
  //       \    | /
  //        [ end ]
  //
  // [end] has two unconditional predecessor arcs and one conditional. The
  // conditional refers to the implicit empty 'else' arc. This conditional
  // arc can also be caused by an empty default block in a switch.
  //
  // In this case, we attempt to sink code from all *unconditional* arcs.
  // If we can sink instructions from these arcs (determined during the scan
  // phase below) we insert a common successor for all unconditional arcs and
  // connect that to [end], to enable sinking:
  //
  //       [if]
  //      /    \
  //    [x(1)] [if]
  //      |     | \
  //      |     |  \
  //      |  [x(2)] |
  //       \   /    |
  //   [sink.split] |
  //         \     /
  //         [ end ]
  //
  SmallVector<BasicBlock*,4> UnconditionalPreds;
  bool HaveNonUnconditionalPredecessors = false;
  for (auto *PredBB : predecessors(BB)) {
    auto *PredBr = dyn_cast<BranchInst>(PredBB->getTerminator());
    if (PredBr && PredBr->isUnconditional())
      UnconditionalPreds.push_back(PredBB);
    else
      HaveNonUnconditionalPredecessors = true;
  }
  if (UnconditionalPreds.size() < 2)
    return false;

  // We take a two-step approach to tail sinking. First we scan from the end of
  // each block upwards in lockstep. If the n'th instruction from the end of each
  // block can be sunk, those instructions are added to ValuesToSink and we
  // carry on. If we can sink an instruction but need to PHI-merge some operands
  // (because they're not identical in each instruction) we add these to
  // PHIOperands.
  // We prepopulate PHIOperands with the phis that already exist in BB.
  DenseMap<const Use *, SmallVector<Value *, 4>> PHIOperands;
  for (PHINode &PN : BB->phis()) {
    SmallDenseMap<BasicBlock *, const Use *, 4> IncomingVals;
    for (const Use &U : PN.incoming_values())
      IncomingVals.insert({PN.getIncomingBlock(U), &U});
    auto &Ops = PHIOperands[IncomingVals[UnconditionalPreds[0]]];
    for (BasicBlock *Pred : UnconditionalPreds)
      Ops.push_back(*IncomingVals[Pred]);
  }

  int ScanIdx = 0;
  SmallPtrSet<Value*,4> InstructionsToSink;
  LockstepReverseIterator<true> LRI(UnconditionalPreds);
  while (LRI.isValid() &&
         canSinkInstructions(*LRI, PHIOperands)) {
    LLVM_DEBUG(dbgs() << "SINK: instruction can be sunk: " << *(*LRI)[0]
                      << "\n");
    InstructionsToSink.insert_range(*LRI);
    ++ScanIdx;
    --LRI;
  }

  // If no instructions can be sunk, early-return.
  if (ScanIdx == 0)
    return false;

  bool followedByDeoptOrUnreachable = IsBlockFollowedByDeoptOrUnreachable(BB);

  if (!followedByDeoptOrUnreachable) {
    // Check whether this is the pointer operand of a load/store.
    auto IsMemOperand = [](Use &U) {
      auto *I = cast<Instruction>(U.getUser());
      if (isa<LoadInst>(I))
        return U.getOperandNo() == LoadInst::getPointerOperandIndex();
      if (isa<StoreInst>(I))
        return U.getOperandNo() == StoreInst::getPointerOperandIndex();
      return false;
    };

    // Okay, we *could* sink last ScanIdx instructions. But how many can we
    // actually sink before encountering instruction that is unprofitable to
    // sink?
    auto ProfitableToSinkInstruction = [&](LockstepReverseIterator<true> &LRI) {
      unsigned NumPHIInsts = 0;
      for (Use &U : (*LRI)[0]->operands()) {
        auto It = PHIOperands.find(&U);
        if (It != PHIOperands.end() && !all_of(It->second, [&](Value *V) {
              return InstructionsToSink.contains(V);
            })) {
          ++NumPHIInsts;
          // Do not separate a load/store from the gep producing the address.
          // The gep can likely be folded into the load/store as an addressing
          // mode. Additionally, a load of a gep is easier to analyze than a
          // load of a phi.
          if (IsMemOperand(U) &&
              any_of(It->second, [](Value *V) { return isa<GEPOperator>(V); }))
            return false;
          // FIXME: this check is overly optimistic. We may end up not sinking
          // said instruction, due to the very same profitability check.
          // See @creating_too_many_phis in sink-common-code.ll.
        }
      }
      LLVM_DEBUG(dbgs() << "SINK: #phi insts: " << NumPHIInsts << "\n");
      return NumPHIInsts <= 1;
    };

    // We've determined that we are going to sink last ScanIdx instructions,
    // and recorded them in InstructionsToSink. Now, some instructions may be
    // unprofitable to sink. But that determination depends on the instructions
    // that we are going to sink.

    // First, forward scan: find the first instruction unprofitable to sink,
    // recording all the ones that are profitable to sink.
    // FIXME: would it be better, after we detect that not all are profitable.
    // to either record the profitable ones, or erase the unprofitable ones?
    // Maybe we need to choose (at runtime) the one that will touch least
    // instrs?
    LRI.reset();
    int Idx = 0;
    SmallPtrSet<Value *, 4> InstructionsProfitableToSink;
    while (Idx < ScanIdx) {
      if (!ProfitableToSinkInstruction(LRI)) {
        // Too many PHIs would be created.
        LLVM_DEBUG(
            dbgs() << "SINK: stopping here, too many PHIs would be created!\n");
        break;
      }
      InstructionsProfitableToSink.insert_range(*LRI);
      --LRI;
      ++Idx;
    }

    // If no instructions can be sunk, early-return.
    if (Idx == 0)
      return false;

    // Did we determine that (only) some instructions are unprofitable to sink?
    if (Idx < ScanIdx) {
      // Okay, some instructions are unprofitable.
      ScanIdx = Idx;
      InstructionsToSink = InstructionsProfitableToSink;

      // But, that may make other instructions unprofitable, too.
      // So, do a backward scan, do any earlier instructions become
      // unprofitable?
      assert(
          !ProfitableToSinkInstruction(LRI) &&
          "We already know that the last instruction is unprofitable to sink");
      ++LRI;
      --Idx;
      while (Idx >= 0) {
        // If we detect that an instruction becomes unprofitable to sink,
        // all earlier instructions won't be sunk either,
        // so preemptively keep InstructionsProfitableToSink in sync.
        // FIXME: is this the most performant approach?
        for (auto *I : *LRI)
          InstructionsProfitableToSink.erase(I);
        if (!ProfitableToSinkInstruction(LRI)) {
          // Everything starting with this instruction won't be sunk.
          ScanIdx = Idx;
          InstructionsToSink = InstructionsProfitableToSink;
        }
        ++LRI;
        --Idx;
      }
    }

    // If no instructions can be sunk, early-return.
    if (ScanIdx == 0)
      return false;
  }

  bool Changed = false;

  if (HaveNonUnconditionalPredecessors) {
    if (!followedByDeoptOrUnreachable) {
      // It is always legal to sink common instructions from unconditional
      // predecessors. However, if not all predecessors are unconditional,
      // this transformation might be pessimizing. So as a rule of thumb,
      // don't do it unless we'd sink at least one non-speculatable instruction.
      // See https://bugs.llvm.org/show_bug.cgi?id=30244
      LRI.reset();
      int Idx = 0;
      bool Profitable = false;
      while (Idx < ScanIdx) {
        if (!isSafeToSpeculativelyExecute((*LRI)[0])) {
          Profitable = true;
          break;
        }
        --LRI;
        ++Idx;
      }
      if (!Profitable)
        return false;
    }

    LLVM_DEBUG(dbgs() << "SINK: Splitting edge\n");
    // We have a conditional edge and we're going to sink some instructions.
    // Insert a new block postdominating all blocks we're going to sink from.
    if (!SplitBlockPredecessors(BB, UnconditionalPreds, ".sink.split", DTU))
      // Edges couldn't be split.
      return false;
    Changed = true;
  }

  // Now that we've analyzed all potential sinking candidates, perform the
  // actual sink. We iteratively sink the last non-terminator of the source
  // blocks into their common successor unless doing so would require too
  // many PHI instructions to be generated (currently only one PHI is allowed
  // per sunk instruction).
  //
  // We can use InstructionsToSink to discount values needing PHI-merging that will
  // actually be sunk in a later iteration. This allows us to be more
  // aggressive in what we sink. This does allow a false positive where we
  // sink presuming a later value will also be sunk, but stop half way through
  // and never actually sink it which means we produce more PHIs than intended.
  // This is unlikely in practice though.
  int SinkIdx = 0;
  for (; SinkIdx != ScanIdx; ++SinkIdx) {
    LLVM_DEBUG(dbgs() << "SINK: Sink: "
                      << *UnconditionalPreds[0]->getTerminator()->getPrevNode()
                      << "\n");

    // Because we've sunk every instruction in turn, the current instruction to
    // sink is always at index 0.
    LRI.reset();

    sinkLastInstruction(UnconditionalPreds);
    NumSinkCommonInstrs++;
    Changed = true;
  }
  if (SinkIdx != 0)
    ++NumSinkCommonCode;
  return Changed;
}

namespace {

struct CompatibleSets {
  using SetTy = SmallVector<InvokeInst *, 2>;

  SmallVector<SetTy, 1> Sets;

  static bool shouldBelongToSameSet(ArrayRef<InvokeInst *> Invokes);

  SetTy &getCompatibleSet(InvokeInst *II);

  void insert(InvokeInst *II);
};

CompatibleSets::SetTy &CompatibleSets::getCompatibleSet(InvokeInst *II) {
  // Perform a linear scan over all the existing sets, see if the new `invoke`
  // is compatible with any particular set. Since we know that all the `invokes`
  // within a set are compatible, only check the first `invoke` in each set.
  // WARNING: at worst, this has quadratic complexity.
  for (CompatibleSets::SetTy &Set : Sets) {
    if (CompatibleSets::shouldBelongToSameSet({Set.front(), II}))
      return Set;
  }

  // Otherwise, we either had no sets yet, or this invoke forms a new set.
  return Sets.emplace_back();
}

void CompatibleSets::insert(InvokeInst *II) {
  getCompatibleSet(II).emplace_back(II);
}

bool CompatibleSets::shouldBelongToSameSet(ArrayRef<InvokeInst *> Invokes) {
  assert(Invokes.size() == 2 && "Always called with exactly two candidates.");

  // Can we theoretically merge these `invoke`s?
  auto IsIllegalToMerge = [](InvokeInst *II) {
    return II->cannotMerge() || II->isInlineAsm();
  };
  if (any_of(Invokes, IsIllegalToMerge))
    return false;

  // Either both `invoke`s must be   direct,
  // or     both `invoke`s must be indirect.
  auto IsIndirectCall = [](InvokeInst *II) { return II->isIndirectCall(); };
  bool HaveIndirectCalls = any_of(Invokes, IsIndirectCall);
  bool AllCallsAreIndirect = all_of(Invokes, IsIndirectCall);
  if (HaveIndirectCalls) {
    if (!AllCallsAreIndirect)
      return false;
  } else {
    // All callees must be identical.
    Value *Callee = nullptr;
    for (InvokeInst *II : Invokes) {
      Value *CurrCallee = II->getCalledOperand();
      assert(CurrCallee && "There is always a called operand.");
      if (!Callee)
        Callee = CurrCallee;
      else if (Callee != CurrCallee)
        return false;
    }
  }

  // Either both `invoke`s must not have a normal destination,
  // or     both `invoke`s must     have a normal destination,
  auto HasNormalDest = [](InvokeInst *II) {
    return !isa<UnreachableInst>(II->getNormalDest()->getFirstNonPHIOrDbg());
  };
  if (any_of(Invokes, HasNormalDest)) {
    // Do not merge `invoke` that does not have a normal destination with one
    // that does have a normal destination, even though doing so would be legal.
    if (!all_of(Invokes, HasNormalDest))
      return false;

    // All normal destinations must be identical.
    BasicBlock *NormalBB = nullptr;
    for (InvokeInst *II : Invokes) {
      BasicBlock *CurrNormalBB = II->getNormalDest();
      assert(CurrNormalBB && "There is always a 'continue to' basic block.");
      if (!NormalBB)
        NormalBB = CurrNormalBB;
      else if (NormalBB != CurrNormalBB)
        return false;
    }

    // In the normal destination, the incoming values for these two `invoke`s
    // must be compatible.
    SmallPtrSet<Value *, 16> EquivalenceSet(llvm::from_range, Invokes);
    if (!incomingValuesAreCompatible(
            NormalBB, {Invokes[0]->getParent(), Invokes[1]->getParent()},
            &EquivalenceSet))
      return false;
  }

#ifndef NDEBUG
  // All unwind destinations must be identical.
  // We know that because we have started from said unwind destination.
  BasicBlock *UnwindBB = nullptr;
  for (InvokeInst *II : Invokes) {
    BasicBlock *CurrUnwindBB = II->getUnwindDest();
    assert(CurrUnwindBB && "There is always an 'unwind to' basic block.");
    if (!UnwindBB)
      UnwindBB = CurrUnwindBB;
    else
      assert(UnwindBB == CurrUnwindBB && "Unexpected unwind destination.");
  }
#endif

  // In the unwind destination, the incoming values for these two `invoke`s
  // must be compatible.
  if (!incomingValuesAreCompatible(
          Invokes.front()->getUnwindDest(),
          {Invokes[0]->getParent(), Invokes[1]->getParent()}))
    return false;

  // Ignoring arguments, these `invoke`s must be identical,
  // including operand bundles.
  const InvokeInst *II0 = Invokes.front();
  for (auto *II : Invokes.drop_front())
    if (!II->isSameOperationAs(II0, Instruction::CompareUsingIntersectedAttrs))
      return false;

  // Can we theoretically form the data operands for the merged `invoke`?
  auto IsIllegalToMergeArguments = [](auto Ops) {
    Use &U0 = std::get<0>(Ops);
    Use &U1 = std::get<1>(Ops);
    if (U0 == U1)
      return false;
    return !canReplaceOperandWithVariable(cast<Instruction>(U0.getUser()),
                                          U0.getOperandNo());
  };
  assert(Invokes.size() == 2 && "Always called with exactly two candidates.");
  if (any_of(zip(Invokes[0]->data_ops(), Invokes[1]->data_ops()),
             IsIllegalToMergeArguments))
    return false;

  return true;
}

} // namespace

// Merge all invokes in the provided set, all of which are compatible
// as per the `CompatibleSets::shouldBelongToSameSet()`.
static void mergeCompatibleInvokesImpl(ArrayRef<InvokeInst *> Invokes,
                                       DomTreeUpdater *DTU) {
  assert(Invokes.size() >= 2 && "Must have at least two invokes to merge.");

  SmallVector<DominatorTree::UpdateType, 8> Updates;
  if (DTU)
    Updates.reserve(2 + 3 * Invokes.size());

  bool HasNormalDest =
      !isa<UnreachableInst>(Invokes[0]->getNormalDest()->getFirstNonPHIOrDbg());

  // Clone one of the invokes into a new basic block.
  // Since they are all compatible, it doesn't matter which invoke is cloned.
  InvokeInst *MergedInvoke = [&Invokes, HasNormalDest]() {
    InvokeInst *II0 = Invokes.front();
    BasicBlock *II0BB = II0->getParent();
    BasicBlock *InsertBeforeBlock =
        II0->getParent()->getIterator()->getNextNode();
    Function *Func = II0BB->getParent();
    LLVMContext &Ctx = II0->getContext();

    BasicBlock *MergedInvokeBB = BasicBlock::Create(
        Ctx, II0BB->getName() + ".invoke", Func, InsertBeforeBlock);

    auto *MergedInvoke = cast<InvokeInst>(II0->clone());
    // NOTE: all invokes have the same attributes, so no handling needed.
    MergedInvoke->insertInto(MergedInvokeBB, MergedInvokeBB->end());

    if (!HasNormalDest) {
      // This set does not have a normal destination,
      // so just form a new block with unreachable terminator.
      BasicBlock *MergedNormalDest = BasicBlock::Create(
          Ctx, II0BB->getName() + ".cont", Func, InsertBeforeBlock);
      auto *UI = new UnreachableInst(Ctx, MergedNormalDest);
      UI->setDebugLoc(DebugLoc::getTemporary());
      MergedInvoke->setNormalDest(MergedNormalDest);
    }

    // The unwind destination, however, remainds identical for all invokes here.

    return MergedInvoke;
  }();

  if (DTU) {
    // Predecessor blocks that contained these invokes will now branch to
    // the new block that contains the merged invoke, ...
    for (InvokeInst *II : Invokes)
      Updates.push_back(
          {DominatorTree::Insert, II->getParent(), MergedInvoke->getParent()});

    // ... which has the new `unreachable` block as normal destination,
    // or unwinds to the (same for all `invoke`s in this set) `landingpad`,
    for (BasicBlock *SuccBBOfMergedInvoke : successors(MergedInvoke))
      Updates.push_back({DominatorTree::Insert, MergedInvoke->getParent(),
                         SuccBBOfMergedInvoke});

    // Since predecessor blocks now unconditionally branch to a new block,
    // they no longer branch to their original successors.
    for (InvokeInst *II : Invokes)
      for (BasicBlock *SuccOfPredBB : successors(II->getParent()))
        Updates.push_back(
            {DominatorTree::Delete, II->getParent(), SuccOfPredBB});
  }

  bool IsIndirectCall = Invokes[0]->isIndirectCall();

  // Form the merged operands for the merged invoke.
  for (Use &U : MergedInvoke->operands()) {
    // Only PHI together the indirect callees and data operands.
    if (MergedInvoke->isCallee(&U)) {
      if (!IsIndirectCall)
        continue;
    } else if (!MergedInvoke->isDataOperand(&U))
      continue;

    // Don't create trivial PHI's with all-identical incoming values.
    bool NeedPHI = any_of(Invokes, [&U](InvokeInst *II) {
      return II->getOperand(U.getOperandNo()) != U.get();
    });
    if (!NeedPHI)
      continue;

    // Form a PHI out of all the data ops under this index.
    PHINode *PN = PHINode::Create(
        U->getType(), /*NumReservedValues=*/Invokes.size(), "", MergedInvoke->getIterator());
    for (InvokeInst *II : Invokes)
      PN->addIncoming(II->getOperand(U.getOperandNo()), II->getParent());

    U.set(PN);
  }

  // We've ensured that each PHI node has compatible (identical) incoming values
  // when coming from each of the `invoke`s in the current merge set,
  // so update the PHI nodes accordingly.
  for (BasicBlock *Succ : successors(MergedInvoke))
    addPredecessorToBlock(Succ, /*NewPred=*/MergedInvoke->getParent(),
                          /*ExistPred=*/Invokes.front()->getParent());

  // And finally, replace the original `invoke`s with an unconditional branch
  // to the block with the merged `invoke`. Also, give that merged `invoke`
  // the merged debugloc of all the original `invoke`s.
  DILocation *MergedDebugLoc = nullptr;
  for (InvokeInst *II : Invokes) {
    // Compute the debug location common to all the original `invoke`s.
    if (!MergedDebugLoc)
      MergedDebugLoc = II->getDebugLoc();
    else
      MergedDebugLoc =
          DebugLoc::getMergedLocation(MergedDebugLoc, II->getDebugLoc());

    // And replace the old `invoke` with an unconditionally branch
    // to the block with the merged `invoke`.
    for (BasicBlock *OrigSuccBB : successors(II->getParent()))
      OrigSuccBB->removePredecessor(II->getParent());
    auto *BI = BranchInst::Create(MergedInvoke->getParent(), II->getParent());
    // The unconditional branch is part of the replacement for the original
    // invoke, so should use its DebugLoc.
    BI->setDebugLoc(II->getDebugLoc());
    bool Success = MergedInvoke->tryIntersectAttributes(II);
    assert(Success && "Merged invokes with incompatible attributes");
    // For NDEBUG Compile
    (void)Success;
    II->replaceAllUsesWith(MergedInvoke);
    II->eraseFromParent();
    ++NumInvokesMerged;
  }
  MergedInvoke->setDebugLoc(MergedDebugLoc);
  ++NumInvokeSetsFormed;

  if (DTU)
    DTU->applyUpdates(Updates);
}

/// If this block is a `landingpad` exception handling block, categorize all
/// the predecessor `invoke`s into sets, with all `invoke`s in each set
/// being "mergeable" together, and then merge invokes in each set together.
///
/// This is a weird mix of hoisting and sinking. Visually, it goes from:
///          [...]        [...]
///            |            |
///        [invoke0]    [invoke1]
///           / \          / \
///     [cont0] [landingpad] [cont1]
/// to:
///      [...] [...]
///          \ /
///       [invoke]
///          / \
///     [cont] [landingpad]
///
/// But of course we can only do that if the invokes share the `landingpad`,
/// edges invoke0->cont0 and invoke1->cont1 are "compatible",
/// and the invoked functions are "compatible".
static bool mergeCompatibleInvokes(BasicBlock *BB, DomTreeUpdater *DTU) {
  if (!EnableMergeCompatibleInvokes)
    return false;

  bool Changed = false;

  // FIXME: generalize to all exception handling blocks?
  if (!BB->isLandingPad())
    return Changed;

  CompatibleSets Grouper;

  // Record all the predecessors of this `landingpad`. As per verifier,
  // the only allowed predecessor is the unwind edge of an `invoke`.
  // We want to group "compatible" `invokes` into the same set to be merged.
  for (BasicBlock *PredBB : predecessors(BB))
    Grouper.insert(cast<InvokeInst>(PredBB->getTerminator()));

  // And now, merge `invoke`s that were grouped togeter.
  for (ArrayRef<InvokeInst *> Invokes : Grouper.Sets) {
    if (Invokes.size() < 2)
      continue;
    Changed = true;
    mergeCompatibleInvokesImpl(Invokes, DTU);
  }

  return Changed;
}

namespace {
/// Track ephemeral values, which should be ignored for cost-modelling
/// purposes. Requires walking instructions in reverse order.
class EphemeralValueTracker {
  SmallPtrSet<const Instruction *, 32> EphValues;

  bool isEphemeral(const Instruction *I) {
    if (isa<AssumeInst>(I))
      return true;
    return !I->mayHaveSideEffects() && !I->isTerminator() &&
           all_of(I->users(), [&](const User *U) {
             return EphValues.count(cast<Instruction>(U));
           });
  }

public:
  bool track(const Instruction *I) {
    if (isEphemeral(I)) {
      EphValues.insert(I);
      return true;
    }
    return false;
  }

  bool contains(const Instruction *I) const { return EphValues.contains(I); }
};
} // namespace

/// Determine if we can hoist sink a sole store instruction out of a
/// conditional block.
///
/// We are looking for code like the following:
///   BrBB:
///     store i32 %add, i32* %arrayidx2
///     ... // No other stores or function calls (we could be calling a memory
///     ... // function).
///     %cmp = icmp ult %x, %y
///     br i1 %cmp, label %EndBB, label %ThenBB
///   ThenBB:
///     store i32 %add5, i32* %arrayidx2
///     br label EndBB
///   EndBB:
///     ...
///   We are going to transform this into:
///   BrBB:
///     store i32 %add, i32* %arrayidx2
///     ... //
///     %cmp = icmp ult %x, %y
///     %add.add5 = select i1 %cmp, i32 %add, %add5
///     store i32 %add.add5, i32* %arrayidx2
///     ...
///
/// \return The pointer to the value of the previous store if the store can be
///         hoisted into the predecessor block. 0 otherwise.
static Value *isSafeToSpeculateStore(Instruction *I, BasicBlock *BrBB,
                                     BasicBlock *StoreBB, BasicBlock *EndBB) {
  StoreInst *StoreToHoist = dyn_cast<StoreInst>(I);
  if (!StoreToHoist)
    return nullptr;

  // Volatile or atomic.
  if (!StoreToHoist->isSimple())
    return nullptr;

  Value *StorePtr = StoreToHoist->getPointerOperand();
  Type *StoreTy = StoreToHoist->getValueOperand()->getType();

  // Look for a store to the same pointer in BrBB.
  unsigned MaxNumInstToLookAt = 9;
  // Skip pseudo probe intrinsic calls which are not really killing any memory
  // accesses.
  for (Instruction &CurI : reverse(BrBB->instructionsWithoutDebug(true))) {
    if (!MaxNumInstToLookAt)
      break;
    --MaxNumInstToLookAt;

    // Could be calling an instruction that affects memory like free().
    if (CurI.mayWriteToMemory() && !isa<StoreInst>(CurI))
      return nullptr;

    if (auto *SI = dyn_cast<StoreInst>(&CurI)) {
      // Found the previous store to same location and type. Make sure it is
      // simple, to avoid introducing a spurious non-atomic write after an
      // atomic write.
      if (SI->getPointerOperand() == StorePtr &&
          SI->getValueOperand()->getType() == StoreTy && SI->isSimple() &&
          SI->getAlign() >= StoreToHoist->getAlign())
        // Found the previous store, return its value operand.
        return SI->getValueOperand();
      return nullptr; // Unknown store.
    }

    if (auto *LI = dyn_cast<LoadInst>(&CurI)) {
      if (LI->getPointerOperand() == StorePtr && LI->getType() == StoreTy &&
          LI->isSimple() && LI->getAlign() >= StoreToHoist->getAlign()) {
        Value *Obj = getUnderlyingObject(StorePtr);
        bool ExplicitlyDereferenceableOnly;
        if (isWritableObject(Obj, ExplicitlyDereferenceableOnly) &&
            capturesNothing(
                PointerMayBeCaptured(Obj, /*ReturnCaptures=*/false,
                                     CaptureComponents::Provenance)) &&
            (!ExplicitlyDereferenceableOnly ||
             isDereferenceablePointer(StorePtr, StoreTy,
                                      LI->getDataLayout()))) {
          // Found a previous load, return it.
          return LI;
        }
      }
      // The load didn't work out, but we may still find a store.
    }
  }

  return nullptr;
}

/// Estimate the cost of the insertion(s) and check that the PHI nodes can be
/// converted to selects.
static bool validateAndCostRequiredSelects(BasicBlock *BB, BasicBlock *ThenBB,
                                           BasicBlock *EndBB,
                                           unsigned &SpeculatedInstructions,
                                           InstructionCost &Cost,
                                           const TargetTransformInfo &TTI) {
  TargetTransformInfo::TargetCostKind CostKind =
    BB->getParent()->hasMinSize()
    ? TargetTransformInfo::TCK_CodeSize
    : TargetTransformInfo::TCK_SizeAndLatency;

  bool HaveRewritablePHIs = false;
  for (PHINode &PN : EndBB->phis()) {
    Value *OrigV = PN.getIncomingValueForBlock(BB);
    Value *ThenV = PN.getIncomingValueForBlock(ThenBB);

    // FIXME: Try to remove some of the duplication with
    // hoistCommonCodeFromSuccessors. Skip PHIs which are trivial.
    if (ThenV == OrigV)
      continue;

    Cost += TTI.getCmpSelInstrCost(Instruction::Select, PN.getType(),
                                   CmpInst::makeCmpResultType(PN.getType()),
                                   CmpInst::BAD_ICMP_PREDICATE, CostKind);

    // Don't convert to selects if we could remove undefined behavior instead.
    if (passingValueIsAlwaysUndefined(OrigV, &PN) ||
        passingValueIsAlwaysUndefined(ThenV, &PN))
      return false;

    HaveRewritablePHIs = true;
    ConstantExpr *OrigCE = dyn_cast<ConstantExpr>(OrigV);
    ConstantExpr *ThenCE = dyn_cast<ConstantExpr>(ThenV);
    if (!OrigCE && !ThenCE)
      continue; // Known cheap (FIXME: Maybe not true for aggregates).

    InstructionCost OrigCost = OrigCE ? computeSpeculationCost(OrigCE, TTI) : 0;
    InstructionCost ThenCost = ThenCE ? computeSpeculationCost(ThenCE, TTI) : 0;
    InstructionCost MaxCost =
        2 * PHINodeFoldingThreshold * TargetTransformInfo::TCC_Basic;
    if (OrigCost + ThenCost > MaxCost)
      return false;

    // Account for the cost of an unfolded ConstantExpr which could end up
    // getting expanded into Instructions.
    // FIXME: This doesn't account for how many operations are combined in the
    // constant expression.
    ++SpeculatedInstructions;
    if (SpeculatedInstructions > 1)
      return false;
  }

  return HaveRewritablePHIs;
}

static bool isProfitableToSpeculate(const BranchInst *BI,
                                    std::optional<bool> Invert,
                                    const TargetTransformInfo &TTI) {
  // If the branch is non-unpredictable, and is predicted to *not* branch to
  // the `then` block, then avoid speculating it.
  if (BI->getMetadata(LLVMContext::MD_unpredictable))
    return true;

  uint64_t TWeight, FWeight;
  if (!extractBranchWeights(*BI, TWeight, FWeight) || (TWeight + FWeight) == 0)
    return true;

  if (!Invert.has_value())
    return false;

  uint64_t EndWeight = *Invert ? TWeight : FWeight;
  BranchProbability BIEndProb =
      BranchProbability::getBranchProbability(EndWeight, TWeight + FWeight);
  BranchProbability Likely = TTI.getPredictableBranchThreshold();
  return BIEndProb < Likely;
}

/// Speculate a conditional basic block flattening the CFG.
///
/// Note that this is a very risky transform currently. Speculating
/// instructions like this is most often not desirable. Instead, there is an MI
/// pass which can do it with full awareness of the resource constraints.
/// However, some cases are "obvious" and we should do directly. An example of
/// this is speculating a single, reasonably cheap instruction.
///
/// There is only one distinct advantage to flattening the CFG at the IR level:
/// it makes very common but simplistic optimizations such as are common in
/// instcombine and the DAG combiner more powerful by removing CFG edges and
/// modeling their effects with easier to reason about SSA value graphs.
///
///
/// An illustration of this transform is turning this IR:
/// \code
///   BB:
///     %cmp = icmp ult %x, %y
///     br i1 %cmp, label %EndBB, label %ThenBB
///   ThenBB:
///     %sub = sub %x, %y
///     br label BB2
///   EndBB:
///     %phi = phi [ %sub, %ThenBB ], [ 0, %BB ]
///     ...
/// \endcode
///
/// Into this IR:
/// \code
///   BB:
///     %cmp = icmp ult %x, %y
///     %sub = sub %x, %y
///     %cond = select i1 %cmp, 0, %sub
///     ...
/// \endcode
///
/// \returns true if the conditional block is removed.
bool SimplifyCFGOpt::speculativelyExecuteBB(BranchInst *BI,
                                            BasicBlock *ThenBB) {
  if (!Options.SpeculateBlocks)
    return false;

  // Be conservative for now. FP select instruction can often be expensive.
  Value *BrCond = BI->getCondition();
  if (isa<FCmpInst>(BrCond))
    return false;

  BasicBlock *BB = BI->getParent();
  BasicBlock *EndBB = ThenBB->getTerminator()->getSuccessor(0);
  InstructionCost Budget =
      PHINodeFoldingThreshold * TargetTransformInfo::TCC_Basic;

  // If ThenBB is actually on the false edge of the conditional branch, remember
  // to swap the select operands later.
  bool Invert = false;
  if (ThenBB != BI->getSuccessor(0)) {
    assert(ThenBB == BI->getSuccessor(1) && "No edge from 'if' block?");
    Invert = true;
  }
  assert(EndBB == BI->getSuccessor(!Invert) && "No edge from to end block");

  if (!isProfitableToSpeculate(BI, Invert, TTI))
    return false;

  // Keep a count of how many times instructions are used within ThenBB when
  // they are candidates for sinking into ThenBB. Specifically:
  // - They are defined in BB, and
  // - They have no side effects, and
  // - All of their uses are in ThenBB.
  SmallDenseMap<Instruction *, unsigned, 4> SinkCandidateUseCounts;

  SmallVector<Instruction *, 4> SpeculatedPseudoProbes;

  unsigned SpeculatedInstructions = 0;
  bool HoistLoadsStores = Options.HoistLoadsStoresWithCondFaulting;
  SmallVector<Instruction *, 2> SpeculatedConditionalLoadsStores;
  Value *SpeculatedStoreValue = nullptr;
  StoreInst *SpeculatedStore = nullptr;
  EphemeralValueTracker EphTracker;
  for (Instruction &I : reverse(drop_end(*ThenBB))) {
    // Skip pseudo probes. The consequence is we lose track of the branch
    // probability for ThenBB, which is fine since the optimization here takes
    // place regardless of the branch probability.
    if (isa<PseudoProbeInst>(I)) {
      // The probe should be deleted so that it will not be over-counted when
      // the samples collected on the non-conditional path are counted towards
      // the conditional path. We leave it for the counts inference algorithm to
      // figure out a proper count for an unknown probe.
      SpeculatedPseudoProbes.push_back(&I);
      continue;
    }

    // Ignore ephemeral values, they will be dropped by the transform.
    if (EphTracker.track(&I))
      continue;

    // Only speculatively execute a single instruction (not counting the
    // terminator) for now.
    bool IsSafeCheapLoadStore = HoistLoadsStores &&
                                isSafeCheapLoadStore(&I, TTI) &&
                                SpeculatedConditionalLoadsStores.size() <
                                    HoistLoadsStoresWithCondFaultingThreshold;
    // Not count load/store into cost if target supports conditional faulting
    // b/c it's cheap to speculate it.
    if (IsSafeCheapLoadStore)
      SpeculatedConditionalLoadsStores.push_back(&I);
    else
      ++SpeculatedInstructions;

    if (SpeculatedInstructions > 1)
      return false;

    // Don't hoist the instruction if it's unsafe or expensive.
    if (!IsSafeCheapLoadStore &&
        !isSafeToSpeculativelyExecute(&I, BI, Options.AC) &&
        !(HoistCondStores && !SpeculatedStoreValue &&
          (SpeculatedStoreValue =
               isSafeToSpeculateStore(&I, BB, ThenBB, EndBB))))
      return false;
    if (!IsSafeCheapLoadStore && !SpeculatedStoreValue &&
        computeSpeculationCost(&I, TTI) >
            PHINodeFoldingThreshold * TargetTransformInfo::TCC_Basic)
      return false;

    // Store the store speculation candidate.
    if (!SpeculatedStore && SpeculatedStoreValue)
      SpeculatedStore = cast<StoreInst>(&I);

    // Do not hoist the instruction if any of its operands are defined but not
    // used in BB. The transformation will prevent the operand from
    // being sunk into the use block.
    for (Use &Op : I.operands()) {
      Instruction *OpI = dyn_cast<Instruction>(Op);
      if (!OpI || OpI->getParent() != BB || OpI->mayHaveSideEffects())
        continue; // Not a candidate for sinking.

      ++SinkCandidateUseCounts[OpI];
    }
  }

  // Consider any sink candidates which are only used in ThenBB as costs for
  // speculation. Note, while we iterate over a DenseMap here, we are summing
  // and so iteration order isn't significant.
  for (const auto &[Inst, Count] : SinkCandidateUseCounts)
    if (Inst->hasNUses(Count)) {
      ++SpeculatedInstructions;
      if (SpeculatedInstructions > 1)
        return false;
    }

  // Check that we can insert the selects and that it's not too expensive to do
  // so.
  bool Convert =
      SpeculatedStore != nullptr || !SpeculatedConditionalLoadsStores.empty();
  InstructionCost Cost = 0;
  Convert |= validateAndCostRequiredSelects(BB, ThenBB, EndBB,
                                            SpeculatedInstructions, Cost, TTI);
  if (!Convert || Cost > Budget)
    return false;

  // If we get here, we can hoist the instruction and if-convert.
  LLVM_DEBUG(dbgs() << "SPECULATIVELY EXECUTING BB" << *ThenBB << "\n";);

  Instruction *Sel = nullptr;
  // Insert a select of the value of the speculated store.
  if (SpeculatedStoreValue) {
    IRBuilder<NoFolder> Builder(BI);
    Value *OrigV = SpeculatedStore->getValueOperand();
    Value *TrueV = SpeculatedStore->getValueOperand();
    Value *FalseV = SpeculatedStoreValue;
    if (Invert)
      std::swap(TrueV, FalseV);
    Value *S = Builder.CreateSelect(
        BrCond, TrueV, FalseV, "spec.store.select", BI);
    Sel = cast<Instruction>(S);
    SpeculatedStore->setOperand(0, S);
    SpeculatedStore->applyMergedLocation(BI->getDebugLoc(),
                                         SpeculatedStore->getDebugLoc());
    // The value stored is still conditional, but the store itself is now
    // unconditonally executed, so we must be sure that any linked dbg.assign
    // intrinsics are tracking the new stored value (the result of the
    // select). If we don't, and the store were to be removed by another pass
    // (e.g. DSE), then we'd eventually end up emitting a location describing
    // the conditional value, unconditionally.
    //
    // === Before this transformation ===
    // pred:
    //   store %one, %x.dest, !DIAssignID !1
    //   dbg.assign %one, "x", ..., !1, ...
    //   br %cond if.then
    //
    // if.then:
    //   store %two, %x.dest, !DIAssignID !2
    //   dbg.assign %two, "x", ..., !2, ...
    //
    // === After this transformation ===
    // pred:
    //   store %one, %x.dest, !DIAssignID !1
    //   dbg.assign %one, "x", ..., !1
    ///  ...
    //   %merge = select %cond, %two, %one
    //   store %merge, %x.dest, !DIAssignID !2
    //   dbg.assign %merge, "x", ..., !2
    for (DbgVariableRecord *DbgAssign :
         at::getDVRAssignmentMarkers(SpeculatedStore))
      if (llvm::is_contained(DbgAssign->location_ops(), OrigV))
        DbgAssign->replaceVariableLocationOp(OrigV, S);
  }

  // Metadata can be dependent on the condition we are hoisting above.
  // Strip all UB-implying metadata on the instruction. Drop the debug loc
  // to avoid making it appear as if the condition is a constant, which would
  // be misleading while debugging.
  // Similarly strip attributes that maybe dependent on condition we are
  // hoisting above.
  for (auto &I : make_early_inc_range(*ThenBB)) {
    if (!SpeculatedStoreValue || &I != SpeculatedStore) {
      I.setDebugLoc(DebugLoc::getDropped());
    }
    I.dropUBImplyingAttrsAndMetadata();

    // Drop ephemeral values.
    if (EphTracker.contains(&I)) {
      I.replaceAllUsesWith(PoisonValue::get(I.getType()));
      I.eraseFromParent();
    }
  }

  // Hoist the instructions.
  // Drop DbgVariableRecords attached to these instructions.
  for (auto &It : *ThenBB)
    for (DbgRecord &DR : make_early_inc_range(It.getDbgRecordRange()))
      // Drop all records except assign-kind DbgVariableRecords (dbg.assign
      // equivalent).
      if (DbgVariableRecord *DVR = dyn_cast<DbgVariableRecord>(&DR);
          !DVR || !DVR->isDbgAssign())
        It.dropOneDbgRecord(&DR);
  BB->splice(BI->getIterator(), ThenBB, ThenBB->begin(),
             std::prev(ThenBB->end()));

  if (!SpeculatedConditionalLoadsStores.empty())
    hoistConditionalLoadsStores(BI, SpeculatedConditionalLoadsStores, Invert,
                                Sel);

  // Insert selects and rewrite the PHI operands.
  IRBuilder<NoFolder> Builder(BI);
  for (PHINode &PN : EndBB->phis()) {
    unsigned OrigI = PN.getBasicBlockIndex(BB);
    unsigned ThenI = PN.getBasicBlockIndex(ThenBB);
    Value *OrigV = PN.getIncomingValue(OrigI);
    Value *ThenV = PN.getIncomingValue(ThenI);

    // Skip PHIs which are trivial.
    if (OrigV == ThenV)
      continue;

    // Create a select whose true value is the speculatively executed value and
    // false value is the pre-existing value. Swap them if the branch
    // destinations were inverted.
    Value *TrueV = ThenV, *FalseV = OrigV;
    if (Invert)
      std::swap(TrueV, FalseV);
    Value *V = Builder.CreateSelect(BrCond, TrueV, FalseV, "spec.select", BI);
    PN.setIncomingValue(OrigI, V);
    PN.setIncomingValue(ThenI, V);
  }

  // Remove speculated pseudo probes.
  for (Instruction *I : SpeculatedPseudoProbes)
    I->eraseFromParent();

  ++NumSpeculations;
  return true;
}

using BlocksSet = SmallPtrSet<BasicBlock *, 8>;

// Return false if number of blocks searched is too much.
static bool findReaching(BasicBlock *BB, BasicBlock *DefBB,
                         BlocksSet &ReachesNonLocalUses) {
  if (BB == DefBB)
    return true;
  if (!ReachesNonLocalUses.insert(BB).second)
    return true;

  if (ReachesNonLocalUses.size() > MaxJumpThreadingLiveBlocks)
    return false;
  for (BasicBlock *Pred : predecessors(BB))
    if (!findReaching(Pred, DefBB, ReachesNonLocalUses))
      return false;
  return true;
}

/// Return true if we can thread a branch across this block.
static bool blockIsSimpleEnoughToThreadThrough(BasicBlock *BB,
                                               BlocksSet &NonLocalUseBlocks) {
  int Size = 0;
  EphemeralValueTracker EphTracker;

  // Walk the loop in reverse so that we can identify ephemeral values properly
  // (values only feeding assumes).
  for (Instruction &I : reverse(BB->instructionsWithoutDebug(false))) {
    // Can't fold blocks that contain noduplicate or convergent calls.
    if (CallInst *CI = dyn_cast<CallInst>(&I))
      if (CI->cannotDuplicate() || CI->isConvergent())
        return false;

    // Ignore ephemeral values which are deleted during codegen.
    // We will delete Phis while threading, so Phis should not be accounted in
    // block's size.
    if (!EphTracker.track(&I) && !isa<PHINode>(I)) {
      if (Size++ > MaxSmallBlockSize)
        return false; // Don't clone large BB's.
    }

    // Record blocks with non-local uses of values defined in the current basic
    // block.
    for (User *U : I.users()) {
      Instruction *UI = cast<Instruction>(U);
      BasicBlock *UsedInBB = UI->getParent();
      if (UsedInBB == BB) {
        if (isa<PHINode>(UI))
          return false;
      } else
        NonLocalUseBlocks.insert(UsedInBB);
    }

    // Looks ok, continue checking.
  }

  return true;
}

static ConstantInt *getKnownValueOnEdge(Value *V, BasicBlock *From,
                                        BasicBlock *To) {
  // Don't look past the block defining the value, we might get the value from
  // a previous loop iteration.
  auto *I = dyn_cast<Instruction>(V);
  if (I && I->getParent() == To)
    return nullptr;

  // We know the value if the From block branches on it.
  auto *BI = dyn_cast<BranchInst>(From->getTerminator());
  if (BI && BI->isConditional() && BI->getCondition() == V &&
      BI->getSuccessor(0) != BI->getSuccessor(1))
    return BI->getSuccessor(0) == To ? ConstantInt::getTrue(BI->getContext())
                                     : ConstantInt::getFalse(BI->getContext());

  return nullptr;
}

/// If we have a conditional branch on something for which we know the constant
/// value in predecessors (e.g. a phi node in the current block), thread edges
/// from the predecessor to their ultimate destination.
static std::optional<bool>
foldCondBranchOnValueKnownInPredecessorImpl(BranchInst *BI, DomTreeUpdater *DTU,
                                            const DataLayout &DL,
                                            AssumptionCache *AC) {
  SmallMapVector<ConstantInt *, SmallSetVector<BasicBlock *, 2>, 2> KnownValues;
  BasicBlock *BB = BI->getParent();
  Value *Cond = BI->getCondition();
  PHINode *PN = dyn_cast<PHINode>(Cond);
  if (PN && PN->getParent() == BB) {
    // Degenerate case of a single entry PHI.
    if (PN->getNumIncomingValues() == 1) {
      FoldSingleEntryPHINodes(PN->getParent());
      return true;
    }

    for (Use &U : PN->incoming_values())
      if (auto *CB = dyn_cast<ConstantInt>(U))
        KnownValues[CB].insert(PN->getIncomingBlock(U));
  } else {
    for (BasicBlock *Pred : predecessors(BB)) {
      if (ConstantInt *CB = getKnownValueOnEdge(Cond, Pred, BB))
        KnownValues[CB].insert(Pred);
    }
  }

  if (KnownValues.empty())
    return false;

  // Now we know that this block has multiple preds and two succs.
  // Check that the block is small enough and record which non-local blocks use
  // values defined in the block.

  BlocksSet NonLocalUseBlocks;
  BlocksSet ReachesNonLocalUseBlocks;
  if (!blockIsSimpleEnoughToThreadThrough(BB, NonLocalUseBlocks))
    return false;

  // Jump-threading can only be done to destinations where no values defined
  // in BB are live.

  // Quickly check if both destinations have uses.  If so, jump-threading cannot
  // be done.
  if (NonLocalUseBlocks.contains(BI->getSuccessor(0)) &&
      NonLocalUseBlocks.contains(BI->getSuccessor(1)))
    return false;

  // Search backward from NonLocalUseBlocks to find which blocks
  // reach non-local uses.
  for (BasicBlock *UseBB : NonLocalUseBlocks)
    // Give up if too many blocks are searched.
    if (!findReaching(UseBB, BB, ReachesNonLocalUseBlocks))
      return false;

  for (const auto &Pair : KnownValues) {
    ConstantInt *CB = Pair.first;
    ArrayRef<BasicBlock *> PredBBs = Pair.second.getArrayRef();
    BasicBlock *RealDest = BI->getSuccessor(!CB->getZExtValue());

    // Okay, we now know that all edges from PredBB should be revectored to
    // branch to RealDest.
    if (RealDest == BB)
      continue; // Skip self loops.

    // Skip if the predecessor's terminator is an indirect branch.
    if (any_of(PredBBs, [](BasicBlock *PredBB) {
          return isa<IndirectBrInst>(PredBB->getTerminator());
        }))
      continue;

    // Only revector to RealDest if no values defined in BB are live.
    if (ReachesNonLocalUseBlocks.contains(RealDest))
      continue;

    LLVM_DEBUG({
      dbgs() << "Condition " << *Cond << " in " << BB->getName()
             << " has value " << *Pair.first << " in predecessors:\n";
      for (const BasicBlock *PredBB : Pair.second)
        dbgs() << "  " << PredBB->getName() << "\n";
      dbgs() << "Threading to destination " << RealDest->getName() << ".\n";
    });

    // Split the predecessors we are threading into a new edge block. We'll
    // clone the instructions into this block, and then redirect it to RealDest.
    BasicBlock *EdgeBB = SplitBlockPredecessors(BB, PredBBs, ".critedge", DTU);

    // TODO: These just exist to reduce test diff, we can drop them if we like.
    EdgeBB->setName(RealDest->getName() + ".critedge");
    EdgeBB->moveBefore(RealDest);

    // Update PHI nodes.
    addPredecessorToBlock(RealDest, EdgeBB, BB);

    // BB may have instructions that are being threaded over.  Clone these
    // instructions into EdgeBB.  We know that there will be no uses of the
    // cloned instructions outside of EdgeBB.
    BasicBlock::iterator InsertPt = EdgeBB->getFirstInsertionPt();
    ValueToValueMapTy TranslateMap; // Track translated values.
    TranslateMap[Cond] = CB;

    // RemoveDIs: track instructions that we optimise away while folding, so
    // that we can copy DbgVariableRecords from them later.
    BasicBlock::iterator SrcDbgCursor = BB->begin();
    for (BasicBlock::iterator BBI = BB->begin(); &*BBI != BI; ++BBI) {
      if (PHINode *PN = dyn_cast<PHINode>(BBI)) {
        TranslateMap[PN] = PN->getIncomingValueForBlock(EdgeBB);
        continue;
      }
      // Clone the instruction.
      Instruction *N = BBI->clone();
      // Insert the new instruction into its new home.
      N->insertInto(EdgeBB, InsertPt);

      if (BBI->hasName())
        N->setName(BBI->getName() + ".c");

      // Update operands due to translation.
      // Key Instructions: Remap all the atom groups.
      if (const DebugLoc &DL = BBI->getDebugLoc())
        mapAtomInstance(DL, TranslateMap);
      RemapInstruction(N, TranslateMap,
                       RF_IgnoreMissingLocals | RF_NoModuleLevelChanges);

      // Check for trivial simplification.
      if (Value *V = simplifyInstruction(N, {DL, nullptr, nullptr, AC})) {
        if (!BBI->use_empty())
          TranslateMap[&*BBI] = V;
        if (!N->mayHaveSideEffects()) {
          N->eraseFromParent(); // Instruction folded away, don't need actual
                                // inst
          N = nullptr;
        }
      } else {
        if (!BBI->use_empty())
          TranslateMap[&*BBI] = N;
      }
      if (N) {
        // Copy all debug-info attached to instructions from the last we
        // successfully clone, up to this instruction (they might have been
        // folded away).
        for (; SrcDbgCursor != BBI; ++SrcDbgCursor)
          N->cloneDebugInfoFrom(&*SrcDbgCursor);
        SrcDbgCursor = std::next(BBI);
        // Clone debug-info on this instruction too.
        N->cloneDebugInfoFrom(&*BBI);

        // Register the new instruction with the assumption cache if necessary.
        if (auto *Assume = dyn_cast<AssumeInst>(N))
          if (AC)
            AC->registerAssumption(Assume);
      }
    }

    for (; &*SrcDbgCursor != BI; ++SrcDbgCursor)
      InsertPt->cloneDebugInfoFrom(&*SrcDbgCursor);
    InsertPt->cloneDebugInfoFrom(BI);

    BB->removePredecessor(EdgeBB);
    BranchInst *EdgeBI = cast<BranchInst>(EdgeBB->getTerminator());
    EdgeBI->setSuccessor(0, RealDest);
    EdgeBI->setDebugLoc(BI->getDebugLoc());

    if (DTU) {
      SmallVector<DominatorTree::UpdateType, 2> Updates;
      Updates.push_back({DominatorTree::Delete, EdgeBB, BB});
      Updates.push_back({DominatorTree::Insert, EdgeBB, RealDest});
      DTU->applyUpdates(Updates);
    }

    // For simplicity, we created a separate basic block for the edge. Merge
    // it back into the predecessor if possible. This not only avoids
    // unnecessary SimplifyCFG iterations, but also makes sure that we don't
    // bypass the check for trivial cycles above.
    MergeBlockIntoPredecessor(EdgeBB, DTU);

    // Signal repeat, simplifying any other constants.
    return std::nullopt;
  }

  return false;
}

bool SimplifyCFGOpt::foldCondBranchOnValueKnownInPredecessor(BranchInst *BI) {
  // Note: If BB is a loop header then there is a risk that threading introduces
  // a non-canonical loop by moving a back edge. So we avoid this optimization
  // for loop headers if NeedCanonicalLoop is set.
  if (Options.NeedCanonicalLoop && is_contained(LoopHeaders, BI->getParent()))
    return false;

  std::optional<bool> Result;
  bool EverChanged = false;
  do {
    // Note that None means "we changed things, but recurse further."
    Result =
        foldCondBranchOnValueKnownInPredecessorImpl(BI, DTU, DL, Options.AC);
    EverChanged |= Result == std::nullopt || *Result;
  } while (Result == std::nullopt);
  return EverChanged;
}

/// Given a BB that starts with the specified two-entry PHI node,
/// see if we can eliminate it.
static bool foldTwoEntryPHINode(PHINode *PN, const TargetTransformInfo &TTI,
                                DomTreeUpdater *DTU, AssumptionCache *AC,
                                const DataLayout &DL,
                                bool SpeculateUnpredictables) {
  // Ok, this is a two entry PHI node.  Check to see if this is a simple "if
  // statement", which has a very simple dominance structure.  Basically, we
  // are trying to find the condition that is being branched on, which
  // subsequently causes this merge to happen.  We really want control
  // dependence information for this check, but simplifycfg can't keep it up
  // to date, and this catches most of the cases we care about anyway.
  BasicBlock *BB = PN->getParent();

  BasicBlock *IfTrue, *IfFalse;
  BranchInst *DomBI = GetIfCondition(BB, IfTrue, IfFalse);
  if (!DomBI)
    return false;
  Value *IfCond = DomBI->getCondition();
  // Don't bother if the branch will be constant folded trivially.
  if (isa<ConstantInt>(IfCond))
    return false;

  BasicBlock *DomBlock = DomBI->getParent();
  SmallVector<BasicBlock *, 2> IfBlocks;
  llvm::copy_if(
      PN->blocks(), std::back_inserter(IfBlocks), [](BasicBlock *IfBlock) {
        return cast<BranchInst>(IfBlock->getTerminator())->isUnconditional();
      });
  assert((IfBlocks.size() == 1 || IfBlocks.size() == 2) &&
         "Will have either one or two blocks to speculate.");

  // If the branch is non-unpredictable, see if we either predictably jump to
  // the merge bb (if we have only a single 'then' block), or if we predictably
  // jump to one specific 'then' block (if we have two of them).
  // It isn't beneficial to speculatively execute the code
  // from the block that we know is predictably not entered.
  bool IsUnpredictable = DomBI->getMetadata(LLVMContext::MD_unpredictable);
  if (!IsUnpredictable) {
    uint64_t TWeight, FWeight;
    if (extractBranchWeights(*DomBI, TWeight, FWeight) &&
        (TWeight + FWeight) != 0) {
      BranchProbability BITrueProb =
          BranchProbability::getBranchProbability(TWeight, TWeight + FWeight);
      BranchProbability Likely = TTI.getPredictableBranchThreshold();
      BranchProbability BIFalseProb = BITrueProb.getCompl();
      if (IfBlocks.size() == 1) {
        BranchProbability BIBBProb =
            DomBI->getSuccessor(0) == BB ? BITrueProb : BIFalseProb;
        if (BIBBProb >= Likely)
          return false;
      } else {
        if (BITrueProb >= Likely || BIFalseProb >= Likely)
          return false;
      }
    }
  }

  // Don't try to fold an unreachable block. For example, the phi node itself
  // can't be the candidate if-condition for a select that we want to form.
  if (auto *IfCondPhiInst = dyn_cast<PHINode>(IfCond))
    if (IfCondPhiInst->getParent() == BB)
      return false;

  // Okay, we found that we can merge this two-entry phi node into a select.
  // Doing so would require us to fold *all* two entry phi nodes in this block.
  // At some point this becomes non-profitable (particularly if the target
  // doesn't support cmov's).  Only do this transformation if there are two or
  // fewer PHI nodes in this block.
  unsigned NumPhis = 0;
  for (BasicBlock::iterator I = BB->begin(); isa<PHINode>(I); ++NumPhis, ++I)
    if (NumPhis > 2)
      return false;

  // Loop over the PHI's seeing if we can promote them all to select
  // instructions.  While we are at it, keep track of the instructions
  // that need to be moved to the dominating block.
  SmallPtrSet<Instruction *, 4> AggressiveInsts;
  SmallPtrSet<Instruction *, 2> ZeroCostInstructions;
  InstructionCost Cost = 0;
  InstructionCost Budget =
      TwoEntryPHINodeFoldingThreshold * TargetTransformInfo::TCC_Basic;
  if (SpeculateUnpredictables && IsUnpredictable)
    Budget += TTI.getBranchMispredictPenalty();

  bool Changed = false;
  for (BasicBlock::iterator II = BB->begin(); isa<PHINode>(II);) {
    PHINode *PN = cast<PHINode>(II++);
    if (Value *V = simplifyInstruction(PN, {DL, PN})) {
      PN->replaceAllUsesWith(V);
      PN->eraseFromParent();
      Changed = true;
      continue;
    }

    if (!dominatesMergePoint(PN->getIncomingValue(0), BB, DomBI,
                             AggressiveInsts, Cost, Budget, TTI, AC,
                             ZeroCostInstructions) ||
        !dominatesMergePoint(PN->getIncomingValue(1), BB, DomBI,
                             AggressiveInsts, Cost, Budget, TTI, AC,
                             ZeroCostInstructions))
      return Changed;
  }

  // If we folded the first phi, PN dangles at this point.  Refresh it.  If
  // we ran out of PHIs then we simplified them all.
  PN = dyn_cast<PHINode>(BB->begin());
  if (!PN)
    return true;

  // Return true if at least one of these is a 'not', and another is either
  // a 'not' too, or a constant.
  auto CanHoistNotFromBothValues = [](Value *V0, Value *V1) {
    if (!match(V0, m_Not(m_Value())))
      std::swap(V0, V1);
    auto Invertible = m_CombineOr(m_Not(m_Value()), m_AnyIntegralConstant());
    return match(V0, m_Not(m_Value())) && match(V1, Invertible);
  };

  // Don't fold i1 branches on PHIs which contain binary operators or
  // (possibly inverted) select form of or/ands,  unless one of
  // the incoming values is an 'not' and another one is freely invertible.
  // These can often be turned into switches and other things.
  auto IsBinOpOrAnd = [](Value *V) {
    return match(
        V, m_CombineOr(m_BinOp(), m_c_Select(m_ImmConstant(), m_Value())));
  };
  if (PN->getType()->isIntegerTy(1) &&
      (IsBinOpOrAnd(PN->getIncomingValue(0)) ||
       IsBinOpOrAnd(PN->getIncomingValue(1)) || IsBinOpOrAnd(IfCond)) &&
      !CanHoistNotFromBothValues(PN->getIncomingValue(0),
                                 PN->getIncomingValue(1)))
    return Changed;

  // If all PHI nodes are promotable, check to make sure that all instructions
  // in the predecessor blocks can be promoted as well. If not, we won't be able
  // to get rid of the control flow, so it's not worth promoting to select
  // instructions.
  for (BasicBlock *IfBlock : IfBlocks)
    for (BasicBlock::iterator I = IfBlock->begin(); !I->isTerminator(); ++I)
      if (!AggressiveInsts.count(&*I) && !I->isDebugOrPseudoInst()) {
        // This is not an aggressive instruction that we can promote.
        // Because of this, we won't be able to get rid of the control flow, so
        // the xform is not worth it.
        return Changed;
      }

  // If either of the blocks has it's address taken, we can't do this fold.
  if (any_of(IfBlocks,
             [](BasicBlock *IfBlock) { return IfBlock->hasAddressTaken(); }))
    return Changed;

  LLVM_DEBUG(dbgs() << "FOUND IF CONDITION!  " << *IfCond;
             if (IsUnpredictable) dbgs() << " (unpredictable)";
             dbgs() << "  T: " << IfTrue->getName()
                    << "  F: " << IfFalse->getName() << "\n");

  // If we can still promote the PHI nodes after this gauntlet of tests,
  // do all of the PHI's now.

  // Move all 'aggressive' instructions, which are defined in the
  // conditional parts of the if's up to the dominating block.
  for (BasicBlock *IfBlock : IfBlocks)
      hoistAllInstructionsInto(DomBlock, DomBI, IfBlock);

  IRBuilder<NoFolder> Builder(DomBI);
  // Propagate fast-math-flags from phi nodes to replacement selects.
  while (PHINode *PN = dyn_cast<PHINode>(BB->begin())) {
    // Change the PHI node into a select instruction.
    Value *TrueVal = PN->getIncomingValueForBlock(IfTrue);
    Value *FalseVal = PN->getIncomingValueForBlock(IfFalse);

    Value *Sel = Builder.CreateSelectFMF(IfCond, TrueVal, FalseVal,
                                         isa<FPMathOperator>(PN) ? PN : nullptr,
                                         "", DomBI);
    PN->replaceAllUsesWith(Sel);
    Sel->takeName(PN);
    PN->eraseFromParent();
  }

  // At this point, all IfBlocks are empty, so our if statement
  // has been flattened.  Change DomBlock to jump directly to our new block to
  // avoid other simplifycfg's kicking in on the diamond.
  Builder.CreateBr(BB);

  SmallVector<DominatorTree::UpdateType, 3> Updates;
  if (DTU) {
    Updates.push_back({DominatorTree::Insert, DomBlock, BB});
    for (auto *Successor : successors(DomBlock))
      Updates.push_back({DominatorTree::Delete, DomBlock, Successor});
  }

  DomBI->eraseFromParent();
  if (DTU)
    DTU->applyUpdates(Updates);

  return true;
}

static Value *createLogicalOp(IRBuilderBase &Builder,
                              Instruction::BinaryOps Opc, Value *LHS,
                              Value *RHS, const Twine &Name = "") {
  // Try to relax logical op to binary op.
  if (impliesPoison(RHS, LHS))
    return Builder.CreateBinOp(Opc, LHS, RHS, Name);
  if (Opc == Instruction::And)
    return Builder.CreateLogicalAnd(LHS, RHS, Name);
  if (Opc == Instruction::Or)
    return Builder.CreateLogicalOr(LHS, RHS, Name);
  llvm_unreachable("Invalid logical opcode");
}

/// Return true if either PBI or BI has branch weight available, and store
/// the weights in {Pred|Succ}{True|False}Weight. If one of PBI and BI does
/// not have branch weight, use 1:1 as its weight.
static bool extractPredSuccWeights(BranchInst *PBI, BranchInst *BI,
                                   uint64_t &PredTrueWeight,
                                   uint64_t &PredFalseWeight,
                                   uint64_t &SuccTrueWeight,
                                   uint64_t &SuccFalseWeight) {
  bool PredHasWeights =
      extractBranchWeights(*PBI, PredTrueWeight, PredFalseWeight);
  bool SuccHasWeights =
      extractBranchWeights(*BI, SuccTrueWeight, SuccFalseWeight);
  if (PredHasWeights || SuccHasWeights) {
    if (!PredHasWeights)
      PredTrueWeight = PredFalseWeight = 1;
    if (!SuccHasWeights)
      SuccTrueWeight = SuccFalseWeight = 1;
    return true;
  } else {
    return false;
  }
}

/// Determine if the two branches share a common destination and deduce a glue
/// that joins the branches' conditions to arrive at the common destination if
/// that would be profitable.
static std::optional<std::tuple<BasicBlock *, Instruction::BinaryOps, bool>>
shouldFoldCondBranchesToCommonDestination(BranchInst *BI, BranchInst *PBI,
                                          const TargetTransformInfo *TTI) {
  assert(BI && PBI && BI->isConditional() && PBI->isConditional() &&
         "Both blocks must end with a conditional branches.");
  assert(is_contained(predecessors(BI->getParent()), PBI->getParent()) &&
         "PredBB must be a predecessor of BB.");

  // We have the potential to fold the conditions together, but if the
  // predecessor branch is predictable, we may not want to merge them.
  uint64_t PTWeight, PFWeight;
  BranchProbability PBITrueProb, Likely;
  if (TTI && !PBI->getMetadata(LLVMContext::MD_unpredictable) &&
      extractBranchWeights(*PBI, PTWeight, PFWeight) &&
      (PTWeight + PFWeight) != 0) {
    PBITrueProb =
        BranchProbability::getBranchProbability(PTWeight, PTWeight + PFWeight);
    Likely = TTI->getPredictableBranchThreshold();
  }

  if (PBI->getSuccessor(0) == BI->getSuccessor(0)) {
    // Speculate the 2nd condition unless the 1st is probably true.
    if (PBITrueProb.isUnknown() || PBITrueProb < Likely)
      return {{BI->getSuccessor(0), Instruction::Or, false}};
  } else if (PBI->getSuccessor(1) == BI->getSuccessor(1)) {
    // Speculate the 2nd condition unless the 1st is probably false.
    if (PBITrueProb.isUnknown() || PBITrueProb.getCompl() < Likely)
      return {{BI->getSuccessor(1), Instruction::And, false}};
  } else if (PBI->getSuccessor(0) == BI->getSuccessor(1)) {
    // Speculate the 2nd condition unless the 1st is probably true.
    if (PBITrueProb.isUnknown() || PBITrueProb < Likely)
      return {{BI->getSuccessor(1), Instruction::And, true}};
  } else if (PBI->getSuccessor(1) == BI->getSuccessor(0)) {
    // Speculate the 2nd condition unless the 1st is probably false.
    if (PBITrueProb.isUnknown() || PBITrueProb.getCompl() < Likely)
      return {{BI->getSuccessor(0), Instruction::Or, true}};
  }
  return std::nullopt;
}

static bool performBranchToCommonDestFolding(BranchInst *BI, BranchInst *PBI,
                                             DomTreeUpdater *DTU,
                                             MemorySSAUpdater *MSSAU,
                                             const TargetTransformInfo *TTI) {
  BasicBlock *BB = BI->getParent();
  BasicBlock *PredBlock = PBI->getParent();

  // Determine if the two branches share a common destination.
  BasicBlock *CommonSucc;
  Instruction::BinaryOps Opc;
  bool InvertPredCond;
  std::tie(CommonSucc, Opc, InvertPredCond) =
      *shouldFoldCondBranchesToCommonDestination(BI, PBI, TTI);

  LLVM_DEBUG(dbgs() << "FOLDING BRANCH TO COMMON DEST:\n" << *PBI << *BB);

  IRBuilder<> Builder(PBI);
  // The builder is used to create instructions to eliminate the branch in BB.
  // If BB's terminator has !annotation metadata, add it to the new
  // instructions.
  Builder.CollectMetadataToCopy(BB->getTerminator(),
                                {LLVMContext::MD_annotation});

  // If we need to invert the condition in the pred block to match, do so now.
  if (InvertPredCond) {
    InvertBranch(PBI, Builder);
  }

  BasicBlock *UniqueSucc =
      PBI->getSuccessor(0) == BB ? BI->getSuccessor(0) : BI->getSuccessor(1);

  // Before cloning instructions, notify the successor basic block that it
  // is about to have a new predecessor. This will update PHI nodes,
  // which will allow us to update live-out uses of bonus instructions.
  addPredecessorToBlock(UniqueSucc, PredBlock, BB, MSSAU);

  // Try to update branch weights.
  uint64_t PredTrueWeight, PredFalseWeight, SuccTrueWeight, SuccFalseWeight;
  if (extractPredSuccWeights(PBI, BI, PredTrueWeight, PredFalseWeight,
                             SuccTrueWeight, SuccFalseWeight)) {
    SmallVector<uint64_t, 8> NewWeights;

    if (PBI->getSuccessor(0) == BB) {
      // PBI: br i1 %x, BB, FalseDest
      // BI:  br i1 %y, UniqueSucc, FalseDest
      // TrueWeight is TrueWeight for PBI * TrueWeight for BI.
      NewWeights.push_back(PredTrueWeight * SuccTrueWeight);
      // FalseWeight is FalseWeight for PBI * TotalWeight for BI +
      //               TrueWeight for PBI * FalseWeight for BI.
      // We assume that total weights of a BranchInst can fit into 32 bits.
      // Therefore, we will not have overflow using 64-bit arithmetic.
      NewWeights.push_back(PredFalseWeight *
                               (SuccFalseWeight + SuccTrueWeight) +
                           PredTrueWeight * SuccFalseWeight);
    } else {
      // PBI: br i1 %x, TrueDest, BB
      // BI:  br i1 %y, TrueDest, UniqueSucc
      // TrueWeight is TrueWeight for PBI * TotalWeight for BI +
      //              FalseWeight for PBI * TrueWeight for BI.
      NewWeights.push_back(PredTrueWeight * (SuccFalseWeight + SuccTrueWeight) +
                           PredFalseWeight * SuccTrueWeight);
      // FalseWeight is FalseWeight for PBI * FalseWeight for BI.
      NewWeights.push_back(PredFalseWeight * SuccFalseWeight);
    }

    // Halve the weights if any of them cannot fit in an uint32_t
    fitWeights(NewWeights);

    SmallVector<uint32_t, 8> MDWeights(NewWeights.begin(), NewWeights.end());
    setBranchWeights(PBI, MDWeights[0], MDWeights[1], /*IsExpected=*/false);

    // TODO: If BB is reachable from all paths through PredBlock, then we
    // could replace PBI's branch probabilities with BI's.
  } else
    PBI->setMetadata(LLVMContext::MD_prof, nullptr);

  // Now, update the CFG.
  PBI->setSuccessor(PBI->getSuccessor(0) != BB, UniqueSucc);

  if (DTU)
    DTU->applyUpdates({{DominatorTree::Insert, PredBlock, UniqueSucc},
                       {DominatorTree::Delete, PredBlock, BB}});

  // If BI was a loop latch, it may have had associated loop metadata.
  // We need to copy it to the new latch, that is, PBI.
  if (MDNode *LoopMD = BI->getMetadata(LLVMContext::MD_loop))
    PBI->setMetadata(LLVMContext::MD_loop, LoopMD);

  ValueToValueMapTy VMap; // maps original values to cloned values
  cloneInstructionsIntoPredecessorBlockAndUpdateSSAUses(BB, PredBlock, VMap);

  Module *M = BB->getModule();

  PredBlock->getTerminator()->cloneDebugInfoFrom(BB->getTerminator());
  for (DbgVariableRecord &DVR :
       filterDbgVars(PredBlock->getTerminator()->getDbgRecordRange())) {
    RemapDbgRecord(M, &DVR, VMap,
                   RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);
  }

  // Now that the Cond was cloned into the predecessor basic block,
  // or/and the two conditions together.
  Value *BICond = VMap[BI->getCondition()];
  PBI->setCondition(
      createLogicalOp(Builder, Opc, PBI->getCondition(), BICond, "or.cond"));

  ++NumFoldBranchToCommonDest;
  return true;
}

/// Return if an instruction's type or any of its operands' types are a vector
/// type.
static bool isVectorOp(Instruction &I) {
  return I.getType()->isVectorTy() || any_of(I.operands(), [](Use &U) {
           return U->getType()->isVectorTy();
         });
}

/// If this basic block is simple enough, and if a predecessor branches to us
/// and one of our successors, fold the block into the predecessor and use
/// logical operations to pick the right destination.
bool llvm::foldBranchToCommonDest(BranchInst *BI, DomTreeUpdater *DTU,
                                  MemorySSAUpdater *MSSAU,
                                  const TargetTransformInfo *TTI,
                                  unsigned BonusInstThreshold) {
  // If this block ends with an unconditional branch,
  // let speculativelyExecuteBB() deal with it.
  if (!BI->isConditional())
    return false;

  BasicBlock *BB = BI->getParent();
  TargetTransformInfo::TargetCostKind CostKind =
    BB->getParent()->hasMinSize() ? TargetTransformInfo::TCK_CodeSize
                                  : TargetTransformInfo::TCK_SizeAndLatency;

  Instruction *Cond = dyn_cast<Instruction>(BI->getCondition());

  if (!Cond || !isa<CmpInst, BinaryOperator, SelectInst, TruncInst>(Cond) ||
      Cond->getParent() != BB || !Cond->hasOneUse())
    return false;

  // Finally, don't infinitely unroll conditional loops.
  if (is_contained(successors(BB), BB))
    return false;

  // With which predecessors will we want to deal with?
  SmallVector<BasicBlock *, 8> Preds;
  for (BasicBlock *PredBlock : predecessors(BB)) {
    BranchInst *PBI = dyn_cast<BranchInst>(PredBlock->getTerminator());

    // Check that we have two conditional branches.  If there is a PHI node in
    // the common successor, verify that the same value flows in from both
    // blocks.
    if (!PBI || PBI->isUnconditional() || !safeToMergeTerminators(BI, PBI))
      continue;

    // Determine if the two branches share a common destination.
    BasicBlock *CommonSucc;
    Instruction::BinaryOps Opc;
    bool InvertPredCond;
    if (auto Recipe = shouldFoldCondBranchesToCommonDestination(BI, PBI, TTI))
      std::tie(CommonSucc, Opc, InvertPredCond) = *Recipe;
    else
      continue;

    // Check the cost of inserting the necessary logic before performing the
    // transformation.
    if (TTI) {
      Type *Ty = BI->getCondition()->getType();
      InstructionCost Cost = TTI->getArithmeticInstrCost(Opc, Ty, CostKind);
      if (InvertPredCond && (!PBI->getCondition()->hasOneUse() ||
                             !isa<CmpInst>(PBI->getCondition())))
        Cost += TTI->getArithmeticInstrCost(Instruction::Xor, Ty, CostKind);

      if (Cost > BranchFoldThreshold)
        continue;
    }

    // Ok, we do want to deal with this predecessor. Record it.
    Preds.emplace_back(PredBlock);
  }

  // If there aren't any predecessors into which we can fold,
  // don't bother checking the cost.
  if (Preds.empty())
    return false;

  // Only allow this transformation if computing the condition doesn't involve
  // too many instructions and these involved instructions can be executed
  // unconditionally. We denote all involved instructions except the condition
  // as "bonus instructions", and only allow this transformation when the
  // number of the bonus instructions we'll need to create when cloning into
  // each predecessor does not exceed a certain threshold.
  unsigned NumBonusInsts = 0;
  bool SawVectorOp = false;
  const unsigned PredCount = Preds.size();
  for (Instruction &I : *BB) {
    // Don't check the branch condition comparison itself.
    if (&I == Cond)
      continue;
    // Ignore the terminator.
    if (isa<BranchInst>(I))
      continue;
    // I must be safe to execute unconditionally.
    if (!isSafeToSpeculativelyExecute(&I))
      return false;
    SawVectorOp |= isVectorOp(I);

    // Account for the cost of duplicating this instruction into each
    // predecessor. Ignore free instructions.
    if (!TTI || TTI->getInstructionCost(&I, CostKind) !=
                    TargetTransformInfo::TCC_Free) {
      NumBonusInsts += PredCount;

      // Early exits once we reach the limit.
      if (NumBonusInsts >
          BonusInstThreshold * BranchFoldToCommonDestVectorMultiplier)
        return false;
    }

    auto IsBCSSAUse = [BB, &I](Use &U) {
      auto *UI = cast<Instruction>(U.getUser());
      if (auto *PN = dyn_cast<PHINode>(UI))
        return PN->getIncomingBlock(U) == BB;
      return UI->getParent() == BB && I.comesBefore(UI);
    };

    // Does this instruction require rewriting of uses?
    if (!all_of(I.uses(), IsBCSSAUse))
      return false;
  }
  if (NumBonusInsts >
      BonusInstThreshold *
          (SawVectorOp ? BranchFoldToCommonDestVectorMultiplier : 1))
    return false;

  // Ok, we have the budget. Perform the transformation.
  for (BasicBlock *PredBlock : Preds) {
    auto *PBI = cast<BranchInst>(PredBlock->getTerminator());
    return performBranchToCommonDestFolding(BI, PBI, DTU, MSSAU, TTI);
  }
  return false;
}

// If there is only one store in BB1 and BB2, return it, otherwise return
// nullptr.
static StoreInst *findUniqueStoreInBlocks(BasicBlock *BB1, BasicBlock *BB2) {
  StoreInst *S = nullptr;
  for (auto *BB : {BB1, BB2}) {
    if (!BB)
      continue;
    for (auto &I : *BB)
      if (auto *SI = dyn_cast<StoreInst>(&I)) {
        if (S)
          // Multiple stores seen.
          return nullptr;
        else
          S = SI;
      }
  }
  return S;
}

static Value *ensureValueAvailableInSuccessor(Value *V, BasicBlock *BB,
                                              Value *AlternativeV = nullptr) {
  // PHI is going to be a PHI node that allows the value V that is defined in
  // BB to be referenced in BB's only successor.
  //
  // If AlternativeV is nullptr, the only value we care about in PHI is V. It
  // doesn't matter to us what the other operand is (it'll never get used). We
  // could just create a new PHI with an undef incoming value, but that could
  // increase register pressure if EarlyCSE/InstCombine can't fold it with some
  // other PHI. So here we directly look for some PHI in BB's successor with V
  // as an incoming operand. If we find one, we use it, else we create a new
  // one.
  //
  // If AlternativeV is not nullptr, we care about both incoming values in PHI.
  // PHI must be exactly: phi <ty> [ %BB, %V ], [ %OtherBB, %AlternativeV]
  // where OtherBB is the single other predecessor of BB's only successor.
  PHINode *PHI = nullptr;
  BasicBlock *Succ = BB->getSingleSuccessor();

  for (auto I = Succ->begin(); isa<PHINode>(I); ++I)
    if (cast<PHINode>(I)->getIncomingValueForBlock(BB) == V) {
      PHI = cast<PHINode>(I);
      if (!AlternativeV)
        break;

      assert(Succ->hasNPredecessors(2));
      auto PredI = pred_begin(Succ);
      BasicBlock *OtherPredBB = *PredI == BB ? *++PredI : *PredI;
      if (PHI->getIncomingValueForBlock(OtherPredBB) == AlternativeV)
        break;
      PHI = nullptr;
    }
  if (PHI)
    return PHI;

  // If V is not an instruction defined in BB, just return it.
  if (!AlternativeV &&
      (!isa<Instruction>(V) || cast<Instruction>(V)->getParent() != BB))
    return V;

  PHI = PHINode::Create(V->getType(), 2, "simplifycfg.merge");
  PHI->insertBefore(Succ->begin());
  PHI->addIncoming(V, BB);
  for (BasicBlock *PredBB : predecessors(Succ))
    if (PredBB != BB)
      PHI->addIncoming(
          AlternativeV ? AlternativeV : PoisonValue::get(V->getType()), PredBB);
  return PHI;
}

static bool mergeConditionalStoreToAddress(
    BasicBlock *PTB, BasicBlock *PFB, BasicBlock *QTB, BasicBlock *QFB,
    BasicBlock *PostBB, Value *Address, bool InvertPCond, bool InvertQCond,
    DomTreeUpdater *DTU, const DataLayout &DL, const TargetTransformInfo &TTI) {
  // For every pointer, there must be exactly two stores, one coming from
  // PTB or PFB, and the other from QTB or QFB. We don't support more than one
  // store (to any address) in PTB,PFB or QTB,QFB.
  // FIXME: We could relax this restriction with a bit more work and performance
  // testing.
  StoreInst *PStore = findUniqueStoreInBlocks(PTB, PFB);
  StoreInst *QStore = findUniqueStoreInBlocks(QTB, QFB);
  if (!PStore || !QStore)
    return false;

  // Now check the stores are compatible.
  if (!QStore->isUnordered() || !PStore->isUnordered() ||
      PStore->getValueOperand()->getType() !=
          QStore->getValueOperand()->getType())
    return false;

  // Check that sinking the store won't cause program behavior changes. Sinking
  // the store out of the Q blocks won't change any behavior as we're sinking
  // from a block to its unconditional successor. But we're moving a store from
  // the P blocks down through the middle block (QBI) and past both QFB and QTB.
  // So we need to check that there are no aliasing loads or stores in
  // QBI, QTB and QFB. We also need to check there are no conflicting memory
  // operations between PStore and the end of its parent block.
  //
  // The ideal way to do this is to query AliasAnalysis, but we don't
  // preserve AA currently so that is dangerous. Be super safe and just
  // check there are no other memory operations at all.
  for (auto &I : *QFB->getSinglePredecessor())
    if (I.mayReadOrWriteMemory())
      return false;
  for (auto &I : *QFB)
    if (&I != QStore && I.mayReadOrWriteMemory())
      return false;
  if (QTB)
    for (auto &I : *QTB)
      if (&I != QStore && I.mayReadOrWriteMemory())
        return false;
  for (auto I = BasicBlock::iterator(PStore), E = PStore->getParent()->end();
       I != E; ++I)
    if (&*I != PStore && I->mayReadOrWriteMemory())
      return false;

  // If we're not in aggressive mode, we only optimize if we have some
  // confidence that by optimizing we'll allow P and/or Q to be if-converted.
  auto IsWorthwhile = [&](BasicBlock *BB, ArrayRef<StoreInst *> FreeStores) {
    if (!BB)
      return true;
    // Heuristic: if the block can be if-converted/phi-folded and the
    // instructions inside are all cheap (arithmetic/GEPs), it's worthwhile to
    // thread this store.
    InstructionCost Cost = 0;
    InstructionCost Budget =
        PHINodeFoldingThreshold * TargetTransformInfo::TCC_Basic;
    for (auto &I : BB->instructionsWithoutDebug(false)) {
      // Consider terminator instruction to be free.
      if (I.isTerminator())
        continue;
      // If this is one the stores that we want to speculate out of this BB,
      // then don't count it's cost, consider it to be free.
      if (auto *S = dyn_cast<StoreInst>(&I))
        if (llvm::find(FreeStores, S))
          continue;
      // Else, we have a white-list of instructions that we are ak speculating.
      if (!isa<BinaryOperator>(I) && !isa<GetElementPtrInst>(I))
        return false; // Not in white-list - not worthwhile folding.
      // And finally, if this is a non-free instruction that we are okay
      // speculating, ensure that we consider the speculation budget.
      Cost +=
          TTI.getInstructionCost(&I, TargetTransformInfo::TCK_SizeAndLatency);
      if (Cost > Budget)
        return false; // Eagerly refuse to fold as soon as we're out of budget.
    }
    assert(Cost <= Budget &&
           "When we run out of budget we will eagerly return from within the "
           "per-instruction loop.");
    return true;
  };

  const std::array<StoreInst *, 2> FreeStores = {PStore, QStore};
  if (!MergeCondStoresAggressively &&
      (!IsWorthwhile(PTB, FreeStores) || !IsWorthwhile(PFB, FreeStores) ||
       !IsWorthwhile(QTB, FreeStores) || !IsWorthwhile(QFB, FreeStores)))
    return false;

  // If PostBB has more than two predecessors, we need to split it so we can
  // sink the store.
  if (std::next(pred_begin(PostBB), 2) != pred_end(PostBB)) {
    // We know that QFB's only successor is PostBB. And QFB has a single
    // predecessor. If QTB exists, then its only successor is also PostBB.
    // If QTB does not exist, then QFB's only predecessor has a conditional
    // branch to QFB and PostBB.
    BasicBlock *TruePred = QTB ? QTB : QFB->getSinglePredecessor();
    BasicBlock *NewBB =
        SplitBlockPredecessors(PostBB, {QFB, TruePred}, "condstore.split", DTU);
    if (!NewBB)
      return false;
    PostBB = NewBB;
  }

  // OK, we're going to sink the stores to PostBB. The store has to be
  // conditional though, so first create the predicate.
  BranchInst *PBranch =
      cast<BranchInst>(PFB->getSinglePredecessor()->getTerminator());
  BranchInst *QBranch =
      cast<BranchInst>(QFB->getSinglePredecessor()->getTerminator());
  Value *PCond = PBranch->getCondition();
  Value *QCond = QBranch->getCondition();

  Value *PPHI = ensureValueAvailableInSuccessor(PStore->getValueOperand(),
                                                PStore->getParent());
  Value *QPHI = ensureValueAvailableInSuccessor(QStore->getValueOperand(),
                                                QStore->getParent(), PPHI);

  BasicBlock::iterator PostBBFirst = PostBB->getFirstInsertionPt();
  IRBuilder<> QB(PostBB, PostBBFirst);
  QB.SetCurrentDebugLocation(PostBBFirst->getStableDebugLoc());

  InvertPCond ^= (PStore->getParent() != PTB);
  InvertQCond ^= (QStore->getParent() != QTB);
  Value *PPred = InvertPCond ? QB.CreateNot(PCond) : PCond;
  Value *QPred = InvertQCond ? QB.CreateNot(QCond) : QCond;

  Value *CombinedPred = QB.CreateOr(PPred, QPred);

  BasicBlock::iterator InsertPt = QB.GetInsertPoint();
  auto *T = SplitBlockAndInsertIfThen(CombinedPred, InsertPt,
                                      /*Unreachable=*/false,
                                      /*BranchWeights=*/nullptr, DTU);

  QB.SetInsertPoint(T);
  StoreInst *SI = cast<StoreInst>(QB.CreateStore(QPHI, Address));
  SI->setAAMetadata(PStore->getAAMetadata().merge(QStore->getAAMetadata()));
  // Choose the minimum alignment. If we could prove both stores execute, we
  // could use biggest one.  In this case, though, we only know that one of the
  // stores executes.  And we don't know it's safe to take the alignment from a
  // store that doesn't execute.
  SI->setAlignment(std::min(PStore->getAlign(), QStore->getAlign()));

  QStore->eraseFromParent();
  PStore->eraseFromParent();

  return true;
}

static bool mergeConditionalStores(BranchInst *PBI, BranchInst *QBI,
                                   DomTreeUpdater *DTU, const DataLayout &DL,
                                   const TargetTransformInfo &TTI) {
  // The intention here is to find diamonds or triangles (see below) where each
  // conditional block contains a store to the same address. Both of these
  // stores are conditional, so they can't be unconditionally sunk. But it may
  // be profitable to speculatively sink the stores into one merged store at the
  // end, and predicate the merged store on the union of the two conditions of
  // PBI and QBI.
  //
  // This can reduce the number of stores executed if both of the conditions are
  // true, and can allow the blocks to become small enough to be if-converted.
  // This optimization will also chain, so that ladders of test-and-set
  // sequences can be if-converted away.
  //
  // We only deal with simple diamonds or triangles:
  //
  //     PBI       or      PBI        or a combination of the two
  //    /   \               | \
  //   PTB  PFB             |  PFB
  //    \   /               | /
  //     QBI                QBI
  //    /  \                | \
  //   QTB  QFB             |  QFB
  //    \  /                | /
  //    PostBB            PostBB
  //
  // We model triangles as a type of diamond with a nullptr "true" block.
  // Triangles are canonicalized so that the fallthrough edge is represented by
  // a true condition, as in the diagram above.
  BasicBlock *PTB = PBI->getSuccessor(0);
  BasicBlock *PFB = PBI->getSuccessor(1);
  BasicBlock *QTB = QBI->getSuccessor(0);
  BasicBlock *QFB = QBI->getSuccessor(1);
  BasicBlock *PostBB = QFB->getSingleSuccessor();

  // Make sure we have a good guess for PostBB. If QTB's only successor is
  // QFB, then QFB is a better PostBB.
  if (QTB->getSingleSuccessor() == QFB)
    PostBB = QFB;

  // If we couldn't find a good PostBB, stop.
  if (!PostBB)
    return false;

  bool InvertPCond = false, InvertQCond = false;
  // Canonicalize fallthroughs to the true branches.
  if (PFB == QBI->getParent()) {
    std::swap(PFB, PTB);
    InvertPCond = true;
  }
  if (QFB == PostBB) {
    std::swap(QFB, QTB);
    InvertQCond = true;
  }

  // From this point on we can assume PTB or QTB may be fallthroughs but PFB
  // and QFB may not. Model fallthroughs as a nullptr block.
  if (PTB == QBI->getParent())
    PTB = nullptr;
  if (QTB == PostBB)
    QTB = nullptr;

  // Legality bailouts. We must have at least the non-fallthrough blocks and
  // the post-dominating block, and the non-fallthroughs must only have one
  // predecessor.
  auto HasOnePredAndOneSucc = [](BasicBlock *BB, BasicBlock *P, BasicBlock *S) {
    return BB->getSinglePredecessor() == P && BB->getSingleSuccessor() == S;
  };
  if (!HasOnePredAndOneSucc(PFB, PBI->getParent(), QBI->getParent()) ||
      !HasOnePredAndOneSucc(QFB, QBI->getParent(), PostBB))
    return false;
  if ((PTB && !HasOnePredAndOneSucc(PTB, PBI->getParent(), QBI->getParent())) ||
      (QTB && !HasOnePredAndOneSucc(QTB, QBI->getParent(), PostBB)))
    return false;
  if (!QBI->getParent()->hasNUses(2))
    return false;

  // OK, this is a sequence of two diamonds or triangles.
  // Check if there are stores in PTB or PFB that are repeated in QTB or QFB.
  SmallPtrSet<Value *, 4> PStoreAddresses, QStoreAddresses;
  for (auto *BB : {PTB, PFB}) {
    if (!BB)
      continue;
    for (auto &I : *BB)
      if (StoreInst *SI = dyn_cast<StoreInst>(&I))
        PStoreAddresses.insert(SI->getPointerOperand());
  }
  for (auto *BB : {QTB, QFB}) {
    if (!BB)
      continue;
    for (auto &I : *BB)
      if (StoreInst *SI = dyn_cast<StoreInst>(&I))
        QStoreAddresses.insert(SI->getPointerOperand());
  }

  set_intersect(PStoreAddresses, QStoreAddresses);
  // set_intersect mutates PStoreAddresses in place. Rename it here to make it
  // clear what it contains.
  auto &CommonAddresses = PStoreAddresses;

  bool Changed = false;
  for (auto *Address : CommonAddresses)
    Changed |=
        mergeConditionalStoreToAddress(PTB, PFB, QTB, QFB, PostBB, Address,
                                       InvertPCond, InvertQCond, DTU, DL, TTI);
  return Changed;
}

/// If the previous block ended with a widenable branch, determine if reusing
/// the target block is profitable and legal.  This will have the effect of
/// "widening" PBI, but doesn't require us to reason about hosting safety.
static bool tryWidenCondBranchToCondBranch(BranchInst *PBI, BranchInst *BI,
                                           DomTreeUpdater *DTU) {
  // TODO: This can be generalized in two important ways:
  // 1) We can allow phi nodes in IfFalseBB and simply reuse all the input
  //    values from the PBI edge.
  // 2) We can sink side effecting instructions into BI's fallthrough
  //    successor provided they doesn't contribute to computation of
  //    BI's condition.
  BasicBlock *IfTrueBB = PBI->getSuccessor(0);
  BasicBlock *IfFalseBB = PBI->getSuccessor(1);
  if (!isWidenableBranch(PBI) || IfTrueBB != BI->getParent() ||
      !BI->getParent()->getSinglePredecessor())
    return false;
  if (!IfFalseBB->phis().empty())
    return false; // TODO
  // This helps avoid infinite loop with SimplifyCondBranchToCondBranch which
  // may undo the transform done here.
  // TODO: There might be a more fine-grained solution to this.
  if (!llvm::succ_empty(IfFalseBB))
    return false;
  // Use lambda to lazily compute expensive condition after cheap ones.
  auto NoSideEffects = [](BasicBlock &BB) {
    return llvm::none_of(BB, [](const Instruction &I) {
        return I.mayWriteToMemory() || I.mayHaveSideEffects();
      });
  };
  if (BI->getSuccessor(1) != IfFalseBB && // no inf looping
      BI->getSuccessor(1)->getTerminatingDeoptimizeCall() && // profitability
      NoSideEffects(*BI->getParent())) {
    auto *OldSuccessor = BI->getSuccessor(1);
    OldSuccessor->removePredecessor(BI->getParent());
    BI->setSuccessor(1, IfFalseBB);
    if (DTU)
      DTU->applyUpdates(
          {{DominatorTree::Insert, BI->getParent(), IfFalseBB},
           {DominatorTree::Delete, BI->getParent(), OldSuccessor}});
    return true;
  }
  if (BI->getSuccessor(0) != IfFalseBB && // no inf looping
      BI->getSuccessor(0)->getTerminatingDeoptimizeCall() && // profitability
      NoSideEffects(*BI->getParent())) {
    auto *OldSuccessor = BI->getSuccessor(0);
    OldSuccessor->removePredecessor(BI->getParent());
    BI->setSuccessor(0, IfFalseBB);
    if (DTU)
      DTU->applyUpdates(
          {{DominatorTree::Insert, BI->getParent(), IfFalseBB},
           {DominatorTree::Delete, BI->getParent(), OldSuccessor}});
    return true;
  }
  return false;
}

/// If we have a conditional branch as a predecessor of another block,
/// this function tries to simplify it.  We know
/// that PBI and BI are both conditional branches, and BI is in one of the
/// successor blocks of PBI - PBI branches to BI.
static bool SimplifyCondBranchToCondBranch(BranchInst *PBI, BranchInst *BI,
                                           DomTreeUpdater *DTU,
                                           const DataLayout &DL,
                                           const TargetTransformInfo &TTI) {
  assert(PBI->isConditional() && BI->isConditional());
  BasicBlock *BB = BI->getParent();

  // If this block ends with a branch instruction, and if there is a
  // predecessor that ends on a branch of the same condition, make
  // this conditional branch redundant.
  if (PBI->getCondition() == BI->getCondition() &&
      PBI->getSuccessor(0) != PBI->getSuccessor(1)) {
    // Okay, the outcome of this conditional branch is statically
    // knowable.  If this block had a single pred, handle specially, otherwise
    // foldCondBranchOnValueKnownInPredecessor() will handle it.
    if (BB->getSinglePredecessor()) {
      // Turn this into a branch on constant.
      bool CondIsTrue = PBI->getSuccessor(0) == BB;
      BI->setCondition(
          ConstantInt::get(Type::getInt1Ty(BB->getContext()), CondIsTrue));
      return true; // Nuke the branch on constant.
    }
  }

  // If the previous block ended with a widenable branch, determine if reusing
  // the target block is profitable and legal.  This will have the effect of
  // "widening" PBI, but doesn't require us to reason about hosting safety.
  if (tryWidenCondBranchToCondBranch(PBI, BI, DTU))
    return true;

  // If both branches are conditional and both contain stores to the same
  // address, remove the stores from the conditionals and create a conditional
  // merged store at the end.
  if (MergeCondStores && mergeConditionalStores(PBI, BI, DTU, DL, TTI))
    return true;

  // If this is a conditional branch in an empty block, and if any
  // predecessors are a conditional branch to one of our destinations,
  // fold the conditions into logical ops and one cond br.

  // Ignore dbg intrinsics.
  if (&*BB->instructionsWithoutDebug(false).begin() != BI)
    return false;

  int PBIOp, BIOp;
  if (PBI->getSuccessor(0) == BI->getSuccessor(0)) {
    PBIOp = 0;
    BIOp = 0;
  } else if (PBI->getSuccessor(0) == BI->getSuccessor(1)) {
    PBIOp = 0;
    BIOp = 1;
  } else if (PBI->getSuccessor(1) == BI->getSuccessor(0)) {
    PBIOp = 1;
    BIOp = 0;
  } else if (PBI->getSuccessor(1) == BI->getSuccessor(1)) {
    PBIOp = 1;
    BIOp = 1;
  } else {
    return false;
  }

  // Check to make sure that the other destination of this branch
  // isn't BB itself.  If so, this is an infinite loop that will
  // keep getting unwound.
  if (PBI->getSuccessor(PBIOp) == BB)
    return false;

  // If predecessor's branch probability to BB is too low don't merge branches.
  SmallVector<uint32_t, 2> PredWeights;
  if (!PBI->getMetadata(LLVMContext::MD_unpredictable) &&
      extractBranchWeights(*PBI, PredWeights) &&
      (static_cast<uint64_t>(PredWeights[0]) + PredWeights[1]) != 0) {

    BranchProbability CommonDestProb = BranchProbability::getBranchProbability(
        PredWeights[PBIOp],
        static_cast<uint64_t>(PredWeights[0]) + PredWeights[1]);

    BranchProbability Likely = TTI.getPredictableBranchThreshold();
    if (CommonDestProb >= Likely)
      return false;
  }

  // Do not perform this transformation if it would require
  // insertion of a large number of select instructions. For targets
  // without predication/cmovs, this is a big pessimization.

  BasicBlock *CommonDest = PBI->getSuccessor(PBIOp);
  BasicBlock *RemovedDest = PBI->getSuccessor(PBIOp ^ 1);
  unsigned NumPhis = 0;
  for (BasicBlock::iterator II = CommonDest->begin(); isa<PHINode>(II);
       ++II, ++NumPhis) {
    if (NumPhis > 2) // Disable this xform.
      return false;
  }

  // Finally, if everything is ok, fold the branches to logical ops.
  BasicBlock *OtherDest = BI->getSuccessor(BIOp ^ 1);

  LLVM_DEBUG(dbgs() << "FOLDING BRs:" << *PBI->getParent()
                    << "AND: " << *BI->getParent());

  SmallVector<DominatorTree::UpdateType, 5> Updates;

  // If OtherDest *is* BB, then BB is a basic block with a single conditional
  // branch in it, where one edge (OtherDest) goes back to itself but the other
  // exits.  We don't *know* that the program avoids the infinite loop
  // (even though that seems likely).  If we do this xform naively, we'll end up
  // recursively unpeeling the loop.  Since we know that (after the xform is
  // done) that the block *is* infinite if reached, we just make it an obviously
  // infinite loop with no cond branch.
  if (OtherDest == BB) {
    // Insert it at the end of the function, because it's either code,
    // or it won't matter if it's hot. :)
    BasicBlock *InfLoopBlock =
        BasicBlock::Create(BB->getContext(), "infloop", BB->getParent());
    BranchInst::Create(InfLoopBlock, InfLoopBlock);
    if (DTU)
      Updates.push_back({DominatorTree::Insert, InfLoopBlock, InfLoopBlock});
    OtherDest = InfLoopBlock;
  }

  LLVM_DEBUG(dbgs() << *PBI->getParent()->getParent());

  // BI may have other predecessors.  Because of this, we leave
  // it alone, but modify PBI.

  // Make sure we get to CommonDest on True&True directions.
  Value *PBICond = PBI->getCondition();
  IRBuilder<NoFolder> Builder(PBI);
  if (PBIOp)
    PBICond = Builder.CreateNot(PBICond, PBICond->getName() + ".not");

  Value *BICond = BI->getCondition();
  if (BIOp)
    BICond = Builder.CreateNot(BICond, BICond->getName() + ".not");

  // Merge the conditions.
  Value *Cond =
      createLogicalOp(Builder, Instruction::Or, PBICond, BICond, "brmerge");

  // Modify PBI to branch on the new condition to the new dests.
  PBI->setCondition(Cond);
  PBI->setSuccessor(0, CommonDest);
  PBI->setSuccessor(1, OtherDest);

  if (DTU) {
    Updates.push_back({DominatorTree::Insert, PBI->getParent(), OtherDest});
    Updates.push_back({DominatorTree::Delete, PBI->getParent(), RemovedDest});

    DTU->applyUpdates(Updates);
  }

  // Update branch weight for PBI.
  uint64_t PredTrueWeight, PredFalseWeight, SuccTrueWeight, SuccFalseWeight;
  uint64_t PredCommon, PredOther, SuccCommon, SuccOther;
  bool HasWeights =
      extractPredSuccWeights(PBI, BI, PredTrueWeight, PredFalseWeight,
                             SuccTrueWeight, SuccFalseWeight);
  if (HasWeights) {
    PredCommon = PBIOp ? PredFalseWeight : PredTrueWeight;
    PredOther = PBIOp ? PredTrueWeight : PredFalseWeight;
    SuccCommon = BIOp ? SuccFalseWeight : SuccTrueWeight;
    SuccOther = BIOp ? SuccTrueWeight : SuccFalseWeight;
    // The weight to CommonDest should be PredCommon * SuccTotal +
    //                                    PredOther * SuccCommon.
    // The weight to OtherDest should be PredOther * SuccOther.
    uint64_t NewWeights[2] = {PredCommon * (SuccCommon + SuccOther) +
                                  PredOther * SuccCommon,
                              PredOther * SuccOther};
    // Halve the weights if any of them cannot fit in an uint32_t
    fitWeights(NewWeights);

    setBranchWeights(PBI, NewWeights[0], NewWeights[1], /*IsExpected=*/false);
  }

  // OtherDest may have phi nodes.  If so, add an entry from PBI's
  // block that are identical to the entries for BI's block.
  addPredecessorToBlock(OtherDest, PBI->getParent(), BB);

  // We know that the CommonDest already had an edge from PBI to
  // it.  If it has PHIs though, the PHIs may have different
  // entries for BB and PBI's BB.  If so, insert a select to make
  // them agree.
  for (PHINode &PN : CommonDest->phis()) {
    Value *BIV = PN.getIncomingValueForBlock(BB);
    unsigned PBBIdx = PN.getBasicBlockIndex(PBI->getParent());
    Value *PBIV = PN.getIncomingValue(PBBIdx);
    if (BIV != PBIV) {
      // Insert a select in PBI to pick the right value.
      SelectInst *NV = cast<SelectInst>(
          Builder.CreateSelect(PBICond, PBIV, BIV, PBIV->getName() + ".mux"));
      PN.setIncomingValue(PBBIdx, NV);
      // Although the select has the same condition as PBI, the original branch
      // weights for PBI do not apply to the new select because the select's
      // 'logical' edges are incoming edges of the phi that is eliminated, not
      // the outgoing edges of PBI.
      if (HasWeights) {
        uint64_t PredCommon = PBIOp ? PredFalseWeight : PredTrueWeight;
        uint64_t PredOther = PBIOp ? PredTrueWeight : PredFalseWeight;
        uint64_t SuccCommon = BIOp ? SuccFalseWeight : SuccTrueWeight;
        uint64_t SuccOther = BIOp ? SuccTrueWeight : SuccFalseWeight;
        // The weight to PredCommonDest should be PredCommon * SuccTotal.
        // The weight to PredOtherDest should be PredOther * SuccCommon.
        uint64_t NewWeights[2] = {PredCommon * (SuccCommon + SuccOther),
                                  PredOther * SuccCommon};

        fitWeights(NewWeights);

        setBranchWeights(NV, NewWeights[0], NewWeights[1],
                         /*IsExpected=*/false);
      }
    }
  }

  LLVM_DEBUG(dbgs() << "INTO: " << *PBI->getParent());
  LLVM_DEBUG(dbgs() << *PBI->getParent()->getParent());

  // This basic block is probably dead.  We know it has at least
  // one fewer predecessor.
  return true;
}

// Simplifies a terminator by replacing it with a branch to TrueBB if Cond is
// true or to FalseBB if Cond is false.
// Takes care of updating the successors and removing the old terminator.
// Also makes sure not to introduce new successors by assuming that edges to
// non-successor TrueBBs and FalseBBs aren't reachable.
bool SimplifyCFGOpt::simplifyTerminatorOnSelect(Instruction *OldTerm,
                                                Value *Cond, BasicBlock *TrueBB,
                                                BasicBlock *FalseBB,
                                                uint32_t TrueWeight,
                                                uint32_t FalseWeight) {
  auto *BB = OldTerm->getParent();
  // Remove any superfluous successor edges from the CFG.
  // First, figure out which successors to preserve.
  // If TrueBB and FalseBB are equal, only try to preserve one copy of that
  // successor.
  BasicBlock *KeepEdge1 = TrueBB;
  BasicBlock *KeepEdge2 = TrueBB != FalseBB ? FalseBB : nullptr;

  SmallSetVector<BasicBlock *, 2> RemovedSuccessors;

  // Then remove the rest.
  for (BasicBlock *Succ : successors(OldTerm)) {
    // Make sure only to keep exactly one copy of each edge.
    if (Succ == KeepEdge1)
      KeepEdge1 = nullptr;
    else if (Succ == KeepEdge2)
      KeepEdge2 = nullptr;
    else {
      Succ->removePredecessor(BB,
                              /*KeepOneInputPHIs=*/true);

      if (Succ != TrueBB && Succ != FalseBB)
        RemovedSuccessors.insert(Succ);
    }
  }

  IRBuilder<> Builder(OldTerm);
  Builder.SetCurrentDebugLocation(OldTerm->getDebugLoc());

  // Insert an appropriate new terminator.
  if (!KeepEdge1 && !KeepEdge2) {
    if (TrueBB == FalseBB) {
      // We were only looking for one successor, and it was present.
      // Create an unconditional branch to it.
      Builder.CreateBr(TrueBB);
    } else {
      // We found both of the successors we were looking for.
      // Create a conditional branch sharing the condition of the select.
      BranchInst *NewBI = Builder.CreateCondBr(Cond, TrueBB, FalseBB);
      if (TrueWeight != FalseWeight)
        setBranchWeights(NewBI, TrueWeight, FalseWeight, /*IsExpected=*/false);
    }
  } else if (KeepEdge1 && (KeepEdge2 || TrueBB == FalseBB)) {
    // Neither of the selected blocks were successors, so this
    // terminator must be unreachable.
    new UnreachableInst(OldTerm->getContext(), OldTerm->getIterator());
  } else {
    // One of the selected values was a successor, but the other wasn't.
    // Insert an unconditional branch to the one that was found;
    // the edge to the one that wasn't must be unreachable.
    if (!KeepEdge1) {
      // Only TrueBB was found.
      Builder.CreateBr(TrueBB);
    } else {
      // Only FalseBB was found.
      Builder.CreateBr(FalseBB);
    }
  }

  eraseTerminatorAndDCECond(OldTerm);

  if (DTU) {
    SmallVector<DominatorTree::UpdateType, 2> Updates;
    Updates.reserve(RemovedSuccessors.size());
    for (auto *RemovedSuccessor : RemovedSuccessors)
      Updates.push_back({DominatorTree::Delete, BB, RemovedSuccessor});
    DTU->applyUpdates(Updates);
  }

  return true;
}

// Replaces
//   (switch (select cond, X, Y)) on constant X, Y
// with a branch - conditional if X and Y lead to distinct BBs,
// unconditional otherwise.
bool SimplifyCFGOpt::simplifySwitchOnSelect(SwitchInst *SI,
                                            SelectInst *Select) {
  // Check for constant integer values in the select.
  ConstantInt *TrueVal = dyn_cast<ConstantInt>(Select->getTrueValue());
  ConstantInt *FalseVal = dyn_cast<ConstantInt>(Select->getFalseValue());
  if (!TrueVal || !FalseVal)
    return false;

  // Find the relevant condition and destinations.
  Value *Condition = Select->getCondition();
  BasicBlock *TrueBB = SI->findCaseValue(TrueVal)->getCaseSuccessor();
  BasicBlock *FalseBB = SI->findCaseValue(FalseVal)->getCaseSuccessor();

  // Get weight for TrueBB and FalseBB.
  uint32_t TrueWeight = 0, FalseWeight = 0;
  SmallVector<uint64_t, 8> Weights;
  bool HasWeights = hasBranchWeightMD(*SI);
  if (HasWeights) {
    getBranchWeights(SI, Weights);
    if (Weights.size() == 1 + SI->getNumCases()) {
      TrueWeight =
          (uint32_t)Weights[SI->findCaseValue(TrueVal)->getSuccessorIndex()];
      FalseWeight =
          (uint32_t)Weights[SI->findCaseValue(FalseVal)->getSuccessorIndex()];
    }
  }

  // Perform the actual simplification.
  return simplifyTerminatorOnSelect(SI, Condition, TrueBB, FalseBB, TrueWeight,
                                    FalseWeight);
}

// Replaces
//   (indirectbr (select cond, blockaddress(@fn, BlockA),
//                             blockaddress(@fn, BlockB)))
// with
//   (br cond, BlockA, BlockB).
bool SimplifyCFGOpt::simplifyIndirectBrOnSelect(IndirectBrInst *IBI,
                                                SelectInst *SI) {
  // Check that both operands of the select are block addresses.
  BlockAddress *TBA = dyn_cast<BlockAddress>(SI->getTrueValue());
  BlockAddress *FBA = dyn_cast<BlockAddress>(SI->getFalseValue());
  if (!TBA || !FBA)
    return false;

  // Extract the actual blocks.
  BasicBlock *TrueBB = TBA->getBasicBlock();
  BasicBlock *FalseBB = FBA->getBasicBlock();

  // Perform the actual simplification.
  return simplifyTerminatorOnSelect(IBI, SI->getCondition(), TrueBB, FalseBB, 0,
                                    0);
}

/// This is called when we find an icmp instruction
/// (a seteq/setne with a constant) as the only instruction in a
/// block that ends with an uncond branch.  We are looking for a very specific
/// pattern that occurs when "A == 1 || A == 2 || A == 3" gets simplified.  In
/// this case, we merge the first two "or's of icmp" into a switch, but then the
/// default value goes to an uncond block with a seteq in it, we get something
/// like:
///
///   switch i8 %A, label %DEFAULT [ i8 1, label %end    i8 2, label %end ]
/// DEFAULT:
///   %tmp = icmp eq i8 %A, 92
///   br label %end
/// end:
///   ... = phi i1 [ true, %entry ], [ %tmp, %DEFAULT ], [ true, %entry ]
///
/// We prefer to split the edge to 'end' so that there is a true/false entry to
/// the PHI, merging the third icmp into the switch.
bool SimplifyCFGOpt::tryToSimplifyUncondBranchWithICmpInIt(
    ICmpInst *ICI, IRBuilder<> &Builder) {
  BasicBlock *BB = ICI->getParent();

  // If the block has any PHIs in it or the icmp has multiple uses, it is too
  // complex.
  if (isa<PHINode>(BB->begin()) || !ICI->hasOneUse())
    return false;

  Value *V = ICI->getOperand(0);
  ConstantInt *Cst = cast<ConstantInt>(ICI->getOperand(1));

  // The pattern we're looking for is where our only predecessor is a switch on
  // 'V' and this block is the default case for the switch.  In this case we can
  // fold the compared value into the switch to simplify things.
  BasicBlock *Pred = BB->getSinglePredecessor();
  if (!Pred || !isa<SwitchInst>(Pred->getTerminator()))
    return false;

  SwitchInst *SI = cast<SwitchInst>(Pred->getTerminator());
  if (SI->getCondition() != V)
    return false;

  // If BB is reachable on a non-default case, then we simply know the value of
  // V in this block.  Substitute it and constant fold the icmp instruction
  // away.
  if (SI->getDefaultDest() != BB) {
    ConstantInt *VVal = SI->findCaseDest(BB);
    assert(VVal && "Should have a unique destination value");
    ICI->setOperand(0, VVal);

    if (Value *V = simplifyInstruction(ICI, {DL, ICI})) {
      ICI->replaceAllUsesWith(V);
      ICI->eraseFromParent();
    }
    // BB is now empty, so it is likely to simplify away.
    return requestResimplify();
  }

  // Ok, the block is reachable from the default dest.  If the constant we're
  // comparing exists in one of the other edges, then we can constant fold ICI
  // and zap it.
  if (SI->findCaseValue(Cst) != SI->case_default()) {
    Value *V;
    if (ICI->getPredicate() == ICmpInst::ICMP_EQ)
      V = ConstantInt::getFalse(BB->getContext());
    else
      V = ConstantInt::getTrue(BB->getContext());

    ICI->replaceAllUsesWith(V);
    ICI->eraseFromParent();
    // BB is now empty, so it is likely to simplify away.
    return requestResimplify();
  }

  // The use of the icmp has to be in the 'end' block, by the only PHI node in
  // the block.
  BasicBlock *SuccBlock = BB->getTerminator()->getSuccessor(0);
  PHINode *PHIUse = dyn_cast<PHINode>(ICI->user_back());
  if (PHIUse == nullptr || PHIUse != &SuccBlock->front() ||
      isa<PHINode>(++BasicBlock::iterator(PHIUse)))
    return false;

  // If the icmp is a SETEQ, then the default dest gets false, the new edge gets
  // true in the PHI.
  Constant *DefaultCst = ConstantInt::getTrue(BB->getContext());
  Constant *NewCst = ConstantInt::getFalse(BB->getContext());

  if (ICI->getPredicate() == ICmpInst::ICMP_EQ)
    std::swap(DefaultCst, NewCst);

  // Replace ICI (which is used by the PHI for the default value) with true or
  // false depending on if it is EQ or NE.
  ICI->replaceAllUsesWith(DefaultCst);
  ICI->eraseFromParent();

  SmallVector<DominatorTree::UpdateType, 2> Updates;

  // Okay, the switch goes to this block on a default value.  Add an edge from
  // the switch to the merge point on the compared value.
  BasicBlock *NewBB =
      BasicBlock::Create(BB->getContext(), "switch.edge", BB->getParent(), BB);
  {
    SwitchInstProfUpdateWrapper SIW(*SI);
    auto W0 = SIW.getSuccessorWeight(0);
    SwitchInstProfUpdateWrapper::CaseWeightOpt NewW;
    if (W0) {
      NewW = ((uint64_t(*W0) + 1) >> 1);
      SIW.setSuccessorWeight(0, *NewW);
    }
    SIW.addCase(Cst, NewBB, NewW);
    if (DTU)
      Updates.push_back({DominatorTree::Insert, Pred, NewBB});
  }

  // NewBB branches to the phi block, add the uncond branch and the phi entry.
  Builder.SetInsertPoint(NewBB);
  Builder.SetCurrentDebugLocation(SI->getDebugLoc());
  Builder.CreateBr(SuccBlock);
  PHIUse->addIncoming(NewCst, NewBB);
  if (DTU) {
    Updates.push_back({DominatorTree::Insert, NewBB, SuccBlock});
    DTU->applyUpdates(Updates);
  }
  return true;
}

/// The specified branch is a conditional branch.
/// Check to see if it is branching on an or/and chain of icmp instructions, and
/// fold it into a switch instruction if so.
bool SimplifyCFGOpt::simplifyBranchOnICmpChain(BranchInst *BI,
                                               IRBuilder<> &Builder,
                                               const DataLayout &DL) {
  Instruction *Cond = dyn_cast<Instruction>(BI->getCondition());
  if (!Cond)
    return false;

  // Change br (X == 0 | X == 1), T, F into a switch instruction.
  // If this is a bunch of seteq's or'd together, or if it's a bunch of
  // 'setne's and'ed together, collect them.

  // Try to gather values from a chain of and/or to be turned into a switch
  ConstantComparesGatherer ConstantCompare(Cond, DL);
  // Unpack the result
  SmallVectorImpl<ConstantInt *> &Values = ConstantCompare.Vals;
  Value *CompVal = ConstantCompare.CompValue;
  unsigned UsedICmps = ConstantCompare.UsedICmps;
  Value *ExtraCase = ConstantCompare.Extra;
  bool TrueWhenEqual = ConstantCompare.IsEq;

  // If we didn't have a multiply compared value, fail.
  if (!CompVal)
    return false;

  // Avoid turning single icmps into a switch.
  if (UsedICmps <= 1)
    return false;

  // There might be duplicate constants in the list, which the switch
  // instruction can't handle, remove them now.
  array_pod_sort(Values.begin(), Values.end(), constantIntSortPredicate);
  Values.erase(llvm::unique(Values), Values.end());

  // If Extra was used, we require at least two switch values to do the
  // transformation.  A switch with one value is just a conditional branch.
  if (ExtraCase && Values.size() < 2)
    return false;

  // TODO: Preserve branch weight metadata, similarly to how
  // foldValueComparisonIntoPredecessors preserves it.

  // Figure out which block is which destination.
  BasicBlock *DefaultBB = BI->getSuccessor(1);
  BasicBlock *EdgeBB = BI->getSuccessor(0);
  if (!TrueWhenEqual)
    std::swap(DefaultBB, EdgeBB);

  BasicBlock *BB = BI->getParent();

  LLVM_DEBUG(dbgs() << "Converting 'icmp' chain with " << Values.size()
                    << " cases into SWITCH.  BB is:\n"
                    << *BB);

  SmallVector<DominatorTree::UpdateType, 2> Updates;

  // If there are any extra values that couldn't be folded into the switch
  // then we evaluate them with an explicit branch first. Split the block
  // right before the condbr to handle it.
  if (ExtraCase) {
    BasicBlock *NewBB = SplitBlock(BB, BI, DTU, /*LI=*/nullptr,
                                   /*MSSAU=*/nullptr, "switch.early.test");

    // Remove the uncond branch added to the old block.
    Instruction *OldTI = BB->getTerminator();
    Builder.SetInsertPoint(OldTI);

    // There can be an unintended UB if extra values are Poison. Before the
    // transformation, extra values may not be evaluated according to the
    // condition, and it will not raise UB. But after transformation, we are
    // evaluating extra values before checking the condition, and it will raise
    // UB. It can be solved by adding freeze instruction to extra values.
    AssumptionCache *AC = Options.AC;

    if (!isGuaranteedNotToBeUndefOrPoison(ExtraCase, AC, BI, nullptr))
      ExtraCase = Builder.CreateFreeze(ExtraCase);

    if (TrueWhenEqual)
      Builder.CreateCondBr(ExtraCase, EdgeBB, NewBB);
    else
      Builder.CreateCondBr(ExtraCase, NewBB, EdgeBB);

    OldTI->eraseFromParent();

    if (DTU)
      Updates.push_back({DominatorTree::Insert, BB, EdgeBB});

    // If there are PHI nodes in EdgeBB, then we need to add a new entry to them
    // for the edge we just added.
    addPredecessorToBlock(EdgeBB, BB, NewBB);

    LLVM_DEBUG(dbgs() << "  ** 'icmp' chain unhandled condition: " << *ExtraCase
                      << "\nEXTRABB = " << *BB);
    BB = NewBB;
  }

  Builder.SetInsertPoint(BI);
  // Convert pointer to int before we switch.
  if (CompVal->getType()->isPointerTy()) {
    CompVal = Builder.CreatePtrToInt(
        CompVal, DL.getIntPtrType(CompVal->getType()), "magicptr");
  }

  // Create the new switch instruction now.
  SwitchInst *New = Builder.CreateSwitch(CompVal, DefaultBB, Values.size());

  // Add all of the 'cases' to the switch instruction.
  for (ConstantInt *Val : Values)
    New->addCase(Val, EdgeBB);

  // We added edges from PI to the EdgeBB.  As such, if there were any
  // PHI nodes in EdgeBB, they need entries to be added corresponding to
  // the number of edges added.
  for (BasicBlock::iterator BBI = EdgeBB->begin(); isa<PHINode>(BBI); ++BBI) {
    PHINode *PN = cast<PHINode>(BBI);
    Value *InVal = PN->getIncomingValueForBlock(BB);
    for (unsigned i = 0, e = Values.size() - 1; i != e; ++i)
      PN->addIncoming(InVal, BB);
  }

  // Erase the old branch instruction.
  eraseTerminatorAndDCECond(BI);
  if (DTU)
    DTU->applyUpdates(Updates);

  LLVM_DEBUG(dbgs() << "  ** 'icmp' chain result is:\n" << *BB << '\n');
  return true;
}

bool SimplifyCFGOpt::simplifyResume(ResumeInst *RI, IRBuilder<> &Builder) {
  if (isa<PHINode>(RI->getValue()))
    return simplifyCommonResume(RI);
  else if (isa<LandingPadInst>(RI->getParent()->getFirstNonPHIIt()) &&
           RI->getValue() == &*RI->getParent()->getFirstNonPHIIt())
    // The resume must unwind the exception that caused control to branch here.
    return simplifySingleResume(RI);

  return false;
}

// Check if cleanup block is empty
static bool isCleanupBlockEmpty(iterator_range<BasicBlock::iterator> R) {
  for (Instruction &I : R) {
    auto *II = dyn_cast<IntrinsicInst>(&I);
    if (!II)
      return false;

    Intrinsic::ID IntrinsicID = II->getIntrinsicID();
    switch (IntrinsicID) {
    case Intrinsic::dbg_declare:
    case Intrinsic::dbg_value:
    case Intrinsic::dbg_label:
    case Intrinsic::lifetime_end:
      break;
    default:
      return false;
    }
  }
  return true;
}

// Simplify resume that is shared by several landing pads (phi of landing pad).
bool SimplifyCFGOpt::simplifyCommonResume(ResumeInst *RI) {
  BasicBlock *BB = RI->getParent();

  // Check that there are no other instructions except for debug and lifetime
  // intrinsics between the phi's and resume instruction.
  if (!isCleanupBlockEmpty(make_range(RI->getParent()->getFirstNonPHIIt(),
                                      BB->getTerminator()->getIterator())))
    return false;

  SmallSetVector<BasicBlock *, 4> TrivialUnwindBlocks;
  auto *PhiLPInst = cast<PHINode>(RI->getValue());

  // Check incoming blocks to see if any of them are trivial.
  for (unsigned Idx = 0, End = PhiLPInst->getNumIncomingValues(); Idx != End;
       Idx++) {
    auto *IncomingBB = PhiLPInst->getIncomingBlock(Idx);
    auto *IncomingValue = PhiLPInst->getIncomingValue(Idx);

    // If the block has other successors, we can not delete it because
    // it has other dependents.
    if (IncomingBB->getUniqueSuccessor() != BB)
      continue;

    auto *LandingPad = dyn_cast<LandingPadInst>(IncomingBB->getFirstNonPHIIt());
    // Not the landing pad that caused the control to branch here.
    if (IncomingValue != LandingPad)
      continue;

    if (isCleanupBlockEmpty(
            make_range(LandingPad->getNextNode(), IncomingBB->getTerminator())))
      TrivialUnwindBlocks.insert(IncomingBB);
  }

  // If no trivial unwind blocks, don't do any simplifications.
  if (TrivialUnwindBlocks.empty())
    return false;

  // Turn all invokes that unwind here into calls.
  for (auto *TrivialBB : TrivialUnwindBlocks) {
    // Blocks that will be simplified should be removed from the phi node.
    // Note there could be multiple edges to the resume block, and we need
    // to remove them all.
    while (PhiLPInst->getBasicBlockIndex(TrivialBB) != -1)
      BB->removePredecessor(TrivialBB, true);

    for (BasicBlock *Pred :
         llvm::make_early_inc_range(predecessors(TrivialBB))) {
      removeUnwindEdge(Pred, DTU);
      ++NumInvokes;
    }

    // In each SimplifyCFG run, only the current processed block can be erased.
    // Otherwise, it will break the iteration of SimplifyCFG pass. So instead
    // of erasing TrivialBB, we only remove the branch to the common resume
    // block so that we can later erase the resume block since it has no
    // predecessors.
    TrivialBB->getTerminator()->eraseFromParent();
    new UnreachableInst(RI->getContext(), TrivialBB);
    if (DTU)
      DTU->applyUpdates({{DominatorTree::Delete, TrivialBB, BB}});
  }

  // Delete the resume block if all its predecessors have been removed.
  if (pred_empty(BB))
    DeleteDeadBlock(BB, DTU);

  return !TrivialUnwindBlocks.empty();
}

// Simplify resume that is only used by a single (non-phi) landing pad.
bool SimplifyCFGOpt::simplifySingleResume(ResumeInst *RI) {
  BasicBlock *BB = RI->getParent();
  auto *LPInst = cast<LandingPadInst>(BB->getFirstNonPHIIt());
  assert(RI->getValue() == LPInst &&
         "Resume must unwind the exception that caused control to here");

  // Check that there are no other instructions except for debug intrinsics.
  if (!isCleanupBlockEmpty(
          make_range<Instruction *>(LPInst->getNextNode(), RI)))
    return false;

  // Turn all invokes that unwind here into calls and delete the basic block.
  for (BasicBlock *Pred : llvm::make_early_inc_range(predecessors(BB))) {
    removeUnwindEdge(Pred, DTU);
    ++NumInvokes;
  }

  // The landingpad is now unreachable.  Zap it.
  DeleteDeadBlock(BB, DTU);
  return true;
}

static bool removeEmptyCleanup(CleanupReturnInst *RI, DomTreeUpdater *DTU) {
  // If this is a trivial cleanup pad that executes no instructions, it can be
  // eliminated.  If the cleanup pad continues to the caller, any predecessor
  // that is an EH pad will be updated to continue to the caller and any
  // predecessor that terminates with an invoke instruction will have its invoke
  // instruction converted to a call instruction.  If the cleanup pad being
  // simplified does not continue to the caller, each predecessor will be
  // updated to continue to the unwind destination of the cleanup pad being
  // simplified.
  BasicBlock *BB = RI->getParent();
  CleanupPadInst *CPInst = RI->getCleanupPad();
  if (CPInst->getParent() != BB)
    // This isn't an empty cleanup.
    return false;

  // We cannot kill the pad if it has multiple uses.  This typically arises
  // from unreachable basic blocks.
  if (!CPInst->hasOneUse())
    return false;

  // Check that there are no other instructions except for benign intrinsics.
  if (!isCleanupBlockEmpty(
          make_range<Instruction *>(CPInst->getNextNode(), RI)))
    return false;

  // If the cleanup return we are simplifying unwinds to the caller, this will
  // set UnwindDest to nullptr.
  BasicBlock *UnwindDest = RI->getUnwindDest();

  // We're about to remove BB from the control flow.  Before we do, sink any
  // PHINodes into the unwind destination.  Doing this before changing the
  // control flow avoids some potentially slow checks, since we can currently
  // be certain that UnwindDest and BB have no common predecessors (since they
  // are both EH pads).
  if (UnwindDest) {
    // First, go through the PHI nodes in UnwindDest and update any nodes that
    // reference the block we are removing
    for (PHINode &DestPN : UnwindDest->phis()) {
      int Idx = DestPN.getBasicBlockIndex(BB);
      // Since BB unwinds to UnwindDest, it has to be in the PHI node.
      assert(Idx != -1);
      // This PHI node has an incoming value that corresponds to a control
      // path through the cleanup pad we are removing.  If the incoming
      // value is in the cleanup pad, it must be a PHINode (because we
      // verified above that the block is otherwise empty).  Otherwise, the
      // value is either a constant or a value that dominates the cleanup
      // pad being removed.
      //
      // Because BB and UnwindDest are both EH pads, all of their
      // predecessors must unwind to these blocks, and since no instruction
      // can have multiple unwind destinations, there will be no overlap in
      // incoming blocks between SrcPN and DestPN.
      Value *SrcVal = DestPN.getIncomingValue(Idx);
      PHINode *SrcPN = dyn_cast<PHINode>(SrcVal);

      bool NeedPHITranslation = SrcPN && SrcPN->getParent() == BB;
      for (auto *Pred : predecessors(BB)) {
        Value *Incoming =
            NeedPHITranslation ? SrcPN->getIncomingValueForBlock(Pred) : SrcVal;
        DestPN.addIncoming(Incoming, Pred);
      }
    }

    // Sink any remaining PHI nodes directly into UnwindDest.
    BasicBlock::iterator InsertPt = UnwindDest->getFirstNonPHIIt();
    for (PHINode &PN : make_early_inc_range(BB->phis())) {
      if (PN.use_empty() || !PN.isUsedOutsideOfBlock(BB))
        // If the PHI node has no uses or all of its uses are in this basic
        // block (meaning they are debug or lifetime intrinsics), just leave
        // it.  It will be erased when we erase BB below.
        continue;

      // Otherwise, sink this PHI node into UnwindDest.
      // Any predecessors to UnwindDest which are not already represented
      // must be back edges which inherit the value from the path through
      // BB.  In this case, the PHI value must reference itself.
      for (auto *pred : predecessors(UnwindDest))
        if (pred != BB)
          PN.addIncoming(&PN, pred);
      PN.moveBefore(InsertPt);
      // Also, add a dummy incoming value for the original BB itself,
      // so that the PHI is well-formed until we drop said predecessor.
      PN.addIncoming(PoisonValue::get(PN.getType()), BB);
    }
  }

  std::vector<DominatorTree::UpdateType> Updates;

  // We use make_early_inc_range here because we will remove all predecessors.
  for (BasicBlock *PredBB : llvm::make_early_inc_range(predecessors(BB))) {
    if (UnwindDest == nullptr) {
      if (DTU) {
        DTU->applyUpdates(Updates);
        Updates.clear();
      }
      removeUnwindEdge(PredBB, DTU);
      ++NumInvokes;
    } else {
      BB->removePredecessor(PredBB);
      Instruction *TI = PredBB->getTerminator();
      TI->replaceUsesOfWith(BB, UnwindDest);
      if (DTU) {
        Updates.push_back({DominatorTree::Insert, PredBB, UnwindDest});
        Updates.push_back({DominatorTree::Delete, PredBB, BB});
      }
    }
  }

  if (DTU)
    DTU->applyUpdates(Updates);

  DeleteDeadBlock(BB, DTU);

  return true;
}

// Try to merge two cleanuppads together.
static bool mergeCleanupPad(CleanupReturnInst *RI) {
  // Skip any cleanuprets which unwind to caller, there is nothing to merge
  // with.
  BasicBlock *UnwindDest = RI->getUnwindDest();
  if (!UnwindDest)
    return false;

  // This cleanupret isn't the only predecessor of this cleanuppad, it wouldn't
  // be safe to merge without code duplication.
  if (UnwindDest->getSinglePredecessor() != RI->getParent())
    return false;

  // Verify that our cleanuppad's unwind destination is another cleanuppad.
  auto *SuccessorCleanupPad = dyn_cast<CleanupPadInst>(&UnwindDest->front());
  if (!SuccessorCleanupPad)
    return false;

  CleanupPadInst *PredecessorCleanupPad = RI->getCleanupPad();
  // Replace any uses of the successor cleanupad with the predecessor pad
  // The only cleanuppad uses should be this cleanupret, it's cleanupret and
  // funclet bundle operands.
  SuccessorCleanupPad->replaceAllUsesWith(PredecessorCleanupPad);
  // Remove the old cleanuppad.
  SuccessorCleanupPad->eraseFromParent();
  // Now, we simply replace the cleanupret with a branch to the unwind
  // destination.
  BranchInst::Create(UnwindDest, RI->getParent());
  RI->eraseFromParent();

  return true;
}

bool SimplifyCFGOpt::simplifyCleanupReturn(CleanupReturnInst *RI) {
  // It is possible to transiantly have an undef cleanuppad operand because we
  // have deleted some, but not all, dead blocks.
  // Eventually, this block will be deleted.
  if (isa<UndefValue>(RI->getOperand(0)))
    return false;

  if (mergeCleanupPad(RI))
    return true;

  if (removeEmptyCleanup(RI, DTU))
    return true;

  return false;
}

// WARNING: keep in sync with InstCombinerImpl::visitUnreachableInst()!
bool SimplifyCFGOpt::simplifyUnreachable(UnreachableInst *UI) {
  BasicBlock *BB = UI->getParent();

  bool Changed = false;

  // Ensure that any debug-info records that used to occur after the Unreachable
  // are moved to in front of it -- otherwise they'll "dangle" at the end of
  // the block.
  BB->flushTerminatorDbgRecords();

  // Debug-info records on the unreachable inst itself should be deleted, as
  // below we delete everything past the final executable instruction.
  UI->dropDbgRecords();

  // If there are any instructions immediately before the unreachable that can
  // be removed, do so.
  while (UI->getIterator() != BB->begin()) {
    BasicBlock::iterator BBI = UI->getIterator();
    --BBI;

    if (!isGuaranteedToTransferExecutionToSuccessor(&*BBI))
      break; // Can not drop any more instructions. We're done here.
    // Otherwise, this instruction can be freely erased,
    // even if it is not side-effect free.

    // Note that deleting EH's here is in fact okay, although it involves a bit
    // of subtle reasoning. If this inst is an EH, all the predecessors of this
    // block will be the unwind edges of Invoke/CatchSwitch/CleanupReturn,
    // and we can therefore guarantee this block will be erased.

    // If we're deleting this, we're deleting any subsequent debug info, so
    // delete DbgRecords.
    BBI->dropDbgRecords();

    // Delete this instruction (any uses are guaranteed to be dead)
    BBI->replaceAllUsesWith(PoisonValue::get(BBI->getType()));
    BBI->eraseFromParent();
    Changed = true;
  }

  // If the unreachable instruction is the first in the block, take a gander
  // at all of the predecessors of this instruction, and simplify them.
  if (&BB->front() != UI)
    return Changed;

  std::vector<DominatorTree::UpdateType> Updates;

  SmallSetVector<BasicBlock *, 8> Preds(pred_begin(BB), pred_end(BB));
  for (BasicBlock *Predecessor : Preds) {
    Instruction *TI = Predecessor->getTerminator();
    IRBuilder<> Builder(TI);
    if (auto *BI = dyn_cast<BranchInst>(TI)) {
      // We could either have a proper unconditional branch,
      // or a degenerate conditional branch with matching destinations.
      if (all_of(BI->successors(),
                 [BB](auto *Successor) { return Successor == BB; })) {
        new UnreachableInst(TI->getContext(), TI->getIterator());
        TI->eraseFromParent();
        Changed = true;
      } else {
        assert(BI->isConditional() && "Can't get here with an uncond branch.");
        Value* Cond = BI->getCondition();
        assert(BI->getSuccessor(0) != BI->getSuccessor(1) &&
               "The destinations are guaranteed to be different here.");
        CallInst *Assumption;
        if (BI->getSuccessor(0) == BB) {
          Assumption = Builder.CreateAssumption(Builder.CreateNot(Cond));
          Builder.CreateBr(BI->getSuccessor(1));
        } else {
          assert(BI->getSuccessor(1) == BB && "Incorrect CFG");
          Assumption = Builder.CreateAssumption(Cond);
          Builder.CreateBr(BI->getSuccessor(0));
        }
        if (Options.AC)
          Options.AC->registerAssumption(cast<AssumeInst>(Assumption));

        eraseTerminatorAndDCECond(BI);
        Changed = true;
      }
      if (DTU)
        Updates.push_back({DominatorTree::Delete, Predecessor, BB});
    } else if (auto *SI = dyn_cast<SwitchInst>(TI)) {
      SwitchInstProfUpdateWrapper SU(*SI);
      for (auto i = SU->case_begin(), e = SU->case_end(); i != e;) {
        if (i->getCaseSuccessor() != BB) {
          ++i;
          continue;
        }
        BB->removePredecessor(SU->getParent());
        i = SU.removeCase(i);
        e = SU->case_end();
        Changed = true;
      }
      // Note that the default destination can't be removed!
      if (DTU && SI->getDefaultDest() != BB)
        Updates.push_back({DominatorTree::Delete, Predecessor, BB});
    } else if (auto *II = dyn_cast<InvokeInst>(TI)) {
      if (II->getUnwindDest() == BB) {
        if (DTU) {
          DTU->applyUpdates(Updates);
          Updates.clear();
        }
        auto *CI = cast<CallInst>(removeUnwindEdge(TI->getParent(), DTU));
        if (!CI->doesNotThrow())
          CI->setDoesNotThrow();
        Changed = true;
      }
    } else if (auto *CSI = dyn_cast<CatchSwitchInst>(TI)) {
      if (CSI->getUnwindDest() == BB) {
        if (DTU) {
          DTU->applyUpdates(Updates);
          Updates.clear();
        }
        removeUnwindEdge(TI->getParent(), DTU);
        Changed = true;
        continue;
      }

      for (CatchSwitchInst::handler_iterator I = CSI->handler_begin(),
                                             E = CSI->handler_end();
           I != E; ++I) {
        if (*I == BB) {
          CSI->removeHandler(I);
          --I;
          --E;
          Changed = true;
        }
      }
      if (DTU)
        Updates.push_back({DominatorTree::Delete, Predecessor, BB});
      if (CSI->getNumHandlers() == 0) {
        if (CSI->hasUnwindDest()) {
          // Redirect all predecessors of the block containing CatchSwitchInst
          // to instead branch to the CatchSwitchInst's unwind destination.
          if (DTU) {
            for (auto *PredecessorOfPredecessor : predecessors(Predecessor)) {
              Updates.push_back({DominatorTree::Insert,
                                 PredecessorOfPredecessor,
                                 CSI->getUnwindDest()});
              Updates.push_back({DominatorTree::Delete,
                                 PredecessorOfPredecessor, Predecessor});
            }
          }
          Predecessor->replaceAllUsesWith(CSI->getUnwindDest());
        } else {
          // Rewrite all preds to unwind to caller (or from invoke to call).
          if (DTU) {
            DTU->applyUpdates(Updates);
            Updates.clear();
          }
          SmallVector<BasicBlock *, 8> EHPreds(predecessors(Predecessor));
          for (BasicBlock *EHPred : EHPreds)
            removeUnwindEdge(EHPred, DTU);
        }
        // The catchswitch is no longer reachable.
        new UnreachableInst(CSI->getContext(), CSI->getIterator());
        CSI->eraseFromParent();
        Changed = true;
      }
    } else if (auto *CRI = dyn_cast<CleanupReturnInst>(TI)) {
      (void)CRI;
      assert(CRI->hasUnwindDest() && CRI->getUnwindDest() == BB &&
             "Expected to always have an unwind to BB.");
      if (DTU)
        Updates.push_back({DominatorTree::Delete, Predecessor, BB});
      new UnreachableInst(TI->getContext(), TI->getIterator());
      TI->eraseFromParent();
      Changed = true;
    }
  }

  if (DTU)
    DTU->applyUpdates(Updates);

  // If this block is now dead, remove it.
  if (pred_empty(BB) && BB != &BB->getParent()->getEntryBlock()) {
    DeleteDeadBlock(BB, DTU);
    return true;
  }

  return Changed;
}

static bool casesAreContiguous(SmallVectorImpl<ConstantInt *> &Cases) {
  assert(Cases.size() >= 1);

  array_pod_sort(Cases.begin(), Cases.end(), constantIntSortPredicate);
  for (size_t I = 1, E = Cases.size(); I != E; ++I) {
    if (Cases[I - 1]->getValue() != Cases[I]->getValue() + 1)
      return false;
  }
  return true;
}

static void createUnreachableSwitchDefault(SwitchInst *Switch,
                                           DomTreeUpdater *DTU,
                                           bool RemoveOrigDefaultBlock = true) {
  LLVM_DEBUG(dbgs() << "SimplifyCFG: switch default is dead.\n");
  auto *BB = Switch->getParent();
  auto *OrigDefaultBlock = Switch->getDefaultDest();
  if (RemoveOrigDefaultBlock)
    OrigDefaultBlock->removePredecessor(BB);
  BasicBlock *NewDefaultBlock = BasicBlock::Create(
      BB->getContext(), BB->getName() + ".unreachabledefault", BB->getParent(),
      OrigDefaultBlock);
  auto *UI = new UnreachableInst(Switch->getContext(), NewDefaultBlock);
  UI->setDebugLoc(DebugLoc::getTemporary());
  Switch->setDefaultDest(&*NewDefaultBlock);
  if (DTU) {
    SmallVector<DominatorTree::UpdateType, 2> Updates;
    Updates.push_back({DominatorTree::Insert, BB, &*NewDefaultBlock});
    if (RemoveOrigDefaultBlock &&
        !is_contained(successors(BB), OrigDefaultBlock))
      Updates.push_back({DominatorTree::Delete, BB, &*OrigDefaultBlock});
    DTU->applyUpdates(Updates);
  }
}

/// Turn a switch into an integer range comparison and branch.
/// Switches with more than 2 destinations are ignored.
/// Switches with 1 destination are also ignored.
bool SimplifyCFGOpt::turnSwitchRangeIntoICmp(SwitchInst *SI,
                                             IRBuilder<> &Builder) {
  assert(SI->getNumCases() > 1 && "Degenerate switch?");

  bool HasDefault = !SI->defaultDestUnreachable();

  auto *BB = SI->getParent();

  // Partition the cases into two sets with different destinations.
  BasicBlock *DestA = HasDefault ? SI->getDefaultDest() : nullptr;
  BasicBlock *DestB = nullptr;
  SmallVector<ConstantInt *, 16> CasesA;
  SmallVector<ConstantInt *, 16> CasesB;

  for (auto Case : SI->cases()) {
    BasicBlock *Dest = Case.getCaseSuccessor();
    if (!DestA)
      DestA = Dest;
    if (Dest == DestA) {
      CasesA.push_back(Case.getCaseValue());
      continue;
    }
    if (!DestB)
      DestB = Dest;
    if (Dest == DestB) {
      CasesB.push_back(Case.getCaseValue());
      continue;
    }
    return false; // More than two destinations.
  }
  if (!DestB)
    return false; // All destinations are the same and the default is unreachable

  assert(DestA && DestB &&
         "Single-destination switch should have been folded.");
  assert(DestA != DestB);
  assert(DestB != SI->getDefaultDest());
  assert(!CasesB.empty() && "There must be non-default cases.");
  assert(!CasesA.empty() || HasDefault);

  // Figure out if one of the sets of cases form a contiguous range.
  SmallVectorImpl<ConstantInt *> *ContiguousCases = nullptr;
  BasicBlock *ContiguousDest = nullptr;
  BasicBlock *OtherDest = nullptr;
  if (!CasesA.empty() && casesAreContiguous(CasesA)) {
    ContiguousCases = &CasesA;
    ContiguousDest = DestA;
    OtherDest = DestB;
  } else if (casesAreContiguous(CasesB)) {
    ContiguousCases = &CasesB;
    ContiguousDest = DestB;
    OtherDest = DestA;
  } else
    return false;

  // Start building the compare and branch.

  Constant *Offset = ConstantExpr::getNeg(ContiguousCases->back());
  Constant *NumCases =
      ConstantInt::get(Offset->getType(), ContiguousCases->size());

  Value *Sub = SI->getCondition();
  if (!Offset->isNullValue())
    Sub = Builder.CreateAdd(Sub, Offset, Sub->getName() + ".off");

  Value *Cmp;
  // If NumCases overflowed, then all possible values jump to the successor.
  if (NumCases->isNullValue() && !ContiguousCases->empty())
    Cmp = ConstantInt::getTrue(SI->getContext());
  else
    Cmp = Builder.CreateICmpULT(Sub, NumCases, "switch");
  BranchInst *NewBI = Builder.CreateCondBr(Cmp, ContiguousDest, OtherDest);

  // Update weight for the newly-created conditional branch.
  if (hasBranchWeightMD(*SI)) {
    SmallVector<uint64_t, 8> Weights;
    getBranchWeights(SI, Weights);
    if (Weights.size() == 1 + SI->getNumCases()) {
      uint64_t TrueWeight = 0;
      uint64_t FalseWeight = 0;
      for (size_t I = 0, E = Weights.size(); I != E; ++I) {
        if (SI->getSuccessor(I) == ContiguousDest)
          TrueWeight += Weights[I];
        else
          FalseWeight += Weights[I];
      }
      while (TrueWeight > UINT32_MAX || FalseWeight > UINT32_MAX) {
        TrueWeight /= 2;
        FalseWeight /= 2;
      }
      setBranchWeights(NewBI, TrueWeight, FalseWeight, /*IsExpected=*/false);
    }
  }

  // Prune obsolete incoming values off the successors' PHI nodes.
  for (auto BBI = ContiguousDest->begin(); isa<PHINode>(BBI); ++BBI) {
    unsigned PreviousEdges = ContiguousCases->size();
    if (ContiguousDest == SI->getDefaultDest())
      ++PreviousEdges;
    for (unsigned I = 0, E = PreviousEdges - 1; I != E; ++I)
      cast<PHINode>(BBI)->removeIncomingValue(SI->getParent());
  }
  for (auto BBI = OtherDest->begin(); isa<PHINode>(BBI); ++BBI) {
    unsigned PreviousEdges = SI->getNumCases() - ContiguousCases->size();
    if (OtherDest == SI->getDefaultDest())
      ++PreviousEdges;
    for (unsigned I = 0, E = PreviousEdges - 1; I != E; ++I)
      cast<PHINode>(BBI)->removeIncomingValue(SI->getParent());
  }

  // Clean up the default block - it may have phis or other instructions before
  // the unreachable terminator.
  if (!HasDefault)
    createUnreachableSwitchDefault(SI, DTU);

  auto *UnreachableDefault = SI->getDefaultDest();

  // Drop the switch.
  SI->eraseFromParent();

  if (!HasDefault && DTU)
    DTU->applyUpdates({{DominatorTree::Delete, BB, UnreachableDefault}});

  return true;
}

/// Compute masked bits for the condition of a switch
/// and use it to remove dead cases.
static bool eliminateDeadSwitchCases(SwitchInst *SI, DomTreeUpdater *DTU,
                                     AssumptionCache *AC,
                                     const DataLayout &DL) {
  Value *Cond = SI->getCondition();
  KnownBits Known = computeKnownBits(Cond, DL, AC, SI);

  // We can also eliminate cases by determining that their values are outside of
  // the limited range of the condition based on how many significant (non-sign)
  // bits are in the condition value.
  unsigned MaxSignificantBitsInCond =
      ComputeMaxSignificantBits(Cond, DL, AC, SI);

  // Gather dead cases.
  SmallVector<ConstantInt *, 8> DeadCases;
  SmallDenseMap<BasicBlock *, int, 8> NumPerSuccessorCases;
  SmallVector<BasicBlock *, 8> UniqueSuccessors;
  for (const auto &Case : SI->cases()) {
    auto *Successor = Case.getCaseSuccessor();
    if (DTU) {
      auto [It, Inserted] = NumPerSuccessorCases.try_emplace(Successor);
      if (Inserted)
        UniqueSuccessors.push_back(Successor);
      ++It->second;
    }
    const APInt &CaseVal = Case.getCaseValue()->getValue();
    if (Known.Zero.intersects(CaseVal) || !Known.One.isSubsetOf(CaseVal) ||
        (CaseVal.getSignificantBits() > MaxSignificantBitsInCond)) {
      DeadCases.push_back(Case.getCaseValue());
      if (DTU)
        --NumPerSuccessorCases[Successor];
      LLVM_DEBUG(dbgs() << "SimplifyCFG: switch case " << CaseVal
                        << " is dead.\n");
    }
  }

  // If we can prove that the cases must cover all possible values, the
  // default destination becomes dead and we can remove it.  If we know some
  // of the bits in the value, we can use that to more precisely compute the
  // number of possible unique case values.
  bool HasDefault = !SI->defaultDestUnreachable();
  const unsigned NumUnknownBits =
      Known.getBitWidth() - (Known.Zero | Known.One).popcount();
  assert(NumUnknownBits <= Known.getBitWidth());
  if (HasDefault && DeadCases.empty() &&
      NumUnknownBits < 64 /* avoid overflow */) {
    uint64_t AllNumCases = 1ULL << NumUnknownBits;
    if (SI->getNumCases() == AllNumCases) {
      createUnreachableSwitchDefault(SI, DTU);
      return true;
    }
    // When only one case value is missing, replace default with that case.
    // Eliminating the default branch will provide more opportunities for
    // optimization, such as lookup tables.
    if (SI->getNumCases() == AllNumCases - 1) {
      assert(NumUnknownBits > 1 && "Should be canonicalized to a branch");
      IntegerType *CondTy = cast<IntegerType>(Cond->getType());
      if (CondTy->getIntegerBitWidth() > 64 ||
          !DL.fitsInLegalInteger(CondTy->getIntegerBitWidth()))
        return false;

      uint64_t MissingCaseVal = 0;
      for (const auto &Case : SI->cases())
        MissingCaseVal ^= Case.getCaseValue()->getValue().getLimitedValue();
      auto *MissingCase =
          cast<ConstantInt>(ConstantInt::get(Cond->getType(), MissingCaseVal));
      SwitchInstProfUpdateWrapper SIW(*SI);
      SIW.addCase(MissingCase, SI->getDefaultDest(), SIW.getSuccessorWeight(0));
      createUnreachableSwitchDefault(SI, DTU, /*RemoveOrigDefaultBlock*/ false);
      SIW.setSuccessorWeight(0, 0);
      return true;
    }
  }

  if (DeadCases.empty())
    return false;

  SwitchInstProfUpdateWrapper SIW(*SI);
  for (ConstantInt *DeadCase : DeadCases) {
    SwitchInst::CaseIt CaseI = SI->findCaseValue(DeadCase);
    assert(CaseI != SI->case_default() &&
           "Case was not found. Probably mistake in DeadCases forming.");
    // Prune unused values from PHI nodes.
    CaseI->getCaseSuccessor()->removePredecessor(SI->getParent());
    SIW.removeCase(CaseI);
  }

  if (DTU) {
    std::vector<DominatorTree::UpdateType> Updates;
    for (auto *Successor : UniqueSuccessors)
      if (NumPerSuccessorCases[Successor] == 0)
        Updates.push_back({DominatorTree::Delete, SI->getParent(), Successor});
    DTU->applyUpdates(Updates);
  }

  return true;
}

/// If BB would be eligible for simplification by
/// TryToSimplifyUncondBranchFromEmptyBlock (i.e. it is empty and terminated
/// by an unconditional branch), look at the phi node for BB in the successor
/// block and see if the incoming value is equal to CaseValue. If so, return
/// the phi node, and set PhiIndex to BB's index in the phi node.
static PHINode *findPHIForConditionForwarding(ConstantInt *CaseValue,
                                              BasicBlock *BB, int *PhiIndex) {
  if (&*BB->getFirstNonPHIIt() != BB->getTerminator())
    return nullptr; // BB must be empty to be a candidate for simplification.
  if (!BB->getSinglePredecessor())
    return nullptr; // BB must be dominated by the switch.

  BranchInst *Branch = dyn_cast<BranchInst>(BB->getTerminator());
  if (!Branch || !Branch->isUnconditional())
    return nullptr; // Terminator must be unconditional branch.

  BasicBlock *Succ = Branch->getSuccessor(0);

  for (PHINode &PHI : Succ->phis()) {
    int Idx = PHI.getBasicBlockIndex(BB);
    assert(Idx >= 0 && "PHI has no entry for predecessor?");

    Value *InValue = PHI.getIncomingValue(Idx);
    if (InValue != CaseValue)
      continue;

    *PhiIndex = Idx;
    return &PHI;
  }

  return nullptr;
}

/// Try to forward the condition of a switch instruction to a phi node
/// dominated by the switch, if that would mean that some of the destination
/// blocks of the switch can be folded away. Return true if a change is made.
static bool forwardSwitchConditionToPHI(SwitchInst *SI) {
  using ForwardingNodesMap = DenseMap<PHINode *, SmallVector<int, 4>>;

  ForwardingNodesMap ForwardingNodes;
  BasicBlock *SwitchBlock = SI->getParent();
  bool Changed = false;
  for (const auto &Case : SI->cases()) {
    ConstantInt *CaseValue = Case.getCaseValue();
    BasicBlock *CaseDest = Case.getCaseSuccessor();

    // Replace phi operands in successor blocks that are using the constant case
    // value rather than the switch condition variable:
    //   switchbb:
    //   switch i32 %x, label %default [
    //     i32 17, label %succ
    //   ...
    //   succ:
    //     %r = phi i32 ... [ 17, %switchbb ] ...
    // -->
    //     %r = phi i32 ... [ %x, %switchbb ] ...

    for (PHINode &Phi : CaseDest->phis()) {
      // This only works if there is exactly 1 incoming edge from the switch to
      // a phi. If there is >1, that means multiple cases of the switch map to 1
      // value in the phi, and that phi value is not the switch condition. Thus,
      // this transform would not make sense (the phi would be invalid because
      // a phi can't have different incoming values from the same block).
      int SwitchBBIdx = Phi.getBasicBlockIndex(SwitchBlock);
      if (Phi.getIncomingValue(SwitchBBIdx) == CaseValue &&
          count(Phi.blocks(), SwitchBlock) == 1) {
        Phi.setIncomingValue(SwitchBBIdx, SI->getCondition());
        Changed = true;
      }
    }

    // Collect phi nodes that are indirectly using this switch's case constants.
    int PhiIdx;
    if (auto *Phi = findPHIForConditionForwarding(CaseValue, CaseDest, &PhiIdx))
      ForwardingNodes[Phi].push_back(PhiIdx);
  }

  for (auto &ForwardingNode : ForwardingNodes) {
    PHINode *Phi = ForwardingNode.first;
    SmallVectorImpl<int> &Indexes = ForwardingNode.second;
    // Check if it helps to fold PHI.
    if (Indexes.size() < 2 && !llvm::is_contained(Phi->incoming_values(), SI->getCondition()))
      continue;

    for (int Index : Indexes)
      Phi->setIncomingValue(Index, SI->getCondition());
    Changed = true;
  }

  return Changed;
}

/// Return true if the backend will be able to handle
/// initializing an array of constants like C.
static bool validLookupTableConstant(Constant *C, const TargetTransformInfo &TTI) {
  if (C->isThreadDependent())
    return false;
  if (C->isDLLImportDependent())
    return false;

  if (!isa<ConstantFP>(C) && !isa<ConstantInt>(C) &&
      !isa<ConstantPointerNull>(C) && !isa<GlobalValue>(C) &&
      !isa<UndefValue>(C) && !isa<ConstantExpr>(C))
    return false;

  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
    // Pointer casts and in-bounds GEPs will not prohibit the backend from
    // materializing the array of constants.
    Constant *StrippedC = cast<Constant>(CE->stripInBoundsConstantOffsets());
    if (StrippedC == C || !validLookupTableConstant(StrippedC, TTI))
      return false;
  }

  if (!TTI.shouldBuildLookupTablesForConstant(C))
    return false;

  return true;
}

/// If V is a Constant, return it. Otherwise, try to look up
/// its constant value in ConstantPool, returning 0 if it's not there.
static Constant *
lookupConstant(Value *V,
               const SmallDenseMap<Value *, Constant *> &ConstantPool) {
  if (Constant *C = dyn_cast<Constant>(V))
    return C;
  return ConstantPool.lookup(V);
}

/// Try to fold instruction I into a constant. This works for
/// simple instructions such as binary operations where both operands are
/// constant or can be replaced by constants from the ConstantPool. Returns the
/// resulting constant on success, 0 otherwise.
static Constant *
constantFold(Instruction *I, const DataLayout &DL,
             const SmallDenseMap<Value *, Constant *> &ConstantPool) {
  if (SelectInst *Select = dyn_cast<SelectInst>(I)) {
    Constant *A = lookupConstant(Select->getCondition(), ConstantPool);
    if (!A)
      return nullptr;
    if (A->isAllOnesValue())
      return lookupConstant(Select->getTrueValue(), ConstantPool);
    if (A->isNullValue())
      return lookupConstant(Select->getFalseValue(), ConstantPool);
    return nullptr;
  }

  SmallVector<Constant *, 4> COps;
  for (unsigned N = 0, E = I->getNumOperands(); N != E; ++N) {
    if (Constant *A = lookupConstant(I->getOperand(N), ConstantPool))
      COps.push_back(A);
    else
      return nullptr;
  }

  return ConstantFoldInstOperands(I, COps, DL);
}

/// Try to determine the resulting constant values in phi nodes
/// at the common destination basic block, *CommonDest, for one of the case
/// destionations CaseDest corresponding to value CaseVal (0 for the default
/// case), of a switch instruction SI.
static bool
getCaseResults(SwitchInst *SI, ConstantInt *CaseVal, BasicBlock *CaseDest,
               BasicBlock **CommonDest,
               SmallVectorImpl<std::pair<PHINode *, Constant *>> &Res,
               const DataLayout &DL, const TargetTransformInfo &TTI) {
  // The block from which we enter the common destination.
  BasicBlock *Pred = SI->getParent();

  // If CaseDest is empty except for some side-effect free instructions through
  // which we can constant-propagate the CaseVal, continue to its successor.
  SmallDenseMap<Value *, Constant *> ConstantPool;
  ConstantPool.insert(std::make_pair(SI->getCondition(), CaseVal));
  for (Instruction &I : CaseDest->instructionsWithoutDebug(false)) {
    if (I.isTerminator()) {
      // If the terminator is a simple branch, continue to the next block.
      if (I.getNumSuccessors() != 1 || I.isSpecialTerminator())
        return false;
      Pred = CaseDest;
      CaseDest = I.getSuccessor(0);
    } else if (Constant *C = constantFold(&I, DL, ConstantPool)) {
      // Instruction is side-effect free and constant.

      // If the instruction has uses outside this block or a phi node slot for
      // the block, it is not safe to bypass the instruction since it would then
      // no longer dominate all its uses.
      for (auto &Use : I.uses()) {
        User *User = Use.getUser();
        if (Instruction *I = dyn_cast<Instruction>(User))
          if (I->getParent() == CaseDest)
            continue;
        if (PHINode *Phi = dyn_cast<PHINode>(User))
          if (Phi->getIncomingBlock(Use) == CaseDest)
            continue;
        return false;
      }

      ConstantPool.insert(std::make_pair(&I, C));
    } else {
      break;
    }
  }

  // If we did not have a CommonDest before, use the current one.
  if (!*CommonDest)
    *CommonDest = CaseDest;
  // If the destination isn't the common one, abort.
  if (CaseDest != *CommonDest)
    return false;

  // Get the values for this case from phi nodes in the destination block.
  for (PHINode &PHI : (*CommonDest)->phis()) {
    int Idx = PHI.getBasicBlockIndex(Pred);
    if (Idx == -1)
      continue;

    Constant *ConstVal =
        lookupConstant(PHI.getIncomingValue(Idx), ConstantPool);
    if (!ConstVal)
      return false;

    // Be conservative about which kinds of constants we support.
    if (!validLookupTableConstant(ConstVal, TTI))
      return false;

    Res.push_back(std::make_pair(&PHI, ConstVal));
  }

  return Res.size() > 0;
}

// Helper function used to add CaseVal to the list of cases that generate
// Result. Returns the updated number of cases that generate this result.
static size_t mapCaseToResult(ConstantInt *CaseVal,
                              SwitchCaseResultVectorTy &UniqueResults,
                              Constant *Result) {
  for (auto &I : UniqueResults) {
    if (I.first == Result) {
      I.second.push_back(CaseVal);
      return I.second.size();
    }
  }
  UniqueResults.push_back(
      std::make_pair(Result, SmallVector<ConstantInt *, 4>(1, CaseVal)));
  return 1;
}

// Helper function that initializes a map containing
// results for the PHI node of the common destination block for a switch
// instruction. Returns false if multiple PHI nodes have been found or if
// there is not a common destination block for the switch.
static bool initializeUniqueCases(SwitchInst *SI, PHINode *&PHI,
                                  BasicBlock *&CommonDest,
                                  SwitchCaseResultVectorTy &UniqueResults,
                                  Constant *&DefaultResult,
                                  const DataLayout &DL,
                                  const TargetTransformInfo &TTI,
                                  uintptr_t MaxUniqueResults) {
  for (const auto &I : SI->cases()) {
    ConstantInt *CaseVal = I.getCaseValue();

    // Resulting value at phi nodes for this case value.
    SwitchCaseResultsTy Results;
    if (!getCaseResults(SI, CaseVal, I.getCaseSuccessor(), &CommonDest, Results,
                        DL, TTI))
      return false;

    // Only one value per case is permitted.
    if (Results.size() > 1)
      return false;

    // Add the case->result mapping to UniqueResults.
    const size_t NumCasesForResult =
        mapCaseToResult(CaseVal, UniqueResults, Results.begin()->second);

    // Early out if there are too many cases for this result.
    if (NumCasesForResult > MaxSwitchCasesPerResult)
      return false;

    // Early out if there are too many unique results.
    if (UniqueResults.size() > MaxUniqueResults)
      return false;

    // Check the PHI consistency.
    if (!PHI)
      PHI = Results[0].first;
    else if (PHI != Results[0].first)
      return false;
  }
  // Find the default result value.
  SmallVector<std::pair<PHINode *, Constant *>, 1> DefaultResults;
  getCaseResults(SI, nullptr, SI->getDefaultDest(), &CommonDest, DefaultResults,
                 DL, TTI);
  // If the default value is not found abort unless the default destination
  // is unreachable.
  DefaultResult =
      DefaultResults.size() == 1 ? DefaultResults.begin()->second : nullptr;

  return DefaultResult || SI->defaultDestUnreachable();
}

// Helper function that checks if it is possible to transform a switch with only
// two cases (or two cases + default) that produces a result into a select.
// TODO: Handle switches with more than 2 cases that map to the same result.
static Value *foldSwitchToSelect(const SwitchCaseResultVectorTy &ResultVector,
                                 Constant *DefaultResult, Value *Condition,
                                 IRBuilder<> &Builder, const DataLayout &DL) {
  // If we are selecting between only two cases transform into a simple
  // select or a two-way select if default is possible.
  // Example:
  // switch (a) {                  %0 = icmp eq i32 %a, 10
  //   case 10: return 42;         %1 = select i1 %0, i32 42, i32 4
  //   case 20: return 2;   ---->  %2 = icmp eq i32 %a, 20
  //   default: return 4;          %3 = select i1 %2, i32 2, i32 %1
  // }
  if (ResultVector.size() == 2 && ResultVector[0].second.size() == 1 &&
      ResultVector[1].second.size() == 1) {
    ConstantInt *FirstCase = ResultVector[0].second[0];
    ConstantInt *SecondCase = ResultVector[1].second[0];
    Value *SelectValue = ResultVector[1].first;
    if (DefaultResult) {
      Value *ValueCompare =
          Builder.CreateICmpEQ(Condition, SecondCase, "switch.selectcmp");
      SelectValue = Builder.CreateSelect(ValueCompare, ResultVector[1].first,
                                         DefaultResult, "switch.select");
    }
    Value *ValueCompare =
        Builder.CreateICmpEQ(Condition, FirstCase, "switch.selectcmp");
    return Builder.CreateSelect(ValueCompare, ResultVector[0].first,
                                SelectValue, "switch.select");
  }

  // Handle the degenerate case where two cases have the same result value.
  if (ResultVector.size() == 1 && DefaultResult) {
    ArrayRef<ConstantInt *> CaseValues = ResultVector[0].second;
    unsigned CaseCount = CaseValues.size();
    // n bits group cases map to the same result:
    // case 0,4      -> Cond & 0b1..1011 == 0 ? result : default
    // case 0,2,4,6  -> Cond & 0b1..1001 == 0 ? result : default
    // case 0,2,8,10 -> Cond & 0b1..0101 == 0 ? result : default
    if (isPowerOf2_32(CaseCount)) {
      ConstantInt *MinCaseVal = CaseValues[0];
      // If there are bits that are set exclusively by CaseValues, we
      // can transform the switch into a select if the conjunction of
      // all the values uniquely identify CaseValues.
      APInt AndMask = APInt::getAllOnes(MinCaseVal->getBitWidth());

      // Find the minimum value and compute the and of all the case values.
      for (auto *Case : CaseValues) {
        if (Case->getValue().slt(MinCaseVal->getValue()))
          MinCaseVal = Case;
        AndMask &= Case->getValue();
      }
      KnownBits Known = computeKnownBits(Condition, DL);

      if (!AndMask.isZero() && Known.getMaxValue().uge(AndMask)) {
        // Compute the number of bits that are free to vary.
        unsigned FreeBits = Known.countMaxActiveBits() - AndMask.popcount();

        // Check if the number of values covered by the mask is equal
        // to the number of cases.
        if (FreeBits == Log2_32(CaseCount)) {
          Value *And = Builder.CreateAnd(Condition, AndMask);
          Value *Cmp = Builder.CreateICmpEQ(
              And, Constant::getIntegerValue(And->getType(), AndMask));
          return Builder.CreateSelect(Cmp, ResultVector[0].first,
                                      DefaultResult);
        }
      }

      // Mark the bits case number touched.
      APInt BitMask = APInt::getZero(MinCaseVal->getBitWidth());
      for (auto *Case : CaseValues)
        BitMask |= (Case->getValue() - MinCaseVal->getValue());

      // Check if cases with the same result can cover all number
      // in touched bits.
      if (BitMask.popcount() == Log2_32(CaseCount)) {
        if (!MinCaseVal->isNullValue())
          Condition = Builder.CreateSub(Condition, MinCaseVal);
        Value *And = Builder.CreateAnd(Condition, ~BitMask, "switch.and");
        Value *Cmp = Builder.CreateICmpEQ(
            And, Constant::getNullValue(And->getType()), "switch.selectcmp");
        return Builder.CreateSelect(Cmp, ResultVector[0].first, DefaultResult);
      }
    }

    // Handle the degenerate case where two cases have the same value.
    if (CaseValues.size() == 2) {
      Value *Cmp1 = Builder.CreateICmpEQ(Condition, CaseValues[0],
                                         "switch.selectcmp.case1");
      Value *Cmp2 = Builder.CreateICmpEQ(Condition, CaseValues[1],
                                         "switch.selectcmp.case2");
      Value *Cmp = Builder.CreateOr(Cmp1, Cmp2, "switch.selectcmp");
      return Builder.CreateSelect(Cmp, ResultVector[0].first, DefaultResult);
    }
  }

  return nullptr;
}

// Helper function to cleanup a switch instruction that has been converted into
// a select, fixing up PHI nodes and basic blocks.
static void removeSwitchAfterSelectFold(SwitchInst *SI, PHINode *PHI,
                                        Value *SelectValue,
                                        IRBuilder<> &Builder,
                                        DomTreeUpdater *DTU) {
  std::vector<DominatorTree::UpdateType> Updates;

  BasicBlock *SelectBB = SI->getParent();
  BasicBlock *DestBB = PHI->getParent();

  if (DTU && !is_contained(predecessors(DestBB), SelectBB))
    Updates.push_back({DominatorTree::Insert, SelectBB, DestBB});
  Builder.CreateBr(DestBB);

  // Remove the switch.

  PHI->removeIncomingValueIf(
      [&](unsigned Idx) { return PHI->getIncomingBlock(Idx) == SelectBB; });
  PHI->addIncoming(SelectValue, SelectBB);

  SmallPtrSet<BasicBlock *, 4> RemovedSuccessors;
  for (unsigned i = 0, e = SI->getNumSuccessors(); i < e; ++i) {
    BasicBlock *Succ = SI->getSuccessor(i);

    if (Succ == DestBB)
      continue;
    Succ->removePredecessor(SelectBB);
    if (DTU && RemovedSuccessors.insert(Succ).second)
      Updates.push_back({DominatorTree::Delete, SelectBB, Succ});
  }
  SI->eraseFromParent();
  if (DTU)
    DTU->applyUpdates(Updates);
}

/// If a switch is only used to initialize one or more phi nodes in a common
/// successor block with only two different constant values, try to replace the
/// switch with a select. Returns true if the fold was made.
static bool trySwitchToSelect(SwitchInst *SI, IRBuilder<> &Builder,
                              DomTreeUpdater *DTU, const DataLayout &DL,
                              const TargetTransformInfo &TTI) {
  Value *const Cond = SI->getCondition();
  PHINode *PHI = nullptr;
  BasicBlock *CommonDest = nullptr;
  Constant *DefaultResult;
  SwitchCaseResultVectorTy UniqueResults;
  // Collect all the cases that will deliver the same value from the switch.
  if (!initializeUniqueCases(SI, PHI, CommonDest, UniqueResults, DefaultResult,
                             DL, TTI, /*MaxUniqueResults*/ 2))
    return false;

  assert(PHI != nullptr && "PHI for value select not found");
  Builder.SetInsertPoint(SI);
  Value *SelectValue =
      foldSwitchToSelect(UniqueResults, DefaultResult, Cond, Builder, DL);
  if (!SelectValue)
    return false;

  removeSwitchAfterSelectFold(SI, PHI, SelectValue, Builder, DTU);
  return true;
}

namespace {

/// This class finds alternatives for switches to ultimately
/// replace the switch.
class SwitchReplacement {
public:
  /// Create a helper for optimizations to use as a switch replacement.
  /// Find a better representation for the content of Values,
  /// using DefaultValue to fill any holes in the table.
  SwitchReplacement(
      Module &M, uint64_t TableSize, ConstantInt *Offset,
      const SmallVectorImpl<std::pair<ConstantInt *, Constant *>> &Values,
      Constant *DefaultValue, const DataLayout &DL, const StringRef &FuncName);

  /// Build instructions with Builder to retrieve values using Index
  /// and replace the switch.
  Value *replaceSwitch(Value *Index, IRBuilder<> &Builder, const DataLayout &DL,
                       Function *Func);

  /// Return true if a table with TableSize elements of
  /// type ElementType would fit in a target-legal register.
  static bool wouldFitInRegister(const DataLayout &DL, uint64_t TableSize,
                                 Type *ElementType);

  /// Return the default value of the switch.
  Constant *getDefaultValue();

private:
  // Depending on the switch, there are different alternatives.
  enum {
    // For switches where each case contains the same value, we just have to
    // store that single value and return it for each lookup.
    SingleValueKind,

    // For switches where there is a linear relationship between table index
    // and values. We calculate the result with a simple multiplication
    // and addition instead of a table lookup.
    LinearMapKind,

    // For small tables with integer elements, we can pack them into a bitmap
    // that fits into a target-legal register. Values are retrieved by
    // shift and mask operations.
    BitMapKind,

    // The table is stored as an array of values. Values are retrieved by load
    // instructions from the table.
    LookupTableKind
  } Kind;

  // The default value of the switch.
  Constant *DefaultValue;

  // The type of the output values.
  Type *ValueType;

  // For SingleValueKind, this is the single value.
  Constant *SingleValue = nullptr;

  // For BitMapKind, this is the bitmap.
  ConstantInt *BitMap = nullptr;
  IntegerType *BitMapElementTy = nullptr;

  // For LinearMapKind, these are the constants used to derive the value.
  ConstantInt *LinearOffset = nullptr;
  ConstantInt *LinearMultiplier = nullptr;
  bool LinearMapValWrapped = false;

  // For LookupTableKind, this is the table.
  Constant *Initializer = nullptr;
};

} // end anonymous namespace

SwitchReplacement::SwitchReplacement(
    Module &M, uint64_t TableSize, ConstantInt *Offset,
    const SmallVectorImpl<std::pair<ConstantInt *, Constant *>> &Values,
    Constant *DefaultValue, const DataLayout &DL, const StringRef &FuncName)
    : DefaultValue(DefaultValue) {
  assert(Values.size() && "Can't build lookup table without values!");
  assert(TableSize >= Values.size() && "Can't fit values in table!");

  // If all values in the table are equal, this is that value.
  SingleValue = Values.begin()->second;

  ValueType = Values.begin()->second->getType();

  // Build up the table contents.
  SmallVector<Constant *, 64> TableContents(TableSize);
  for (const auto &[CaseVal, CaseRes] : Values) {
    assert(CaseRes->getType() == ValueType);

    uint64_t Idx = (CaseVal->getValue() - Offset->getValue()).getLimitedValue();
    TableContents[Idx] = CaseRes;

    if (SingleValue && !isa<PoisonValue>(CaseRes) && CaseRes != SingleValue)
      SingleValue = isa<PoisonValue>(SingleValue) ? CaseRes : nullptr;
  }

  // Fill in any holes in the table with the default result.
  if (Values.size() < TableSize) {
    assert(DefaultValue &&
           "Need a default value to fill the lookup table holes.");
    assert(DefaultValue->getType() == ValueType);
    for (uint64_t I = 0; I < TableSize; ++I) {
      if (!TableContents[I])
        TableContents[I] = DefaultValue;
    }

    // If the default value is poison, all the holes are poison.
    bool DefaultValueIsPoison = isa<PoisonValue>(DefaultValue);

    if (DefaultValue != SingleValue && !DefaultValueIsPoison)
      SingleValue = nullptr;
  }

  // If each element in the table contains the same value, we only need to store
  // that single value.
  if (SingleValue) {
    Kind = SingleValueKind;
    return;
  }

  // Check if we can derive the value with a linear transformation from the
  // table index.
  if (isa<IntegerType>(ValueType)) {
    bool LinearMappingPossible = true;
    APInt PrevVal;
    APInt DistToPrev;
    // When linear map is monotonic and signed overflow doesn't happen on
    // maximum index, we can attach nsw on Add and Mul.
    bool NonMonotonic = false;
    assert(TableSize >= 2 && "Should be a SingleValue table.");
    // Check if there is the same distance between two consecutive values.
    for (uint64_t I = 0; I < TableSize; ++I) {
      ConstantInt *ConstVal = dyn_cast<ConstantInt>(TableContents[I]);

      if (!ConstVal && isa<PoisonValue>(TableContents[I])) {
        // This is an poison, so it's (probably) a lookup table hole.
        // To prevent any regressions from before we switched to using poison as
        // the default value, holes will fall back to using the first value.
        // This can be removed once we add proper handling for poisons in lookup
        // tables.
        ConstVal = dyn_cast<ConstantInt>(Values[0].second);
      }

      if (!ConstVal) {
        // This is an undef. We could deal with it, but undefs in lookup tables
        // are very seldom. It's probably not worth the additional complexity.
        LinearMappingPossible = false;
        break;
      }
      const APInt &Val = ConstVal->getValue();
      if (I != 0) {
        APInt Dist = Val - PrevVal;
        if (I == 1) {
          DistToPrev = Dist;
        } else if (Dist != DistToPrev) {
          LinearMappingPossible = false;
          break;
        }
        NonMonotonic |=
            Dist.isStrictlyPositive() ? Val.sle(PrevVal) : Val.sgt(PrevVal);
      }
      PrevVal = Val;
    }
    if (LinearMappingPossible) {
      LinearOffset = cast<ConstantInt>(TableContents[0]);
      LinearMultiplier = ConstantInt::get(M.getContext(), DistToPrev);
      APInt M = LinearMultiplier->getValue();
      bool MayWrap = true;
      if (isIntN(M.getBitWidth(), TableSize - 1))
        (void)M.smul_ov(APInt(M.getBitWidth(), TableSize - 1), MayWrap);
      LinearMapValWrapped = NonMonotonic || MayWrap;
      Kind = LinearMapKind;
      ++NumLinearMaps;
      return;
    }
  }

  // If the type is integer and the table fits in a register, build a bitmap.
  if (wouldFitInRegister(DL, TableSize, ValueType)) {
    IntegerType *IT = cast<IntegerType>(ValueType);
    APInt TableInt(TableSize * IT->getBitWidth(), 0);
    for (uint64_t I = TableSize; I > 0; --I) {
      TableInt <<= IT->getBitWidth();
      // Insert values into the bitmap. Undef values are set to zero.
      if (!isa<UndefValue>(TableContents[I - 1])) {
        ConstantInt *Val = cast<ConstantInt>(TableContents[I - 1]);
        TableInt |= Val->getValue().zext(TableInt.getBitWidth());
      }
    }
    BitMap = ConstantInt::get(M.getContext(), TableInt);
    BitMapElementTy = IT;
    Kind = BitMapKind;
    ++NumBitMaps;
    return;
  }

  // Store the table in an array.
  auto *TableTy = ArrayType::get(ValueType, TableSize);
  Initializer = ConstantArray::get(TableTy, TableContents);

  Kind = LookupTableKind;
}

Value *SwitchReplacement::replaceSwitch(Value *Index, IRBuilder<> &Builder,
                                        const DataLayout &DL, Function *Func) {
  switch (Kind) {
  case SingleValueKind:
    return SingleValue;
  case LinearMapKind: {
    // Derive the result value from the input value.
    Value *Result = Builder.CreateIntCast(Index, LinearMultiplier->getType(),
                                          false, "switch.idx.cast");
    if (!LinearMultiplier->isOne())
      Result = Builder.CreateMul(Result, LinearMultiplier, "switch.idx.mult",
                                 /*HasNUW = */ false,
                                 /*HasNSW = */ !LinearMapValWrapped);

    if (!LinearOffset->isZero())
      Result = Builder.CreateAdd(Result, LinearOffset, "switch.offset",
                                 /*HasNUW = */ false,
                                 /*HasNSW = */ !LinearMapValWrapped);
    return Result;
  }
  case BitMapKind: {
    // Type of the bitmap (e.g. i59).
    IntegerType *MapTy = BitMap->getIntegerType();

    // Cast Index to the same type as the bitmap.
    // Note: The Index is <= the number of elements in the table, so
    // truncating it to the width of the bitmask is safe.
    Value *ShiftAmt = Builder.CreateZExtOrTrunc(Index, MapTy, "switch.cast");

    // Multiply the shift amount by the element width. NUW/NSW can always be
    // set, because wouldFitInRegister guarantees Index * ShiftAmt is in
    // BitMap's bit width.
    ShiftAmt = Builder.CreateMul(
        ShiftAmt, ConstantInt::get(MapTy, BitMapElementTy->getBitWidth()),
        "switch.shiftamt",/*HasNUW =*/true,/*HasNSW =*/true);

    // Shift down.
    Value *DownShifted =
        Builder.CreateLShr(BitMap, ShiftAmt, "switch.downshift");
    // Mask off.
    return Builder.CreateTrunc(DownShifted, BitMapElementTy, "switch.masked");
  }
  case LookupTableKind: {
    auto *Table =
        new GlobalVariable(*Func->getParent(), Initializer->getType(),
                           /*isConstant=*/true, GlobalVariable::PrivateLinkage,
                           Initializer, "switch.table." + Func->getName());
    Table->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    // Set the alignment to that of an array items. We will be only loading one
    // value out of it.
    Table->setAlignment(DL.getPrefTypeAlign(ValueType));
    Type *IndexTy = DL.getIndexType(Table->getType());
    auto *ArrayTy = cast<ArrayType>(Table->getValueType());

    if (Index->getType() != IndexTy) {
      unsigned OldBitWidth = Index->getType()->getIntegerBitWidth();
      Index = Builder.CreateZExtOrTrunc(Index, IndexTy);
      if (auto *Zext = dyn_cast<ZExtInst>(Index))
        Zext->setNonNeg(
            isUIntN(OldBitWidth - 1, ArrayTy->getNumElements() - 1));
    }

    Value *GEPIndices[] = {ConstantInt::get(IndexTy, 0), Index};
    Value *GEP =
        Builder.CreateInBoundsGEP(ArrayTy, Table, GEPIndices, "switch.gep");
    return Builder.CreateLoad(ArrayTy->getElementType(), GEP, "switch.load");
  }
  }
  llvm_unreachable("Unknown helper kind!");
}

bool SwitchReplacement::wouldFitInRegister(const DataLayout &DL,
                                           uint64_t TableSize,
                                           Type *ElementType) {
  auto *IT = dyn_cast<IntegerType>(ElementType);
  if (!IT)
    return false;
  // FIXME: If the type is wider than it needs to be, e.g. i8 but all values
  // are <= 15, we could try to narrow the type.

  // Avoid overflow, fitsInLegalInteger uses unsigned int for the width.
  if (TableSize >= UINT_MAX / IT->getBitWidth())
    return false;
  return DL.fitsInLegalInteger(TableSize * IT->getBitWidth());
}

static bool isTypeLegalForLookupTable(Type *Ty, const TargetTransformInfo &TTI,
                                      const DataLayout &DL) {
  // Allow any legal type.
  if (TTI.isTypeLegal(Ty))
    return true;

  auto *IT = dyn_cast<IntegerType>(Ty);
  if (!IT)
    return false;

  // Also allow power of 2 integer types that have at least 8 bits and fit in
  // a register. These types are common in frontend languages and targets
  // usually support loads of these types.
  // TODO: We could relax this to any integer that fits in a register and rely
  // on ABI alignment and padding in the table to allow the load to be widened.
  // Or we could widen the constants and truncate the load.
  unsigned BitWidth = IT->getBitWidth();
  return BitWidth >= 8 && isPowerOf2_32(BitWidth) &&
         DL.fitsInLegalInteger(IT->getBitWidth());
}

Constant *SwitchReplacement::getDefaultValue() { return DefaultValue; }

static bool isSwitchDense(uint64_t NumCases, uint64_t CaseRange) {
  // 40% is the default density for building a jump table in optsize/minsize
  // mode. See also TargetLoweringBase::isSuitableForJumpTable(), which this
  // function was based on.
  const uint64_t MinDensity = 40;

  if (CaseRange >= UINT64_MAX / 100)
    return false; // Avoid multiplication overflows below.

  return NumCases * 100 >= CaseRange * MinDensity;
}

static bool isSwitchDense(ArrayRef<int64_t> Values) {
  uint64_t Diff = (uint64_t)Values.back() - (uint64_t)Values.front();
  uint64_t Range = Diff + 1;
  if (Range < Diff)
    return false; // Overflow.

  return isSwitchDense(Values.size(), Range);
}

/// Determine whether a lookup table should be built for this switch, based on
/// the number of cases, size of the table, and the types of the results.
// TODO: We could support larger than legal types by limiting based on the
// number of loads required and/or table size. If the constants are small we
// could use smaller table entries and extend after the load.
static bool shouldBuildLookupTable(SwitchInst *SI, uint64_t TableSize,
                                   const TargetTransformInfo &TTI,
                                   const DataLayout &DL,
                                   const SmallVector<Type *> &ResultTypes) {
  if (SI->getNumCases() > TableSize)
    return false; // TableSize overflowed.

  bool AllTablesFitInRegister = true;
  bool HasIllegalType = false;
  for (const auto &Ty : ResultTypes) {
    // Saturate this flag to true.
    HasIllegalType = HasIllegalType || !isTypeLegalForLookupTable(Ty, TTI, DL);

    // Saturate this flag to false.
    AllTablesFitInRegister =
        AllTablesFitInRegister &&
        SwitchReplacement::wouldFitInRegister(DL, TableSize, Ty);

    // If both flags saturate, we're done. NOTE: This *only* works with
    // saturating flags, and all flags have to saturate first due to the
    // non-deterministic behavior of iterating over a dense map.
    if (HasIllegalType && !AllTablesFitInRegister)
      break;
  }

  // If each table would fit in a register, we should build it anyway.
  if (AllTablesFitInRegister)
    return true;

  // Don't build a table that doesn't fit in-register if it has illegal types.
  if (HasIllegalType)
    return false;

  return isSwitchDense(SI->getNumCases(), TableSize);
}

static bool shouldUseSwitchConditionAsTableIndex(
    ConstantInt &MinCaseVal, const ConstantInt &MaxCaseVal,
    bool HasDefaultResults, const SmallVector<Type *> &ResultTypes,
    const DataLayout &DL, const TargetTransformInfo &TTI) {
  if (MinCaseVal.isNullValue())
    return true;
  if (MinCaseVal.isNegative() ||
      MaxCaseVal.getLimitedValue() == std::numeric_limits<uint64_t>::max() ||
      !HasDefaultResults)
    return false;
  return all_of(ResultTypes, [&](const auto &ResultType) {
    return SwitchReplacement::wouldFitInRegister(
        DL, MaxCaseVal.getLimitedValue() + 1 /* TableSize */, ResultType);
  });
}

/// Try to reuse the switch table index compare. Following pattern:
/// \code
///     if (idx < tablesize)
///        r = table[idx]; // table does not contain default_value
///     else
///        r = default_value;
///     if (r != default_value)
///        ...
/// \endcode
/// Is optimized to:
/// \code
///     cond = idx < tablesize;
///     if (cond)
///        r = table[idx];
///     else
///        r = default_value;
///     if (cond)
///        ...
/// \endcode
/// Jump threading will then eliminate the second if(cond).
static void reuseTableCompare(
    User *PhiUser, BasicBlock *PhiBlock, BranchInst *RangeCheckBranch,
    Constant *DefaultValue,
    const SmallVectorImpl<std::pair<ConstantInt *, Constant *>> &Values) {
  ICmpInst *CmpInst = dyn_cast<ICmpInst>(PhiUser);
  if (!CmpInst)
    return;

  // We require that the compare is in the same block as the phi so that jump
  // threading can do its work afterwards.
  if (CmpInst->getParent() != PhiBlock)
    return;

  Constant *CmpOp1 = dyn_cast<Constant>(CmpInst->getOperand(1));
  if (!CmpOp1)
    return;

  Value *RangeCmp = RangeCheckBranch->getCondition();
  Constant *TrueConst = ConstantInt::getTrue(RangeCmp->getType());
  Constant *FalseConst = ConstantInt::getFalse(RangeCmp->getType());

  // Check if the compare with the default value is constant true or false.
  const DataLayout &DL = PhiBlock->getDataLayout();
  Constant *DefaultConst = ConstantFoldCompareInstOperands(
      CmpInst->getPredicate(), DefaultValue, CmpOp1, DL);
  if (DefaultConst != TrueConst && DefaultConst != FalseConst)
    return;

  // Check if the compare with the case values is distinct from the default
  // compare result.
  for (auto ValuePair : Values) {
    Constant *CaseConst = ConstantFoldCompareInstOperands(
        CmpInst->getPredicate(), ValuePair.second, CmpOp1, DL);
    if (!CaseConst || CaseConst == DefaultConst ||
        (CaseConst != TrueConst && CaseConst != FalseConst))
      return;
  }

  // Check if the branch instruction dominates the phi node. It's a simple
  // dominance check, but sufficient for our needs.
  // Although this check is invariant in the calling loops, it's better to do it
  // at this late stage. Practically we do it at most once for a switch.
  BasicBlock *BranchBlock = RangeCheckBranch->getParent();
  for (BasicBlock *Pred : predecessors(PhiBlock)) {
    if (Pred != BranchBlock && Pred->getUniquePredecessor() != BranchBlock)
      return;
  }

  if (DefaultConst == FalseConst) {
    // The compare yields the same result. We can replace it.
    CmpInst->replaceAllUsesWith(RangeCmp);
    ++NumTableCmpReuses;
  } else {
    // The compare yields the same result, just inverted. We can replace it.
    Value *InvertedTableCmp = BinaryOperator::CreateXor(
        RangeCmp, ConstantInt::get(RangeCmp->getType(), 1), "inverted.cmp",
        RangeCheckBranch->getIterator());
    CmpInst->replaceAllUsesWith(InvertedTableCmp);
    ++NumTableCmpReuses;
  }
}

/// If the switch is only used to initialize one or more phi nodes in a common
/// successor block with different constant values, replace the switch with
/// lookup tables.
static bool simplifySwitchLookup(SwitchInst *SI, IRBuilder<> &Builder,
                                 DomTreeUpdater *DTU, const DataLayout &DL,
                                 const TargetTransformInfo &TTI) {
  assert(SI->getNumCases() > 1 && "Degenerate switch?");

  BasicBlock *BB = SI->getParent();
  Function *Fn = BB->getParent();
  // Only build lookup table when we have a target that supports it or the
  // attribute is not set.
  if (!TTI.shouldBuildLookupTables() ||
      (Fn->getFnAttribute("no-jump-tables").getValueAsBool()))
    return false;

  // FIXME: If the switch is too sparse for a lookup table, perhaps we could
  // split off a dense part and build a lookup table for that.

  // FIXME: This creates arrays of GEPs to constant strings, which means each
  // GEP needs a runtime relocation in PIC code. We should just build one big
  // string and lookup indices into that.

  // Ignore switches with less than three cases. Lookup tables will not make
  // them faster, so we don't analyze them.
  if (SI->getNumCases() < 3)
    return false;

  // Figure out the corresponding result for each case value and phi node in the
  // common destination, as well as the min and max case values.
  assert(!SI->cases().empty());
  SwitchInst::CaseIt CI = SI->case_begin();
  ConstantInt *MinCaseVal = CI->getCaseValue();
  ConstantInt *MaxCaseVal = CI->getCaseValue();

  BasicBlock *CommonDest = nullptr;

  using ResultListTy = SmallVector<std::pair<ConstantInt *, Constant *>, 4>;
  SmallDenseMap<PHINode *, ResultListTy> ResultLists;

  SmallDenseMap<PHINode *, Constant *> DefaultResults;
  SmallVector<Type *> ResultTypes;
  SmallVector<PHINode *, 4> PHIs;

  for (SwitchInst::CaseIt E = SI->case_end(); CI != E; ++CI) {
    ConstantInt *CaseVal = CI->getCaseValue();
    if (CaseVal->getValue().slt(MinCaseVal->getValue()))
      MinCaseVal = CaseVal;
    if (CaseVal->getValue().sgt(MaxCaseVal->getValue()))
      MaxCaseVal = CaseVal;

    // Resulting value at phi nodes for this case value.
    using ResultsTy = SmallVector<std::pair<PHINode *, Constant *>, 4>;
    ResultsTy Results;
    if (!getCaseResults(SI, CaseVal, CI->getCaseSuccessor(), &CommonDest,
                        Results, DL, TTI))
      return false;

    // Append the result and result types from this case to the list for each
    // phi.
    for (const auto &I : Results) {
      PHINode *PHI = I.first;
      Constant *Value = I.second;
      auto [It, Inserted] = ResultLists.try_emplace(PHI);
      if (Inserted)
        PHIs.push_back(PHI);
      It->second.push_back(std::make_pair(CaseVal, Value));
      ResultTypes.push_back(PHI->getType());
    }
  }

  // If the table has holes, we need a constant result for the default case
  // or a bitmask that fits in a register.
  SmallVector<std::pair<PHINode *, Constant *>, 4> DefaultResultsList;
  bool HasDefaultResults =
      getCaseResults(SI, nullptr, SI->getDefaultDest(), &CommonDest,
                     DefaultResultsList, DL, TTI);
  for (const auto &I : DefaultResultsList) {
    PHINode *PHI = I.first;
    Constant *Result = I.second;
    DefaultResults[PHI] = Result;
  }

  bool UseSwitchConditionAsTableIndex = shouldUseSwitchConditionAsTableIndex(
      *MinCaseVal, *MaxCaseVal, HasDefaultResults, ResultTypes, DL, TTI);
  uint64_t TableSize;
  ConstantInt *TableIndexOffset;
  if (UseSwitchConditionAsTableIndex) {
    TableSize = MaxCaseVal->getLimitedValue() + 1;
    TableIndexOffset = ConstantInt::get(MaxCaseVal->getIntegerType(), 0);
  } else {
    TableSize =
        (MaxCaseVal->getValue() - MinCaseVal->getValue()).getLimitedValue() + 1;

    TableIndexOffset = MinCaseVal;
  }

  // If the default destination is unreachable, or if the lookup table covers
  // all values of the conditional variable, branch directly to the lookup table
  // BB. Otherwise, check that the condition is within the case range.
  uint64_t NumResults = ResultLists[PHIs[0]].size();
  bool DefaultIsReachable = !SI->defaultDestUnreachable();

  bool TableHasHoles = (NumResults < TableSize);

  // If the table has holes but the default destination doesn't produce any
  // constant results, the lookup table entries corresponding to the holes will
  // contain poison.
  bool AllHolesArePoison = TableHasHoles && !HasDefaultResults;

  // If the default destination doesn't produce a constant result but is still
  // reachable, and the lookup table has holes, we need to use a mask to
  // determine if the current index should load from the lookup table or jump
  // to the default case.
  // The mask is unnecessary if the table has holes but the default destination
  // is unreachable, as in that case the holes must also be unreachable.
  bool NeedMask = AllHolesArePoison && DefaultIsReachable;
  if (NeedMask) {
    // As an extra penalty for the validity test we require more cases.
    if (SI->getNumCases() < 4) // FIXME: Find best threshold value (benchmark).
      return false;
    if (!DL.fitsInLegalInteger(TableSize))
      return false;
  }

  if (!shouldBuildLookupTable(SI, TableSize, TTI, DL, ResultTypes))
    return false;

  // Compute the table index value.
  Value *TableIndex;
  if (UseSwitchConditionAsTableIndex) {
    TableIndex = SI->getCondition();
    if (HasDefaultResults) {
      // Grow the table to cover all possible index values to avoid the range
      // check. It will use the default result to fill in the table hole later,
      // so make sure it exist.
      ConstantRange CR =
          computeConstantRange(TableIndex, /* ForSigned */ false);
      // Grow the table shouldn't have any size impact by checking
      // wouldFitInRegister.
      // TODO: Consider growing the table also when it doesn't fit in a register
      // if no optsize is specified.
      const uint64_t UpperBound = CR.getUpper().getLimitedValue();
      if (!CR.isUpperWrapped() &&
          all_of(ResultTypes, [&](const auto &ResultType) {
            return SwitchReplacement::wouldFitInRegister(DL, UpperBound,
                                                         ResultType);
          })) {
        // There may be some case index larger than the UpperBound (unreachable
        // case), so make sure the table size does not get smaller.
        TableSize = std::max(UpperBound, TableSize);
        // The default branch is unreachable after we enlarge the lookup table.
        // Adjust DefaultIsReachable to reuse code path.
        DefaultIsReachable = false;
      }
    }
  }

  // Keep track of the switch replacement for each phi
  SmallDenseMap<PHINode *, SwitchReplacement> PhiToReplacementMap;
  for (PHINode *PHI : PHIs) {
    const auto &ResultList = ResultLists[PHI];

    Type *ResultType = ResultList.begin()->second->getType();
    // Use any value to fill the lookup table holes.
    Constant *DefaultVal =
        AllHolesArePoison ? PoisonValue::get(ResultType) : DefaultResults[PHI];
    StringRef FuncName = Fn->getName();
    SwitchReplacement Replacement(*Fn->getParent(), TableSize, TableIndexOffset,
                                  ResultList, DefaultVal, DL, FuncName);
    PhiToReplacementMap.insert({PHI, Replacement});
  }

  Builder.SetInsertPoint(SI);
  // TableIndex is the switch condition - TableIndexOffset if we don't
  // use the condition directly
  if (!UseSwitchConditionAsTableIndex) {
    // If the default is unreachable, all case values are s>= MinCaseVal. Then
    // we can try to attach nsw.
    bool MayWrap = true;
    if (!DefaultIsReachable) {
      APInt Res =
          MaxCaseVal->getValue().ssub_ov(MinCaseVal->getValue(), MayWrap);
      (void)Res;
    }
    TableIndex = Builder.CreateSub(SI->getCondition(), TableIndexOffset,
                                   "switch.tableidx", /*HasNUW =*/false,
                                   /*HasNSW =*/!MayWrap);
  }

  std::vector<DominatorTree::UpdateType> Updates;

  // Compute the maximum table size representable by the integer type we are
  // switching upon.
  unsigned CaseSize = MinCaseVal->getType()->getPrimitiveSizeInBits();
  uint64_t MaxTableSize = CaseSize > 63 ? UINT64_MAX : 1ULL << CaseSize;
  assert(MaxTableSize >= TableSize &&
         "It is impossible for a switch to have more entries than the max "
         "representable value of its input integer type's size.");

  // Create the BB that does the lookups.
  Module &Mod = *CommonDest->getParent()->getParent();
  BasicBlock *LookupBB = BasicBlock::Create(
      Mod.getContext(), "switch.lookup", CommonDest->getParent(), CommonDest);

  BranchInst *RangeCheckBranch = nullptr;

  Builder.SetInsertPoint(SI);
  const bool GeneratingCoveredLookupTable = (MaxTableSize == TableSize);
  if (!DefaultIsReachable || GeneratingCoveredLookupTable) {
    Builder.CreateBr(LookupBB);
    if (DTU)
      Updates.push_back({DominatorTree::Insert, BB, LookupBB});
    // Note: We call removeProdecessor later since we need to be able to get the
    // PHI value for the default case in case we're using a bit mask.
  } else {
    Value *Cmp = Builder.CreateICmpULT(
        TableIndex, ConstantInt::get(MinCaseVal->getType(), TableSize));
    RangeCheckBranch =
        Builder.CreateCondBr(Cmp, LookupBB, SI->getDefaultDest());
    if (DTU)
      Updates.push_back({DominatorTree::Insert, BB, LookupBB});
  }

  // Populate the BB that does the lookups.
  Builder.SetInsertPoint(LookupBB);

  if (NeedMask) {
    // Before doing the lookup, we do the hole check. The LookupBB is therefore
    // re-purposed to do the hole check, and we create a new LookupBB.
    BasicBlock *MaskBB = LookupBB;
    MaskBB->setName("switch.hole_check");
    LookupBB = BasicBlock::Create(Mod.getContext(), "switch.lookup",
                                  CommonDest->getParent(), CommonDest);

    // Make the mask's bitwidth at least 8-bit and a power-of-2 to avoid
    // unnecessary illegal types.
    uint64_t TableSizePowOf2 = NextPowerOf2(std::max(7ULL, TableSize - 1ULL));
    APInt MaskInt(TableSizePowOf2, 0);
    APInt One(TableSizePowOf2, 1);
    // Build bitmask; fill in a 1 bit for every case.
    const ResultListTy &ResultList = ResultLists[PHIs[0]];
    for (const auto &Result : ResultList) {
      uint64_t Idx = (Result.first->getValue() - TableIndexOffset->getValue())
                         .getLimitedValue();
      MaskInt |= One << Idx;
    }
    ConstantInt *TableMask = ConstantInt::get(Mod.getContext(), MaskInt);

    // Get the TableIndex'th bit of the bitmask.
    // If this bit is 0 (meaning hole) jump to the default destination,
    // else continue with table lookup.
    IntegerType *MapTy = TableMask->getIntegerType();
    Value *MaskIndex =
        Builder.CreateZExtOrTrunc(TableIndex, MapTy, "switch.maskindex");
    Value *Shifted = Builder.CreateLShr(TableMask, MaskIndex, "switch.shifted");
    Value *LoBit = Builder.CreateTrunc(
        Shifted, Type::getInt1Ty(Mod.getContext()), "switch.lobit");
    Builder.CreateCondBr(LoBit, LookupBB, SI->getDefaultDest());
    if (DTU) {
      Updates.push_back({DominatorTree::Insert, MaskBB, LookupBB});
      Updates.push_back({DominatorTree::Insert, MaskBB, SI->getDefaultDest()});
    }
    Builder.SetInsertPoint(LookupBB);
    addPredecessorToBlock(SI->getDefaultDest(), MaskBB, BB);
  }

  if (!DefaultIsReachable || GeneratingCoveredLookupTable) {
    // We cached PHINodes in PHIs. To avoid accessing deleted PHINodes later,
    // do not delete PHINodes here.
    SI->getDefaultDest()->removePredecessor(BB,
                                            /*KeepOneInputPHIs=*/true);
    if (DTU)
      Updates.push_back({DominatorTree::Delete, BB, SI->getDefaultDest()});
  }

  for (PHINode *PHI : PHIs) {
    const ResultListTy &ResultList = ResultLists[PHI];
    auto Replacement = PhiToReplacementMap.at(PHI);
    auto *Result = Replacement.replaceSwitch(TableIndex, Builder, DL, Fn);
    // Do a small peephole optimization: re-use the switch table compare if
    // possible.
    if (!TableHasHoles && HasDefaultResults && RangeCheckBranch) {
      BasicBlock *PhiBlock = PHI->getParent();
      // Search for compare instructions which use the phi.
      for (auto *User : PHI->users()) {
        reuseTableCompare(User, PhiBlock, RangeCheckBranch,
                          Replacement.getDefaultValue(), ResultList);
      }
    }

    PHI->addIncoming(Result, LookupBB);
  }

  Builder.CreateBr(CommonDest);
  if (DTU)
    Updates.push_back({DominatorTree::Insert, LookupBB, CommonDest});

  // Remove the switch.
  SmallPtrSet<BasicBlock *, 8> RemovedSuccessors;
  for (unsigned i = 0, e = SI->getNumSuccessors(); i < e; ++i) {
    BasicBlock *Succ = SI->getSuccessor(i);

    if (Succ == SI->getDefaultDest())
      continue;
    Succ->removePredecessor(BB);
    if (DTU && RemovedSuccessors.insert(Succ).second)
      Updates.push_back({DominatorTree::Delete, BB, Succ});
  }
  SI->eraseFromParent();

  if (DTU)
    DTU->applyUpdates(Updates);

  ++NumLookupTables;
  if (NeedMask)
    ++NumLookupTablesHoles;
  return true;
}

/// Try to transform a switch that has "holes" in it to a contiguous sequence
/// of cases.
///
/// A switch such as: switch(i) {case 5: case 9: case 13: case 17:} can be
/// range-reduced to: switch ((i-5) / 4) {case 0: case 1: case 2: case 3:}.
///
/// This converts a sparse switch into a dense switch which allows better
/// lowering and could also allow transforming into a lookup table.
static bool reduceSwitchRange(SwitchInst *SI, IRBuilder<> &Builder,
                              const DataLayout &DL,
                              const TargetTransformInfo &TTI) {
  auto *CondTy = cast<IntegerType>(SI->getCondition()->getType());
  if (CondTy->getIntegerBitWidth() > 64 ||
      !DL.fitsInLegalInteger(CondTy->getIntegerBitWidth()))
    return false;
  // Only bother with this optimization if there are more than 3 switch cases;
  // SDAG will only bother creating jump tables for 4 or more cases.
  if (SI->getNumCases() < 4)
    return false;

  // This transform is agnostic to the signedness of the input or case values. We
  // can treat the case values as signed or unsigned. We can optimize more common
  // cases such as a sequence crossing zero {-4,0,4,8} if we interpret case values
  // as signed.
  SmallVector<int64_t,4> Values;
  for (const auto &C : SI->cases())
    Values.push_back(C.getCaseValue()->getValue().getSExtValue());
  llvm::sort(Values);

  // If the switch is already dense, there's nothing useful to do here.
  if (isSwitchDense(Values))
    return false;

  // First, transform the values such that they start at zero and ascend.
  int64_t Base = Values[0];
  for (auto &V : Values)
    V -= (uint64_t)(Base);

  // Now we have signed numbers that have been shifted so that, given enough
  // precision, there are no negative values. Since the rest of the transform
  // is bitwise only, we switch now to an unsigned representation.

  // This transform can be done speculatively because it is so cheap - it
  // results in a single rotate operation being inserted.

  // countTrailingZeros(0) returns 64. As Values is guaranteed to have more than
  // one element and LLVM disallows duplicate cases, Shift is guaranteed to be
  // less than 64.
  unsigned Shift = 64;
  for (auto &V : Values)
    Shift = std::min(Shift, (unsigned)llvm::countr_zero((uint64_t)V));
  assert(Shift < 64);
  if (Shift > 0)
    for (auto &V : Values)
      V = (int64_t)((uint64_t)V >> Shift);

  if (!isSwitchDense(Values))
    // Transform didn't create a dense switch.
    return false;

  // The obvious transform is to shift the switch condition right and emit a
  // check that the condition actually cleanly divided by GCD, i.e.
  //   C & (1 << Shift - 1) == 0
  // inserting a new CFG edge to handle the case where it didn't divide cleanly.
  //
  // A cheaper way of doing this is a simple ROTR(C, Shift). This performs the
  // shift and puts the shifted-off bits in the uppermost bits. If any of these
  // are nonzero then the switch condition will be very large and will hit the
  // default case.

  auto *Ty = cast<IntegerType>(SI->getCondition()->getType());
  Builder.SetInsertPoint(SI);
  Value *Sub =
      Builder.CreateSub(SI->getCondition(), ConstantInt::get(Ty, Base));
  Value *Rot = Builder.CreateIntrinsic(
      Ty, Intrinsic::fshl,
      {Sub, Sub, ConstantInt::get(Ty, Ty->getBitWidth() - Shift)});
  SI->replaceUsesOfWith(SI->getCondition(), Rot);

  for (auto Case : SI->cases()) {
    auto *Orig = Case.getCaseValue();
    auto Sub = Orig->getValue() - APInt(Ty->getBitWidth(), Base, true);
    Case.setValue(cast<ConstantInt>(ConstantInt::get(Ty, Sub.lshr(Shift))));
  }
  return true;
}

/// Tries to transform switch of powers of two to reduce switch range.
/// For example, switch like:
/// switch (C) { case 1: case 2: case 64: case 128: }
/// will be transformed to:
/// switch (count_trailing_zeros(C)) { case 0: case 1: case 6: case 7: }
///
/// This transformation allows better lowering and may transform the switch
/// instruction into a sequence of bit manipulation and a smaller
/// log2(C)-indexed value table (instead of traditionally emitting a load of the
/// address of the jump target, and indirectly jump to it).
static bool simplifySwitchOfPowersOfTwo(SwitchInst *SI, IRBuilder<> &Builder,
                                        const DataLayout &DL,
                                        const TargetTransformInfo &TTI) {
  Value *Condition = SI->getCondition();
  LLVMContext &Context = SI->getContext();
  auto *CondTy = cast<IntegerType>(Condition->getType());

  if (CondTy->getIntegerBitWidth() > 64 ||
      !DL.fitsInLegalInteger(CondTy->getIntegerBitWidth()))
    return false;

  // Ensure trailing zeroes count intrinsic emission is not too expensive.
  IntrinsicCostAttributes Attrs(Intrinsic::cttz, CondTy,
                                {Condition, ConstantInt::getTrue(Context)});
  if (TTI.getIntrinsicInstrCost(Attrs, TTI::TCK_SizeAndLatency) >
      TTI::TCC_Basic * 2)
    return false;

  // Only bother with this optimization if there are more than 3 switch cases.
  // SDAG will start emitting jump tables for 4 or more cases.
  if (SI->getNumCases() < 4)
    return false;

  // We perform this optimization only for switches with
  // unreachable default case.
  // This assumtion will save us from checking if `Condition` is a power of two.
  if (!SI->defaultDestUnreachable())
    return false;

  // Check that switch cases are powers of two.
  SmallVector<uint64_t, 4> Values;
  for (const auto &Case : SI->cases()) {
    uint64_t CaseValue = Case.getCaseValue()->getValue().getZExtValue();
    if (llvm::has_single_bit(CaseValue))
      Values.push_back(CaseValue);
    else
      return false;
  }

  // isSwichDense requires case values to be sorted.
  llvm::sort(Values);
  if (!isSwitchDense(Values.size(), llvm::countr_zero(Values.back()) -
                                        llvm::countr_zero(Values.front()) + 1))
    // Transform is unable to generate dense switch.
    return false;

  Builder.SetInsertPoint(SI);

  // Replace each case with its trailing zeros number.
  for (auto &Case : SI->cases()) {
    auto *OrigValue = Case.getCaseValue();
    Case.setValue(ConstantInt::get(OrigValue->getIntegerType(),
                                   OrigValue->getValue().countr_zero()));
  }

  // Replace condition with its trailing zeros number.
  auto *ConditionTrailingZeros = Builder.CreateIntrinsic(
      Intrinsic::cttz, {CondTy}, {Condition, ConstantInt::getTrue(Context)});

  SI->setCondition(ConditionTrailingZeros);

  return true;
}

/// Fold switch over ucmp/scmp intrinsic to br if two of the switch arms have
/// the same destination.
static bool simplifySwitchOfCmpIntrinsic(SwitchInst *SI, IRBuilderBase &Builder,
                                         DomTreeUpdater *DTU) {
  auto *Cmp = dyn_cast<CmpIntrinsic>(SI->getCondition());
  if (!Cmp || !Cmp->hasOneUse())
    return false;

  SmallVector<uint32_t, 4> Weights;
  bool HasWeights = extractBranchWeights(getBranchWeightMDNode(*SI), Weights);
  if (!HasWeights)
    Weights.resize(4); // Avoid checking HasWeights everywhere.

  // Normalize to [us]cmp == Res ? Succ : OtherSucc.
  int64_t Res;
  BasicBlock *Succ, *OtherSucc;
  uint32_t SuccWeight = 0, OtherSuccWeight = 0;
  BasicBlock *Unreachable = nullptr;

  if (SI->getNumCases() == 2) {
    // Find which of 1, 0 or -1 is missing (handled by default dest).
    SmallSet<int64_t, 3> Missing;
    Missing.insert(1);
    Missing.insert(0);
    Missing.insert(-1);

    Succ = SI->getDefaultDest();
    SuccWeight = Weights[0];
    OtherSucc = nullptr;
    for (auto &Case : SI->cases()) {
      std::optional<int64_t> Val =
          Case.getCaseValue()->getValue().trySExtValue();
      if (!Val)
        return false;
      if (!Missing.erase(*Val))
        return false;
      if (OtherSucc && OtherSucc != Case.getCaseSuccessor())
        return false;
      OtherSucc = Case.getCaseSuccessor();
      OtherSuccWeight += Weights[Case.getSuccessorIndex()];
    }

    assert(Missing.size() == 1 && "Should have one case left");
    Res = *Missing.begin();
  } else if (SI->getNumCases() == 3 && SI->defaultDestUnreachable()) {
    // Normalize so that Succ is taken once and OtherSucc twice.
    Unreachable = SI->getDefaultDest();
    Succ = OtherSucc = nullptr;
    for (auto &Case : SI->cases()) {
      BasicBlock *NewSucc = Case.getCaseSuccessor();
      uint32_t Weight = Weights[Case.getSuccessorIndex()];
      if (!OtherSucc || OtherSucc == NewSucc) {
        OtherSucc = NewSucc;
        OtherSuccWeight += Weight;
      } else if (!Succ) {
        Succ = NewSucc;
        SuccWeight = Weight;
      } else if (Succ == NewSucc) {
        std::swap(Succ, OtherSucc);
        std::swap(SuccWeight, OtherSuccWeight);
      } else
        return false;
    }
    for (auto &Case : SI->cases()) {
      std::optional<int64_t> Val =
          Case.getCaseValue()->getValue().trySExtValue();
      if (!Val || (Val != 1 && Val != 0 && Val != -1))
        return false;
      if (Case.getCaseSuccessor() == Succ) {
        Res = *Val;
        break;
      }
    }
  } else {
    return false;
  }

  // Determine predicate for the missing case.
  ICmpInst::Predicate Pred;
  switch (Res) {
  case 1:
    Pred = ICmpInst::ICMP_UGT;
    break;
  case 0:
    Pred = ICmpInst::ICMP_EQ;
    break;
  case -1:
    Pred = ICmpInst::ICMP_ULT;
    break;
  }
  if (Cmp->isSigned())
    Pred = ICmpInst::getSignedPredicate(Pred);

  MDNode *NewWeights = nullptr;
  if (HasWeights)
    NewWeights = MDBuilder(SI->getContext())
                     .createBranchWeights(SuccWeight, OtherSuccWeight);

  BasicBlock *BB = SI->getParent();
  Builder.SetInsertPoint(SI->getIterator());
  Value *ICmp = Builder.CreateICmp(Pred, Cmp->getLHS(), Cmp->getRHS());
  Builder.CreateCondBr(ICmp, Succ, OtherSucc, NewWeights,
                       SI->getMetadata(LLVMContext::MD_unpredictable));
  OtherSucc->removePredecessor(BB);
  if (Unreachable)
    Unreachable->removePredecessor(BB);
  SI->eraseFromParent();
  Cmp->eraseFromParent();
  if (DTU && Unreachable)
    DTU->applyUpdates({{DominatorTree::Delete, BB, Unreachable}});
  return true;
}

/// Checking whether two cases of SI are equal depends on the contents of the
/// BasicBlock and the incoming values of their successor PHINodes.
/// PHINode::getIncomingValueForBlock is O(|Preds|), so we'd like to avoid
/// calling this function on each BasicBlock every time isEqual is called,
/// especially since the same BasicBlock may be passed as an argument multiple
/// times. To do this, we can precompute a map of PHINode -> Pred BasicBlock ->
/// IncomingValue and add it in the Wrapper so isEqual can do O(1) checking
/// of the incoming values.
struct SwitchSuccWrapper {
  BasicBlock *Dest;
  DenseMap<PHINode *, SmallDenseMap<BasicBlock *, Value *, 8>> *PhiPredIVs;
};

namespace llvm {
template <> struct DenseMapInfo<const SwitchSuccWrapper *> {
  static const SwitchSuccWrapper *getEmptyKey() {
    return static_cast<SwitchSuccWrapper *>(
        DenseMapInfo<void *>::getEmptyKey());
  }
  static const SwitchSuccWrapper *getTombstoneKey() {
    return static_cast<SwitchSuccWrapper *>(
        DenseMapInfo<void *>::getTombstoneKey());
  }
  static unsigned getHashValue(const SwitchSuccWrapper *SSW) {
    BasicBlock *Succ = SSW->Dest;
    BranchInst *BI = cast<BranchInst>(Succ->getTerminator());
    assert(BI->isUnconditional() &&
           "Only supporting unconditional branches for now");
    assert(BI->getNumSuccessors() == 1 &&
           "Expected unconditional branches to have one successor");
    assert(Succ->size() == 1 && "Expected just a single branch in the BB");

    // Since we assume the BB is just a single BranchInst with a single
    // successor, we hash as the BB and the incoming Values of its successor
    // PHIs. Initially, we tried to just use the successor BB as the hash, but
    // including the incoming PHI values leads to better performance.
    // We also tried to build a map from BB -> Succs.IncomingValues ahead of
    // time and passing it in SwitchSuccWrapper, but this slowed down the
    // average compile time without having any impact on the worst case compile
    // time.
    BasicBlock *BB = BI->getSuccessor(0);
    SmallVector<Value *> PhiValsForBB;
    for (PHINode &Phi : BB->phis())
      PhiValsForBB.emplace_back((*SSW->PhiPredIVs)[&Phi][BB]);

    return hash_combine(BB, hash_combine_range(PhiValsForBB));
  }
  static bool isEqual(const SwitchSuccWrapper *LHS,
                      const SwitchSuccWrapper *RHS) {
    auto EKey = DenseMapInfo<SwitchSuccWrapper *>::getEmptyKey();
    auto TKey = DenseMapInfo<SwitchSuccWrapper *>::getTombstoneKey();
    if (LHS == EKey || RHS == EKey || LHS == TKey || RHS == TKey)
      return LHS == RHS;

    BasicBlock *A = LHS->Dest;
    BasicBlock *B = RHS->Dest;

    // FIXME: we checked that the size of A and B are both 1 in
    // simplifyDuplicateSwitchArms to make the Case list smaller to
    // improve performance. If we decide to support BasicBlocks with more
    // than just a single instruction, we need to check that A.size() ==
    // B.size() here, and we need to check more than just the BranchInsts
    // for equality.

    BranchInst *ABI = cast<BranchInst>(A->getTerminator());
    BranchInst *BBI = cast<BranchInst>(B->getTerminator());
    assert(ABI->isUnconditional() && BBI->isUnconditional() &&
           "Only supporting unconditional branches for now");
    if (ABI->getSuccessor(0) != BBI->getSuccessor(0))
      return false;

    // Need to check that PHIs in successor have matching values
    BasicBlock *Succ = ABI->getSuccessor(0);
    for (PHINode &Phi : Succ->phis()) {
      auto &PredIVs = (*LHS->PhiPredIVs)[&Phi];
      if (PredIVs[A] != PredIVs[B])
        return false;
    }

    return true;
  }
};
} // namespace llvm

bool SimplifyCFGOpt::simplifyDuplicateSwitchArms(SwitchInst *SI,
                                                 DomTreeUpdater *DTU) {
  // Build Cases. Skip BBs that are not candidates for simplification. Mark
  // PHINodes which need to be processed into PhiPredIVs. We decide to process
  // an entire PHI at once after the loop, opposed to calling
  // getIncomingValueForBlock inside this loop, since each call to
  // getIncomingValueForBlock is O(|Preds|).
  SmallPtrSet<PHINode *, 8> Phis;
  SmallPtrSet<BasicBlock *, 8> Seen;
  DenseMap<PHINode *, SmallDenseMap<BasicBlock *, Value *, 8>> PhiPredIVs;
  DenseMap<BasicBlock *, SmallVector<unsigned, 32>> BBToSuccessorIndexes;
  SmallVector<SwitchSuccWrapper> Cases;
  Cases.reserve(SI->getNumSuccessors());

  for (unsigned I = 0; I < SI->getNumSuccessors(); ++I) {
    BasicBlock *BB = SI->getSuccessor(I);

    // FIXME: Support more than just a single BranchInst. One way we could do
    // this is by taking a hashing approach of all insts in BB.
    if (BB->size() != 1)
      continue;

    // FIXME: Relax that the terminator is a BranchInst by checking for equality
    // on other kinds of terminators. We decide to only support unconditional
    // branches for now for compile time reasons.
    auto *BI = dyn_cast<BranchInst>(BB->getTerminator());
    if (!BI || BI->isConditional())
      continue;

    if (!Seen.insert(BB).second) {
      auto It = BBToSuccessorIndexes.find(BB);
      if (It != BBToSuccessorIndexes.end())
        It->second.emplace_back(I);
      continue;
    }

    // FIXME: This case needs some extra care because the terminators other than
    // SI need to be updated. For now, consider only backedges to the SI.
    if (BB->getUniquePredecessor() != SI->getParent())
      continue;

    // Keep track of which PHIs we need as keys in PhiPredIVs below.
    for (BasicBlock *Succ : BI->successors())
      Phis.insert_range(llvm::make_pointer_range(Succ->phis()));

    // Add the successor only if not previously visited.
    Cases.emplace_back(SwitchSuccWrapper{BB, &PhiPredIVs});
    BBToSuccessorIndexes[BB].emplace_back(I);
  }

  // Precompute a data structure to improve performance of isEqual for
  // SwitchSuccWrapper.
  PhiPredIVs.reserve(Phis.size());
  for (PHINode *Phi : Phis) {
    auto &IVs =
        PhiPredIVs.try_emplace(Phi, Phi->getNumIncomingValues()).first->second;
    for (auto &IV : Phi->incoming_values())
      IVs.insert({Phi->getIncomingBlock(IV), IV.get()});
  }

  // Build a set such that if the SwitchSuccWrapper exists in the set and
  // another SwitchSuccWrapper isEqual, then the equivalent SwitchSuccWrapper
  // which is not in the set should be replaced with the one in the set. If the
  // SwitchSuccWrapper is not in the set, then it should be added to the set so
  // other SwitchSuccWrappers can check against it in the same manner. We use
  // SwitchSuccWrapper instead of just BasicBlock because we'd like to pass
  // around information to isEquality, getHashValue, and when doing the
  // replacement with better performance.
  DenseSet<const SwitchSuccWrapper *> ReplaceWith;
  ReplaceWith.reserve(Cases.size());

  SmallVector<DominatorTree::UpdateType> Updates;
  Updates.reserve(ReplaceWith.size());
  bool MadeChange = false;
  for (auto &SSW : Cases) {
    // SSW is a candidate for simplification. If we find a duplicate BB,
    // replace it.
    const auto [It, Inserted] = ReplaceWith.insert(&SSW);
    if (!Inserted) {
      // We know that SI's parent BB no longer dominates the old case successor
      // since we are making it dead.
      Updates.push_back({DominatorTree::Delete, SI->getParent(), SSW.Dest});
      const auto &Successors = BBToSuccessorIndexes.at(SSW.Dest);
      for (unsigned Idx : Successors)
        SI->setSuccessor(Idx, (*It)->Dest);
      MadeChange = true;
    }
  }

  if (DTU)
    DTU->applyUpdates(Updates);

  return MadeChange;
}

bool SimplifyCFGOpt::simplifySwitch(SwitchInst *SI, IRBuilder<> &Builder) {
  BasicBlock *BB = SI->getParent();

  if (isValueEqualityComparison(SI)) {
    // If we only have one predecessor, and if it is a branch on this value,
    // see if that predecessor totally determines the outcome of this switch.
    if (BasicBlock *OnlyPred = BB->getSinglePredecessor())
      if (simplifyEqualityComparisonWithOnlyPredecessor(SI, OnlyPred, Builder))
        return requestResimplify();

    Value *Cond = SI->getCondition();
    if (SelectInst *Select = dyn_cast<SelectInst>(Cond))
      if (simplifySwitchOnSelect(SI, Select))
        return requestResimplify();

    // If the block only contains the switch, see if we can fold the block
    // away into any preds.
    if (SI == &*BB->instructionsWithoutDebug(false).begin())
      if (foldValueComparisonIntoPredecessors(SI, Builder))
        return requestResimplify();
  }

  // Try to transform the switch into an icmp and a branch.
  // The conversion from switch to comparison may lose information on
  // impossible switch values, so disable it early in the pipeline.
  if (Options.ConvertSwitchRangeToICmp && turnSwitchRangeIntoICmp(SI, Builder))
    return requestResimplify();

  // Remove unreachable cases.
  if (eliminateDeadSwitchCases(SI, DTU, Options.AC, DL))
    return requestResimplify();

  if (simplifySwitchOfCmpIntrinsic(SI, Builder, DTU))
    return requestResimplify();

  if (trySwitchToSelect(SI, Builder, DTU, DL, TTI))
    return requestResimplify();

  if (Options.ForwardSwitchCondToPhi && forwardSwitchConditionToPHI(SI))
    return requestResimplify();

  // The conversion from switch to lookup tables results in difficult-to-analyze
  // code and makes pruning branches much harder. This is a problem if the
  // switch expression itself can still be restricted as a result of inlining or
  // CVP. Therefore, only apply this transformation during late stages of the
  // optimisation pipeline.
  if (Options.ConvertSwitchToLookupTable &&
      simplifySwitchLookup(SI, Builder, DTU, DL, TTI))
    return requestResimplify();

  if (simplifySwitchOfPowersOfTwo(SI, Builder, DL, TTI))
    return requestResimplify();

  if (reduceSwitchRange(SI, Builder, DL, TTI))
    return requestResimplify();

  if (HoistCommon &&
      hoistCommonCodeFromSuccessors(SI, !Options.HoistCommonInsts))
    return requestResimplify();

  if (simplifyDuplicateSwitchArms(SI, DTU))
    return requestResimplify();

  return false;
}

bool SimplifyCFGOpt::simplifyIndirectBr(IndirectBrInst *IBI) {
  BasicBlock *BB = IBI->getParent();
  bool Changed = false;

  // Eliminate redundant destinations.
  SmallPtrSet<Value *, 8> Succs;
  SmallSetVector<BasicBlock *, 8> RemovedSuccs;
  for (unsigned i = 0, e = IBI->getNumDestinations(); i != e; ++i) {
    BasicBlock *Dest = IBI->getDestination(i);
    if (!Dest->hasAddressTaken() || !Succs.insert(Dest).second) {
      if (!Dest->hasAddressTaken())
        RemovedSuccs.insert(Dest);
      Dest->removePredecessor(BB);
      IBI->removeDestination(i);
      --i;
      --e;
      Changed = true;
    }
  }

  if (DTU) {
    std::vector<DominatorTree::UpdateType> Updates;
    Updates.reserve(RemovedSuccs.size());
    for (auto *RemovedSucc : RemovedSuccs)
      Updates.push_back({DominatorTree::Delete, BB, RemovedSucc});
    DTU->applyUpdates(Updates);
  }

  if (IBI->getNumDestinations() == 0) {
    // If the indirectbr has no successors, change it to unreachable.
    new UnreachableInst(IBI->getContext(), IBI->getIterator());
    eraseTerminatorAndDCECond(IBI);
    return true;
  }

  if (IBI->getNumDestinations() == 1) {
    // If the indirectbr has one successor, change it to a direct branch.
    BranchInst::Create(IBI->getDestination(0), IBI->getIterator());
    eraseTerminatorAndDCECond(IBI);
    return true;
  }

  if (SelectInst *SI = dyn_cast<SelectInst>(IBI->getAddress())) {
    if (simplifyIndirectBrOnSelect(IBI, SI))
      return requestResimplify();
  }
  return Changed;
}

/// Given an block with only a single landing pad and a unconditional branch
/// try to find another basic block which this one can be merged with.  This
/// handles cases where we have multiple invokes with unique landing pads, but
/// a shared handler.
///
/// We specifically choose to not worry about merging non-empty blocks
/// here.  That is a PRE/scheduling problem and is best solved elsewhere.  In
/// practice, the optimizer produces empty landing pad blocks quite frequently
/// when dealing with exception dense code.  (see: instcombine, gvn, if-else
/// sinking in this file)
///
/// This is primarily a code size optimization.  We need to avoid performing
/// any transform which might inhibit optimization (such as our ability to
/// specialize a particular handler via tail commoning).  We do this by not
/// merging any blocks which require us to introduce a phi.  Since the same
/// values are flowing through both blocks, we don't lose any ability to
/// specialize.  If anything, we make such specialization more likely.
///
/// TODO - This transformation could remove entries from a phi in the target
/// block when the inputs in the phi are the same for the two blocks being
/// merged.  In some cases, this could result in removal of the PHI entirely.
static bool tryToMergeLandingPad(LandingPadInst *LPad, BranchInst *BI,
                                 BasicBlock *BB, DomTreeUpdater *DTU) {
  auto Succ = BB->getUniqueSuccessor();
  assert(Succ);
  // If there's a phi in the successor block, we'd likely have to introduce
  // a phi into the merged landing pad block.
  if (isa<PHINode>(*Succ->begin()))
    return false;

  for (BasicBlock *OtherPred : predecessors(Succ)) {
    if (BB == OtherPred)
      continue;
    BasicBlock::iterator I = OtherPred->begin();
    LandingPadInst *LPad2 = dyn_cast<LandingPadInst>(I);
    if (!LPad2 || !LPad2->isIdenticalTo(LPad))
      continue;
    ++I;
    BranchInst *BI2 = dyn_cast<BranchInst>(I);
    if (!BI2 || !BI2->isIdenticalTo(BI))
      continue;

    std::vector<DominatorTree::UpdateType> Updates;

    // We've found an identical block.  Update our predecessors to take that
    // path instead and make ourselves dead.
    SmallSetVector<BasicBlock *, 16> UniquePreds(pred_begin(BB), pred_end(BB));
    for (BasicBlock *Pred : UniquePreds) {
      InvokeInst *II = cast<InvokeInst>(Pred->getTerminator());
      assert(II->getNormalDest() != BB && II->getUnwindDest() == BB &&
             "unexpected successor");
      II->setUnwindDest(OtherPred);
      if (DTU) {
        Updates.push_back({DominatorTree::Insert, Pred, OtherPred});
        Updates.push_back({DominatorTree::Delete, Pred, BB});
      }
    }

    SmallSetVector<BasicBlock *, 16> UniqueSuccs(succ_begin(BB), succ_end(BB));
    for (BasicBlock *Succ : UniqueSuccs) {
      Succ->removePredecessor(BB);
      if (DTU)
        Updates.push_back({DominatorTree::Delete, BB, Succ});
    }

    IRBuilder<> Builder(BI);
    Builder.CreateUnreachable();
    BI->eraseFromParent();
    if (DTU)
      DTU->applyUpdates(Updates);
    return true;
  }
  return false;
}

bool SimplifyCFGOpt::simplifyBranch(BranchInst *Branch, IRBuilder<> &Builder) {
  return Branch->isUnconditional() ? simplifyUncondBranch(Branch, Builder)
                                   : simplifyCondBranch(Branch, Builder);
}

bool SimplifyCFGOpt::simplifyUncondBranch(BranchInst *BI,
                                          IRBuilder<> &Builder) {
  BasicBlock *BB = BI->getParent();
  BasicBlock *Succ = BI->getSuccessor(0);

  // If the Terminator is the only non-phi instruction, simplify the block.
  // If LoopHeader is provided, check if the block or its successor is a loop
  // header. (This is for early invocations before loop simplify and
  // vectorization to keep canonical loop forms for nested loops. These blocks
  // can be eliminated when the pass is invoked later in the back-end.)
  // Note that if BB has only one predecessor then we do not introduce new
  // backedge, so we can eliminate BB.
  bool NeedCanonicalLoop =
      Options.NeedCanonicalLoop &&
      (!LoopHeaders.empty() && BB->hasNPredecessorsOrMore(2) &&
       (is_contained(LoopHeaders, BB) || is_contained(LoopHeaders, Succ)));
  BasicBlock::iterator I = BB->getFirstNonPHIOrDbg();
  if (I->isTerminator() && BB != &BB->getParent()->getEntryBlock() &&
      !NeedCanonicalLoop && TryToSimplifyUncondBranchFromEmptyBlock(BB, DTU))
    return true;

  // If the only instruction in the block is a seteq/setne comparison against a
  // constant, try to simplify the block.
  if (ICmpInst *ICI = dyn_cast<ICmpInst>(I))
    if (ICI->isEquality() && isa<ConstantInt>(ICI->getOperand(1))) {
      ++I;
      if (I->isTerminator() &&
          tryToSimplifyUncondBranchWithICmpInIt(ICI, Builder))
        return true;
    }

  // See if we can merge an empty landing pad block with another which is
  // equivalent.
  if (LandingPadInst *LPad = dyn_cast<LandingPadInst>(I)) {
    ++I;
    if (I->isTerminator() && tryToMergeLandingPad(LPad, BI, BB, DTU))
      return true;
  }

  // If this basic block is ONLY a compare and a branch, and if a predecessor
  // branches to us and our successor, fold the comparison into the
  // predecessor and use logical operations to update the incoming value
  // for PHI nodes in common successor.
  if (Options.SpeculateBlocks &&
      foldBranchToCommonDest(BI, DTU, /*MSSAU=*/nullptr, &TTI,
                             Options.BonusInstThreshold))
    return requestResimplify();
  return false;
}

static BasicBlock *allPredecessorsComeFromSameSource(BasicBlock *BB) {
  BasicBlock *PredPred = nullptr;
  for (auto *P : predecessors(BB)) {
    BasicBlock *PPred = P->getSinglePredecessor();
    if (!PPred || (PredPred && PredPred != PPred))
      return nullptr;
    PredPred = PPred;
  }
  return PredPred;
}

/// Fold the following pattern:
/// bb0:
///   br i1 %cond1, label %bb1, label %bb2
/// bb1:
///   br i1 %cond2, label %bb3, label %bb4
/// bb2:
///   br i1 %cond2, label %bb4, label %bb3
/// bb3:
///   ...
/// bb4:
///   ...
/// into
/// bb0:
///   %cond = xor i1 %cond1, %cond2
///   br i1 %cond, label %bb4, label %bb3
/// bb3:
///   ...
/// bb4:
///   ...
/// NOTE: %cond2 always dominates the terminator of bb0.
static bool mergeNestedCondBranch(BranchInst *BI, DomTreeUpdater *DTU) {
  BasicBlock *BB = BI->getParent();
  BasicBlock *BB1 = BI->getSuccessor(0);
  BasicBlock *BB2 = BI->getSuccessor(1);
  auto IsSimpleSuccessor = [BB](BasicBlock *Succ, BranchInst *&SuccBI) {
    if (Succ == BB)
      return false;
    if (&Succ->front() != Succ->getTerminator())
      return false;
    SuccBI = dyn_cast<BranchInst>(Succ->getTerminator());
    if (!SuccBI || !SuccBI->isConditional())
      return false;
    BasicBlock *Succ1 = SuccBI->getSuccessor(0);
    BasicBlock *Succ2 = SuccBI->getSuccessor(1);
    return Succ1 != Succ && Succ2 != Succ && Succ1 != BB && Succ2 != BB &&
           !isa<PHINode>(Succ1->front()) && !isa<PHINode>(Succ2->front());
  };
  BranchInst *BB1BI, *BB2BI;
  if (!IsSimpleSuccessor(BB1, BB1BI) || !IsSimpleSuccessor(BB2, BB2BI))
    return false;

  if (BB1BI->getCondition() != BB2BI->getCondition() ||
      BB1BI->getSuccessor(0) != BB2BI->getSuccessor(1) ||
      BB1BI->getSuccessor(1) != BB2BI->getSuccessor(0))
    return false;

  BasicBlock *BB3 = BB1BI->getSuccessor(0);
  BasicBlock *BB4 = BB1BI->getSuccessor(1);
  IRBuilder<> Builder(BI);
  BI->setCondition(
      Builder.CreateXor(BI->getCondition(), BB1BI->getCondition()));
  BB1->removePredecessor(BB);
  BI->setSuccessor(0, BB4);
  BB2->removePredecessor(BB);
  BI->setSuccessor(1, BB3);
  if (DTU) {
    SmallVector<DominatorTree::UpdateType, 4> Updates;
    Updates.push_back({DominatorTree::Delete, BB, BB1});
    Updates.push_back({DominatorTree::Insert, BB, BB4});
    Updates.push_back({DominatorTree::Delete, BB, BB2});
    Updates.push_back({DominatorTree::Insert, BB, BB3});

    DTU->applyUpdates(Updates);
  }
  bool HasWeight = false;
  uint64_t BBTWeight, BBFWeight;
  if (extractBranchWeights(*BI, BBTWeight, BBFWeight))
    HasWeight = true;
  else
    BBTWeight = BBFWeight = 1;
  uint64_t BB1TWeight, BB1FWeight;
  if (extractBranchWeights(*BB1BI, BB1TWeight, BB1FWeight))
    HasWeight = true;
  else
    BB1TWeight = BB1FWeight = 1;
  uint64_t BB2TWeight, BB2FWeight;
  if (extractBranchWeights(*BB2BI, BB2TWeight, BB2FWeight))
    HasWeight = true;
  else
    BB2TWeight = BB2FWeight = 1;
  if (HasWeight) {
    uint64_t Weights[2] = {BBTWeight * BB1FWeight + BBFWeight * BB2TWeight,
                           BBTWeight * BB1TWeight + BBFWeight * BB2FWeight};
    fitWeights(Weights);
    setBranchWeights(BI, Weights[0], Weights[1], /*IsExpected=*/false);
  }
  return true;
}

bool SimplifyCFGOpt::simplifyCondBranch(BranchInst *BI, IRBuilder<> &Builder) {
  assert(
      !isa<ConstantInt>(BI->getCondition()) &&
      BI->getSuccessor(0) != BI->getSuccessor(1) &&
      "Tautological conditional branch should have been eliminated already.");

  BasicBlock *BB = BI->getParent();
  if (!Options.SimplifyCondBranch ||
      BI->getFunction()->hasFnAttribute(Attribute::OptForFuzzing))
    return false;

  // Conditional branch
  if (isValueEqualityComparison(BI)) {
    // If we only have one predecessor, and if it is a branch on this value,
    // see if that predecessor totally determines the outcome of this
    // switch.
    if (BasicBlock *OnlyPred = BB->getSinglePredecessor())
      if (simplifyEqualityComparisonWithOnlyPredecessor(BI, OnlyPred, Builder))
        return requestResimplify();

    // This block must be empty, except for the setcond inst, if it exists.
    // Ignore dbg and pseudo intrinsics.
    auto I = BB->instructionsWithoutDebug(true).begin();
    if (&*I == BI) {
      if (foldValueComparisonIntoPredecessors(BI, Builder))
        return requestResimplify();
    } else if (&*I == cast<Instruction>(BI->getCondition())) {
      ++I;
      if (&*I == BI && foldValueComparisonIntoPredecessors(BI, Builder))
        return requestResimplify();
    }
  }

  // Try to turn "br (X == 0 | X == 1), T, F" into a switch instruction.
  if (simplifyBranchOnICmpChain(BI, Builder, DL))
    return true;

  // If this basic block has dominating predecessor blocks and the dominating
  // blocks' conditions imply BI's condition, we know the direction of BI.
  std::optional<bool> Imp = isImpliedByDomCondition(BI->getCondition(), BI, DL);
  if (Imp) {
    // Turn this into a branch on constant.
    auto *OldCond = BI->getCondition();
    ConstantInt *TorF = *Imp ? ConstantInt::getTrue(BB->getContext())
                             : ConstantInt::getFalse(BB->getContext());
    BI->setCondition(TorF);
    RecursivelyDeleteTriviallyDeadInstructions(OldCond);
    return requestResimplify();
  }

  // If this basic block is ONLY a compare and a branch, and if a predecessor
  // branches to us and one of our successors, fold the comparison into the
  // predecessor and use logical operations to pick the right destination.
  if (Options.SpeculateBlocks &&
      foldBranchToCommonDest(BI, DTU, /*MSSAU=*/nullptr, &TTI,
                             Options.BonusInstThreshold))
    return requestResimplify();

  // We have a conditional branch to two blocks that are only reachable
  // from BI.  We know that the condbr dominates the two blocks, so see if
  // there is any identical code in the "then" and "else" blocks.  If so, we
  // can hoist it up to the branching block.
  if (BI->getSuccessor(0)->getSinglePredecessor()) {
    if (BI->getSuccessor(1)->getSinglePredecessor()) {
      if (HoistCommon &&
          hoistCommonCodeFromSuccessors(BI, !Options.HoistCommonInsts))
        return requestResimplify();

      if (BI && Options.HoistLoadsStoresWithCondFaulting &&
          isProfitableToSpeculate(BI, std::nullopt, TTI)) {
        SmallVector<Instruction *, 2> SpeculatedConditionalLoadsStores;
        auto CanSpeculateConditionalLoadsStores = [&]() {
          for (auto *Succ : successors(BB)) {
            for (Instruction &I : *Succ) {
              if (I.isTerminator()) {
                if (I.getNumSuccessors() > 1)
                  return false;
                continue;
              } else if (!isSafeCheapLoadStore(&I, TTI) ||
                         SpeculatedConditionalLoadsStores.size() ==
                             HoistLoadsStoresWithCondFaultingThreshold) {
                return false;
              }
              SpeculatedConditionalLoadsStores.push_back(&I);
            }
          }
          return !SpeculatedConditionalLoadsStores.empty();
        };

        if (CanSpeculateConditionalLoadsStores()) {
          hoistConditionalLoadsStores(BI, SpeculatedConditionalLoadsStores,
                                      std::nullopt, nullptr);
          return requestResimplify();
        }
      }
    } else {
      // If Successor #1 has multiple preds, we may be able to conditionally
      // execute Successor #0 if it branches to Successor #1.
      Instruction *Succ0TI = BI->getSuccessor(0)->getTerminator();
      if (Succ0TI->getNumSuccessors() == 1 &&
          Succ0TI->getSuccessor(0) == BI->getSuccessor(1))
        if (speculativelyExecuteBB(BI, BI->getSuccessor(0)))
          return requestResimplify();
    }
  } else if (BI->getSuccessor(1)->getSinglePredecessor()) {
    // If Successor #0 has multiple preds, we may be able to conditionally
    // execute Successor #1 if it branches to Successor #0.
    Instruction *Succ1TI = BI->getSuccessor(1)->getTerminator();
    if (Succ1TI->getNumSuccessors() == 1 &&
        Succ1TI->getSuccessor(0) == BI->getSuccessor(0))
      if (speculativelyExecuteBB(BI, BI->getSuccessor(1)))
        return requestResimplify();
  }

  // If this is a branch on something for which we know the constant value in
  // predecessors (e.g. a phi node in the current block), thread control
  // through this block.
  if (foldCondBranchOnValueKnownInPredecessor(BI))
    return requestResimplify();

  // Scan predecessor blocks for conditional branches.
  for (BasicBlock *Pred : predecessors(BB))
    if (BranchInst *PBI = dyn_cast<BranchInst>(Pred->getTerminator()))
      if (PBI != BI && PBI->isConditional())
        if (SimplifyCondBranchToCondBranch(PBI, BI, DTU, DL, TTI))
          return requestResimplify();

  // Look for diamond patterns.
  if (MergeCondStores)
    if (BasicBlock *PrevBB = allPredecessorsComeFromSameSource(BB))
      if (BranchInst *PBI = dyn_cast<BranchInst>(PrevBB->getTerminator()))
        if (PBI != BI && PBI->isConditional())
          if (mergeConditionalStores(PBI, BI, DTU, DL, TTI))
            return requestResimplify();

  // Look for nested conditional branches.
  if (mergeNestedCondBranch(BI, DTU))
    return requestResimplify();

  return false;
}

/// Check if passing a value to an instruction will cause undefined behavior.
static bool passingValueIsAlwaysUndefined(Value *V, Instruction *I, bool PtrValueMayBeModified) {
  assert(V->getType() == I->getType() && "Mismatched types");
  Constant *C = dyn_cast<Constant>(V);
  if (!C)
    return false;

  if (I->use_empty())
    return false;

  if (C->isNullValue() || isa<UndefValue>(C)) {
    // Only look at the first use we can handle, avoid hurting compile time with
    // long uselists
    auto FindUse = llvm::find_if(I->uses(), [](auto &U) {
      auto *Use = cast<Instruction>(U.getUser());
      // Change this list when we want to add new instructions.
      switch (Use->getOpcode()) {
      default:
        return false;
      case Instruction::GetElementPtr:
      case Instruction::Ret:
      case Instruction::BitCast:
      case Instruction::Load:
      case Instruction::Store:
      case Instruction::Call:
      case Instruction::CallBr:
      case Instruction::Invoke:
      case Instruction::UDiv:
      case Instruction::URem:
        // Note: signed div/rem of INT_MIN / -1 is also immediate UB, not
        // implemented to avoid code complexity as it is unclear how useful such
        // logic is.
      case Instruction::SDiv:
      case Instruction::SRem:
        return true;
      }
    });
    if (FindUse == I->use_end())
      return false;
    auto &Use = *FindUse;
    auto *User = cast<Instruction>(Use.getUser());
    // Bail out if User is not in the same BB as I or User == I or User comes
    // before I in the block. The latter two can be the case if User is a
    // PHI node.
    if (User->getParent() != I->getParent() || User == I ||
        User->comesBefore(I))
      return false;

    // Now make sure that there are no instructions in between that can alter
    // control flow (eg. calls)
    auto InstrRange =
        make_range(std::next(I->getIterator()), User->getIterator());
    if (any_of(InstrRange, [](Instruction &I) {
          return !isGuaranteedToTransferExecutionToSuccessor(&I);
        }))
      return false;

    // Look through GEPs. A load from a GEP derived from NULL is still undefined
    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(User))
      if (GEP->getPointerOperand() == I) {
        // The type of GEP may differ from the type of base pointer.
        // Bail out on vector GEPs, as they are not handled by other checks.
        if (GEP->getType()->isVectorTy())
          return false;
        // The current base address is null, there are four cases to consider:
        // getelementptr (TY, null, 0)                 -> null
        // getelementptr (TY, null, not zero)          -> may be modified
        // getelementptr inbounds (TY, null, 0)        -> null
        // getelementptr inbounds (TY, null, not zero) -> poison iff null is
        // undefined?
        if (!GEP->hasAllZeroIndices() &&
            (!GEP->isInBounds() ||
             NullPointerIsDefined(GEP->getFunction(),
                                  GEP->getPointerAddressSpace())))
          PtrValueMayBeModified = true;
        return passingValueIsAlwaysUndefined(V, GEP, PtrValueMayBeModified);
      }

    // Look through return.
    if (ReturnInst *Ret = dyn_cast<ReturnInst>(User)) {
      bool HasNoUndefAttr =
          Ret->getFunction()->hasRetAttribute(Attribute::NoUndef);
      // Return undefined to a noundef return value is undefined.
      if (isa<UndefValue>(C) && HasNoUndefAttr)
        return true;
      // Return null to a nonnull+noundef return value is undefined.
      if (C->isNullValue() && HasNoUndefAttr &&
          Ret->getFunction()->hasRetAttribute(Attribute::NonNull)) {
        return !PtrValueMayBeModified;
      }
    }

    // Load from null is undefined.
    if (LoadInst *LI = dyn_cast<LoadInst>(User))
      if (!LI->isVolatile())
        return !NullPointerIsDefined(LI->getFunction(),
                                     LI->getPointerAddressSpace());

    // Store to null is undefined.
    if (StoreInst *SI = dyn_cast<StoreInst>(User))
      if (!SI->isVolatile())
        return (!NullPointerIsDefined(SI->getFunction(),
                                      SI->getPointerAddressSpace())) &&
               SI->getPointerOperand() == I;

    // llvm.assume(false/undef) always triggers immediate UB.
    if (auto *Assume = dyn_cast<AssumeInst>(User)) {
      // Ignore assume operand bundles.
      if (I == Assume->getArgOperand(0))
        return true;
    }

    if (auto *CB = dyn_cast<CallBase>(User)) {
      if (C->isNullValue() && NullPointerIsDefined(CB->getFunction()))
        return false;
      // A call to null is undefined.
      if (CB->getCalledOperand() == I)
        return true;

      if (CB->isArgOperand(&Use)) {
        unsigned ArgIdx = CB->getArgOperandNo(&Use);
        // Passing null to a nonnnull+noundef argument is undefined.
        if (isa<ConstantPointerNull>(C) &&
            CB->paramHasNonNullAttr(ArgIdx, /*AllowUndefOrPoison=*/false))
          return !PtrValueMayBeModified;
        // Passing undef to a noundef argument is undefined.
        if (isa<UndefValue>(C) && CB->isPassingUndefUB(ArgIdx))
          return true;
      }
    }
    // Div/Rem by zero is immediate UB
    if (match(User, m_BinOp(m_Value(), m_Specific(I))) && User->isIntDivRem())
      return true;
  }
  return false;
}

/// If BB has an incoming value that will always trigger undefined behavior
/// (eg. null pointer dereference), remove the branch leading here.
static bool removeUndefIntroducingPredecessor(BasicBlock *BB,
                                              DomTreeUpdater *DTU,
                                              AssumptionCache *AC) {
  for (PHINode &PHI : BB->phis())
    for (unsigned i = 0, e = PHI.getNumIncomingValues(); i != e; ++i)
      if (passingValueIsAlwaysUndefined(PHI.getIncomingValue(i), &PHI)) {
        BasicBlock *Predecessor = PHI.getIncomingBlock(i);
        Instruction *T = Predecessor->getTerminator();
        IRBuilder<> Builder(T);
        if (BranchInst *BI = dyn_cast<BranchInst>(T)) {
          BB->removePredecessor(Predecessor);
          // Turn unconditional branches into unreachables and remove the dead
          // destination from conditional branches.
          if (BI->isUnconditional())
            Builder.CreateUnreachable();
          else {
            // Preserve guarding condition in assume, because it might not be
            // inferrable from any dominating condition.
            Value *Cond = BI->getCondition();
            CallInst *Assumption;
            if (BI->getSuccessor(0) == BB)
              Assumption = Builder.CreateAssumption(Builder.CreateNot(Cond));
            else
              Assumption = Builder.CreateAssumption(Cond);
            if (AC)
              AC->registerAssumption(cast<AssumeInst>(Assumption));
            Builder.CreateBr(BI->getSuccessor(0) == BB ? BI->getSuccessor(1)
                                                       : BI->getSuccessor(0));
          }
          BI->eraseFromParent();
          if (DTU)
            DTU->applyUpdates({{DominatorTree::Delete, Predecessor, BB}});
          return true;
        } else if (SwitchInst *SI = dyn_cast<SwitchInst>(T)) {
          // Redirect all branches leading to UB into
          // a newly created unreachable block.
          BasicBlock *Unreachable = BasicBlock::Create(
              Predecessor->getContext(), "unreachable", BB->getParent(), BB);
          Builder.SetInsertPoint(Unreachable);
          // The new block contains only one instruction: Unreachable
          Builder.CreateUnreachable();
          for (const auto &Case : SI->cases())
            if (Case.getCaseSuccessor() == BB) {
              BB->removePredecessor(Predecessor);
              Case.setSuccessor(Unreachable);
            }
          if (SI->getDefaultDest() == BB) {
            BB->removePredecessor(Predecessor);
            SI->setDefaultDest(Unreachable);
          }

          if (DTU)
            DTU->applyUpdates(
                { { DominatorTree::Insert, Predecessor, Unreachable },
                  { DominatorTree::Delete, Predecessor, BB } });
          return true;
        }
      }

  return false;
}

bool SimplifyCFGOpt::simplifyOnce(BasicBlock *BB) {
  bool Changed = false;

  assert(BB && BB->getParent() && "Block not embedded in function!");
  assert(BB->getTerminator() && "Degenerate basic block encountered!");

  // Remove basic blocks that have no predecessors (except the entry block)...
  // or that just have themself as a predecessor.  These are unreachable.
  if ((pred_empty(BB) && BB != &BB->getParent()->getEntryBlock()) ||
      BB->getSinglePredecessor() == BB) {
    LLVM_DEBUG(dbgs() << "Removing BB: \n" << *BB);
    DeleteDeadBlock(BB, DTU);
    return true;
  }

  // Check to see if we can constant propagate this terminator instruction
  // away...
  Changed |= ConstantFoldTerminator(BB, /*DeleteDeadConditions=*/true,
                                    /*TLI=*/nullptr, DTU);

  // Check for and eliminate duplicate PHI nodes in this block.
  Changed |= EliminateDuplicatePHINodes(BB);

  // Check for and remove branches that will always cause undefined behavior.
  if (removeUndefIntroducingPredecessor(BB, DTU, Options.AC))
    return requestResimplify();

  // Merge basic blocks into their predecessor if there is only one distinct
  // pred, and if there is only one distinct successor of the predecessor, and
  // if there are no PHI nodes.
  if (MergeBlockIntoPredecessor(BB, DTU))
    return true;

  if (SinkCommon && Options.SinkCommonInsts)
    if (sinkCommonCodeFromPredecessors(BB, DTU) ||
        mergeCompatibleInvokes(BB, DTU)) {
      // sinkCommonCodeFromPredecessors() does not automatically CSE PHI's,
      // so we may now how duplicate PHI's.
      // Let's rerun EliminateDuplicatePHINodes() first,
      // before foldTwoEntryPHINode() potentially converts them into select's,
      // after which we'd need a whole EarlyCSE pass run to cleanup them.
      return true;
    }

  IRBuilder<> Builder(BB);

  if (Options.SpeculateBlocks &&
      !BB->getParent()->hasFnAttribute(Attribute::OptForFuzzing)) {
    // If there is a trivial two-entry PHI node in this basic block, and we can
    // eliminate it, do so now.
    if (auto *PN = dyn_cast<PHINode>(BB->begin()))
      if (PN->getNumIncomingValues() == 2)
        if (foldTwoEntryPHINode(PN, TTI, DTU, Options.AC, DL,
                                Options.SpeculateUnpredictables))
          return true;
  }

  Instruction *Terminator = BB->getTerminator();
  Builder.SetInsertPoint(Terminator);
  switch (Terminator->getOpcode()) {
  case Instruction::Br:
    Changed |= simplifyBranch(cast<BranchInst>(Terminator), Builder);
    break;
  case Instruction::Resume:
    Changed |= simplifyResume(cast<ResumeInst>(Terminator), Builder);
    break;
  case Instruction::CleanupRet:
    Changed |= simplifyCleanupReturn(cast<CleanupReturnInst>(Terminator));
    break;
  case Instruction::Switch:
    Changed |= simplifySwitch(cast<SwitchInst>(Terminator), Builder);
    break;
  case Instruction::Unreachable:
    Changed |= simplifyUnreachable(cast<UnreachableInst>(Terminator));
    break;
  case Instruction::IndirectBr:
    Changed |= simplifyIndirectBr(cast<IndirectBrInst>(Terminator));
    break;
  }

  return Changed;
}

bool SimplifyCFGOpt::run(BasicBlock *BB) {
  bool Changed = false;

  // Repeated simplify BB as long as resimplification is requested.
  do {
    Resimplify = false;

    // Perform one round of simplifcation. Resimplify flag will be set if
    // another iteration is requested.
    Changed |= simplifyOnce(BB);
  } while (Resimplify);

  return Changed;
}

bool llvm::simplifyCFG(BasicBlock *BB, const TargetTransformInfo &TTI,
                       DomTreeUpdater *DTU, const SimplifyCFGOptions &Options,
                       ArrayRef<WeakVH> LoopHeaders) {
  return SimplifyCFGOpt(TTI, DTU, BB->getDataLayout(), LoopHeaders,
                        Options)
      .run(BB);
}
