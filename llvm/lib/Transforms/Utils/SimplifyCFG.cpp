//===- SimplifyCFG.cpp - Code to perform CFG simplification ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Peephole optimize the CFG.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/Local.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <algorithm>
#include <map>
#include <set>
using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "simplifycfg"

static cl::opt<unsigned>
PHINodeFoldingThreshold("phi-node-folding-threshold", cl::Hidden, cl::init(1),
   cl::desc("Control the amount of phi node folding to perform (default = 1)"));

static cl::opt<bool>
DupRet("simplifycfg-dup-ret", cl::Hidden, cl::init(false),
       cl::desc("Duplicate return instructions into unconditional branches"));

static cl::opt<bool>
SinkCommon("simplifycfg-sink-common", cl::Hidden, cl::init(true),
       cl::desc("Sink common instructions down to the end block"));

static cl::opt<bool> HoistCondStores(
    "simplifycfg-hoist-cond-stores", cl::Hidden, cl::init(true),
    cl::desc("Hoist conditional stores if an unconditional store precedes"));

STATISTIC(NumBitMaps, "Number of switch instructions turned into bitmaps");
STATISTIC(NumLookupTables, "Number of switch instructions turned into lookup tables");
STATISTIC(NumLookupTablesHoles, "Number of switch instructions turned into lookup tables (holes checked)");
STATISTIC(NumSinkCommons, "Number of common instructions sunk down to the end block");
STATISTIC(NumSpeculations, "Number of speculative executed instructions");

namespace {
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
  const DataLayout *const DL;
  Value *isValueEqualityComparison(TerminatorInst *TI);
  BasicBlock *GetValueEqualityComparisonCases(TerminatorInst *TI,
                               std::vector<ValueEqualityComparisonCase> &Cases);
  bool SimplifyEqualityComparisonWithOnlyPredecessor(TerminatorInst *TI,
                                                     BasicBlock *Pred,
                                                     IRBuilder<> &Builder);
  bool FoldValueComparisonIntoPredecessors(TerminatorInst *TI,
                                           IRBuilder<> &Builder);

  bool SimplifyReturn(ReturnInst *RI, IRBuilder<> &Builder);
  bool SimplifyResume(ResumeInst *RI, IRBuilder<> &Builder);
  bool SimplifyUnreachable(UnreachableInst *UI);
  bool SimplifySwitch(SwitchInst *SI, IRBuilder<> &Builder);
  bool SimplifyIndirectBr(IndirectBrInst *IBI);
  bool SimplifyUncondBranch(BranchInst *BI, IRBuilder <> &Builder);
  bool SimplifyCondBranch(BranchInst *BI, IRBuilder <>&Builder);

public:
  SimplifyCFGOpt(const TargetTransformInfo &TTI, const DataLayout *DL)
      : TTI(TTI), DL(DL) {}
  bool run(BasicBlock *BB);
};
}

/// SafeToMergeTerminators - Return true if it is safe to merge these two
/// terminator instructions together.
///
static bool SafeToMergeTerminators(TerminatorInst *SI1, TerminatorInst *SI2) {
  if (SI1 == SI2) return false;  // Can't merge with self!

  // It is not safe to merge these two switch instructions if they have a common
  // successor, and if that successor has a PHI node, and if *that* PHI node has
  // conflicting incoming values from the two switch blocks.
  BasicBlock *SI1BB = SI1->getParent();
  BasicBlock *SI2BB = SI2->getParent();
  SmallPtrSet<BasicBlock*, 16> SI1Succs(succ_begin(SI1BB), succ_end(SI1BB));

  for (succ_iterator I = succ_begin(SI2BB), E = succ_end(SI2BB); I != E; ++I)
    if (SI1Succs.count(*I))
      for (BasicBlock::iterator BBI = (*I)->begin();
           isa<PHINode>(BBI); ++BBI) {
        PHINode *PN = cast<PHINode>(BBI);
        if (PN->getIncomingValueForBlock(SI1BB) !=
            PN->getIncomingValueForBlock(SI2BB))
          return false;
      }

  return true;
}

/// isProfitableToFoldUnconditional - Return true if it is safe and profitable
/// to merge these two terminator instructions together, where SI1 is an
/// unconditional branch. PhiNodes will store all PHI nodes in common
/// successors.
///
static bool isProfitableToFoldUnconditional(BranchInst *SI1,
                                          BranchInst *SI2,
                                          Instruction *Cond,
                                          SmallVectorImpl<PHINode*> &PhiNodes) {
  if (SI1 == SI2) return false;  // Can't merge with self!
  assert(SI1->isUnconditional() && SI2->isConditional());

  // We fold the unconditional branch if we can easily update all PHI nodes in
  // common successors:
  // 1> We have a constant incoming value for the conditional branch;
  // 2> We have "Cond" as the incoming value for the unconditional branch;
  // 3> SI2->getCondition() and Cond have same operands.
  CmpInst *Ci2 = dyn_cast<CmpInst>(SI2->getCondition());
  if (!Ci2) return false;
  if (!(Cond->getOperand(0) == Ci2->getOperand(0) &&
        Cond->getOperand(1) == Ci2->getOperand(1)) &&
      !(Cond->getOperand(0) == Ci2->getOperand(1) &&
        Cond->getOperand(1) == Ci2->getOperand(0)))
    return false;

  BasicBlock *SI1BB = SI1->getParent();
  BasicBlock *SI2BB = SI2->getParent();
  SmallPtrSet<BasicBlock*, 16> SI1Succs(succ_begin(SI1BB), succ_end(SI1BB));
  for (succ_iterator I = succ_begin(SI2BB), E = succ_end(SI2BB); I != E; ++I)
    if (SI1Succs.count(*I))
      for (BasicBlock::iterator BBI = (*I)->begin();
           isa<PHINode>(BBI); ++BBI) {
        PHINode *PN = cast<PHINode>(BBI);
        if (PN->getIncomingValueForBlock(SI1BB) != Cond ||
            !isa<ConstantInt>(PN->getIncomingValueForBlock(SI2BB)))
          return false;
        PhiNodes.push_back(PN);
      }
  return true;
}

/// AddPredecessorToBlock - Update PHI nodes in Succ to indicate that there will
/// now be entries in it from the 'NewPred' block.  The values that will be
/// flowing into the PHI nodes will be the same as those coming in from
/// ExistPred, an existing predecessor of Succ.
static void AddPredecessorToBlock(BasicBlock *Succ, BasicBlock *NewPred,
                                  BasicBlock *ExistPred) {
  if (!isa<PHINode>(Succ->begin())) return; // Quick exit if nothing to do

  PHINode *PN;
  for (BasicBlock::iterator I = Succ->begin();
       (PN = dyn_cast<PHINode>(I)); ++I)
    PN->addIncoming(PN->getIncomingValueForBlock(ExistPred), NewPred);
}

/// ComputeSpeculationCost - Compute an abstract "cost" of speculating the
/// given instruction, which is assumed to be safe to speculate. 1 means
/// cheap, 2 means less cheap, and UINT_MAX means prohibitively expensive.
static unsigned ComputeSpeculationCost(const User *I, const DataLayout *DL) {
  assert(isSafeToSpeculativelyExecute(I, DL) &&
         "Instruction is not safe to speculatively execute!");
  switch (Operator::getOpcode(I)) {
  default:
    // In doubt, be conservative.
    return UINT_MAX;
  case Instruction::GetElementPtr:
    // GEPs are cheap if all indices are constant.
    if (!cast<GEPOperator>(I)->hasAllConstantIndices())
      return UINT_MAX;
    return 1;
  case Instruction::ExtractValue:
  case Instruction::Load:
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
  case Instruction::ICmp:
  case Instruction::Trunc:
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::BitCast:
  case Instruction::ExtractElement:
  case Instruction::InsertElement:
    return 1; // These are all cheap.

  case Instruction::Call:
  case Instruction::Select:
    return 2;
  }
}

/// DominatesMergePoint - If we have a merge point of an "if condition" as
/// accepted above, return true if the specified value dominates the block.  We
/// don't handle the true generality of domination here, just a special case
/// which works well enough for us.
///
/// If AggressiveInsts is non-null, and if V does not dominate BB, we check to
/// see if V (which must be an instruction) and its recursive operands
/// that do not dominate BB have a combined cost lower than CostRemaining and
/// are non-trapping.  If both are true, the instruction is inserted into the
/// set and true is returned.
///
/// The cost for most non-trapping instructions is defined as 1 except for
/// Select whose cost is 2.
///
/// After this function returns, CostRemaining is decreased by the cost of
/// V plus its non-dominating operands.  If that cost is greater than
/// CostRemaining, false is returned and CostRemaining is undefined.
static bool DominatesMergePoint(Value *V, BasicBlock *BB,
                                SmallPtrSet<Instruction*, 4> *AggressiveInsts,
                                unsigned &CostRemaining,
                                const DataLayout *DL) {
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I) {
    // Non-instructions all dominate instructions, but not all constantexprs
    // can be executed unconditionally.
    if (ConstantExpr *C = dyn_cast<ConstantExpr>(V))
      if (C->canTrap())
        return false;
    return true;
  }
  BasicBlock *PBB = I->getParent();

  // We don't want to allow weird loops that might have the "if condition" in
  // the bottom of this block.
  if (PBB == BB) return false;

  // If this instruction is defined in a block that contains an unconditional
  // branch to BB, then it must be in the 'conditional' part of the "if
  // statement".  If not, it definitely dominates the region.
  BranchInst *BI = dyn_cast<BranchInst>(PBB->getTerminator());
  if (!BI || BI->isConditional() || BI->getSuccessor(0) != BB)
    return true;

  // If we aren't allowing aggressive promotion anymore, then don't consider
  // instructions in the 'if region'.
  if (!AggressiveInsts) return false;

  // If we have seen this instruction before, don't count it again.
  if (AggressiveInsts->count(I)) return true;

  // Okay, it looks like the instruction IS in the "condition".  Check to
  // see if it's a cheap instruction to unconditionally compute, and if it
  // only uses stuff defined outside of the condition.  If so, hoist it out.
  if (!isSafeToSpeculativelyExecute(I, DL))
    return false;

  unsigned Cost = ComputeSpeculationCost(I, DL);

  if (Cost > CostRemaining)
    return false;

  CostRemaining -= Cost;

  // Okay, we can only really hoist these out if their operands do
  // not take us over the cost threshold.
  for (User::op_iterator i = I->op_begin(), e = I->op_end(); i != e; ++i)
    if (!DominatesMergePoint(*i, BB, AggressiveInsts, CostRemaining, DL))
      return false;
  // Okay, it's safe to do this!  Remember this instruction.
  AggressiveInsts->insert(I);
  return true;
}

/// GetConstantInt - Extract ConstantInt from value, looking through IntToPtr
/// and PointerNullValue. Return NULL if value is not a constant int.
static ConstantInt *GetConstantInt(Value *V, const DataLayout *DL) {
  // Normal constant int.
  ConstantInt *CI = dyn_cast<ConstantInt>(V);
  if (CI || !DL || !isa<Constant>(V) || !V->getType()->isPointerTy())
    return CI;

  // This is some kind of pointer constant. Turn it into a pointer-sized
  // ConstantInt if possible.
  IntegerType *PtrTy = cast<IntegerType>(DL->getIntPtrType(V->getType()));

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
          return cast<ConstantInt>
            (ConstantExpr::getIntegerCast(CI, PtrTy, /*isSigned=*/false));
      }
  return nullptr;
}

/// GatherConstantCompares - Given a potentially 'or'd or 'and'd together
/// collection of icmp eq/ne instructions that compare a value against a
/// constant, return the value being compared, and stick the constant into the
/// Values vector.
static Value *
GatherConstantCompares(Value *V, std::vector<ConstantInt*> &Vals, Value *&Extra,
                       const DataLayout *DL, bool isEQ, unsigned &UsedICmps) {
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I) return nullptr;

  // If this is an icmp against a constant, handle this as one of the cases.
  if (ICmpInst *ICI = dyn_cast<ICmpInst>(I)) {
    if (ConstantInt *C = GetConstantInt(I->getOperand(1), DL)) {
      Value *RHSVal;
      ConstantInt *RHSC;

      if (ICI->getPredicate() == (isEQ ? ICmpInst::ICMP_EQ:ICmpInst::ICMP_NE)) {
        // (x & ~2^x) == y --> x == y || x == y|2^x
        // This undoes a transformation done by instcombine to fuse 2 compares.
        if (match(ICI->getOperand(0),
                  m_And(m_Value(RHSVal), m_ConstantInt(RHSC)))) {
          APInt Not = ~RHSC->getValue();
          if (Not.isPowerOf2()) {
            Vals.push_back(C);
            Vals.push_back(
                ConstantInt::get(C->getContext(), C->getValue() | Not));
            UsedICmps++;
            return RHSVal;
          }
        }

        UsedICmps++;
        Vals.push_back(C);
        return I->getOperand(0);
      }

      // If we have "x ult 3" comparison, for example, then we can add 0,1,2 to
      // the set.
      ConstantRange Span =
        ConstantRange::makeICmpRegion(ICI->getPredicate(), C->getValue());

      // Shift the range if the compare is fed by an add. This is the range
      // compare idiom as emitted by instcombine.
      bool hasAdd =
          match(I->getOperand(0), m_Add(m_Value(RHSVal), m_ConstantInt(RHSC)));
      if (hasAdd)
        Span = Span.subtract(RHSC->getValue());

      // If this is an and/!= check then we want to optimize "x ugt 2" into
      // x != 0 && x != 1.
      if (!isEQ)
        Span = Span.inverse();

      // If there are a ton of values, we don't want to make a ginormous switch.
      if (Span.getSetSize().ugt(8) || Span.isEmptySet())
        return nullptr;

      for (APInt Tmp = Span.getLower(); Tmp != Span.getUpper(); ++Tmp)
        Vals.push_back(ConstantInt::get(V->getContext(), Tmp));
      UsedICmps++;
      return hasAdd ? RHSVal : I->getOperand(0);
    }
    return nullptr;
  }

  // Otherwise, we can only handle an | or &, depending on isEQ.
  if (I->getOpcode() != (isEQ ? Instruction::Or : Instruction::And))
    return nullptr;

  unsigned NumValsBeforeLHS = Vals.size();
  unsigned UsedICmpsBeforeLHS = UsedICmps;
  if (Value *LHS = GatherConstantCompares(I->getOperand(0), Vals, Extra, DL,
                                          isEQ, UsedICmps)) {
    unsigned NumVals = Vals.size();
    unsigned UsedICmpsBeforeRHS = UsedICmps;
    if (Value *RHS = GatherConstantCompares(I->getOperand(1), Vals, Extra, DL,
                                            isEQ, UsedICmps)) {
      if (LHS == RHS)
        return LHS;
      Vals.resize(NumVals);
      UsedICmps = UsedICmpsBeforeRHS;
    }

    // The RHS of the or/and can't be folded in and we haven't used "Extra" yet,
    // set it and return success.
    if (Extra == nullptr || Extra == I->getOperand(1)) {
      Extra = I->getOperand(1);
      return LHS;
    }

    Vals.resize(NumValsBeforeLHS);
    UsedICmps = UsedICmpsBeforeLHS;
    return nullptr;
  }

  // If the LHS can't be folded in, but Extra is available and RHS can, try to
  // use LHS as Extra.
  if (Extra == nullptr || Extra == I->getOperand(0)) {
    Value *OldExtra = Extra;
    Extra = I->getOperand(0);
    if (Value *RHS = GatherConstantCompares(I->getOperand(1), Vals, Extra, DL,
                                            isEQ, UsedICmps))
      return RHS;
    assert(Vals.size() == NumValsBeforeLHS);
    Extra = OldExtra;
  }

  return nullptr;
}

static void EraseTerminatorInstAndDCECond(TerminatorInst *TI) {
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
  if (Cond) RecursivelyDeleteTriviallyDeadInstructions(Cond);
}

/// isValueEqualityComparison - Return true if the specified terminator checks
/// to see if a value is equal to constant integer value.
Value *SimplifyCFGOpt::isValueEqualityComparison(TerminatorInst *TI) {
  Value *CV = nullptr;
  if (SwitchInst *SI = dyn_cast<SwitchInst>(TI)) {
    // Do not permit merging of large switch instructions into their
    // predecessors unless there is only one predecessor.
    if (SI->getNumSuccessors()*std::distance(pred_begin(SI->getParent()),
                                             pred_end(SI->getParent())) <= 128)
      CV = SI->getCondition();
  } else if (BranchInst *BI = dyn_cast<BranchInst>(TI))
    if (BI->isConditional() && BI->getCondition()->hasOneUse())
      if (ICmpInst *ICI = dyn_cast<ICmpInst>(BI->getCondition()))
        if (ICI->isEquality() && GetConstantInt(ICI->getOperand(1), DL))
          CV = ICI->getOperand(0);

  // Unwrap any lossless ptrtoint cast.
  if (DL && CV) {
    if (PtrToIntInst *PTII = dyn_cast<PtrToIntInst>(CV)) {
      Value *Ptr = PTII->getPointerOperand();
      if (PTII->getType() == DL->getIntPtrType(Ptr->getType()))
        CV = Ptr;
    }
  }
  return CV;
}

/// GetValueEqualityComparisonCases - Given a value comparison instruction,
/// decode all of the 'cases' that it represents and return the 'default' block.
BasicBlock *SimplifyCFGOpt::
GetValueEqualityComparisonCases(TerminatorInst *TI,
                                std::vector<ValueEqualityComparisonCase>
                                                                       &Cases) {
  if (SwitchInst *SI = dyn_cast<SwitchInst>(TI)) {
    Cases.reserve(SI->getNumCases());
    for (SwitchInst::CaseIt i = SI->case_begin(), e = SI->case_end(); i != e; ++i)
      Cases.push_back(ValueEqualityComparisonCase(i.getCaseValue(),
                                                  i.getCaseSuccessor()));
    return SI->getDefaultDest();
  }

  BranchInst *BI = cast<BranchInst>(TI);
  ICmpInst *ICI = cast<ICmpInst>(BI->getCondition());
  BasicBlock *Succ = BI->getSuccessor(ICI->getPredicate() == ICmpInst::ICMP_NE);
  Cases.push_back(ValueEqualityComparisonCase(GetConstantInt(ICI->getOperand(1),
                                                             DL),
                                              Succ));
  return BI->getSuccessor(ICI->getPredicate() == ICmpInst::ICMP_EQ);
}


/// EliminateBlockCases - Given a vector of bb/value pairs, remove any entries
/// in the list that match the specified block.
static void EliminateBlockCases(BasicBlock *BB,
                              std::vector<ValueEqualityComparisonCase> &Cases) {
  Cases.erase(std::remove(Cases.begin(), Cases.end(), BB), Cases.end());
}

/// ValuesOverlap - Return true if there are any keys in C1 that exist in C2 as
/// well.
static bool
ValuesOverlap(std::vector<ValueEqualityComparisonCase> &C1,
              std::vector<ValueEqualityComparisonCase > &C2) {
  std::vector<ValueEqualityComparisonCase> *V1 = &C1, *V2 = &C2;

  // Make V1 be smaller than V2.
  if (V1->size() > V2->size())
    std::swap(V1, V2);

  if (V1->size() == 0) return false;
  if (V1->size() == 1) {
    // Just scan V2.
    ConstantInt *TheVal = (*V1)[0].Value;
    for (unsigned i = 0, e = V2->size(); i != e; ++i)
      if (TheVal == (*V2)[i].Value)
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

/// SimplifyEqualityComparisonWithOnlyPredecessor - If TI is known to be a
/// terminator instruction and its block is known to only have a single
/// predecessor block, check to see if that predecessor is also a value
/// comparison with the same value, and if that comparison determines the
/// outcome of this comparison.  If so, simplify TI.  This does a very limited
/// form of jump threading.
bool SimplifyCFGOpt::
SimplifyEqualityComparisonWithOnlyPredecessor(TerminatorInst *TI,
                                              BasicBlock *Pred,
                                              IRBuilder<> &Builder) {
  Value *PredVal = isValueEqualityComparison(Pred->getTerminator());
  if (!PredVal) return false;  // Not a value comparison in predecessor.

  Value *ThisVal = isValueEqualityComparison(TI);
  assert(ThisVal && "This isn't a value comparison!!");
  if (ThisVal != PredVal) return false;  // Different predicates.

  // TODO: Preserve branch weight metadata, similarly to how
  // FoldValueComparisonIntoPredecessors preserves it.

  // Find out information about when control will move from Pred to TI's block.
  std::vector<ValueEqualityComparisonCase> PredCases;
  BasicBlock *PredDef = GetValueEqualityComparisonCases(Pred->getTerminator(),
                                                        PredCases);
  EliminateBlockCases(PredDef, PredCases);  // Remove default from cases.

  // Find information about how control leaves this block.
  std::vector<ValueEqualityComparisonCase> ThisCases;
  BasicBlock *ThisDef = GetValueEqualityComparisonCases(TI, ThisCases);
  EliminateBlockCases(ThisDef, ThisCases);  // Remove default from cases.

  // If TI's block is the default block from Pred's comparison, potentially
  // simplify TI based on this knowledge.
  if (PredDef == TI->getParent()) {
    // If we are here, we know that the value is none of those cases listed in
    // PredCases.  If there are any cases in ThisCases that are in PredCases, we
    // can simplify TI.
    if (!ValuesOverlap(PredCases, ThisCases))
      return false;

    if (isa<BranchInst>(TI)) {
      // Okay, one of the successors of this condbr is dead.  Convert it to a
      // uncond br.
      assert(ThisCases.size() == 1 && "Branch can only have one case!");
      // Insert the new branch.
      Instruction *NI = Builder.CreateBr(ThisDef);
      (void) NI;

      // Remove PHI node entries for the dead edge.
      ThisCases[0].Dest->removePredecessor(TI->getParent());

      DEBUG(dbgs() << "Threading pred instr: " << *Pred->getTerminator()
           << "Through successor TI: " << *TI << "Leaving: " << *NI << "\n");

      EraseTerminatorInstAndDCECond(TI);
      return true;
    }

    SwitchInst *SI = cast<SwitchInst>(TI);
    // Okay, TI has cases that are statically dead, prune them away.
    SmallPtrSet<Constant*, 16> DeadCases;
    for (unsigned i = 0, e = PredCases.size(); i != e; ++i)
      DeadCases.insert(PredCases[i].Value);

    DEBUG(dbgs() << "Threading pred instr: " << *Pred->getTerminator()
                 << "Through successor TI: " << *TI);

    // Collect branch weights into a vector.
    SmallVector<uint32_t, 8> Weights;
    MDNode* MD = SI->getMetadata(LLVMContext::MD_prof);
    bool HasWeight = MD && (MD->getNumOperands() == 2 + SI->getNumCases());
    if (HasWeight)
      for (unsigned MD_i = 1, MD_e = MD->getNumOperands(); MD_i < MD_e;
           ++MD_i) {
        ConstantInt* CI = dyn_cast<ConstantInt>(MD->getOperand(MD_i));
        assert(CI);
        Weights.push_back(CI->getValue().getZExtValue());
      }
    for (SwitchInst::CaseIt i = SI->case_end(), e = SI->case_begin(); i != e;) {
      --i;
      if (DeadCases.count(i.getCaseValue())) {
        if (HasWeight) {
          std::swap(Weights[i.getCaseIndex()+1], Weights.back());
          Weights.pop_back();
        }
        i.getCaseSuccessor()->removePredecessor(TI->getParent());
        SI->removeCase(i);
      }
    }
    if (HasWeight && Weights.size() >= 2)
      SI->setMetadata(LLVMContext::MD_prof,
                      MDBuilder(SI->getParent()->getContext()).
                      createBranchWeights(Weights));

    DEBUG(dbgs() << "Leaving: " << *TI << "\n");
    return true;
  }

  // Otherwise, TI's block must correspond to some matched value.  Find out
  // which value (or set of values) this is.
  ConstantInt *TIV = nullptr;
  BasicBlock *TIBB = TI->getParent();
  for (unsigned i = 0, e = PredCases.size(); i != e; ++i)
    if (PredCases[i].Dest == TIBB) {
      if (TIV)
        return false;  // Cannot handle multiple values coming to this block.
      TIV = PredCases[i].Value;
    }
  assert(TIV && "No edge from pred to succ?");

  // Okay, we found the one constant that our value can be if we get into TI's
  // BB.  Find out which successor will unconditionally be branched to.
  BasicBlock *TheRealDest = nullptr;
  for (unsigned i = 0, e = ThisCases.size(); i != e; ++i)
    if (ThisCases[i].Value == TIV) {
      TheRealDest = ThisCases[i].Dest;
      break;
    }

  // If not handled by any explicit cases, it is handled by the default case.
  if (!TheRealDest) TheRealDest = ThisDef;

  // Remove PHI node entries for dead edges.
  BasicBlock *CheckEdge = TheRealDest;
  for (succ_iterator SI = succ_begin(TIBB), e = succ_end(TIBB); SI != e; ++SI)
    if (*SI != CheckEdge)
      (*SI)->removePredecessor(TIBB);
    else
      CheckEdge = nullptr;

  // Insert the new branch.
  Instruction *NI = Builder.CreateBr(TheRealDest);
  (void) NI;

  DEBUG(dbgs() << "Threading pred instr: " << *Pred->getTerminator()
            << "Through successor TI: " << *TI << "Leaving: " << *NI << "\n");

  EraseTerminatorInstAndDCECond(TI);
  return true;
}

namespace {
  /// ConstantIntOrdering - This class implements a stable ordering of constant
  /// integers that does not depend on their address.  This is important for
  /// applications that sort ConstantInt's to ensure uniqueness.
  struct ConstantIntOrdering {
    bool operator()(const ConstantInt *LHS, const ConstantInt *RHS) const {
      return LHS->getValue().ult(RHS->getValue());
    }
  };
}

static int ConstantIntSortPredicate(ConstantInt *const *P1,
                                    ConstantInt *const *P2) {
  const ConstantInt *LHS = *P1;
  const ConstantInt *RHS = *P2;
  if (LHS->getValue().ult(RHS->getValue()))
    return 1;
  if (LHS->getValue() == RHS->getValue())
    return 0;
  return -1;
}

static inline bool HasBranchWeights(const Instruction* I) {
  MDNode* ProfMD = I->getMetadata(LLVMContext::MD_prof);
  if (ProfMD && ProfMD->getOperand(0))
    if (MDString* MDS = dyn_cast<MDString>(ProfMD->getOperand(0)))
      return MDS->getString().equals("branch_weights");

  return false;
}

/// Get Weights of a given TerminatorInst, the default weight is at the front
/// of the vector. If TI is a conditional eq, we need to swap the branch-weight
/// metadata.
static void GetBranchWeights(TerminatorInst *TI,
                             SmallVectorImpl<uint64_t> &Weights) {
  MDNode* MD = TI->getMetadata(LLVMContext::MD_prof);
  assert(MD);
  for (unsigned i = 1, e = MD->getNumOperands(); i < e; ++i) {
    ConstantInt *CI = cast<ConstantInt>(MD->getOperand(i));
    Weights.push_back(CI->getValue().getZExtValue());
  }

  // If TI is a conditional eq, the default case is the false case,
  // and the corresponding branch-weight data is at index 2. We swap the
  // default weight to be the first entry.
  if (BranchInst* BI = dyn_cast<BranchInst>(TI)) {
    assert(Weights.size() == 2);
    ICmpInst *ICI = cast<ICmpInst>(BI->getCondition());
    if (ICI->getPredicate() == ICmpInst::ICMP_EQ)
      std::swap(Weights.front(), Weights.back());
  }
}

/// Keep halving the weights until all can fit in uint32_t.
static void FitWeights(MutableArrayRef<uint64_t> Weights) {
  uint64_t Max = *std::max_element(Weights.begin(), Weights.end());
  if (Max > UINT_MAX) {
    unsigned Offset = 32 - countLeadingZeros(Max);
    for (uint64_t &I : Weights)
      I >>= Offset;
  }
}

/// FoldValueComparisonIntoPredecessors - The specified terminator is a value
/// equality comparison instruction (either a switch or a branch on "X == c").
/// See if any of the predecessors of the terminator block are value comparisons
/// on the same value.  If so, and if safe to do so, fold them together.
bool SimplifyCFGOpt::FoldValueComparisonIntoPredecessors(TerminatorInst *TI,
                                                         IRBuilder<> &Builder) {
  BasicBlock *BB = TI->getParent();
  Value *CV = isValueEqualityComparison(TI);  // CondVal
  assert(CV && "Not a comparison?");
  bool Changed = false;

  SmallVector<BasicBlock*, 16> Preds(pred_begin(BB), pred_end(BB));
  while (!Preds.empty()) {
    BasicBlock *Pred = Preds.pop_back_val();

    // See if the predecessor is a comparison with the same value.
    TerminatorInst *PTI = Pred->getTerminator();
    Value *PCV = isValueEqualityComparison(PTI);  // PredCondVal

    if (PCV == CV && SafeToMergeTerminators(TI, PTI)) {
      // Figure out which 'cases' to copy from SI to PSI.
      std::vector<ValueEqualityComparisonCase> BBCases;
      BasicBlock *BBDefault = GetValueEqualityComparisonCases(TI, BBCases);

      std::vector<ValueEqualityComparisonCase> PredCases;
      BasicBlock *PredDefault = GetValueEqualityComparisonCases(PTI, PredCases);

      // Based on whether the default edge from PTI goes to BB or not, fill in
      // PredCases and PredDefault with the new switch cases we would like to
      // build.
      SmallVector<BasicBlock*, 8> NewSuccessors;

      // Update the branch weight metadata along the way
      SmallVector<uint64_t, 8> Weights;
      bool PredHasWeights = HasBranchWeights(PTI);
      bool SuccHasWeights = HasBranchWeights(TI);

      if (PredHasWeights) {
        GetBranchWeights(PTI, Weights);
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
        GetBranchWeights(TI, SuccWeights);
        // branch-weight metadata is inconsistent here.
        if (SuccWeights.size() != 1 + BBCases.size())
          PredHasWeights = SuccHasWeights = false;
      } else if (PredHasWeights)
        SuccWeights.assign(1 + BBCases.size(), 1);

      if (PredDefault == BB) {
        // If this is the default destination from PTI, only the edges in TI
        // that don't occur in PTI, or that branch to BB will be activated.
        std::set<ConstantInt*, ConstantIntOrdering> PTIHandled;
        for (unsigned i = 0, e = PredCases.size(); i != e; ++i)
          if (PredCases[i].Dest != BB)
            PTIHandled.insert(PredCases[i].Value);
          else {
            // The default destination is BB, we don't need explicit targets.
            std::swap(PredCases[i], PredCases.back());

            if (PredHasWeights || SuccHasWeights) {
              // Increase weight for the default case.
              Weights[0] += Weights[i+1];
              std::swap(Weights[i+1], Weights.back());
              Weights.pop_back();
            }

            PredCases.pop_back();
            --i; --e;
          }

        // Reconstruct the new switch statement we will be building.
        if (PredDefault != BBDefault) {
          PredDefault->removePredecessor(Pred);
          PredDefault = BBDefault;
          NewSuccessors.push_back(BBDefault);
        }

        unsigned CasesFromPred = Weights.size();
        uint64_t ValidTotalSuccWeight = 0;
        for (unsigned i = 0, e = BBCases.size(); i != e; ++i)
          if (!PTIHandled.count(BBCases[i].Value) &&
              BBCases[i].Dest != BBDefault) {
            PredCases.push_back(BBCases[i]);
            NewSuccessors.push_back(BBCases[i].Dest);
            if (SuccHasWeights || PredHasWeights) {
              // The default weight is at index 0, so weight for the ith case
              // should be at index i+1. Scale the cases from successor by
              // PredDefaultWeight (Weights[0]).
              Weights.push_back(Weights[0] * SuccWeights[i+1]);
              ValidTotalSuccWeight += SuccWeights[i+1];
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
        std::set<ConstantInt*, ConstantIntOrdering> PTIHandled;
        std::map<ConstantInt*, uint64_t> WeightsForHandled;
        for (unsigned i = 0, e = PredCases.size(); i != e; ++i)
          if (PredCases[i].Dest == BB) {
            PTIHandled.insert(PredCases[i].Value);

            if (PredHasWeights || SuccHasWeights) {
              WeightsForHandled[PredCases[i].Value] = Weights[i+1];
              std::swap(Weights[i+1], Weights.back());
              Weights.pop_back();
            }

            std::swap(PredCases[i], PredCases.back());
            PredCases.pop_back();
            --i; --e;
          }

        // Okay, now we know which constants were sent to BB from the
        // predecessor.  Figure out where they will all go now.
        for (unsigned i = 0, e = BBCases.size(); i != e; ++i)
          if (PTIHandled.count(BBCases[i].Value)) {
            // If this is one we are capable of getting...
            if (PredHasWeights || SuccHasWeights)
              Weights.push_back(WeightsForHandled[BBCases[i].Value]);
            PredCases.push_back(BBCases[i]);
            NewSuccessors.push_back(BBCases[i].Dest);
            PTIHandled.erase(BBCases[i].Value);// This constant is taken care of
          }

        // If there are any constants vectored to BB that TI doesn't handle,
        // they must go to the default destination of TI.
        for (std::set<ConstantInt*, ConstantIntOrdering>::iterator I =
                                    PTIHandled.begin(),
               E = PTIHandled.end(); I != E; ++I) {
          if (PredHasWeights || SuccHasWeights)
            Weights.push_back(WeightsForHandled[*I]);
          PredCases.push_back(ValueEqualityComparisonCase(*I, BBDefault));
          NewSuccessors.push_back(BBDefault);
        }
      }

      // Okay, at this point, we know which new successor Pred will get.  Make
      // sure we update the number of entries in the PHI nodes for these
      // successors.
      for (unsigned i = 0, e = NewSuccessors.size(); i != e; ++i)
        AddPredecessorToBlock(NewSuccessors[i], Pred, BB);

      Builder.SetInsertPoint(PTI);
      // Convert pointer to int before we switch.
      if (CV->getType()->isPointerTy()) {
        assert(DL && "Cannot switch on pointer without DataLayout");
        CV = Builder.CreatePtrToInt(CV, DL->getIntPtrType(CV->getType()),
                                    "magicptr");
      }

      // Now that the successors are updated, create the new Switch instruction.
      SwitchInst *NewSI = Builder.CreateSwitch(CV, PredDefault,
                                               PredCases.size());
      NewSI->setDebugLoc(PTI->getDebugLoc());
      for (unsigned i = 0, e = PredCases.size(); i != e; ++i)
        NewSI->addCase(PredCases[i].Value, PredCases[i].Dest);

      if (PredHasWeights || SuccHasWeights) {
        // Halve the weights if any of them cannot fit in an uint32_t
        FitWeights(Weights);

        SmallVector<uint32_t, 8> MDWeights(Weights.begin(), Weights.end());

        NewSI->setMetadata(LLVMContext::MD_prof,
                           MDBuilder(BB->getContext()).
                           createBranchWeights(MDWeights));
      }

      EraseTerminatorInstAndDCECond(PTI);

      // Okay, last check.  If BB is still a successor of PSI, then we must
      // have an infinite loop case.  If so, add an infinitely looping block
      // to handle the case to preserve the behavior of the code.
      BasicBlock *InfLoopBlock = nullptr;
      for (unsigned i = 0, e = NewSI->getNumSuccessors(); i != e; ++i)
        if (NewSI->getSuccessor(i) == BB) {
          if (!InfLoopBlock) {
            // Insert it at the end of the function, because it's either code,
            // or it won't matter if it's hot. :)
            InfLoopBlock = BasicBlock::Create(BB->getContext(),
                                              "infloop", BB->getParent());
            BranchInst::Create(InfLoopBlock, InfLoopBlock);
          }
          NewSI->setSuccessor(i, InfLoopBlock);
        }

      Changed = true;
    }
  }
  return Changed;
}

// isSafeToHoistInvoke - If we would need to insert a select that uses the
// value of this invoke (comments in HoistThenElseCodeToIf explain why we
// would need to do this), we can't hoist the invoke, as there is nowhere
// to put the select in this case.
static bool isSafeToHoistInvoke(BasicBlock *BB1, BasicBlock *BB2,
                                Instruction *I1, Instruction *I2) {
  for (succ_iterator SI = succ_begin(BB1), E = succ_end(BB1); SI != E; ++SI) {
    PHINode *PN;
    for (BasicBlock::iterator BBI = SI->begin();
         (PN = dyn_cast<PHINode>(BBI)); ++BBI) {
      Value *BB1V = PN->getIncomingValueForBlock(BB1);
      Value *BB2V = PN->getIncomingValueForBlock(BB2);
      if (BB1V != BB2V && (BB1V==I1 || BB2V==I2)) {
        return false;
      }
    }
  }
  return true;
}

/// HoistThenElseCodeToIf - Given a conditional branch that goes to BB1 and
/// BB2, hoist any common code in the two blocks up into the branch block.  The
/// caller of this function guarantees that BI's block dominates BB1 and BB2.
static bool HoistThenElseCodeToIf(BranchInst *BI, const DataLayout *DL) {
  // This does very trivial matching, with limited scanning, to find identical
  // instructions in the two blocks.  In particular, we don't want to get into
  // O(M*N) situations here where M and N are the sizes of BB1 and BB2.  As
  // such, we currently just scan for obviously identical instructions in an
  // identical order.
  BasicBlock *BB1 = BI->getSuccessor(0);  // The true destination.
  BasicBlock *BB2 = BI->getSuccessor(1);  // The false destination

  BasicBlock::iterator BB1_Itr = BB1->begin();
  BasicBlock::iterator BB2_Itr = BB2->begin();

  Instruction *I1 = BB1_Itr++, *I2 = BB2_Itr++;
  // Skip debug info if it is not identical.
  DbgInfoIntrinsic *DBI1 = dyn_cast<DbgInfoIntrinsic>(I1);
  DbgInfoIntrinsic *DBI2 = dyn_cast<DbgInfoIntrinsic>(I2);
  if (!DBI1 || !DBI2 || !DBI1->isIdenticalToWhenDefined(DBI2)) {
    while (isa<DbgInfoIntrinsic>(I1))
      I1 = BB1_Itr++;
    while (isa<DbgInfoIntrinsic>(I2))
      I2 = BB2_Itr++;
  }
  if (isa<PHINode>(I1) || !I1->isIdenticalToWhenDefined(I2) ||
      (isa<InvokeInst>(I1) && !isSafeToHoistInvoke(BB1, BB2, I1, I2)))
    return false;

  BasicBlock *BIParent = BI->getParent();

  bool Changed = false;
  do {
    // If we are hoisting the terminator instruction, don't move one (making a
    // broken BB), instead clone it, and remove BI.
    if (isa<TerminatorInst>(I1))
      goto HoistTerminator;

    // For a normal instruction, we just move one to right before the branch,
    // then replace all uses of the other with the first.  Finally, we remove
    // the now redundant second instruction.
    BIParent->getInstList().splice(BI, BB1->getInstList(), I1);
    if (!I2->use_empty())
      I2->replaceAllUsesWith(I1);
    I1->intersectOptionalDataWith(I2);
    I2->eraseFromParent();
    Changed = true;

    I1 = BB1_Itr++;
    I2 = BB2_Itr++;
    // Skip debug info if it is not identical.
    DbgInfoIntrinsic *DBI1 = dyn_cast<DbgInfoIntrinsic>(I1);
    DbgInfoIntrinsic *DBI2 = dyn_cast<DbgInfoIntrinsic>(I2);
    if (!DBI1 || !DBI2 || !DBI1->isIdenticalToWhenDefined(DBI2)) {
      while (isa<DbgInfoIntrinsic>(I1))
        I1 = BB1_Itr++;
      while (isa<DbgInfoIntrinsic>(I2))
        I2 = BB2_Itr++;
    }
  } while (I1->isIdenticalToWhenDefined(I2));

  return true;

HoistTerminator:
  // It may not be possible to hoist an invoke.
  if (isa<InvokeInst>(I1) && !isSafeToHoistInvoke(BB1, BB2, I1, I2))
    return Changed;

  for (succ_iterator SI = succ_begin(BB1), E = succ_end(BB1); SI != E; ++SI) {
    PHINode *PN;
    for (BasicBlock::iterator BBI = SI->begin();
         (PN = dyn_cast<PHINode>(BBI)); ++BBI) {
      Value *BB1V = PN->getIncomingValueForBlock(BB1);
      Value *BB2V = PN->getIncomingValueForBlock(BB2);
      if (BB1V == BB2V)
        continue;

      if (isa<ConstantExpr>(BB1V) && !isSafeToSpeculativelyExecute(BB1V, DL))
        return Changed;
      if (isa<ConstantExpr>(BB2V) && !isSafeToSpeculativelyExecute(BB2V, DL))
        return Changed;
    }
  }

  // Okay, it is safe to hoist the terminator.
  Instruction *NT = I1->clone();
  BIParent->getInstList().insert(BI, NT);
  if (!NT->getType()->isVoidTy()) {
    I1->replaceAllUsesWith(NT);
    I2->replaceAllUsesWith(NT);
    NT->takeName(I1);
  }

  IRBuilder<true, NoFolder> Builder(NT);
  // Hoisting one of the terminators from our successor is a great thing.
  // Unfortunately, the successors of the if/else blocks may have PHI nodes in
  // them.  If they do, all PHI entries for BB1/BB2 must agree for all PHI
  // nodes, so we insert select instruction to compute the final result.
  std::map<std::pair<Value*,Value*>, SelectInst*> InsertedSelects;
  for (succ_iterator SI = succ_begin(BB1), E = succ_end(BB1); SI != E; ++SI) {
    PHINode *PN;
    for (BasicBlock::iterator BBI = SI->begin();
         (PN = dyn_cast<PHINode>(BBI)); ++BBI) {
      Value *BB1V = PN->getIncomingValueForBlock(BB1);
      Value *BB2V = PN->getIncomingValueForBlock(BB2);
      if (BB1V == BB2V) continue;

      // These values do not agree.  Insert a select instruction before NT
      // that determines the right value.
      SelectInst *&SI = InsertedSelects[std::make_pair(BB1V, BB2V)];
      if (!SI)
        SI = cast<SelectInst>
          (Builder.CreateSelect(BI->getCondition(), BB1V, BB2V,
                                BB1V->getName()+"."+BB2V->getName()));

      // Make the PHI node use the select for all incoming values for BB1/BB2
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
        if (PN->getIncomingBlock(i) == BB1 || PN->getIncomingBlock(i) == BB2)
          PN->setIncomingValue(i, SI);
    }
  }

  // Update any PHI nodes in our new successors.
  for (succ_iterator SI = succ_begin(BB1), E = succ_end(BB1); SI != E; ++SI)
    AddPredecessorToBlock(*SI, BIParent, BB1);

  EraseTerminatorInstAndDCECond(BI);
  return true;
}

/// SinkThenElseCodeToEnd - Given an unconditional branch that goes to BBEnd,
/// check whether BBEnd has only two predecessors and the other predecessor
/// ends with an unconditional branch. If it is true, sink any common code
/// in the two predecessors to BBEnd.
static bool SinkThenElseCodeToEnd(BranchInst *BI1) {
  assert(BI1->isUnconditional());
  BasicBlock *BB1 = BI1->getParent();
  BasicBlock *BBEnd = BI1->getSuccessor(0);

  // Check that BBEnd has two predecessors and the other predecessor ends with
  // an unconditional branch.
  pred_iterator PI = pred_begin(BBEnd), PE = pred_end(BBEnd);
  BasicBlock *Pred0 = *PI++;
  if (PI == PE) // Only one predecessor.
    return false;
  BasicBlock *Pred1 = *PI++;
  if (PI != PE) // More than two predecessors.
    return false;
  BasicBlock *BB2 = (Pred0 == BB1) ? Pred1 : Pred0;
  BranchInst *BI2 = dyn_cast<BranchInst>(BB2->getTerminator());
  if (!BI2 || !BI2->isUnconditional())
    return false;

  // Gather the PHI nodes in BBEnd.
  std::map<Value*, std::pair<Value*, PHINode*> > MapValueFromBB1ToBB2;
  Instruction *FirstNonPhiInBBEnd = nullptr;
  for (BasicBlock::iterator I = BBEnd->begin(), E = BBEnd->end();
       I != E; ++I) {
    if (PHINode *PN = dyn_cast<PHINode>(I)) {
      Value *BB1V = PN->getIncomingValueForBlock(BB1);
      Value *BB2V = PN->getIncomingValueForBlock(BB2);
      MapValueFromBB1ToBB2[BB1V] = std::make_pair(BB2V, PN);
    } else {
      FirstNonPhiInBBEnd = &*I;
      break;
    }
  }
  if (!FirstNonPhiInBBEnd)
    return false;


  // This does very trivial matching, with limited scanning, to find identical
  // instructions in the two blocks.  We scan backward for obviously identical
  // instructions in an identical order.
  BasicBlock::InstListType::reverse_iterator RI1 = BB1->getInstList().rbegin(),
      RE1 = BB1->getInstList().rend(), RI2 = BB2->getInstList().rbegin(),
      RE2 = BB2->getInstList().rend();
  // Skip debug info.
  while (RI1 != RE1 && isa<DbgInfoIntrinsic>(&*RI1)) ++RI1;
  if (RI1 == RE1)
    return false;
  while (RI2 != RE2 && isa<DbgInfoIntrinsic>(&*RI2)) ++RI2;
  if (RI2 == RE2)
    return false;
  // Skip the unconditional branches.
  ++RI1;
  ++RI2;

  bool Changed = false;
  while (RI1 != RE1 && RI2 != RE2) {
    // Skip debug info.
    while (RI1 != RE1 && isa<DbgInfoIntrinsic>(&*RI1)) ++RI1;
    if (RI1 == RE1)
      return Changed;
    while (RI2 != RE2 && isa<DbgInfoIntrinsic>(&*RI2)) ++RI2;
    if (RI2 == RE2)
      return Changed;

    Instruction *I1 = &*RI1, *I2 = &*RI2;
    // I1 and I2 should have a single use in the same PHI node, and they
    // perform the same operation.
    // Cannot move control-flow-involving, volatile loads, vaarg, etc.
    if (isa<PHINode>(I1) || isa<PHINode>(I2) ||
        isa<TerminatorInst>(I1) || isa<TerminatorInst>(I2) ||
        isa<LandingPadInst>(I1) || isa<LandingPadInst>(I2) ||
        isa<AllocaInst>(I1) || isa<AllocaInst>(I2) ||
        I1->mayHaveSideEffects() || I2->mayHaveSideEffects() ||
        I1->mayReadOrWriteMemory() || I2->mayReadOrWriteMemory() ||
        !I1->hasOneUse() || !I2->hasOneUse() ||
        MapValueFromBB1ToBB2.find(I1) == MapValueFromBB1ToBB2.end() ||
        MapValueFromBB1ToBB2[I1].first != I2)
      return Changed;

    // Check whether we should swap the operands of ICmpInst.
    ICmpInst *ICmp1 = dyn_cast<ICmpInst>(I1), *ICmp2 = dyn_cast<ICmpInst>(I2);
    bool SwapOpnds = false;
    if (ICmp1 && ICmp2 &&
        ICmp1->getOperand(0) != ICmp2->getOperand(0) &&
        ICmp1->getOperand(1) != ICmp2->getOperand(1) &&
        (ICmp1->getOperand(0) == ICmp2->getOperand(1) ||
         ICmp1->getOperand(1) == ICmp2->getOperand(0))) {
      ICmp2->swapOperands();
      SwapOpnds = true;
    }
    if (!I1->isSameOperationAs(I2)) {
      if (SwapOpnds)
        ICmp2->swapOperands();
      return Changed;
    }

    // The operands should be either the same or they need to be generated
    // with a PHI node after sinking. We only handle the case where there is
    // a single pair of different operands.
    Value *DifferentOp1 = nullptr, *DifferentOp2 = nullptr;
    unsigned Op1Idx = 0;
    for (unsigned I = 0, E = I1->getNumOperands(); I != E; ++I) {
      if (I1->getOperand(I) == I2->getOperand(I))
        continue;
      // Early exit if we have more-than one pair of different operands or
      // the different operand is already in MapValueFromBB1ToBB2.
      // Early exit if we need a PHI node to replace a constant.
      if (DifferentOp1 ||
          MapValueFromBB1ToBB2.find(I1->getOperand(I)) !=
          MapValueFromBB1ToBB2.end() ||
          isa<Constant>(I1->getOperand(I)) ||
          isa<Constant>(I2->getOperand(I))) {
        // If we can't sink the instructions, undo the swapping.
        if (SwapOpnds)
          ICmp2->swapOperands();
        return Changed;
      }
      DifferentOp1 = I1->getOperand(I);
      Op1Idx = I;
      DifferentOp2 = I2->getOperand(I);
    }

    // We insert the pair of different operands to MapValueFromBB1ToBB2 and
    // remove (I1, I2) from MapValueFromBB1ToBB2.
    if (DifferentOp1) {
      PHINode *NewPN = PHINode::Create(DifferentOp1->getType(), 2,
                                       DifferentOp1->getName() + ".sink",
                                       BBEnd->begin());
      MapValueFromBB1ToBB2[DifferentOp1] = std::make_pair(DifferentOp2, NewPN);
      // I1 should use NewPN instead of DifferentOp1.
      I1->setOperand(Op1Idx, NewPN);
      NewPN->addIncoming(DifferentOp1, BB1);
      NewPN->addIncoming(DifferentOp2, BB2);
      DEBUG(dbgs() << "Create PHI node " << *NewPN << "\n";);
    }
    PHINode *OldPN = MapValueFromBB1ToBB2[I1].second;
    MapValueFromBB1ToBB2.erase(I1);

    DEBUG(dbgs() << "SINK common instructions " << *I1 << "\n";);
    DEBUG(dbgs() << "                         " << *I2 << "\n";);
    // We need to update RE1 and RE2 if we are going to sink the first
    // instruction in the basic block down.
    bool UpdateRE1 = (I1 == BB1->begin()), UpdateRE2 = (I2 == BB2->begin());
    // Sink the instruction.
    BBEnd->getInstList().splice(FirstNonPhiInBBEnd, BB1->getInstList(), I1);
    if (!OldPN->use_empty())
      OldPN->replaceAllUsesWith(I1);
    OldPN->eraseFromParent();

    if (!I2->use_empty())
      I2->replaceAllUsesWith(I1);
    I1->intersectOptionalDataWith(I2);
    I2->eraseFromParent();

    if (UpdateRE1)
      RE1 = BB1->getInstList().rend();
    if (UpdateRE2)
      RE2 = BB2->getInstList().rend();
    FirstNonPhiInBBEnd = I1;
    NumSinkCommons++;
    Changed = true;
  }
  return Changed;
}

/// \brief Determine if we can hoist sink a sole store instruction out of a
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

  // Look for a store to the same pointer in BrBB.
  unsigned MaxNumInstToLookAt = 10;
  for (BasicBlock::reverse_iterator RI = BrBB->rbegin(),
       RE = BrBB->rend(); RI != RE && (--MaxNumInstToLookAt); ++RI) {
    Instruction *CurI = &*RI;

    // Could be calling an instruction that effects memory like free().
    if (CurI->mayHaveSideEffects() && !isa<StoreInst>(CurI))
      return nullptr;

    StoreInst *SI = dyn_cast<StoreInst>(CurI);
    // Found the previous store make sure it stores to the same location.
    if (SI && SI->getPointerOperand() == StorePtr)
      // Found the previous store, return its value operand.
      return SI->getValueOperand();
    else if (SI)
      return nullptr; // Unknown store.
  }

  return nullptr;
}

/// \brief Speculate a conditional basic block flattening the CFG.
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
///     %phi = phi [ %sub, %ThenBB ], [ 0, %EndBB ]
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
static bool SpeculativelyExecuteBB(BranchInst *BI, BasicBlock *ThenBB,
                                   const DataLayout *DL) {
  // Be conservative for now. FP select instruction can often be expensive.
  Value *BrCond = BI->getCondition();
  if (isa<FCmpInst>(BrCond))
    return false;

  BasicBlock *BB = BI->getParent();
  BasicBlock *EndBB = ThenBB->getTerminator()->getSuccessor(0);

  // If ThenBB is actually on the false edge of the conditional branch, remember
  // to swap the select operands later.
  bool Invert = false;
  if (ThenBB != BI->getSuccessor(0)) {
    assert(ThenBB == BI->getSuccessor(1) && "No edge from 'if' block?");
    Invert = true;
  }
  assert(EndBB == BI->getSuccessor(!Invert) && "No edge from to end block");

  // Keep a count of how many times instructions are used within CondBB when
  // they are candidates for sinking into CondBB. Specifically:
  // - They are defined in BB, and
  // - They have no side effects, and
  // - All of their uses are in CondBB.
  SmallDenseMap<Instruction *, unsigned, 4> SinkCandidateUseCounts;

  unsigned SpeculationCost = 0;
  Value *SpeculatedStoreValue = nullptr;
  StoreInst *SpeculatedStore = nullptr;
  for (BasicBlock::iterator BBI = ThenBB->begin(),
                            BBE = std::prev(ThenBB->end());
       BBI != BBE; ++BBI) {
    Instruction *I = BBI;
    // Skip debug info.
    if (isa<DbgInfoIntrinsic>(I))
      continue;

    // Only speculatively execution a single instruction (not counting the
    // terminator) for now.
    ++SpeculationCost;
    if (SpeculationCost > 1)
      return false;

    // Don't hoist the instruction if it's unsafe or expensive.
    if (!isSafeToSpeculativelyExecute(I, DL) &&
        !(HoistCondStores &&
          (SpeculatedStoreValue = isSafeToSpeculateStore(I, BB, ThenBB,
                                                         EndBB))))
      return false;
    if (!SpeculatedStoreValue &&
        ComputeSpeculationCost(I, DL) > PHINodeFoldingThreshold)
      return false;

    // Store the store speculation candidate.
    if (SpeculatedStoreValue)
      SpeculatedStore = cast<StoreInst>(I);

    // Do not hoist the instruction if any of its operands are defined but not
    // used in BB. The transformation will prevent the operand from
    // being sunk into the use block.
    for (User::op_iterator i = I->op_begin(), e = I->op_end();
         i != e; ++i) {
      Instruction *OpI = dyn_cast<Instruction>(*i);
      if (!OpI || OpI->getParent() != BB ||
          OpI->mayHaveSideEffects())
        continue; // Not a candidate for sinking.

      ++SinkCandidateUseCounts[OpI];
    }
  }

  // Consider any sink candidates which are only used in CondBB as costs for
  // speculation. Note, while we iterate over a DenseMap here, we are summing
  // and so iteration order isn't significant.
  for (SmallDenseMap<Instruction *, unsigned, 4>::iterator I =
           SinkCandidateUseCounts.begin(), E = SinkCandidateUseCounts.end();
       I != E; ++I)
    if (I->first->getNumUses() == I->second) {
      ++SpeculationCost;
      if (SpeculationCost > 1)
        return false;
    }

  // Check that the PHI nodes can be converted to selects.
  bool HaveRewritablePHIs = false;
  for (BasicBlock::iterator I = EndBB->begin();
       PHINode *PN = dyn_cast<PHINode>(I); ++I) {
    Value *OrigV = PN->getIncomingValueForBlock(BB);
    Value *ThenV = PN->getIncomingValueForBlock(ThenBB);

    // FIXME: Try to remove some of the duplication with HoistThenElseCodeToIf.
    // Skip PHIs which are trivial.
    if (ThenV == OrigV)
      continue;

    HaveRewritablePHIs = true;
    ConstantExpr *OrigCE = dyn_cast<ConstantExpr>(OrigV);
    ConstantExpr *ThenCE = dyn_cast<ConstantExpr>(ThenV);
    if (!OrigCE && !ThenCE)
      continue; // Known safe and cheap.

    if ((ThenCE && !isSafeToSpeculativelyExecute(ThenCE, DL)) ||
        (OrigCE && !isSafeToSpeculativelyExecute(OrigCE, DL)))
      return false;
    unsigned OrigCost = OrigCE ? ComputeSpeculationCost(OrigCE, DL) : 0;
    unsigned ThenCost = ThenCE ? ComputeSpeculationCost(ThenCE, DL) : 0;
    if (OrigCost + ThenCost > 2 * PHINodeFoldingThreshold)
      return false;

    // Account for the cost of an unfolded ConstantExpr which could end up
    // getting expanded into Instructions.
    // FIXME: This doesn't account for how many operations are combined in the
    // constant expression.
    ++SpeculationCost;
    if (SpeculationCost > 1)
      return false;
  }

  // If there are no PHIs to process, bail early. This helps ensure idempotence
  // as well.
  if (!HaveRewritablePHIs && !(HoistCondStores && SpeculatedStoreValue))
    return false;

  // If we get here, we can hoist the instruction and if-convert.
  DEBUG(dbgs() << "SPECULATIVELY EXECUTING BB" << *ThenBB << "\n";);

  // Insert a select of the value of the speculated store.
  if (SpeculatedStoreValue) {
    IRBuilder<true, NoFolder> Builder(BI);
    Value *TrueV = SpeculatedStore->getValueOperand();
    Value *FalseV = SpeculatedStoreValue;
    if (Invert)
      std::swap(TrueV, FalseV);
    Value *S = Builder.CreateSelect(BrCond, TrueV, FalseV, TrueV->getName() +
                                    "." + FalseV->getName());
    SpeculatedStore->setOperand(0, S);
  }

  // Hoist the instructions.
  BB->getInstList().splice(BI, ThenBB->getInstList(), ThenBB->begin(),
                           std::prev(ThenBB->end()));

  // Insert selects and rewrite the PHI operands.
  IRBuilder<true, NoFolder> Builder(BI);
  for (BasicBlock::iterator I = EndBB->begin();
       PHINode *PN = dyn_cast<PHINode>(I); ++I) {
    unsigned OrigI = PN->getBasicBlockIndex(BB);
    unsigned ThenI = PN->getBasicBlockIndex(ThenBB);
    Value *OrigV = PN->getIncomingValue(OrigI);
    Value *ThenV = PN->getIncomingValue(ThenI);

    // Skip PHIs which are trivial.
    if (OrigV == ThenV)
      continue;

    // Create a select whose true value is the speculatively executed value and
    // false value is the preexisting value. Swap them if the branch
    // destinations were inverted.
    Value *TrueV = ThenV, *FalseV = OrigV;
    if (Invert)
      std::swap(TrueV, FalseV);
    Value *V = Builder.CreateSelect(BrCond, TrueV, FalseV,
                                    TrueV->getName() + "." + FalseV->getName());
    PN->setIncomingValue(OrigI, V);
    PN->setIncomingValue(ThenI, V);
  }

  ++NumSpeculations;
  return true;
}

/// \returns True if this block contains a CallInst with the NoDuplicate
/// attribute.
static bool HasNoDuplicateCall(const BasicBlock *BB) {
  for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
    const CallInst *CI = dyn_cast<CallInst>(I);
    if (!CI)
      continue;
    if (CI->cannotDuplicate())
      return true;
  }
  return false;
}

/// BlockIsSimpleEnoughToThreadThrough - Return true if we can thread a branch
/// across this block.
static bool BlockIsSimpleEnoughToThreadThrough(BasicBlock *BB) {
  BranchInst *BI = cast<BranchInst>(BB->getTerminator());
  unsigned Size = 0;

  for (BasicBlock::iterator BBI = BB->begin(); &*BBI != BI; ++BBI) {
    if (isa<DbgInfoIntrinsic>(BBI))
      continue;
    if (Size > 10) return false;  // Don't clone large BB's.
    ++Size;

    // We can only support instructions that do not define values that are
    // live outside of the current basic block.
    for (User *U : BBI->users()) {
      Instruction *UI = cast<Instruction>(U);
      if (UI->getParent() != BB || isa<PHINode>(UI)) return false;
    }

    // Looks ok, continue checking.
  }

  return true;
}

/// FoldCondBranchOnPHI - If we have a conditional branch on a PHI node value
/// that is defined in the same block as the branch and if any PHI entries are
/// constants, thread edges corresponding to that entry to be branches to their
/// ultimate destination.
static bool FoldCondBranchOnPHI(BranchInst *BI, const DataLayout *DL) {
  BasicBlock *BB = BI->getParent();
  PHINode *PN = dyn_cast<PHINode>(BI->getCondition());
  // NOTE: we currently cannot transform this case if the PHI node is used
  // outside of the block.
  if (!PN || PN->getParent() != BB || !PN->hasOneUse())
    return false;

  // Degenerate case of a single entry PHI.
  if (PN->getNumIncomingValues() == 1) {
    FoldSingleEntryPHINodes(PN->getParent());
    return true;
  }

  // Now we know that this block has multiple preds and two succs.
  if (!BlockIsSimpleEnoughToThreadThrough(BB)) return false;

  if (HasNoDuplicateCall(BB)) return false;

  // Okay, this is a simple enough basic block.  See if any phi values are
  // constants.
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
    ConstantInt *CB = dyn_cast<ConstantInt>(PN->getIncomingValue(i));
    if (!CB || !CB->getType()->isIntegerTy(1)) continue;

    // Okay, we now know that all edges from PredBB should be revectored to
    // branch to RealDest.
    BasicBlock *PredBB = PN->getIncomingBlock(i);
    BasicBlock *RealDest = BI->getSuccessor(!CB->getZExtValue());

    if (RealDest == BB) continue;  // Skip self loops.
    // Skip if the predecessor's terminator is an indirect branch.
    if (isa<IndirectBrInst>(PredBB->getTerminator())) continue;

    // The dest block might have PHI nodes, other predecessors and other
    // difficult cases.  Instead of being smart about this, just insert a new
    // block that jumps to the destination block, effectively splitting
    // the edge we are about to create.
    BasicBlock *EdgeBB = BasicBlock::Create(BB->getContext(),
                                            RealDest->getName()+".critedge",
                                            RealDest->getParent(), RealDest);
    BranchInst::Create(RealDest, EdgeBB);

    // Update PHI nodes.
    AddPredecessorToBlock(RealDest, EdgeBB, BB);

    // BB may have instructions that are being threaded over.  Clone these
    // instructions into EdgeBB.  We know that there will be no uses of the
    // cloned instructions outside of EdgeBB.
    BasicBlock::iterator InsertPt = EdgeBB->begin();
    DenseMap<Value*, Value*> TranslateMap;  // Track translated values.
    for (BasicBlock::iterator BBI = BB->begin(); &*BBI != BI; ++BBI) {
      if (PHINode *PN = dyn_cast<PHINode>(BBI)) {
        TranslateMap[PN] = PN->getIncomingValueForBlock(PredBB);
        continue;
      }
      // Clone the instruction.
      Instruction *N = BBI->clone();
      if (BBI->hasName()) N->setName(BBI->getName()+".c");

      // Update operands due to translation.
      for (User::op_iterator i = N->op_begin(), e = N->op_end();
           i != e; ++i) {
        DenseMap<Value*, Value*>::iterator PI = TranslateMap.find(*i);
        if (PI != TranslateMap.end())
          *i = PI->second;
      }

      // Check for trivial simplification.
      if (Value *V = SimplifyInstruction(N, DL)) {
        TranslateMap[BBI] = V;
        delete N;   // Instruction folded away, don't need actual inst
      } else {
        // Insert the new instruction into its new home.
        EdgeBB->getInstList().insert(InsertPt, N);
        if (!BBI->use_empty())
          TranslateMap[BBI] = N;
      }
    }

    // Loop over all of the edges from PredBB to BB, changing them to branch
    // to EdgeBB instead.
    TerminatorInst *PredBBTI = PredBB->getTerminator();
    for (unsigned i = 0, e = PredBBTI->getNumSuccessors(); i != e; ++i)
      if (PredBBTI->getSuccessor(i) == BB) {
        BB->removePredecessor(PredBB);
        PredBBTI->setSuccessor(i, EdgeBB);
      }

    // Recurse, simplifying any other constants.
    return FoldCondBranchOnPHI(BI, DL) | true;
  }

  return false;
}

/// FoldTwoEntryPHINode - Given a BB that starts with the specified two-entry
/// PHI node, see if we can eliminate it.
static bool FoldTwoEntryPHINode(PHINode *PN, const DataLayout *DL) {
  // Ok, this is a two entry PHI node.  Check to see if this is a simple "if
  // statement", which has a very simple dominance structure.  Basically, we
  // are trying to find the condition that is being branched on, which
  // subsequently causes this merge to happen.  We really want control
  // dependence information for this check, but simplifycfg can't keep it up
  // to date, and this catches most of the cases we care about anyway.
  BasicBlock *BB = PN->getParent();
  BasicBlock *IfTrue, *IfFalse;
  Value *IfCond = GetIfCondition(BB, IfTrue, IfFalse);
  if (!IfCond ||
      // Don't bother if the branch will be constant folded trivially.
      isa<ConstantInt>(IfCond))
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
  SmallPtrSet<Instruction*, 4> AggressiveInsts;
  unsigned MaxCostVal0 = PHINodeFoldingThreshold,
           MaxCostVal1 = PHINodeFoldingThreshold;

  for (BasicBlock::iterator II = BB->begin(); isa<PHINode>(II);) {
    PHINode *PN = cast<PHINode>(II++);
    if (Value *V = SimplifyInstruction(PN, DL)) {
      PN->replaceAllUsesWith(V);
      PN->eraseFromParent();
      continue;
    }

    if (!DominatesMergePoint(PN->getIncomingValue(0), BB, &AggressiveInsts,
                             MaxCostVal0, DL) ||
        !DominatesMergePoint(PN->getIncomingValue(1), BB, &AggressiveInsts,
                             MaxCostVal1, DL))
      return false;
  }

  // If we folded the first phi, PN dangles at this point.  Refresh it.  If
  // we ran out of PHIs then we simplified them all.
  PN = dyn_cast<PHINode>(BB->begin());
  if (!PN) return true;

  // Don't fold i1 branches on PHIs which contain binary operators.  These can
  // often be turned into switches and other things.
  if (PN->getType()->isIntegerTy(1) &&
      (isa<BinaryOperator>(PN->getIncomingValue(0)) ||
       isa<BinaryOperator>(PN->getIncomingValue(1)) ||
       isa<BinaryOperator>(IfCond)))
    return false;

  // If we all PHI nodes are promotable, check to make sure that all
  // instructions in the predecessor blocks can be promoted as well.  If
  // not, we won't be able to get rid of the control flow, so it's not
  // worth promoting to select instructions.
  BasicBlock *DomBlock = nullptr;
  BasicBlock *IfBlock1 = PN->getIncomingBlock(0);
  BasicBlock *IfBlock2 = PN->getIncomingBlock(1);
  if (cast<BranchInst>(IfBlock1->getTerminator())->isConditional()) {
    IfBlock1 = nullptr;
  } else {
    DomBlock = *pred_begin(IfBlock1);
    for (BasicBlock::iterator I = IfBlock1->begin();!isa<TerminatorInst>(I);++I)
      if (!AggressiveInsts.count(I) && !isa<DbgInfoIntrinsic>(I)) {
        // This is not an aggressive instruction that we can promote.
        // Because of this, we won't be able to get rid of the control
        // flow, so the xform is not worth it.
        return false;
      }
  }

  if (cast<BranchInst>(IfBlock2->getTerminator())->isConditional()) {
    IfBlock2 = nullptr;
  } else {
    DomBlock = *pred_begin(IfBlock2);
    for (BasicBlock::iterator I = IfBlock2->begin();!isa<TerminatorInst>(I);++I)
      if (!AggressiveInsts.count(I) && !isa<DbgInfoIntrinsic>(I)) {
        // This is not an aggressive instruction that we can promote.
        // Because of this, we won't be able to get rid of the control
        // flow, so the xform is not worth it.
        return false;
      }
  }

  DEBUG(dbgs() << "FOUND IF CONDITION!  " << *IfCond << "  T: "
               << IfTrue->getName() << "  F: " << IfFalse->getName() << "\n");

  // If we can still promote the PHI nodes after this gauntlet of tests,
  // do all of the PHI's now.
  Instruction *InsertPt = DomBlock->getTerminator();
  IRBuilder<true, NoFolder> Builder(InsertPt);

  // Move all 'aggressive' instructions, which are defined in the
  // conditional parts of the if's up to the dominating block.
  if (IfBlock1)
    DomBlock->getInstList().splice(InsertPt,
                                   IfBlock1->getInstList(), IfBlock1->begin(),
                                   IfBlock1->getTerminator());
  if (IfBlock2)
    DomBlock->getInstList().splice(InsertPt,
                                   IfBlock2->getInstList(), IfBlock2->begin(),
                                   IfBlock2->getTerminator());

  while (PHINode *PN = dyn_cast<PHINode>(BB->begin())) {
    // Change the PHI node into a select instruction.
    Value *TrueVal  = PN->getIncomingValue(PN->getIncomingBlock(0) == IfFalse);
    Value *FalseVal = PN->getIncomingValue(PN->getIncomingBlock(0) == IfTrue);

    SelectInst *NV =
      cast<SelectInst>(Builder.CreateSelect(IfCond, TrueVal, FalseVal, ""));
    PN->replaceAllUsesWith(NV);
    NV->takeName(PN);
    PN->eraseFromParent();
  }

  // At this point, IfBlock1 and IfBlock2 are both empty, so our if statement
  // has been flattened.  Change DomBlock to jump directly to our new block to
  // avoid other simplifycfg's kicking in on the diamond.
  TerminatorInst *OldTI = DomBlock->getTerminator();
  Builder.SetInsertPoint(OldTI);
  Builder.CreateBr(BB);
  OldTI->eraseFromParent();
  return true;
}

/// SimplifyCondBranchToTwoReturns - If we found a conditional branch that goes
/// to two returning blocks, try to merge them together into one return,
/// introducing a select if the return values disagree.
static bool SimplifyCondBranchToTwoReturns(BranchInst *BI,
                                           IRBuilder<> &Builder) {
  assert(BI->isConditional() && "Must be a conditional branch");
  BasicBlock *TrueSucc = BI->getSuccessor(0);
  BasicBlock *FalseSucc = BI->getSuccessor(1);
  ReturnInst *TrueRet = cast<ReturnInst>(TrueSucc->getTerminator());
  ReturnInst *FalseRet = cast<ReturnInst>(FalseSucc->getTerminator());

  // Check to ensure both blocks are empty (just a return) or optionally empty
  // with PHI nodes.  If there are other instructions, merging would cause extra
  // computation on one path or the other.
  if (!TrueSucc->getFirstNonPHIOrDbg()->isTerminator())
    return false;
  if (!FalseSucc->getFirstNonPHIOrDbg()->isTerminator())
    return false;

  Builder.SetInsertPoint(BI);
  // Okay, we found a branch that is going to two return nodes.  If
  // there is no return value for this function, just change the
  // branch into a return.
  if (FalseRet->getNumOperands() == 0) {
    TrueSucc->removePredecessor(BI->getParent());
    FalseSucc->removePredecessor(BI->getParent());
    Builder.CreateRetVoid();
    EraseTerminatorInstAndDCECond(BI);
    return true;
  }

  // Otherwise, figure out what the true and false return values are
  // so we can insert a new select instruction.
  Value *TrueValue = TrueRet->getReturnValue();
  Value *FalseValue = FalseRet->getReturnValue();

  // Unwrap any PHI nodes in the return blocks.
  if (PHINode *TVPN = dyn_cast_or_null<PHINode>(TrueValue))
    if (TVPN->getParent() == TrueSucc)
      TrueValue = TVPN->getIncomingValueForBlock(BI->getParent());
  if (PHINode *FVPN = dyn_cast_or_null<PHINode>(FalseValue))
    if (FVPN->getParent() == FalseSucc)
      FalseValue = FVPN->getIncomingValueForBlock(BI->getParent());

  // In order for this transformation to be safe, we must be able to
  // unconditionally execute both operands to the return.  This is
  // normally the case, but we could have a potentially-trapping
  // constant expression that prevents this transformation from being
  // safe.
  if (ConstantExpr *TCV = dyn_cast_or_null<ConstantExpr>(TrueValue))
    if (TCV->canTrap())
      return false;
  if (ConstantExpr *FCV = dyn_cast_or_null<ConstantExpr>(FalseValue))
    if (FCV->canTrap())
      return false;

  // Okay, we collected all the mapped values and checked them for sanity, and
  // defined to really do this transformation.  First, update the CFG.
  TrueSucc->removePredecessor(BI->getParent());
  FalseSucc->removePredecessor(BI->getParent());

  // Insert select instructions where needed.
  Value *BrCond = BI->getCondition();
  if (TrueValue) {
    // Insert a select if the results differ.
    if (TrueValue == FalseValue || isa<UndefValue>(FalseValue)) {
    } else if (isa<UndefValue>(TrueValue)) {
      TrueValue = FalseValue;
    } else {
      TrueValue = Builder.CreateSelect(BrCond, TrueValue,
                                       FalseValue, "retval");
    }
  }

  Value *RI = !TrueValue ?
    Builder.CreateRetVoid() : Builder.CreateRet(TrueValue);

  (void) RI;

  DEBUG(dbgs() << "\nCHANGING BRANCH TO TWO RETURNS INTO SELECT:"
               << "\n  " << *BI << "NewRet = " << *RI
               << "TRUEBLOCK: " << *TrueSucc << "FALSEBLOCK: "<< *FalseSucc);

  EraseTerminatorInstAndDCECond(BI);

  return true;
}

/// ExtractBranchMetadata - Given a conditional BranchInstruction, retrieve the
/// probabilities of the branch taking each edge. Fills in the two APInt
/// parameters and return true, or returns false if no or invalid metadata was
/// found.
static bool ExtractBranchMetadata(BranchInst *BI,
                                  uint64_t &ProbTrue, uint64_t &ProbFalse) {
  assert(BI->isConditional() &&
         "Looking for probabilities on unconditional branch?");
  MDNode *ProfileData = BI->getMetadata(LLVMContext::MD_prof);
  if (!ProfileData || ProfileData->getNumOperands() != 3) return false;
  ConstantInt *CITrue = dyn_cast<ConstantInt>(ProfileData->getOperand(1));
  ConstantInt *CIFalse = dyn_cast<ConstantInt>(ProfileData->getOperand(2));
  if (!CITrue || !CIFalse) return false;
  ProbTrue = CITrue->getValue().getZExtValue();
  ProbFalse = CIFalse->getValue().getZExtValue();
  return true;
}

/// checkCSEInPredecessor - Return true if the given instruction is available
/// in its predecessor block. If yes, the instruction will be removed.
///
static bool checkCSEInPredecessor(Instruction *Inst, BasicBlock *PB) {
  if (!isa<BinaryOperator>(Inst) && !isa<CmpInst>(Inst))
    return false;
  for (BasicBlock::iterator I = PB->begin(), E = PB->end(); I != E; I++) {
    Instruction *PBI = &*I;
    // Check whether Inst and PBI generate the same value.
    if (Inst->isIdenticalTo(PBI)) {
      Inst->replaceAllUsesWith(PBI);
      Inst->eraseFromParent();
      return true;
    }
  }
  return false;
}

/// FoldBranchToCommonDest - If this basic block is simple enough, and if a
/// predecessor branches to us and one of our successors, fold the block into
/// the predecessor and use logical operations to pick the right destination.
bool llvm::FoldBranchToCommonDest(BranchInst *BI, const DataLayout *DL) {
  BasicBlock *BB = BI->getParent();

  Instruction *Cond = nullptr;
  if (BI->isConditional())
    Cond = dyn_cast<Instruction>(BI->getCondition());
  else {
    // For unconditional branch, check for a simple CFG pattern, where
    // BB has a single predecessor and BB's successor is also its predecessor's
    // successor. If such pattern exisits, check for CSE between BB and its
    // predecessor.
    if (BasicBlock *PB = BB->getSinglePredecessor())
      if (BranchInst *PBI = dyn_cast<BranchInst>(PB->getTerminator()))
        if (PBI->isConditional() &&
            (BI->getSuccessor(0) == PBI->getSuccessor(0) ||
             BI->getSuccessor(0) == PBI->getSuccessor(1))) {
          for (BasicBlock::iterator I = BB->begin(), E = BB->end();
               I != E; ) {
            Instruction *Curr = I++;
            if (isa<CmpInst>(Curr)) {
              Cond = Curr;
              break;
            }
            // Quit if we can't remove this instruction.
            if (!checkCSEInPredecessor(Curr, PB))
              return false;
          }
        }

    if (!Cond)
      return false;
  }

  if (!Cond || (!isa<CmpInst>(Cond) && !isa<BinaryOperator>(Cond)) ||
      Cond->getParent() != BB || !Cond->hasOneUse())
  return false;

  // Only allow this if the condition is a simple instruction that can be
  // executed unconditionally.  It must be in the same block as the branch, and
  // must be at the front of the block.
  BasicBlock::iterator FrontIt = BB->front();

  // Ignore dbg intrinsics.
  while (isa<DbgInfoIntrinsic>(FrontIt)) ++FrontIt;

  // Allow a single instruction to be hoisted in addition to the compare
  // that feeds the branch.  We later ensure that any values that _it_ uses
  // were also live in the predecessor, so that we don't unnecessarily create
  // register pressure or inhibit out-of-order execution.
  Instruction *BonusInst = nullptr;
  if (&*FrontIt != Cond &&
      FrontIt->hasOneUse() && FrontIt->user_back() == Cond &&
      isSafeToSpeculativelyExecute(FrontIt, DL)) {
    BonusInst = &*FrontIt;
    ++FrontIt;

    // Ignore dbg intrinsics.
    while (isa<DbgInfoIntrinsic>(FrontIt)) ++FrontIt;
  }

  // Only a single bonus inst is allowed.
  if (&*FrontIt != Cond)
    return false;

  // Make sure the instruction after the condition is the cond branch.
  BasicBlock::iterator CondIt = Cond; ++CondIt;

  // Ignore dbg intrinsics.
  while (isa<DbgInfoIntrinsic>(CondIt)) ++CondIt;

  if (&*CondIt != BI)
    return false;

  // Cond is known to be a compare or binary operator.  Check to make sure that
  // neither operand is a potentially-trapping constant expression.
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Cond->getOperand(0)))
    if (CE->canTrap())
      return false;
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Cond->getOperand(1)))
    if (CE->canTrap())
      return false;

  // Finally, don't infinitely unroll conditional loops.
  BasicBlock *TrueDest  = BI->getSuccessor(0);
  BasicBlock *FalseDest = (BI->isConditional()) ? BI->getSuccessor(1) : nullptr;
  if (TrueDest == BB || FalseDest == BB)
    return false;

  for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI) {
    BasicBlock *PredBlock = *PI;
    BranchInst *PBI = dyn_cast<BranchInst>(PredBlock->getTerminator());

    // Check that we have two conditional branches.  If there is a PHI node in
    // the common successor, verify that the same value flows in from both
    // blocks.
    SmallVector<PHINode*, 4> PHIs;
    if (!PBI || PBI->isUnconditional() ||
        (BI->isConditional() &&
         !SafeToMergeTerminators(BI, PBI)) ||
        (!BI->isConditional() &&
         !isProfitableToFoldUnconditional(BI, PBI, Cond, PHIs)))
      continue;

    // Determine if the two branches share a common destination.
    Instruction::BinaryOps Opc = Instruction::BinaryOpsEnd;
    bool InvertPredCond = false;

    if (BI->isConditional()) {
      if (PBI->getSuccessor(0) == TrueDest)
        Opc = Instruction::Or;
      else if (PBI->getSuccessor(1) == FalseDest)
        Opc = Instruction::And;
      else if (PBI->getSuccessor(0) == FalseDest)
        Opc = Instruction::And, InvertPredCond = true;
      else if (PBI->getSuccessor(1) == TrueDest)
        Opc = Instruction::Or, InvertPredCond = true;
      else
        continue;
    } else {
      if (PBI->getSuccessor(0) != TrueDest && PBI->getSuccessor(1) != TrueDest)
        continue;
    }

    // Ensure that any values used in the bonus instruction are also used
    // by the terminator of the predecessor.  This means that those values
    // must already have been resolved, so we won't be inhibiting the
    // out-of-order core by speculating them earlier. We also allow
    // instructions that are used by the terminator's condition because it
    // exposes more merging opportunities.
    bool UsedByBranch = (BonusInst && BonusInst->hasOneUse() &&
                         BonusInst->user_back() == Cond);

    if (BonusInst && !UsedByBranch) {
      // Collect the values used by the bonus inst
      SmallPtrSet<Value*, 4> UsedValues;
      for (Instruction::op_iterator OI = BonusInst->op_begin(),
           OE = BonusInst->op_end(); OI != OE; ++OI) {
        Value *V = *OI;
        if (!isa<Constant>(V) && !isa<Argument>(V))
          UsedValues.insert(V);
      }

      SmallVector<std::pair<Value*, unsigned>, 4> Worklist;
      Worklist.push_back(std::make_pair(PBI->getOperand(0), 0));

      // Walk up to four levels back up the use-def chain of the predecessor's
      // terminator to see if all those values were used.  The choice of four
      // levels is arbitrary, to provide a compile-time-cost bound.
      while (!Worklist.empty()) {
        std::pair<Value*, unsigned> Pair = Worklist.back();
        Worklist.pop_back();

        if (Pair.second >= 4) continue;
        UsedValues.erase(Pair.first);
        if (UsedValues.empty()) break;

        if (Instruction *I = dyn_cast<Instruction>(Pair.first)) {
          for (Instruction::op_iterator OI = I->op_begin(), OE = I->op_end();
               OI != OE; ++OI)
            Worklist.push_back(std::make_pair(OI->get(), Pair.second+1));
        }
      }

      if (!UsedValues.empty()) return false;
    }

    DEBUG(dbgs() << "FOLDING BRANCH TO COMMON DEST:\n" << *PBI << *BB);
    IRBuilder<> Builder(PBI);

    // If we need to invert the condition in the pred block to match, do so now.
    if (InvertPredCond) {
      Value *NewCond = PBI->getCondition();

      if (NewCond->hasOneUse() && isa<CmpInst>(NewCond)) {
        CmpInst *CI = cast<CmpInst>(NewCond);
        CI->setPredicate(CI->getInversePredicate());
      } else {
        NewCond = Builder.CreateNot(NewCond,
                                    PBI->getCondition()->getName()+".not");
      }

      PBI->setCondition(NewCond);
      PBI->swapSuccessors();
    }

    // If we have a bonus inst, clone it into the predecessor block.
    Instruction *NewBonus = nullptr;
    if (BonusInst) {
      NewBonus = BonusInst->clone();

      // If we moved a load, we cannot any longer claim any knowledge about
      // its potential value. The previous information might have been valid
      // only given the branch precondition.
      // For an analogous reason, we must also drop all the metadata whose
      // semantics we don't understand.
      NewBonus->dropUnknownMetadata(LLVMContext::MD_dbg);

      PredBlock->getInstList().insert(PBI, NewBonus);
      NewBonus->takeName(BonusInst);
      BonusInst->setName(BonusInst->getName()+".old");
    }

    // Clone Cond into the predecessor basic block, and or/and the
    // two conditions together.
    Instruction *New = Cond->clone();
    if (BonusInst) New->replaceUsesOfWith(BonusInst, NewBonus);
    PredBlock->getInstList().insert(PBI, New);
    New->takeName(Cond);
    Cond->setName(New->getName()+".old");

    if (BI->isConditional()) {
      Instruction *NewCond =
        cast<Instruction>(Builder.CreateBinOp(Opc, PBI->getCondition(),
                                            New, "or.cond"));
      PBI->setCondition(NewCond);

      uint64_t PredTrueWeight, PredFalseWeight, SuccTrueWeight, SuccFalseWeight;
      bool PredHasWeights = ExtractBranchMetadata(PBI, PredTrueWeight,
                                                  PredFalseWeight);
      bool SuccHasWeights = ExtractBranchMetadata(BI, SuccTrueWeight,
                                                  SuccFalseWeight);
      SmallVector<uint64_t, 8> NewWeights;

      if (PBI->getSuccessor(0) == BB) {
        if (PredHasWeights && SuccHasWeights) {
          // PBI: br i1 %x, BB, FalseDest
          // BI:  br i1 %y, TrueDest, FalseDest
          //TrueWeight is TrueWeight for PBI * TrueWeight for BI.
          NewWeights.push_back(PredTrueWeight * SuccTrueWeight);
          //FalseWeight is FalseWeight for PBI * TotalWeight for BI +
          //               TrueWeight for PBI * FalseWeight for BI.
          // We assume that total weights of a BranchInst can fit into 32 bits.
          // Therefore, we will not have overflow using 64-bit arithmetic.
          NewWeights.push_back(PredFalseWeight * (SuccFalseWeight +
               SuccTrueWeight) + PredTrueWeight * SuccFalseWeight);
        }
        AddPredecessorToBlock(TrueDest, PredBlock, BB);
        PBI->setSuccessor(0, TrueDest);
      }
      if (PBI->getSuccessor(1) == BB) {
        if (PredHasWeights && SuccHasWeights) {
          // PBI: br i1 %x, TrueDest, BB
          // BI:  br i1 %y, TrueDest, FalseDest
          //TrueWeight is TrueWeight for PBI * TotalWeight for BI +
          //              FalseWeight for PBI * TrueWeight for BI.
          NewWeights.push_back(PredTrueWeight * (SuccFalseWeight +
              SuccTrueWeight) + PredFalseWeight * SuccTrueWeight);
          //FalseWeight is FalseWeight for PBI * FalseWeight for BI.
          NewWeights.push_back(PredFalseWeight * SuccFalseWeight);
        }
        AddPredecessorToBlock(FalseDest, PredBlock, BB);
        PBI->setSuccessor(1, FalseDest);
      }
      if (NewWeights.size() == 2) {
        // Halve the weights if any of them cannot fit in an uint32_t
        FitWeights(NewWeights);

        SmallVector<uint32_t, 8> MDWeights(NewWeights.begin(),NewWeights.end());
        PBI->setMetadata(LLVMContext::MD_prof,
                         MDBuilder(BI->getContext()).
                         createBranchWeights(MDWeights));
      } else
        PBI->setMetadata(LLVMContext::MD_prof, nullptr);
    } else {
      // Update PHI nodes in the common successors.
      for (unsigned i = 0, e = PHIs.size(); i != e; ++i) {
        ConstantInt *PBI_C = cast<ConstantInt>(
          PHIs[i]->getIncomingValueForBlock(PBI->getParent()));
        assert(PBI_C->getType()->isIntegerTy(1));
        Instruction *MergedCond = nullptr;
        if (PBI->getSuccessor(0) == TrueDest) {
          // Create (PBI_Cond and PBI_C) or (!PBI_Cond and BI_Value)
          // PBI_C is true: PBI_Cond or (!PBI_Cond and BI_Value)
          //       is false: !PBI_Cond and BI_Value
          Instruction *NotCond =
            cast<Instruction>(Builder.CreateNot(PBI->getCondition(),
                                "not.cond"));
          MergedCond =
            cast<Instruction>(Builder.CreateBinOp(Instruction::And,
                                NotCond, New,
                                "and.cond"));
          if (PBI_C->isOne())
            MergedCond =
              cast<Instruction>(Builder.CreateBinOp(Instruction::Or,
                                  PBI->getCondition(), MergedCond,
                                  "or.cond"));
        } else {
          // Create (PBI_Cond and BI_Value) or (!PBI_Cond and PBI_C)
          // PBI_C is true: (PBI_Cond and BI_Value) or (!PBI_Cond)
          //       is false: PBI_Cond and BI_Value
          MergedCond =
            cast<Instruction>(Builder.CreateBinOp(Instruction::And,
                                PBI->getCondition(), New,
                                "and.cond"));
          if (PBI_C->isOne()) {
            Instruction *NotCond =
              cast<Instruction>(Builder.CreateNot(PBI->getCondition(),
                                  "not.cond"));
            MergedCond =
              cast<Instruction>(Builder.CreateBinOp(Instruction::Or,
                                  NotCond, MergedCond,
                                  "or.cond"));
          }
        }
        // Update PHI Node.
        PHIs[i]->setIncomingValue(PHIs[i]->getBasicBlockIndex(PBI->getParent()),
                                  MergedCond);
      }
      // Change PBI from Conditional to Unconditional.
      BranchInst *New_PBI = BranchInst::Create(TrueDest, PBI);
      EraseTerminatorInstAndDCECond(PBI);
      PBI = New_PBI;
    }

    // TODO: If BB is reachable from all paths through PredBlock, then we
    // could replace PBI's branch probabilities with BI's.

    // Copy any debug value intrinsics into the end of PredBlock.
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
      if (isa<DbgInfoIntrinsic>(*I))
        I->clone()->insertBefore(PBI);

    return true;
  }
  return false;
}

/// SimplifyCondBranchToCondBranch - If we have a conditional branch as a
/// predecessor of another block, this function tries to simplify it.  We know
/// that PBI and BI are both conditional branches, and BI is in one of the
/// successor blocks of PBI - PBI branches to BI.
static bool SimplifyCondBranchToCondBranch(BranchInst *PBI, BranchInst *BI) {
  assert(PBI->isConditional() && BI->isConditional());
  BasicBlock *BB = BI->getParent();

  // If this block ends with a branch instruction, and if there is a
  // predecessor that ends on a branch of the same condition, make
  // this conditional branch redundant.
  if (PBI->getCondition() == BI->getCondition() &&
      PBI->getSuccessor(0) != PBI->getSuccessor(1)) {
    // Okay, the outcome of this conditional branch is statically
    // knowable.  If this block had a single pred, handle specially.
    if (BB->getSinglePredecessor()) {
      // Turn this into a branch on constant.
      bool CondIsTrue = PBI->getSuccessor(0) == BB;
      BI->setCondition(ConstantInt::get(Type::getInt1Ty(BB->getContext()),
                                        CondIsTrue));
      return true;  // Nuke the branch on constant.
    }

    // Otherwise, if there are multiple predecessors, insert a PHI that merges
    // in the constant and simplify the block result.  Subsequent passes of
    // simplifycfg will thread the block.
    if (BlockIsSimpleEnoughToThreadThrough(BB)) {
      pred_iterator PB = pred_begin(BB), PE = pred_end(BB);
      PHINode *NewPN = PHINode::Create(Type::getInt1Ty(BB->getContext()),
                                       std::distance(PB, PE),
                                       BI->getCondition()->getName() + ".pr",
                                       BB->begin());
      // Okay, we're going to insert the PHI node.  Since PBI is not the only
      // predecessor, compute the PHI'd conditional value for all of the preds.
      // Any predecessor where the condition is not computable we keep symbolic.
      for (pred_iterator PI = PB; PI != PE; ++PI) {
        BasicBlock *P = *PI;
        if ((PBI = dyn_cast<BranchInst>(P->getTerminator())) &&
            PBI != BI && PBI->isConditional() &&
            PBI->getCondition() == BI->getCondition() &&
            PBI->getSuccessor(0) != PBI->getSuccessor(1)) {
          bool CondIsTrue = PBI->getSuccessor(0) == BB;
          NewPN->addIncoming(ConstantInt::get(Type::getInt1Ty(BB->getContext()),
                                              CondIsTrue), P);
        } else {
          NewPN->addIncoming(BI->getCondition(), P);
        }
      }

      BI->setCondition(NewPN);
      return true;
    }
  }

  // If this is a conditional branch in an empty block, and if any
  // predecessors are a conditional branch to one of our destinations,
  // fold the conditions into logical ops and one cond br.
  BasicBlock::iterator BBI = BB->begin();
  // Ignore dbg intrinsics.
  while (isa<DbgInfoIntrinsic>(BBI))
    ++BBI;
  if (&*BBI != BI)
    return false;


  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(BI->getCondition()))
    if (CE->canTrap())
      return false;

  int PBIOp, BIOp;
  if (PBI->getSuccessor(0) == BI->getSuccessor(0))
    PBIOp = BIOp = 0;
  else if (PBI->getSuccessor(0) == BI->getSuccessor(1))
    PBIOp = 0, BIOp = 1;
  else if (PBI->getSuccessor(1) == BI->getSuccessor(0))
    PBIOp = 1, BIOp = 0;
  else if (PBI->getSuccessor(1) == BI->getSuccessor(1))
    PBIOp = BIOp = 1;
  else
    return false;

  // Check to make sure that the other destination of this branch
  // isn't BB itself.  If so, this is an infinite loop that will
  // keep getting unwound.
  if (PBI->getSuccessor(PBIOp) == BB)
    return false;

  // Do not perform this transformation if it would require
  // insertion of a large number of select instructions. For targets
  // without predication/cmovs, this is a big pessimization.

  // Also do not perform this transformation if any phi node in the common
  // destination block can trap when reached by BB or PBB (PR17073). In that
  // case, it would be unsafe to hoist the operation into a select instruction.

  BasicBlock *CommonDest = PBI->getSuccessor(PBIOp);
  unsigned NumPhis = 0;
  for (BasicBlock::iterator II = CommonDest->begin();
       isa<PHINode>(II); ++II, ++NumPhis) {
    if (NumPhis > 2) // Disable this xform.
      return false;

    PHINode *PN = cast<PHINode>(II);
    Value *BIV = PN->getIncomingValueForBlock(BB);
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(BIV))
      if (CE->canTrap())
        return false;

    unsigned PBBIdx = PN->getBasicBlockIndex(PBI->getParent());
    Value *PBIV = PN->getIncomingValue(PBBIdx);
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(PBIV))
      if (CE->canTrap())
        return false;
  }

  // Finally, if everything is ok, fold the branches to logical ops.
  BasicBlock *OtherDest = BI->getSuccessor(BIOp ^ 1);

  DEBUG(dbgs() << "FOLDING BRs:" << *PBI->getParent()
               << "AND: " << *BI->getParent());


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
    BasicBlock *InfLoopBlock = BasicBlock::Create(BB->getContext(),
                                                  "infloop", BB->getParent());
    BranchInst::Create(InfLoopBlock, InfLoopBlock);
    OtherDest = InfLoopBlock;
  }

  DEBUG(dbgs() << *PBI->getParent()->getParent());

  // BI may have other predecessors.  Because of this, we leave
  // it alone, but modify PBI.

  // Make sure we get to CommonDest on True&True directions.
  Value *PBICond = PBI->getCondition();
  IRBuilder<true, NoFolder> Builder(PBI);
  if (PBIOp)
    PBICond = Builder.CreateNot(PBICond, PBICond->getName()+".not");

  Value *BICond = BI->getCondition();
  if (BIOp)
    BICond = Builder.CreateNot(BICond, BICond->getName()+".not");

  // Merge the conditions.
  Value *Cond = Builder.CreateOr(PBICond, BICond, "brmerge");

  // Modify PBI to branch on the new condition to the new dests.
  PBI->setCondition(Cond);
  PBI->setSuccessor(0, CommonDest);
  PBI->setSuccessor(1, OtherDest);

  // Update branch weight for PBI.
  uint64_t PredTrueWeight, PredFalseWeight, SuccTrueWeight, SuccFalseWeight;
  bool PredHasWeights = ExtractBranchMetadata(PBI, PredTrueWeight,
                                              PredFalseWeight);
  bool SuccHasWeights = ExtractBranchMetadata(BI, SuccTrueWeight,
                                              SuccFalseWeight);
  if (PredHasWeights && SuccHasWeights) {
    uint64_t PredCommon = PBIOp ? PredFalseWeight : PredTrueWeight;
    uint64_t PredOther = PBIOp ?PredTrueWeight : PredFalseWeight;
    uint64_t SuccCommon = BIOp ? SuccFalseWeight : SuccTrueWeight;
    uint64_t SuccOther = BIOp ? SuccTrueWeight : SuccFalseWeight;
    // The weight to CommonDest should be PredCommon * SuccTotal +
    //                                    PredOther * SuccCommon.
    // The weight to OtherDest should be PredOther * SuccOther.
    SmallVector<uint64_t, 2> NewWeights;
    NewWeights.push_back(PredCommon * (SuccCommon + SuccOther) +
                         PredOther * SuccCommon);
    NewWeights.push_back(PredOther * SuccOther);
    // Halve the weights if any of them cannot fit in an uint32_t
    FitWeights(NewWeights);

    SmallVector<uint32_t, 2> MDWeights(NewWeights.begin(),NewWeights.end());
    PBI->setMetadata(LLVMContext::MD_prof,
                     MDBuilder(BI->getContext()).
                     createBranchWeights(MDWeights));
  }

  // OtherDest may have phi nodes.  If so, add an entry from PBI's
  // block that are identical to the entries for BI's block.
  AddPredecessorToBlock(OtherDest, PBI->getParent(), BB);

  // We know that the CommonDest already had an edge from PBI to
  // it.  If it has PHIs though, the PHIs may have different
  // entries for BB and PBI's BB.  If so, insert a select to make
  // them agree.
  PHINode *PN;
  for (BasicBlock::iterator II = CommonDest->begin();
       (PN = dyn_cast<PHINode>(II)); ++II) {
    Value *BIV = PN->getIncomingValueForBlock(BB);
    unsigned PBBIdx = PN->getBasicBlockIndex(PBI->getParent());
    Value *PBIV = PN->getIncomingValue(PBBIdx);
    if (BIV != PBIV) {
      // Insert a select in PBI to pick the right value.
      Value *NV = cast<SelectInst>
        (Builder.CreateSelect(PBICond, PBIV, BIV, PBIV->getName()+".mux"));
      PN->setIncomingValue(PBBIdx, NV);
    }
  }

  DEBUG(dbgs() << "INTO: " << *PBI->getParent());
  DEBUG(dbgs() << *PBI->getParent()->getParent());

  // This basic block is probably dead.  We know it has at least
  // one fewer predecessor.
  return true;
}

// SimplifyTerminatorOnSelect - Simplifies a terminator by replacing it with a
// branch to TrueBB if Cond is true or to FalseBB if Cond is false.
// Takes care of updating the successors and removing the old terminator.
// Also makes sure not to introduce new successors by assuming that edges to
// non-successor TrueBBs and FalseBBs aren't reachable.
static bool SimplifyTerminatorOnSelect(TerminatorInst *OldTerm, Value *Cond,
                                       BasicBlock *TrueBB, BasicBlock *FalseBB,
                                       uint32_t TrueWeight,
                                       uint32_t FalseWeight){
  // Remove any superfluous successor edges from the CFG.
  // First, figure out which successors to preserve.
  // If TrueBB and FalseBB are equal, only try to preserve one copy of that
  // successor.
  BasicBlock *KeepEdge1 = TrueBB;
  BasicBlock *KeepEdge2 = TrueBB != FalseBB ? FalseBB : nullptr;

  // Then remove the rest.
  for (unsigned I = 0, E = OldTerm->getNumSuccessors(); I != E; ++I) {
    BasicBlock *Succ = OldTerm->getSuccessor(I);
    // Make sure only to keep exactly one copy of each edge.
    if (Succ == KeepEdge1)
      KeepEdge1 = nullptr;
    else if (Succ == KeepEdge2)
      KeepEdge2 = nullptr;
    else
      Succ->removePredecessor(OldTerm->getParent());
  }

  IRBuilder<> Builder(OldTerm);
  Builder.SetCurrentDebugLocation(OldTerm->getDebugLoc());

  // Insert an appropriate new terminator.
  if (!KeepEdge1 && !KeepEdge2) {
    if (TrueBB == FalseBB)
      // We were only looking for one successor, and it was present.
      // Create an unconditional branch to it.
      Builder.CreateBr(TrueBB);
    else {
      // We found both of the successors we were looking for.
      // Create a conditional branch sharing the condition of the select.
      BranchInst *NewBI = Builder.CreateCondBr(Cond, TrueBB, FalseBB);
      if (TrueWeight != FalseWeight)
        NewBI->setMetadata(LLVMContext::MD_prof,
                           MDBuilder(OldTerm->getContext()).
                           createBranchWeights(TrueWeight, FalseWeight));
    }
  } else if (KeepEdge1 && (KeepEdge2 || TrueBB == FalseBB)) {
    // Neither of the selected blocks were successors, so this
    // terminator must be unreachable.
    new UnreachableInst(OldTerm->getContext(), OldTerm);
  } else {
    // One of the selected values was a successor, but the other wasn't.
    // Insert an unconditional branch to the one that was found;
    // the edge to the one that wasn't must be unreachable.
    if (!KeepEdge1)
      // Only TrueBB was found.
      Builder.CreateBr(TrueBB);
    else
      // Only FalseBB was found.
      Builder.CreateBr(FalseBB);
  }

  EraseTerminatorInstAndDCECond(OldTerm);
  return true;
}

// SimplifySwitchOnSelect - Replaces
//   (switch (select cond, X, Y)) on constant X, Y
// with a branch - conditional if X and Y lead to distinct BBs,
// unconditional otherwise.
static bool SimplifySwitchOnSelect(SwitchInst *SI, SelectInst *Select) {
  // Check for constant integer values in the select.
  ConstantInt *TrueVal = dyn_cast<ConstantInt>(Select->getTrueValue());
  ConstantInt *FalseVal = dyn_cast<ConstantInt>(Select->getFalseValue());
  if (!TrueVal || !FalseVal)
    return false;

  // Find the relevant condition and destinations.
  Value *Condition = Select->getCondition();
  BasicBlock *TrueBB = SI->findCaseValue(TrueVal).getCaseSuccessor();
  BasicBlock *FalseBB = SI->findCaseValue(FalseVal).getCaseSuccessor();

  // Get weight for TrueBB and FalseBB.
  uint32_t TrueWeight = 0, FalseWeight = 0;
  SmallVector<uint64_t, 8> Weights;
  bool HasWeights = HasBranchWeights(SI);
  if (HasWeights) {
    GetBranchWeights(SI, Weights);
    if (Weights.size() == 1 + SI->getNumCases()) {
      TrueWeight = (uint32_t)Weights[SI->findCaseValue(TrueVal).
                                     getSuccessorIndex()];
      FalseWeight = (uint32_t)Weights[SI->findCaseValue(FalseVal).
                                      getSuccessorIndex()];
    }
  }

  // Perform the actual simplification.
  return SimplifyTerminatorOnSelect(SI, Condition, TrueBB, FalseBB,
                                    TrueWeight, FalseWeight);
}

// SimplifyIndirectBrOnSelect - Replaces
//   (indirectbr (select cond, blockaddress(@fn, BlockA),
//                             blockaddress(@fn, BlockB)))
// with
//   (br cond, BlockA, BlockB).
static bool SimplifyIndirectBrOnSelect(IndirectBrInst *IBI, SelectInst *SI) {
  // Check that both operands of the select are block addresses.
  BlockAddress *TBA = dyn_cast<BlockAddress>(SI->getTrueValue());
  BlockAddress *FBA = dyn_cast<BlockAddress>(SI->getFalseValue());
  if (!TBA || !FBA)
    return false;

  // Extract the actual blocks.
  BasicBlock *TrueBB = TBA->getBasicBlock();
  BasicBlock *FalseBB = FBA->getBasicBlock();

  // Perform the actual simplification.
  return SimplifyTerminatorOnSelect(IBI, SI->getCondition(), TrueBB, FalseBB,
                                    0, 0);
}

/// TryToSimplifyUncondBranchWithICmpInIt - This is called when we find an icmp
/// instruction (a seteq/setne with a constant) as the only instruction in a
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
static bool TryToSimplifyUncondBranchWithICmpInIt(
    ICmpInst *ICI, IRBuilder<> &Builder, const TargetTransformInfo &TTI,
    const DataLayout *DL) {
  BasicBlock *BB = ICI->getParent();

  // If the block has any PHIs in it or the icmp has multiple uses, it is too
  // complex.
  if (isa<PHINode>(BB->begin()) || !ICI->hasOneUse()) return false;

  Value *V = ICI->getOperand(0);
  ConstantInt *Cst = cast<ConstantInt>(ICI->getOperand(1));

  // The pattern we're looking for is where our only predecessor is a switch on
  // 'V' and this block is the default case for the switch.  In this case we can
  // fold the compared value into the switch to simplify things.
  BasicBlock *Pred = BB->getSinglePredecessor();
  if (!Pred || !isa<SwitchInst>(Pred->getTerminator())) return false;

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

    if (Value *V = SimplifyInstruction(ICI, DL)) {
      ICI->replaceAllUsesWith(V);
      ICI->eraseFromParent();
    }
    // BB is now empty, so it is likely to simplify away.
    return SimplifyCFG(BB, TTI, DL) | true;
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
    return SimplifyCFG(BB, TTI, DL) | true;
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
  Constant *NewCst     = ConstantInt::getFalse(BB->getContext());

  if (ICI->getPredicate() == ICmpInst::ICMP_EQ)
    std::swap(DefaultCst, NewCst);

  // Replace ICI (which is used by the PHI for the default value) with true or
  // false depending on if it is EQ or NE.
  ICI->replaceAllUsesWith(DefaultCst);
  ICI->eraseFromParent();

  // Okay, the switch goes to this block on a default value.  Add an edge from
  // the switch to the merge point on the compared value.
  BasicBlock *NewBB = BasicBlock::Create(BB->getContext(), "switch.edge",
                                         BB->getParent(), BB);
  SmallVector<uint64_t, 8> Weights;
  bool HasWeights = HasBranchWeights(SI);
  if (HasWeights) {
    GetBranchWeights(SI, Weights);
    if (Weights.size() == 1 + SI->getNumCases()) {
      // Split weight for default case to case for "Cst".
      Weights[0] = (Weights[0]+1) >> 1;
      Weights.push_back(Weights[0]);

      SmallVector<uint32_t, 8> MDWeights(Weights.begin(), Weights.end());
      SI->setMetadata(LLVMContext::MD_prof,
                      MDBuilder(SI->getContext()).
                      createBranchWeights(MDWeights));
    }
  }
  SI->addCase(Cst, NewBB);

  // NewBB branches to the phi block, add the uncond branch and the phi entry.
  Builder.SetInsertPoint(NewBB);
  Builder.SetCurrentDebugLocation(SI->getDebugLoc());
  Builder.CreateBr(SuccBlock);
  PHIUse->addIncoming(NewCst, NewBB);
  return true;
}

/// SimplifyBranchOnICmpChain - The specified branch is a conditional branch.
/// Check to see if it is branching on an or/and chain of icmp instructions, and
/// fold it into a switch instruction if so.
static bool SimplifyBranchOnICmpChain(BranchInst *BI, const DataLayout *DL,
                                      IRBuilder<> &Builder) {
  Instruction *Cond = dyn_cast<Instruction>(BI->getCondition());
  if (!Cond) return false;


  // Change br (X == 0 | X == 1), T, F into a switch instruction.
  // If this is a bunch of seteq's or'd together, or if it's a bunch of
  // 'setne's and'ed together, collect them.
  Value *CompVal = nullptr;
  std::vector<ConstantInt*> Values;
  bool TrueWhenEqual = true;
  Value *ExtraCase = nullptr;
  unsigned UsedICmps = 0;

  if (Cond->getOpcode() == Instruction::Or) {
    CompVal = GatherConstantCompares(Cond, Values, ExtraCase, DL, true,
                                     UsedICmps);
  } else if (Cond->getOpcode() == Instruction::And) {
    CompVal = GatherConstantCompares(Cond, Values, ExtraCase, DL, false,
                                     UsedICmps);
    TrueWhenEqual = false;
  }

  // If we didn't have a multiply compared value, fail.
  if (!CompVal) return false;

  // Avoid turning single icmps into a switch.
  if (UsedICmps <= 1)
    return false;

  // There might be duplicate constants in the list, which the switch
  // instruction can't handle, remove them now.
  array_pod_sort(Values.begin(), Values.end(), ConstantIntSortPredicate);
  Values.erase(std::unique(Values.begin(), Values.end()), Values.end());

  // If Extra was used, we require at least two switch values to do the
  // transformation.  A switch with one value is just an cond branch.
  if (ExtraCase && Values.size() < 2) return false;

  // TODO: Preserve branch weight metadata, similarly to how
  // FoldValueComparisonIntoPredecessors preserves it.

  // Figure out which block is which destination.
  BasicBlock *DefaultBB = BI->getSuccessor(1);
  BasicBlock *EdgeBB    = BI->getSuccessor(0);
  if (!TrueWhenEqual) std::swap(DefaultBB, EdgeBB);

  BasicBlock *BB = BI->getParent();

  DEBUG(dbgs() << "Converting 'icmp' chain with " << Values.size()
               << " cases into SWITCH.  BB is:\n" << *BB);

  // If there are any extra values that couldn't be folded into the switch
  // then we evaluate them with an explicit branch first.  Split the block
  // right before the condbr to handle it.
  if (ExtraCase) {
    BasicBlock *NewBB = BB->splitBasicBlock(BI, "switch.early.test");
    // Remove the uncond branch added to the old block.
    TerminatorInst *OldTI = BB->getTerminator();
    Builder.SetInsertPoint(OldTI);

    if (TrueWhenEqual)
      Builder.CreateCondBr(ExtraCase, EdgeBB, NewBB);
    else
      Builder.CreateCondBr(ExtraCase, NewBB, EdgeBB);

    OldTI->eraseFromParent();

    // If there are PHI nodes in EdgeBB, then we need to add a new entry to them
    // for the edge we just added.
    AddPredecessorToBlock(EdgeBB, BB, NewBB);

    DEBUG(dbgs() << "  ** 'icmp' chain unhandled condition: " << *ExtraCase
          << "\nEXTRABB = " << *BB);
    BB = NewBB;
  }

  Builder.SetInsertPoint(BI);
  // Convert pointer to int before we switch.
  if (CompVal->getType()->isPointerTy()) {
    assert(DL && "Cannot switch on pointer without DataLayout");
    CompVal = Builder.CreatePtrToInt(CompVal,
                                     DL->getIntPtrType(CompVal->getType()),
                                     "magicptr");
  }

  // Create the new switch instruction now.
  SwitchInst *New = Builder.CreateSwitch(CompVal, DefaultBB, Values.size());

  // Add all of the 'cases' to the switch instruction.
  for (unsigned i = 0, e = Values.size(); i != e; ++i)
    New->addCase(Values[i], EdgeBB);

  // We added edges from PI to the EdgeBB.  As such, if there were any
  // PHI nodes in EdgeBB, they need entries to be added corresponding to
  // the number of edges added.
  for (BasicBlock::iterator BBI = EdgeBB->begin();
       isa<PHINode>(BBI); ++BBI) {
    PHINode *PN = cast<PHINode>(BBI);
    Value *InVal = PN->getIncomingValueForBlock(BB);
    for (unsigned i = 0, e = Values.size()-1; i != e; ++i)
      PN->addIncoming(InVal, BB);
  }

  // Erase the old branch instruction.
  EraseTerminatorInstAndDCECond(BI);

  DEBUG(dbgs() << "  ** 'icmp' chain result is:\n" << *BB << '\n');
  return true;
}

bool SimplifyCFGOpt::SimplifyResume(ResumeInst *RI, IRBuilder<> &Builder) {
  // If this is a trivial landing pad that just continues unwinding the caught
  // exception then zap the landing pad, turning its invokes into calls.
  BasicBlock *BB = RI->getParent();
  LandingPadInst *LPInst = dyn_cast<LandingPadInst>(BB->getFirstNonPHI());
  if (RI->getValue() != LPInst)
    // Not a landing pad, or the resume is not unwinding the exception that
    // caused control to branch here.
    return false;

  // Check that there are no other instructions except for debug intrinsics.
  BasicBlock::iterator I = LPInst, E = RI;
  while (++I != E)
    if (!isa<DbgInfoIntrinsic>(I))
      return false;

  // Turn all invokes that unwind here into calls and delete the basic block.
  bool InvokeRequiresTableEntry = false;
  bool Changed = false;
  for (pred_iterator PI = pred_begin(BB), PE = pred_end(BB); PI != PE;) {
    InvokeInst *II = cast<InvokeInst>((*PI++)->getTerminator());

    if (II->hasFnAttr(Attribute::UWTable)) {
      // Don't remove an `invoke' instruction if the ABI requires an entry into
      // the table.
      InvokeRequiresTableEntry = true;
      continue;
    }

    SmallVector<Value*, 8> Args(II->op_begin(), II->op_end() - 3);

    // Insert a call instruction before the invoke.
    CallInst *Call = CallInst::Create(II->getCalledValue(), Args, "", II);
    Call->takeName(II);
    Call->setCallingConv(II->getCallingConv());
    Call->setAttributes(II->getAttributes());
    Call->setDebugLoc(II->getDebugLoc());

    // Anything that used the value produced by the invoke instruction now uses
    // the value produced by the call instruction.  Note that we do this even
    // for void functions and calls with no uses so that the callgraph edge is
    // updated.
    II->replaceAllUsesWith(Call);
    BB->removePredecessor(II->getParent());

    // Insert a branch to the normal destination right before the invoke.
    BranchInst::Create(II->getNormalDest(), II);

    // Finally, delete the invoke instruction!
    II->eraseFromParent();
    Changed = true;
  }

  if (!InvokeRequiresTableEntry)
    // The landingpad is now unreachable.  Zap it.
    BB->eraseFromParent();

  return Changed;
}

bool SimplifyCFGOpt::SimplifyReturn(ReturnInst *RI, IRBuilder<> &Builder) {
  BasicBlock *BB = RI->getParent();
  if (!BB->getFirstNonPHIOrDbg()->isTerminator()) return false;

  // Find predecessors that end with branches.
  SmallVector<BasicBlock*, 8> UncondBranchPreds;
  SmallVector<BranchInst*, 8> CondBranchPreds;
  for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI) {
    BasicBlock *P = *PI;
    TerminatorInst *PTI = P->getTerminator();
    if (BranchInst *BI = dyn_cast<BranchInst>(PTI)) {
      if (BI->isUnconditional())
        UncondBranchPreds.push_back(P);
      else
        CondBranchPreds.push_back(BI);
    }
  }

  // If we found some, do the transformation!
  if (!UncondBranchPreds.empty() && DupRet) {
    while (!UncondBranchPreds.empty()) {
      BasicBlock *Pred = UncondBranchPreds.pop_back_val();
      DEBUG(dbgs() << "FOLDING: " << *BB
            << "INTO UNCOND BRANCH PRED: " << *Pred);
      (void)FoldReturnIntoUncondBranch(RI, BB, Pred);
    }

    // If we eliminated all predecessors of the block, delete the block now.
    if (pred_begin(BB) == pred_end(BB))
      // We know there are no successors, so just nuke the block.
      BB->eraseFromParent();

    return true;
  }

  // Check out all of the conditional branches going to this return
  // instruction.  If any of them just select between returns, change the
  // branch itself into a select/return pair.
  while (!CondBranchPreds.empty()) {
    BranchInst *BI = CondBranchPreds.pop_back_val();

    // Check to see if the non-BB successor is also a return block.
    if (isa<ReturnInst>(BI->getSuccessor(0)->getTerminator()) &&
        isa<ReturnInst>(BI->getSuccessor(1)->getTerminator()) &&
        SimplifyCondBranchToTwoReturns(BI, Builder))
      return true;
  }
  return false;
}

bool SimplifyCFGOpt::SimplifyUnreachable(UnreachableInst *UI) {
  BasicBlock *BB = UI->getParent();

  bool Changed = false;

  // If there are any instructions immediately before the unreachable that can
  // be removed, do so.
  while (UI != BB->begin()) {
    BasicBlock::iterator BBI = UI;
    --BBI;
    // Do not delete instructions that can have side effects which might cause
    // the unreachable to not be reachable; specifically, calls and volatile
    // operations may have this effect.
    if (isa<CallInst>(BBI) && !isa<DbgInfoIntrinsic>(BBI)) break;

    if (BBI->mayHaveSideEffects()) {
      if (StoreInst *SI = dyn_cast<StoreInst>(BBI)) {
        if (SI->isVolatile())
          break;
      } else if (LoadInst *LI = dyn_cast<LoadInst>(BBI)) {
        if (LI->isVolatile())
          break;
      } else if (AtomicRMWInst *RMWI = dyn_cast<AtomicRMWInst>(BBI)) {
        if (RMWI->isVolatile())
          break;
      } else if (AtomicCmpXchgInst *CXI = dyn_cast<AtomicCmpXchgInst>(BBI)) {
        if (CXI->isVolatile())
          break;
      } else if (!isa<FenceInst>(BBI) && !isa<VAArgInst>(BBI) &&
                 !isa<LandingPadInst>(BBI)) {
        break;
      }
      // Note that deleting LandingPad's here is in fact okay, although it
      // involves a bit of subtle reasoning. If this inst is a LandingPad,
      // all the predecessors of this block will be the unwind edges of Invokes,
      // and we can therefore guarantee this block will be erased.
    }

    // Delete this instruction (any uses are guaranteed to be dead)
    if (!BBI->use_empty())
      BBI->replaceAllUsesWith(UndefValue::get(BBI->getType()));
    BBI->eraseFromParent();
    Changed = true;
  }

  // If the unreachable instruction is the first in the block, take a gander
  // at all of the predecessors of this instruction, and simplify them.
  if (&BB->front() != UI) return Changed;

  SmallVector<BasicBlock*, 8> Preds(pred_begin(BB), pred_end(BB));
  for (unsigned i = 0, e = Preds.size(); i != e; ++i) {
    TerminatorInst *TI = Preds[i]->getTerminator();
    IRBuilder<> Builder(TI);
    if (BranchInst *BI = dyn_cast<BranchInst>(TI)) {
      if (BI->isUnconditional()) {
        if (BI->getSuccessor(0) == BB) {
          new UnreachableInst(TI->getContext(), TI);
          TI->eraseFromParent();
          Changed = true;
        }
      } else {
        if (BI->getSuccessor(0) == BB) {
          Builder.CreateBr(BI->getSuccessor(1));
          EraseTerminatorInstAndDCECond(BI);
        } else if (BI->getSuccessor(1) == BB) {
          Builder.CreateBr(BI->getSuccessor(0));
          EraseTerminatorInstAndDCECond(BI);
          Changed = true;
        }
      }
    } else if (SwitchInst *SI = dyn_cast<SwitchInst>(TI)) {
      for (SwitchInst::CaseIt i = SI->case_begin(), e = SI->case_end();
           i != e; ++i)
        if (i.getCaseSuccessor() == BB) {
          BB->removePredecessor(SI->getParent());
          SI->removeCase(i);
          --i; --e;
          Changed = true;
        }
      // If the default value is unreachable, figure out the most popular
      // destination and make it the default.
      if (SI->getDefaultDest() == BB) {
        std::map<BasicBlock*, std::pair<unsigned, unsigned> > Popularity;
        for (SwitchInst::CaseIt i = SI->case_begin(), e = SI->case_end();
             i != e; ++i) {
          std::pair<unsigned, unsigned> &entry =
              Popularity[i.getCaseSuccessor()];
          if (entry.first == 0) {
            entry.first = 1;
            entry.second = i.getCaseIndex();
          } else {
            entry.first++;
          }
        }

        // Find the most popular block.
        unsigned MaxPop = 0;
        unsigned MaxIndex = 0;
        BasicBlock *MaxBlock = nullptr;
        for (std::map<BasicBlock*, std::pair<unsigned, unsigned> >::iterator
             I = Popularity.begin(), E = Popularity.end(); I != E; ++I) {
          if (I->second.first > MaxPop ||
              (I->second.first == MaxPop && MaxIndex > I->second.second)) {
            MaxPop = I->second.first;
            MaxIndex = I->second.second;
            MaxBlock = I->first;
          }
        }
        if (MaxBlock) {
          // Make this the new default, allowing us to delete any explicit
          // edges to it.
          SI->setDefaultDest(MaxBlock);
          Changed = true;

          // If MaxBlock has phinodes in it, remove MaxPop-1 entries from
          // it.
          if (isa<PHINode>(MaxBlock->begin()))
            for (unsigned i = 0; i != MaxPop-1; ++i)
              MaxBlock->removePredecessor(SI->getParent());

          for (SwitchInst::CaseIt i = SI->case_begin(), e = SI->case_end();
               i != e; ++i)
            if (i.getCaseSuccessor() == MaxBlock) {
              SI->removeCase(i);
              --i; --e;
            }
        }
      }
    } else if (InvokeInst *II = dyn_cast<InvokeInst>(TI)) {
      if (II->getUnwindDest() == BB) {
        // Convert the invoke to a call instruction.  This would be a good
        // place to note that the call does not throw though.
        BranchInst *BI = Builder.CreateBr(II->getNormalDest());
        II->removeFromParent();   // Take out of symbol table

        // Insert the call now...
        SmallVector<Value*, 8> Args(II->op_begin(), II->op_end()-3);
        Builder.SetInsertPoint(BI);
        CallInst *CI = Builder.CreateCall(II->getCalledValue(),
                                          Args, II->getName());
        CI->setCallingConv(II->getCallingConv());
        CI->setAttributes(II->getAttributes());
        // If the invoke produced a value, the call does now instead.
        II->replaceAllUsesWith(CI);
        delete II;
        Changed = true;
      }
    }
  }

  // If this block is now dead, remove it.
  if (pred_begin(BB) == pred_end(BB) &&
      BB != &BB->getParent()->getEntryBlock()) {
    // We know there are no successors, so just nuke the block.
    BB->eraseFromParent();
    return true;
  }

  return Changed;
}

/// TurnSwitchRangeIntoICmp - Turns a switch with that contains only a
/// integer range comparison into a sub, an icmp and a branch.
static bool TurnSwitchRangeIntoICmp(SwitchInst *SI, IRBuilder<> &Builder) {
  assert(SI->getNumCases() > 1 && "Degenerate switch?");

  // Make sure all cases point to the same destination and gather the values.
  SmallVector<ConstantInt *, 16> Cases;
  SwitchInst::CaseIt I = SI->case_begin();
  Cases.push_back(I.getCaseValue());
  SwitchInst::CaseIt PrevI = I++;
  for (SwitchInst::CaseIt E = SI->case_end(); I != E; PrevI = I++) {
    if (PrevI.getCaseSuccessor() != I.getCaseSuccessor())
      return false;
    Cases.push_back(I.getCaseValue());
  }
  assert(Cases.size() == SI->getNumCases() && "Not all cases gathered");

  // Sort the case values, then check if they form a range we can transform.
  array_pod_sort(Cases.begin(), Cases.end(), ConstantIntSortPredicate);
  for (unsigned I = 1, E = Cases.size(); I != E; ++I) {
    if (Cases[I-1]->getValue() != Cases[I]->getValue()+1)
      return false;
  }

  Constant *Offset = ConstantExpr::getNeg(Cases.back());
  Constant *NumCases = ConstantInt::get(Offset->getType(), SI->getNumCases());

  Value *Sub = SI->getCondition();
  if (!Offset->isNullValue())
    Sub = Builder.CreateAdd(Sub, Offset, Sub->getName()+".off");
  Value *Cmp;
  // If NumCases overflowed, then all possible values jump to the successor.
  if (NumCases->isNullValue() && SI->getNumCases() != 0)
    Cmp = ConstantInt::getTrue(SI->getContext());
  else
    Cmp = Builder.CreateICmpULT(Sub, NumCases, "switch");
  BranchInst *NewBI = Builder.CreateCondBr(
      Cmp, SI->case_begin().getCaseSuccessor(), SI->getDefaultDest());

  // Update weight for the newly-created conditional branch.
  SmallVector<uint64_t, 8> Weights;
  bool HasWeights = HasBranchWeights(SI);
  if (HasWeights) {
    GetBranchWeights(SI, Weights);
    if (Weights.size() == 1 + SI->getNumCases()) {
      // Combine all weights for the cases to be the true weight of NewBI.
      // We assume that the sum of all weights for a Terminator can fit into 32
      // bits.
      uint32_t NewTrueWeight = 0;
      for (unsigned I = 1, E = Weights.size(); I != E; ++I)
        NewTrueWeight += (uint32_t)Weights[I];
      NewBI->setMetadata(LLVMContext::MD_prof,
                         MDBuilder(SI->getContext()).
                         createBranchWeights(NewTrueWeight,
                                             (uint32_t)Weights[0]));
    }
  }

  // Prune obsolete incoming values off the successor's PHI nodes.
  for (BasicBlock::iterator BBI = SI->case_begin().getCaseSuccessor()->begin();
       isa<PHINode>(BBI); ++BBI) {
    for (unsigned I = 0, E = SI->getNumCases()-1; I != E; ++I)
      cast<PHINode>(BBI)->removeIncomingValue(SI->getParent());
  }
  SI->eraseFromParent();

  return true;
}

/// EliminateDeadSwitchCases - Compute masked bits for the condition of a switch
/// and use it to remove dead cases.
static bool EliminateDeadSwitchCases(SwitchInst *SI) {
  Value *Cond = SI->getCondition();
  unsigned Bits = Cond->getType()->getIntegerBitWidth();
  APInt KnownZero(Bits, 0), KnownOne(Bits, 0);
  computeKnownBits(Cond, KnownZero, KnownOne);

  // Gather dead cases.
  SmallVector<ConstantInt*, 8> DeadCases;
  for (SwitchInst::CaseIt I = SI->case_begin(), E = SI->case_end(); I != E; ++I) {
    if ((I.getCaseValue()->getValue() & KnownZero) != 0 ||
        (I.getCaseValue()->getValue() & KnownOne) != KnownOne) {
      DeadCases.push_back(I.getCaseValue());
      DEBUG(dbgs() << "SimplifyCFG: switch case '"
                   << I.getCaseValue() << "' is dead.\n");
    }
  }

  SmallVector<uint64_t, 8> Weights;
  bool HasWeight = HasBranchWeights(SI);
  if (HasWeight) {
    GetBranchWeights(SI, Weights);
    HasWeight = (Weights.size() == 1 + SI->getNumCases());
  }

  // Remove dead cases from the switch.
  for (unsigned I = 0, E = DeadCases.size(); I != E; ++I) {
    SwitchInst::CaseIt Case = SI->findCaseValue(DeadCases[I]);
    assert(Case != SI->case_default() &&
           "Case was not found. Probably mistake in DeadCases forming.");
    if (HasWeight) {
      std::swap(Weights[Case.getCaseIndex()+1], Weights.back());
      Weights.pop_back();
    }

    // Prune unused values from PHI nodes.
    Case.getCaseSuccessor()->removePredecessor(SI->getParent());
    SI->removeCase(Case);
  }
  if (HasWeight && Weights.size() >= 2) {
    SmallVector<uint32_t, 8> MDWeights(Weights.begin(), Weights.end());
    SI->setMetadata(LLVMContext::MD_prof,
                    MDBuilder(SI->getParent()->getContext()).
                    createBranchWeights(MDWeights));
  }

  return !DeadCases.empty();
}

/// FindPHIForConditionForwarding - If BB would be eligible for simplification
/// by TryToSimplifyUncondBranchFromEmptyBlock (i.e. it is empty and terminated
/// by an unconditional branch), look at the phi node for BB in the successor
/// block and see if the incoming value is equal to CaseValue. If so, return
/// the phi node, and set PhiIndex to BB's index in the phi node.
static PHINode *FindPHIForConditionForwarding(ConstantInt *CaseValue,
                                              BasicBlock *BB,
                                              int *PhiIndex) {
  if (BB->getFirstNonPHIOrDbg() != BB->getTerminator())
    return nullptr; // BB must be empty to be a candidate for simplification.
  if (!BB->getSinglePredecessor())
    return nullptr; // BB must be dominated by the switch.

  BranchInst *Branch = dyn_cast<BranchInst>(BB->getTerminator());
  if (!Branch || !Branch->isUnconditional())
    return nullptr; // Terminator must be unconditional branch.

  BasicBlock *Succ = Branch->getSuccessor(0);

  BasicBlock::iterator I = Succ->begin();
  while (PHINode *PHI = dyn_cast<PHINode>(I++)) {
    int Idx = PHI->getBasicBlockIndex(BB);
    assert(Idx >= 0 && "PHI has no entry for predecessor?");

    Value *InValue = PHI->getIncomingValue(Idx);
    if (InValue != CaseValue) continue;

    *PhiIndex = Idx;
    return PHI;
  }

  return nullptr;
}

/// ForwardSwitchConditionToPHI - Try to forward the condition of a switch
/// instruction to a phi node dominated by the switch, if that would mean that
/// some of the destination blocks of the switch can be folded away.
/// Returns true if a change is made.
static bool ForwardSwitchConditionToPHI(SwitchInst *SI) {
  typedef DenseMap<PHINode*, SmallVector<int,4> > ForwardingNodesMap;
  ForwardingNodesMap ForwardingNodes;

  for (SwitchInst::CaseIt I = SI->case_begin(), E = SI->case_end(); I != E; ++I) {
    ConstantInt *CaseValue = I.getCaseValue();
    BasicBlock *CaseDest = I.getCaseSuccessor();

    int PhiIndex;
    PHINode *PHI = FindPHIForConditionForwarding(CaseValue, CaseDest,
                                                 &PhiIndex);
    if (!PHI) continue;

    ForwardingNodes[PHI].push_back(PhiIndex);
  }

  bool Changed = false;

  for (ForwardingNodesMap::iterator I = ForwardingNodes.begin(),
       E = ForwardingNodes.end(); I != E; ++I) {
    PHINode *Phi = I->first;
    SmallVectorImpl<int> &Indexes = I->second;

    if (Indexes.size() < 2) continue;

    for (size_t I = 0, E = Indexes.size(); I != E; ++I)
      Phi->setIncomingValue(Indexes[I], SI->getCondition());
    Changed = true;
  }

  return Changed;
}

/// ValidLookupTableConstant - Return true if the backend will be able to handle
/// initializing an array of constants like C.
static bool ValidLookupTableConstant(Constant *C) {
  if (C->isThreadDependent())
    return false;
  if (C->isDLLImportDependent())
    return false;

  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C))
    return CE->isGEPWithNoNotionalOverIndexing();

  return isa<ConstantFP>(C) ||
      isa<ConstantInt>(C) ||
      isa<ConstantPointerNull>(C) ||
      isa<GlobalValue>(C) ||
      isa<UndefValue>(C);
}

/// LookupConstant - If V is a Constant, return it. Otherwise, try to look up
/// its constant value in ConstantPool, returning 0 if it's not there.
static Constant *LookupConstant(Value *V,
                         const SmallDenseMap<Value*, Constant*>& ConstantPool) {
  if (Constant *C = dyn_cast<Constant>(V))
    return C;
  return ConstantPool.lookup(V);
}

/// ConstantFold - Try to fold instruction I into a constant. This works for
/// simple instructions such as binary operations where both operands are
/// constant or can be replaced by constants from the ConstantPool. Returns the
/// resulting constant on success, 0 otherwise.
static Constant *
ConstantFold(Instruction *I,
             const SmallDenseMap<Value *, Constant *> &ConstantPool,
             const DataLayout *DL) {
  if (SelectInst *Select = dyn_cast<SelectInst>(I)) {
    Constant *A = LookupConstant(Select->getCondition(), ConstantPool);
    if (!A)
      return nullptr;
    if (A->isAllOnesValue())
      return LookupConstant(Select->getTrueValue(), ConstantPool);
    if (A->isNullValue())
      return LookupConstant(Select->getFalseValue(), ConstantPool);
    return nullptr;
  }

  SmallVector<Constant *, 4> COps;
  for (unsigned N = 0, E = I->getNumOperands(); N != E; ++N) {
    if (Constant *A = LookupConstant(I->getOperand(N), ConstantPool))
      COps.push_back(A);
    else
      return nullptr;
  }

  if (CmpInst *Cmp = dyn_cast<CmpInst>(I))
    return ConstantFoldCompareInstOperands(Cmp->getPredicate(), COps[0],
                                           COps[1], DL);

  return ConstantFoldInstOperands(I->getOpcode(), I->getType(), COps, DL);
}

/// GetCaseResults - Try to determine the resulting constant values in phi nodes
/// at the common destination basic block, *CommonDest, for one of the case
/// destionations CaseDest corresponding to value CaseVal (0 for the default
/// case), of a switch instruction SI.
static bool
GetCaseResults(SwitchInst *SI,
               ConstantInt *CaseVal,
               BasicBlock *CaseDest,
               BasicBlock **CommonDest,
               SmallVectorImpl<std::pair<PHINode *, Constant *> > &Res,
               const DataLayout *DL) {
  // The block from which we enter the common destination.
  BasicBlock *Pred = SI->getParent();

  // If CaseDest is empty except for some side-effect free instructions through
  // which we can constant-propagate the CaseVal, continue to its successor.
  SmallDenseMap<Value*, Constant*> ConstantPool;
  ConstantPool.insert(std::make_pair(SI->getCondition(), CaseVal));
  for (BasicBlock::iterator I = CaseDest->begin(), E = CaseDest->end(); I != E;
       ++I) {
    if (TerminatorInst *T = dyn_cast<TerminatorInst>(I)) {
      // If the terminator is a simple branch, continue to the next block.
      if (T->getNumSuccessors() != 1)
        return false;
      Pred = CaseDest;
      CaseDest = T->getSuccessor(0);
    } else if (isa<DbgInfoIntrinsic>(I)) {
      // Skip debug intrinsic.
      continue;
    } else if (Constant *C = ConstantFold(I, ConstantPool, DL)) {
      // Instruction is side-effect free and constant.
      ConstantPool.insert(std::make_pair(I, C));
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
  BasicBlock::iterator I = (*CommonDest)->begin();
  while (PHINode *PHI = dyn_cast<PHINode>(I++)) {
    int Idx = PHI->getBasicBlockIndex(Pred);
    if (Idx == -1)
      continue;

    Constant *ConstVal = LookupConstant(PHI->getIncomingValue(Idx),
                                        ConstantPool);
    if (!ConstVal)
      return false;

    // Note: If the constant comes from constant-propagating the case value
    // through the CaseDest basic block, it will be safe to remove the
    // instructions in that block. They cannot be used (except in the phi nodes
    // we visit) outside CaseDest, because that block does not dominate its
    // successor. If it did, we would not be in this phi node.

    // Be conservative about which kinds of constants we support.
    if (!ValidLookupTableConstant(ConstVal))
      return false;

    Res.push_back(std::make_pair(PHI, ConstVal));
  }

  return Res.size() > 0;
}

namespace {
  /// SwitchLookupTable - This class represents a lookup table that can be used
  /// to replace a switch.
  class SwitchLookupTable {
  public:
    /// SwitchLookupTable - Create a lookup table to use as a switch replacement
    /// with the contents of Values, using DefaultValue to fill any holes in the
    /// table.
    SwitchLookupTable(Module &M,
                      uint64_t TableSize,
                      ConstantInt *Offset,
             const SmallVectorImpl<std::pair<ConstantInt*, Constant*> >& Values,
                      Constant *DefaultValue,
                      const DataLayout *DL);

    /// BuildLookup - Build instructions with Builder to retrieve the value at
    /// the position given by Index in the lookup table.
    Value *BuildLookup(Value *Index, IRBuilder<> &Builder);

    /// WouldFitInRegister - Return true if a table with TableSize elements of
    /// type ElementType would fit in a target-legal register.
    static bool WouldFitInRegister(const DataLayout *DL,
                                   uint64_t TableSize,
                                   const Type *ElementType);

  private:
    // Depending on the contents of the table, it can be represented in
    // different ways.
    enum {
      // For tables where each element contains the same value, we just have to
      // store that single value and return it for each lookup.
      SingleValueKind,

      // For small tables with integer elements, we can pack them into a bitmap
      // that fits into a target-legal register. Values are retrieved by
      // shift and mask operations.
      BitMapKind,

      // The table is stored as an array of values. Values are retrieved by load
      // instructions from the table.
      ArrayKind
    } Kind;

    // For SingleValueKind, this is the single value.
    Constant *SingleValue;

    // For BitMapKind, this is the bitmap.
    ConstantInt *BitMap;
    IntegerType *BitMapElementTy;

    // For ArrayKind, this is the array.
    GlobalVariable *Array;
  };
}

SwitchLookupTable::SwitchLookupTable(Module &M,
                                     uint64_t TableSize,
                                     ConstantInt *Offset,
             const SmallVectorImpl<std::pair<ConstantInt*, Constant*> >& Values,
                                     Constant *DefaultValue,
                                     const DataLayout *DL)
    : SingleValue(nullptr), BitMap(nullptr), BitMapElementTy(nullptr),
      Array(nullptr) {
  assert(Values.size() && "Can't build lookup table without values!");
  assert(TableSize >= Values.size() && "Can't fit values in table!");

  // If all values in the table are equal, this is that value.
  SingleValue = Values.begin()->second;

  Type *ValueType = Values.begin()->second->getType();

  // Build up the table contents.
  SmallVector<Constant*, 64> TableContents(TableSize);
  for (size_t I = 0, E = Values.size(); I != E; ++I) {
    ConstantInt *CaseVal = Values[I].first;
    Constant *CaseRes = Values[I].second;
    assert(CaseRes->getType() == ValueType);

    uint64_t Idx = (CaseVal->getValue() - Offset->getValue())
                   .getLimitedValue();
    TableContents[Idx] = CaseRes;

    if (CaseRes != SingleValue)
      SingleValue = nullptr;
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

    if (DefaultValue != SingleValue)
      SingleValue = nullptr;
  }

  // If each element in the table contains the same value, we only need to store
  // that single value.
  if (SingleValue) {
    Kind = SingleValueKind;
    return;
  }

  // If the type is integer and the table fits in a register, build a bitmap.
  if (WouldFitInRegister(DL, TableSize, ValueType)) {
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
  ArrayType *ArrayTy = ArrayType::get(ValueType, TableSize);
  Constant *Initializer = ConstantArray::get(ArrayTy, TableContents);

  Array = new GlobalVariable(M, ArrayTy, /*constant=*/ true,
                             GlobalVariable::PrivateLinkage,
                             Initializer,
                             "switch.table");
  Array->setUnnamedAddr(true);
  Kind = ArrayKind;
}

Value *SwitchLookupTable::BuildLookup(Value *Index, IRBuilder<> &Builder) {
  switch (Kind) {
    case SingleValueKind:
      return SingleValue;
    case BitMapKind: {
      // Type of the bitmap (e.g. i59).
      IntegerType *MapTy = BitMap->getType();

      // Cast Index to the same type as the bitmap.
      // Note: The Index is <= the number of elements in the table, so
      // truncating it to the width of the bitmask is safe.
      Value *ShiftAmt = Builder.CreateZExtOrTrunc(Index, MapTy, "switch.cast");

      // Multiply the shift amount by the element width.
      ShiftAmt = Builder.CreateMul(ShiftAmt,
                      ConstantInt::get(MapTy, BitMapElementTy->getBitWidth()),
                                   "switch.shiftamt");

      // Shift down.
      Value *DownShifted = Builder.CreateLShr(BitMap, ShiftAmt,
                                              "switch.downshift");
      // Mask off.
      return Builder.CreateTrunc(DownShifted, BitMapElementTy,
                                 "switch.masked");
    }
    case ArrayKind: {
      // Make sure the table index will not overflow when treated as signed.
      IntegerType *IT = cast<IntegerType>(Index->getType());
      uint64_t TableSize = Array->getInitializer()->getType()
                                ->getArrayNumElements();
      if (TableSize > (1ULL << (IT->getBitWidth() - 1)))
        Index = Builder.CreateZExt(Index,
                                   IntegerType::get(IT->getContext(),
                                                    IT->getBitWidth() + 1),
                                   "switch.tableidx.zext");

      Value *GEPIndices[] = { Builder.getInt32(0), Index };
      Value *GEP = Builder.CreateInBoundsGEP(Array, GEPIndices,
                                             "switch.gep");
      return Builder.CreateLoad(GEP, "switch.load");
    }
  }
  llvm_unreachable("Unknown lookup table kind!");
}

bool SwitchLookupTable::WouldFitInRegister(const DataLayout *DL,
                                           uint64_t TableSize,
                                           const Type *ElementType) {
  if (!DL)
    return false;
  const IntegerType *IT = dyn_cast<IntegerType>(ElementType);
  if (!IT)
    return false;
  // FIXME: If the type is wider than it needs to be, e.g. i8 but all values
  // are <= 15, we could try to narrow the type.

  // Avoid overflow, fitsInLegalInteger uses unsigned int for the width.
  if (TableSize >= UINT_MAX/IT->getBitWidth())
    return false;
  return DL->fitsInLegalInteger(TableSize * IT->getBitWidth());
}

/// ShouldBuildLookupTable - Determine whether a lookup table should be built
/// for this switch, based on the number of cases, size of the table and the
/// types of the results.
static bool ShouldBuildLookupTable(SwitchInst *SI,
                                   uint64_t TableSize,
                                   const TargetTransformInfo &TTI,
                                   const DataLayout *DL,
                            const SmallDenseMap<PHINode*, Type*>& ResultTypes) {
  if (SI->getNumCases() > TableSize || TableSize >= UINT64_MAX / 10)
    return false; // TableSize overflowed, or mul below might overflow.

  bool AllTablesFitInRegister = true;
  bool HasIllegalType = false;
  for (SmallDenseMap<PHINode*, Type*>::const_iterator I = ResultTypes.begin(),
       E = ResultTypes.end(); I != E; ++I) {
    Type *Ty = I->second;

    // Saturate this flag to true.
    HasIllegalType = HasIllegalType || !TTI.isTypeLegal(Ty);

    // Saturate this flag to false.
    AllTablesFitInRegister = AllTablesFitInRegister &&
      SwitchLookupTable::WouldFitInRegister(DL, TableSize, Ty);

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

  // The table density should be at least 40%. This is the same criterion as for
  // jump tables, see SelectionDAGBuilder::handleJTSwitchCase.
  // FIXME: Find the best cut-off.
  return SI->getNumCases() * 10 >= TableSize * 4;
}

/// SwitchToLookupTable - If the switch is only used to initialize one or more
/// phi nodes in a common successor block with different constant values,
/// replace the switch with lookup tables.
static bool SwitchToLookupTable(SwitchInst *SI,
                                IRBuilder<> &Builder,
                                const TargetTransformInfo &TTI,
                                const DataLayout* DL) {
  assert(SI->getNumCases() > 1 && "Degenerate switch?");

  // Only build lookup table when we have a target that supports it.
  if (!TTI.shouldBuildLookupTables())
    return false;

  // FIXME: If the switch is too sparse for a lookup table, perhaps we could
  // split off a dense part and build a lookup table for that.

  // FIXME: This creates arrays of GEPs to constant strings, which means each
  // GEP needs a runtime relocation in PIC code. We should just build one big
  // string and lookup indices into that.

  // Ignore switches with less than three cases. Lookup tables will not make them
  // faster, so we don't analyze them.
  if (SI->getNumCases() < 3)
    return false;

  // Figure out the corresponding result for each case value and phi node in the
  // common destination, as well as the the min and max case values.
  assert(SI->case_begin() != SI->case_end());
  SwitchInst::CaseIt CI = SI->case_begin();
  ConstantInt *MinCaseVal = CI.getCaseValue();
  ConstantInt *MaxCaseVal = CI.getCaseValue();

  BasicBlock *CommonDest = nullptr;
  typedef SmallVector<std::pair<ConstantInt*, Constant*>, 4> ResultListTy;
  SmallDenseMap<PHINode*, ResultListTy> ResultLists;
  SmallDenseMap<PHINode*, Constant*> DefaultResults;
  SmallDenseMap<PHINode*, Type*> ResultTypes;
  SmallVector<PHINode*, 4> PHIs;

  for (SwitchInst::CaseIt E = SI->case_end(); CI != E; ++CI) {
    ConstantInt *CaseVal = CI.getCaseValue();
    if (CaseVal->getValue().slt(MinCaseVal->getValue()))
      MinCaseVal = CaseVal;
    if (CaseVal->getValue().sgt(MaxCaseVal->getValue()))
      MaxCaseVal = CaseVal;

    // Resulting value at phi nodes for this case value.
    typedef SmallVector<std::pair<PHINode*, Constant*>, 4> ResultsTy;
    ResultsTy Results;
    if (!GetCaseResults(SI, CaseVal, CI.getCaseSuccessor(), &CommonDest,
                        Results, DL))
      return false;

    // Append the result from this case to the list for each phi.
    for (ResultsTy::iterator I = Results.begin(), E = Results.end(); I!=E; ++I) {
      if (!ResultLists.count(I->first))
        PHIs.push_back(I->first);
      ResultLists[I->first].push_back(std::make_pair(CaseVal, I->second));
    }
  }

  // Keep track of the result types.
  for (size_t I = 0, E = PHIs.size(); I != E; ++I) {
    PHINode *PHI = PHIs[I];
    ResultTypes[PHI] = ResultLists[PHI][0].second->getType();
  }

  uint64_t NumResults = ResultLists[PHIs[0]].size();
  APInt RangeSpread = MaxCaseVal->getValue() - MinCaseVal->getValue();
  uint64_t TableSize = RangeSpread.getLimitedValue() + 1;
  bool TableHasHoles = (NumResults < TableSize);

  // If the table has holes, we need a constant result for the default case
  // or a bitmask that fits in a register.
  SmallVector<std::pair<PHINode*, Constant*>, 4> DefaultResultsList;
  bool HasDefaultResults = false;
  if (TableHasHoles) {
    HasDefaultResults = GetCaseResults(SI, nullptr, SI->getDefaultDest(),
                                       &CommonDest, DefaultResultsList, DL);
  }
  bool NeedMask = (TableHasHoles && !HasDefaultResults);
  if (NeedMask) {
    // As an extra penalty for the validity test we require more cases.
    if (SI->getNumCases() < 4)  // FIXME: Find best threshold value (benchmark).
      return false;
    if (!(DL && DL->fitsInLegalInteger(TableSize)))
      return false;
  }

  for (size_t I = 0, E = DefaultResultsList.size(); I != E; ++I) {
    PHINode *PHI = DefaultResultsList[I].first;
    Constant *Result = DefaultResultsList[I].second;
    DefaultResults[PHI] = Result;
  }

  if (!ShouldBuildLookupTable(SI, TableSize, TTI, DL, ResultTypes))
    return false;

  // Create the BB that does the lookups.
  Module &Mod = *CommonDest->getParent()->getParent();
  BasicBlock *LookupBB = BasicBlock::Create(Mod.getContext(),
                                            "switch.lookup",
                                            CommonDest->getParent(),
                                            CommonDest);

  // Compute the table index value.
  Builder.SetInsertPoint(SI);
  Value *TableIndex = Builder.CreateSub(SI->getCondition(), MinCaseVal,
                                        "switch.tableidx");

  // Compute the maximum table size representable by the integer type we are
  // switching upon.
  unsigned CaseSize = MinCaseVal->getType()->getPrimitiveSizeInBits();
  uint64_t MaxTableSize = CaseSize > 63 ? UINT64_MAX : 1ULL << CaseSize;
  assert(MaxTableSize >= TableSize &&
         "It is impossible for a switch to have more entries than the max "
         "representable value of its input integer type's size.");

  // If we have a fully covered lookup table, unconditionally branch to the
  // lookup table BB. Otherwise, check if the condition value is within the case
  // range. If it is so, branch to the new BB. Otherwise branch to SI's default
  // destination.
  const bool GeneratingCoveredLookupTable = MaxTableSize == TableSize;
  if (GeneratingCoveredLookupTable) {
    Builder.CreateBr(LookupBB);
    SI->getDefaultDest()->removePredecessor(SI->getParent());
  } else {
    Value *Cmp = Builder.CreateICmpULT(TableIndex, ConstantInt::get(
                                       MinCaseVal->getType(), TableSize));
    Builder.CreateCondBr(Cmp, LookupBB, SI->getDefaultDest());
  }

  // Populate the BB that does the lookups.
  Builder.SetInsertPoint(LookupBB);

  if (NeedMask) {
    // Before doing the lookup we do the hole check.
    // The LookupBB is therefore re-purposed to do the hole check
    // and we create a new LookupBB.
    BasicBlock *MaskBB = LookupBB;
    MaskBB->setName("switch.hole_check");
    LookupBB = BasicBlock::Create(Mod.getContext(),
                                  "switch.lookup",
                                  CommonDest->getParent(),
                                  CommonDest);

    // Build bitmask; fill in a 1 bit for every case.
    APInt MaskInt(TableSize, 0);
    APInt One(TableSize, 1);
    const ResultListTy &ResultList = ResultLists[PHIs[0]];
    for (size_t I = 0, E = ResultList.size(); I != E; ++I) {
      uint64_t Idx = (ResultList[I].first->getValue() -
                      MinCaseVal->getValue()).getLimitedValue();
      MaskInt |= One << Idx;
    }
    ConstantInt *TableMask = ConstantInt::get(Mod.getContext(), MaskInt);

    // Get the TableIndex'th bit of the bitmask.
    // If this bit is 0 (meaning hole) jump to the default destination,
    // else continue with table lookup.
    IntegerType *MapTy = TableMask->getType();
    Value *MaskIndex = Builder.CreateZExtOrTrunc(TableIndex, MapTy,
                                                 "switch.maskindex");
    Value *Shifted = Builder.CreateLShr(TableMask, MaskIndex,
                                        "switch.shifted");
    Value *LoBit = Builder.CreateTrunc(Shifted,
                                       Type::getInt1Ty(Mod.getContext()),
                                       "switch.lobit");
    Builder.CreateCondBr(LoBit, LookupBB, SI->getDefaultDest());

    Builder.SetInsertPoint(LookupBB);
    AddPredecessorToBlock(SI->getDefaultDest(), MaskBB, SI->getParent());
  }

  bool ReturnedEarly = false;
  for (size_t I = 0, E = PHIs.size(); I != E; ++I) {
    PHINode *PHI = PHIs[I];

    // If using a bitmask, use any value to fill the lookup table holes.
    Constant *DV = NeedMask ? ResultLists[PHI][0].second : DefaultResults[PHI];
    SwitchLookupTable Table(Mod, TableSize, MinCaseVal, ResultLists[PHI],
                            DV, DL);

    Value *Result = Table.BuildLookup(TableIndex, Builder);

    // If the result is used to return immediately from the function, we want to
    // do that right here.
    if (PHI->hasOneUse() && isa<ReturnInst>(*PHI->user_begin()) &&
        PHI->user_back() == CommonDest->getFirstNonPHIOrDbg()) {
      Builder.CreateRet(Result);
      ReturnedEarly = true;
      break;
    }

    PHI->addIncoming(Result, LookupBB);
  }

  if (!ReturnedEarly)
    Builder.CreateBr(CommonDest);

  // Remove the switch.
  for (unsigned i = 0, e = SI->getNumSuccessors(); i < e; ++i) {
    BasicBlock *Succ = SI->getSuccessor(i);

    if (Succ == SI->getDefaultDest())
      continue;
    Succ->removePredecessor(SI->getParent());
  }
  SI->eraseFromParent();

  ++NumLookupTables;
  if (NeedMask)
    ++NumLookupTablesHoles;
  return true;
}

bool SimplifyCFGOpt::SimplifySwitch(SwitchInst *SI, IRBuilder<> &Builder) {
  BasicBlock *BB = SI->getParent();

  if (isValueEqualityComparison(SI)) {
    // If we only have one predecessor, and if it is a branch on this value,
    // see if that predecessor totally determines the outcome of this switch.
    if (BasicBlock *OnlyPred = BB->getSinglePredecessor())
      if (SimplifyEqualityComparisonWithOnlyPredecessor(SI, OnlyPred, Builder))
        return SimplifyCFG(BB, TTI, DL) | true;

    Value *Cond = SI->getCondition();
    if (SelectInst *Select = dyn_cast<SelectInst>(Cond))
      if (SimplifySwitchOnSelect(SI, Select))
        return SimplifyCFG(BB, TTI, DL) | true;

    // If the block only contains the switch, see if we can fold the block
    // away into any preds.
    BasicBlock::iterator BBI = BB->begin();
    // Ignore dbg intrinsics.
    while (isa<DbgInfoIntrinsic>(BBI))
      ++BBI;
    if (SI == &*BBI)
      if (FoldValueComparisonIntoPredecessors(SI, Builder))
        return SimplifyCFG(BB, TTI, DL) | true;
  }

  // Try to transform the switch into an icmp and a branch.
  if (TurnSwitchRangeIntoICmp(SI, Builder))
    return SimplifyCFG(BB, TTI, DL) | true;

  // Remove unreachable cases.
  if (EliminateDeadSwitchCases(SI))
    return SimplifyCFG(BB, TTI, DL) | true;

  if (ForwardSwitchConditionToPHI(SI))
    return SimplifyCFG(BB, TTI, DL) | true;

  if (SwitchToLookupTable(SI, Builder, TTI, DL))
    return SimplifyCFG(BB, TTI, DL) | true;

  return false;
}

bool SimplifyCFGOpt::SimplifyIndirectBr(IndirectBrInst *IBI) {
  BasicBlock *BB = IBI->getParent();
  bool Changed = false;

  // Eliminate redundant destinations.
  SmallPtrSet<Value *, 8> Succs;
  for (unsigned i = 0, e = IBI->getNumDestinations(); i != e; ++i) {
    BasicBlock *Dest = IBI->getDestination(i);
    if (!Dest->hasAddressTaken() || !Succs.insert(Dest)) {
      Dest->removePredecessor(BB);
      IBI->removeDestination(i);
      --i; --e;
      Changed = true;
    }
  }

  if (IBI->getNumDestinations() == 0) {
    // If the indirectbr has no successors, change it to unreachable.
    new UnreachableInst(IBI->getContext(), IBI);
    EraseTerminatorInstAndDCECond(IBI);
    return true;
  }

  if (IBI->getNumDestinations() == 1) {
    // If the indirectbr has one successor, change it to a direct branch.
    BranchInst::Create(IBI->getDestination(0), IBI);
    EraseTerminatorInstAndDCECond(IBI);
    return true;
  }

  if (SelectInst *SI = dyn_cast<SelectInst>(IBI->getAddress())) {
    if (SimplifyIndirectBrOnSelect(IBI, SI))
      return SimplifyCFG(BB, TTI, DL) | true;
  }
  return Changed;
}

bool SimplifyCFGOpt::SimplifyUncondBranch(BranchInst *BI, IRBuilder<> &Builder){
  BasicBlock *BB = BI->getParent();

  if (SinkCommon && SinkThenElseCodeToEnd(BI))
    return true;

  // If the Terminator is the only non-phi instruction, simplify the block.
  BasicBlock::iterator I = BB->getFirstNonPHIOrDbg();
  if (I->isTerminator() && BB != &BB->getParent()->getEntryBlock() &&
      TryToSimplifyUncondBranchFromEmptyBlock(BB))
    return true;

  // If the only instruction in the block is a seteq/setne comparison
  // against a constant, try to simplify the block.
  if (ICmpInst *ICI = dyn_cast<ICmpInst>(I))
    if (ICI->isEquality() && isa<ConstantInt>(ICI->getOperand(1))) {
      for (++I; isa<DbgInfoIntrinsic>(I); ++I)
        ;
      if (I->isTerminator() &&
          TryToSimplifyUncondBranchWithICmpInIt(ICI, Builder, TTI, DL))
        return true;
    }

  // If this basic block is ONLY a compare and a branch, and if a predecessor
  // branches to us and our successor, fold the comparison into the
  // predecessor and use logical operations to update the incoming value
  // for PHI nodes in common successor.
  if (FoldBranchToCommonDest(BI, DL))
    return SimplifyCFG(BB, TTI, DL) | true;
  return false;
}


bool SimplifyCFGOpt::SimplifyCondBranch(BranchInst *BI, IRBuilder<> &Builder) {
  BasicBlock *BB = BI->getParent();

  // Conditional branch
  if (isValueEqualityComparison(BI)) {
    // If we only have one predecessor, and if it is a branch on this value,
    // see if that predecessor totally determines the outcome of this
    // switch.
    if (BasicBlock *OnlyPred = BB->getSinglePredecessor())
      if (SimplifyEqualityComparisonWithOnlyPredecessor(BI, OnlyPred, Builder))
        return SimplifyCFG(BB, TTI, DL) | true;

    // This block must be empty, except for the setcond inst, if it exists.
    // Ignore dbg intrinsics.
    BasicBlock::iterator I = BB->begin();
    // Ignore dbg intrinsics.
    while (isa<DbgInfoIntrinsic>(I))
      ++I;
    if (&*I == BI) {
      if (FoldValueComparisonIntoPredecessors(BI, Builder))
        return SimplifyCFG(BB, TTI, DL) | true;
    } else if (&*I == cast<Instruction>(BI->getCondition())){
      ++I;
      // Ignore dbg intrinsics.
      while (isa<DbgInfoIntrinsic>(I))
        ++I;
      if (&*I == BI && FoldValueComparisonIntoPredecessors(BI, Builder))
        return SimplifyCFG(BB, TTI, DL) | true;
    }
  }

  // Try to turn "br (X == 0 | X == 1), T, F" into a switch instruction.
  if (SimplifyBranchOnICmpChain(BI, DL, Builder))
    return true;

  // If this basic block is ONLY a compare and a branch, and if a predecessor
  // branches to us and one of our successors, fold the comparison into the
  // predecessor and use logical operations to pick the right destination.
  if (FoldBranchToCommonDest(BI, DL))
    return SimplifyCFG(BB, TTI, DL) | true;

  // We have a conditional branch to two blocks that are only reachable
  // from BI.  We know that the condbr dominates the two blocks, so see if
  // there is any identical code in the "then" and "else" blocks.  If so, we
  // can hoist it up to the branching block.
  if (BI->getSuccessor(0)->getSinglePredecessor()) {
    if (BI->getSuccessor(1)->getSinglePredecessor()) {
      if (HoistThenElseCodeToIf(BI, DL))
        return SimplifyCFG(BB, TTI, DL) | true;
    } else {
      // If Successor #1 has multiple preds, we may be able to conditionally
      // execute Successor #0 if it branches to Successor #1.
      TerminatorInst *Succ0TI = BI->getSuccessor(0)->getTerminator();
      if (Succ0TI->getNumSuccessors() == 1 &&
          Succ0TI->getSuccessor(0) == BI->getSuccessor(1))
        if (SpeculativelyExecuteBB(BI, BI->getSuccessor(0), DL))
          return SimplifyCFG(BB, TTI, DL) | true;
    }
  } else if (BI->getSuccessor(1)->getSinglePredecessor()) {
    // If Successor #0 has multiple preds, we may be able to conditionally
    // execute Successor #1 if it branches to Successor #0.
    TerminatorInst *Succ1TI = BI->getSuccessor(1)->getTerminator();
    if (Succ1TI->getNumSuccessors() == 1 &&
        Succ1TI->getSuccessor(0) == BI->getSuccessor(0))
      if (SpeculativelyExecuteBB(BI, BI->getSuccessor(1), DL))
        return SimplifyCFG(BB, TTI, DL) | true;
  }

  // If this is a branch on a phi node in the current block, thread control
  // through this block if any PHI node entries are constants.
  if (PHINode *PN = dyn_cast<PHINode>(BI->getCondition()))
    if (PN->getParent() == BI->getParent())
      if (FoldCondBranchOnPHI(BI, DL))
        return SimplifyCFG(BB, TTI, DL) | true;

  // Scan predecessor blocks for conditional branches.
  for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI)
    if (BranchInst *PBI = dyn_cast<BranchInst>((*PI)->getTerminator()))
      if (PBI != BI && PBI->isConditional())
        if (SimplifyCondBranchToCondBranch(PBI, BI))
          return SimplifyCFG(BB, TTI, DL) | true;

  return false;
}

/// Check if passing a value to an instruction will cause undefined behavior.
static bool passingValueIsAlwaysUndefined(Value *V, Instruction *I) {
  Constant *C = dyn_cast<Constant>(V);
  if (!C)
    return false;

  if (I->use_empty())
    return false;

  if (C->isNullValue()) {
    // Only look at the first use, avoid hurting compile time with long uselists
    User *Use = *I->user_begin();

    // Now make sure that there are no instructions in between that can alter
    // control flow (eg. calls)
    for (BasicBlock::iterator i = ++BasicBlock::iterator(I); &*i != Use; ++i)
      if (i == I->getParent()->end() || i->mayHaveSideEffects())
        return false;

    // Look through GEPs. A load from a GEP derived from NULL is still undefined
    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Use))
      if (GEP->getPointerOperand() == I)
        return passingValueIsAlwaysUndefined(V, GEP);

    // Look through bitcasts.
    if (BitCastInst *BC = dyn_cast<BitCastInst>(Use))
      return passingValueIsAlwaysUndefined(V, BC);

    // Load from null is undefined.
    if (LoadInst *LI = dyn_cast<LoadInst>(Use))
      if (!LI->isVolatile())
        return LI->getPointerAddressSpace() == 0;

    // Store to null is undefined.
    if (StoreInst *SI = dyn_cast<StoreInst>(Use))
      if (!SI->isVolatile())
        return SI->getPointerAddressSpace() == 0 && SI->getPointerOperand() == I;
  }
  return false;
}

/// If BB has an incoming value that will always trigger undefined behavior
/// (eg. null pointer dereference), remove the branch leading here.
static bool removeUndefIntroducingPredecessor(BasicBlock *BB) {
  for (BasicBlock::iterator i = BB->begin();
       PHINode *PHI = dyn_cast<PHINode>(i); ++i)
    for (unsigned i = 0, e = PHI->getNumIncomingValues(); i != e; ++i)
      if (passingValueIsAlwaysUndefined(PHI->getIncomingValue(i), PHI)) {
        TerminatorInst *T = PHI->getIncomingBlock(i)->getTerminator();
        IRBuilder<> Builder(T);
        if (BranchInst *BI = dyn_cast<BranchInst>(T)) {
          BB->removePredecessor(PHI->getIncomingBlock(i));
          // Turn uncoditional branches into unreachables and remove the dead
          // destination from conditional branches.
          if (BI->isUnconditional())
            Builder.CreateUnreachable();
          else
            Builder.CreateBr(BI->getSuccessor(0) == BB ? BI->getSuccessor(1) :
                                                         BI->getSuccessor(0));
          BI->eraseFromParent();
          return true;
        }
        // TODO: SwitchInst.
      }

  return false;
}

bool SimplifyCFGOpt::run(BasicBlock *BB) {
  bool Changed = false;

  assert(BB && BB->getParent() && "Block not embedded in function!");
  assert(BB->getTerminator() && "Degenerate basic block encountered!");

  // Remove basic blocks that have no predecessors (except the entry block)...
  // or that just have themself as a predecessor.  These are unreachable.
  if ((pred_begin(BB) == pred_end(BB) &&
       BB != &BB->getParent()->getEntryBlock()) ||
      BB->getSinglePredecessor() == BB) {
    DEBUG(dbgs() << "Removing BB: \n" << *BB);
    DeleteDeadBlock(BB);
    return true;
  }

  // Check to see if we can constant propagate this terminator instruction
  // away...
  Changed |= ConstantFoldTerminator(BB, true);

  // Check for and eliminate duplicate PHI nodes in this block.
  Changed |= EliminateDuplicatePHINodes(BB);

  // Check for and remove branches that will always cause undefined behavior.
  Changed |= removeUndefIntroducingPredecessor(BB);

  // Merge basic blocks into their predecessor if there is only one distinct
  // pred, and if there is only one distinct successor of the predecessor, and
  // if there are no PHI nodes.
  //
  if (MergeBlockIntoPredecessor(BB))
    return true;

  IRBuilder<> Builder(BB);

  // If there is a trivial two-entry PHI node in this basic block, and we can
  // eliminate it, do so now.
  if (PHINode *PN = dyn_cast<PHINode>(BB->begin()))
    if (PN->getNumIncomingValues() == 2)
      Changed |= FoldTwoEntryPHINode(PN, DL);

  Builder.SetInsertPoint(BB->getTerminator());
  if (BranchInst *BI = dyn_cast<BranchInst>(BB->getTerminator())) {
    if (BI->isUnconditional()) {
      if (SimplifyUncondBranch(BI, Builder)) return true;
    } else {
      if (SimplifyCondBranch(BI, Builder)) return true;
    }
  } else if (ReturnInst *RI = dyn_cast<ReturnInst>(BB->getTerminator())) {
    if (SimplifyReturn(RI, Builder)) return true;
  } else if (ResumeInst *RI = dyn_cast<ResumeInst>(BB->getTerminator())) {
    if (SimplifyResume(RI, Builder)) return true;
  } else if (SwitchInst *SI = dyn_cast<SwitchInst>(BB->getTerminator())) {
    if (SimplifySwitch(SI, Builder)) return true;
  } else if (UnreachableInst *UI =
               dyn_cast<UnreachableInst>(BB->getTerminator())) {
    if (SimplifyUnreachable(UI)) return true;
  } else if (IndirectBrInst *IBI =
               dyn_cast<IndirectBrInst>(BB->getTerminator())) {
    if (SimplifyIndirectBr(IBI)) return true;
  }

  return Changed;
}

/// SimplifyCFG - This function is used to do simplification of a CFG.  For
/// example, it adjusts branches to branches to eliminate the extra hop, it
/// eliminates unreachable basic blocks, and does other "peephole" optimization
/// of the CFG.  It returns true if a modification was made.
///
bool llvm::SimplifyCFG(BasicBlock *BB, const TargetTransformInfo &TTI,
                       const DataLayout *DL) {
  return SimplifyCFGOpt(TTI, DL).run(BB);
}
