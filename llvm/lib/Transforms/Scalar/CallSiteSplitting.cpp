//===- CallSiteSplitting.cpp ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a transformation that tries to split a call-site to pass
// more constrained arguments if its argument is predicated in the control flow
// so that we can expose better context to the later passes (e.g, inliner, jump
// threading, or IPA-CP based function cloning, etc.).
// As of now we support two cases :
//
// 1) Try to a split call-site with constrained arguments, if any constraints
// on any argument can be found by following the single predecessors of the
// all site's predecessors. Currently this pass only handles call-sites with 2
// predecessors. For example, in the code below, we try to split the call-site
// since we can predicate the argument(ptr) based on the OR condition.
//
// Split from :
//   if (!ptr || c)
//     callee(ptr);
// to :
//   if (!ptr)
//     callee(null)         // set the known constant value
//   else if (c)
//     callee(nonnull ptr)  // set non-null attribute in the argument
//
// 2) We can also split a call-site based on constant incoming values of a PHI
// For example,
// from :
//   Header:
//    %c = icmp eq i32 %i1, %i2
//    br i1 %c, label %Tail, label %TBB
//   TBB:
//    br label Tail%
//   Tail:
//    %p = phi i32 [ 0, %Header], [ 1, %TBB]
//    call void @bar(i32 %p)
// to
//   Header:
//    %c = icmp eq i32 %i1, %i2
//    br i1 %c, label %Tail-split0, label %TBB
//   TBB:
//    br label %Tail-split1
//   Tail-split0:
//    call void @bar(i32 0)
//    br label %Tail
//   Tail-split1:
//    call void @bar(i32 1)
//    br label %Tail
//   Tail:
//    %p = phi i32 [ 0, %Tail-split0 ], [ 1, %Tail-split1 ]
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/CallSiteSplitting.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "callsite-splitting"

STATISTIC(NumCallSiteSplit, "Number of call-site split");

static void addNonNullAttribute(CallSite CS, Value *Op) {
  unsigned ArgNo = 0;
  for (auto &I : CS.args()) {
    if (&*I == Op)
      CS.addParamAttr(ArgNo, Attribute::NonNull);
    ++ArgNo;
  }
}

static void setConstantInArgument(CallSite CS, Value *Op,
                                  Constant *ConstValue) {
  unsigned ArgNo = 0;
  for (auto &I : CS.args()) {
    if (&*I == Op)
      CS.setArgument(ArgNo, ConstValue);
    ++ArgNo;
  }
}

static bool isCondRelevantToAnyCallArgument(ICmpInst *Cmp, CallSite CS) {
  assert(isa<Constant>(Cmp->getOperand(1)) && "Expected a constant operand.");
  Value *Op0 = Cmp->getOperand(0);
  unsigned ArgNo = 0;
  for (CallSite::arg_iterator I = CS.arg_begin(), E = CS.arg_end(); I != E;
       ++I, ++ArgNo) {
    // Don't consider constant or arguments that are already known non-null.
    if (isa<Constant>(*I) || CS.paramHasAttr(ArgNo, Attribute::NonNull))
      continue;

    if (*I == Op0)
      return true;
  }
  return false;
}

typedef std::pair<ICmpInst *, unsigned> ConditionTy;
typedef SmallVector<ConditionTy, 2> ConditionsTy;

/// If From has a conditional jump to To, add the condition to Conditions,
/// if it is relevant to any argument at CS.
static void recordCondition(CallSite CS, BasicBlock *From, BasicBlock *To,
                            ConditionsTy &Conditions) {
  auto *BI = dyn_cast<BranchInst>(From->getTerminator());
  if (!BI || !BI->isConditional())
    return;

  CmpInst::Predicate Pred;
  Value *Cond = BI->getCondition();
  if (!match(Cond, m_ICmp(Pred, m_Value(), m_Constant())))
    return;

  ICmpInst *Cmp = cast<ICmpInst>(Cond);
  if (Pred == ICmpInst::ICMP_EQ || Pred == ICmpInst::ICMP_NE)
    if (isCondRelevantToAnyCallArgument(Cmp, CS))
      Conditions.push_back({Cmp, From->getTerminator()->getSuccessor(0) == To
                                     ? Pred
                                     : Cmp->getInversePredicate()});
}

/// Record ICmp conditions relevant to any argument in CS following Pred's
/// single successors. If there are conflicting conditions along a path, like
/// x == 1 and x == 0, the first condition will be used.
static void recordConditions(CallSite CS, BasicBlock *Pred,
                             ConditionsTy &Conditions) {
  recordCondition(CS, Pred, CS.getInstruction()->getParent(), Conditions);
  BasicBlock *From = Pred;
  BasicBlock *To = Pred;
  SmallPtrSet<BasicBlock *, 4> Visited = {From};
  while (!Visited.count(From->getSinglePredecessor()) &&
         (From = From->getSinglePredecessor())) {
    recordCondition(CS, From, To, Conditions);
    To = From;
  }
}

static void addConditions(CallSite CS, const ConditionsTy &Conditions) {
  for (auto &Cond : Conditions) {
    Value *Arg = Cond.first->getOperand(0);
    Constant *ConstVal = cast<Constant>(Cond.first->getOperand(1));
    if (Cond.second == ICmpInst::ICMP_EQ)
      setConstantInArgument(CS, Arg, ConstVal);
    else if (ConstVal->getType()->isPointerTy() && ConstVal->isNullValue()) {
      assert(Cond.second == ICmpInst::ICMP_NE);
      addNonNullAttribute(CS, Arg);
    }
  }
}

static SmallVector<BasicBlock *, 2> getTwoPredecessors(BasicBlock *BB) {
  SmallVector<BasicBlock *, 2> Preds(predecessors((BB)));
  assert(Preds.size() == 2 && "Expected exactly 2 predecessors!");
  return Preds;
}

static bool canSplitCallSite(CallSite CS) {
  // FIXME: As of now we handle only CallInst. InvokeInst could be handled
  // without too much effort.
  Instruction *Instr = CS.getInstruction();
  if (!isa<CallInst>(Instr))
    return false;

  // Allow splitting a call-site only when there is no instruction before the
  // call-site in the basic block. Based on this constraint, we only clone the
  // call instruction, and we do not move a call-site across any other
  // instruction.
  BasicBlock *CallSiteBB = Instr->getParent();
  if (Instr != CallSiteBB->getFirstNonPHIOrDbg())
    return false;

  // Need 2 predecessors and cannot split an edge from an IndirectBrInst.
  SmallVector<BasicBlock *, 2> Preds(predecessors(CallSiteBB));
  if (Preds.size() != 2 || isa<IndirectBrInst>(Preds[0]->getTerminator()) ||
      isa<IndirectBrInst>(Preds[1]->getTerminator()))
    return false;

  return CallSiteBB->canSplitPredecessors();
}

/// Return true if the CS is split into its new predecessors.
///
/// For each (predecessor, conditions from predecessors) pair, it will split the
/// basic block containing the call site, hook it up to the predecessor and
/// replace the call instruction with new call instructions, which contain
/// constraints based on the conditions from their predecessors.
/// For example, in the IR below with an OR condition, the call-site can
/// be split. In this case, Preds for Tail is [(Header, a == null),
/// (TBB, a != null, b == null)]. Tail is replaced by 2 split blocks, containing
/// CallInst1, which has constraints based on the conditions from Head and
/// CallInst2, which has constraints based on the conditions coming from TBB.
///
/// From :
///
///   Header:
///     %c = icmp eq i32* %a, null
///     br i1 %c %Tail, %TBB
///   TBB:
///     %c2 = icmp eq i32* %b, null
///     br i1 %c %Tail, %End
///   Tail:
///     %ca = call i1  @callee (i32* %a, i32* %b)
///
///  to :
///
///   Header:                          // PredBB1 is Header
///     %c = icmp eq i32* %a, null
///     br i1 %c %Tail-split1, %TBB
///   TBB:                             // PredBB2 is TBB
///     %c2 = icmp eq i32* %b, null
///     br i1 %c %Tail-split2, %End
///   Tail-split1:
///     %ca1 = call @callee (i32* null, i32* %b)         // CallInst1
///    br %Tail
///   Tail-split2:
///     %ca2 = call @callee (i32* nonnull %a, i32* null) // CallInst2
///    br %Tail
///   Tail:
///    %p = phi i1 [%ca1, %Tail-split1],[%ca2, %Tail-split2]
///
/// Note that in case any arguments at the call-site are constrained by its
/// predecessors, new call-sites with more constrained arguments will be
/// created in createCallSitesOnPredicatedArgument().
static void splitCallSite(
    CallSite CS,
    const SmallVectorImpl<std::pair<BasicBlock *, ConditionsTy>> &Preds) {
  Instruction *Instr = CS.getInstruction();
  BasicBlock *TailBB = Instr->getParent();

  PHINode *CallPN = nullptr;
  if (Instr->getNumUses())
    CallPN = PHINode::Create(Instr->getType(), Preds.size(), "phi.call");

  DEBUG(dbgs() << "split call-site : " << *Instr << " into \n");
  for (const auto &P : Preds) {
    BasicBlock *PredBB = P.first;
    BasicBlock *SplitBlock =
        SplitBlockPredecessors(TailBB, PredBB, ".predBB.split");
    assert(SplitBlock && "Unexpected new basic block split.");

    Instruction *NewCI = Instr->clone();
    CallSite NewCS(NewCI);
    addConditions(NewCS, P.second);
    NewCI->insertBefore(&*SplitBlock->getFirstInsertionPt());

    // Handle PHIs used as arguments in the call-site.
    for (PHINode &PN : TailBB->phis()) {
      unsigned ArgNo = 0;
      for (auto &CI : CS.args()) {
        if (&*CI == &PN) {
          NewCS.setArgument(ArgNo, PN.getIncomingValueForBlock(SplitBlock));
        }
        ++ArgNo;
      }
    }
    DEBUG(dbgs() << "    " << *NewCI << " in " << SplitBlock->getName()
                 << "\n");
    if (CallPN)
      CallPN->addIncoming(NewCI, SplitBlock);
  }

  // Replace users of the original call with a PHI mering call-sites split.
  if (CallPN) {
    CallPN->insertBefore(TailBB->getFirstNonPHI());
    Instr->replaceAllUsesWith(CallPN);
  }

  Instr->eraseFromParent();
  NumCallSiteSplit++;
}

// Return true if the call-site has an argument which is a PHI with only
// constant incoming values.
static bool isPredicatedOnPHI(CallSite CS) {
  Instruction *Instr = CS.getInstruction();
  BasicBlock *Parent = Instr->getParent();
  if (Instr != Parent->getFirstNonPHIOrDbg())
    return false;

  for (auto &BI : *Parent) {
    if (PHINode *PN = dyn_cast<PHINode>(&BI)) {
      for (auto &I : CS.args())
        if (&*I == PN) {
          assert(PN->getNumIncomingValues() == 2 &&
                 "Unexpected number of incoming values");
          if (PN->getIncomingBlock(0) == PN->getIncomingBlock(1))
            return false;
          if (PN->getIncomingValue(0) == PN->getIncomingValue(1))
            continue;
          if (isa<Constant>(PN->getIncomingValue(0)) &&
              isa<Constant>(PN->getIncomingValue(1)))
            return true;
        }
    }
    break;
  }
  return false;
}

static bool tryToSplitOnPHIPredicatedArgument(CallSite CS) {
  if (!isPredicatedOnPHI(CS))
    return false;

  auto Preds = getTwoPredecessors(CS.getInstruction()->getParent());
  SmallVector<std::pair<BasicBlock *, ConditionsTy>, 2> PredsCS = {
      {Preds[0], {}}, {Preds[1], {}}};
  splitCallSite(CS, PredsCS);
  return true;
}

static bool tryToSplitOnPredicatedArgument(CallSite CS) {
  auto Preds = getTwoPredecessors(CS.getInstruction()->getParent());
  if (Preds[0] == Preds[1])
    return false;

  SmallVector<std::pair<BasicBlock *, ConditionsTy>, 2> PredsCS;
  for (auto *Pred : make_range(Preds.rbegin(), Preds.rend())) {
    ConditionsTy Conditions;
    recordConditions(CS, Pred, Conditions);
    PredsCS.push_back({Pred, Conditions});
  }

  if (std::all_of(PredsCS.begin(), PredsCS.end(),
                  [](const std::pair<BasicBlock *, ConditionsTy> &P) {
                    return P.second.empty();
                  }))
    return false;

  splitCallSite(CS, PredsCS);
  return true;
}

static bool tryToSplitCallSite(CallSite CS) {
  if (!CS.arg_size() || !canSplitCallSite(CS))
    return false;
  return tryToSplitOnPredicatedArgument(CS) ||
         tryToSplitOnPHIPredicatedArgument(CS);
}

static bool doCallSiteSplitting(Function &F, TargetLibraryInfo &TLI) {
  bool Changed = false;
  for (Function::iterator BI = F.begin(), BE = F.end(); BI != BE;) {
    BasicBlock &BB = *BI++;
    for (BasicBlock::iterator II = BB.begin(), IE = BB.end(); II != IE;) {
      Instruction *I = &*II++;
      CallSite CS(cast<Value>(I));
      if (!CS || isa<IntrinsicInst>(I) || isInstructionTriviallyDead(I, &TLI))
        continue;

      Function *Callee = CS.getCalledFunction();
      if (!Callee || Callee->isDeclaration())
        continue;
      Changed |= tryToSplitCallSite(CS);
    }
  }
  return Changed;
}

namespace {
struct CallSiteSplittingLegacyPass : public FunctionPass {
  static char ID;
  CallSiteSplittingLegacyPass() : FunctionPass(ID) {
    initializeCallSiteSplittingLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    FunctionPass::getAnalysisUsage(AU);
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
    return doCallSiteSplitting(F, TLI);
  }
};
} // namespace

char CallSiteSplittingLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(CallSiteSplittingLegacyPass, "callsite-splitting",
                      "Call-site splitting", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(CallSiteSplittingLegacyPass, "callsite-splitting",
                    "Call-site splitting", false, false)
FunctionPass *llvm::createCallSiteSplittingPass() {
  return new CallSiteSplittingLegacyPass();
}

PreservedAnalyses CallSiteSplittingPass::run(Function &F,
                                             FunctionAnalysisManager &AM) {
  auto &TLI = AM.getResult<TargetLibraryAnalysis>(F);

  if (!doCallSiteSplitting(F, TLI))
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  return PA;
}
