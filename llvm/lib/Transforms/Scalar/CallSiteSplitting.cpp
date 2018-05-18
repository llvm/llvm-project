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

static void addNonNullAttribute(Instruction *CallI, Instruction *NewCallI,
                                Value *Op) {
  CallSite CS(NewCallI);
  unsigned ArgNo = 0;
  for (auto &I : CS.args()) {
    if (&*I == Op)
      CS.addParamAttr(ArgNo, Attribute::NonNull);
    ++ArgNo;
  }
}

static void setConstantInArgument(Instruction *CallI, Instruction *NewCallI,
                                  Value *Op, Constant *ConstValue) {
  CallSite CS(NewCallI);
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

/// If From has a conditional jump to To, add the condition to Conditions,
/// if it is relevant to any argument at CS.
static void
recordCondition(const CallSite &CS, BasicBlock *From, BasicBlock *To,
                SmallVectorImpl<std::pair<ICmpInst *, unsigned>> &Conditions) {
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
static void
recordConditions(const CallSite &CS, BasicBlock *Pred,
                 SmallVectorImpl<std::pair<ICmpInst *, unsigned>> &Conditions) {
  recordCondition(CS, Pred, CS.getInstruction()->getParent(), Conditions);
  BasicBlock *From = Pred;
  BasicBlock *To = Pred;
  SmallPtrSet<BasicBlock *, 4> Visited;
  while (!Visited.count(From->getSinglePredecessor()) &&
         (From = From->getSinglePredecessor())) {
    recordCondition(CS, From, To, Conditions);
    Visited.insert(From);
    To = From;
  }
}

static Instruction *
addConditions(CallSite &CS,
              SmallVectorImpl<std::pair<ICmpInst *, unsigned>> &Conditions) {
  if (Conditions.empty())
    return nullptr;

  Instruction *NewCI = CS.getInstruction()->clone();
  for (auto &Cond : Conditions) {
    Value *Arg = Cond.first->getOperand(0);
    Constant *ConstVal = cast<Constant>(Cond.first->getOperand(1));
    if (Cond.second == ICmpInst::ICMP_EQ)
      setConstantInArgument(CS.getInstruction(), NewCI, Arg, ConstVal);
    else if (ConstVal->getType()->isPointerTy() && ConstVal->isNullValue()) {
      assert(Cond.second == ICmpInst::ICMP_NE);
      addNonNullAttribute(CS.getInstruction(), NewCI, Arg);
    }
  }
  return NewCI;
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

static Instruction *cloneInstForMustTail(Instruction *I, Instruction *Before,
                                         Value *V) {
  Instruction *Copy = I->clone();
  Copy->setName(I->getName());
  Copy->insertBefore(Before);
  if (V)
    Copy->setOperand(0, V);
  return Copy;
}

/// Copy mandatory `musttail` return sequence that follows original `CI`, and
/// link it up to `NewCI` value instead:
///
///   * (optional) `bitcast NewCI to ...`
///   * `ret bitcast or NewCI`
///
/// Insert this sequence right before `SplitBB`'s terminator, which will be
/// cleaned up later in `splitCallSite` below.
static void copyMustTailReturn(BasicBlock *SplitBB, Instruction *CI,
                               Instruction *NewCI) {
  bool IsVoid = SplitBB->getParent()->getReturnType()->isVoidTy();
  auto II = std::next(CI->getIterator());

  BitCastInst *BCI = dyn_cast<BitCastInst>(&*II);
  if (BCI)
    ++II;

  ReturnInst *RI = dyn_cast<ReturnInst>(&*II);
  assert(RI && "`musttail` call must be followed by `ret` instruction");

  TerminatorInst *TI = SplitBB->getTerminator();
  Value *V = NewCI;
  if (BCI)
    V = cloneInstForMustTail(BCI, TI, V);
  cloneInstForMustTail(RI, TI, IsVoid ? nullptr : V);

  // FIXME: remove TI here, `DuplicateInstructionsInSplitBetween` has a bug
  // that prevents doing this now.
}

/// Return true if the CS is split into its new predecessors which are directly
/// hooked to each of its original predecessors pointed by PredBB1 and PredBB2.
/// CallInst1 and CallInst2 will be the new call-sites placed in the new
/// predecessors split for PredBB1 and PredBB2, respectively.
/// For example, in the IR below with an OR condition, the call-site can
/// be split. Assuming PredBB1=Header and PredBB2=TBB, CallInst1 will be the
/// call-site placed between Header and Tail, and CallInst2 will be the
/// call-site between TBB and Tail.
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
static void splitCallSite(CallSite CS, BasicBlock *PredBB1, BasicBlock *PredBB2,
                          Instruction *CallInst1, Instruction *CallInst2) {
  Instruction *Instr = CS.getInstruction();
  BasicBlock *TailBB = Instr->getParent();
  bool IsMustTailCall = CS.isMustTailCall();
  assert(Instr == (TailBB->getFirstNonPHIOrDbg()) && "Unexpected call-site");

  BasicBlock *SplitBlock1 =
      SplitBlockPredecessors(TailBB, PredBB1, ".predBB1.split");
  BasicBlock *SplitBlock2 =
      SplitBlockPredecessors(TailBB, PredBB2, ".predBB2.split");

  assert((SplitBlock1 && SplitBlock2) && "Unexpected new basic block split.");

  if (!CallInst1)
    CallInst1 = Instr->clone();
  if (!CallInst2)
    CallInst2 = Instr->clone();

  CallInst1->insertBefore(&*SplitBlock1->getFirstInsertionPt());
  CallInst2->insertBefore(&*SplitBlock2->getFirstInsertionPt());

  CallSite CS1(CallInst1);
  CallSite CS2(CallInst2);

  // Handle PHIs used as arguments in the call-site.
  for (PHINode &PN : TailBB->phis()) {
    unsigned ArgNo = 0;
    for (auto &CI : CS.args()) {
      if (&*CI == &PN) {
        CS1.setArgument(ArgNo, PN.getIncomingValueForBlock(SplitBlock1));
        CS2.setArgument(ArgNo, PN.getIncomingValueForBlock(SplitBlock2));
      }
      ++ArgNo;
    }
  }
  // Clone and place bitcast and return instructions before `TI`
  if (IsMustTailCall) {
    copyMustTailReturn(SplitBlock1, CS.getInstruction(), CallInst1);
    copyMustTailReturn(SplitBlock2, CS.getInstruction(), CallInst2);
  }

  // Replace users of the original call with a PHI mering call-sites split.
  if (!IsMustTailCall && Instr->getNumUses()) {
    PHINode *PN = PHINode::Create(Instr->getType(), 2, "phi.call",
                                  TailBB->getFirstNonPHI());
    PN->addIncoming(CallInst1, SplitBlock1);
    PN->addIncoming(CallInst2, SplitBlock2);
    Instr->replaceAllUsesWith(PN);
  }
  DEBUG(dbgs() << "split call-site : " << *Instr << " into \n");
  DEBUG(dbgs() << "    " << *CallInst1 << " in " << SplitBlock1->getName()
               << "\n");
  DEBUG(dbgs() << "    " << *CallInst2 << " in " << SplitBlock2->getName()
               << "\n");

  NumCallSiteSplit++;

  // FIXME: remove TI in `copyMustTailReturn`
  if (IsMustTailCall) {
    // Remove superfluous `br` terminators from the end of the Split blocks
    // NOTE: Removing terminator removes the SplitBlock from the TailBB's
    // predecessors. Therefore we must get complete list of Splits before
    // attempting removal.
    SmallVector<BasicBlock *, 2> Splits(predecessors((TailBB)));
    assert(Splits.size() == 2 && "Expected exactly 2 splits!");
    for (unsigned i = 0; i < Splits.size(); i++)
      Splits[i]->getTerminator()->eraseFromParent();

    // Erase the tail block once done with musttail patching
    TailBB->eraseFromParent();
    return;
  }
  Instr->eraseFromParent();
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
  splitCallSite(CS, Preds[0], Preds[1], nullptr, nullptr);
  return true;
}

static bool tryToSplitOnPredicatedArgument(CallSite CS) {
  auto Preds = getTwoPredecessors(CS.getInstruction()->getParent());
  if (Preds[0] == Preds[1])
    return false;

  SmallVector<std::pair<ICmpInst *, unsigned>, 2> C1, C2;
  recordConditions(CS, Preds[0], C1);
  recordConditions(CS, Preds[1], C2);

  Instruction *CallInst1 = addConditions(CS, C1);
  Instruction *CallInst2 = addConditions(CS, C2);
  if (!CallInst1 && !CallInst2)
    return false;

  splitCallSite(CS, Preds[1], Preds[0], CallInst2, CallInst1);
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

      // Successful musttail call-site splits result in erased CI and erased BB.
      // Check if such path is possible before attempting the splitting.
      bool IsMustTail = CS.isMustTailCall();

      Changed |= tryToSplitCallSite(CS);

      // There're no interesting instructions after this. The call site
      // itself might have been erased on splitting.
      if (IsMustTail)
        break;
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
