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
// 1) If a call site is dominated by an OR condition and if any of its arguments
// are predicated on this OR condition, try to split the condition with more
// constrained arguments. For example, in the code below, we try to split the
// call site since we can predicate the argument(ptr) based on the OR condition.
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

static void addNonNullAttribute(Instruction *CallI, Instruction *&NewCallI,
                                Value *Op) {
  if (!NewCallI)
    NewCallI = CallI->clone();
  CallSite CS(NewCallI);
  unsigned ArgNo = 0;
  for (auto &I : CS.args()) {
    if (&*I == Op)
      CS.addParamAttr(ArgNo, Attribute::NonNull);
    ++ArgNo;
  }
}

static void setConstantInArgument(Instruction *CallI, Instruction *&NewCallI,
                                  Value *Op, Constant *ConstValue) {
  if (!NewCallI)
    NewCallI = CallI->clone();
  CallSite CS(NewCallI);
  unsigned ArgNo = 0;
  for (auto &I : CS.args()) {
    if (&*I == Op)
      CS.setArgument(ArgNo, ConstValue);
    ++ArgNo;
  }
}

static bool createCallSitesOnOrPredicatedArgument(
    CallSite CS, Instruction *&NewCSTakenFromHeader,
    Instruction *&NewCSTakenFromNextCond,
    SmallVectorImpl<BranchInst *> &BranchInsts, BasicBlock *HeaderBB) {
  assert(BranchInsts.size() <= 2 &&
         "Unexpected number of blocks in the OR predicated condition");
  Instruction *Instr = CS.getInstruction();
  BasicBlock *CallSiteBB = Instr->getParent();
  TerminatorInst *HeaderTI = HeaderBB->getTerminator();
  bool IsCSInTakenPath = CallSiteBB == HeaderTI->getSuccessor(0);

  for (unsigned I = 0, E = BranchInsts.size(); I != E; ++I) {
    BranchInst *PBI = BranchInsts[I];
    assert(isa<ICmpInst>(PBI->getCondition()) &&
           "Unexpected condition in a conditional branch.");
    ICmpInst *Cmp = cast<ICmpInst>(PBI->getCondition());
    Value *Arg = Cmp->getOperand(0);
    assert(isa<Constant>(Cmp->getOperand(1)) &&
           "Expected op1 to be a constant.");
    Constant *ConstVal = cast<Constant>(Cmp->getOperand(1));
    CmpInst::Predicate Pred = Cmp->getPredicate();

    if (PBI->getParent() == HeaderBB) {
      Instruction *&CallTakenFromHeader =
          IsCSInTakenPath ? NewCSTakenFromHeader : NewCSTakenFromNextCond;
      Instruction *&CallUntakenFromHeader =
          IsCSInTakenPath ? NewCSTakenFromNextCond : NewCSTakenFromHeader;

      assert(Pred == ICmpInst::ICMP_EQ ||
             Pred == ICmpInst::ICMP_NE &&
                 "Unexpected predicate in an OR condition");

      // Set the constant value for agruments in the call predicated based on
      // the OR condition.
      Instruction *&CallToSetConst = Pred == ICmpInst::ICMP_EQ
                                         ? CallTakenFromHeader
                                         : CallUntakenFromHeader;
      setConstantInArgument(Instr, CallToSetConst, Arg, ConstVal);

      // Add the NonNull attribute if compared with the null pointer.
      if (ConstVal->getType()->isPointerTy() && ConstVal->isNullValue()) {
        Instruction *&CallToSetAttr = Pred == ICmpInst::ICMP_EQ
                                          ? CallUntakenFromHeader
                                          : CallTakenFromHeader;
        addNonNullAttribute(Instr, CallToSetAttr, Arg);
      }
      continue;
    }

    if (Pred == ICmpInst::ICMP_EQ) {
      if (PBI->getSuccessor(0) == Instr->getParent()) {
        // Set the constant value for the call taken from the second block in
        // the OR condition.
        setConstantInArgument(Instr, NewCSTakenFromNextCond, Arg, ConstVal);
      } else {
        // Add the NonNull attribute if compared with the null pointer for the
        // call taken from the second block in the OR condition.
        if (ConstVal->getType()->isPointerTy() && ConstVal->isNullValue())
          addNonNullAttribute(Instr, NewCSTakenFromNextCond, Arg);
      }
    } else {
      if (PBI->getSuccessor(0) == Instr->getParent()) {
        // Add the NonNull attribute if compared with the null pointer for the
        // call taken from the second block in the OR condition.
        if (ConstVal->getType()->isPointerTy() && ConstVal->isNullValue())
          addNonNullAttribute(Instr, NewCSTakenFromNextCond, Arg);
      } else if (Pred == ICmpInst::ICMP_NE) {
        // Set the constant value for the call in the untaken path from the
        // header block.
        setConstantInArgument(Instr, NewCSTakenFromNextCond, Arg, ConstVal);
      } else
        llvm_unreachable("Unexpected condition");
    }
  }
  return NewCSTakenFromHeader || NewCSTakenFromNextCond;
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
  if (Instr != CallSiteBB->getFirstNonPHI())
    return false;

  pred_iterator PII = pred_begin(CallSiteBB);
  pred_iterator PIE = pred_end(CallSiteBB);
  unsigned NumPreds = std::distance(PII, PIE);

  // Allow only one extra call-site. No more than two from one call-site.
  if (NumPreds != 2)
    return false;

  // Cannot split an edge from an IndirectBrInst.
  BasicBlock *Preds[2] = {*PII++, *PII};
  if (isa<IndirectBrInst>(Preds[0]->getTerminator()) ||
      isa<IndirectBrInst>(Preds[1]->getTerminator()))
    return false;

  return CallSiteBB->canSplitPredecessors();
}

/// Return true if the CS is split into its new predecessors which are directly
/// hooked to each of its orignial predecessors pointed by PredBB1 and PredBB2.
/// Note that PredBB1 and PredBB2 are decided in findPredicatedArgument(),
/// especially for the OR predicated case where PredBB1 will point the header,
/// and PredBB2 will point the the second compare block. CallInst1 and CallInst2
/// will be the new call-sites placed in the new predecessors split for PredBB1
/// and PredBB2, repectively. Therefore, CallInst1 will be the call-site placed
/// between Header and Tail, and CallInst2 will be the call-site between TBB and
/// Tail. For example, in the IR below with an OR condition, the call-site can
/// be split
///
/// from :
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
/// Note that for an OR predicated case, CallInst1 and CallInst2 should be
/// created with more constrained arguments in
/// createCallSitesOnOrPredicatedArgument().
static void splitCallSite(CallSite CS, BasicBlock *PredBB1, BasicBlock *PredBB2,
                          Instruction *CallInst1, Instruction *CallInst2) {
  Instruction *Instr = CS.getInstruction();
  BasicBlock *TailBB = Instr->getParent();
  assert(Instr == (TailBB->getFirstNonPHI()) && "Unexpected call-site");

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
  for (auto &PI : *TailBB) {
    PHINode *PN = dyn_cast<PHINode>(&PI);
    if (!PN)
      break;
    unsigned ArgNo = 0;
    for (auto &CI : CS.args()) {
      if (&*CI == PN) {
        CS1.setArgument(ArgNo, PN->getIncomingValueForBlock(SplitBlock1));
        CS2.setArgument(ArgNo, PN->getIncomingValueForBlock(SplitBlock2));
      }
      ++ArgNo;
    }
  }

  // Replace users of the original call with a PHI mering call-sites split.
  if (Instr->getNumUses()) {
    PHINode *PN = PHINode::Create(Instr->getType(), 2, "phi.call", Instr);
    PN->addIncoming(CallInst1, SplitBlock1);
    PN->addIncoming(CallInst2, SplitBlock2);
    Instr->replaceAllUsesWith(PN);
  }
  DEBUG(dbgs() << "split call-site : " << *Instr << " into \n");
  DEBUG(dbgs() << "    " << *CallInst1 << " in " << SplitBlock1->getName()
               << "\n");
  DEBUG(dbgs() << "    " << *CallInst2 << " in " << SplitBlock2->getName()
               << "\n");
  Instr->eraseFromParent();
  NumCallSiteSplit++;
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

static void findOrCondRelevantToCallArgument(
    CallSite CS, BasicBlock *PredBB, BasicBlock *OtherPredBB,
    SmallVectorImpl<BranchInst *> &BranchInsts, BasicBlock *&HeaderBB) {
  auto *PBI = dyn_cast<BranchInst>(PredBB->getTerminator());
  if (!PBI || !PBI->isConditional())
    return;

  if (PBI->getSuccessor(0) == OtherPredBB ||
      PBI->getSuccessor(1) == OtherPredBB)
    if (PredBB == OtherPredBB->getSinglePredecessor()) {
      assert(!HeaderBB && "Expect to find only a single header block");
      HeaderBB = PredBB;
    }

  CmpInst::Predicate Pred;
  Value *Cond = PBI->getCondition();
  if (!match(Cond, m_ICmp(Pred, m_Value(), m_Constant())))
    return;
  ICmpInst *Cmp = cast<ICmpInst>(Cond);
  if (Pred == ICmpInst::ICMP_EQ || Pred == ICmpInst::ICMP_NE)
    if (isCondRelevantToAnyCallArgument(Cmp, CS))
      BranchInsts.push_back(PBI);
}

// Return true if the call-site has an argument which is a PHI with only
// constant incoming values.
static bool isPredicatedOnPHI(CallSite CS) {
  Instruction *Instr = CS.getInstruction();
  BasicBlock *Parent = Instr->getParent();
  if (Instr != Parent->getFirstNonPHI())
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

// Return true if an agument in CS is predicated on an 'or' condition.
// Create new call-site with arguments constrained based on the OR condition.
static bool findPredicatedOnOrCondition(CallSite CS, BasicBlock *PredBB1,
                                        BasicBlock *PredBB2,
                                        Instruction *&NewCallTakenFromHeader,
                                        Instruction *&NewCallTakenFromNextCond,
                                        BasicBlock *&HeaderBB) {
  SmallVector<BranchInst *, 4> BranchInsts;
  findOrCondRelevantToCallArgument(CS, PredBB1, PredBB2, BranchInsts, HeaderBB);
  findOrCondRelevantToCallArgument(CS, PredBB2, PredBB1, BranchInsts, HeaderBB);
  if (BranchInsts.empty() || !HeaderBB)
    return false;

  // If an OR condition is detected, try to create call sites with constrained
  // arguments (e.g., NonNull attribute or constant value).
  return createCallSitesOnOrPredicatedArgument(CS, NewCallTakenFromHeader,
                                               NewCallTakenFromNextCond,
                                               BranchInsts, HeaderBB);
}

static bool findPredicatedArgument(CallSite CS, Instruction *&CallInst1,
                                   Instruction *&CallInst2,
                                   BasicBlock *&PredBB1, BasicBlock *&PredBB2) {
  BasicBlock *CallSiteBB = CS.getInstruction()->getParent();
  pred_iterator PII = pred_begin(CallSiteBB);
  pred_iterator PIE = pred_end(CallSiteBB);
  assert(std::distance(PII, PIE) == 2 && "Expect only two predecessors.");
  (void)PIE;
  BasicBlock *Preds[2] = {*PII++, *PII};
  BasicBlock *&HeaderBB = PredBB1;
  if (!findPredicatedOnOrCondition(CS, Preds[0], Preds[1], CallInst1, CallInst2,
                                   HeaderBB) &&
      !isPredicatedOnPHI(CS))
    return false;

  if (!PredBB1)
    PredBB1 = Preds[0];

  PredBB2 = PredBB1 == Preds[0] ? Preds[1] : Preds[0];
  return true;
}

static bool tryToSplitCallSite(CallSite CS) {
  if (!CS.arg_size())
    return false;

  BasicBlock *PredBB1 = nullptr;
  BasicBlock *PredBB2 = nullptr;
  Instruction *CallInst1 = nullptr;
  Instruction *CallInst2 = nullptr;
  if (!canSplitCallSite(CS) ||
      !findPredicatedArgument(CS, CallInst1, CallInst2, PredBB1, PredBB2)) {
    assert(!CallInst1 && !CallInst2 && "Unexpected new call-sites cloned.");
    return false;
  }
  splitCallSite(CS, PredBB1, PredBB2, CallInst1, CallInst2);
  return true;
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
