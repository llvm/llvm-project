//===-- SimplifyIndVar.cpp - Induction variable simplification ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements induction variable simplification. It does
// not define any actual pass or policy, but provides a single function to
// simplify a loop's induction variables based on ScalarEvolution.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/SimplifyIndVar.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "indvars"

STATISTIC(NumElimIdentity, "Number of IV identities eliminated");
STATISTIC(NumElimOperand,  "Number of IV operands folded into a use");
STATISTIC(NumElimRem     , "Number of IV remainder operations eliminated");
STATISTIC(
    NumSimplifiedSDiv,
    "Number of IV signed division operations converted to unsigned division");
STATISTIC(NumElimCmp     , "Number of IV comparisons eliminated");

namespace {
  /// This is a utility for simplifying induction variables
  /// based on ScalarEvolution. It is the primary instrument of the
  /// IndvarSimplify pass, but it may also be directly invoked to cleanup after
  /// other loop passes that preserve SCEV.
  class SimplifyIndvar {
    Loop             *L;
    LoopInfo         *LI;
    ScalarEvolution  *SE;
    DominatorTree    *DT;

    SmallVectorImpl<WeakTrackingVH> &DeadInsts;

    bool Changed;

  public:
    SimplifyIndvar(Loop *Loop, ScalarEvolution *SE, DominatorTree *DT,
                   LoopInfo *LI, SmallVectorImpl<WeakTrackingVH> &Dead)
        : L(Loop), LI(LI), SE(SE), DT(DT), DeadInsts(Dead), Changed(false) {
      assert(LI && "IV simplification requires LoopInfo");
    }

    bool hasChanged() const { return Changed; }

    /// Iteratively perform simplification on a worklist of users of the
    /// specified induction variable. This is the top-level driver that applies
    /// all simplifications to users of an IV.
    void simplifyUsers(PHINode *CurrIV, IVVisitor *V = nullptr);

    Value *foldIVUser(Instruction *UseInst, Instruction *IVOperand);

    bool eliminateIdentitySCEV(Instruction *UseInst, Instruction *IVOperand);

    bool eliminateOverflowIntrinsic(CallInst *CI);
    bool eliminateIVUser(Instruction *UseInst, Instruction *IVOperand);
    void eliminateIVComparison(ICmpInst *ICmp, Value *IVOperand);
    void eliminateIVRemainder(BinaryOperator *Rem, Value *IVOperand,
                              bool IsSigned);
    bool eliminateSDiv(BinaryOperator *SDiv);
    bool strengthenOverflowingOperation(BinaryOperator *OBO, Value *IVOperand);
    bool strengthenRightShift(BinaryOperator *BO, Value *IVOperand);
  };
}

/// Fold an IV operand into its use.  This removes increments of an
/// aligned IV when used by a instruction that ignores the low bits.
///
/// IVOperand is guaranteed SCEVable, but UseInst may not be.
///
/// Return the operand of IVOperand for this induction variable if IVOperand can
/// be folded (in case more folding opportunities have been exposed).
/// Otherwise return null.
Value *SimplifyIndvar::foldIVUser(Instruction *UseInst, Instruction *IVOperand) {
  Value *IVSrc = nullptr;
  unsigned OperIdx = 0;
  const SCEV *FoldedExpr = nullptr;
  switch (UseInst->getOpcode()) {
  default:
    return nullptr;
  case Instruction::UDiv:
  case Instruction::LShr:
    // We're only interested in the case where we know something about
    // the numerator and have a constant denominator.
    if (IVOperand != UseInst->getOperand(OperIdx) ||
        !isa<ConstantInt>(UseInst->getOperand(1)))
      return nullptr;

    // Attempt to fold a binary operator with constant operand.
    // e.g. ((I + 1) >> 2) => I >> 2
    if (!isa<BinaryOperator>(IVOperand)
        || !isa<ConstantInt>(IVOperand->getOperand(1)))
      return nullptr;

    IVSrc = IVOperand->getOperand(0);
    // IVSrc must be the (SCEVable) IV, since the other operand is const.
    assert(SE->isSCEVable(IVSrc->getType()) && "Expect SCEVable IV operand");

    ConstantInt *D = cast<ConstantInt>(UseInst->getOperand(1));
    if (UseInst->getOpcode() == Instruction::LShr) {
      // Get a constant for the divisor. See createSCEV.
      uint32_t BitWidth = cast<IntegerType>(UseInst->getType())->getBitWidth();
      if (D->getValue().uge(BitWidth))
        return nullptr;

      D = ConstantInt::get(UseInst->getContext(),
                           APInt::getOneBitSet(BitWidth, D->getZExtValue()));
    }
    FoldedExpr = SE->getUDivExpr(SE->getSCEV(IVSrc), SE->getSCEV(D));
  }
  // We have something that might fold it's operand. Compare SCEVs.
  if (!SE->isSCEVable(UseInst->getType()))
    return nullptr;

  // Bypass the operand if SCEV can prove it has no effect.
  if (SE->getSCEV(UseInst) != FoldedExpr)
    return nullptr;

  DEBUG(dbgs() << "INDVARS: Eliminated IV operand: " << *IVOperand
        << " -> " << *UseInst << '\n');

  UseInst->setOperand(OperIdx, IVSrc);
  assert(SE->getSCEV(UseInst) == FoldedExpr && "bad SCEV with folded oper");

  ++NumElimOperand;
  Changed = true;
  if (IVOperand->use_empty())
    DeadInsts.emplace_back(IVOperand);
  return IVSrc;
}

/// SimplifyIVUsers helper for eliminating useless
/// comparisons against an induction variable.
void SimplifyIndvar::eliminateIVComparison(ICmpInst *ICmp, Value *IVOperand) {
  unsigned IVOperIdx = 0;
  ICmpInst::Predicate Pred = ICmp->getPredicate();
  ICmpInst::Predicate OriginalPred = Pred;
  if (IVOperand != ICmp->getOperand(0)) {
    // Swapped
    assert(IVOperand == ICmp->getOperand(1) && "Can't find IVOperand");
    IVOperIdx = 1;
    Pred = ICmpInst::getSwappedPredicate(Pred);
  }

  // Get the SCEVs for the ICmp operands.
  const SCEV *S = SE->getSCEV(ICmp->getOperand(IVOperIdx));
  const SCEV *X = SE->getSCEV(ICmp->getOperand(1 - IVOperIdx));

  // Simplify unnecessary loops away.
  const Loop *ICmpLoop = LI->getLoopFor(ICmp->getParent());
  S = SE->getSCEVAtScope(S, ICmpLoop);
  X = SE->getSCEVAtScope(X, ICmpLoop);

  ICmpInst::Predicate InvariantPredicate;
  const SCEV *InvariantLHS, *InvariantRHS;

  // If the condition is always true or always false, replace it with
  // a constant value.
  if (SE->isKnownPredicate(Pred, S, X)) {
    ICmp->replaceAllUsesWith(ConstantInt::getTrue(ICmp->getContext()));
    DeadInsts.emplace_back(ICmp);
    DEBUG(dbgs() << "INDVARS: Eliminated comparison: " << *ICmp << '\n');
  } else if (SE->isKnownPredicate(ICmpInst::getInversePredicate(Pred), S, X)) {
    ICmp->replaceAllUsesWith(ConstantInt::getFalse(ICmp->getContext()));
    DeadInsts.emplace_back(ICmp);
    DEBUG(dbgs() << "INDVARS: Eliminated comparison: " << *ICmp << '\n');
  } else if (isa<PHINode>(IVOperand) &&
             SE->isLoopInvariantPredicate(Pred, S, X, L, InvariantPredicate,
                                          InvariantLHS, InvariantRHS)) {

    // Rewrite the comparison to a loop invariant comparison if it can be done
    // cheaply, where cheaply means "we don't need to emit any new
    // instructions".

    Value *NewLHS = nullptr, *NewRHS = nullptr;

    if (S == InvariantLHS || X == InvariantLHS)
      NewLHS =
          ICmp->getOperand(S == InvariantLHS ? IVOperIdx : (1 - IVOperIdx));

    if (S == InvariantRHS || X == InvariantRHS)
      NewRHS =
          ICmp->getOperand(S == InvariantRHS ? IVOperIdx : (1 - IVOperIdx));

    auto *PN = cast<PHINode>(IVOperand);
    for (unsigned i = 0, e = PN->getNumIncomingValues();
         i != e && (!NewLHS || !NewRHS);
         ++i) {

      // If this is a value incoming from the backedge, then it cannot be a loop
      // invariant value (since we know that IVOperand is an induction variable).
      if (L->contains(PN->getIncomingBlock(i)))
        continue;

      // NB! This following assert does not fundamentally have to be true, but
      // it is true today given how SCEV analyzes induction variables.
      // Specifically, today SCEV will *not* recognize %iv as an induction
      // variable in the following case:
      //
      // define void @f(i32 %k) {
      // entry:
      //   br i1 undef, label %r, label %l
      //
      // l:
      //   %k.inc.l = add i32 %k, 1
      //   br label %loop
      //
      // r:
      //   %k.inc.r = add i32 %k, 1
      //   br label %loop
      //
      // loop:
      //   %iv = phi i32 [ %k.inc.l, %l ], [ %k.inc.r, %r ], [ %iv.inc, %loop ]
      //   %iv.inc = add i32 %iv, 1
      //   br label %loop
      // }
      //
      // but if it starts to, at some point, then the assertion below will have
      // to be changed to a runtime check.

      Value *Incoming = PN->getIncomingValue(i);

#ifndef NDEBUG
      if (auto *I = dyn_cast<Instruction>(Incoming))
        assert(DT->dominates(I, ICmp) && "Should be a unique loop dominating value!");
#endif

      const SCEV *IncomingS = SE->getSCEV(Incoming);

      if (!NewLHS && IncomingS == InvariantLHS)
        NewLHS = Incoming;
      if (!NewRHS && IncomingS == InvariantRHS)
        NewRHS = Incoming;
    }

    if (!NewLHS || !NewRHS)
      // We could not find an existing value to replace either LHS or RHS.
      // Generating new instructions has subtler tradeoffs, so avoid doing that
      // for now.
      return;

    DEBUG(dbgs() << "INDVARS: Simplified comparison: " << *ICmp << '\n');
    ICmp->setPredicate(InvariantPredicate);
    ICmp->setOperand(0, NewLHS);
    ICmp->setOperand(1, NewRHS);
  } else if (ICmpInst::isSigned(OriginalPred) &&
             SE->isKnownNonNegative(S) && SE->isKnownNonNegative(X)) {
    // If we were unable to make anything above, all we can is to canonicalize
    // the comparison hoping that it will open the doors for other
    // optimizations. If we find out that we compare two non-negative values,
    // we turn the instruction's predicate to its unsigned version. Note that
    // we cannot rely on Pred here unless we check if we have swapped it.
    assert(ICmp->getPredicate() == OriginalPred && "Predicate changed?");
    DEBUG(dbgs() << "INDVARS: Turn to unsigned comparison: " << *ICmp << '\n');
    ICmp->setPredicate(ICmpInst::getUnsignedPredicate(OriginalPred));
  } else
    return;

  ++NumElimCmp;
  Changed = true;
}

bool SimplifyIndvar::eliminateSDiv(BinaryOperator *SDiv) {
  // Get the SCEVs for the ICmp operands.
  auto *N = SE->getSCEV(SDiv->getOperand(0));
  auto *D = SE->getSCEV(SDiv->getOperand(1));

  // Simplify unnecessary loops away.
  const Loop *L = LI->getLoopFor(SDiv->getParent());
  N = SE->getSCEVAtScope(N, L);
  D = SE->getSCEVAtScope(D, L);

  // Replace sdiv by udiv if both of the operands are non-negative
  if (SE->isKnownNonNegative(N) && SE->isKnownNonNegative(D)) {
    auto *UDiv = BinaryOperator::Create(
        BinaryOperator::UDiv, SDiv->getOperand(0), SDiv->getOperand(1),
        SDiv->getName() + ".udiv", SDiv);
    UDiv->setIsExact(SDiv->isExact());
    SDiv->replaceAllUsesWith(UDiv);
    DEBUG(dbgs() << "INDVARS: Simplified sdiv: " << *SDiv << '\n');
    ++NumSimplifiedSDiv;
    Changed = true;
    DeadInsts.push_back(SDiv);
    return true;
  }

  return false;
}

/// SimplifyIVUsers helper for eliminating useless
/// remainder operations operating on an induction variable.
void SimplifyIndvar::eliminateIVRemainder(BinaryOperator *Rem,
                                      Value *IVOperand,
                                      bool IsSigned) {
  // We're only interested in the case where we know something about
  // the numerator.
  if (IVOperand != Rem->getOperand(0))
    return;

  // Get the SCEVs for the ICmp operands.
  const SCEV *S = SE->getSCEV(Rem->getOperand(0));
  const SCEV *X = SE->getSCEV(Rem->getOperand(1));

  // Simplify unnecessary loops away.
  const Loop *ICmpLoop = LI->getLoopFor(Rem->getParent());
  S = SE->getSCEVAtScope(S, ICmpLoop);
  X = SE->getSCEVAtScope(X, ICmpLoop);

  // i % n  -->  i  if i is in [0,n).
  if ((!IsSigned || SE->isKnownNonNegative(S)) &&
      SE->isKnownPredicate(IsSigned ? ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT,
                           S, X))
    Rem->replaceAllUsesWith(Rem->getOperand(0));
  else {
    // (i+1) % n  -->  (i+1)==n?0:(i+1)  if i is in [0,n).
    const SCEV *LessOne = SE->getMinusSCEV(S, SE->getOne(S->getType()));
    if (IsSigned && !SE->isKnownNonNegative(LessOne))
      return;

    if (!SE->isKnownPredicate(IsSigned ?
                              ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT,
                              LessOne, X))
      return;

    ICmpInst *ICmp = new ICmpInst(Rem, ICmpInst::ICMP_EQ,
                                  Rem->getOperand(0), Rem->getOperand(1));
    SelectInst *Sel =
      SelectInst::Create(ICmp,
                         ConstantInt::get(Rem->getType(), 0),
                         Rem->getOperand(0), "tmp", Rem);
    Rem->replaceAllUsesWith(Sel);
  }

  DEBUG(dbgs() << "INDVARS: Simplified rem: " << *Rem << '\n');
  ++NumElimRem;
  Changed = true;
  DeadInsts.emplace_back(Rem);
}

bool SimplifyIndvar::eliminateOverflowIntrinsic(CallInst *CI) {
  auto *F = CI->getCalledFunction();
  if (!F)
    return false;

  typedef const SCEV *(ScalarEvolution::*OperationFunctionTy)(
      const SCEV *, const SCEV *, SCEV::NoWrapFlags, unsigned);
  typedef const SCEV *(ScalarEvolution::*ExtensionFunctionTy)(
      const SCEV *, Type *, unsigned);

  OperationFunctionTy Operation;
  ExtensionFunctionTy Extension;

  Instruction::BinaryOps RawOp;

  // We always have exactly one of nsw or nuw.  If NoSignedOverflow is false, we
  // have nuw.
  bool NoSignedOverflow;

  switch (F->getIntrinsicID()) {
  default:
    return false;

  case Intrinsic::sadd_with_overflow:
    Operation = &ScalarEvolution::getAddExpr;
    Extension = &ScalarEvolution::getSignExtendExpr;
    RawOp = Instruction::Add;
    NoSignedOverflow = true;
    break;

  case Intrinsic::uadd_with_overflow:
    Operation = &ScalarEvolution::getAddExpr;
    Extension = &ScalarEvolution::getZeroExtendExpr;
    RawOp = Instruction::Add;
    NoSignedOverflow = false;
    break;

  case Intrinsic::ssub_with_overflow:
    Operation = &ScalarEvolution::getMinusSCEV;
    Extension = &ScalarEvolution::getSignExtendExpr;
    RawOp = Instruction::Sub;
    NoSignedOverflow = true;
    break;

  case Intrinsic::usub_with_overflow:
    Operation = &ScalarEvolution::getMinusSCEV;
    Extension = &ScalarEvolution::getZeroExtendExpr;
    RawOp = Instruction::Sub;
    NoSignedOverflow = false;
    break;
  }

  const SCEV *LHS = SE->getSCEV(CI->getArgOperand(0));
  const SCEV *RHS = SE->getSCEV(CI->getArgOperand(1));

  auto *NarrowTy = cast<IntegerType>(LHS->getType());
  auto *WideTy =
    IntegerType::get(NarrowTy->getContext(), NarrowTy->getBitWidth() * 2);

  const SCEV *A =
      (SE->*Extension)((SE->*Operation)(LHS, RHS, SCEV::FlagAnyWrap, 0),
                       WideTy, 0);
  const SCEV *B =
      (SE->*Operation)((SE->*Extension)(LHS, WideTy, 0),
                       (SE->*Extension)(RHS, WideTy, 0), SCEV::FlagAnyWrap, 0);

  if (A != B)
    return false;

  // Proved no overflow, nuke the overflow check and, if possible, the overflow
  // intrinsic as well.

  BinaryOperator *NewResult = BinaryOperator::Create(
      RawOp, CI->getArgOperand(0), CI->getArgOperand(1), "", CI);

  if (NoSignedOverflow)
    NewResult->setHasNoSignedWrap(true);
  else
    NewResult->setHasNoUnsignedWrap(true);

  SmallVector<ExtractValueInst *, 4> ToDelete;

  for (auto *U : CI->users()) {
    if (auto *EVI = dyn_cast<ExtractValueInst>(U)) {
      if (EVI->getIndices()[0] == 1)
        EVI->replaceAllUsesWith(ConstantInt::getFalse(CI->getContext()));
      else {
        assert(EVI->getIndices()[0] == 0 && "Only two possibilities!");
        EVI->replaceAllUsesWith(NewResult);
      }
      ToDelete.push_back(EVI);
    }
  }

  for (auto *EVI : ToDelete)
    EVI->eraseFromParent();

  if (CI->use_empty())
    CI->eraseFromParent();

  return true;
}

/// Eliminate an operation that consumes a simple IV and has no observable
/// side-effect given the range of IV values.  IVOperand is guaranteed SCEVable,
/// but UseInst may not be.
bool SimplifyIndvar::eliminateIVUser(Instruction *UseInst,
                                     Instruction *IVOperand) {
  if (ICmpInst *ICmp = dyn_cast<ICmpInst>(UseInst)) {
    eliminateIVComparison(ICmp, IVOperand);
    return true;
  }
  if (BinaryOperator *Bin = dyn_cast<BinaryOperator>(UseInst)) {
    bool IsSRem = Bin->getOpcode() == Instruction::SRem;
    if (IsSRem || Bin->getOpcode() == Instruction::URem) {
      eliminateIVRemainder(Bin, IVOperand, IsSRem);
      return true;
    }

    if (Bin->getOpcode() == Instruction::SDiv)
      return eliminateSDiv(Bin);
  }

  if (auto *CI = dyn_cast<CallInst>(UseInst))
    if (eliminateOverflowIntrinsic(CI))
      return true;

  if (eliminateIdentitySCEV(UseInst, IVOperand))
    return true;

  return false;
}

/// Eliminate any operation that SCEV can prove is an identity function.
bool SimplifyIndvar::eliminateIdentitySCEV(Instruction *UseInst,
                                           Instruction *IVOperand) {
  if (!SE->isSCEVable(UseInst->getType()) ||
      (UseInst->getType() != IVOperand->getType()) ||
      (SE->getSCEV(UseInst) != SE->getSCEV(IVOperand)))
    return false;

  // getSCEV(X) == getSCEV(Y) does not guarantee that X and Y are related in the
  // dominator tree, even if X is an operand to Y.  For instance, in
  //
  //     %iv = phi i32 {0,+,1}
  //     br %cond, label %left, label %merge
  //
  //   left:
  //     %X = add i32 %iv, 0
  //     br label %merge
  //
  //   merge:
  //     %M = phi (%X, %iv)
  //
  // getSCEV(%M) == getSCEV(%X) == {0,+,1}, but %X does not dominate %M, and
  // %M.replaceAllUsesWith(%X) would be incorrect.

  if (isa<PHINode>(UseInst))
    // If UseInst is not a PHI node then we know that IVOperand dominates
    // UseInst directly from the legality of SSA.
    if (!DT || !DT->dominates(IVOperand, UseInst))
      return false;

  if (!LI->replacementPreservesLCSSAForm(UseInst, IVOperand))
    return false;

  DEBUG(dbgs() << "INDVARS: Eliminated identity: " << *UseInst << '\n');

  UseInst->replaceAllUsesWith(IVOperand);
  ++NumElimIdentity;
  Changed = true;
  DeadInsts.emplace_back(UseInst);
  return true;
}

/// Annotate BO with nsw / nuw if it provably does not signed-overflow /
/// unsigned-overflow.  Returns true if anything changed, false otherwise.
bool SimplifyIndvar::strengthenOverflowingOperation(BinaryOperator *BO,
                                                    Value *IVOperand) {

  // Fastpath: we don't have any work to do if `BO` is `nuw` and `nsw`.
  if (BO->hasNoUnsignedWrap() && BO->hasNoSignedWrap())
    return false;

  const SCEV *(ScalarEvolution::*GetExprForBO)(const SCEV *, const SCEV *,
                                               SCEV::NoWrapFlags, unsigned);
  switch (BO->getOpcode()) {
  default:
    return false;

  case Instruction::Add:
    GetExprForBO = &ScalarEvolution::getAddExpr;
    break;

  case Instruction::Sub:
    GetExprForBO = &ScalarEvolution::getMinusSCEV;
    break;

  case Instruction::Mul:
    GetExprForBO = &ScalarEvolution::getMulExpr;
    break;
  }

  unsigned BitWidth = cast<IntegerType>(BO->getType())->getBitWidth();
  Type *WideTy = IntegerType::get(BO->getContext(), BitWidth * 2);
  const SCEV *LHS = SE->getSCEV(BO->getOperand(0));
  const SCEV *RHS = SE->getSCEV(BO->getOperand(1));

  bool Changed = false;

  if (!BO->hasNoUnsignedWrap()) {
    const SCEV *ExtendAfterOp = SE->getZeroExtendExpr(SE->getSCEV(BO), WideTy);
    const SCEV *OpAfterExtend = (SE->*GetExprForBO)(
      SE->getZeroExtendExpr(LHS, WideTy), SE->getZeroExtendExpr(RHS, WideTy),
      SCEV::FlagAnyWrap, 0u);
    if (ExtendAfterOp == OpAfterExtend) {
      BO->setHasNoUnsignedWrap();
      SE->forgetValue(BO);
      Changed = true;
    }
  }

  if (!BO->hasNoSignedWrap()) {
    const SCEV *ExtendAfterOp = SE->getSignExtendExpr(SE->getSCEV(BO), WideTy);
    const SCEV *OpAfterExtend = (SE->*GetExprForBO)(
      SE->getSignExtendExpr(LHS, WideTy), SE->getSignExtendExpr(RHS, WideTy),
      SCEV::FlagAnyWrap, 0u);
    if (ExtendAfterOp == OpAfterExtend) {
      BO->setHasNoSignedWrap();
      SE->forgetValue(BO);
      Changed = true;
    }
  }

  return Changed;
}

/// Annotate the Shr in (X << IVOperand) >> C as exact using the
/// information from the IV's range. Returns true if anything changed, false
/// otherwise.
bool SimplifyIndvar::strengthenRightShift(BinaryOperator *BO,
                                          Value *IVOperand) {
  using namespace llvm::PatternMatch;

  if (BO->getOpcode() == Instruction::Shl) {
    bool Changed = false;
    ConstantRange IVRange = SE->getUnsignedRange(SE->getSCEV(IVOperand));
    for (auto *U : BO->users()) {
      const APInt *C;
      if (match(U,
                m_AShr(m_Shl(m_Value(), m_Specific(IVOperand)), m_APInt(C))) ||
          match(U,
                m_LShr(m_Shl(m_Value(), m_Specific(IVOperand)), m_APInt(C)))) {
        BinaryOperator *Shr = cast<BinaryOperator>(U);
        if (!Shr->isExact() && IVRange.getUnsignedMin().uge(*C)) {
          Shr->setIsExact(true);
          Changed = true;
        }
      }
    }
    return Changed;
  }

  return false;
}

/// Add all uses of Def to the current IV's worklist.
static void pushIVUsers(
  Instruction *Def,
  SmallPtrSet<Instruction*,16> &Simplified,
  SmallVectorImpl< std::pair<Instruction*,Instruction*> > &SimpleIVUsers) {

  for (User *U : Def->users()) {
    Instruction *UI = cast<Instruction>(U);

    // Avoid infinite or exponential worklist processing.
    // Also ensure unique worklist users.
    // If Def is a LoopPhi, it may not be in the Simplified set, so check for
    // self edges first.
    if (UI != Def && Simplified.insert(UI).second)
      SimpleIVUsers.push_back(std::make_pair(UI, Def));
  }
}

/// Return true if this instruction generates a simple SCEV
/// expression in terms of that IV.
///
/// This is similar to IVUsers' isInteresting() but processes each instruction
/// non-recursively when the operand is already known to be a simpleIVUser.
///
static bool isSimpleIVUser(Instruction *I, const Loop *L, ScalarEvolution *SE) {
  if (!SE->isSCEVable(I->getType()))
    return false;

  // Get the symbolic expression for this instruction.
  const SCEV *S = SE->getSCEV(I);

  // Only consider affine recurrences.
  const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(S);
  if (AR && AR->getLoop() == L)
    return true;

  return false;
}

/// Iteratively perform simplification on a worklist of users
/// of the specified induction variable. Each successive simplification may push
/// more users which may themselves be candidates for simplification.
///
/// This algorithm does not require IVUsers analysis. Instead, it simplifies
/// instructions in-place during analysis. Rather than rewriting induction
/// variables bottom-up from their users, it transforms a chain of IVUsers
/// top-down, updating the IR only when it encounters a clear optimization
/// opportunity.
///
/// Once DisableIVRewrite is default, LSR will be the only client of IVUsers.
///
void SimplifyIndvar::simplifyUsers(PHINode *CurrIV, IVVisitor *V) {
  if (!SE->isSCEVable(CurrIV->getType()))
    return;

  // Instructions processed by SimplifyIndvar for CurrIV.
  SmallPtrSet<Instruction*,16> Simplified;

  // Use-def pairs if IV users waiting to be processed for CurrIV.
  SmallVector<std::pair<Instruction*, Instruction*>, 8> SimpleIVUsers;

  // Push users of the current LoopPhi. In rare cases, pushIVUsers may be
  // called multiple times for the same LoopPhi. This is the proper thing to
  // do for loop header phis that use each other.
  pushIVUsers(CurrIV, Simplified, SimpleIVUsers);

  while (!SimpleIVUsers.empty()) {
    std::pair<Instruction*, Instruction*> UseOper =
      SimpleIVUsers.pop_back_val();
    Instruction *UseInst = UseOper.first;

    // Bypass back edges to avoid extra work.
    if (UseInst == CurrIV) continue;

    Instruction *IVOperand = UseOper.second;
    for (unsigned N = 0; IVOperand; ++N) {
      assert(N <= Simplified.size() && "runaway iteration");

      Value *NewOper = foldIVUser(UseOper.first, IVOperand);
      if (!NewOper)
        break; // done folding
      IVOperand = dyn_cast<Instruction>(NewOper);
    }
    if (!IVOperand)
      continue;

    if (eliminateIVUser(UseOper.first, IVOperand)) {
      pushIVUsers(IVOperand, Simplified, SimpleIVUsers);
      continue;
    }

    if (BinaryOperator *BO = dyn_cast<BinaryOperator>(UseOper.first)) {
      if ((isa<OverflowingBinaryOperator>(BO) &&
           strengthenOverflowingOperation(BO, IVOperand)) ||
          (isa<ShlOperator>(BO) && strengthenRightShift(BO, IVOperand))) {
        // re-queue uses of the now modified binary operator and fall
        // through to the checks that remain.
        pushIVUsers(IVOperand, Simplified, SimpleIVUsers);
      }
    }

    CastInst *Cast = dyn_cast<CastInst>(UseOper.first);
    if (V && Cast) {
      V->visitCast(Cast);
      continue;
    }
    if (isSimpleIVUser(UseOper.first, L, SE)) {
      pushIVUsers(UseOper.first, Simplified, SimpleIVUsers);
    }
  }
}

namespace llvm {

void IVVisitor::anchor() { }

/// Simplify instructions that use this induction variable
/// by using ScalarEvolution to analyze the IV's recurrence.
bool simplifyUsersOfIV(PHINode *CurrIV, ScalarEvolution *SE, DominatorTree *DT,
                       LoopInfo *LI, SmallVectorImpl<WeakTrackingVH> &Dead,
                       IVVisitor *V) {
  SimplifyIndvar SIV(LI->getLoopFor(CurrIV->getParent()), SE, DT, LI, Dead);
  SIV.simplifyUsers(CurrIV, V);
  return SIV.hasChanged();
}

/// Simplify users of induction variables within this
/// loop. This does not actually change or add IVs.
bool simplifyLoopIVs(Loop *L, ScalarEvolution *SE, DominatorTree *DT,
                     LoopInfo *LI, SmallVectorImpl<WeakTrackingVH> &Dead) {
  bool Changed = false;
  for (BasicBlock::iterator I = L->getHeader()->begin(); isa<PHINode>(I); ++I) {
    Changed |= simplifyUsersOfIV(cast<PHINode>(I), SE, DT, LI, Dead);
  }
  return Changed;
}

} // namespace llvm
