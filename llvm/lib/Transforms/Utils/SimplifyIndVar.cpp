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
#include "llvm/Analysis/IVUsers.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "indvars"

STATISTIC(NumElimIdentity, "Number of IV identities eliminated");
STATISTIC(NumElimOperand,  "Number of IV operands folded into a use");
STATISTIC(NumElimRem     , "Number of IV remainder operations eliminated");
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
    const DataLayout *DL; // May be NULL

    SmallVectorImpl<WeakVH> &DeadInsts;

    bool Changed;

  public:
    SimplifyIndvar(Loop *Loop, ScalarEvolution *SE, LPPassManager *LPM,
                   SmallVectorImpl<WeakVH> &Dead, IVUsers *IVU = nullptr) :
      L(Loop),
      LI(LPM->getAnalysisIfAvailable<LoopInfo>()),
      SE(SE),
      DeadInsts(Dead),
      Changed(false) {
      DataLayoutPass *DLP = LPM->getAnalysisIfAvailable<DataLayoutPass>();
      DL = DLP ? &DLP->getDataLayout() : nullptr;
      assert(LI && "IV simplification requires LoopInfo");
    }

    bool hasChanged() const { return Changed; }

    /// Iteratively perform simplification on a worklist of users of the
    /// specified induction variable. This is the top-level driver that applies
    /// all simplicitions to users of an IV.
    void simplifyUsers(PHINode *CurrIV, IVVisitor *V = nullptr);

    Value *foldIVUser(Instruction *UseInst, Instruction *IVOperand);

    bool eliminateIVUser(Instruction *UseInst, Instruction *IVOperand);
    void eliminateIVComparison(ICmpInst *ICmp, Value *IVOperand);
    void eliminateIVRemainder(BinaryOperator *Rem, Value *IVOperand,
                              bool IsSigned);
    bool strengthenOverflowingOperation(BinaryOperator *OBO, Value *IVOperand);

    Instruction *splitOverflowIntrinsic(Instruction *IVUser,
                                        const DominatorTree *DT);
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
    DeadInsts.push_back(IVOperand);
  return IVSrc;
}

/// SimplifyIVUsers helper for eliminating useless
/// comparisons against an induction variable.
void SimplifyIndvar::eliminateIVComparison(ICmpInst *ICmp, Value *IVOperand) {
  unsigned IVOperIdx = 0;
  ICmpInst::Predicate Pred = ICmp->getPredicate();
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

  // If the condition is always true or always false, replace it with
  // a constant value.
  if (SE->isKnownPredicate(Pred, S, X))
    ICmp->replaceAllUsesWith(ConstantInt::getTrue(ICmp->getContext()));
  else if (SE->isKnownPredicate(ICmpInst::getInversePredicate(Pred), S, X))
    ICmp->replaceAllUsesWith(ConstantInt::getFalse(ICmp->getContext()));
  else
    return;

  DEBUG(dbgs() << "INDVARS: Eliminated comparison: " << *ICmp << '\n');
  ++NumElimCmp;
  Changed = true;
  DeadInsts.push_back(ICmp);
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
    const SCEV *LessOne =
      SE->getMinusSCEV(S, SE->getConstant(S->getType(), 1));
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
  DeadInsts.push_back(Rem);
}

/// Eliminate an operation that consumes a simple IV and has
/// no observable side-effect given the range of IV values.
/// IVOperand is guaranteed SCEVable, but UseInst may not be.
bool SimplifyIndvar::eliminateIVUser(Instruction *UseInst,
                                     Instruction *IVOperand) {
  if (ICmpInst *ICmp = dyn_cast<ICmpInst>(UseInst)) {
    eliminateIVComparison(ICmp, IVOperand);
    return true;
  }
  if (BinaryOperator *Rem = dyn_cast<BinaryOperator>(UseInst)) {
    bool IsSigned = Rem->getOpcode() == Instruction::SRem;
    if (IsSigned || Rem->getOpcode() == Instruction::URem) {
      eliminateIVRemainder(Rem, IVOperand, IsSigned);
      return true;
    }
  }

  // Eliminate any operation that SCEV can prove is an identity function.
  if (!SE->isSCEVable(UseInst->getType()) ||
      (UseInst->getType() != IVOperand->getType()) ||
      (SE->getSCEV(UseInst) != SE->getSCEV(IVOperand)))
    return false;

  DEBUG(dbgs() << "INDVARS: Eliminated identity: " << *UseInst << '\n');

  UseInst->replaceAllUsesWith(IVOperand);
  ++NumElimIdentity;
  Changed = true;
  DeadInsts.push_back(UseInst);
  return true;
}

/// Annotate BO with nsw / nuw if it provably does not signed-overflow /
/// unsigned-overflow.  Returns true if anything changed, false otherwise.
bool SimplifyIndvar::strengthenOverflowingOperation(BinaryOperator *BO,
                                                    Value *IVOperand) {

  // Currently we only handle instructions of the form "add <indvar> <value>"
  unsigned Op = BO->getOpcode();
  if (Op != Instruction::Add)
    return false;

  // If BO is already both nuw and nsw then there is nothing left to do
  if (BO->hasNoUnsignedWrap() && BO->hasNoSignedWrap())
    return false;

  IntegerType *IT = cast<IntegerType>(IVOperand->getType());
  Value *OtherOperand = nullptr;
  int OtherOperandIdx = -1;
  if (BO->getOperand(0) == IVOperand) {
    OtherOperand = BO->getOperand(1);
    OtherOperandIdx = 1;
  } else {
    assert(BO->getOperand(1) == IVOperand && "only other use!");
    OtherOperand = BO->getOperand(0);
    OtherOperandIdx = 0;
  }

  bool Changed = false;
  const SCEV *OtherOpSCEV = SE->getSCEV(OtherOperand);
  if (OtherOpSCEV == SE->getCouldNotCompute())
    return false;

  const SCEV *IVOpSCEV = SE->getSCEV(IVOperand);
  const SCEV *ZeroSCEV = SE->getConstant(IVOpSCEV->getType(), 0);

  if (!BO->hasNoSignedWrap()) {
    // Upgrade the add to an "add nsw" if we can prove that it will never
    // sign-overflow or sign-underflow.

    const SCEV *SignedMax =
      SE->getConstant(APInt::getSignedMaxValue(IT->getBitWidth()));
    const SCEV *SignedMin =
      SE->getConstant(APInt::getSignedMinValue(IT->getBitWidth()));

    // The addition "IVOperand + OtherOp" does not sign-overflow if the result
    // is sign-representable in 2's complement in the given bit-width.
    //
    // If OtherOp is SLT 0, then for an IVOperand in [SignedMin - OtherOp,
    // SignedMax], "IVOperand + OtherOp" is in [SignedMin, SignedMax + OtherOp].
    // Everything in [SignedMin, SignedMax + OtherOp] is representable since
    // SignedMax + OtherOp is at least -1.
    //
    // If OtherOp is SGE 0, then for an IVOperand in [SignedMin, SignedMax -
    // OtherOp], "IVOperand + OtherOp" is in [SignedMin + OtherOp, SignedMax].
    // Everything in [SignedMin + OtherOp, SignedMax] is representable since
    // SignedMin + OtherOp is at most -1.
    //
    // It follows that for all values of IVOperand in [SignedMin - smin(0,
    // OtherOp), SignedMax - smax(0, OtherOp)] the result of the add is
    // representable (i.e. there is no sign-overflow).

    const SCEV *UpperDelta = SE->getSMaxExpr(ZeroSCEV, OtherOpSCEV);
    const SCEV *UpperLimit = SE->getMinusSCEV(SignedMax, UpperDelta);

    bool NeverSignedOverflows =
      SE->isKnownPredicate(ICmpInst::ICMP_SLE, IVOpSCEV, UpperLimit);

    if (NeverSignedOverflows) {
      const SCEV *LowerDelta = SE->getSMinExpr(ZeroSCEV, OtherOpSCEV);
      const SCEV *LowerLimit = SE->getMinusSCEV(SignedMin, LowerDelta);

      bool NeverSignedUnderflows =
        SE->isKnownPredicate(ICmpInst::ICMP_SGE, IVOpSCEV, LowerLimit);
      if (NeverSignedUnderflows) {
        BO->setHasNoSignedWrap(true);
        Changed = true;
      }
    }
  }

  if (!BO->hasNoUnsignedWrap()) {
    // Upgrade the add computing "IVOperand + OtherOp" to an "add nuw" if we can
    // prove that it will never unsigned-overflow (i.e. the result will always
    // be representable in the given bit-width).
    //
    // "IVOperand + OtherOp" is unsigned-representable in 2's complement iff it
    // does not produce a carry.  "IVOperand + OtherOp" produces no carry iff
    // IVOperand ULE (UnsignedMax - OtherOp).

    const SCEV *UnsignedMax =
      SE->getConstant(APInt::getMaxValue(IT->getBitWidth()));
    const SCEV *UpperLimit = SE->getMinusSCEV(UnsignedMax, OtherOpSCEV);

    bool NeverUnsignedOverflows =
        SE->isKnownPredicate(ICmpInst::ICMP_ULE, IVOpSCEV, UpperLimit);

    if (NeverUnsignedOverflows) {
      BO->setHasNoUnsignedWrap(true);
      Changed = true;
    }
  }

  return Changed;
}

/// \brief Split sadd.with.overflow into add + sadd.with.overflow to allow
/// analysis and optimization.
///
/// \return A new value representing the non-overflowing add if possible,
/// otherwise return the original value.
Instruction *SimplifyIndvar::splitOverflowIntrinsic(Instruction *IVUser,
                                                    const DominatorTree *DT) {
  IntrinsicInst *II = dyn_cast<IntrinsicInst>(IVUser);
  if (!II || II->getIntrinsicID() != Intrinsic::sadd_with_overflow)
    return IVUser;

  // Find a branch guarded by the overflow check.
  BranchInst *Branch = nullptr;
  Instruction *AddVal = nullptr;
  for (User *U : II->users()) {
    if (ExtractValueInst *ExtractInst = dyn_cast<ExtractValueInst>(U)) {
      if (ExtractInst->getNumIndices() != 1)
        continue;
      if (ExtractInst->getIndices()[0] == 0)
        AddVal = ExtractInst;
      else if (ExtractInst->getIndices()[0] == 1 && ExtractInst->hasOneUse())
        Branch = dyn_cast<BranchInst>(ExtractInst->user_back());
    }
  }
  if (!AddVal || !Branch)
    return IVUser;

  BasicBlock *ContinueBB = Branch->getSuccessor(1);
  if (std::next(pred_begin(ContinueBB)) != pred_end(ContinueBB))
    return IVUser;

  // Check if all users of the add are provably NSW.
  bool AllNSW = true;
  for (Use &U : AddVal->uses()) {
    if (Instruction *UseInst = dyn_cast<Instruction>(U.getUser())) {
      BasicBlock *UseBB = UseInst->getParent();
      if (PHINode *PHI = dyn_cast<PHINode>(UseInst))
        UseBB = PHI->getIncomingBlock(U);
      if (!DT->dominates(ContinueBB, UseBB)) {
        AllNSW = false;
        break;
      }
    }
  }
  if (!AllNSW)
    return IVUser;

  // Go for it...
  IRBuilder<> Builder(IVUser);
  Instruction *AddInst = dyn_cast<Instruction>(
    Builder.CreateNSWAdd(II->getOperand(0), II->getOperand(1)));

  // The caller expects the new add to have the same form as the intrinsic. The
  // IV operand position must be the same.
  assert((AddInst->getOpcode() == Instruction::Add &&
          AddInst->getOperand(0) == II->getOperand(0)) &&
         "Bad add instruction created from overflow intrinsic.");

  AddVal->replaceAllUsesWith(AddInst);
  DeadInsts.push_back(AddVal);
  return AddInst;
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
/// top-down, updating the IR only when it encouters a clear optimization
/// opportunitiy.
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

    if (V && V->shouldSplitOverflowInstrinsics()) {
      UseInst = splitOverflowIntrinsic(UseInst, V->getDomTree());
      if (!UseInst)
        continue;
    }

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
      if (isa<OverflowingBinaryOperator>(BO) &&
          strengthenOverflowingOperation(BO, IVOperand)) {
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
bool simplifyUsersOfIV(PHINode *CurrIV, ScalarEvolution *SE, LPPassManager *LPM,
                       SmallVectorImpl<WeakVH> &Dead, IVVisitor *V)
{
  LoopInfo *LI = &LPM->getAnalysis<LoopInfo>();
  SimplifyIndvar SIV(LI->getLoopFor(CurrIV->getParent()), SE, LPM, Dead);
  SIV.simplifyUsers(CurrIV, V);
  return SIV.hasChanged();
}

/// Simplify users of induction variables within this
/// loop. This does not actually change or add IVs.
bool simplifyLoopIVs(Loop *L, ScalarEvolution *SE, LPPassManager *LPM,
                     SmallVectorImpl<WeakVH> &Dead) {
  bool Changed = false;
  for (BasicBlock::iterator I = L->getHeader()->begin(); isa<PHINode>(I); ++I) {
    Changed |= simplifyUsersOfIV(cast<PHINode>(I), SE, LPM, Dead);
  }
  return Changed;
}

} // namespace llvm
