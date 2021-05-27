//===- AggressiveInstCombine.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the aggressive expression pattern combiner classes.
// Currently, it handles expression patterns for:
//  * Truncate instruction
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/AggressiveInstCombine/AggressiveInstCombine.h"
#include "AggressiveInstCombineInternal.h"
#include "llvm-c/Initialization.h"
#include "llvm-c/Transforms/AggressiveInstCombine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "aggressive-instcombine"

STATISTIC(NumAnyOrAllBitsSet, "Number of any/all-bits-set patterns folded");
STATISTIC(NumGuardedRotates,
          "Number of guarded rotates transformed into funnel shifts");
STATISTIC(NumGuardedFunnelShifts,
          "Number of guarded funnel shifts transformed into funnel shifts");
STATISTIC(NumPopCountRecognized, "Number of popcount idioms recognized");

namespace {
/// Contains expression pattern combiner logic.
/// This class provides both the logic to combine expression patterns and
/// combine them. It differs from InstCombiner class in that each pattern
/// combiner runs only once as opposed to InstCombine's multi-iteration,
/// which allows pattern combiner to have higher complexity than the O(1)
/// required by the instruction combiner.
class AggressiveInstCombinerLegacyPass : public FunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid

  AggressiveInstCombinerLegacyPass() : FunctionPass(ID) {
    initializeAggressiveInstCombinerLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  /// Run all expression pattern optimizations on the given /p F function.
  ///
  /// \param F function to optimize.
  /// \returns true if the IR is changed.
  bool runOnFunction(Function &F) override;
};
} // namespace

/// Match a pattern for a bitwise funnel/rotate operation that partially guards
/// against undefined behavior by branching around the funnel-shift/rotation
/// when the shift amount is 0.
static bool foldGuardedFunnelShift(Instruction &I, const DominatorTree &DT) {
  if (I.getOpcode() != Instruction::PHI || I.getNumOperands() != 2)
    return false;

  // As with the one-use checks below, this is not strictly necessary, but we
  // are being cautious to avoid potential perf regressions on targets that
  // do not actually have a funnel/rotate instruction (where the funnel shift
  // would be expanded back into math/shift/logic ops).
  if (!isPowerOf2_32(I.getType()->getScalarSizeInBits()))
    return false;

  // Match V to funnel shift left/right and capture the source operands and
  // shift amount.
  auto matchFunnelShift = [](Value *V, Value *&ShVal0, Value *&ShVal1,
                             Value *&ShAmt) {
    Value *SubAmt;
    unsigned Width = V->getType()->getScalarSizeInBits();

    // fshl(ShVal0, ShVal1, ShAmt)
    //  == (ShVal0 << ShAmt) | (ShVal1 >> (Width -ShAmt))
    if (match(V, m_OneUse(m_c_Or(
                     m_Shl(m_Value(ShVal0), m_Value(ShAmt)),
                     m_LShr(m_Value(ShVal1),
                            m_Sub(m_SpecificInt(Width), m_Value(SubAmt))))))) {
      if (ShAmt == SubAmt) // TODO: Use m_Specific
        return Intrinsic::fshl;
    }

    // fshr(ShVal0, ShVal1, ShAmt)
    //  == (ShVal0 >> ShAmt) | (ShVal1 << (Width - ShAmt))
    if (match(V,
              m_OneUse(m_c_Or(m_Shl(m_Value(ShVal0), m_Sub(m_SpecificInt(Width),
                                                           m_Value(SubAmt))),
                              m_LShr(m_Value(ShVal1), m_Value(ShAmt)))))) {
      if (ShAmt == SubAmt) // TODO: Use m_Specific
        return Intrinsic::fshr;
    }

    return Intrinsic::not_intrinsic;
  };

  // One phi operand must be a funnel/rotate operation, and the other phi
  // operand must be the source value of that funnel/rotate operation:
  // phi [ rotate(RotSrc, ShAmt), FunnelBB ], [ RotSrc, GuardBB ]
  // phi [ fshl(ShVal0, ShVal1, ShAmt), FunnelBB ], [ ShVal0, GuardBB ]
  // phi [ fshr(ShVal0, ShVal1, ShAmt), FunnelBB ], [ ShVal1, GuardBB ]
  PHINode &Phi = cast<PHINode>(I);
  unsigned FunnelOp = 0, GuardOp = 1;
  Value *P0 = Phi.getOperand(0), *P1 = Phi.getOperand(1);
  Value *ShVal0, *ShVal1, *ShAmt;
  Intrinsic::ID IID = matchFunnelShift(P0, ShVal0, ShVal1, ShAmt);
  if (IID == Intrinsic::not_intrinsic ||
      (IID == Intrinsic::fshl && ShVal0 != P1) ||
      (IID == Intrinsic::fshr && ShVal1 != P1)) {
    IID = matchFunnelShift(P1, ShVal0, ShVal1, ShAmt);
    if (IID == Intrinsic::not_intrinsic ||
        (IID == Intrinsic::fshl && ShVal0 != P0) ||
        (IID == Intrinsic::fshr && ShVal1 != P0))
      return false;
    assert((IID == Intrinsic::fshl || IID == Intrinsic::fshr) &&
           "Pattern must match funnel shift left or right");
    std::swap(FunnelOp, GuardOp);
  }

  // The incoming block with our source operand must be the "guard" block.
  // That must contain a cmp+branch to avoid the funnel/rotate when the shift
  // amount is equal to 0. The other incoming block is the block with the
  // funnel/rotate.
  BasicBlock *GuardBB = Phi.getIncomingBlock(GuardOp);
  BasicBlock *FunnelBB = Phi.getIncomingBlock(FunnelOp);
  Instruction *TermI = GuardBB->getTerminator();

  // Ensure that the shift values dominate each block.
  if (!DT.dominates(ShVal0, TermI) || !DT.dominates(ShVal1, TermI))
    return false;

  ICmpInst::Predicate Pred;
  BasicBlock *PhiBB = Phi.getParent();
  if (!match(TermI, m_Br(m_ICmp(Pred, m_Specific(ShAmt), m_ZeroInt()),
                         m_SpecificBB(PhiBB), m_SpecificBB(FunnelBB))))
    return false;

  if (Pred != CmpInst::ICMP_EQ)
    return false;

  IRBuilder<> Builder(PhiBB, PhiBB->getFirstInsertionPt());

  if (ShVal0 == ShVal1)
    ++NumGuardedRotates;
  else
    ++NumGuardedFunnelShifts;

  // If this is not a rotate then the select was blocking poison from the
  // 'shift-by-zero' non-TVal, but a funnel shift won't - so freeze it.
  bool IsFshl = IID == Intrinsic::fshl;
  if (ShVal0 != ShVal1) {
    if (IsFshl && !llvm::isGuaranteedNotToBePoison(ShVal1))
      ShVal1 = Builder.CreateFreeze(ShVal1);
    else if (!IsFshl && !llvm::isGuaranteedNotToBePoison(ShVal0))
      ShVal0 = Builder.CreateFreeze(ShVal0);
  }

  // We matched a variation of this IR pattern:
  // GuardBB:
  //   %cmp = icmp eq i32 %ShAmt, 0
  //   br i1 %cmp, label %PhiBB, label %FunnelBB
  // FunnelBB:
  //   %sub = sub i32 32, %ShAmt
  //   %shr = lshr i32 %ShVal1, %sub
  //   %shl = shl i32 %ShVal0, %ShAmt
  //   %fsh = or i32 %shr, %shl
  //   br label %PhiBB
  // PhiBB:
  //   %cond = phi i32 [ %fsh, %FunnelBB ], [ %ShVal0, %GuardBB ]
  // -->
  // llvm.fshl.i32(i32 %ShVal0, i32 %ShVal1, i32 %ShAmt)
  Function *F = Intrinsic::getDeclaration(Phi.getModule(), IID, Phi.getType());
  Phi.replaceAllUsesWith(Builder.CreateCall(F, {ShVal0, ShVal1, ShAmt}));
  return true;
}

/// This is used by foldAnyOrAllBitsSet() to capture a source value (Root) and
/// the bit indexes (Mask) needed by a masked compare. If we're matching a chain
/// of 'and' ops, then we also need to capture the fact that we saw an
/// "and X, 1", so that's an extra return value for that case.
struct MaskOps {
  Value *Root;
  APInt Mask;
  bool MatchAndChain;
  bool FoundAnd1;

  MaskOps(unsigned BitWidth, bool MatchAnds)
      : Root(nullptr), Mask(APInt::getNullValue(BitWidth)),
        MatchAndChain(MatchAnds), FoundAnd1(false) {}
};

/// This is a recursive helper for foldAnyOrAllBitsSet() that walks through a
/// chain of 'and' or 'or' instructions looking for shift ops of a common source
/// value. Examples:
///   or (or (or X, (X >> 3)), (X >> 5)), (X >> 8)
/// returns { X, 0x129 }
///   and (and (X >> 1), 1), (X >> 4)
/// returns { X, 0x12 }
static bool matchAndOrChain(Value *V, MaskOps &MOps) {
  Value *Op0, *Op1;
  if (MOps.MatchAndChain) {
    // Recurse through a chain of 'and' operands. This requires an extra check
    // vs. the 'or' matcher: we must find an "and X, 1" instruction somewhere
    // in the chain to know that all of the high bits are cleared.
    if (match(V, m_And(m_Value(Op0), m_One()))) {
      MOps.FoundAnd1 = true;
      return matchAndOrChain(Op0, MOps);
    }
    if (match(V, m_And(m_Value(Op0), m_Value(Op1))))
      return matchAndOrChain(Op0, MOps) && matchAndOrChain(Op1, MOps);
  } else {
    // Recurse through a chain of 'or' operands.
    if (match(V, m_Or(m_Value(Op0), m_Value(Op1))))
      return matchAndOrChain(Op0, MOps) && matchAndOrChain(Op1, MOps);
  }

  // We need a shift-right or a bare value representing a compare of bit 0 of
  // the original source operand.
  Value *Candidate;
  const APInt *BitIndex = nullptr;
  if (!match(V, m_LShr(m_Value(Candidate), m_APInt(BitIndex))))
    Candidate = V;

  // Initialize result source operand.
  if (!MOps.Root)
    MOps.Root = Candidate;

  // The shift constant is out-of-range? This code hasn't been simplified.
  if (BitIndex && BitIndex->uge(MOps.Mask.getBitWidth()))
    return false;

  // Fill in the mask bit derived from the shift constant.
  MOps.Mask.setBit(BitIndex ? BitIndex->getZExtValue() : 0);
  return MOps.Root == Candidate;
}

/// Match patterns that correspond to "any-bits-set" and "all-bits-set".
/// These will include a chain of 'or' or 'and'-shifted bits from a
/// common source value:
/// and (or  (lshr X, C), ...), 1 --> (X & CMask) != 0
/// and (and (lshr X, C), ...), 1 --> (X & CMask) == CMask
/// Note: "any-bits-clear" and "all-bits-clear" are variations of these patterns
/// that differ only with a final 'not' of the result. We expect that final
/// 'not' to be folded with the compare that we create here (invert predicate).
static bool foldAnyOrAllBitsSet(Instruction &I) {
  // The 'any-bits-set' ('or' chain) pattern is simpler to match because the
  // final "and X, 1" instruction must be the final op in the sequence.
  bool MatchAllBitsSet;
  if (match(&I, m_c_And(m_OneUse(m_And(m_Value(), m_Value())), m_Value())))
    MatchAllBitsSet = true;
  else if (match(&I, m_And(m_OneUse(m_Or(m_Value(), m_Value())), m_One())))
    MatchAllBitsSet = false;
  else
    return false;

  MaskOps MOps(I.getType()->getScalarSizeInBits(), MatchAllBitsSet);
  if (MatchAllBitsSet) {
    if (!matchAndOrChain(cast<BinaryOperator>(&I), MOps) || !MOps.FoundAnd1)
      return false;
  } else {
    if (!matchAndOrChain(cast<BinaryOperator>(&I)->getOperand(0), MOps))
      return false;
  }

  // The pattern was found. Create a masked compare that replaces all of the
  // shift and logic ops.
  IRBuilder<> Builder(&I);
  Constant *Mask = ConstantInt::get(I.getType(), MOps.Mask);
  Value *And = Builder.CreateAnd(MOps.Root, Mask);
  Value *Cmp = MatchAllBitsSet ? Builder.CreateICmpEQ(And, Mask)
                               : Builder.CreateIsNotNull(And);
  Value *Zext = Builder.CreateZExt(Cmp, I.getType());
  I.replaceAllUsesWith(Zext);
  ++NumAnyOrAllBitsSet;
  return true;
}

// Try to recognize below function as popcount intrinsic.
// This is the "best" algorithm from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
// Also used in TargetLowering::expandCTPOP().
//
// int popcount(unsigned int i) {
//   i = i - ((i >> 1) & 0x55555555);
//   i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
//   i = ((i + (i >> 4)) & 0x0F0F0F0F);
//   return (i * 0x01010101) >> 24;
// }
static bool tryToRecognizePopCount(Instruction &I) {
  if (I.getOpcode() != Instruction::LShr)
    return false;

  Type *Ty = I.getType();
  if (!Ty->isIntOrIntVectorTy())
    return false;

  unsigned Len = Ty->getScalarSizeInBits();
  // FIXME: fix Len == 8 and other irregular type lengths.
  if (!(Len <= 128 && Len > 8 && Len % 8 == 0))
    return false;

  APInt Mask55 = APInt::getSplat(Len, APInt(8, 0x55));
  APInt Mask33 = APInt::getSplat(Len, APInt(8, 0x33));
  APInt Mask0F = APInt::getSplat(Len, APInt(8, 0x0F));
  APInt Mask01 = APInt::getSplat(Len, APInt(8, 0x01));
  APInt MaskShift = APInt(Len, Len - 8);

  Value *Op0 = I.getOperand(0);
  Value *Op1 = I.getOperand(1);
  Value *MulOp0;
  // Matching "(i * 0x01010101...) >> 24".
  if ((match(Op0, m_Mul(m_Value(MulOp0), m_SpecificInt(Mask01)))) &&
       match(Op1, m_SpecificInt(MaskShift))) {
    Value *ShiftOp0;
    // Matching "((i + (i >> 4)) & 0x0F0F0F0F...)".
    if (match(MulOp0, m_And(m_c_Add(m_LShr(m_Value(ShiftOp0), m_SpecificInt(4)),
                                    m_Deferred(ShiftOp0)),
                            m_SpecificInt(Mask0F)))) {
      Value *AndOp0;
      // Matching "(i & 0x33333333...) + ((i >> 2) & 0x33333333...)".
      if (match(ShiftOp0,
                m_c_Add(m_And(m_Value(AndOp0), m_SpecificInt(Mask33)),
                        m_And(m_LShr(m_Deferred(AndOp0), m_SpecificInt(2)),
                              m_SpecificInt(Mask33))))) {
        Value *Root, *SubOp1;
        // Matching "i - ((i >> 1) & 0x55555555...)".
        if (match(AndOp0, m_Sub(m_Value(Root), m_Value(SubOp1))) &&
            match(SubOp1, m_And(m_LShr(m_Specific(Root), m_SpecificInt(1)),
                                m_SpecificInt(Mask55)))) {
          LLVM_DEBUG(dbgs() << "Recognized popcount intrinsic\n");
          IRBuilder<> Builder(&I);
          Function *Func = Intrinsic::getDeclaration(
              I.getModule(), Intrinsic::ctpop, I.getType());
          I.replaceAllUsesWith(Builder.CreateCall(Func, {Root}));
          ++NumPopCountRecognized;
          return true;
        }
      }
    }
  }

  return false;
}

/// This is the entry point for folds that could be implemented in regular
/// InstCombine, but they are separated because they are not expected to
/// occur frequently and/or have more than a constant-length pattern match.
static bool foldUnusualPatterns(Function &F, DominatorTree &DT) {
  bool MadeChange = false;
  for (BasicBlock &BB : F) {
    // Ignore unreachable basic blocks.
    if (!DT.isReachableFromEntry(&BB))
      continue;
    // Do not delete instructions under here and invalidate the iterator.
    // Walk the block backwards for efficiency. We're matching a chain of
    // use->defs, so we're more likely to succeed by starting from the bottom.
    // Also, we want to avoid matching partial patterns.
    // TODO: It would be more efficient if we removed dead instructions
    // iteratively in this loop rather than waiting until the end.
    for (Instruction &I : make_range(BB.rbegin(), BB.rend())) {
      MadeChange |= foldAnyOrAllBitsSet(I);
      MadeChange |= foldGuardedFunnelShift(I, DT);
      MadeChange |= tryToRecognizePopCount(I); 
    }
  }

  // We're done with transforms, so remove dead instructions.
  if (MadeChange)
    for (BasicBlock &BB : F)
      SimplifyInstructionsInBlock(&BB);

  return MadeChange;
}

/// This is the entry point for all transforms. Pass manager differences are
/// handled in the callers of this function.
static bool runImpl(Function &F, TargetLibraryInfo &TLI, DominatorTree &DT) {
  bool MadeChange = false;
  const DataLayout &DL = F.getParent()->getDataLayout();
  TruncInstCombine TIC(TLI, DL, DT);
  MadeChange |= TIC.run(F);
  MadeChange |= foldUnusualPatterns(F, DT);
  return MadeChange;
}

void AggressiveInstCombinerLegacyPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addRequired<TargetLibraryInfoWrapperPass>();
  AU.addPreserved<AAResultsWrapperPass>();
  AU.addPreserved<BasicAAWrapperPass>();
  AU.addPreserved<DominatorTreeWrapperPass>();
  AU.addPreserved<GlobalsAAWrapperPass>();
}

bool AggressiveInstCombinerLegacyPass::runOnFunction(Function &F) {
  auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  return runImpl(F, TLI, DT);
}

PreservedAnalyses AggressiveInstCombinePass::run(Function &F,
                                                 FunctionAnalysisManager &AM) {
  auto &TLI = AM.getResult<TargetLibraryAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  if (!runImpl(F, TLI, DT)) {
    // No changes, all analyses are preserved.
    return PreservedAnalyses::all();
  }
  // Mark all the analyses that instcombine updates as preserved.
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

char AggressiveInstCombinerLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(AggressiveInstCombinerLegacyPass,
                      "aggressive-instcombine",
                      "Combine pattern based expressions", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(AggressiveInstCombinerLegacyPass, "aggressive-instcombine",
                    "Combine pattern based expressions", false, false)

// Initialization Routines
void llvm::initializeAggressiveInstCombine(PassRegistry &Registry) {
  initializeAggressiveInstCombinerLegacyPassPass(Registry);
}

void LLVMInitializeAggressiveInstCombiner(LLVMPassRegistryRef R) {
  initializeAggressiveInstCombinerLegacyPassPass(*unwrap(R));
}

FunctionPass *llvm::createAggressiveInstCombinerPass() {
  return new AggressiveInstCombinerLegacyPass();
}

void LLVMAddAggressiveInstCombinerPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createAggressiveInstCombinerPass());
}
