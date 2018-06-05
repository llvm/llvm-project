//===- AggressiveInstCombine.cpp ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Pass.h"
using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "aggressive-instcombine"

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

/// This is used by foldAnyOrAllBitsSet() to capture a source value (Root) and
/// the bit indexes (Mask) needed by a masked compare. If we're matching a chain
/// of 'and' ops, then we also need to capture the fact that we saw an
/// "and X, 1", so that's an extra return value for that case.
struct MaskOps {
  Value *Root;
  APInt Mask;
  bool MatchAndChain;
  bool FoundAnd1;

  MaskOps(unsigned BitWidth, bool MatchAnds) :
      Root(nullptr), Mask(APInt::getNullValue(BitWidth)),
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
  uint64_t BitIndex = 0;
  if (!match(V, m_LShr(m_Value(Candidate), m_ConstantInt(BitIndex))))
    Candidate = V;

  // Initialize result source operand.
  if (!MOps.Root)
    MOps.Root = Candidate;

  // The shift constant is out-of-range? This code hasn't been simplified.
  if (BitIndex >= MOps.Mask.getBitWidth())
    return false;

  // Fill in the mask bit derived from the shift constant.
  MOps.Mask.setBit(BitIndex);
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
  Value *Cmp = MatchAllBitsSet ? Builder.CreateICmpEQ(And, Mask) :
                                 Builder.CreateIsNotNull(And);
  Value *Zext = Builder.CreateZExt(Cmp, I.getType());
  I.replaceAllUsesWith(Zext);
  return true;
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
    for (Instruction &I : make_range(BB.rbegin(), BB.rend()))
      MadeChange |= foldAnyOrAllBitsSet(I);
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
  auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
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
  PA.preserve<AAManager>();
  PA.preserve<GlobalsAA>();
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
