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
#include "llvm/Analysis/Utils/Local.h"
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

/// This is a recursive helper for 'and X, 1' that walks through a chain of 'or'
/// instructions looking for shift ops of a common source value (first member of
/// the pair). The second member of the pair is a mask constant for all of the
/// bits that are being compared. So this:
/// or (or (or X, (X >> 3)), (X >> 5)), (X >> 8)
/// returns {X, 0x129} and those are the operands of an 'and' that is compared
/// to zero.
static bool matchMaskedCmpOp(Value *V, std::pair<Value *, APInt> &Result) {
  // Recurse through a chain of 'or' operands.
  Value *Op0, *Op1;
  if (match(V, m_Or(m_Value(Op0), m_Value(Op1))))
    return matchMaskedCmpOp(Op0, Result) && matchMaskedCmpOp(Op1, Result);

  // We need a shift-right or a bare value representing a compare of bit 0 of
  // the original source operand.
  Value *Candidate;
  uint64_t BitIndex = 0;
  if (!match(V, m_LShr(m_Value(Candidate), m_ConstantInt(BitIndex))))
    Candidate = V;

  // Initialize result source operand.
  if (!Result.first)
    Result.first = Candidate;

  // Fill in the mask bit derived from the shift constant.
  Result.second |= (1 << BitIndex);
  return Result.first == Candidate;
}

/// Match an 'and' of a chain of or-shifted bits from a common source value into
/// a masked compare:
/// and (or (lshr X, C), ...), 1 --> (X & C') != 0
static bool foldToMaskedCmp(Instruction &I) {
  // TODO: This is only looking for 'any-bits-set' and 'all-bits-clear'.
  // We should also match 'all-bits-set' and 'any-bits-clear' by looking for a
  // a chain of 'and'.
  if (!match(&I, m_And(m_OneUse(m_Or(m_Value(), m_Value())), m_One())))
    return false;

  std::pair<Value *, APInt>
  MaskOps(nullptr, APInt::getNullValue(I.getType()->getScalarSizeInBits()));
  if (!matchMaskedCmpOp(cast<BinaryOperator>(&I)->getOperand(0), MaskOps))
    return false;

  IRBuilder<> Builder(&I);
  Value *Mask = Builder.CreateAnd(MaskOps.first, MaskOps.second);
  Value *CmpZero = Builder.CreateIsNotNull(Mask);
  Value *Zext = Builder.CreateZExt(CmpZero, I.getType());
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
    for (Instruction &I : BB)
      MadeChange |= foldToMaskedCmp(I);
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
