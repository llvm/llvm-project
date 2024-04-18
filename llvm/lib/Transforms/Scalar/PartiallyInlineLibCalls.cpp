//===--- PartiallyInlineLibCalls.cpp - Partially inline libcalls ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass tries to partially inline the fast path of well-known library
// functions, such as using square-root instructions for cases where sqrt()
// does not need to set errno.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/PartiallyInlineLibCalls.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DebugCounter.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "partially-inline-libcalls"

DEBUG_COUNTER(PILCounter, "partially-inline-libcalls-transform",
              "Controls transformations in partially-inline-libcalls");

static bool optimizeSQRT(CallInst *Call, Function *CalledFunc,
                         BasicBlock &CurrBB, Function::iterator &BB,
                         const TargetTransformInfo *TTI, DomTreeUpdater *DTU) {
  // There is no need to change the IR, since backend will emit sqrt
  // instruction if the call has already been marked read-only.
  if (Call->onlyReadsMemory())
    return false;

  if (!DebugCounter::shouldExecute(PILCounter))
    return false;

  // Do the following transformation:
  //
  // (before)
  // dst = sqrt(src)
  //
  // (after)
  // v0 = sqrt_noreadmem(src) # native sqrt instruction.
  // [if (v0 is a NaN) || if (src < 0)]
  //   v1 = sqrt(src)         # library call.
  // dst = phi(v0, v1)
  //

  Type *Ty = Call->getType();
  IRBuilder<> Builder(Call->getNextNode());

  // Split CurrBB right after the call, create a 'then' block (that branches
  // back to split-off tail of CurrBB) into which we'll insert a libcall.
  Instruction *LibCallTerm = SplitBlockAndInsertIfThen(
      Builder.getTrue(), Call->getNextNode(), /*Unreachable=*/false,
      /*BranchWeights*/ nullptr, DTU);

  auto *CurrBBTerm = cast<BranchInst>(CurrBB.getTerminator());
  // We want an 'else' block though, not a 'then' block.
  cast<BranchInst>(CurrBBTerm)->swapSuccessors();

  // Create phi that will merge results of either sqrt and replace all uses.
  BasicBlock *JoinBB = LibCallTerm->getSuccessor(0);
  JoinBB->setName(CurrBB.getName() + ".split");
  Builder.SetInsertPoint(JoinBB, JoinBB->begin());
  PHINode *Phi = Builder.CreatePHI(Ty, 2);
  Call->replaceAllUsesWith(Phi);

  // Finally, insert the libcall into 'else' block.
  BasicBlock *LibCallBB = LibCallTerm->getParent();
  LibCallBB->setName("call.sqrt");
  Builder.SetInsertPoint(LibCallTerm);
  Instruction *LibCall = Call->clone();
  Builder.Insert(LibCall);

  // Add memory(none) attribute, so that the backend can use a native sqrt
  // instruction for this call.
  Call->setDoesNotAccessMemory();

  // Insert a FP compare instruction and use it as the CurrBB branch condition.
  Builder.SetInsertPoint(CurrBBTerm);
  Value *FCmp = TTI->isFCmpOrdCheaperThanFCmpZero(Ty)
                    ? Builder.CreateFCmpORD(Call, Call)
                    : Builder.CreateFCmpOGE(Call->getOperand(0),
                                            ConstantFP::get(Ty, 0.0));
  CurrBBTerm->setCondition(FCmp);

  // Add phi operands.
  Phi->addIncoming(Call, &CurrBB);
  Phi->addIncoming(LibCall, LibCallBB);

  BB = JoinBB->getIterator();
  return true;
}

static cl::opt<unsigned> StrNCmpInlineThreshold(
    "strncmp-inline-threshold", cl::init(3), cl::Hidden,
    cl::desc("The maximum length of a constant string for a builtin string cmp "
             "call eligible for inlining. The default value is 3."));

namespace {
class StrNCmpInliner {
public:
  StrNCmpInliner(CallInst *CI, LibFunc Func, Function::iterator &BBNext,
                 DomTreeUpdater *DTU, const DataLayout &DL)
      : CI(CI), Func(Func), BBNext(BBNext), DTU(DTU), DL(DL) {}

  bool optimizeStrNCmp();

private:
  bool inlineCompare(Value *LHS, StringRef RHS, uint64_t N, bool Switched);

  CallInst *CI;
  LibFunc Func;
  Function::iterator &BBNext;
  DomTreeUpdater *DTU;
  const DataLayout &DL;
};

} // namespace

/// First we normalize calls to strncmp/strcmp to the form of
/// compare(s1, s2, N), which means comparing first N bytes of s1 and s2
/// (without considering '\0')
///
/// Examples:
///
/// \code
///   strncmp(s, "a", 3) -> compare(s, "a", 2)
///   strncmp(s, "abc", 3) -> compare(s, "abc", 3)
///   strncmp(s, "a\0b", 3) -> compare(s, "a\0b", 2)
///   strcmp(s, "a") -> compare(s, "a", 2)
///
///   char s2[] = {'a'}
///   strncmp(s, s2, 3) -> compare(s, s2, 3)
///
///   char s2[] = {'a', 'b', 'c', 'd'}
///   strncmp(s, s2, 3) -> compare(s, s2, 3)
/// \endcode
///
/// We only handle cases that N and exactly one of s1 and s2 are constant. Cases
/// that s1 and s2 are both constant are already handled by the instcombine
/// pass.
///
/// We do not handle cases that N > StrNCmpInlineThreshold.
///
/// We also do not handles cases that N < 2, which are already
/// handled by the instcombine pass.
///
bool StrNCmpInliner::optimizeStrNCmp() {
  if (StrNCmpInlineThreshold < 2)
    return false;

  Value *Str1P = CI->getArgOperand(0);
  Value *Str2P = CI->getArgOperand(1);
  // should be handled elsewhere
  if (Str1P == Str2P)
    return false;

  StringRef Str1, Str2;
  bool HasStr1 = getConstantStringInfo(Str1P, Str1, false);
  bool HasStr2 = getConstantStringInfo(Str2P, Str2, false);
  if (!(HasStr1 ^ HasStr2))
    return false;

  // note that '\0' and characters after it are not trimmed
  StringRef Str = HasStr1 ? Str1 : Str2;

  size_t Idx = Str.find('\0');
  uint64_t N = Idx == StringRef::npos ? UINT64_MAX : Idx + 1;
  if (Func == LibFunc_strncmp) {
    if (!isa<ConstantInt>(CI->getArgOperand(2)))
      return false;
    N = std::min(N, cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue());
  }
  // now N means how many bytes we need to compare at most
  if (N > Str.size() || N < 2 || N > StrNCmpInlineThreshold)
    return false;

  Value *StrP = HasStr1 ? Str2P : Str1P;

  // cases that StrP has two or more dereferenceable bytes might be better
  // optimized elsewhere
  bool CanBeNull = false, CanBeFreed = false;
  if (StrP->getPointerDereferenceableBytes(DL, CanBeNull, CanBeFreed) > 1)
    return false;

  return inlineCompare(StrP, Str, N, HasStr1);
}

/// Convert
///
/// \code
///   ret = compare(s1, s2, N)
/// \endcode
///
/// into
///
/// \code
///   ret = (int)s1[0] - (int)s2[0]
///   if (ret != 0)
///     goto NE
///   ...
///   ret = (int)s1[N-2] - (int)s2[N-2]
///   if (ret != 0)
///     goto NE
///   ret = (int)s1[N-1] - (int)s2[N-1]
///   NE:
/// \endcode
///
/// CFG before and after the transformation:
///
/// (before)
/// BBCI
///
/// (after)
/// BBBefore -> BBSubs[0] (sub,icmp) --NE-> BBNE -> BBCI
///                      |                    ^
///                      E                    |
///                      |                    |
///             BBSubs[1] (sub,icmp) --NE-----+
///                     ...                   |
///             BBSubs[N-1]    (sub) ---------+
///
bool StrNCmpInliner::inlineCompare(Value *LHS, StringRef RHS, uint64_t N,
                                   bool Switched) {
  IRBuilder<> B(CI->getContext());

  BasicBlock *BBCI = CI->getParent();
  bool IsEntry = BBCI->isEntryBlock();
  BasicBlock *BBBefore = splitBlockBefore(BBCI, CI, DTU, nullptr, nullptr,
                                          BBCI->getName() + ".before");

  SmallVector<BasicBlock *> BBSubs;
  for (uint64_t i = 0; i < N + 1; ++i)
    BBSubs.push_back(
        BasicBlock::Create(CI->getContext(), "sub", BBCI->getParent(), BBCI));
  BasicBlock *BBNE = BBSubs[N];

  cast<BranchInst>(BBBefore->getTerminator())->setSuccessor(0, BBSubs[0]);

  B.SetInsertPoint(BBNE);
  PHINode *Phi = B.CreatePHI(CI->getType(), N);
  B.CreateBr(BBCI);

  Value *Base = LHS;
  for (uint64_t i = 0; i < N; ++i) {
    B.SetInsertPoint(BBSubs[i]);
    Value *VL = B.CreateZExt(
        B.CreateLoad(B.getInt8Ty(),
                     B.CreateInBoundsGEP(B.getInt8Ty(), Base, B.getInt64(i))),
        CI->getType());
    Value *VR = ConstantInt::get(CI->getType(), RHS[i]);
    Value *Sub = Switched ? B.CreateSub(VR, VL) : B.CreateSub(VL, VR);
    if (i < N - 1)
      B.CreateCondBr(B.CreateICmpNE(Sub, ConstantInt::get(CI->getType(), 0)),
                     BBNE, BBSubs[i + 1]);
    else
      B.CreateBr(BBNE);

    Phi->addIncoming(Sub, BBSubs[i]);
  }

  CI->replaceAllUsesWith(Phi);
  CI->eraseFromParent();

  BBNext = BBCI->getIterator();

  // Update DomTree
  if (DTU) {
    if (IsEntry) {
      DTU->recalculate(*BBCI->getParent());
    } else {
      SmallVector<DominatorTree::UpdateType, 8> Updates;
      Updates.push_back({DominatorTree::Delete, BBBefore, BBCI});
      Updates.push_back({DominatorTree::Insert, BBBefore, BBSubs[0]});
      for (uint64_t i = 0; i < N; ++i) {
        if (i < N - 1)
          Updates.push_back({DominatorTree::Insert, BBSubs[i], BBSubs[i + 1]});
        Updates.push_back({DominatorTree::Insert, BBSubs[i], BBNE});
      }
      Updates.push_back({DominatorTree::Insert, BBNE, BBCI});
      DTU->applyUpdates(Updates);
    }
  }
  return true;
}

static bool runPartiallyInlineLibCalls(Function &F, TargetLibraryInfo *TLI,
                                       const TargetTransformInfo *TTI,
                                       DominatorTree *DT) {
  std::optional<DomTreeUpdater> DTU;
  if (DT)
    DTU.emplace(DT, DomTreeUpdater::UpdateStrategy::Lazy);

  bool Changed = false;

  Function::iterator CurrBB;
  for (Function::iterator BB = F.begin(), BE = F.end(); BB != BE;) {
    CurrBB = BB++;

    for (BasicBlock::iterator II = CurrBB->begin(), IE = CurrBB->end();
         II != IE; ++II) {
      CallInst *Call = dyn_cast<CallInst>(&*II);
      Function *CalledFunc;

      if (!Call || !(CalledFunc = Call->getCalledFunction()))
        continue;

      if (Call->isNoBuiltin() || Call->isStrictFP())
        continue;

      if (Call->isMustTailCall())
        continue;

      // Skip if function either has local linkage or is not a known library
      // function.
      LibFunc LF;
      if (CalledFunc->hasLocalLinkage() ||
          !TLI->getLibFunc(*CalledFunc, LF) || !TLI->has(LF))
        continue;

      switch (LF) {
      case LibFunc_sqrtf:
      case LibFunc_sqrt:
        if (TTI->haveFastSqrt(Call->getType()) &&
            optimizeSQRT(Call, CalledFunc, *CurrBB, BB, TTI,
                         DTU ? &*DTU : nullptr))
          break;
        continue;
      case LibFunc_strcmp:
      case LibFunc_strncmp: {
        auto DT = DTU ? &*DTU : nullptr;
        auto &DL = F.getParent()->getDataLayout();
        if (StrNCmpInliner(Call, LF, BB, DT, DL).optimizeStrNCmp())
          break;
        continue;
      }
      default:
        continue;
      }

      Changed = true;
      break;
    }
  }

  return Changed;
}

PreservedAnalyses
PartiallyInlineLibCallsPass::run(Function &F, FunctionAnalysisManager &AM) {
  auto &TLI = AM.getResult<TargetLibraryAnalysis>(F);
  auto &TTI = AM.getResult<TargetIRAnalysis>(F);
  auto *DT = AM.getCachedResult<DominatorTreeAnalysis>(F);
  if (!runPartiallyInlineLibCalls(F, &TLI, &TTI, DT))
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  return PA;
}

namespace {
class PartiallyInlineLibCallsLegacyPass : public FunctionPass {
public:
  static char ID;

  PartiallyInlineLibCallsLegacyPass() : FunctionPass(ID) {
    initializePartiallyInlineLibCallsLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
    FunctionPass::getAnalysisUsage(AU);
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    TargetLibraryInfo *TLI =
        &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
    const TargetTransformInfo *TTI =
        &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    DominatorTree *DT = nullptr;
    if (auto *DTWP = getAnalysisIfAvailable<DominatorTreeWrapperPass>())
      DT = &DTWP->getDomTree();
    return runPartiallyInlineLibCalls(F, TLI, TTI, DT);
  }
};
}

char PartiallyInlineLibCallsLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(PartiallyInlineLibCallsLegacyPass,
                      "partially-inline-libcalls",
                      "Partially inline calls to library functions", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(PartiallyInlineLibCallsLegacyPass,
                    "partially-inline-libcalls",
                    "Partially inline calls to library functions", false, false)

FunctionPass *llvm::createPartiallyInlineLibCallsPass() {
  return new PartiallyInlineLibCallsLegacyPass();
}
