//===- AIRSimdDivergenceGuard.cpp - Keep SIMD ops under their branch ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The result of a SIMD-group function depends on which threads of the
// SIMD-group execute it. On an M2 Max, Metal's compiler if-converts a branch
// whose sides have no side effects: it runs both sides with every thread
// active and selects the results, so `if (tid < 16) m = ballot(true); else
// m = ballot(true);` gives every thread 0xffffffff instead of 0xffff /
// 0xffff0000. Measured on the device: air.simd_ballot (activemask, vote),
// air.simd_is_first (elect), air.simd_broadcast_first and
// air.simd_active_threads_mask are speculated this way; simd_sum,
// simd_min, simd_prefix_exclusive_sum and quad_sum were not, and a shuffle
// cannot tell (it reads the same lanes either way). Marking the functions
// convergent (AIRWriter) does not stop it; any side effect in the block does.
// Every air.simd_* / air.quad_* call is guarded alike, so a future compiler
// that speculates more of them is covered too.
//
// This pass puts one `call void @llvm.sideeffect()` before the first
// air.simd_* / air.quad_* call of each block whose execution depends on a
// branch: a block that does not post-dominate the entry block, or that lies on
// a cycle (a later iteration can run with fewer threads). Blocks every thread
// runs exactly once (straight-line kernel code) are left alone. llvm.sideeffect
// generates no code; it only keeps the block from being speculated.
//
// First, air.simd_vote_all / air.quad_vote_all of a mask become a comparison
// with the ballot of true: in AIR (and on AGX) the vote is over the active
// threads, but Metal compares the mask with every thread of the SIMD-group
// (M2 Max: air.simd_vote_all.i64(0xffff) is false with lanes 0-15 active).
//
// Metal pipeline only: the portable AIR (addAIRPortablePasses) is unchanged.
//
//===----------------------------------------------------------------------===//

#include "AIRSimdDivergenceGuard.h"
#include "AIRKernelParams.h"
#include "llvm/ADT/DenseMap.h"
#include "AIR.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "air-simd-divergence-guard"

namespace {
// The defined functions with SIMD-group calls, in them or in the functions
// they call: a call of one is a SIMD-group call too.
using SimdFunctions = DenseMap<const Function *, bool>;
} // namespace

static bool isSimdGroupCall(const Instruction &I, SimdFunctions &Funcs);

static bool hasSimdGroupCalls(const Function &F, SimdFunctions &Funcs) {
  if (auto It = Funcs.find(&F); It != Funcs.end())
    return It->second;
  Funcs[&F] = false; // (no recursion in AIR)
  bool Has = any_of(instructions(F), [&](const Instruction &I) {
    return isSimdGroupCall(I, Funcs);
  });
  return Funcs[&F] = Has;
}

static bool isSimdGroupCall(const Instruction &I, SimdFunctions &Funcs) {
  const auto *CI = dyn_cast<CallInst>(&I);
  if (!CI)
    return false;
  const Function *Callee = CI->getCalledFunction();
  if (!Callee)
    return false;
  if (!Callee->isDeclaration())
    return hasSimdGroupCalls(*Callee, Funcs);
  StringRef Name = Callee->getName();
  return Name.starts_with("air.simd_") || Name.starts_with("air.quad_");
}

static bool guardFunction(Function &F, SimdFunctions &Funcs) {
  SmallVector<Instruction *, 8> First;
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (isSimdGroupCall(I, Funcs)) {
        First.push_back(&I);
        break;
      }
  if (First.empty())
    return false;

  // Blocks on a cycle.
  SmallPtrSet<const BasicBlock *, 16> OnCycle;
  for (scc_iterator<Function *> It = scc_begin(&F); !It.isAtEnd(); ++It)
    if (It.hasCycle())
      for (BasicBlock *BB : *It)
        OnCycle.insert(BB);

  PostDominatorTree PDT(F);
  const BasicBlock *Entry = &F.getEntryBlock();
  // A function a kernel calls may be called under a branch: all its blocks
  // depend on one.
  bool Called = isCalledFunction(F);
  bool Changed = false;
  for (Instruction *I : First) {
    BasicBlock *BB = I->getParent();
    if (!Called && !OnCycle.count(BB) && PDT.dominates(BB, Entry))
      continue;
    // Already guarded (the pass ran before).
    if (auto *Prev = dyn_cast_or_null<IntrinsicInst>(I->getPrevNode()))
      if (Prev->getIntrinsicID() == Intrinsic::sideeffect)
        continue;
    CallInst::Create(
        Intrinsic::getOrInsertDeclaration(F.getParent(), Intrinsic::sideeffect),
        "", I->getIterator());
    Changed = true;
  }
  return Changed;
}

static bool voteAllOverActiveThreads(Module &M) {
  SmallVector<CallInst *, 4> Votes;
  for (Function &F : M)
    for (Instruction &I : instructions(F))
      if (auto *CI = dyn_cast<CallInst>(&I))
        if (Function *Callee = CI->getCalledFunction())
          if ((Callee->getName().starts_with("air.simd_vote_all") ||
               Callee->getName().starts_with("air.quad_vote_all")) &&
              CI->arg_size() == 1 && CI->getType()->isIntegerTy(1) &&
              CI->getArgOperand(0)->getType()->isIntegerTy())
            Votes.push_back(CI);
  for (CallInst *CI : Votes) {
    IRBuilder<> B(CI);
    bool Quad = CI->getCalledFunction()->getName().starts_with("air.quad_");
    FunctionCallee Ballot =
        Quad ? M.getOrInsertFunction("air.quad_ballot", B.getInt16Ty(),
                                     B.getInt1Ty())
             : M.getOrInsertFunction("air.simd_ballot.i64", B.getInt64Ty(),
                                     B.getInt1Ty());
    Value *Mask = CI->getArgOperand(0);
    Value *Active = B.CreateZExtOrTrunc(
        B.CreateCall(Ballot, {B.getTrue()}, "active"), Mask->getType());
    Value *All = B.CreateICmpEQ(Mask, Active);
    All->takeName(CI);
    CI->replaceAllUsesWith(All);
    Function *Old = CI->getCalledFunction();
    CI->eraseFromParent();
    if (Old->use_empty())
      Old->eraseFromParent();
  }
  return !Votes.empty();
}

static bool guardModule(Module &M) {
  bool Changed = voteAllOverActiveThreads(M);
  SimdFunctions Funcs;
  for (Function &F : M)
    if (!F.isDeclaration())
      Changed |= guardFunction(F, Funcs);
  return Changed;
}

PreservedAnalyses AIRSimdDivergenceGuardPass::run(Module &M,
                                                  ModuleAnalysisManager &AM) {
  return guardModule(M) ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

bool AIRSimdDivergenceGuardLegacy::runOnModule(Module &M) {
  return guardModule(M);
}

char AIRSimdDivergenceGuardLegacy::ID = 0;

INITIALIZE_PASS(AIRSimdDivergenceGuardLegacy, DEBUG_TYPE,
                "AIR SIMD-group divergence guard", false, false)

ModulePass *llvm::createAIRSimdDivergenceGuardLegacyPass() {
  return new AIRSimdDivergenceGuardLegacy();
}
