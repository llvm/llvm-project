//===-- AMDGPUUniformIntrinsicCombine.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass simplifies certain intrinsic calls when the arguments are uniform.
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/UniformityAnalysis.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define DEBUG_TYPE "amdgpu-uniform-intrinsic-combine"

using namespace llvm;
using namespace llvm::AMDGPU;
using namespace llvm::PatternMatch;

/// Optimizes uniform intrinsics.
static bool optimizeUniformIntrinsic(IntrinsicInst &II,
                                     const UniformityInfo &UI) {
  llvm::Intrinsic::ID IID = II.getIntrinsicID();

  switch (IID) {
  case Intrinsic::amdgcn_permlane64:
  case Intrinsic::amdgcn_readfirstlane:
  case Intrinsic::amdgcn_readlane: {
    Value *Src = II.getArgOperand(0);
    // Check if the argument use is divergent
    if (UI.isDivergentUse(II.getOperandUse(0)))
      return false;
    LLVM_DEBUG(dbgs() << "Replacing " << II << " with " << *Src << "\n");
    II.replaceAllUsesWith(Src);
    II.eraseFromParent();
    return true;
  }
  case Intrinsic::amdgcn_ballot: {
    Value *Src = II.getArgOperand(0);
    if (UI.isDivergentUse(II.getOperandUse(0)))
      return false;
    LLVM_DEBUG(dbgs() << "Found uniform ballot intrinsic: " << II << "\n");

    // If there are no ICmp users, return early.
    if (II.user_empty() ||
        none_of(II.users(), [](User *U) { return isa<ICmpInst>(U); }))
      return false;

    bool Changed = false;
    for (User *U : make_early_inc_range(II.users())) {
      if (auto *ICmp = dyn_cast<ICmpInst>(U)) {
        Value *Op0 = ICmp->getOperand(0);
        Value *Op1 = ICmp->getOperand(1);
        ICmpInst::Predicate Pred = ICmp->getPredicate();

        Value *OtherOp = (Op0 == &II ? Op1 : Op0);

        // Case (icmp eq %ballot, 0) -->  xor %ballot_arg, 1
        if (Pred == ICmpInst::ICMP_EQ && match(OtherOp, m_Zero())) {
          Instruction *NotOp =
              BinaryOperator::CreateNot(Src, "", ICmp->getIterator());
          LLVM_DEBUG(dbgs() << "Replacing ICMP_EQ: " << *NotOp << "\n");
          ICmp->replaceAllUsesWith(NotOp);
          ICmp->eraseFromParent();
          Changed = true;
        }
        // (icmp ne %ballot, 0)  -->  %ballot_arg
        else if (Pred == ICmpInst::ICMP_NE && match(OtherOp, m_Zero())) {
          LLVM_DEBUG(dbgs() << "Replacing ICMP_NE with ballot argument: "
                            << *Src << "\n");
          ICmp->replaceAllUsesWith(Src);
          ICmp->eraseFromParent();
          Changed = true;
        }
      }
    }
    // Erase the intrinsic if it has no remaining uses.
    if (II.use_empty())
      II.eraseFromParent();
    return Changed;
  }
  default:
    llvm_unreachable("Unexpected intrinsic ID in optimizeUniformIntrinsic");
  }
  return false;
}

/// Iterates over the Intrinsics use in the function to optimise.
static bool runUniformIntrinsicCombine(Function &F, const UniformityInfo &UI) {
  Module *M = F.getParent();
  // List of AMDGPU intrinsics to optimize if their arguments are uniform.
  constexpr Intrinsic::ID Intrinsics[] = {
      Intrinsic::amdgcn_permlane64, Intrinsic::amdgcn_readfirstlane,
      Intrinsic::amdgcn_readlane, Intrinsic::amdgcn_ballot};

  bool IsChanged = false;
  for (Function &Func : M->functions()) {
    // Continue if intrinsic doesn't exists or not in the intrinsic list.
    Intrinsic::ID IID = Func.getIntrinsicID();
    if (!llvm::is_contained(Intrinsics, IID))
      continue;
    for (User *U : Func.users()) {
      auto *II = cast<IntrinsicInst>(U);
      if (II->getFunction() == &F)
        IsChanged |= optimizeUniformIntrinsic(*II, UI);
    }
  }
  return IsChanged;
}

PreservedAnalyses
AMDGPUUniformIntrinsicCombinePass::run(Function &F,
                                       FunctionAnalysisManager &AM) {
  const auto &UI = AM.getResult<UniformityInfoAnalysis>(F);
  bool IsChanged = runUniformIntrinsicCombine(F, UI);

  if (!IsChanged)
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserve<UniformityInfoAnalysis>();
  return PA;
}
