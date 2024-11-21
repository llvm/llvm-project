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
/// also, this pass relies on the fact that uniformity analysis remains safe
/// across valid transformations in LLVM. A transformation does not alter
/// program behavior across threads: each instruction in the original IR
/// continues to have a well-defined counterpart in the transformed IR, both
/// statically and dynamically.
///
/// Valid transformations respect three invariants:
/// 1. Use-def relationships are preserved. If one instruction produces a value
///    and another consumes it, that dependency must remain intact.
/// 2. Uniformity classification is preserved. Certain values are always uniform
///    (constants, kernel arguments, convergent operations), while others are
///    always divergent (atomics, most function calls). Transformations may turn
///    divergent computations into uniform ones, but never the reverse.
/// 3. Uniformity must hold not only at the point of value computation but also
///    at all later uses of that value, consistently across the same set of
///    threads.
///
/// Together, these invariants ensure that transformations in this pass are
/// correctness-preserving and remain safe for uniformity analysis.
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
    LLVM_DEBUG(dbgs() << "Replacing " << II << " with " << *Src << '\n');
    II.replaceAllUsesWith(Src);
    II.eraseFromParent();
    return true;
  }
  case Intrinsic::amdgcn_ballot: {
    Value *Src = II.getArgOperand(0);
    if (UI.isDivergentUse(II.getOperandUse(0)))
      return false;
    LLVM_DEBUG(dbgs() << "Found uniform ballot intrinsic: " << II << '\n');

    // If there are no ICmp users, return early.
    if (none_of(II.users(), [](User *U) { return isa<ICmpInst>(U); }))
      return false;

    bool Changed = false;
    for (User *U : make_early_inc_range(II.users())) {
      if (auto *ICmp = dyn_cast<ICmpInst>(U)) {
        Value *Op0 = ICmp->getOperand(0);
        Value *Op1 = ICmp->getOperand(1);
        ICmpInst::Predicate Pred = ICmp->getPredicate();
        Value *OtherOp = Op0 == &II ? Op1 : Op0;

        if (Pred == ICmpInst::ICMP_EQ && match(OtherOp, m_Zero())) {
          // Case (icmp eq %ballot, 0) -->  xor %ballot_arg, 1
          Instruction *NotOp =
              BinaryOperator::CreateNot(Src, "", ICmp->getIterator());
          LLVM_DEBUG(dbgs() << "Replacing ICMP_EQ: " << *NotOp << '\n');
          ICmp->replaceAllUsesWith(NotOp);
          ICmp->eraseFromParent();
          Changed = true;
        } else if (Pred == ICmpInst::ICMP_NE && match(OtherOp, m_Zero())) {
          // (icmp ne %ballot, 0)  -->  %ballot_arg
          LLVM_DEBUG(dbgs() << "Replacing ICMP_NE with ballot argument: "
                            << *Src << '\n');
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

/// Iterate over the Intrinsics use in the Module to optimise.
static bool runUniformIntrinsicCombine(Module &M, ModuleAnalysisManager &AM) {
  bool IsChanged = false;
  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  for (Function &F : M) {
    switch (F.getIntrinsicID()) {
    case Intrinsic::amdgcn_permlane64:
    case Intrinsic::amdgcn_readfirstlane:
    case Intrinsic::amdgcn_readlane:
    case Intrinsic::amdgcn_ballot:
      break;
    default:
      continue;
    }

    for (User *U : F.users()) {
      auto *II = cast<IntrinsicInst>(U);
      Function *ParentF = II->getFunction();
      if (ParentF->isDeclaration())
        continue;

      const auto &UI = FAM.getResult<UniformityInfoAnalysis>(*ParentF);
      IsChanged |= optimizeUniformIntrinsic(*II, UI);
    }
  }
  return IsChanged;
}

PreservedAnalyses
AMDGPUUniformIntrinsicCombinePass::run(Module &M, ModuleAnalysisManager &AM) {
  if (!runUniformIntrinsicCombine(M, AM))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserve<UniformityInfoAnalysis>();
  return PA;
}
