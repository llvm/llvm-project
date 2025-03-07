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

namespace {
class AMDGPUUniformIntrinsicCombineLegacy : public FunctionPass {
public:
  static char ID;
  AMDGPUUniformIntrinsicCombineLegacy() : FunctionPass(ID) {
    initializeAMDGPUUniformIntrinsicCombineLegacyPass(
        *PassRegistry::getPassRegistry());
  }
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<UniformityInfoWrapperPass>();
    AU.addRequired<TargetPassConfig>();
  }
};

class AMDGPUUniformIntrinsicCombineImpl
    : public InstVisitor<AMDGPUUniformIntrinsicCombineImpl> {
private:
  const UniformityInfo *UI;
  bool optimizeUniformIntrinsicInst(IntrinsicInst &II) const;

public:
  AMDGPUUniformIntrinsicCombineImpl() = delete;
  AMDGPUUniformIntrinsicCombineImpl(const UniformityInfo *UI) : UI(UI) {}
  bool run(Function &F);
};
} // namespace

char AMDGPUUniformIntrinsicCombineLegacy::ID = 0;
char &llvm::AMDGPUUniformIntrinsicCombineLegacyPassID =
    AMDGPUUniformIntrinsicCombineLegacy::ID;

bool AMDGPUUniformIntrinsicCombineLegacy::runOnFunction(Function &F) {
  if (skipFunction(F)) {
    return false;
  }
  const UniformityInfo *UI =
      &getAnalysis<UniformityInfoWrapperPass>().getUniformityInfo();
  return AMDGPUUniformIntrinsicCombineImpl(UI).run(F);
}

PreservedAnalyses
AMDGPUUniformIntrinsicCombinePass::run(Function &F,
                                       FunctionAnalysisManager &AM) {
  const auto *UI = &AM.getResult<UniformityInfoAnalysis>(F);
  bool IsChanged = AMDGPUUniformIntrinsicCombineImpl(UI).run(F);

  if (!IsChanged) {
    return PreservedAnalyses::all();
  }
  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<LoopAnalysis>();
  PA.preserve<ScalarEvolutionAnalysis>();
  PA.preserve<UniformityInfoAnalysis>();
  PA.preserve<TargetLibraryAnalysis>();
  return PA;
}

bool AMDGPUUniformIntrinsicCombineImpl::run(Function &F) {
  bool IsChanged{false};

  // Iterate over each instruction in the function to get the desired intrinsic
  // inst to check for optimization.
  for (Instruction &I : make_early_inc_range(instructions(F))) {
    if (auto *Intrinsic = dyn_cast<IntrinsicInst>(&I)) {
      IsChanged |= optimizeUniformIntrinsicInst(*Intrinsic);
    }
  }
  return IsChanged;
}

bool AMDGPUUniformIntrinsicCombineImpl::optimizeUniformIntrinsicInst(
    IntrinsicInst &II) const {
  llvm::Intrinsic::ID IID = II.getIntrinsicID();

  switch (IID) {
  case Intrinsic::amdgcn_permlane64:
  case Intrinsic::amdgcn_readfirstlane:
  case Intrinsic::amdgcn_readlane: {
    Value *Src = II.getArgOperand(0);
    // Check if the argument use is divergent
    if (UI->isDivergentUse(II.getOperandUse(0)))
      return false;
    LLVM_DEBUG(dbgs() << "Replacing " << II << " with " << *Src << "\n");
    II.replaceAllUsesWith(Src);
    return true;
  }
  case Intrinsic::amdgcn_ballot: {
    Value *Src = II.getArgOperand(0);
    if (UI->isDivergentUse(II.getOperandUse(0)))
      return false;
    LLVM_DEBUG(dbgs() << "Found uniform ballot intrinsic: " << II << "\n");

    bool Changed = false;
    for (User *U : make_early_inc_range(II.users())) {
      if (auto *ICmp = dyn_cast<ICmpInst>(U)) {
        Value *Op0 = ICmp->getOperand(0);
        Value *Op1 = ICmp->getOperand(1);
        ICmpInst::Predicate Pred = ICmp->getPredicate();

        // Ensure ballot is one of the operands
        Value *OtherOp = nullptr;
        if (Op0 == &II)
          OtherOp = Op1;
        else if (Op1 == &II)
          OtherOp = Op0;
        else
          continue; // Skip if ballot isn't involved

        // Case (icmp eq %ballot, 0) OR (icmp ne %ballot, 1)  -->  xor
        // %ballot_arg, 1
        if ((Pred == ICmpInst::ICMP_EQ && match(OtherOp, m_Zero())) ||
            (Pred == ICmpInst::ICMP_NE && match(OtherOp, m_One()))) {
          Instruction *NotOp = BinaryOperator::CreateNot(Src);
          NotOp->insertInto(ICmp->getParent(), ICmp->getIterator());
          LLVM_DEBUG(dbgs() << "Replacing ICMP_EQ/ICMP_NE with NOT: " << *NotOp
                            << "\n");
          ICmp->replaceAllUsesWith(NotOp);
          Changed = true;
        }
        // Case (icmp eq %ballot, 1) OR (icmp ne %ballot, 0)  -->  %ballot_arg
        else if ((Pred == ICmpInst::ICMP_EQ && match(OtherOp, m_One())) ||
                 (Pred == ICmpInst::ICMP_NE && match(OtherOp, m_Zero()))) {
          LLVM_DEBUG(dbgs()
                     << "Replacing ICMP_EQ/ICMP_NE with ballot argument: "
                     << *Src << "\n");
          ICmp->replaceAllUsesWith(Src);
          Changed = true;
        }
      }
    }
    return Changed;
  }
  }
  return false;
}

INITIALIZE_PASS_BEGIN(AMDGPUUniformIntrinsicCombineLegacy, DEBUG_TYPE,
                      "AMDGPU uniformIntrinsic Combine", false, false)
INITIALIZE_PASS_DEPENDENCY(UniformityInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(AMDGPUUniformIntrinsicCombineLegacy, DEBUG_TYPE,
                    "AMDGPU uniformIntrinsic Combine", false, false)

FunctionPass *llvm::createAMDGPUUniformIntrinsicCombineLegacyPass() {
  return new AMDGPUUniformIntrinsicCombineLegacy();
}
