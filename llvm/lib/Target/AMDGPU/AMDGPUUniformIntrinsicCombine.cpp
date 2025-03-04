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

        if (ICmp->getPredicate() == ICmpInst::ICMP_EQ &&
            ((Op0 == &II && match(Op1, m_Zero())) ||
             (Op1 == &II && match(Op0, m_Zero())))) {

          IRBuilder<> Builder(ICmp);
          Value *Xor = Builder.CreateXor(Src, Builder.getTrue());
          LLVM_DEBUG(dbgs() << "Replacing with XOR: " << *Xor << "\n");
          ICmp->replaceAllUsesWith(Xor);
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
