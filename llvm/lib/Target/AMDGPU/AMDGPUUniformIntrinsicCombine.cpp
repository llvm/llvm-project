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
/// It's true that this pass has transforms that can lead to a situation where
/// some instruction whose operand was previously recognized as statically
/// uniform is later on no longer recognized as statically uniform. However, the
/// semantics of how programs execute don't (and must not, for this precise
/// reason) care about static uniformity, they only ever care about dynamic
/// uniformity. And every instruction that's downstream and cares about dynamic
/// uniformity must be convergent (and isel will introduce v_readfirstlane for
/// them if their operands can't be proven statically uniform).
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

/// Wrapper for querying uniformity info that first checks locally tracked
/// instructions.
static bool
isDivergentUseWithNew(const Use &U, const UniformityInfo &UI,
                      const ValueMap<const Value *, bool> &Tracker) {
  Value *V = U.get();
  if (auto It = Tracker.find(V); It != Tracker.end())
    return !It->second; // divergent if marked false
  return UI.isDivergentUse(U);
}

/// Optimizes uniform intrinsics calls if their operand can be proven uniform.
static bool optimizeUniformIntrinsic(IntrinsicInst &II,
                                     const UniformityInfo &UI,
                                     ValueMap<const Value *, bool> &Tracker) {
  llvm::Intrinsic::ID IID = II.getIntrinsicID();
  /// We deliberately do not simplify readfirstlane with a uniform argument, so
  /// that frontends can use it to force a copy to SGPR and thereby prevent the
  /// backend from generating unwanted waterfall loops.
  switch (IID) {
  case Intrinsic::amdgcn_permlane64:
  case Intrinsic::amdgcn_readlane: {
    Value *Src = II.getArgOperand(0);
    if (isDivergentUseWithNew(II.getOperandUse(0), UI, Tracker))
      return false;
    LLVM_DEBUG(dbgs() << "Replacing " << II << " with " << *Src << '\n');
    II.replaceAllUsesWith(Src);
    II.eraseFromParent();
    return true;
  }
  case Intrinsic::amdgcn_ballot: {
    Value *Src = II.getArgOperand(0);
    if (isDivergentUseWithNew(II.getOperandUse(0), UI, Tracker))
      return false;
    LLVM_DEBUG(dbgs() << "Found uniform ballot intrinsic: " << II << '\n');

    bool Changed = false;
    for (User *U : make_early_inc_range(II.users())) {
      if (auto *ICmp = dyn_cast<ICmpInst>(U)) {
        Value *Op0 = ICmp->getOperand(0);
        Value *Op1 = ICmp->getOperand(1);
        ICmpInst::Predicate Pred = ICmp->getPredicate();
        Value *OtherOp = Op0 == &II ? Op1 : Op0;

        if (Pred == ICmpInst::ICMP_EQ && match(OtherOp, m_Zero())) {
          // Case: (icmp eq %ballot, 0) -> xor %ballot_arg, 1
          Instruction *NotOp =
              BinaryOperator::CreateNot(Src, "", ICmp->getIterator());
          Tracker[NotOp] = true; // NOT preserves uniformity
          LLVM_DEBUG(dbgs() << "Replacing ICMP_EQ: " << *NotOp << '\n');
          ICmp->replaceAllUsesWith(NotOp);
          Changed = true;
        } else if (Pred == ICmpInst::ICMP_NE && match(OtherOp, m_Zero())) {
          // Case: (icmp ne %ballot, 0) -> %ballot_arg
          LLVM_DEBUG(dbgs() << "Replacing ICMP_NE with ballot argument: "
                            << *Src << '\n');
          ICmp->replaceAllUsesWith(Src);
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
    return false;
  }
  return false;
}

/// Iterates over intrinsic calls in the Function to optimize.
static bool runUniformIntrinsicCombine(Function &F, const UniformityInfo &UI) {
  bool IsChanged = false;
  ValueMap<const Value *, bool> Tracker;

  for (Instruction &I : make_early_inc_range(instructions(F))) {
    auto *II = dyn_cast<IntrinsicInst>(&I);
    if (!II)
      continue;
    IsChanged |= optimizeUniformIntrinsic(*II, UI, Tracker);
  }
  return IsChanged;
}

PreservedAnalyses
AMDGPUUniformIntrinsicCombinePass::run(Function &F,
                                       FunctionAnalysisManager &AM) {
  const auto &UI = AM.getResult<UniformityInfoAnalysis>(F);
  if (!runUniformIntrinsicCombine(F, UI))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserve<UniformityInfoAnalysis>();
  return PA;
}

namespace {
class AMDGPUUniformIntrinsicCombineLegacy : public FunctionPass {
public:
  static char ID;
  AMDGPUUniformIntrinsicCombineLegacy() : FunctionPass(ID) {
    initializeAMDGPUUniformIntrinsicCombineLegacyPass(
        *PassRegistry::getPassRegistry());
  }

private:
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<UniformityInfoWrapperPass>();
    AU.addRequired<TargetPassConfig>();
  }
};
} // namespace

char AMDGPUUniformIntrinsicCombineLegacy::ID = 0;
char &llvm::AMDGPUUniformIntrinsicCombineLegacyPassID =
    AMDGPUUniformIntrinsicCombineLegacy::ID;

bool AMDGPUUniformIntrinsicCombineLegacy::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;
  const UniformityInfo &UI =
      getAnalysis<UniformityInfoWrapperPass>().getUniformityInfo();
  return runUniformIntrinsicCombine(F, UI);
}

INITIALIZE_PASS_BEGIN(AMDGPUUniformIntrinsicCombineLegacy, DEBUG_TYPE,
                      "AMDGPU Uniform Intrinsic Combine", false, false)
INITIALIZE_PASS_DEPENDENCY(UniformityInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(AMDGPUUniformIntrinsicCombineLegacy, DEBUG_TYPE,
                    "AMDGPU Uniform Intrinsic Combine", false, false)

FunctionPass *llvm::createAMDGPUUniformIntrinsicCombineLegacyPass() {
  return new AMDGPUUniformIntrinsicCombineLegacy();
}
