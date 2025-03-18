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
  Module *M = F.getParent();
  llvm::LLVMContext &Ctx = M->getContext();
  // List of AMDGPU intrinsics to optimize if their arguments are uniform.
  std::vector<Intrinsic::ID> Intrinsics = {
      Intrinsic::amdgcn_permlane64, Intrinsic::amdgcn_readfirstlane,
      Intrinsic::amdgcn_readlane, Intrinsic::amdgcn_ballot};

  bool IsChanged = false;

  // Iterate over each intrinsic in the list and process its uses within F.
  for (Intrinsic::ID IID : Intrinsics) {
    // Determine the correct return type for the intrinsic.
    // Most intrinsics return i32, but amdgcn_ballot returns i64.
    llvm::Type *IntrinsicTy = (IID == Intrinsic::amdgcn_ballot)
                                  ? llvm::Type::getInt64Ty(Ctx)
                                  : llvm::Type::getInt32Ty(Ctx);

    // Check if the intrinsic is declared in the module with the expected type.
    if (Function *Intr =
            Intrinsic::getDeclarationIfExists(M, IID, {IntrinsicTy})) {
      // Iterate over all users of the intrinsic.
      for (User *U : Intr->users()) {
        // Ensure the user is an intrinsic call within function F.
        if (auto *II = dyn_cast<IntrinsicInst>(U)) {
          if (II->getFunction() == &F) {
            IsChanged |= optimizeUniformIntrinsicInst(*II);
          }
        }
      }
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
    II.eraseFromParent();
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

        Value *OtherOp = (Op0 == &II ? Op1 : Op0);

        // Case (icmp eq %ballot, 0) -->  xor %ballot_arg, 1
        if (Pred == ICmpInst::ICMP_EQ && match(OtherOp, m_Zero())) {
          Instruction *NotOp =
              BinaryOperator::CreateNot(Src, "", ICmp->getIterator());
          LLVM_DEBUG(dbgs() << "Replacing ICMP_EQ: " << *NotOp << "\n");
          ICmp->replaceAllUsesWith(NotOp);
          ICmp->eraseFromParent();
          II.eraseFromParent();
          Changed = true;
        }
        // (icmp ne %ballot, 0)  -->  %ballot_arg
        else if (Pred == ICmpInst::ICMP_NE && match(OtherOp, m_Zero())) {
          LLVM_DEBUG(dbgs() << "Replacing ICMP_NE with ballot argument: "
                            << *Src << "\n");
          ICmp->replaceAllUsesWith(Src);
          ICmp->eraseFromParent();
          II.eraseFromParent();
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
