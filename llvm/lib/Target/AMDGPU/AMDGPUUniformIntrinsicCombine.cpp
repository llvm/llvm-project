//===-- AMDGPUUniformIntrinsicCombine.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass combines uniform intrinsic instructions.
/// Uniform Intrinsic Combine uses pattern match to identify and optimize
/// redundant intrinsic instructions.
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

class AMDGPUUniformIntrinsicCombine : public FunctionPass {
public:
  static char ID;
  AMDGPUUniformIntrinsicCombine() : FunctionPass(ID) {}

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

char AMDGPUUniformIntrinsicCombine::ID = 0;

char &llvm::AMDGPUUniformIntrinsicCombineID = AMDGPUUniformIntrinsicCombine::ID;

bool AMDGPUUniformIntrinsicCombine::runOnFunction(Function &F) {
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
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (auto *Call = dyn_cast<CallInst>(&I)) {
        if (auto *Intrinsic = dyn_cast<IntrinsicInst>(Call)) {
          IsChanged |= optimizeUniformIntrinsicInst(*Intrinsic);
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
    // The below part may not be safe if the exec is not same between the def
    // and use. Is this part stilll required??
    Instruction *SrcInst = dyn_cast<Instruction>(Src);
    if (SrcInst && SrcInst->getParent() != II.getParent())
      break;

    // readfirstlane (readfirstlane x) -> readfirstlane x
    // readfirstlane (readlane x, y) -> readlane x, y
    // readlane (readfirstlane x), y -> readfirstlane x
    // readlane (readlane x, y), z -> readlane x, y
    // All these cases are identical and are dependent on the inner intrinsic
    // results value.(i.e.irrespective of the which of these case is inner
    // intrinsic will write the same value across all output lane indexes)
    if (UI->isUniform(II.getOperandUse(0))) {
      II.replaceAllUsesWith(Src);
      return true;
    }
    break;
  }
  }
  return false;
}

INITIALIZE_PASS_BEGIN(AMDGPUUniformIntrinsicCombine, DEBUG_TYPE,
                      "AMDGPU uniformIntrinsic Combine", false, false)
INITIALIZE_PASS_DEPENDENCY(UniformityInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(AMDGPUUniformIntrinsicCombine, DEBUG_TYPE,
                    "AMDGPU uniformIntrinsic Combine", false, false)

FunctionPass *llvm::createAMDGPUUniformIntrinsicCombinePass() {
  return new AMDGPUUniformIntrinsicCombine();
}
