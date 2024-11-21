//===-- AMDGPUUniformIntrinsicCombine.cpp
//-----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass combines uniform intrinsic instructions.
/// Unifrom Intrinsic combine uses pattern match to identify and optimize
/// redundent intrinsic instruction.
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "llvm/Analysis/DomTreeUpdater.h"
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
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addRequired<UniformityInfoWrapperPass>();
    AU.addRequired<TargetPassConfig>();
  }
};

class AMDGPUUniformIntrinsicCombineImpl
    : public InstVisitor<AMDGPUUniformIntrinsicCombineImpl> {
private:
  const UniformityInfo *UI;

  void optimizeUniformIntrinsicInst(IntrinsicInst &II) const;

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

  // @todo check if it is required that this method must return bool, if so
  // figure out what can be returned.
  bool IsChanged = AMDGPUUniformIntrinsicCombineImpl(UI).run(F);

  if (!IsChanged) {
    return PreservedAnalyses::all();
  }

  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  return PA;
}

bool AMDGPUUniformIntrinsicCombineImpl::run(Function &F) {

  // @todo check if it is required that this method must return bool, if so
  // figure out what can be returned.
  const bool IsChanged{false};

  // Iterate over each instruction in the function to get the desired intrinsic
  // inst to check for optimization.
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (auto *Call = dyn_cast<CallInst>(&I)) {
        if (auto *Intrinsic = dyn_cast<IntrinsicInst>(Call)) {
          optimizeUniformIntrinsicInst(*Intrinsic);
        }
      }
    }
  }

  return IsChanged;
}

void AMDGPUUniformIntrinsicCombineImpl::optimizeUniformIntrinsicInst(
    IntrinsicInst &II) const {
  llvm::Intrinsic::ID IID = II.getIntrinsicID();

  switch (IID) {
  case Intrinsic::amdgcn_permlane64: {
    Value *Src = II.getOperand(0);
    if (UI->isUniform(Src)) {
      return II.replaceAllUsesWith(Src);
    }
    break;
  }
  case Intrinsic::amdgcn_readfirstlane:
  case Intrinsic::amdgcn_readlane: {
    Value *Srcv = II.getOperand(0);
    if (UI->isUniform(Srcv)) {
      return II.replaceAllUsesWith(Srcv);
    }

    // The rest of these may not be safe if the exec may not be the same between
    // the def and use.
    Value *Src = II.getArgOperand(0);
    Instruction *SrcInst = dyn_cast<Instruction>(Src);
    if (SrcInst && SrcInst->getParent() != II.getParent())
      break;

    // readfirstlane (readfirstlane x) -> readfirstlane x
    // readlane (readfirstlane x), y -> readfirstlane x
    if (match(Src,
              PatternMatch::m_Intrinsic<Intrinsic::amdgcn_readfirstlane>())) {
      return II.replaceAllUsesWith(Src);
    }

    if (IID == Intrinsic::amdgcn_readfirstlane) {
      // readfirstlane (readlane x, y) -> readlane x, y
      if (match(Src, PatternMatch::m_Intrinsic<Intrinsic::amdgcn_readlane>())) {
        return II.replaceAllUsesWith(Src);
      }
    } else {
      // readlane (readlane x, y), y -> readlane x, y
      if (match(Src, PatternMatch::m_Intrinsic<Intrinsic::amdgcn_readlane>(
                         PatternMatch::m_Value(),
                         PatternMatch::m_Specific(II.getArgOperand(1))))) {
        return II.replaceAllUsesWith(Src);
      }
    }
    break;
  }
  }
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
