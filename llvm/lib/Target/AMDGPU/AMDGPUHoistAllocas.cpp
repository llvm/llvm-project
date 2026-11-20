//===- AMDGPUHoistAllocas.cpp - Hoist fixed size allocas to the entry -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Instruction selection only assigns a fixed stack slot to allocas in the entry
// block. Any other alloca is lowered as a dynamic stack allocation, which is
// not included in the private segment size reported for the function. On
// targets where the runtime cannot grow the stack (e.g. PAL), this
// under-allocates scratch memory and lets lanes overwrite each other's private
// data.
//
// Move fixed size allocas that execute at most once per invocation into the
// entry block so they become static stack objects. Allocas inside a cycle are
// left in place, since each iteration allocates new memory.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "llvm/Analysis/CycleAnalysis.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-hoist-allocas"

namespace {

class AMDGPUHoistAllocasLegacy : public FunctionPass {
public:
  static char ID;
  AMDGPUHoistAllocasLegacy() : FunctionPass(ID) {}
  bool runOnFunction(Function &F) override;
  StringRef getPassName() const override { return "AMDGPU Hoist Allocas"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<CycleInfoWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.setPreservesCFG();
  }
};

} // end anonymous namespace

char AMDGPUHoistAllocasLegacy::ID = 0;
char &llvm::AMDGPUHoistAllocasLegacyPassID = AMDGPUHoistAllocasLegacy::ID;

INITIALIZE_PASS_BEGIN(AMDGPUHoistAllocasLegacy, DEBUG_TYPE,
                      "AMDGPU Hoist Allocas", false, false)
INITIALIZE_PASS_DEPENDENCY(CycleInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(AMDGPUHoistAllocasLegacy, DEBUG_TYPE,
                    "AMDGPU Hoist Allocas", false, false)

static bool hoistAllocas(Function &F, const CycleInfo &CI,
                         const DominatorTree &DT) {
  BasicBlock &Entry = F.getEntryBlock();
  SmallVector<AllocaInst *> ToHoist;
  for (BasicBlock &BB : F) {
    if (&BB == &Entry || !DT.isReachableFromEntry(&BB) || CI.getCycle(&BB))
      continue;
    for (Instruction &I : BB) {
      auto *AI = dyn_cast<AllocaInst>(&I);
      if (AI && isa<ConstantInt>(AI->getArraySize()))
        ToHoist.push_back(AI);
    }
  }

  if (ToHoist.empty())
    return false;

  BasicBlock::iterator InsertPt = Entry.getFirstNonPHIOrAlloca();
  for (AllocaInst *AI : ToHoist)
    AI->moveBefore(InsertPt);
  return true;
}

bool AMDGPUHoistAllocasLegacy::runOnFunction(Function &F) {
  const CycleInfo &CI = getAnalysis<CycleInfoWrapperPass>().getResult();
  const DominatorTree &DT =
      getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  return hoistAllocas(F, CI, DT);
}

PreservedAnalyses AMDGPUHoistAllocasPass::run(Function &F,
                                              FunctionAnalysisManager &AM) {
  const CycleInfo &CI = AM.getResult<CycleAnalysis>(F);
  const DominatorTree &DT = AM.getResult<DominatorTreeAnalysis>(F);
  if (!hoistAllocas(F, CI, DT))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

FunctionPass *llvm::createAMDGPUHoistAllocasLegacyPass() {
  return new AMDGPUHoistAllocasLegacy();
}
