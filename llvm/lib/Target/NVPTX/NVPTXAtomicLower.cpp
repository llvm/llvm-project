//===-- NVPTXAtomicLower.cpp - Lower atomics of local memory ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Lower atomics of local memory to simple load/stores
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "llvm/CodeGen/StackProtector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Utils/LowerAtomic.h"

#include "MCTargetDesc/NVPTXBaseInfo.h"
using namespace llvm;

static bool lowerLocalMemoryAtomics(Function &F) {
  SmallVector<AtomicRMWInst *> LocalMemoryAtomics;
  for (Instruction &I : instructions(F))
    if (AtomicRMWInst *RMWI = dyn_cast<AtomicRMWInst>(&I))
      if (RMWI->getPointerAddressSpace() == ADDRESS_SPACE_LOCAL)
        LocalMemoryAtomics.push_back(RMWI);

  bool Changed = false;
  for (AtomicRMWInst *RMWI : LocalMemoryAtomics)
    Changed |= lowerAtomicRMWInst(RMWI);
  return Changed;
}

namespace {
class NVPTXAtomicLowerLegacyPass : public FunctionPass {
public:
  static char ID; // Pass ID
  NVPTXAtomicLowerLegacyPass() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }

  StringRef getPassName() const override {
    return "NVPTX lower atomics of local memory";
  }

  bool runOnFunction(Function &F) override {
    return lowerLocalMemoryAtomics(F);
  }
};
} // namespace

char NVPTXAtomicLowerLegacyPass::ID = 0;

INITIALIZE_PASS(NVPTXAtomicLowerLegacyPass, "nvptx-atomic-lower",
                "Lower atomics of local memory to simple load/stores", false,
                false)

FunctionPass *llvm::createNVPTXAtomicLowerLegacyPass() {
  return new NVPTXAtomicLowerLegacyPass();
}

PreservedAnalyses NVPTXAtomicLowerPass::run(Function &F,
                                            FunctionAnalysisManager &FAM) {
  if (!lowerLocalMemoryAtomics(F))
    return PreservedAnalyses::all();
  return PreservedAnalyses::none().preserveSet<CFGAnalyses>();
}
