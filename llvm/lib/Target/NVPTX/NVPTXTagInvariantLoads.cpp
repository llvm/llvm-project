//===------ NVPTXTagInvariantLoads.cpp - Tag invariant loads --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements invaraint load tagging. It traverses load instructions
// in a function, and determines if each load can be tagged as invariant.
//
// We currently infer invariance for loads from
//  - constant global variables, and
//  - kernel function pointer params that are noalias (i.e. __restrict) and
//    never written to.
//
// TODO: Perform a more powerful invariance analysis (ideally IPO).
//
//===----------------------------------------------------------------------===//

#include "NVPTXUtilities.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/NVPTXAddrSpace.h"

using namespace llvm;

static void markLoadsAsInvariant(LoadInst *LI) {
  LI->setMetadata(LLVMContext::MD_invariant_load,
                  MDNode::get(LI->getContext(), {}));
}

static bool tagInvariantLoads(Function &F) {
  const bool IsKernelFn = isKernelFunction(F);

  bool Changed = false;
  for (auto &I : instructions(F)) {
    if (auto *LI = dyn_cast<LoadInst>(&I)) {

      // Don't bother with non-global loads
      if (LI->getPointerAddressSpace() != NVPTXAS::ADDRESS_SPACE_GLOBAL)
        continue;

      if (LI->getMetadata(LLVMContext::MD_invariant_load))
        continue;

      SmallVector<const Value *, 8> Objs;

      // We use getUnderlyingObjects() here instead of getUnderlyingObject()
      // mainly because the former looks through phi nodes while the latter does
      // not. We need to look through phi nodes to handle pointer induction
      // variables.

      getUnderlyingObjects(LI->getPointerOperand(), Objs);

      const bool IsInvariant = all_of(Objs, [&](const Value *V) {
        if (const auto *A = dyn_cast<const Argument>(V))
          return IsKernelFn && A->onlyReadsMemory() && A->hasNoAliasAttr();
        if (const auto *GV = dyn_cast<const GlobalVariable>(V))
          return GV->isConstant();
        return false;
      });

      if (IsInvariant) {
        markLoadsAsInvariant(LI);
        Changed = true;
      }
    }
  }

  return Changed;
}

namespace {

struct NVPTXTagInvariantLoadLegacyPass : public FunctionPass {
  static char ID;

  NVPTXTagInvariantLoadLegacyPass() : FunctionPass(ID) {}
  bool runOnFunction(Function &F) override;
};

} // namespace

INITIALIZE_PASS(NVPTXTagInvariantLoadLegacyPass, "nvptx-tag-invariant-loads",
                "NVPTX Tag Invariant Loads", false, false)

bool NVPTXTagInvariantLoadLegacyPass::runOnFunction(Function &F) {
  return tagInvariantLoads(F);
}

char NVPTXTagInvariantLoadLegacyPass::ID = 0;

FunctionPass *llvm::createNVPTXTagInvariantLoadsPass() {
  return new NVPTXTagInvariantLoadLegacyPass();
}

PreservedAnalyses NVPTXTagInvariantLoadsPass::run(Function &F,
                                                  FunctionAnalysisManager &) {
  return tagInvariantLoads(F) ? PreservedAnalyses::none()
                              : PreservedAnalyses::all();
}
