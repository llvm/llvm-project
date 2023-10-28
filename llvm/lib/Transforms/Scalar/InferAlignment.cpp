//===- InferAlignment.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Infer alignment for load, stores and other memory operations based on
// trailing zero known bits information.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/InferAlignment.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Instructions.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

#define DEBUG_TYPE "infer-alignment"

static bool tryToImproveAlign(
    const DataLayout &DL, Instruction *I,
    function_ref<Align(Value *PtrOp, Align OldAlign, Align PrefAlign)> Fn) {
  if (auto *LI = dyn_cast<LoadInst>(I)) {
    Value *PtrOp = LI->getPointerOperand();
    Align OldAlign = LI->getAlign();
    Align NewAlign = Fn(PtrOp, OldAlign, DL.getPrefTypeAlign(LI->getType()));
    if (NewAlign > OldAlign) {
      LI->setAlignment(NewAlign);
      return true;
    }
  } else if (auto *SI = dyn_cast<StoreInst>(I)) {
    Value *PtrOp = SI->getPointerOperand();
    Value *ValOp = SI->getValueOperand();
    Align OldAlign = SI->getAlign();
    Align NewAlign = Fn(PtrOp, OldAlign, DL.getPrefTypeAlign(ValOp->getType()));
    if (NewAlign > OldAlign) {
      SI->setAlignment(NewAlign);
      return true;
    }
  }
  // TODO: Also handle memory intrinsics.
  return false;
}

bool inferAlignment(Function &F, AssumptionCache &AC, DominatorTree &DT) {
  const DataLayout &DL = F.getParent()->getDataLayout();
  bool Changed = false;

  // Enforce preferred type alignment if possible. We do this as a separate
  // pass first, because it may improve the alignments we infer below.
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      Changed |= tryToImproveAlign(
          DL, &I, [&](Value *PtrOp, Align OldAlign, Align PrefAlign) {
            if (PrefAlign > OldAlign)
              return std::max(OldAlign,
                              tryEnforceAlignment(PtrOp, PrefAlign, DL));
            return OldAlign;
          });
    }
  }

  // Compute alignment from known bits.
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      Changed |= tryToImproveAlign(
          DL, &I, [&](Value *PtrOp, Align OldAlign, Align PrefAlign) {
            KnownBits Known = computeKnownBits(PtrOp, DL, 0, &AC, &I, &DT);
            unsigned TrailZ = std::min(Known.countMinTrailingZeros(),
                                       +Value::MaxAlignmentExponent);
            return Align(1ull << std::min(Known.getBitWidth() - 1, TrailZ));
          });
    }
  }

  return Changed;
}

PreservedAnalyses InferAlignmentPass::run(Function &F,
                                          FunctionAnalysisManager &AM) {
  AssumptionCache &AC = AM.getResult<AssumptionAnalysis>(F);
  DominatorTree &DT = AM.getResult<DominatorTreeAnalysis>(F);
  inferAlignment(F, AC, DT);
  // Changes to alignment shouldn't invalidated analyses.
  return PreservedAnalyses::all();
}

class InferAligments : public FunctionPass {

public:
  static char ID;

  InferAligments() : FunctionPass(ID) {
    initializeInferAligmentsPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addRequired<AssumptionCacheTracker>();
  }

  bool runOnFunction(Function &F) override;
};

char InferAligments::ID = 0;

INITIALIZE_PASS_BEGIN(InferAligments, DEBUG_TYPE, "Infer Alignments", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_END(InferAligments, DEBUG_TYPE, "Infer Alignments", false,
                    false)

bool InferAligments::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  auto *DTWP = getAnalysisIfAvailable<DominatorTreeWrapperPass>();
  DominatorTree *DT = DTWP ? &DTWP->getDomTree() : nullptr;
  return inferAlignment(
      F, getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F), *DT);
}

FunctionPass *llvm::createInferAlignmentsPass() { return new InferAligments(); }