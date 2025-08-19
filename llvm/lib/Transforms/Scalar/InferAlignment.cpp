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
#include "llvm/Support/KnownBits.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

static bool tryToImproveAlign(
    const DataLayout &DL, Instruction *I,
    function_ref<Align(Value *PtrOp, Align OldAlign, Align PrefAlign)> Fn) {

  if (auto *PtrOp = getLoadStorePointerOperand(I)) {
    Align OldAlign = getLoadStoreAlignment(I);
    Align PrefAlign = DL.getPrefTypeAlign(getLoadStoreType(I));

    Align NewAlign = Fn(PtrOp, OldAlign, PrefAlign);
    if (NewAlign > OldAlign) {
      setLoadStoreAlignment(I, NewAlign);
      return true;
    }
  }
  // TODO: Also handle memory intrinsics.
  return false;
}

bool inferAlignment(Function &F, AssumptionCache &AC, DominatorTree &DT) {
  const DataLayout &DL = F.getDataLayout();
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
  auto InferFromKnownBits = [&](Instruction &I, Value *PtrOp) {
    KnownBits Known = computeKnownBits(PtrOp, DL, &AC, &I, &DT);
    unsigned TrailZ =
        std::min(Known.countMinTrailingZeros(), +Value::MaxAlignmentExponent);
    return Align(1ull << std::min(Known.getBitWidth() - 1, TrailZ));
  };

  // Propagate alignment between loads and stores that originate from the
  // same base pointer.
  DenseMap<Value *, Align> BestBasePointerAligns;
  auto InferFromBasePointer = [&](Value *PtrOp, Align LoadStoreAlign) {
    APInt OffsetFromBase(DL.getIndexTypeSizeInBits(PtrOp->getType()), 0);
    PtrOp = PtrOp->stripAndAccumulateConstantOffsets(DL, OffsetFromBase, true);
    // Derive the base pointer alignment from the load/store alignment
    // and the offset from the base pointer.
    Align BasePointerAlign =
        commonAlignment(LoadStoreAlign, OffsetFromBase.getLimitedValue());

    auto [It, Inserted] =
        BestBasePointerAligns.try_emplace(PtrOp, BasePointerAlign);
    if (!Inserted) {
      // If the stored base pointer alignment is better than the
      // base pointer alignment we derived, we may be able to use it
      // to improve the load/store alignment. If not, store the
      // improved base pointer alignment for future iterations.
      if (It->second > BasePointerAlign) {
        Align BetterLoadStoreAlign =
            commonAlignment(It->second, OffsetFromBase.getLimitedValue());
        return BetterLoadStoreAlign;
      }
      It->second = BasePointerAlign;
    }
    return LoadStoreAlign;
  };

  for (BasicBlock &BB : F) {
    // We need to reset the map for each block because alignment information
    // can only be propagated from instruction A to B if A dominates B.
    // This is because control flow (and exception throwing) could be dependent
    // on the address (and its alignment) at runtime. Some sort of dominator
    // tree approach could be better, but doing a simple forward pass through a
    // single basic block is correct too.
    BestBasePointerAligns.clear();

    for (Instruction &I : BB) {
      Changed |= tryToImproveAlign(
          DL, &I, [&](Value *PtrOp, Align OldAlign, Align PrefAlign) {
            return std::max(InferFromKnownBits(I, PtrOp),
                            InferFromBasePointer(PtrOp, OldAlign));
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
