//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This pass merges a run of fixed-size, textually adjacent 'allocas' in one
/// block outside a function's entry block (e.g., because a loop body allocates
/// fresh stack storage every iteration) that can't be folded into the
/// prologue/epilogue. This prevents the instruction selector from lowering
/// each 'alloca' into it's own 'ISD::DYNAMIC_STACKALLOC' node, which is then
/// lowered into a series of nodes that each read the stack pointer, adjusts
/// it, and writes it back.
///
/// Instead, this pass converts a series of 'allocas' into a single 'alloca',
/// sized to their alignment-padded sum, plus one constant-offset
/// 'getelementptr' per original 'alloca'. Instruction selection then only ever
/// sees one dynamic stack allocation per run.
///
/// This pass runs at the IR level so that every backend benefits (SelectionDAG
/// and GlobalISel alike).
///
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MergeAllocas.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/LibcallLoweringInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/StackProtector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

#define DEBUG_TYPE "merge-allocas"

STATISTIC(NumRunsMerged, "Number of adjacent-alloca runs merged into one");
STATISTIC(NumAllocasEliminated,
          "Number of individual allocas eliminated by merging");

/// True if \p AI is a candidate to be folded into a merged alloca: it must not
/// already be free (i.e., AllocaInst::isStaticAlloca()) and its size must be a
/// compile-time-constant, non-scalable byte count. Also excludes `inalloca`,
/// whose call-lowering contract depends on the exact stack address of the
/// individual alloca, which merging would change.
static bool isMergeCandidate(const AllocaInst &AI, const DataLayout &DL) {
  if (AI.isStaticAlloca() || AI.isUsedWithInAlloca())
    return false;

  std::optional<TypeSize> Size = AI.getAllocationSize(DL);
  return Size && !Size->isScalable();
}

namespace {

/// One alloca in a run being merged, with the byte offset (within the
/// merged buffer) its replacement getelementptr will use.
struct MergedAlloca {
  AllocaInst *Old;
  uint64_t Offset;
};

} // end anonymous namespace

/// Scans forward from \p Iter (which must point at a merge candidate) for the
/// maximal run of adjacent merge-candidate allocas, computing each one's
/// offset within the combined buffer via struct-style layout (each
/// element's offset rounded up to its own alignment). Appends one
/// MergedAlloca per element of the run to Elems (so a "run" of just one
/// alloca is possible -- the caller checks Elems.size() before acting) and
/// returns the size and alignment of the combined buffer, plus an iterator
/// to the first instruction after the run.
static std::tuple<uint64_t, Align, BasicBlock::iterator>
collectMergeRun(BasicBlock::iterator Iter, const DataLayout &DL,
                SmallVectorImpl<MergedAlloca> &Elems) {
  uint64_t TotalSize = 0;
  Align MergedAlign(1);
  BasicBlock *BB = Iter->getParent();

  for (auto End = BB->end(); Iter != End; ++Iter) {
    auto *AI = dyn_cast<AllocaInst>(&*Iter);
    if (!AI || !isMergeCandidate(*AI, DL))
      break;

    uint64_t Size = AI->getAllocationSize(DL)->getFixedValue();
    Align A = AI->getAlign();
    uint64_t Offset = alignTo(TotalSize, A);

    // Overflow-safe: alignTo only ever grows TotalSize, so the only new
    // wraparound risk is Offset + Size itself.
    uint64_t NewTotal = Offset + Size;
    if (NewTotal < Offset)
      break;

    Elems.push_back({AI, Offset});
    TotalSize = NewTotal;
    MergedAlign = std::max(MergedAlign, A);
  }

  return {TotalSize, MergedAlign, Iter};
}

/// Replaces the run of allocas described by Elems/TotalSize/MergedAlign
/// with a single alloca and one constant-offset getelementptr per original
/// alloca, inserted where the first one used to be.
static void mergeRun(ArrayRef<MergedAlloca> Elems, uint64_t TotalSize,
                     Align MergedAlign) {
  AllocaInst *First = Elems.front().Old;
  IRBuilder<> Builder(First);
  AllocaInst *Merged = Builder.CreateAlloca(
      Builder.getInt8Ty(), Builder.getInt64(TotalSize), "merged.alloca");
  Merged->setAlignment(MergedAlign);

  // Build every replacement pointer before erasing any original alloca:
  // Builder's insertion point is fixed at First's original position, so
  // erasing First (Elems.front().Old, the common case of an at-offset-0
  // element) while more GEPs still need to be inserted there would leave
  // Builder inserting relative to an already-erased instruction.
  SmallVector<Value *, 8> NewPtrs;
  NewPtrs.reserve(Elems.size());
  for (const MergedAlloca &Elem : Elems) {
    NewPtrs.push_back(Elem.Offset == 0
                          ? Merged
                          : Builder.CreateConstInBoundsGEP1_64(
                                Builder.getInt8Ty(), Merged, Elem.Offset,
                                Elem.Old->getName() + ".merged"));
  }

  for (auto [Elem, NewPtr] : llvm::zip_equal(Elems, NewPtrs)) {
    Elem.Old->replaceAllUsesWith(NewPtr);
    Elem.Old->eraseFromParent();
  }

  ++NumRunsMerged;
  NumAllocasEliminated += Elems.size();
}

static bool mergeAllocasInBlock(BasicBlock &BB, const DataLayout &DL) {
  bool Changed = false;

  for (BasicBlock::iterator Iter = BB.begin(), End = BB.end(); Iter != End;) {
    auto *AI = dyn_cast<AllocaInst>(&*Iter);
    if (!AI || !isMergeCandidate(*AI, DL)) {
      ++Iter;
      continue;
    }

    SmallVector<MergedAlloca, 8> Elems;
    auto [TotalSize, MergedAlign, RunEnd] = collectMergeRun(Iter, DL, Elems);
    if (Elems.size() < 2) {
      // Nothing to merge here. collectMergeRun always consumes at least
      // the one candidate Iter pointed at, so this always makes progress.
      Iter = RunEnd;
      continue;
    }

    mergeRun(Elems, TotalSize, MergedAlign);
    Changed = true;
    Iter = RunEnd;
  }

  return Changed;
}

static bool runImpl(Function &F) {
  const DataLayout &DL = F.getDataLayout();
  bool Changed = false;

  for (BasicBlock &BB : F)
    Changed |= mergeAllocasInBlock(BB, DL);

  return Changed;
}

namespace {

class MergeAllocasLegacy : public FunctionPass {
public:
  static char ID;
  MergeAllocasLegacy() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<StackProtector>();
    AU.addPreserved<StackProtector>();
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<BranchProbabilityInfoWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addPreserved<TargetLibraryInfoWrapperPass>();
    AU.addRequired<LibcallLoweringInfoWrapper>();
  }

  bool runOnFunction(Function &F) override { return runImpl(F); }
};

char MergeAllocasLegacy::ID = 0;

} // end anonymous namespace

INITIALIZE_PASS(MergeAllocasLegacy, "merge-allocas",
                "Merge adjacent fixed-size allocas", false, false)

FunctionPass *llvm::createMergeAllocasPass() {
  return new MergeAllocasLegacy();
}

PreservedAnalyses MergeAllocasPass::run(Function &F,
                                        FunctionAnalysisManager &FAM) {
  if (!runImpl(F))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
