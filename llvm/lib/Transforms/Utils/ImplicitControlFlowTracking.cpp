//===-- ImplicitControlFlowTracking.cpp -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This class allows to keep track on instructions with implicit control flow.
// These are instructions that may not pass execution to their successors. For
// example, throwing calls and guards do not always do this. If we need to know
// for sure that some instruction is guaranteed to execute if the given block
// is reached, then we need to make sure that there is no implicit control flow
// instruction (ICFI) preceeding it. For example, this check is required if we
// perform PRE moving non-speculable instruction to other place.
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Transforms/Utils/ImplicitControlFlowTracking.h"

using namespace llvm;

const Instruction *
ImplicitControlFlowTracking::getFirstICFI(const BasicBlock *BB) {
  if (!KnownBlocks.count(BB))
    fill(BB);
  auto *FirstICF = FirstImplicitControlFlowInsts.lookup(BB);
  assert((!FirstICF || FirstICF->getParent() == BB) && "Inconsistent cache!");
  return FirstICF;
}

bool ImplicitControlFlowTracking::hasICF(const BasicBlock *BB) {
  return getFirstICFI(BB) != nullptr;
}

bool ImplicitControlFlowTracking::isDominatedByICFIFromSameBlock(
    const Instruction *Insn) {
  const Instruction *MaybeFirstICF = getFirstICFI(Insn->getParent());
  return MaybeFirstICF && OI.dominates(MaybeFirstICF, Insn);
}

void ImplicitControlFlowTracking::fill(const BasicBlock *BB) {
  auto MayNotTransferExecutionToSuccessor = [&](const Instruction *I) {
    // If a block's instruction doesn't always pass the control to its successor
    // instruction, mark the block as having implicit control flow. We use them
    // to avoid wrong assumptions of sort "if A is executed and B post-dominates
    // A, then B is also executed". This is not true is there is an implicit
    // control flow instruction (e.g. a guard) between them.
    //
    // TODO: Currently, isGuaranteedToTransferExecutionToSuccessor returns false
    // for volatile stores and loads because they can trap. The discussion on
    // whether or not it is correct is still ongoing. We might want to get rid
    // of this logic in the future. Anyways, trapping instructions shouldn't
    // introduce implicit control flow, so we explicitly allow them here. This
    // must be removed once isGuaranteedToTransferExecutionToSuccessor is fixed.
    if (isGuaranteedToTransferExecutionToSuccessor(I))
      return false;
    if (isa<LoadInst>(I)) {
      assert(cast<LoadInst>(I)->isVolatile() &&
             "Non-volatile load should transfer execution to successor!");
      return false;
    }
    if (isa<StoreInst>(I)) {
      assert(cast<StoreInst>(I)->isVolatile() &&
             "Non-volatile store should transfer execution to successor!");
      return false;
    }
    return true;
  };
  FirstImplicitControlFlowInsts.erase(BB);

  for (auto &I : *BB)
    if (MayNotTransferExecutionToSuccessor(&I)) {
      FirstImplicitControlFlowInsts[BB] = &I;
      break;
    }

  // Mark this block as having a known result.
  KnownBlocks.insert(BB);
}

void ImplicitControlFlowTracking::invalidateBlock(const BasicBlock *BB) {
  OI.invalidateBlock(BB);
  FirstImplicitControlFlowInsts.erase(BB);
  KnownBlocks.erase(BB);
}

void ImplicitControlFlowTracking::clear() {
  for (auto It : FirstImplicitControlFlowInsts)
    OI.invalidateBlock(It.first);
  FirstImplicitControlFlowInsts.clear();
  KnownBlocks.clear();
}
