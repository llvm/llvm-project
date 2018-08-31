//===-- InstructionPrecedenceTracking.cpp -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Implements a class that is able to define some instructions as "special"
// (e.g. as having implicit control flow, or writing memory, or having another
// interesting property) and then efficiently answers queries of the types:
// 1. Are there any special instructions in the block of interest?
// 2. Return first of the special instructions in the given block;
// 3. Check if the given instruction is preceeded by the first special
//    instruction in the same block.
// The class provides caching that allows to answer these queries quickly. The
// user must make sure that the cached data is invalidated properly whenever
// a content of some tracked block is changed.
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/InstructionPrecedenceTracking.h"
#include "llvm/Analysis/ValueTracking.h"

using namespace llvm;

const Instruction *InstructionPrecedenceTracking::getFirstSpecialInstruction(
    const BasicBlock *BB) {
  if (!KnownBlocks.count(BB))
    fill(BB);
  auto *FirstICF = FirstSpecialInsts.lookup(BB);
  assert((!FirstICF || FirstICF->getParent() == BB) && "Inconsistent cache!");
  return FirstICF;
}

bool InstructionPrecedenceTracking::hasSpecialInstructions(
    const BasicBlock *BB) {
  return getFirstSpecialInstruction(BB) != nullptr;
}

bool InstructionPrecedenceTracking::isPreceededBySpecialInstruction(
    const Instruction *Insn) {
  const Instruction *MaybeFirstICF =
      getFirstSpecialInstruction(Insn->getParent());
  return MaybeFirstICF && OI.dominates(MaybeFirstICF, Insn);
}

void InstructionPrecedenceTracking::fill(const BasicBlock *BB) {
  FirstSpecialInsts.erase(BB);
  for (auto &I : *BB)
    if (isSpecialInstruction(&I)) {
      FirstSpecialInsts[BB] = &I;
      break;
    }

  // Mark this block as having a known result.
  KnownBlocks.insert(BB);
}

void InstructionPrecedenceTracking::invalidateBlock(const BasicBlock *BB) {
  OI.invalidateBlock(BB);
  FirstSpecialInsts.erase(BB);
  KnownBlocks.erase(BB);
}

void InstructionPrecedenceTracking::clear() {
  for (auto It : FirstSpecialInsts)
    OI.invalidateBlock(It.first);
  FirstSpecialInsts.clear();
  KnownBlocks.clear();
}

bool ImplicitControlFlowTracking::isSpecialInstruction(
    const Instruction *Insn) const {
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
  if (isGuaranteedToTransferExecutionToSuccessor(Insn))
    return false;
  if (isa<LoadInst>(Insn)) {
    assert(cast<LoadInst>(Insn)->isVolatile() &&
           "Non-volatile load should transfer execution to successor!");
    return false;
  }
  if (isa<StoreInst>(Insn)) {
    assert(cast<StoreInst>(Insn)->isVolatile() &&
           "Non-volatile store should transfer execution to successor!");
    return false;
  }
  return true;
}
