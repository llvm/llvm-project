//===-- InstructionPrecedenceTracking.cpp -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "AnalysisOptions.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/PatternMatch.h"

using namespace llvm;

const Instruction *InstructionPrecedenceTracking::getFirstSpecialInstruction(
    const BasicBlock *BB) {
#ifndef NDEBUG
  // If there is a bug connected to invalid cache, turn on
  // -ipt-expensive-asserts to catch this situation as early as possible.
  if (getAnalysisOptions(BB->getContext()).ipt_expensive_asserts)
    validateAll();
  else
    validate(BB);
#endif

  auto [It, Inserted] = FirstSpecialInsts.try_emplace(BB);
  if (Inserted) {
    for (const auto &I : *BB) {
      if (isSpecialInstruction(&I)) {
        It->second = &I;
        break;
      }
    }
  }
  return It->second;
}

bool InstructionPrecedenceTracking::hasSpecialInstructions(
    const BasicBlock *BB) {
  return getFirstSpecialInstruction(BB) != nullptr;
}

bool InstructionPrecedenceTracking::isPreceededBySpecialInstruction(
    const Instruction *Insn) {
  const Instruction *MaybeFirstSpecial =
      getFirstSpecialInstruction(Insn->getParent());
  return MaybeFirstSpecial && MaybeFirstSpecial->comesBefore(Insn);
}

#ifndef NDEBUG
void InstructionPrecedenceTracking::validate(const BasicBlock *BB) const {
  auto It = FirstSpecialInsts.find(BB);
  // Bail if we don't have anything cached for this block.
  if (It == FirstSpecialInsts.end())
    return;

  const Instruction *Cached = It->second;
  assert((!Cached || Cached->getParent() == BB) &&
         "Cached first special instruction is not in the block!");

  // The cache is allowed to be conservative: the cached instruction may be
  // earlier than the actual first special instruction (e.g. if an instruction
  // stopped being special after a RAUW of one of its operands), but it must
  // never be later than it.
  for (const Instruction &Insn : *BB) {
    if (&Insn == Cached)
      return;
    if (isSpecialInstruction(&Insn)) {
      assert(Cached && "Block is marked as having no special instructions but "
                       "in fact it has some!");
      llvm_unreachable("Cached first special instruction is too late!");
    }
  }
}

void InstructionPrecedenceTracking::validateAll() const {
  // Check that for every known block the cached value is correct.
  for (const auto &It : FirstSpecialInsts)
    validate(It.first);
}
#endif

void InstructionPrecedenceTracking::insertInstructionTo(const Instruction *Inst,
                                                        const BasicBlock *BB) {
  if (isSpecialInstruction(Inst))
    FirstSpecialInsts.erase(BB);
}

void InstructionPrecedenceTracking::removeInstruction(const Instruction *Inst) {
  auto *BB = Inst->getParent();
  assert(BB && "must be called before instruction is actually removed");
  auto It = FirstSpecialInsts.find(BB);
  if (It != FirstSpecialInsts.end() && It->second == Inst)
    FirstSpecialInsts.erase(It);
}

void InstructionPrecedenceTracking::removeUsersOf(const Instruction *Inst) {
  for (const Instruction *U : Inst->users())
    removeInstruction(U);
}

void InstructionPrecedenceTracking::clear() {
  FirstSpecialInsts.clear();
#ifndef NDEBUG
  // The map should be valid after clearing (at least empty).
  validateAll();
#endif
}

bool ImplicitControlFlowTracking::isSpecialInstruction(
    const Instruction *Insn) const {
  // If a block's instruction doesn't always pass the control to its successor
  // instruction, mark the block as having implicit control flow. We use them
  // to avoid wrong assumptions of sort "if A is executed and B post-dominates
  // A, then B is also executed". This is not true is there is an implicit
  // control flow instruction (e.g. a guard) between them.
  return !isGuaranteedToTransferExecutionToSuccessor(Insn);
}

bool MemoryWriteTracking::isSpecialInstruction(
    const Instruction *Insn) const {
  using namespace PatternMatch;
  if (match(Insn, m_Intrinsic<Intrinsic::experimental_widenable_condition>()))
    return false;
  return Insn->mayWriteToMemory();
}
