//=== InstructionWorklist.h - Worklist for InstCombine & others -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_INSTRUCTIONWORKLIST_H
#define LLVM_TRANSFORMS_UTILS_INSTRUCTIONWORKLIST_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Config/abi-breaking.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

/// InstructionWorklist - This is the worklist management logic for
/// InstCombine and other simplification passes.
class InstructionWorklist {
  SmallVector<Instruction *, 256> Worklist;
  // 1 + index into Worklist, indexed by Instruction::getNumber(); 0 if absent.
  // An instruction must not be removed and reinserted, which assigns a new
  // number, while in the worklist.
  SmallVector<unsigned, 0> WorklistMap;
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
  // The function and number epoch of the instructions in WorklistMap.
  const Function *NumberedFn = nullptr;
  unsigned NumberEpoch = 0;
#endif
  /// These instructions will be added in reverse order after the current
  /// combine has finished. This means that these instructions will be visited
  /// in the order they have been added.
  SmallSetVector<Instruction *, 16> Deferred;

  unsigned getIndex(const Instruction *I) {
    unsigned N = I->getNumber();
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
    const Function *F = I->getFunction();
    if (!NumberedFn) {
      NumberedFn = F;
      NumberEpoch = F->getInstructionNumberEpoch();
    }
    assert(F == NumberedFn && F->getInstructionNumberEpoch() == NumberEpoch &&
           "instruction renumbered while in the worklist");
#endif
    return N;
  }

public:
  InstructionWorklist() = default;

  InstructionWorklist(InstructionWorklist &&) = default;
  InstructionWorklist &operator=(InstructionWorklist &&) = default;

  bool isEmpty() const { return Worklist.empty() && Deferred.empty(); }

  /// Add instruction to the worklist.
  /// Instructions will be visited in the order they are added.
  /// You likely want to use this method.
  void add(Instruction *I) {
    if (Deferred.insert(I))
      LLVM_DEBUG(dbgs() << "ADD DEFERRED: " << *I << '\n');
  }

  /// Add value to the worklist if it is an instruction.
  /// Instructions will be visited in the order they are added.
  void addValue(Value *V) {
    if (Instruction *I = dyn_cast<Instruction>(V))
      add(I);
  }

  /// Push the instruction onto the worklist stack.
  /// Instructions that have been added first will be visited last.
  void push(Instruction *I) {
    assert(I);
    assert(I->getParent() && "Instruction not inserted yet?");

    unsigned N = getIndex(I);
    if (N >= WorklistMap.size())
      WorklistMap.resize(N + 1);
    if (!WorklistMap[N]) {
      LLVM_DEBUG(dbgs() << "ADD: " << *I << '\n');
      Worklist.push_back(I);
      WorklistMap[N] = Worklist.size();
    }
  }

  void pushValue(Value *V) {
    if (Instruction *I = dyn_cast<Instruction>(V))
      push(I);
  }

  Instruction *popDeferred() {
    if (Deferred.empty())
      return nullptr;
    return Deferred.pop_back_val();
  }

  void reserve(size_t Size) {
    Worklist.reserve(Size + 16);
  }

  /// Remove I from the worklist if it exists.
  void remove(Instruction *I) {
    unsigned N = getIndex(I);
    if (N < WorklistMap.size() && WorklistMap[N]) {
      assert(Worklist[WorklistMap[N] - 1] == I &&
             "instruction renumbered while in the worklist");
      // Don't bother moving everything down, just null out the slot.
      Worklist[WorklistMap[N] - 1] = nullptr;
      WorklistMap[N] = 0;
    }
#ifdef EXPENSIVE_CHECKS
    assert(!is_contained(Worklist, I) &&
           "instruction renumbered while in the worklist");
#endif

    Deferred.remove(I);
  }

  Instruction *removeOne() {
    if (Worklist.empty())
      return nullptr;
    Instruction *I = Worklist.pop_back_val();
    if (I) {
      unsigned N = getIndex(I);
      assert(N < WorklistMap.size() && WorklistMap[N] == Worklist.size() + 1 &&
             "instruction renumbered while in the worklist");
      if (N < WorklistMap.size())
        WorklistMap[N] = 0;
    }
    return I;
  }

  /// When an instruction is simplified, add all users of the instruction
  /// to the work lists because they might get more simplified now.
  void pushUsersToWorkList(Instruction &I) {
    for (Instruction *U : I.users())
      push(U);
  }

  /// Should be called *after* decrementing the use-count on V.
  void handleUseCountDecrement(Value *V) {
    if (auto *I = dyn_cast<Instruction>(V)) {
      add(I);
      // Many folds have one-use limitations. If there's only one use left,
      // revisit that use.
      if (I->hasOneUse())
        add(cast<Instruction>(*I->user_begin()));
    }
  }

  /// Check that the worklist is empty and nuke the backing store for the map.
  void zap() {
    assert(Worklist.empty() && "Worklist not empty?");
    assert(all_of(WorklistMap, [](unsigned Slot) { return Slot == 0; }) &&
           "Worklist empty, but map not?");
    assert(Deferred.empty() && "Deferred instructions left over");

    WorklistMap.clear();
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
    NumberedFn = nullptr;
#endif
  }
};

} // end namespace llvm.

#endif
