//===- InstCombineWorklist.h - Worklist for InstCombine pass ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTCOMBINE_INSTCOMBINEWORKLIST_H
#define LLVM_TRANSFORMS_INSTCOMBINE_INSTCOMBINEWORKLIST_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "instcombine"

namespace llvm {

/// InstCombineWorklist - This is the worklist management logic for
/// InstCombine.
class InstCombineWorklist {
  SmallVector<Instruction *, 256> Worklist;
  DenseMap<Instruction *, unsigned> WorklistMap;
  /// These instructions will be added in reverse order after the current
  /// combine has finished. This means that these instructions will be visited
  /// in the order they have been added.
  SmallSetVector<Instruction *, 16> Deferred;

public:
  InstCombineWorklist() = default;

  InstCombineWorklist(InstCombineWorklist &&) = default;
  InstCombineWorklist &operator=(InstCombineWorklist &&) = default;

  bool isEmpty() const { return Worklist.empty(); }

  /// Add - Add the specified instruction to the worklist if it isn't already
  /// in it.
  void Add(Instruction *I) {
    assert(I);
    assert(I->getParent() && "Instruction not inserted yet?");

    if (WorklistMap.insert(std::make_pair(I, Worklist.size())).second) {
      LLVM_DEBUG(dbgs() << "IC: ADD: " << *I << '\n');
      Worklist.push_back(I);
    }
  }

  void AddValue(Value *V) {
    if (Instruction *I = dyn_cast<Instruction>(V))
      Add(I);
  }

  void AddDeferred(Instruction *I) {
    if (Deferred.insert(I))
      LLVM_DEBUG(dbgs() << "IC: ADD DEFERRED: " << *I << '\n');
  }

  void AddDeferredInstructions() {
    for (Instruction *I : reverse(Deferred))
      Add(I);
    Deferred.clear();
  }

  /// AddInitialGroup - Add the specified batch of stuff in reverse order.
  /// which should only be done when the worklist is empty and when the group
  /// has no duplicates.
  void AddInitialGroup(ArrayRef<Instruction *> List) {
    assert(Worklist.empty() && "Worklist must be empty to add initial group");
    Worklist.reserve(List.size()+16);
    WorklistMap.reserve(List.size());
    LLVM_DEBUG(dbgs() << "IC: ADDING: " << List.size()
                      << " instrs to worklist\n");
    unsigned Idx = 0;
    for (Instruction *I : reverse(List)) {
      WorklistMap.insert(std::make_pair(I, Idx++));
      Worklist.push_back(I);
    }
  }

  // Remove - remove I from the worklist if it exists.
  void Remove(Instruction *I) {
    DenseMap<Instruction*, unsigned>::iterator It = WorklistMap.find(I);
    if (It == WorklistMap.end()) return; // Not in worklist.

    // Don't bother moving everything down, just null out the slot.
    Worklist[It->second] = nullptr;

    WorklistMap.erase(It);
    Deferred.remove(I);
  }

  Instruction *RemoveOne() {
    Instruction *I = Worklist.pop_back_val();
    WorklistMap.erase(I);
    return I;
  }

  /// AddUsersToWorkList - When an instruction is simplified, add all users of
  /// the instruction to the work lists because they might get more simplified
  /// now.
  ///
  void AddUsersToWorkList(Instruction &I) {
    for (User *U : I.users())
      Add(cast<Instruction>(U));
  }


  /// Zap - check that the worklist is empty and nuke the backing store for
  /// the map if it is large.
  void Zap() {
    assert(WorklistMap.empty() && "Worklist empty, but map not?");
    assert(Deferred.empty() && "Deferred instructions left over");

    // Do an explicit clear, this shrinks the map if needed.
    WorklistMap.clear();
  }
};

} // end namespace llvm.

#undef DEBUG_TYPE

#endif
