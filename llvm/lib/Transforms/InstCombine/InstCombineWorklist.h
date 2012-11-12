//===- InstCombineWorklist.h - Worklist for the InstCombine pass ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef INSTCOMBINE_WORKLIST_H
#define INSTCOMBINE_WORKLIST_H

#define DEBUG_TYPE "instcombine"
#include "llvm/Instruction.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Compiler.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
  
/// InstCombineWorklist - This is the worklist management logic for
/// InstCombine.
class LLVM_LIBRARY_VISIBILITY InstCombineWorklist {
  SmallVector<Instruction*, 256> Worklist;
  DenseMap<Instruction*, unsigned> WorklistMap;
  
  void operator=(const InstCombineWorklist&RHS) LLVM_DELETED_FUNCTION;
  InstCombineWorklist(const InstCombineWorklist&) LLVM_DELETED_FUNCTION;
public:
  InstCombineWorklist() {}
  
  bool isEmpty() const { return Worklist.empty(); }
  
  /// Add - Add the specified instruction to the worklist if it isn't already
  /// in it.
  void Add(Instruction *I) {
    if (WorklistMap.insert(std::make_pair(I, Worklist.size())).second) {
      DEBUG(errs() << "IC: ADD: " << *I << '\n');
      Worklist.push_back(I);
    }
  }
  
  void AddValue(Value *V) {
    if (Instruction *I = dyn_cast<Instruction>(V))
      Add(I);
  }
  
  /// AddInitialGroup - Add the specified batch of stuff in reverse order.
  /// which should only be done when the worklist is empty and when the group
  /// has no duplicates.
  void AddInitialGroup(Instruction *const *List, unsigned NumEntries) {
    assert(Worklist.empty() && "Worklist must be empty to add initial group");
    Worklist.reserve(NumEntries+16);
    WorklistMap.resize(NumEntries);
    DEBUG(errs() << "IC: ADDING: " << NumEntries << " instrs to worklist\n");
    for (unsigned Idx = 0; NumEntries; --NumEntries) {
      Instruction *I = List[NumEntries-1];
      WorklistMap.insert(std::make_pair(I, Idx++));
      Worklist.push_back(I);
    }
  }
  
  // Remove - remove I from the worklist if it exists.
  void Remove(Instruction *I) {
    DenseMap<Instruction*, unsigned>::iterator It = WorklistMap.find(I);
    if (It == WorklistMap.end()) return; // Not in worklist.
    
    // Don't bother moving everything down, just null out the slot.
    Worklist[It->second] = 0;
    
    WorklistMap.erase(It);
  }
  
  Instruction *RemoveOne() {
    Instruction *I = Worklist.back();
    Worklist.pop_back();
    WorklistMap.erase(I);
    return I;
  }
  
  /// AddUsersToWorkList - When an instruction is simplified, add all users of
  /// the instruction to the work lists because they might get more simplified
  /// now.
  ///
  void AddUsersToWorkList(Instruction &I) {
    for (Value::use_iterator UI = I.use_begin(), UE = I.use_end();
         UI != UE; ++UI)
      Add(cast<Instruction>(*UI));
  }
  
  
  /// Zap - check that the worklist is empty and nuke the backing store for
  /// the map if it is large.
  void Zap() {
    assert(WorklistMap.empty() && "Worklist empty, but map not?");
    
    // Do an explicit clear, this shrinks the map if needed.
    WorklistMap.clear();
  }
};
  
} // end namespace llvm.

#endif
